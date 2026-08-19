#!/usr/bin/env python3
"""Evaluate the frozen v10 composite on fresh clean T01 development data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from train_t01_rgb_target_evidence import (  # noqa: E402
    binary_evidence,
    build_model,
    episode_score,
    examples,
    predict_examples,
)
from interactive_perception.action_outcome import (  # noqa: E402
    HierarchicalActionOutcomePredictor,
    PI05_PATCH_EQUIVALENT_TARGET_PIXELS,
    combine_v10_outcome_sets,
    label_temporal_information_outcome_v10,
)
from interactive_perception.capability_gate import (  # noqa: E402
    exact_binomial_lower_bound,
)
from interactive_perception.semantic_conformal import (  # noqa: E402
    MondrianSemanticConformalCalibrator,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_reference(value: dict) -> Path:
    path = ROOT / value["path"]
    actual = digest(path)
    if actual != value["sha256"]:
        raise ValueError(f"frozen dependency changed: {path}: {actual}")
    return path


def v10_truth(row: dict) -> str:
    evaluator = row["evaluator_only"]
    target = tuple(
        tuple(point["target_pixels"].values())
        for point in evaluator["visibility_history"]
    )
    coverage = tuple(
        max(point["target_pixels"].values())
        >= PI05_PATCH_EQUIVALENT_TARGET_PIXELS
        for point in evaluator["counterfactual_visibility_history"]
    )
    coverage = coverage or (False,) * len(target)
    return label_temporal_information_outcome_v10(
        full_executor=bool(row["full_executor"]),
        target_pixel_history=target,
        minimum_target_pixels=PI05_PATCH_EQUIVALENT_TARGET_PIXELS,
        searched_region_coverage_history=coverage,
    ).value


def class_metrics(truth, predictions, *, confidence: float) -> dict:
    output = {}
    for label in ("FAILED", "REVEALED", "EMPTY"):
        indices = [index for index, value in enumerate(truth) if value == label]
        coverage = sum(label in predictions[index] for index in indices)
        singleton = sum(predictions[index] == (label,) for index in indices)
        trials = len(indices)
        output[label] = {
            "trials": trials,
            "correct_label_retained": coverage,
            "coverage": coverage / trials,
            "singleton_correct": singleton,
            "singleton_accuracy": singleton / trials,
            "singleton_one_sided_95_lower": exact_binomial_lower_bound(
                singleton, trials, confidence
            ),
            "mean_prediction_set_size": float(
                np.mean([len(predictions[index]) for index in indices])
            ),
        }
    return output


def physical_branch(rows, truth, regime, desired, confidence):
    selected = [
        value
        for row, value in zip(rows, truth, strict=True)
        if row["regime"] == regime
    ]
    successes = sum(value == desired for value in selected)
    trials = len(selected)
    lower = exact_binomial_lower_bound(successes, trials, confidence)
    return {
        "endpoint": (
            "prompt-resolvable target at any public history point"
            if desired == "REVEALED"
            else "local middle-layer searched-region coverage with no target evidence"
        ),
        "successes": successes,
        "trials": trials,
        "rate": successes / trials,
        "one_sided_95_lower": lower,
        "passes_0.80": lower >= 0.80,
        "passes_original_0.90": lower >= 0.90,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/calibration/t01_open_and_observe_effect_v10_clean.jsonl",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT
        / "outputs/t01_open_and_observe_effect_v10_clean/pi05_temporal_embeddings_v5.npz",
    )
    parser.add_argument(
        "--composite",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_v10_composite_candidate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_clean_development_v10.json",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    for name in ("dataset", "embeddings", "composite", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output.exists():
        raise FileExistsError(f"immutable clean result already exists: {args.output}")

    import torch
    from PIL import Image

    composite = json.loads(args.composite.read_text())
    composite_sha = digest(args.composite)
    target_artifact_path = resolve_reference(
        composite["target_evidence"]["artifact"]
    )
    target_checkpoint_path = resolve_reference(
        composite["target_evidence"]["checkpoint"]
    )
    effect_artifact_path = resolve_reference(
        composite["observation_effect"]["artifact"]
    )
    target_artifact = json.loads(target_artifact_path.read_text())
    target_calibrator = MondrianSemanticConformalCalibrator.from_dict(
        target_artifact["conformal"]
    )
    checkpoint = torch.load(
        target_checkpoint_path, map_location="cpu", weights_only=True
    )
    target_model = build_model(torch)
    target_model.load_state_dict(checkpoint["state_dict"])
    effect_predictor = HierarchicalActionOutcomePredictor.from_artifact(
        json.loads(effect_artifact_path.read_text())
    )

    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line]
    if len(rows) != 120 or sorted(set(int(row["seed"]) for row in rows)) != list(
        range(1400, 1440)
    ):
        raise ValueError("clean v10 dataset must contain 40 seeds x 3 regimes")
    for row in rows:
        if row["frozen_critic_sha256"] != composite_sha:
            raise ValueError("row was not collected under the frozen composite")
        if row["online_oracle_inputs"]:
            raise ValueError("controller row declares privileged online input")

    data = np.load(args.embeddings)
    if len(rows) != len(data["seed"]):
        raise ValueError("dataset and embeddings are not aligned")
    for row, seed, regime in zip(
        rows, data["seed"], data["regime"], strict=True
    ):
        if int(seed) != int(row["seed"]) or str(regime) != row["regime"]:
            raise ValueError("dataset/embedding order mismatch")

    image_items = examples(rows, ("agentview",))
    target_scores = predict_examples(
        target_model,
        image_items,
        device=torch.device("cpu"),
        size=int(target_artifact["model"]["image_size"]),
        batch_size=128,
        torch=torch,
        Image=Image,
    )
    truth = []
    predictions = []
    records = []
    for index, row in enumerate(rows):
        score = episode_score(row, target_scores, ("agentview",))
        target_set = target_calibrator.predict(binary_evidence(score))
        effect = effect_predictor.predict_effect_history(
            data["history_features"][index], data["robot_state_history"][index]
        )
        prediction = combine_v10_outcome_sets(target_set, effect.prediction_set)
        observed = v10_truth(row)
        truth.append(observed)
        predictions.append(prediction)
        records.append(
            {
                "regime": row["regime"],
                "seed": int(row["seed"]),
                "truth": observed,
                "target_maximum_frame_logit": score,
                "target_prediction_set": list(target_set),
                "effect_prediction_set": list(effect.prediction_set),
                "outcome_prediction_set": list(prediction),
                "return_phase": row["return_status"]["phase"],
                "drawer_opened_final_diagnostic": bool(
                    row["evaluator_only"]["drawer_opened"]
                ),
            }
        )

    per_class = class_metrics(truth, predictions, confidence=args.confidence)
    false_empty = sum(
        prediction == ("EMPTY",) and observed != "EMPTY"
        for prediction, observed in zip(predictions, truth, strict=True)
    )
    false_revealed = sum(
        prediction == ("REVEALED",) and observed != "REVEALED"
        for prediction, observed in zip(predictions, truth, strict=True)
    )
    physical = {
        "REVEALED": physical_branch(
            rows, truth, "revealed_full", "REVEALED", args.confidence
        ),
        "EMPTY": physical_branch(
            rows, truth, "empty_full", "EMPTY", args.confidence
        ),
    }
    revealed_rows = [row for row in rows if row["regime"] == "revealed_full"]
    any_history = sum(
        any(
            max(point["target_pixels"].values())
            >= PI05_PATCH_EQUIVALENT_TARGET_PIXELS
            for point in row["evaluator_only"]["visibility_history"]
        )
        for row in revealed_rows
    )
    final_only = sum(
        max(
            revealed["evaluator_only"]["visibility_history"][-1][
                "target_pixels"
            ].values()
        )
        >= PI05_PATCH_EQUIVALENT_TARGET_PIXELS
        for revealed in revealed_rows
    )
    gates = {
        "correct_label_retention": all(
            value["coverage"] >= 0.90 for value in per_class.values()
        ),
        "singleton_lower_0.80": all(
            value["singleton_one_sided_95_lower"] >= 0.80
            for value in per_class.values()
        ),
        "zero_false_singleton_EMPTY": false_empty == 0,
        "zero_false_singleton_REVEALED": false_revealed == 0,
        "physical_information_lower_0.80": all(
            value["passes_0.80"] for value in physical.values()
        ),
        "privileged_online_read_count_zero": True,
    }
    passed = all(gates.values())
    artifact = {
        "schema_version": "interactive-perception.t01-v10-clean-development.v1",
        "decision": "GO" if passed else "NOT_GO",
        "sealed_audit_authorized": passed,
        "dataset": {"path": str(args.dataset.relative_to(ROOT)), "sha256": digest(args.dataset)},
        "embeddings": {"path": str(args.embeddings.relative_to(ROOT)), "sha256": digest(args.embeddings)},
        "frozen_composite": {"path": str(args.composite.relative_to(ROOT)), "sha256": composite_sha},
        "samples": len(rows),
        "seeds": "1400-1439",
        "per_class": per_class,
        "false_singleton_EMPTY": false_empty,
        "false_singleton_REVEALED": false_revealed,
        "physical_information_acquisition": physical,
        "history_ablation": {
            "six_frame_resolvable_reveals": any_history,
            "final_frame_only_resolvable_reveals": final_only,
            "trials": len(revealed_rows),
        },
        "privileged_input_audit": {
            "controller_online_oracle_read_count": 0,
            "critic_online_oracle_inputs": [],
            "evaluator_timing": "separate replay after controller termination",
        },
        "gates": gates,
        "rows": records,
        "interpretation": {
            "proves": "clean T01 public-RGB OPEN_AND_OBSERVE outcome recognition and target-observability information acquisition",
            "does_not_prove": [
                "final butter placement success",
                "global NOT_FOUND",
                "cross-scene or cross-object generalization",
                "multi-information-action planning",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({key: artifact[key] for key in ("decision", "per_class", "physical_information_acquisition", "history_ablation", "gates")}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
