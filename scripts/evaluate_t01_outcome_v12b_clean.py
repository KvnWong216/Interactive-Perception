#!/usr/bin/env python3
"""Evaluate the frozen all-public-RGB v12b cascade on fresh T01 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from evaluate_t01_outcome_v10_clean import (  # noqa: E402
    class_metrics,
    physical_branch,
    v10_truth,
)
from train_t01_rgb_target_evidence import (  # noqa: E402
    binary_evidence,
    build_model,
    episode_score,
    examples,
    predict_examples,
)
from interactive_perception.action_outcome import (  # noqa: E402
    PI05_PATCH_EQUIVALENT_TARGET_PIXELS,
)
from interactive_perception.rgb_outcome_critic import (  # noqa: E402
    resolve_v12b_cascade,
)
from interactive_perception.semantic_conformal import (  # noqa: E402
    MondrianSemanticConformalCalibrator,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(value: dict) -> Path:
    path = ROOT / value["path"]
    if digest(path) != value["sha256"]:
        raise ValueError(f"frozen v12b dependency changed: {path}")
    return path


def load_head(artifact_path, checkpoint_path, rows, camera, evidence_kind, torch, Image):
    artifact = json.loads(artifact_path.read_text())
    calibrator = MondrianSemanticConformalCalibrator.from_dict(
        artifact["conformal"]
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(torch)
    model.load_state_dict(checkpoint["state_dict"])
    scores = predict_examples(
        model,
        examples(rows, (camera,), evidence_kind),
        device=torch.device("cpu"),
        size=int(artifact["model"]["image_size"]),
        batch_size=128,
        torch=torch,
        Image=Image,
    )
    return calibrator, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--composite",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_v12b_composite_candidate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--sealed-audit", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    if args.dataset is None:
        args.dataset = ROOT / (
            "data/calibration/t01_open_and_observe_effect_v12b_sealed_audit.jsonl"
            if args.sealed_audit
            else "data/calibration/t01_open_and_observe_effect_v12b_clean.jsonl"
        )
    if args.output is None:
        args.output = ROOT / (
            "results/calibration/t01_open_and_observe_sealed_audit_v12b.json"
            if args.sealed_audit
            else "results/calibration/t01_open_and_observe_clean_development_v12b.json"
        )
    for name in ("dataset", "composite", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch
    from PIL import Image

    composite = json.loads(args.composite.read_text())
    if composite["schema_version"] != (
        "interactive-perception.action-outcome-composite.v12b-candidate"
    ):
        raise ValueError("unexpected v12b composite schema")
    if composite.get("online_oracle_inputs"):
        raise ValueError("v12b composite declares privileged online inputs")
    composite_sha = digest(args.composite)
    if args.sealed_audit:
        authorization_path = (
            ROOT
            / "results/calibration/t01_open_and_observe_v12b_sealed_authorization.json"
        )
        authorization = json.loads(authorization_path.read_text())
        if authorization["decision"] != "AUTHORIZED" or not all(
            authorization["clean_gates"].values()
        ):
            raise ValueError("sealed audit was not authorized by every clean gate")
        for key in ("clean_result", "clean_manifest", "frozen_composite"):
            reference = authorization[key]
            path = ROOT / reference["path"]
            if digest(path) != reference["sha256"]:
                raise ValueError(f"sealed authorization dependency changed: {path}")
        if authorization["frozen_composite"]["sha256"] != composite_sha:
            raise ValueError("sealed audit composite differs from clean authorization")
    dependencies = {
        name: resolve(value)
        for name, value in composite["dependencies"].items()
    }
    model_selection = ROOT / composite["model_selection_evidence"]["path"]
    if digest(model_selection) != composite["model_selection_evidence"]["sha256"]:
        raise ValueError("frozen v12b model-selection report changed")
    rows = [
        json.loads(line)
        for line in args.dataset.read_text().splitlines()
        if line
    ]
    seed_start, seed_stop = (900, 1000) if args.sealed_audit else (1900, 1940)
    expected_split = (
        "sealed_audit_v12b"
        if args.sealed_audit
        else "fresh_v12b_clean_development"
    )
    expected_samples = 3 * (seed_stop - seed_start)
    if len(rows) != expected_samples or sorted(
        set(int(row["seed"]) for row in rows)
    ) != list(range(seed_start, seed_stop)):
        raise ValueError(
            f"v12b {'sealed' if args.sealed_audit else 'clean'} data must "
            f"contain seeds {seed_start}-{seed_stop - 1} x 3 regimes"
        )
    expected_order = [
        (regime, seed)
        for regime in ("revealed_full", "empty_full", "failed_truncated_control")
        for seed in range(seed_start, seed_stop)
    ]
    observed_order = [(row["regime"], int(row["seed"])) for row in rows]
    if observed_order != expected_order:
        raise ValueError("v12b rows violate the frozen regime-major call order")
    for row in rows:
        if row["schema_version"] != (
            "interactive-perception.open-and-observe-effect.v12b"
        ):
            raise ValueError("row does not use the v12b schema")
        if row["split"] != expected_split:
            raise ValueError("row does not declare the frozen v12b split")
        if row["frozen_critic_sha256"] != composite_sha:
            raise ValueError("row was not collected under the frozen v12b composite")
        if row["online_oracle_inputs"]:
            raise ValueError("row declares a privileged controller input")

    agent_cal, agent_scores = load_head(
        dependencies["agentview_target_artifact"],
        dependencies["agentview_target_checkpoint"],
        rows,
        "agentview",
        "target",
        torch,
        Image,
    )
    wrist_cal, wrist_scores = load_head(
        dependencies["wrist_target_artifact"],
        dependencies["wrist_target_checkpoint"],
        rows,
        "wrist",
        "target",
        torch,
        Image,
    )
    coverage_cal, coverage_scores = load_head(
        dependencies["coverage_artifact"],
        dependencies["coverage_checkpoint"],
        rows,
        "agentview",
        "coverage",
        torch,
        Image,
    )

    truth: list[str] = []
    predictions: list[tuple[str, ...]] = []
    records = []
    target_sources: dict[str, int] = {}
    for row in rows:
        agent_logit = episode_score(row, agent_scores, ("agentview",))
        wrist_logit = episode_score(row, wrist_scores, ("wrist",))
        coverage_logit = episode_score(row, coverage_scores, ("agentview",))
        agent_set = tuple(agent_cal.predict(binary_evidence(agent_logit)))
        wrist_set = tuple(wrist_cal.predict(binary_evidence(wrist_logit)))
        coverage_set = tuple(
            coverage_cal.predict(
                binary_evidence(coverage_logit, "COMPLETED", "FAILED")
            )
        )
        outcome, source = resolve_v12b_cascade(
            agent_set, wrist_set, coverage_set
        )
        target_sources[source] = target_sources.get(source, 0) + 1
        observed = v10_truth(row)
        truth.append(observed)
        predictions.append(outcome)
        records.append(
            {
                "regime": row["regime"],
                "seed": int(row["seed"]),
                "truth": observed,
                "agentview_target_set": list(agent_set),
                "wrist_target_set": list(wrist_set),
                "target_source": source,
                "coverage_set": list(coverage_set),
                "outcome_prediction_set": list(outcome),
                "controller_action_for_multilabel": (
                    "SAFE_STOP" if len(outcome) != 1 else None
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
    ambiguous = sum(len(prediction) != 1 for prediction in predictions)
    mean_set_size = sum(map(len, predictions)) / len(predictions)
    physical = {
        "REVEALED": physical_branch(
            rows, truth, "revealed_full", "REVEALED", args.confidence
        ),
        "EMPTY": physical_branch(
            rows, truth, "empty_full", "EMPTY", args.confidence
        ),
    }
    revealed_rows = [row for row in rows if row["regime"] == "revealed_full"]
    six_frame = sum(v10_truth(row) == "REVEALED" for row in revealed_rows)
    final_frame = sum(
        max(row["evaluator_only"]["visibility_history"][-1]["target_pixels"].values())
        >= PI05_PATCH_EQUIVALENT_TARGET_PIXELS
        for row in revealed_rows
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
    result = {
        "schema_version": (
            "interactive-perception.t01-v12b-sealed-audit.v1"
            if args.sealed_audit
            else "interactive-perception.t01-v12b-clean-development.v1"
        ),
        "decision": "GO" if passed else "NOT_GO",
        "sealed_audit_authorized": passed if not args.sealed_audit else None,
        "sealed_audit_passed": passed if args.sealed_audit else None,
        "dataset": {
            "path": str(args.dataset.relative_to(ROOT)),
            "sha256": digest(args.dataset),
        },
        "frozen_composite": {
            "path": str(args.composite.relative_to(ROOT)),
            "sha256": composite_sha,
        },
        "samples": len(rows),
        "seeds": f"{seed_start}-{seed_stop - 1}",
        "per_class": per_class,
        "false_singleton_EMPTY": false_empty,
        "false_singleton_REVEALED": false_revealed,
        "mean_prediction_set_size": mean_set_size,
        "ambiguous_safe_stop_count": ambiguous,
        "ambiguous_safe_stop_rate": ambiguous / len(predictions),
        "target_sources": target_sources,
        "physical_information_acquisition": physical,
        "history_ablation": {
            "six_frame_resolvable_reveals": six_frame,
            "final_frame_only_resolvable_reveals": final_frame,
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
            "proves_if_GO": (
                f"{'sealed' if args.sealed_audit else 'fresh clean'} T01 "
                "public-RGB OPEN_AND_OBSERVE outcome recognition "
                "for the target-observability endpoint"
            ),
            "does_not_prove": [
                "final butter placement success",
                "global NOT_FOUND",
                "cross-scene or cross-object generalization",
                "broad pi0.5 cross-seed robustness",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "decision",
                    "per_class",
                    "false_singleton_EMPTY",
                    "false_singleton_REVEALED",
                    "mean_prediction_set_size",
                    "ambiguous_safe_stop_rate",
                    "physical_information_acquisition",
                    "history_ablation",
                    "gates",
                )
            },
            indent=2,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
