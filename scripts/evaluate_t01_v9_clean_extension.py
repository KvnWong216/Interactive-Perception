#!/usr/bin/env python3
"""Evaluate the frozen v9 candidate once on seeds 660--699 without refitting."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interactive_perception.action_outcome import (  # noqa: E402
    HierarchicalActionOutcomePredictor,
    PI05_PATCH_EQUIVALENT_TARGET_PIXELS,
    label_temporal_information_outcome,
)
from interactive_perception.capability_gate import (  # noqa: E402
    exact_binomial_lower_bound,
)
from interactive_perception.seed_registry import (  # noqa: E402
    load_seed_registry,
    seeds_for_blocks,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observable_outcome(row: dict) -> str:
    evaluator = row["evaluator_only"]
    return label_temporal_information_outcome(
        full_executor=bool(row["full_executor"]),
        opened=bool(evaluator["drawer_opened"]),
        return_complete=row["return_status"]["phase"] == "COMPLETE",
        target_pixel_history=tuple(
            tuple(point["target_pixels"].values())
            for point in evaluator["visibility_history"]
        ),
        minimum_target_pixels=PI05_PATCH_EQUIVALENT_TARGET_PIXELS,
        empty_coverage_certified=bool(
            evaluator["empty_counterfactual_reveal_certified"]
        ),
    ).value


def prediction_metrics(
    truth: np.ndarray,
    predictions: list[tuple[str, ...]],
    *,
    confidence: float,
    minimum_singleton_lower: float,
    alpha: float,
) -> dict:
    per_class = {}
    passed = True
    for label in ("FAILED", "REVEALED", "EMPTY"):
        mask = truth == label
        selected = [item for item, keep in zip(predictions, mask, strict=True) if keep]
        trials = len(selected)
        if trials == 0:
            raise ValueError(f"clean extension contains no {label} ground truth")
        covered = sum(label in item for item in selected)
        singleton = sum(item == (label,) for item in selected)
        singleton_lower = exact_binomial_lower_bound(singleton, trials, confidence)
        row = {
            "trials": trials,
            "covered": covered,
            "coverage": covered / trials,
            "singleton_correct": singleton,
            "singleton_rate": singleton / trials,
            "singleton_one_sided_95_lower": singleton_lower,
            "mean_prediction_set_size": float(
                np.mean([len(item) for item in selected])
            ),
            "prediction_set_histogram": dict(
                sorted(collections.Counter("|".join(item) for item in selected).items())
            ),
            "passed": (
                covered / trials >= 1.0 - alpha
                and singleton_lower >= minimum_singleton_lower
            ),
        }
        per_class[label] = row
        passed = passed and row["passed"]
    ambiguous = sum(len(prediction) != 1 for prediction in predictions)
    return {
        "per_class": per_class,
        "mean_prediction_set_size": float(
            np.mean([len(item) for item in predictions])
        ),
        "ambiguous_safe_stop_count": ambiguous,
        "ambiguous_safe_stop_rate": ambiguous / len(predictions),
        "false_empty": sum(
            prediction == ("EMPTY",) and observed != "EMPTY"
            for prediction, observed in zip(predictions, truth, strict=True)
        ),
        "false_revealed": sum(
            prediction == ("REVEALED",) and observed != "REVEALED"
            for prediction, observed in zip(predictions, truth, strict=True)
        ),
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_critic_v9_candidate_visual.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT
        / "data/calibration/t01_open_and_observe_effect_v4_extension.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "data/calibration/t01_open_and_observe_effect_v4_extension.manifest.json",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT
        / "outputs/t01_open_and_observe_effect_v4_extension/pi05_temporal_embeddings_v5.npz",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_clean_extension_v9.json",
    )
    parser.add_argument(
        "--frozen-artifact",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_critic_v9.json",
    )
    parser.add_argument(
        "--effect-candidate",
        type=Path,
        default=ROOT / "results/calibration/t01_open_and_observe_effect_v3.json",
    )
    parser.add_argument(
        "--full-history-artifact",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_critic_v9_candidate.json",
    )
    parser.add_argument(
        "--global-history-artifact",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_critic_v9_candidate_global.json",
    )
    parser.add_argument(
        "--no-history-artifact",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_critic_v9_candidate_no_history.json",
    )
    parser.add_argument(
        "--frozen-effect-artifact",
        type=Path,
        default=ROOT / "results/calibration/t01_open_and_observe_effect_v4.json",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--minimum-singleton-lower", type=float, default=0.8)
    parser.add_argument("--minimum-action-lower", type=float, default=0.8)
    args = parser.parse_args()
    for name in (
        "candidate",
        "dataset",
        "manifest",
        "embeddings",
        "result",
        "frozen_artifact",
        "effect_candidate",
        "full_history_artifact",
        "global_history_artifact",
        "no_history_artifact",
        "frozen_effect_artifact",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if (
        args.result.exists()
        or args.frozen_artifact.exists()
        or args.frozen_effect_artifact.exists()
    ):
        raise FileExistsError("clean-extension outputs are immutable")

    registry_path = ROOT / "benchmarks/rss_v1/seed_registry.yaml"
    registry = load_seed_registry(registry_path)
    expected_seeds = seeds_for_blocks(
        registry, ["t01_open_observe_clean_extension"]
    )
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema_version") != "interactive-perception.open-and-observe-manifest.v4":
        raise ValueError("clean extension requires the v4 manifest")
    if manifest.get("phase") != "heldout_development_extension":
        raise ValueError("manifest is not the clean development extension")
    if manifest.get("seeds") != expected_seeds:
        raise ValueError("manifest seeds differ from the authoritative registry")
    if manifest.get("dataset_sha256") != digest(args.dataset):
        raise ValueError("dataset hash differs from its immutable manifest")

    candidate = json.loads(args.candidate.read_text())
    if (
        candidate.get("schema_version")
        != "interactive-perception.action-outcome-critic.v9-candidate"
    ):
        raise ValueError("expected the frozen v9 candidate")
    if candidate.get("development", {}).get("passed", False):
        raise ValueError("candidate unexpectedly contains a passed development decision")
    if manifest.get("audit_artifact_sha256") != digest(args.candidate):
        raise ValueError("clean extension was not collected against this frozen candidate")
    predictor = HierarchicalActionOutcomePredictor.from_artifact(candidate)
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line]
    data = np.load(args.embeddings)
    expected_row_seeds = np.asarray([row["seed"] for row in rows], dtype=np.int64)
    expected_regimes = np.asarray([row["regime"] for row in rows], dtype=str)
    if len(rows) != 3 * len(expected_seeds):
        raise ValueError("clean extension must contain three regimes per seed")
    if not np.array_equal(np.asarray(data["seed"]), expected_row_seeds):
        raise ValueError("embedding seeds are not aligned to the dataset")
    if not np.array_equal(data["regime"].astype(str), expected_regimes):
        raise ValueError("embedding regimes are not aligned to the dataset")

    truth = np.asarray([observable_outcome(row) for row in rows], dtype=str)
    predictions = [
        predictor.predict_history(history, robot).prediction_set
        for history, robot in zip(
            data["history_features"], data["robot_state_history"], strict=True
        )
    ]
    primary_metrics = prediction_metrics(
        truth,
        predictions,
        confidence=args.confidence,
        minimum_singleton_lower=args.minimum_singleton_lower,
        alpha=predictor.effect_conformal.alpha,
    )
    per_class = primary_metrics["per_class"]
    critic_passed = bool(primary_metrics["passed"])

    ablations = {}
    for name, path in (
        ("full_visual_plus_robot_history", args.full_history_artifact),
        ("global_visual_history", args.global_history_artifact),
        ("no_history_final_frame", args.no_history_artifact),
    ):
        artifact = json.loads(path.read_text())
        ablation_predictor = HierarchicalActionOutcomePredictor.from_artifact(artifact)
        ablation_predictions = [
            ablation_predictor.predict_history(history, robot).prediction_set
            for history, robot in zip(
                data["history_features"], data["robot_state_history"], strict=True
            )
        ]
        ablations[name] = {
            "artifact": str(path.relative_to(ROOT)),
            "artifact_sha256": digest(path),
            **prediction_metrics(
                truth,
                ablation_predictions,
                confidence=args.confidence,
                minimum_singleton_lower=args.minimum_singleton_lower,
                alpha=ablation_predictor.effect_conformal.alpha,
            ),
        }

    visibility_by_history_point = {}
    for index, name in enumerate(
        [point["name"] for point in rows[0]["evaluator_only"]["visibility_history"]]
    ):
        visible = [
            max(row["evaluator_only"]["visibility_history"][index]["target_pixels"].values())
            >= PI05_PATCH_EQUIVALENT_TARGET_PIXELS
            for row in rows
        ]
        visibility_by_history_point[name] = {
            "all_rows": sum(visible),
            "revealed_full_rows": sum(
                item and row["intended_outcome"] == "REVEALED"
                for item, row in zip(visible, rows, strict=True)
            ),
            "trials": len(rows),
        }

    final_frame_predictions = []
    for row in rows:
        evaluator = row["evaluator_only"]
        final_visible = (
            max(evaluator["visibility_history"][-1]["target_pixels"].values())
            >= PI05_PATCH_EQUIVALENT_TARGET_PIXELS
        )
        if not row["full_executor"] or not evaluator["drawer_opened"]:
            label = "FAILED"
        elif final_visible:
            label = "REVEALED"
        elif (
            row["return_status"]["phase"] == "COMPLETE"
            and evaluator["empty_counterfactual_reveal_certified"]
        ):
            label = "EMPTY"
        else:
            label = "FAILED"
        final_frame_predictions.append((label,))
    final_frame_endpoint_baseline = prediction_metrics(
        truth,
        final_frame_predictions,
        confidence=args.confidence,
        minimum_singleton_lower=args.minimum_singleton_lower,
        alpha=predictor.effect_conformal.alpha,
    )

    physical = {}
    physical_passed = True
    for intended in ("REVEALED", "EMPTY"):
        selected = [
            observed
            for row, observed in zip(rows, truth, strict=True)
            if row["full_executor"] and row["intended_outcome"] == intended
        ]
        successes = sum(observed == intended for observed in selected)
        lower = exact_binomial_lower_bound(successes, len(selected), args.confidence)
        result = {
            "successes": successes,
            "trials": len(selected),
            "empirical_rate": successes / len(selected),
            "one_sided_95_lower": lower,
            "passes_0.80": lower >= args.minimum_action_lower,
            "passes_original_0.90": lower >= 0.9,
        }
        physical[intended] = result
        physical_passed = physical_passed and result["passes_0.80"]

    passed = bool(critic_passed and physical_passed)
    report = {
        "schema_version": "interactive-perception.v9-clean-extension.v1",
        "status": "GO" if passed else "NOT_GO",
        "paper_eligible": False,
        "candidate": str(args.candidate.relative_to(ROOT)),
        "candidate_sha256": digest(args.candidate),
        "effect_candidate": str(args.effect_candidate.relative_to(ROOT)),
        "effect_candidate_sha256": digest(args.effect_candidate),
        "dataset": str(args.dataset.relative_to(ROOT)),
        "dataset_sha256": digest(args.dataset),
        "manifest": str(args.manifest.relative_to(ROOT)),
        "manifest_sha256": digest(args.manifest),
        "embeddings": str(args.embeddings.relative_to(ROOT)),
        "embeddings_sha256": digest(args.embeddings),
        "seed_registry": str(registry_path.relative_to(ROOT)),
        "seed_blocks": ["t01_open_observe_clean_extension"],
        "seeds": expected_seeds,
        "critic": per_class,
        "selected_input": "visual-only six-point temporal history",
        "ablations": ablations,
        "visibility_by_history_point": visibility_by_history_point,
        "final_frame_only_evaluator_endpoint_baseline": final_frame_endpoint_baseline,
        "physical": physical,
        "mean_prediction_set_size": primary_metrics["mean_prediction_set_size"],
        "ambiguous_safe_stop_count": primary_metrics[
            "ambiguous_safe_stop_count"
        ],
        "ambiguous_safe_stop_rate": primary_metrics["ambiguous_safe_stop_rate"],
        "false_empty": primary_metrics["false_empty"],
        "false_revealed": primary_metrics["false_revealed"],
        "online_oracle_inputs": [],
        "sealed_audit_opened": False,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(report, indent=2) + "\n")
    if passed:
        frozen = dict(candidate)
        frozen["schema_version"] = "interactive-perception.action-outcome-critic.v9"
        frozen["clean_development"] = report
        frozen["development"] = {
            "passed": True,
            "decision_source": str(args.result.relative_to(ROOT)),
            "decision_source_sha256": digest(args.result),
        }
        frozen["fp3_passed"] = False
        args.frozen_artifact.write_text(json.dumps(frozen, indent=2) + "\n")
        effect = json.loads(args.effect_candidate.read_text())
        if not effect.get("physical_effect_gate_passed", False):
            raise ValueError("the pre-extension physical effect candidate is NOT-GO")
        effect["schema_version"] = "interactive-perception.temporal-effect-registry.v4"
        effect["clean_development"] = {
            "source": str(args.result.relative_to(ROOT)),
            "source_sha256": digest(args.result),
            "physical": physical,
            "passed": physical_passed,
            "model_refit_on_extension": False,
        }
        effect["physical_effect_gate_passed"] = physical_passed
        args.frozen_effect_artifact.write_text(json.dumps(effect, indent=2) + "\n")
    print(
        json.dumps(
            {"status": report["status"], "critic": per_class, "physical": physical},
            indent=2,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
