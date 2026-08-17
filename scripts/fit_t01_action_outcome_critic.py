#!/usr/bin/env python3
"""Fit and freeze the paired-RGB T01 action-outcome critic."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interactive_perception.action_outcome import (  # noqa: E402
    BalancedRidgeMulticlass,
    FeatureStandardizer,
    transition_feature_block,
)
from interactive_perception.capability_gate import exact_binomial_lower_bound  # noqa: E402
from interactive_perception.semantic_conformal import (  # noqa: E402
    MondrianSemanticConformalCalibrator,
)

BLOCKS = ("after_prompt", "prompt_delta", "prompt_history", "all_delta", "all_history")
SPATIAL_BLOCKS = (
    "after",
    "delta",
    "history",
    "spatial_after",
    "spatial_delta",
    "spatial_history",
)
REGULARIZATION_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def balanced_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.mean(
            [np.mean(prediction[truth == label] == label) for label in sorted(set(truth))]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--spatial-v2", action="store_true")
    parser.add_argument("--standardized-v3", action="store_true")
    parser.add_argument("--cognitive-v4", action="store_true")
    args = parser.parse_args()
    if args.cognitive_v4:
        args.standardized_v3 = True
    if args.standardized_v3:
        args.spatial_v2 = True
    if args.embeddings is None:
        args.embeddings = ROOT / (
            "outputs/t01_action_effect_v4/pi05_transition_cognitive_query_embeddings.npz"
            if args.cognitive_v4
            else "outputs/t01_action_effect_v3/pi05_transition_target_query_spatial_embeddings.npz"
            if args.standardized_v3
            else "outputs/t01_action_effect_v2/pi05_transition_spatial_embeddings.npz"
            if args.spatial_v2
            else "outputs/t01_action_effect_v1/pi05_transition_embeddings.npz"
        )
    if args.output is None:
        args.output = ROOT / (
            "results/calibration/t01_action_outcome_critic_v4.json"
            if args.cognitive_v4
            else "results/calibration/t01_action_outcome_critic_v3.json"
            if args.standardized_v3
            else "results/calibration/t01_action_outcome_critic_v2.json"
            if args.spatial_v2
            else "results/calibration/t01_action_outcome_critic_v1.json"
        )
    for name in ("embeddings", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)

    data = np.load(args.embeddings)
    before = np.asarray(data["before_features"], dtype=np.float64)
    after = np.asarray(data["after_features"], dtype=np.float64)
    labels = np.asarray(data["outcome"], dtype=str)
    intended = np.asarray(data["intended_outcome"], dtype=str)
    seeds = np.asarray(data["seed"], dtype=np.int64)
    splits = np.asarray(data["split"], dtype=str)
    full_executor = np.asarray(data["full_executor"], dtype=bool)
    physical_subtype = np.asarray(data["physical_outcome_subtype"], dtype=str)
    if args.spatial_v2:
        train = (seeds >= 400) & (seeds <= 419)
        head_refit = train
        calibration = (seeds >= 420) & (seeds <= 452)
        validation = (seeds >= 453) & (seeds <= 459)
        blocks = SPATIAL_BLOCKS
    else:
        train = splits == "prototype_train"
        head_refit = train | (splits == "probability_calibration")
        calibration = splits == "conformal_calibration"
        validation = splits == "heldout_validation"
        blocks = BLOCKS
    if not all(mask.any() for mask in (train, head_refit, calibration, validation)):
        raise ValueError("one or more frozen development splits are empty")

    searches = []
    for block in blocks:
        values = transition_feature_block(before, after, block)
        for regularization in REGULARIZATION_GRID:
            fold_scores = []
            for fold in range(5):
                fit_mask = train & (seeds % 5 != fold)
                test_mask = train & (seeds % 5 == fold)
                standardizer = (
                    FeatureStandardizer.fit(values[fit_mask])
                    if args.standardized_v3
                    else None
                )
                fit_values = (
                    standardizer.transform(values[fit_mask])
                    if standardizer is not None
                    else values[fit_mask]
                )
                test_values = (
                    standardizer.transform(values[test_mask])
                    if standardizer is not None
                    else values[test_mask]
                )
                model = BalancedRidgeMulticlass.fit(
                    fit_values,
                    labels[fit_mask],
                    regularization=regularization,
                )
                fold_scores.append(
                    balanced_accuracy(labels[test_mask], model.predict(test_values))
                )
            searches.append(
                {
                    "block": block,
                    "regularization": regularization,
                    "mean_grouped_cv_balanced_accuracy": float(np.mean(fold_scores)),
                    "fold_balanced_accuracy": fold_scores,
                    "dimension": int(values.shape[1]),
                }
            )
    selected = sorted(
        searches,
        key=lambda item: (
            -item["mean_grouped_cv_balanced_accuracy"],
            item["dimension"],
            item["regularization"],
            item["block"],
        ),
    )[0]
    values = transition_feature_block(before, after, str(selected["block"]))
    standardizer = (
        FeatureStandardizer.fit(values[head_refit])
        if args.standardized_v3
        else None
    )
    model_values = (
        standardizer.transform(values) if standardizer is not None else values
    )
    model = BalancedRidgeMulticlass.fit(
        model_values[head_refit],
        labels[head_refit],
        regularization=float(selected["regularization"]),
    )
    conformal = MondrianSemanticConformalCalibrator.fit(
        list(
            zip(
                model.evidence(model_values[calibration]),
                labels[calibration],
                strict=True,
            )
        ),
        alpha=args.alpha,
        policy_id="pi05_libero_prefix_transition",
        split_id=(
            "t01_action_effect_v4_cognitive_query_seed420_452"
            if args.cognitive_v4
            else "t01_action_effect_v3_target_query_seed420_452"
            if args.standardized_v3
            else "t01_action_effect_v2_seed420_452"
            if args.spatial_v2
            else "t01_action_effect_v1_seed430_439"
        ),
    )
    if args.spatial_v2 and min(conformal.calibration_size_per_class.values()) < 30:
        raise ValueError("spatial v2 requires at least 30 conformal examples per class")
    validation_evidence = model.evidence(model_values[validation])
    prediction_sets = [conformal.predict(item) for item in validation_evidence]
    truth = labels[validation]
    covered = np.asarray(
        [label in predicted for label, predicted in zip(truth, prediction_sets, strict=True)]
    )
    singleton = np.asarray(
        [predicted == (label,) for label, predicted in zip(truth, prediction_sets, strict=True)]
    )
    per_class = {}
    for label in model.labels:
        mask = truth == label
        per_class[label] = {
            "trials": int(mask.sum()),
            "coverage": float(np.mean(covered[mask])),
            "singleton_accuracy": float(np.mean(singleton[mask])),
            "mean_set_size": float(
                np.mean(
                    [
                        len(predicted)
                        for predicted, keep in zip(prediction_sets, mask, strict=True)
                        if keep
                    ]
                )
            ),
        }
    development_passed = bool(
        min(item["coverage"] for item in per_class.values()) >= 0.9
        and min(item["singleton_accuracy"] for item in per_class.values()) >= 0.9
    )

    physical_validation = {}
    for label in ("REVEALED", "EMPTY"):
        mask = validation & full_executor & (intended == label)
        successes = int(np.sum(labels[mask] == label))
        trials = int(mask.sum())
        lower = exact_binomial_lower_bound(successes, trials, 0.95)
        physical_validation[label] = {
            "successes": successes,
            "trials": trials,
            "empirical_rate": successes / trials,
            "one_sided_95_lower": lower,
            "passes_0.90": lower >= 0.9,
        }

    artifact = {
        "schema_version": (
            "interactive-perception.action-outcome-critic.v4"
            if args.cognitive_v4
            else "interactive-perception.action-outcome-critic.v3"
            if args.standardized_v3
            else "interactive-perception.action-outcome-critic.v2"
            if args.spatial_v2
            else "interactive-perception.action-outcome-critic.v1"
        ),
        "claim": "paired policy-RGB recognition of FAILED, REVEALED, and EMPTY",
        "non_claim": "scene-disjoint generalization or final task completion",
        "development_validation_note": (
            "Seeds 453-459 were inspected during outcome-head development and are "
            "diagnostic only; they cannot authorize the frozen audit. A new "
            "post-freeze confirmation split is required."
            if args.standardized_v3
            else "frozen development sanity split"
        ),
        "embeddings": str(args.embeddings.relative_to(ROOT)),
        "embeddings_sha256": digest(args.embeddings),
        "input_feature_dimension": int(before.shape[1]),
        "spatial_v2": args.spatial_v2,
        "perception_subtask_prompt": args.standardized_v3,
        "target_query_v3": bool(args.standardized_v3 and not args.cognitive_v4),
        "cognitive_query_v4": args.cognitive_v4,
        "feature_standardizer": (
            standardizer.to_dict() if standardizer is not None else None
        ),
        "alpha": args.alpha,
        "labels": list(model.labels),
        "split_contract": (
            {
                "prototype_train": "seeds 400-419",
                "head_refit_addition": "none",
                "conformal_calibration": "seeds 420-452; at least 30 examples per observed class",
                "heldout_validation": "seeds 453-459; development sanity only",
                "future_frozen_audit": "seeds 500-599; not collected before this artifact is frozen",
            }
            if args.spatial_v2
            else {
                "prototype_train": "seeds 400-419",
                "head_refit_addition": "seeds 420-429",
                "conformal_calibration": "seeds 430-439",
                "heldout_validation": "seeds 440-459",
                "future_frozen_audit": "seeds 500-599; not collected before this artifact is frozen",
            }
        ),
        "model_selection": {
            "rule": (
                "five-fold grouped-by-seed balanced accuracy on prototype_train only; "
                "feature standardization is fitted inside each training fold"
                if args.standardized_v3
                else "five-fold grouped-by-seed balanced accuracy on prototype_train only"
            ),
            "candidates": searches,
            "selected": selected,
        },
        "critic": model.to_dict(),
        "conformal": conformal.to_dict(),
        "validation": {
            "trials": int(validation.sum()),
            "coverage": float(np.mean(covered)),
            "singleton_accuracy": float(np.mean(singleton)),
            "mean_set_size": float(np.mean([len(value) for value in prediction_sets])),
            "per_class": per_class,
            "physical_full_executor": physical_validation,
            "physical_failure_subtypes": {
                label: int(
                    np.sum(validation & full_executor & (physical_subtype == label))
                )
                for label in ("NO_EFFECT", "OPENED_UNOBSERVED")
            },
            "passed_development_critic_gate": development_passed,
            "passed_strict_physical_effect_gate": all(
                item["passes_0.90"] for item in physical_validation.values()
            ),
        },
        "online_inputs": ["paired stock policy RGB", "prompt", "action label"],
        "online_oracle_inputs": [],
        "evaluator_only_diagnostic_note": (
            "FAILED is split after evaluation into NO_EFFECT versus "
            "OPENED_UNOBSERVED; neither subtype enters the critic input"
        ),
        "fp3_passed": False,
        "fp3_note": "requires the frozen 100-seed-per-regime audit before it can pass",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "selected": selected,
                "validation": artifact["validation"],
                "artifact": str(args.output),
            },
            indent=2,
        )
    )
    if not development_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
