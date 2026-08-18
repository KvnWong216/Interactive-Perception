#!/usr/bin/env python3
"""Fit the v9 candidate: completion first, then resolvable content."""

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
    HierarchicalActionOutcomePredictor,
    PI05_PATCH_EQUIVALENT_TARGET_PIXELS,
    label_temporal_information_outcome,
    temporal_history_feature_block,
)
from interactive_perception.semantic_conformal import (  # noqa: E402
    MondrianSemanticConformalCalibrator,
)

GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def balanced_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.mean(
            [
                np.mean(prediction[truth == label] == label)
                for label in sorted(set(truth))
            ]
        )
    )


def select_regularization(values, labels, train, seeds):
    rows = []
    for regularization in GRID:
        scores = []
        for fold in range(5):
            fit = train & (seeds % 5 != fold)
            check = train & (seeds % 5 == fold)
            scaler = FeatureStandardizer.fit(values[fit])
            model = BalancedRidgeMulticlass.fit(
                scaler.transform(values[fit]),
                labels[fit],
                regularization=regularization,
            )
            scores.append(
                balanced_accuracy(
                    labels[check], model.predict(scaler.transform(values[check]))
                )
            )
        rows.append(
            {
                "regularization": regularization,
                "fold_balanced_accuracy": scores,
                "mean_grouped_cv_balanced_accuracy": float(np.mean(scores)),
            }
        )
    selected = sorted(
        rows,
        key=lambda row: (
            -row["mean_grouped_cv_balanced_accuracy"],
            row["regularization"],
        ),
    )[0]
    return selected, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/calibration/t01_open_and_observe_effect_v3.jsonl",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT
        / "outputs/t01_open_and_observe_effect_v3/pi05_temporal_embeddings_v5.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_critic_v9_candidate.json",
    )
    parser.add_argument(
        "--block",
        choices=(
            "temporal_history",
            "visual_only_history",
            "global_visual_history",
            "no_history",
        ),
        default="temporal_history",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    for name in ("dataset", "embeddings", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)

    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line]
    data = np.load(args.embeddings)
    if len(rows) != len(data["seed"]):
        raise ValueError("dataset and embeddings are not aligned")
    history = np.asarray(data["history_features"], dtype=np.float64)
    robot = np.asarray(data["robot_state_history"], dtype=np.float64)
    seeds = np.asarray(data["seed"], dtype=np.int64)
    values = temporal_history_feature_block(history, robot, args.block)

    corrected_outcomes = []
    correction_reasons = []
    for row in rows:
        outcome = label_temporal_information_outcome(
            full_executor=bool(row["full_executor"]),
            opened=bool(row["evaluator_only"]["drawer_opened"]),
            return_complete=row["return_status"]["phase"] == "COMPLETE",
            target_pixel_history=tuple(
                tuple(point["target_pixels"].values())
                for point in row["evaluator_only"]["visibility_history"]
            ),
            minimum_target_pixels=PI05_PATCH_EQUIVALENT_TARGET_PIXELS,
            empty_coverage_certified=bool(
                row["evaluator_only"]["empty_counterfactual_reveal_certified"]
            ),
        ).value
        corrected_outcomes.append(outcome)
        if outcome == "REVEALED":
            correction_reasons.append("target visibly acquired")
        elif outcome == "EMPTY":
            correction_reasons.append(
                "opened searched region plus completed calibrated observation return with no target evidence"
            )
        else:
            correction_reasons.append("no certified prompt-relevant information")
    corrected = np.asarray(corrected_outcomes, dtype=str)
    effect_labels = np.where(corrected == "FAILED", "FAILED", "COMPLETED")
    content_labels = corrected.copy()

    train = (seeds >= 600) & (seeds <= 619)
    calibration = (seeds >= 620) & (seeds <= 652)
    development = (seeds >= 653) & (seeds <= 659)
    content_train = train & (effect_labels == "COMPLETED")
    content_calibration = calibration & (effect_labels == "COMPLETED")

    effect_selected, effect_search = select_regularization(
        values, effect_labels, train, seeds
    )
    content_selected, content_search = select_regularization(
        values, content_labels, content_train, seeds
    )
    scaler = FeatureStandardizer.fit(values[train])
    scaled = scaler.transform(values)
    effect_model = BalancedRidgeMulticlass.fit(
        scaled[train],
        effect_labels[train],
        regularization=float(effect_selected["regularization"]),
    )
    content_model = BalancedRidgeMulticlass.fit(
        scaled[content_train],
        content_labels[content_train],
        regularization=float(content_selected["regularization"]),
    )
    effect_conformal = MondrianSemanticConformalCalibrator.fit(
        list(
            zip(
                effect_model.evidence(scaled[calibration]),
                effect_labels[calibration],
                strict=True,
            )
        ),
        alpha=args.alpha,
        policy_id="pi05_libero_hierarchical_effect_v9",
        split_id="t01_effect_seed620_652",
    )
    content_conformal = MondrianSemanticConformalCalibrator.fit(
        list(
            zip(
                content_model.evidence(scaled[content_calibration]),
                content_labels[content_calibration],
                strict=True,
            )
        ),
        alpha=args.alpha,
        policy_id="pi05_libero_hierarchical_content_v9",
        split_id="t01_content_seed620_652",
    )
    if min(effect_conformal.calibration_size_per_class.values()) < 30:
        raise ValueError("effect head requires at least 30 examples per class")
    if min(content_conformal.calibration_size_per_class.values()) < 30:
        raise ValueError("content head requires at least 30 examples per class")

    artifact = {
        "schema_version": "interactive-perception.action-outcome-critic.v9-candidate",
        "claim": "hierarchical physical completion then prompt-resolvable content",
        "dataset": str(args.dataset.relative_to(ROOT)),
        "dataset_sha256": digest(args.dataset),
        "embeddings": str(args.embeddings.relative_to(ROOT)),
        "embeddings_sha256": digest(args.embeddings),
        "input_feature_dimension": int(history.shape[2]),
        "model_selection": {
            "block": args.block,
            "effect": effect_selected,
            "content": content_selected,
            "effect_candidates": effect_search,
            "content_candidates": content_search,
        },
        "feature_standardizer": scaler.to_dict(),
        "effect_head": {
            "labels": ["FAILED", "COMPLETED"],
            "critic": effect_model.to_dict(),
            "conformal": effect_conformal.to_dict(),
        },
        "content_head": {
            "condition": "effect includes COMPLETED",
            "labels": ["REVEALED", "EMPTY"],
            "critic": content_model.to_dict(),
            "conformal": content_conformal.to_dict(),
        },
        "label_contract": {
            "REVEALED": (
                "target occupies at least one pi0.5 visual-token footprint "
                "at any public history point"
            ),
            "EMPTY": "target never visible; region open; return complete; visibility coverage certified",
            "FAILED": "no certified prompt-relevant information",
            "minimum_resolvable_target_pixels": PI05_PATCH_EQUIVALENT_TARGET_PIXELS,
            "threshold_derivation": "(256 policy pixels / 16 visual tokens per side)^2",
            "change_from_v8": (
                "five-pixel simulator visibility is retained as a diagnostic but "
                "does not count as prompt-resolvable evidence"
            ),
            "corrected_rows": [
                {
                    "regime": row["regime"],
                    "seed": row["seed"],
                    "old": row["outcome"],
                    "new": new,
                    "reason": reason,
                }
                for row, new, reason in zip(
                    rows, corrected_outcomes, correction_reasons, strict=True
                )
                if row["outcome"] != new
            ],
        },
        "online_inputs": [
            "six stock policy RGB pairs",
            "six public robot-state vectors",
            "final prompt",
            "executed option role",
        ],
        "online_oracle_inputs": [],
        "fp3_passed": False,
    }
    predictor = HierarchicalActionOutcomePredictor.from_artifact(artifact)
    prediction_sets = [
        predictor.predict_history(history[index], robot[index]).prediction_set
        for index in np.flatnonzero(development)
    ]
    truth = corrected[development]
    per_class = {}
    for label in ("FAILED", "REVEALED", "EMPTY"):
        mask = truth == label
        selected_sets = [
            item
            for item, keep in zip(prediction_sets, mask, strict=True)
            if keep
        ]
        per_class[label] = {
            "trials": int(mask.sum()),
            "coverage": float(
                np.mean([label in item for item in selected_sets])
            ),
            "singleton_accuracy": float(
                np.mean([item == (label,) for item in selected_sets])
            ),
            "mean_set_size": float(np.mean([len(item) for item in selected_sets])),
        }
    development_passed = bool(
        min(item["coverage"] for item in per_class.values()) >= 0.9
        and min(item["singleton_accuracy"] for item in per_class.values()) >= 0.8
    )
    diagnostic_rows = [
        {
            "regime": rows[index]["regime"],
            "seed": int(rows[index]["seed"]),
            "truth": str(label),
            "prediction_set": list(prediction),
            "maximum_target_pixels": max(
                value
                for point in rows[index]["evaluator_only"]["visibility_history"]
                for value in point["target_pixels"].values()
            ),
        }
        for index, label, prediction in zip(
            np.flatnonzero(development), truth, prediction_sets, strict=True
        )
    ]
    artifact["development_diagnostic"] = {
        "per_class": per_class,
        "would_pass_numeric_gate": development_passed,
        "minimum_singleton_rate": 0.8,
        "rows": diagnostic_rows,
        "note": (
            "seeds 653-659 diagnosed v8 and motivated the architecture-derived "
            "resolvability contract; they cannot certify v9"
        ),
    }
    artifact["development"] = {
        "passed": False,
        "note": "requires untouched extension seeds before sealed audit",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"frozen v8 artifact exists: {args.output}")
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "effect_selected": effect_selected,
                "content_selected": content_selected,
                "development_diagnostic": artifact["development_diagnostic"],
                "corrected_rows": artifact["label_contract"]["corrected_rows"],
            },
            indent=2,
        )
    )
if __name__ == "__main__":
    main()
