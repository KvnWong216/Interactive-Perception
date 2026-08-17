#!/usr/bin/env python3
"""Fit and conformalize the prompt-conditioned T01 target-state probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interactive_perception.prompt_state import (  # noqa: E402
    BalancedRidgeBinary,
    GaussianScoreCalibrator,
)
from interactive_perception.semantic_conformal import (  # noqa: E402
    MondrianSemanticConformalCalibrator,
)

BLOCKS = {
    "last_prompt": (0, 2048),
    "prompt_mean": (2048, 4096),
    "base_mean": (4096, 6144),
    "wrist_mean": (6144, 8192),
    "all": (0, 8192),
}
REGULARIZATION_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def subset(features, start, stop):
    return np.asarray(features[:, start:stop], dtype=np.float64)


def balanced_accuracy(truth, prediction):
    labels = sorted(set(truth.tolist()))
    return float(
        np.mean(
            [np.mean(prediction[truth == label] == label) for label in labels]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT / "outputs/t01_prompt_state_v1/pi05_prefix_embeddings.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/calibration/prompt_state_belief_t01_v1.json",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    data = np.load(args.embeddings)
    features = np.asarray(data["features"], dtype=np.float64)
    labels = np.asarray(data["target_state"], dtype=str)
    conditions = np.asarray(data["condition"], dtype=str)
    seeds = np.asarray(data["seed"], dtype=np.int64)
    splits = np.asarray(data["split"], dtype=str)
    negative_label = "MANIPULATION_ONLY"
    positive_label = "OBSERVED"
    train = splits == "prototype_train"
    probability_calibration = (seeds >= 240) & (seeds < 250)
    conformal_calibration = (seeds >= 250) & (seeds < 260)
    validation = splits == "heldout_validation"
    if not all(mask.any() for mask in (train, probability_calibration, conformal_calibration, validation)):
        raise ValueError("required frozen split is empty")

    searches = []
    for block, (start, stop) in BLOCKS.items():
        values = subset(features, start, stop)
        for regularization in REGULARIZATION_GRID:
            fold_scores = []
            for fold in range(5):
                fit_mask = train & (seeds % 5 != fold)
                test_mask = train & (seeds % 5 == fold)
                probe = BalancedRidgeBinary.fit(
                    values[fit_mask],
                    labels[fit_mask],
                    negative_label=negative_label,
                    positive_label=positive_label,
                    regularization=regularization,
                )
                prediction = probe.predict(values[test_mask])
                fold_scores.append(
                    balanced_accuracy(labels[test_mask], prediction)
                )
            searches.append(
                {
                    "block": block,
                    "regularization": regularization,
                    "mean_grouped_cv_balanced_accuracy": float(np.mean(fold_scores)),
                    "fold_balanced_accuracy": fold_scores,
                }
            )
    selected = sorted(
        searches,
        key=lambda item: (
            -item["mean_grouped_cv_balanced_accuracy"],
            BLOCKS[item["block"]][1] - BLOCKS[item["block"]][0],
            item["regularization"],
            item["block"],
        ),
    )[0]
    start, stop = BLOCKS[str(selected["block"])]
    values = subset(features, start, stop)
    probe = BalancedRidgeBinary.fit(
        values[train],
        labels[train],
        negative_label=negative_label,
        positive_label=positive_label,
        regularization=float(selected["regularization"]),
    )
    probability_model = GaussianScoreCalibrator.fit(
        probe.score(values[probability_calibration]),
        labels[probability_calibration],
        ordered_labels=(negative_label, positive_label),
    )
    conformal_examples = list(
        zip(
            probability_model.evidence(probe.score(values[conformal_calibration])),
            labels[conformal_calibration].tolist(),
            strict=True,
        )
    )
    conformal = MondrianSemanticConformalCalibrator.fit(
        conformal_examples,
        alpha=args.alpha,
        policy_id="pi05_libero_prefix",
        split_id="t01_prompt_state_v1_seed250_259",
    )
    validation_evidence = probability_model.evidence(probe.score(values[validation]))
    validation_sets = [conformal.predict(item) for item in validation_evidence]
    validation_truth = labels[validation]
    validation_condition = conditions[validation]
    covered = np.asarray(
        [truth in predicted for truth, predicted in zip(validation_truth, validation_sets, strict=True)]
    )
    singleton_correct = np.asarray(
        [predicted == (truth,) for truth, predicted in zip(validation_truth, validation_sets, strict=True)]
    )
    condition_metrics = {}
    for condition in sorted(set(validation_condition.tolist())):
        mask = validation_condition == condition
        condition_metrics[condition] = {
            "trials": int(mask.sum()),
            "coverage": float(np.mean(covered[mask])),
            "singleton_accuracy": float(np.mean(singleton_correct[mask])),
            "mean_set_size": float(
                np.mean([len(value) for value, keep in zip(validation_sets, mask, strict=True) if keep])
            ),
        }
    per_class_coverage = {
        label: float(np.mean(covered[validation_truth == label]))
        for label in sorted(set(validation_truth.tolist()))
    }
    artifact = {
        "schema_version": "interactive-perception.prompt-state-belief.v1",
        "claim": "prompt-conditioned target observability state from frozen pi0.5 prefix",
        "non_claim": "scene-disjoint generalization or downstream task success",
        "embeddings": str(args.embeddings.relative_to(ROOT)),
        "embedding_sha256": hashlib.sha256(args.embeddings.read_bytes()).hexdigest(),
        "alpha": args.alpha,
        "labels": [negative_label, positive_label],
        "split_contract": {
            "prototype_train": "seeds 220-239",
            "probability_calibration": "seeds 240-249",
            "conformal_calibration": "seeds 250-259",
            "heldout_validation": "seeds 260-269",
            "future_audit": "seeds 280-309; not collected when this artifact was fit",
        },
        "model_selection": {
            "rule": "five-fold grouped-by-seed balanced accuracy on prototype_train only",
            "candidates": searches,
            "selected": selected,
        },
        "feature_block": {"name": selected["block"], "start": start, "stop": stop},
        "probe": probe.to_dict(),
        "probability_model": probability_model.to_dict(),
        "conformal": conformal.to_dict(),
        "validation": {
            "trials": int(validation.sum()),
            "coverage": float(np.mean(covered)),
            "minimum_class_coverage": min(per_class_coverage.values()),
            "per_class_coverage": per_class_coverage,
            "singleton_accuracy": float(np.mean(singleton_correct)),
            "mean_set_size": float(np.mean([len(value) for value in validation_sets])),
            "by_condition": condition_metrics,
            "prediction_sets": [list(value) for value in validation_sets],
            "passed_development_gate": bool(
                np.mean(covered) >= 0.9
                and min(per_class_coverage.values()) >= 0.9
                and np.mean(singleton_correct) >= 0.9
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "selected": selected,
                "validation": artifact["validation"],
            },
            indent=2,
        )
    )
    if not artifact["validation"]["passed_development_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
