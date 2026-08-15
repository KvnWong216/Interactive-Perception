#!/usr/bin/env python3
"""Fit prototypes, conformalize intent scores, and validate a frozen G4 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interactive_perception.semantic_conformal import (  # noqa: E402
    MondrianSemanticConformalCalibrator,
    SemanticConformalCalibrator,
)


def vector(row):
    return np.asarray(row["chunk_features"], dtype=np.float64).mean(axis=0)


def evidence(value, prototypes, scale):
    distances = {label: float(np.linalg.norm(value - center) / scale) for label, center in prototypes.items()}
    weights = {label: float(np.exp(-distance)) for label, distance in distances.items()}
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--minimum-validation-coverage", type=float, default=0.9)
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--method", choices=["marginal", "mondrian"], default="marginal")
    parser.add_argument("--validation-split", default="heldout_validation")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional preregistered action scope; rows outside it are excluded.",
    )
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for dataset in args.datasets
        for line in dataset.read_text().splitlines()
        if line
    ]
    available_labels = sorted({row["true_intent"] for row in rows})
    if args.labels is not None:
        requested = set(args.labels)
        unknown = requested - set(available_labels)
        if unknown:
            raise ValueError(f"unknown labels: {sorted(unknown)}")
        rows = [row for row in rows if row["true_intent"] in requested]
    labels = sorted({row["true_intent"] for row in rows})
    train = [row for row in rows if row["split"] == "prototype_train"]
    calibration = [row for row in rows if row["split"] == "conformal_calibration"]
    validation = [row for row in rows if row["split"] == args.validation_split]
    if not validation:
        raise ValueError(f"no examples in validation split {args.validation_split!r}")
    prototypes = {
        label: np.mean([vector(row) for row in train if row["true_intent"] == label], axis=0)
        for label in labels
    }
    within = [np.linalg.norm(vector(row) - prototypes[row["true_intent"]]) for row in train]
    scale = max(float(np.median(within)), 1e-9)
    examples = [(evidence(vector(row), prototypes, scale), row["true_intent"]) for row in calibration]
    calibrator_class = (
        MondrianSemanticConformalCalibrator
        if args.method == "mondrian"
        else SemanticConformalCalibrator
    )
    calibrator = calibrator_class.fit(
        examples,
        alpha=args.alpha,
        policy_id="pi05_libero",
        split_id=args.split_id,
    )
    predictions = [calibrator.predict(evidence(vector(row), prototypes, scale)) for row in validation]
    covered = [row["true_intent"] in prediction for row, prediction in zip(validation, predictions, strict=True)]
    coverage = float(np.mean(covered))
    mean_size = float(np.mean([len(value) for value in predictions]))
    per_class = {
        label: {
            "count": sum(row["true_intent"] == label for row in validation),
            "coverage": float(
                np.mean(
                    [
                        row["true_intent"] in prediction
                        for row, prediction in zip(validation, predictions, strict=True)
                        if row["true_intent"] == label
                    ]
                )
            ),
            "mean_set_size": float(
                np.mean(
                    [
                        len(prediction)
                        for row, prediction in zip(validation, predictions, strict=True)
                        if row["true_intent"] == label
                    ]
                )
            ),
        }
        for label in labels
    }
    minimum_class_coverage = min(value["coverage"] for value in per_class.values())
    passed = bool(
        coverage >= args.minimum_validation_coverage
        and minimum_class_coverage >= args.minimum_validation_coverage
    )
    artifact = {
        **calibrator.to_dict(),
        "scope": args.scope,
        "action_scope": labels,
        "excluded_labels": sorted(set(available_labels) - set(labels)),
        "datasets": [str(path) for path in args.datasets],
        "dataset_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.datasets
        },
        "feature": "mean black-box action-chunk temporal statistics",
        "prototype_scale": scale,
        "prototypes": {label: center.tolist() for label, center in prototypes.items()},
        "split_counts": {name: sum(row["split"] == name for row in rows) for name in {row["split"] for row in rows}},
        "validation_coverage": coverage,
        "validation_split": args.validation_split,
        "validation_minimum_class_coverage": minimum_class_coverage,
        "validation_mean_set_size": mean_size,
        "validation_per_class": per_class,
        "validation_predictions": predictions,
        "g4_passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
