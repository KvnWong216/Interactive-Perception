#!/usr/bin/env python3
"""Fit the v5 prompt-attended temporal OPEN_AND_OBSERVE outcome critic."""

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
    temporal_history_feature_block,
)
from interactive_perception.semantic_conformal import (  # noqa: E402
    MondrianSemanticConformalCalibrator,
)

REGULARIZATION_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT
        / "outputs/t01_open_and_observe_effect_v1/pi05_temporal_embeddings_v5.npz",
    )
    parser.add_argument(
        "--effect-artifact",
        type=Path,
        default=ROOT / "results/calibration/t01_open_and_observe_effect_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_critic_v5.json",
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
    for name in ("embeddings", "effect_artifact", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)

    effect = json.loads(args.effect_artifact.read_text())
    if not effect.get("physical_effect_gate_passed", False):
        raise ValueError("physical OPEN_AND_OBSERVE development gate must pass first")
    data = np.load(args.embeddings)
    history = np.asarray(data["history_features"], dtype=np.float64)
    robot = np.asarray(data["robot_state_history"], dtype=np.float64)
    labels = np.asarray(data["outcome"], dtype=str)
    seeds = np.asarray(data["seed"], dtype=np.int64)
    if sorted(set(seeds.tolist())) != list(range(600, 660)):
        raise ValueError("v5 development embeddings require seeds 600-659")
    values = temporal_history_feature_block(history, robot, args.block)
    train = (seeds >= 600) & (seeds <= 619)
    calibration = (seeds >= 620) & (seeds <= 652)
    development = (seeds >= 653) & (seeds <= 659)
    if not all(mask.any() for mask in (train, calibration, development)):
        raise ValueError("one or more frozen v5 splits are empty")

    search = []
    for regularization in REGULARIZATION_GRID:
        folds = []
        for fold in range(5):
            fit = train & (seeds % 5 != fold)
            check = train & (seeds % 5 == fold)
            scaler = FeatureStandardizer.fit(values[fit])
            model = BalancedRidgeMulticlass.fit(
                scaler.transform(values[fit]),
                labels[fit],
                regularization=regularization,
            )
            folds.append(
                balanced_accuracy(
                    labels[check], model.predict(scaler.transform(values[check]))
                )
            )
        search.append(
            {
                "regularization": regularization,
                "fold_balanced_accuracy": folds,
                "mean_grouped_cv_balanced_accuracy": float(np.mean(folds)),
            }
        )
    selected = sorted(
        search,
        key=lambda row: (
            -row["mean_grouped_cv_balanced_accuracy"],
            row["regularization"],
        ),
    )[0]
    scaler = FeatureStandardizer.fit(values[train])
    scaled = scaler.transform(values)
    model = BalancedRidgeMulticlass.fit(
        scaled[train],
        labels[train],
        regularization=float(selected["regularization"]),
    )
    conformal = MondrianSemanticConformalCalibrator.fit(
        list(zip(model.evidence(scaled[calibration]), labels[calibration], strict=True)),
        alpha=args.alpha,
        policy_id="pi05_libero_prompt_attended_temporal_v5",
        split_id="t01_open_and_observe_seed620_652",
    )
    if min(conformal.calibration_size_per_class.values()) < 30:
        raise ValueError("v5 requires at least 30 conformal examples per class")

    evidence = model.evidence(scaled[development])
    prediction_sets = [conformal.predict(item) for item in evidence]
    truth = labels[development]
    per_class = {}
    for label in model.labels:
        mask = truth == label
        selected_sets = [
            predicted
            for predicted, keep in zip(prediction_sets, mask, strict=True)
            if keep
        ]
        per_class[label] = {
            "trials": int(mask.sum()),
            "coverage": float(
                np.mean([label in predicted for predicted in selected_sets])
            ),
            "singleton_accuracy": float(
                np.mean([predicted == (label,) for predicted in selected_sets])
            ),
            "mean_set_size": float(np.mean([len(item) for item in selected_sets])),
        }
    development_passed = bool(
        min(row["coverage"] for row in per_class.values()) >= 0.9
        and min(row["singleton_accuracy"] for row in per_class.values()) >= 0.9
    )
    artifact = {
        "schema_version": "interactive-perception.action-outcome-critic.v5",
        "claim": "prompt-attended six-frame RGB/proprioceptive option-outcome recognition",
        "embeddings": str(args.embeddings.relative_to(ROOT)),
        "embeddings_sha256": digest(args.embeddings),
        "effect_artifact": str(args.effect_artifact.relative_to(ROOT)),
        "effect_artifact_sha256": digest(args.effect_artifact),
        "input_feature_dimension": int(history.shape[2]),
        "history_points": int(history.shape[1]),
        "model_selection": {
            "rule": "five-fold grouped-by-seed prototype-only balanced accuracy",
            "selected": {"block": args.block, **selected},
            "candidates": search,
        },
        "feature_standardizer": scaler.to_dict(),
        "critic": model.to_dict(),
        "conformal": conformal.to_dict(),
        "split_contract": {
            "prototype_train": "seeds 600-619",
            "conformal_calibration": "seeds 620-652; 33 examples per class",
            "heldout_development": "seeds 653-659; development decision only",
            "sealed_audit": "seeds 700-799; untouched until this artifact passes development",
        },
        "development": {
            "per_class": per_class,
            "passed": development_passed,
        },
        "online_inputs": [
            "six stock policy RGB pairs",
            "six public robot-state vectors",
            "final prompt",
            "executed option role",
        ],
        "online_oracle_inputs": [],
        "fp3_passed": False,
        "fp3_note": "requires the one-time sealed audit",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"frozen critic artifact already exists: {args.output}")
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "selected": artifact["model_selection"]["selected"],
                "development": artifact["development"],
            },
            indent=2,
        )
    )
    if not development_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
