#!/usr/bin/env python3
"""Evaluate the frozen prompt-state artifact once on its preregistered audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interactive_perception.capability_gate import exact_binomial_lower_bound  # noqa: E402
from interactive_perception.prompt_state import (  # noqa: E402
    BalancedRidgeBinary,
    GaussianScoreCalibrator,
)
from interactive_perception.semantic_conformal import (  # noqa: E402
    MondrianSemanticConformalCalibrator,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "results/calibration/prompt_state_belief_t01_v1.json",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        nargs="+",
        default=[
            ROOT / "outputs/t01_prompt_state_v1_audit/pi05_prefix_embeddings.npz"
        ],
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        nargs="+",
        default=[ROOT / "data/calibration/t01_prompt_state_v1_audit.manifest.json"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/calibration/prompt_state_belief_t01_audit_v1.json",
    )
    parser.add_argument("--required-reliability", type=float, default=0.9)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    for name in ("artifact", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    args.embeddings = [
        path if path.is_absolute() else ROOT / path for path in args.embeddings
    ]
    args.manifest = [
        path if path.is_absolute() else ROOT / path for path in args.manifest
    ]

    artifact = json.loads(args.artifact.read_text())
    if len(args.embeddings) != len(args.manifest):
        raise ValueError("each embedding file requires its audit manifest")
    manifests = [json.loads(path.read_text()) for path in args.manifest]
    if any(
        digest(args.artifact) != value["frozen_artifact_sha256_before_audit"]
        for value in manifests
    ):
        raise ValueError("belief artifact changed after audit observations were collected")
    embedded_parts = [np.load(path) for path in args.embeddings]
    feature = artifact["feature_block"]
    values = np.concatenate(
        [
            np.asarray(
                embedded["features"][
                    :, int(feature["start"]) : int(feature["stop"])
                ],
                dtype=np.float64,
            )
            for embedded in embedded_parts
        ]
    )
    truth = np.concatenate(
        [np.asarray(embedded["target_state"], dtype=str) for embedded in embedded_parts]
    )
    conditions = np.concatenate(
        [np.asarray(embedded["condition"], dtype=str) for embedded in embedded_parts]
    )
    seeds = np.concatenate(
        [np.asarray(embedded["seed"], dtype=np.int64) for embedded in embedded_parts]
    )
    probe = BalancedRidgeBinary.from_dict(artifact["probe"])
    probability_model = GaussianScoreCalibrator.from_dict(
        artifact["probability_model"]
    )
    conformal_value = artifact["conformal"]
    conformal = MondrianSemanticConformalCalibrator(
        alpha=float(conformal_value["alpha"]),
        thresholds={
            key: float(value) for key, value in conformal_value["thresholds"].items()
        },
        labels=tuple(str(value) for value in conformal_value["labels"]),
        calibration_size_per_class={
            key: int(value)
            for key, value in conformal_value["calibration_size_per_class"].items()
        },
        policy_id=str(conformal_value["policy_id"]),
        split_id=str(conformal_value["split_id"]),
    )
    evidence = probability_model.evidence(probe.score(values))
    predictions = [conformal.predict(item) for item in evidence]
    probabilities = probability_model.probabilities(probe.score(values))
    covered = np.asarray(
        [label in prediction for label, prediction in zip(truth, predictions, strict=True)]
    )
    singleton = np.asarray(
        [prediction == (label,) for label, prediction in zip(truth, predictions, strict=True)]
    )
    by_condition = {}
    for condition in sorted(set(conditions.tolist())):
        mask = conditions == condition
        successes = int(np.sum(singleton[mask]))
        trials = int(mask.sum())
        lower_bound = exact_binomial_lower_bound(successes, trials, args.confidence)
        by_condition[condition] = {
            "coverage": float(np.mean(covered[mask])),
            "mean_set_size": float(
                np.mean([len(value) for value, keep in zip(predictions, mask, strict=True) if keep])
            ),
            "singleton_correct_gate": {
                "successes": successes,
                "trials": trials,
                "empirical_rate": successes / trials,
                "confidence": args.confidence,
                "lower_bound": lower_bound,
                "required_reliability": args.required_reliability,
                "passed": lower_bound >= args.required_reliability,
                "interpretation": "prompt-state singleton reliability in this counterfactual condition",
            },
        }
    rows = []
    for index in range(len(truth)):
        rows.append(
            {
                "condition": str(conditions[index]),
                "seed": int(seeds[index]),
                "true_state": str(truth[index]),
                "probabilities": {
                    label: float(probabilities[index, class_index])
                    for class_index, label in enumerate(probability_model.labels)
                },
                "prediction_set": list(predictions[index]),
                "covered": bool(covered[index]),
                "singleton_correct": bool(singleton[index]),
            }
        )
    passed = all(
        value["singleton_correct_gate"]["passed"]
        for value in by_condition.values()
    )
    report = {
        "schema_version": "interactive-perception.prompt-state-belief-audit.v1",
        "claim": "T01 prompt/state counterfactual discrimination",
        "non_claim": "scene-disjoint target belief or physical task completion",
        "artifact": str(args.artifact.relative_to(ROOT)),
        "artifact_sha256": digest(args.artifact),
        "embeddings": [str(path.relative_to(ROOT)) for path in args.embeddings],
        "embeddings_sha256": {
            str(path.relative_to(ROOT)): digest(path) for path in args.embeddings
        },
        "audit_manifests": [str(path.relative_to(ROOT)) for path in args.manifest],
        "audit_dataset_sha256": [value["dataset_sha256"] for value in manifests],
        "required_reliability": args.required_reliability,
        "confidence": args.confidence,
        "controller_oracle_inputs": [],
        "by_condition": by_condition,
        "audit_passed": passed,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"by_condition": by_condition, "audit_passed": passed}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
