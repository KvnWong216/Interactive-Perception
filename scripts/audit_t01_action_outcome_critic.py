#!/usr/bin/env python3
"""Audit a frozen T01 outcome critic and its physical effect reliability."""

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
    transition_feature_block,
)
from interactive_perception.capability_gate import exact_binomial_lower_bound  # noqa: E402
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
        default=ROOT / "results/calibration/t01_action_outcome_critic_v1.json",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT / "outputs/t01_action_effect_v1_audit/pi05_transition_embeddings.npz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/calibration/t01_action_effect_v1_audit.manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/calibration/t01_action_outcome_critic_audit_v1.json",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--required-reliability", type=float, default=0.9)
    args = parser.parse_args()
    for name in ("artifact", "embeddings", "manifest", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)

    artifact_sha = digest(args.artifact)
    manifest = json.loads(args.manifest.read_text())
    if manifest["frozen_artifact_sha256_before_audit"] != artifact_sha:
        raise ValueError("critic artifact changed after audit collection")
    artifact = json.loads(args.artifact.read_text())
    model = BalancedRidgeMulticlass.from_dict(artifact["critic"])
    conformal_data = artifact["conformal"]
    conformal = MondrianSemanticConformalCalibrator(
        alpha=float(conformal_data["alpha"]),
        thresholds={key: float(value) for key, value in conformal_data["thresholds"].items()},
        labels=tuple(str(value) for value in conformal_data["labels"]),
        calibration_size_per_class={
            key: int(value)
            for key, value in conformal_data["calibration_size_per_class"].items()
        },
        policy_id=str(conformal_data["policy_id"]),
        split_id=str(conformal_data["split_id"]),
    )
    data = np.load(args.embeddings)
    before = np.asarray(data["before_features"], dtype=np.float64)
    after = np.asarray(data["after_features"], dtype=np.float64)
    labels = np.asarray(data["outcome"], dtype=str)
    intended = np.asarray(data["intended_outcome"], dtype=str)
    full = np.asarray(data["full_executor"], dtype=bool)
    physical_subtype = np.asarray(data["physical_outcome_subtype"], dtype=str)
    seeds = np.asarray(data["seed"], dtype=np.int64)
    if sorted(set(seeds.tolist())) != list(range(500, 600)):
        raise ValueError("frozen audit embeddings must contain exactly seeds 500-599")
    block = str(artifact["model_selection"]["selected"]["block"])
    values = transition_feature_block(before, after, block)
    evidence = model.evidence(values)
    prediction_sets = [conformal.predict(item) for item in evidence]

    by_class = {}
    class_gate = True
    for label in model.labels:
        mask = labels == label
        trials = int(mask.sum())
        covered = int(
            sum(
                label in predicted
                for predicted, keep in zip(prediction_sets, mask, strict=True)
                if keep
            )
        )
        singleton = int(
            sum(
                predicted == (label,)
                for predicted, keep in zip(prediction_sets, mask, strict=True)
                if keep
            )
        )
        singleton_lower = exact_binomial_lower_bound(
            singleton, trials, args.confidence
        )
        row = {
            "trials": trials,
            "coverage": covered / trials,
            "singleton_correct": singleton,
            "singleton_accuracy": singleton / trials,
            "singleton_one_sided_lower": singleton_lower,
            "passes_0.90": singleton_lower >= args.required_reliability,
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
        by_class[label] = row
        class_gate = class_gate and row["passes_0.90"]

    physical = {}
    physical_gate = True
    for label in ("REVEALED", "EMPTY"):
        mask = full & (intended == label)
        trials = int(mask.sum())
        successes = int(np.sum(labels[mask] == label))
        lower = exact_binomial_lower_bound(successes, trials, args.confidence)
        row = {
            "successes": successes,
            "trials": trials,
            "empirical_rate": successes / trials,
            "one_sided_lower": lower,
            "passes_0.90": lower >= args.required_reliability,
        }
        physical[label] = row
        physical_gate = physical_gate and row["passes_0.90"]

    report = {
        "schema_version": "interactive-perception.action-outcome-critic-audit.v1",
        "artifact": str(args.artifact.relative_to(ROOT)),
        "artifact_sha256": artifact_sha,
        "embeddings": str(args.embeddings.relative_to(ROOT)),
        "embeddings_sha256": digest(args.embeddings),
        "audit_manifest": str(args.manifest.relative_to(ROOT)),
        "audit_manifest_sha256": digest(args.manifest),
        "required_reliability": args.required_reliability,
        "confidence": args.confidence,
        "critic_by_class": by_class,
        "physical_full_executor": physical,
        "physical_failure_subtypes": {
            label: int(np.sum(full & (physical_subtype == label)))
            for label in ("NO_EFFECT", "OPENED_UNOBSERVED")
        },
        "critic_gate_passed": class_gate,
        "physical_effect_gate_passed": physical_gate,
        "fp3_passed": bool(class_gate and physical_gate),
        "online_oracle_inputs": [],
        "failure_subtype_interpretation": {
            "NO_EFFECT": "retry or replace the physical executor",
            "OPENED_UNOBSERVED": "the world changed but information is still missing; use viewpoint discovery or stop safely",
        },
        "non_claim": "paper test-set or scene-disjoint generalization",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["fp3_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
