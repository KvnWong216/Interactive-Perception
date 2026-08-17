#!/usr/bin/env python3
"""One-time sealed audit for the temporal hierarchical outcome critic."""

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
    label_temporal_information_outcome,
)
from interactive_perception.capability_gate import exact_binomial_lower_bound  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observable_outcome(row: dict) -> str:
    return label_temporal_information_outcome(
        full_executor=bool(row["full_executor"]),
        opened=bool(row["evaluator_only"]["drawer_opened"]),
        return_complete=row["return_status"]["phase"] == "COMPLETE",
        target_pixel_history=tuple(
            tuple(point["target_pixels"].values())
            for point in row["evaluator_only"]["visibility_history"]
        ),
        minimum_target_pixels=int(
            row["evaluator_only"]["minimum_revealed_target_pixels"]
        ),
        empty_coverage_certified=bool(
            row["evaluator_only"]["empty_counterfactual_reveal_certified"]
        ),
    ).value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_critic_v8.json",
    )
    parser.add_argument(
        "--effect-artifact",
        type=Path,
        default=ROOT / "results/calibration/t01_open_and_observe_effect_v3.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT
        / "data/calibration/t01_open_and_observe_effect_v3_audit.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "data/calibration/t01_open_and_observe_effect_v3_audit.manifest.json",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT
        / "outputs/t01_open_and_observe_effect_v3_audit/pi05_temporal_embeddings_v5.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_audit_v8.json",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--minimum-action-reliability", type=float, default=0.8)
    parser.add_argument("--minimum-singleton-reliability", type=float, default=0.8)
    args = parser.parse_args()
    for name in (
        "artifact",
        "effect_artifact",
        "dataset",
        "manifest",
        "embeddings",
        "output",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)

    artifact_sha = digest(args.artifact)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("phase") != "heldout_audit" or manifest.get("seeds") != list(
        range(900, 1000)
    ):
        raise ValueError("audit manifest must contain the sealed seeds 900-999")
    if manifest.get("audit_artifact_sha256") != artifact_sha:
        raise ValueError("v8 critic changed after audit collection")
    if manifest.get("dataset_sha256") != digest(args.dataset):
        raise ValueError("audit dataset hash differs from its manifest")
    artifact = json.loads(args.artifact.read_text())
    if not artifact.get("development", {}).get("passed", False):
        raise ValueError("v8 must pass development before audit")
    predictor = HierarchicalActionOutcomePredictor.from_artifact(artifact)
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line]
    data = np.load(args.embeddings)
    if len(rows) != len(data["seed"]):
        raise ValueError("audit dataset and embeddings are not aligned")
    expected_seeds = np.asarray([row["seed"] for row in rows], dtype=np.int64)
    expected_regimes = np.asarray([row["regime"] for row in rows], dtype=str)
    if not np.array_equal(data["seed"], expected_seeds) or not np.array_equal(
        data["regime"].astype(str), expected_regimes
    ):
        raise ValueError("embedding row order differs from the immutable audit dataset")
    predictions = [
        predictor.predict_history(history, robot).prediction_set
        for history, robot in zip(
            data["history_features"], data["robot_state_history"], strict=True
        )
    ]
    truth = np.asarray([observable_outcome(row) for row in rows], dtype=str)
    critic = {}
    critic_gate = True
    for label in ("FAILED", "REVEALED", "EMPTY"):
        mask = truth == label
        trials = int(mask.sum())
        if trials == 0:
            raise ValueError(f"audit contains no {label} examples")
        selected = [
            item for item, keep in zip(predictions, mask, strict=True) if keep
        ]
        covered = sum(label in item for item in selected)
        singleton = sum(item == (label,) for item in selected)
        lower = exact_binomial_lower_bound(singleton, trials, args.confidence)
        row = {
            "trials": trials,
            "coverage": covered / trials,
            "singleton_correct": singleton,
            "singleton_rate": singleton / trials,
            "singleton_one_sided_lower": lower,
            "mean_set_size": float(np.mean([len(item) for item in selected])),
            "prediction_set_histogram": dict(
                sorted(
                    collections.Counter("|".join(item) for item in selected).items()
                )
            ),
            "passes": (
                covered / trials >= 1.0 - predictor.effect_conformal.alpha
                and lower >= args.minimum_singleton_reliability
            ),
        }
        critic[label] = row
        critic_gate = critic_gate and row["passes"]

    physical = {}
    physical_gate = True
    for intended in ("REVEALED", "EMPTY"):
        selected = [
            (row, label)
            for row, label in zip(rows, truth, strict=True)
            if row["full_executor"] and row["intended_outcome"] == intended
        ]
        successes = sum(label == intended for _, label in selected)
        if not selected:
            raise ValueError(f"audit contains no physical {intended} branch")
        lower = exact_binomial_lower_bound(
            successes, len(selected), args.confidence
        )
        row = {
            "successes": successes,
            "trials": len(selected),
            "one_sided_lower": lower,
            "passes_0.80": lower >= args.minimum_action_reliability,
            "passes_original_0.90": lower >= 0.9,
        }
        physical[intended] = row
        physical_gate = physical_gate and row["passes_0.80"]

    retention_mask = np.asarray(
        [
            label == "REVEALED"
            and max(row["evaluator_only"]["after_target_pixels"].values())
            < int(row["evaluator_only"]["minimum_revealed_target_pixels"])
            for row, label in zip(rows, truth, strict=True)
        ],
        dtype=bool,
    )
    retention_sets = [
        item for item, keep in zip(predictions, retention_mask, strict=True) if keep
    ]
    evidence_retention = {
        "trials": int(retention_mask.sum()),
        "coverage": (
            float(np.mean(["REVEALED" in item for item in retention_sets]))
            if retention_sets
            else None
        ),
        "singleton_rate": (
            float(np.mean([item == ("REVEALED",) for item in retention_sets]))
            if retention_sets
            else None
        ),
        "definition": "target visible in history but absent from both final policy views",
    }

    report = {
        "schema_version": "interactive-perception.open-and-observe-audit.v8",
        "artifact": str(args.artifact.relative_to(ROOT)),
        "artifact_sha256": artifact_sha,
        "effect_artifact": str(args.effect_artifact.relative_to(ROOT)),
        "effect_artifact_sha256": digest(args.effect_artifact),
        "dataset": str(args.dataset.relative_to(ROOT)),
        "dataset_sha256": digest(args.dataset),
        "embeddings": str(args.embeddings.relative_to(ROOT)),
        "embeddings_sha256": digest(args.embeddings),
        "observable_truth_counts": dict(
            sorted(collections.Counter(truth.tolist()).items())
        ),
        "critic_by_observable_outcome": critic,
        "physical_task_branch": physical,
        "evidence_retention_subset": evidence_retention,
        "critic_gate_passed": critic_gate,
        "physical_effect_gate_passed": physical_gate,
        "fp3_passed": bool(critic_gate and physical_gate),
        "minimum_action_reliability": args.minimum_action_reliability,
        "minimum_singleton_reliability": args.minimum_singleton_reliability,
        "conformal_error_rate": predictor.effect_conformal.alpha,
        "online_oracle_inputs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"immutable audit result exists: {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["fp3_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
