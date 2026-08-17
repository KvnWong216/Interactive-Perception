#!/usr/bin/env python3
"""One-time audit of the frozen v5 temporal outcome critic and option effect."""

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
    ActionOutcomePredictor,
    temporal_history_feature_block,
)
from interactive_perception.capability_gate import exact_binomial_lower_bound  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_critic_v5.json",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT
        / "outputs/t01_open_and_observe_effect_v1_audit/pi05_temporal_embeddings_v5.npz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "data/calibration/t01_open_and_observe_effect_v1_audit.manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_audit_v5.json",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--minimum-action-reliability", type=float, default=0.8
    )
    parser.add_argument(
        "--minimum-singleton-reliability", type=float, default=0.9
    )
    args = parser.parse_args()
    for name in ("artifact", "embeddings", "manifest", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)

    artifact_sha = digest(args.artifact)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("phase") != "heldout_audit" or manifest.get("seeds") != list(
        range(700, 800)
    ):
        raise ValueError("audit manifest does not match sealed seeds 700-799")
    if manifest.get("audit_artifact_sha256") != artifact_sha:
        raise ValueError("critic artifact changed after audit collection")

    artifact = json.loads(args.artifact.read_text())
    if not artifact.get("development", {}).get("passed", False):
        raise ValueError("the frozen v5 artifact did not pass development")
    predictor = ActionOutcomePredictor.from_artifact(artifact)
    data = np.load(args.embeddings)
    history = np.asarray(data["history_features"], dtype=np.float64)
    robot = np.asarray(data["robot_state_history"], dtype=np.float64)
    labels = np.asarray(data["outcome"], dtype=str)
    intended = np.asarray(data["intended_outcome"], dtype=str)
    full = np.asarray(data["full_executor"], dtype=bool)
    seeds = np.asarray(data["seed"], dtype=np.int64)
    if sorted(set(seeds.tolist())) != list(range(700, 800)):
        raise ValueError("audit embeddings require exactly seeds 700-799")

    values = temporal_history_feature_block(history, robot, predictor.block)
    if predictor.standardizer is not None:
        values = predictor.standardizer.transform(values)
    evidence = predictor.critic.evidence(values)
    prediction_sets = [predictor.conformal.predict(item) for item in evidence]

    critic_by_class = {}
    critic_gate = True
    for label in predictor.critic.labels:
        mask = labels == label
        trials = int(mask.sum())
        covered = sum(
            label in predicted
            for predicted, keep in zip(prediction_sets, mask, strict=True)
            if keep
        )
        singleton = sum(
            predicted == (label,)
            for predicted, keep in zip(prediction_sets, mask, strict=True)
            if keep
        )
        singleton_lower = exact_binomial_lower_bound(
            singleton, trials, args.confidence
        )
        row = {
            "trials": trials,
            "coverage": covered / trials,
            "singleton_correct": singleton,
            "singleton_rate": singleton / trials,
            "singleton_one_sided_lower": singleton_lower,
            "mean_set_size": float(
                np.mean(
                    [
                        len(predicted)
                        for predicted, keep in zip(
                            prediction_sets, mask, strict=True
                        )
                        if keep
                    ]
                )
            ),
            "passes": (
                covered / trials >= 1.0 - predictor.conformal.alpha
                and singleton_lower >= args.minimum_singleton_reliability
            ),
        }
        critic_by_class[label] = row
        critic_gate = critic_gate and row["passes"]

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
            "passes": lower >= args.minimum_action_reliability,
            "passes_original_0.90_reliability": lower >= 0.9,
        }
        physical[label] = row
        physical_gate = physical_gate and row["passes"]

    report = {
        "schema_version": "interactive-perception.open-and-observe-audit.v5",
        "artifact": str(args.artifact.relative_to(ROOT)),
        "artifact_sha256": artifact_sha,
        "embeddings": str(args.embeddings.relative_to(ROOT)),
        "embeddings_sha256": digest(args.embeddings),
        "manifest": str(args.manifest.relative_to(ROOT)),
        "manifest_sha256": digest(args.manifest),
        "critic_by_class": critic_by_class,
        "physical_effect_by_branch": physical,
        "minimum_action_reliability": args.minimum_action_reliability,
        "minimum_singleton_reliability": args.minimum_singleton_reliability,
        "critic_gate_passed": critic_gate,
        "physical_effect_gate_passed": physical_gate,
        "fp3_passed": bool(critic_gate and physical_gate),
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
