#!/usr/bin/env python3
"""Freeze the v10 target-evidence + observation-effect candidate contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-artifact",
        type=Path,
        default=ROOT
        / "results/calibration/t01_rgb_target_evidence_v2_candidate_agentview.json",
    )
    parser.add_argument(
        "--target-checkpoint",
        type=Path,
        default=ROOT
        / "results/models/t01_rgb_target_evidence_v2_candidate_agentview.pt",
    )
    parser.add_argument(
        "--effect-artifact",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_critic_v10_candidate_visual.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_v10_composite_candidate.json",
    )
    args = parser.parse_args()
    for name in ("target_artifact", "target_checkpoint", "effect_artifact", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output.exists():
        raise FileExistsError(f"frozen candidate already exists: {args.output}")
    for path in (args.target_artifact, args.target_checkpoint, args.effect_artifact):
        if not path.exists():
            raise FileNotFoundError(path)

    target = json.loads(args.target_artifact.read_text())
    effect = json.loads(args.effect_artifact.read_text())
    if target["model"]["camera_mode"] != "agentview":
        raise ValueError("v10 candidate must use the selected stable agentview head")
    if effect["label_contract"].get("version") != "v10":
        raise ValueError("effect artifact does not implement the v10 label contract")
    artifact = {
        "schema_version": "interactive-perception.action-outcome-composite.v10-candidate",
        "status": "FROZEN_FOR_FRESH_CLEAN_DEVELOPMENT",
        "claim_eligible": False,
        "target_evidence": {
            "artifact": reference(args.target_artifact),
            "checkpoint": reference(args.target_checkpoint),
            "online_inputs": target["online_inputs"],
            "temporal_rule": target["temporal_rule"],
        },
        "observation_effect": {
            "artifact": reference(args.effect_artifact),
            "head": "effect_head only: FAILED vs COMPLETED",
            "online_inputs": effect["online_inputs"],
        },
        "decision_rule": [
            "singleton REVEALED -> REVEALED, regardless of later reocclusion or motor endpoint",
            "singleton NOT_REVEALED + singleton COMPLETED -> EMPTY",
            "singleton NOT_REVEALED + singleton FAILED -> FAILED",
            "any multi-label level -> preserve all compatible labels; controller SAFE_STOP",
        ],
        "offline_only_labels": [
            "target segmentation pixels for frame supervision and final scoring",
            "seed-matched same-camera target rendering for searched-region coverage supervision and scoring",
            "drawer joint and hidden task state for motor diagnostics only",
        ],
        "online_oracle_inputs": [],
        "fresh_clean_seed_block": "t01_open_observe_v10_clean_development",
        "sealed_seed_block": "t01_open_observe_sealed_audit",
        "sealed_open_condition": "fresh v10 clean development passes every registered gate",
        "gates": {
            "per_class_correct_label_retention": 0.90,
            "per_class_singleton_one_sided_95_lower": 0.80,
            "also_report_original_singleton_lower": 0.90,
            "false_singleton_EMPTY": 0,
            "false_singleton_REVEALED": 0,
            "partially_passed_is_not_go": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"frozen": reference(args.output), "status": artifact["status"]}, indent=2))


if __name__ == "__main__":
    main()
