#!/usr/bin/env python3
"""Freeze the all-public-RGB v11 T01 outcome cascade."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_v11_composite_candidate.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    files = {
        "agentview_target_artifact": ROOT
        / "results/calibration/t01_rgb_target_evidence_v2_candidate_agentview.json",
        "agentview_target_checkpoint": ROOT
        / "results/models/t01_rgb_target_evidence_v2_candidate_agentview.pt",
        "wrist_target_artifact": ROOT
        / "results/calibration/t01_rgb_wrist_target_evidence_v11b_candidate.json",
        "wrist_target_checkpoint": ROOT
        / "results/models/t01_rgb_wrist_target_evidence_v11b_candidate.pt",
        "coverage_artifact": ROOT
        / "results/calibration/t01_rgb_coverage_evidence_v11_candidate.json",
        "coverage_checkpoint": ROOT
        / "results/models/t01_rgb_coverage_evidence_v11_candidate.pt",
    }
    if any(not path.exists() for path in files.values()):
        raise FileNotFoundError("one or more frozen v11 dependencies are missing")
    artifact = {
        "schema_version": "interactive-perception.action-outcome-composite.v11-candidate",
        "status": "FROZEN_FOR_FRESH_CLEAN_DEVELOPMENT",
        "claim_eligible": False,
        "dependencies": {name: ref(path) for name, path in files.items()},
        "online_inputs": [
            "six stock agentview RGB frames",
            "six stock wrist RGB frames",
            "target query fixed by the T01 task protocol",
        ],
        "online_oracle_inputs": [],
        "target_cascade": [
            "agentview singleton REVEALED -> REVEALED",
            "otherwise wrist singleton REVEALED -> REVEALED",
            "otherwise agentview singleton NOT_REVEALED -> NOT_REVEALED",
            "otherwise preserve REVEALED/NOT_REVEALED ambiguity -> SAFE_STOP",
        ],
        "outcome_cascade": [
            "singleton target REVEALED -> REVEALED",
            "singleton NOT_REVEALED + singleton coverage COMPLETED -> EMPTY",
            "singleton NOT_REVEALED + singleton coverage FAILED -> FAILED",
            "any remaining ambiguity -> preserve compatible set -> SAFE_STOP",
        ],
        "design_reason": (
            "v10 seed 1420 was target-resolvable only in wrist RGB and also "
            "contained a separate empty-regime coverage failure; v11 separates "
            "camera-specific positive target evidence from RGB coverage evidence"
        ),
        "development_data_status": (
            "600-699 and 1400-1439 were used for model selection; none may "
            "support the v11 clean claim"
        ),
        "fresh_clean_seed_block": "t01_open_observe_v11_clean_development",
        "sealed_seed_block": "t01_open_observe_sealed_audit",
        "gates": {
            "per_class_correct_label_retention": 0.90,
            "per_class_singleton_one_sided_95_lower": 0.80,
            "false_singleton_EMPTY": 0,
            "false_singleton_REVEALED": 0,
            "physical_branch_one_sided_95_lower": 0.80,
            "partially_passed_is_not_go": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"frozen": ref(output), "status": artifact["status"]}, indent=2))


if __name__ == "__main__":
    main()
