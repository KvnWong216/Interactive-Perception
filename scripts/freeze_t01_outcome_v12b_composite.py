#!/usr/bin/env python3
"""Freeze the singleton-conflict all-public-RGB v12b T01 outcome cascade."""

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
        / "results/calibration/t01_open_and_observe_outcome_v12b_composite_candidate.json",
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
        "implementation_source": ROOT
        / "src/interactive_perception/rgb_outcome_critic.py",
        "protocol": ROOT
        / "benchmarks/calibration/t01_open_and_observe_v12b.yaml",
    }
    if any(not path.exists() for path in files.values()):
        raise FileNotFoundError("one or more frozen v12b dependencies are missing")
    artifact = {
        "schema_version": "interactive-perception.action-outcome-composite.v12b-candidate",
        "status": "FROZEN_FOR_FRESH_CLEAN_DEVELOPMENT",
        "claim_eligible": False,
        "dependencies": {name: ref(path) for name, path in files.items()},
        "online_inputs": [
            "six stock agentview RGB frames",
            "six stock wrist RGB frames",
            "target query fixed by the T01 task protocol",
        ],
        "online_oracle_inputs": [],
        "target_composition": [
            "agentview singleton REVEALED -> REVEALED",
            "agentview singleton NOT_REVEALED plus wrist singleton REVEALED -> preserve camera conflict",
            "agentview singleton NOT_REVEALED plus non-singleton wrist abstention -> NOT_REVEALED",
            "wrist singleton REVEALED may resolve only an ambiguous agentview set",
            "remaining target ambiguity is preserved",
        ],
        "outcome_composition": [
            "each plausible target/effect branch contributes its compatible outcome",
            "REVEALED target branch -> REVEALED",
            "NOT_REVEALED plus COMPLETED coverage branch -> EMPTY",
            "NOT_REVEALED plus FAILED coverage branch -> FAILED",
            "multi-label outcome -> SAFE_STOP; never select a convenient member",
        ],
        "design_reason": (
            "V11 clean seeds 1455 and 1464 exposed singleton wrist false positives. "
            "The first v12 model-selection rule then treated wrist multi-label "
            "abstention as counterevidence and made 50/120 rows ambiguous. V12b "
            "preserves only singleton positive cross-camera conflict: it neither "
            "tunes a score threshold nor discards a concrete wrist-only reveal."
        ),
        "model_selection_evidence": {
            **ref(
                ROOT
                / "results/calibration/t01_open_and_observe_v12b_model_selection_on_v11.json"
            ),
            "role": "diagnostic_only_not_claim_evidence",
            "selected_arm": "singleton_conflict_v12b",
        },
        "development_data_status": (
            "600-699, 1400-1439, and 1440-1479 were used for model/rule selection; "
            "none may support the v12b clean claim"
        ),
        "fresh_clean_seed_block": "t01_open_observe_v12b_clean_development",
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
