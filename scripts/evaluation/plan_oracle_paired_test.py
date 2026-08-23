#!/usr/bin/env python3
"""Prospectively size the formal oracle test from an independent pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from calibrated_interaction.paired_power import (
    smallest_prospective_group_count,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(config_path: Path, pilot_path: Path, search_limit: int) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != (
        "calibrated-interaction.oracle-target-prompt-pilot.v2"
    ):
        raise ValueError("formal planning requires the v2 oracle pilot protocol")
    pilot = json.loads(pilot_path.read_text())
    if pilot.get("status") != "INDEPENDENT_DEVELOPMENT_PILOT_COMPLETE":
        raise ValueError("pilot summary is not complete independent-pilot evidence")
    confirmation = pilot["confirmation"]
    comparison = confirmation["paired_target_grasp_contact_comparison"]
    trials = int(confirmation["aggregates"]["target_grasp_contact"]["trials"])
    left_only = int(comparison["left_only"])
    right_only = int(comparison["right_only"])
    left_probability = left_only / trials
    right_probability = right_only / trials
    formal = config["formal_followup"]
    result = smallest_prospective_group_count(
        intervention_only_probability=left_probability,
        baseline_only_probability=right_probability,
        alpha=float(formal["alpha"]),
        target_power=float(formal["target_power"]),
        search_limit=search_limit,
    )
    return {
        "schema_version": "calibrated-interaction.oracle-formal-test-plan.v1",
        "status": (
            "PROSPECTIVE_GROUP_COUNT_FROZEN"
            if result
            else "NO_PLAN_WITHIN_NUMERICAL_SEARCH_BOUND"
        ),
        "claim_scope": "DESIGN_ONLY_NO_FORMAL_OUTCOME_DATA",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": sha256(config_path),
        },
        "pilot": {
            "path": str(pilot_path.relative_to(ROOT)),
            "sha256": sha256(pilot_path),
            "trials": trials,
            "intervention_only": left_only,
            "baseline_only": right_only,
        },
        "pilot_maximum_likelihood_discordant_probabilities": {
            "intervention_only": left_probability,
            "baseline_only": right_probability,
        },
        "alpha": float(formal["alpha"]),
        "target_power": float(formal["target_power"]),
        "prospective_group_count": result[0] if result else None,
        "power_at_frozen_count": result[1] if result else None,
        "search_limit": search_limit,
        "search_limit_role": "numerical resource bound; never a success threshold",
        "test": formal["test"],
        "data_contract": formal["data"],
        "warning": (
            "Pilot effect estimates determine design only. Pilot groups are excluded "
            "from the formal p-value and effect estimate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/experiments/original_drawer_oracle_target_prompt_pilot_v2.yaml",
    )
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-limit", type=int, default=200)
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    pilot = args.pilot if args.pilot.is_absolute() else ROOT / args.pilot
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"formal test plan is immutable: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan(config, pilot, args.search_limit), indent=2) + "\n"
    )
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
