#!/usr/bin/env python3
"""Freeze the learned-data split from an external outcome-free resource budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.splits import (
    LEARNING_SPLIT_ROLES,
    load_learning_collection_budget,
    load_split_manifest,
    validate_split_manifest,
)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--exclude-split", type=Path, action="append", required=True)
    parser.add_argument(
        "--oracle-protocol",
        type=Path,
        default=ROOT
        / "configs/experiments/original_drawer_oracle_target_prompt_pilot_v2.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    budget_path = resolve(args.budget)
    protocol_path = resolve(args.oracle_protocol)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("learning split manifests are immutable")
    budget = load_learning_collection_budget(budget_path)
    excluded_seeds: set[int] = set()
    excluded_groups: set[str] = set()
    excluded_roles: set[str] = set()
    exclusions = []
    for raw in args.exclude_split:
        path = resolve(raw)
        manifest = load_split_manifest(path)
        exclusions.append({"path": portable(path), "sha256": sha256(path)})
        for row in manifest["assignments"]:
            excluded_seeds.add(int(row["seed"]))
            excluded_groups.add(str(row["initial_state_group"]))
            excluded_roles.add(str(row["split_role"]))
    if not {"primitive_qualification", "oracle_formal"} <= excluded_roles:
        raise ValueError(
            "learning split must exclude qualification and formal-oracle cohorts"
        )
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol.get("schema_version") != (
        "calibrated-interaction.oracle-target-prompt-pilot.v2"
    ):
        raise ValueError("unsupported oracle protocol exclusion source")
    excluded_seeds.update(int(seed) for seed in protocol["preflight"]["source_seeds"])
    assignments = []
    next_seed = int(budget["seed_start"])
    for role in LEARNING_SPLIT_ROLES:
        for index in range(int(budget["groups_per_role"][role])):
            while next_seed in excluded_seeds:
                next_seed += 1
            group = f"piu-{role}-{index:04d}"
            if group in excluded_groups:
                raise ValueError("generated learning group collides with an exclusion")
            assignments.append(
                {
                    "initial_state_group": group,
                    "seed": next_seed,
                    "split_role": role,
                }
            )
            next_seed += 1
    result = validate_split_manifest(
        {
            "schema_version": "piu.group-split-manifest.v1",
            "status": "FROZEN_BEFORE_COLLECTION",
            "allocation_method": "prospective_without_outcome_access",
            "scenario": str(budget["scenario"]),
            "required_roles": list(LEARNING_SPLIT_ROLES),
            "assignments": assignments,
            "allocation_contract": {
                "path": portable(budget_path),
                "sha256": sha256(budget_path),
            },
            "exclusion_split_manifests": exclusions,
            "oracle_protocol": {
                "path": portable(protocol_path),
                "sha256": sha256(protocol_path),
            },
            "outcomes_loaded": False,
            "model_predictions_loaded": False,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
