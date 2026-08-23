#!/usr/bin/env python3
"""Freeze oracle-formal or sealed-test groups from an exact prospective plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.oracle_formal import load_oracle_formal_plan
from piu.splits import load_split_manifest, validate_split_manifest


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def planned_count(path: Path, purpose: str) -> tuple[int, set[int]]:
    if purpose == "oracle_formal":
        plan, _config, _pilot = load_oracle_formal_plan(
            path, repository_root=ROOT
        )
        if plan["status"] != "PROSPECTIVE_GROUP_COUNT_FROZEN":
            raise ValueError("oracle formal plan did not freeze a cohort size")
        return (
            int(plan["prospective_group_count"]),
            {int(seed) for seed in _config["preflight"]["source_seeds"]},
        )
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != "piu.formal-paired-test-plan.v1"
        or value.get("status") != "PROSPECTIVE_GROUP_COUNT_FROZEN"
    ):
        raise ValueError("main formal plan did not freeze a cohort size")
    design = value.get("design")
    if not isinstance(design, dict):
        raise TypeError("main formal plan lacks its frozen design")
    count = design.get("prospective_group_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("main formal plan has an invalid cohort size")
    return count, set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--purpose", choices=("oracle_formal", "sealed_test"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--group-prefix", required=True)
    parser.add_argument("--exclude-split", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan_path = resolve(args.plan)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("planned split manifests are immutable")
    if args.seed_start < 0 or not " ".join(args.group_prefix.split()):
        raise ValueError("seed start and group prefix must be valid")
    count, plan_excluded_seeds = planned_count(plan_path, args.purpose)
    excluded_seeds: set[int] = set(plan_excluded_seeds)
    excluded_groups: set[str] = set()
    excluded_roles: set[str] = set()
    exclusions = []
    for raw in args.exclude_split:
        path = resolve(raw)
        value = load_split_manifest(path)
        exclusions.append({"path": portable(path), "sha256": sha256(path)})
        for row in value["assignments"]:
            excluded_seeds.add(int(row["seed"]))
            excluded_groups.add(str(row["initial_state_group"]))
            excluded_roles.add(str(row["split_role"]))
    if args.purpose == "oracle_formal":
        if "primitive_qualification" not in excluded_roles:
            raise ValueError("oracle formal split must exclude OPEN qualification")
    elif not {
        "train",
        "development",
        "calibration_temperature",
        "calibration_conformal",
        "effect_calibration_temperature",
        "effect_calibration_conformal",
        "oracle_formal",
        "primitive_qualification",
    } <= excluded_roles:
        raise ValueError("sealed split does not exclude every prior mainline role")
    assignments = []
    next_seed = args.seed_start
    for index in range(count):
        while next_seed in excluded_seeds:
            next_seed += 1
        group = f"{args.group_prefix}-{index:04d}"
        if group in excluded_groups:
            raise ValueError("generated planned group collides with an exclusion")
        assignments.append(
            {
                "initial_state_group": group,
                "seed": next_seed,
                "split_role": args.purpose,
            }
        )
        next_seed += 1
    result = validate_split_manifest(
        {
            "schema_version": "piu.group-split-manifest.v1",
            "status": "FROZEN_BEFORE_COLLECTION",
            "allocation_method": "prospective_without_outcome_access",
            "scenario": "original_cluttered_drawer",
            "required_roles": [args.purpose],
            "assignments": assignments,
            "sample_size_plan": {
                "path": portable(plan_path),
                "sha256": sha256(plan_path),
            },
            "seed_start": args.seed_start,
            "seed_start_interpretation": "namespace_allocation_not_seed_search",
            "exclusion_split_manifests": exclusions,
            "outcomes_loaded": False,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
