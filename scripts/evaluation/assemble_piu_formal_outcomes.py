#!/usr/bin/env python3
"""Assemble authorized one-row artifacts into a complete sealed B0--B8 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.splits import load_split_manifest
from piu.statistics import load_formal_outcomes


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, nargs="+", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--formal-schedule", type=Path, required=True)
    parser.add_argument("--sealed-authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row_paths = [resolve(path) for path in args.rows]
    split_path = resolve(args.split_manifest)
    schedule_path = resolve(args.formal_schedule)
    authorization_path = resolve(args.sealed_authorization)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("formal outcome matrices are immutable")
    if len({path.resolve() for path in row_paths}) != len(row_paths):
        raise ValueError("formal row path list contains duplicates")
    rows = []
    for path in row_paths:
        loaded = load_formal_outcomes(path)
        if len(loaded) != 1:
            raise ValueError("each formal row artifact must contain exactly one row")
        row = loaded[0]
        episode = row.get("episode")
        sealed = row.get("sealed_authorization")
        if not isinstance(episode, dict) or not isinstance(sealed, dict):
            raise TypeError("formal row lacks episode/authorization provenance")
        episode_path = resolve(Path(episode["path"]))
        sealed_path = resolve(Path(sealed["path"]))
        if sha256(episode_path) != episode.get("sha256") or sha256(
            sealed_path
        ) != sealed.get("sha256"):
            raise ValueError("formal row provenance artifact differs from its hash")
        sealed_value = json.loads(sealed_path.read_text())
        if sealed_value.get("schema_version") != (
            "piu.formal-row-sealed-authorization.v1"
        ):
            raise ValueError("formal row carries an unsupported authorization")
        expected_row_authorization = {
            "episode_sha256": sha256(episode_path),
            "source_state_sha256": row["source_state_sha256"],
            "action_history_sha256": row["action_history_sha256"],
            "method_id": row["method_id"],
            "single_use_output": portable(path),
        }
        for name, value in expected_row_authorization.items():
            if sealed_value.get(name) != value:
                raise ValueError(f"formal row authorization differs at {name}")
        rows.append(row)
    split_manifest = load_split_manifest(split_path)
    sealed_groups = {
        row["initial_state_group"]
        for row in split_manifest["assignments"]
        if row["split_role"] == "sealed_test"
    }
    if {row["initial_state_group"] for row in rows} != sealed_groups:
        raise ValueError("formal rows differ from the frozen sealed-test cohort")
    methods = {f"B{index}" for index in range(9)}
    expected = {(group, method) for group in sealed_groups for method in methods}
    observed = {(row["initial_state_group"], row["method_id"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError("formal matrix must contain every sealed group x B0--B8 row")
    schedule = json.loads(schedule_path.read_text())
    if (
        schedule.get("schema_version") != "piu.formal-execution-schedule.v1"
        or schedule.get("status")
        != "FROZEN_BEFORE_FORMAL_OUTCOME_COLLECTION"
        or schedule.get("outcomes_loaded") is not False
    ):
        raise ValueError("formal matrix requires a frozen outcome-independent schedule")
    if schedule.get("inputs", {}).get("split_manifest", {}).get("sha256") != sha256(
        split_path
    ):
        raise ValueError("formal schedule used another split manifest")
    schedule_entries = schedule.get("entries")
    if not isinstance(schedule_entries, list):
        raise TypeError("formal schedule entries must be a list")
    scheduled = {
        (row.get("initial_state_group"), row.get("method_id")): row.get(
            "simulator_seed"
        )
        for row in schedule_entries
    }
    if set(scheduled) != expected or len(schedule_entries) != len(expected):
        raise ValueError("formal schedule differs from the complete method matrix")
    if any(
        scheduled[(row["initial_state_group"], row["method_id"])]
        != row["simulator_seed"]
        for row in rows
    ):
        raise ValueError("formal rows differ from scheduled simulator seeds")
    schedule_policy = schedule.get("inputs", {}).get("policy_identity", {}).get(
        "sha256"
    )
    if {row["policy_identity_sha256"] for row in rows} != {schedule_policy}:
        raise ValueError("formal rows differ from the scheduled frozen policy")
    for group in sealed_groups:
        source_hashes = {
            row["source_state_sha256"]
            for row in rows
            if row["initial_state_group"] == group
        }
        if len(source_hashes) != 1:
            raise ValueError("formal paired group mixes source-state hashes")
    authorization = json.loads(authorization_path.read_text())
    if (
        authorization.get("schema_version")
        != "piu.formal-matrix-sealed-authorization.v1"
    ):
        raise ValueError("unsupported formal-matrix sealed authorization")
    row_hashes = [sha256(path) for path in sorted(row_paths)]
    expected_authorization = {
        "row_sha256_sorted": sorted(row_hashes),
        "split_manifest_sha256": sha256(split_path),
        "formal_schedule_sha256": sha256(schedule_path),
        "single_use_output": portable(output),
    }
    for name, value in expected_authorization.items():
        if authorization.get(name) != value:
            raise ValueError(f"formal-matrix authorization differs at {name}")
    ordered = sorted(
        rows, key=lambda row: (row["initial_state_group"], row["method_id"])
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered)
    )
    load_formal_outcomes(output)
    print(
        json.dumps(
            {
                "output": portable(output),
                "sha256": sha256(output),
                "groups": len(sealed_groups),
                "rows": len(ordered),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
