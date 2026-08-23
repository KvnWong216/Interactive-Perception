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

from piu.formal_attempt import load_formal_schedule, validate_attempt_close
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
        ticket = row.get("formal_attempt_ticket")
        close = row.get("formal_attempt_close")
        if not all(
            isinstance(value, dict) for value in (episode, sealed, ticket, close)
        ):
            raise TypeError("formal row lacks episode/authorization provenance")
        episode_path = resolve(Path(episode["path"]))
        sealed_path = resolve(Path(sealed["path"]))
        ticket_path = resolve(Path(ticket["path"]))
        close_path = resolve(Path(close["path"]))
        if sha256(episode_path) != episode.get("sha256") or sha256(
            sealed_path
        ) != sealed.get("sha256"):
            raise ValueError("formal row provenance artifact differs from its hash")
        if sha256(ticket_path) != ticket.get("sha256") or sha256(
            close_path
        ) != close.get("sha256"):
            raise ValueError("formal attempt provenance differs from its hash")
        validate_attempt_close(
            close_path,
            ticket_path=ticket_path,
            episode_path=episode_path,
        )
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
            "formal_attempt_close_sha256": sha256(close_path),
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
    schedule = load_formal_schedule(schedule_path, repository_root=ROOT)
    if schedule.get("inputs", {}).get("split_manifest", {}).get("sha256") != sha256(
        split_path
    ):
        raise ValueError("formal schedule used another split manifest")
    schedule_entries = schedule.get("entries")
    if not isinstance(schedule_entries, list):
        raise TypeError("formal schedule entries must be a list")
    scheduled = {
        (row.get("initial_state_group"), row.get("method_id")): row
        for row in schedule_entries
    }
    if set(scheduled) != expected or len(schedule_entries) != len(expected):
        raise ValueError("formal schedule differs from the complete method matrix")
    if any(
        scheduled[(row["initial_state_group"], row["method_id"])].get(
            "simulator_seed"
        )
        != row["simulator_seed"]
        for row in rows
    ):
        raise ValueError("formal rows differ from scheduled simulator seeds")
    if any(
        scheduled[(row["initial_state_group"], row["method_id"])]
        .get("source_state", {})
        .get("sha256")
        != row["source_state_sha256"]
        for row in rows
    ):
        raise ValueError("formal rows differ from scheduled opaque source states")
    schedule_policy = schedule.get("inputs", {}).get("policy_identity", {}).get(
        "sha256"
    )
    if {row["policy_identity_sha256"] for row in rows} != {schedule_policy}:
        raise ValueError("formal rows differ from the scheduled frozen policy")
    schedule_digest = sha256(schedule_path)
    ordered_attempts = []
    for row in rows:
        ticket_path = resolve(Path(row["formal_attempt_ticket"]["path"]))
        close_path = resolve(Path(row["formal_attempt_close"]["path"]))
        ticket = json.loads(ticket_path.read_text())
        close = json.loads(close_path.read_text())
        index = ticket.get("execution_index")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or ticket.get("schedule", {}).get("sha256") != schedule_digest
            or not 0 <= index < len(schedule_entries)
            or ticket.get("entry") != schedule_entries[index]
            or close.get("execution_index") != index
        ):
            raise ValueError("formal attempt differs from the frozen execution order")
        ordered_attempts.append((index, ticket_path, close_path, ticket))
    if {row[0] for row in ordered_attempts} != set(range(len(schedule_entries))):
        raise ValueError("formal attempt indices do not cover the frozen schedule")
    ledger_dirs = {str(row[3].get("ledger_dir", "")) for row in ordered_attempts}
    if len(ledger_dirs) != 1:
        raise ValueError("formal attempts do not share one ordered ledger")
    previous_close_sha256 = None
    for index, ticket_path, close_path, ticket in sorted(ordered_attempts):
        if ticket.get("previous_close_sha256") != previous_close_sha256:
            raise ValueError("formal attempt close chain is not sequential")
        if ticket_path.resolve() != (
            Path(next(iter(ledger_dirs))) / f"{index:05d}.started.json"
        ).resolve():
            raise ValueError("formal attempt ticket path differs from ledger index")
        if close_path.resolve() != (
            Path(next(iter(ledger_dirs))) / f"{index:05d}.closed.json"
        ).resolve():
            raise ValueError("formal attempt close path differs from ledger index")
        previous_close_sha256 = sha256(close_path)
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
