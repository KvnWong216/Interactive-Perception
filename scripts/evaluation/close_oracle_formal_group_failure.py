#!/usr/bin/env python3
"""Close an interrupted oracle-formal group conservatively as paired failure."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.oracle_formal import (
    ARMS,
    RECEIPT_SCHEMA,
    artifact,
    load_oracle_formal_group_receipt,
    load_oracle_formal_schedule,
    resolve,
    sha256,
)


def local(path: Path) -> Path:
    return resolve(path, repository_root=ROOT)


def maybe_artifact(path: Path) -> dict[str, str] | None:
    return artifact(path, repository_root=ROOT) if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--execution-index", type=int, required=True)
    parser.add_argument("--endpoint-check", type=Path, required=True)
    args = parser.parse_args()
    schedule_path = local(args.schedule)
    endpoint_path = local(args.endpoint_check)
    schedule = load_oracle_formal_schedule(schedule_path, repository_root=ROOT)
    if not 0 <= args.execution_index < len(schedule["entries"]):
        raise ValueError("oracle formal execution index is outside the schedule")
    entry = schedule["entries"][args.execution_index]
    started_path = local(Path(entry["expected_started_ticket"]))
    receipt_path = local(Path(entry["expected_group_receipt"]))
    if not started_path.is_file():
        raise ValueError("an unstarted oracle formal group cannot be closed")
    if receipt_path.exists():
        raise FileExistsError("oracle formal group already has an immutable close")
    started = json.loads(started_path.read_text())
    if started.get("endpoint_check_sha256") != sha256(endpoint_path):
        raise ValueError("failure close uses another endpoint check")
    endpoint = json.loads(endpoint_path.read_text())
    session = endpoint.get("identity", {}).get("server_session_id")
    false_metrics = {
        "target_grasp_contact": False,
        "wrong_object_grasp_contact": False,
        "target_destination_final": False,
        "task_success": False,
        "target_maximum_lift_m": 0.0,
    }
    value = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "CLOSED_SINGLE_USE",
        "claim_scope": "FORMAL_ORACLE_INTENTION_TO_TREAT",
        "execution_index": args.execution_index,
        "initial_state_group": entry["initial_state_group"],
        "schedule_sha256": sha256(schedule_path),
        "started_ticket_sha256": sha256(started_path),
        "endpoint_check": artifact(endpoint_path, repository_root=ROOT),
        "server_session_id": session,
        "source_open_status": "INTERRUPTED_UNVERIFIED",
        "arm_status": {arm: "INTERRUPTED_UNVERIFIED" for arm in ARMS},
        "reports": {
            "source_open": maybe_artifact(local(Path(entry["expected_open_report"]))),
            **{
                arm: maybe_artifact(local(Path(entry["expected_arm_reports"][arm])))
                for arm in ARMS
            },
        },
        "post_open_state_sha256": None,
        "derived_outcomes": {arm: dict(false_metrics) for arm in ARMS},
        "errors": {
            "source_open": "attempt interrupted; completion could not be verified",
            **{
                arm: "attempt interrupted; completion could not be verified"
                for arm in ARMS
            },
        },
        "outcomes_entered_manually": False,
    }
    pending: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=receipt_path.parent,
            prefix=f".{receipt_path.name}.",
            suffix=".pending",
            delete=False,
        ) as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            pending = Path(handle.name)
        load_oracle_formal_group_receipt(
            pending,
            schedule_path=schedule_path,
            schedule=schedule,
            execution_index=args.execution_index,
            repository_root=ROOT,
        )
        os.link(pending, receipt_path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
