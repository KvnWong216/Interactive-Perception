#!/usr/bin/env python3
"""Count an interrupted single-use qualification attempt as a failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.primitive_registry import (
    load_primitive_qualification_execution_receipt,
    load_primitive_qualification_schedule,
)


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
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--execution-index", type=int, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    schedule_path = resolve(args.schedule)
    schedule = load_primitive_qualification_schedule(
        schedule_path, repository_root=ROOT
    )
    entry = schedule["entries"][args.execution_index]
    receipt_path = resolve(Path(entry["expected_execution_receipt"]))
    run_dir = receipt_path.parent
    attempt_path = run_dir / "started.json"
    semantic_path = run_dir / "semantic_report.json"
    if receipt_path.exists() or semantic_path.exists() or not attempt_path.is_file():
        raise ValueError("only a started, unreported qualification may be closed")
    attempt = json.loads(attempt_path.read_text())
    if (
        attempt.get("schema_version")
        != "piu.primitive-qualification-attempt.v1"
        or attempt.get("execution_index") != args.execution_index
        or attempt.get("entry") != entry
        or attempt.get("schedule_sha256") != sha256(schedule_path)
    ):
        raise ValueError("qualification failure close differs from attempt ticket")
    reason = " ".join(args.reason.split())
    if not reason:
        raise ValueError("qualification failure close requires a reason")
    receipt = {
        "schema_version": "piu.primitive-qualification-execution.v1",
        "status": "FAILED_ATTEMPT_COUNTED",
        "execution_index": args.execution_index,
        "initial_state_group": entry["initial_state_group"],
        "simulator_seed": entry["simulator_seed"],
        "candidate_id": entry["candidate_id"],
        "primitive": entry["primitive"],
        "context": entry["context"],
        "source_state_sha256": entry["source_state"]["sha256"],
        "controller_report_sha256": entry["controller_report"]["sha256"],
        "structured_subtask_sha256": entry["structured_subtask_sha256"],
        "schedule": {"path": portable(schedule_path), "sha256": sha256(schedule_path)},
        "attempt": {"path": portable(attempt_path), "sha256": sha256(attempt_path)},
        "semantic_report": None,
        "failure": {"reason": reason, "returncode": None},
    }
    with receipt_path.open("x") as handle:
        handle.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    load_primitive_qualification_execution_receipt(
        receipt_path,
        schedule_path=schedule_path,
        schedule=schedule,
        execution_index=args.execution_index,
        repository_root=ROOT,
    )
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
