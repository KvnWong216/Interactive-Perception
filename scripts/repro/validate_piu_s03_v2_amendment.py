#!/usr/bin/env python3
"""Statically validate the sealed-v1/public-v2 S03 amendment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.s03_execution import validate_s03_receipts, validate_s03_single_use_policy  # noqa: E402
from piu.s03_v2_amendment import (  # noqa: E402
    V1_OUTPUT_ROOT,
    V2_OUTPUT_ROOT,
    V2_PLAN_PATH,
    validate_s03_v1_seal,
    validate_s03_v2_runner_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-plan", type=Path, default=V2_PLAN_PATH)
    parser.add_argument("--execution-index", type=int, default=0)
    args = parser.parse_args()
    plan_path = args.execution_plan if args.execution_plan.is_absolute() else ROOT / args.execution_plan
    seal = validate_s03_v1_seal(repository_root=ROOT)
    plan, identity, model_identity, schedule, request = validate_s03_v2_runner_preflight(
        plan_path=plan_path,
        execution_index=args.execution_index,
        repository_root=ROOT,
    )
    manifest_path = ROOT / plan["logical_manifest"]["path"]
    schedule_path = ROOT / plan["parent_logical_schedule"]["path"]
    model_identity_path = ROOT / identity["model_identity"]["path"]
    ledger = validate_s03_receipts(
        ROOT / V2_OUTPUT_ROOT,
        schedule=schedule,
        schedule_path=schedule_path,
        manifest_path=manifest_path,
        identity_path=model_identity_path,
        identity=model_identity,
        repository_root=ROOT,
    )
    validate_s03_single_use_policy(
        ROOT / V2_OUTPUT_ROOT,
        execution_index=args.execution_index,
        ledger_status=ledger,
        record_id=request["record_id"],
    )
    v1_retry_rejected = False
    try:
        validate_s03_single_use_policy(
            ROOT / V1_OUTPUT_ROOT,
            execution_index=0,
            ledger_status={"next_execution_index": 1},
            record_id="s03-a-000",
        )
    except (FileExistsError, ValueError):
        v1_retry_rejected = True
    if not v1_retry_rejected:
        raise ValueError("S03 v1 index 0 unexpectedly became rerun-eligible")
    print(json.dumps({
        "status": "FROZEN_READY_BEFORE_S03_V2_OUTCOMES",
        "execution_version": identity["execution_version"],
        "v1_seal": seal,
        "v1_index_0_retry_rejected": True,
        "v2_record_count": plan["record_count"],
        "v2_ledger": ledger,
        "legacy_oracle_actionable": False,
        "inference_executed": False,
        "outcomes_generated": 0,
        "certificate_present": False,
        "files_written": False,
        "paper_claim_ready": False,
    }, indent=2))


if __name__ == "__main__":
    main()
