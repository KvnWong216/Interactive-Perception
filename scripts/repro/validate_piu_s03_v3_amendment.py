#!/usr/bin/env python3
"""Validate the prospective S03 public-v3 amendment without inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.s03_v3_amendment import (  # noqa: E402
    V3_PLAN_PATH,
    validate_s03_v2_seal,
    validate_s03_v3_runner_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-plan", type=Path, default=V3_PLAN_PATH)
    parser.add_argument("--execution-index", type=int, default=0)
    args = parser.parse_args()
    plan_path = args.execution_plan if args.execution_plan.is_absolute() else ROOT / args.execution_plan
    plan, identity, model, schedule, request, ledger, state = validate_s03_v3_runner_preflight(
        plan_path=plan_path,
        execution_index=args.execution_index,
        repository_root=ROOT,
    )
    print(
        json.dumps(
            {
                "status": "VALID",
                "execution_version": plan["execution_version"],
                "execution_index_validated": args.execution_index,
                "record_id": request["record_id"],
                "record_count": len(schedule["records"]),
                "runner_identity": identity["identity_id"],
                "model_identity": model["identity_id"],
                "v2_seal": validate_s03_v2_seal(repository_root=ROOT),
                "lifecycle_state": state,
                "ledger": ledger,
                "legacy_oracle_actionable": False,
                "inference_executed": False,
                "outcome_present": False,
                "certificate_present": False,
                "files_written": False,
                "paper_claim_ready": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
