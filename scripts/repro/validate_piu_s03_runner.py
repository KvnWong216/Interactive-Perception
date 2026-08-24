#!/usr/bin/env python3
"""Statically validate the frozen S03 runner without creating outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.s03_execution import (  # noqa: E402
    DEFAULT_CONTRACT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SCHEDULE,
    validate_s03_model_identity_contract,
    validate_s03_no_legacy_oracle_dependency,
    validate_s03_outcome_schema,
    validate_s03_receipts,
    validate_s03_runner_preflight,
    validate_s03_single_use_policy,
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--execution-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--outcome-artifact", type=Path)
    args = parser.parse_args()
    contract_path = _resolve(args.contract)
    schedule_path = _resolve(args.schedule)
    output_root = _resolve(args.output_root)
    contract, identity, schedule, request = validate_s03_runner_preflight(
        contract_path=contract_path,
        schedule_path=schedule_path,
        execution_index=args.execution_index,
        repository_root=ROOT,
    )
    identity_path = _resolve(Path(request["model_identity"]["path"]))
    manifest_path = _resolve(Path(request["manifest"]["path"]))
    validate_s03_model_identity_contract(identity_path, repository_root=ROOT)
    validate_s03_no_legacy_oracle_dependency(identity)
    ledger = validate_s03_receipts(
        output_root,
        schedule=schedule,
        schedule_path=schedule_path,
        manifest_path=manifest_path,
        identity_path=identity_path,
        identity=identity,
        repository_root=ROOT,
    )
    if ledger["in_flight"] is None:
        validate_s03_single_use_policy(
            output_root,
            execution_index=args.execution_index,
            ledger_status=ledger,
            record_id=request["record_id"],
        )
    if args.outcome_artifact is not None:
        validate_s03_outcome_schema(
            _resolve(args.outcome_artifact),
            repository_root=ROOT,
            identity=identity,
        )
    print(
        json.dumps(
            {
                "status": "FROZEN_READY_BEFORE_S03_OUTCOMES",
                "runner_id": contract["runner_id"],
                "execution_index_validated": args.execution_index,
                "ledger": ledger,
                "legacy_oracle_dependency": False,
                "inference_executed": False,
                "outcome_present": False,
                "files_written": False,
                "paper_claim_ready": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
