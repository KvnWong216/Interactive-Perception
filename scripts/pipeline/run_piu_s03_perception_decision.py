#!/usr/bin/env python3
"""Execute one frozen public-input S03 record with ordered single-use receipts.

``--validate-only`` and ``--dry-run`` are guaranteed not to create files or
invoke the perception backend.  Outcome-bearing execution additionally
requires ``--allow-outcome-write``; this task freezes that path but does not
invoke it.
"""

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
    build_record_artifact,
    build_started_receipt,
    canonical_sha256,
    execute_s03_record_backend,
    existing_artifact_references,
    sha256,
    validate_s03_outcome_schema,
    validate_s03_receipts,
    validate_s03_runner_preflight,
    validate_s03_single_use_policy,
    write_record_and_close,
    write_started_receipt,
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--execution-index", type=int, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument(
        "--adjudicate-interrupted-as-infrastructure-failure",
        action="store_true",
        help="close an already-started index without rerunning inference",
    )
    parser.add_argument(
        "--allow-outcome-write",
        action="store_true",
        help="required for either execution or interruption adjudication",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    schedule_path = _resolve(args.schedule)
    contract_path = _resolve(args.contract)
    output_root = _resolve(args.output_root)
    contract, identity, schedule, request = validate_s03_runner_preflight(
        contract_path=contract_path,
        schedule_path=schedule_path,
        execution_index=args.execution_index,
        repository_root=ROOT,
    )
    canonical_output_root = _resolve(Path(contract["output"]["root"]))
    if output_root.resolve() != canonical_output_root.resolve():
        raise ValueError("S03 runner output root must be the frozen canonical directory")
    manifest_path = _resolve(Path(request["manifest"]["path"]))
    identity_path = _resolve(Path(request["model_identity"]["path"]))
    ledger = validate_s03_receipts(
        output_root,
        schedule=schedule,
        schedule_path=schedule_path,
        manifest_path=manifest_path,
        identity_path=identity_path,
        identity=identity,
        repository_root=ROOT,
    )
    summary = {
        "runner_status": "FROZEN_READY_BEFORE_S03_OUTCOMES",
        "mode": "validate-only" if args.validate_only else "dry-run" if args.dry_run else "adjudicate" if args.adjudicate_interrupted_as_infrastructure_failure else "execute",
        "execution_index": args.execution_index,
        "record_id": request["record_id"],
        "request_sha256": canonical_sha256(request),
        "schedule_sha256": sha256(schedule_path),
        "model_identity_sha256": sha256(identity_path),
        "ledger": ledger,
        "inference_executed": False,
        "outcome_present": False,
        "files_written": False,
        "paper_claim_ready": False,
    }
    if args.validate_only:
        print(json.dumps(summary, indent=2))
        return
    if args.dry_run:
        summary["policy_request"] = request["policy_request"]
        print(json.dumps(summary, indent=2))
        return
    if not args.allow_outcome_write:
        raise PermissionError(
            "outcome-bearing S03 operation requires --allow-outcome-write"
        )

    row = schedule["records"][args.execution_index]
    manifest = json.loads(manifest_path.read_text())
    manifest_row = manifest["records"][args.execution_index]
    started_path = output_root / "_receipts" / f"{args.execution_index:03d}.started.json"
    closed_path = output_root / "_receipts" / f"{args.execution_index:03d}.closed.json"
    record_dir = output_root / "records" / f"{args.execution_index:03d}_{request['record_id']}"

    if args.adjudicate_interrupted_as_infrastructure_failure:
        if ledger["in_flight"] != args.execution_index or not started_path.is_file() or closed_path.exists():
            raise ValueError("only the one current unclosed S03 index may be adjudicated")
        started = json.loads(started_path.read_text())
        failure = {
            "stage": "INTERRUPTED_AFTER_STARTED_RECEIPT",
            "category": "OPERATOR_ADJUDICATED_INFRASTRUCTURE_FAILURE",
            "message": "started execution did not produce a valid immutable record artifact",
            "consumes_execution_index": True,
            "adjudication": "closed_without_rerun_or_replacement",
        }
        record = build_record_artifact(
            request=request,
            schedule_row=row,
            prediction=None,
            outcome=None,
            artifacts=existing_artifact_references(
                record_dir, repository_root=ROOT
            ),
            identity=identity,
            infrastructure_failure=failure,
            inference_executed=False,
        )
        record_path = record_dir / contract["output"]["infrastructure_failure_artifact_name"]
        validate_s03_outcome_schema(record, repository_root=ROOT, identity=identity)
        write_record_and_close(
            output_root=output_root,
            record_path=record_path,
            record=record,
            started=started,
            repository_root=ROOT,
        )
        print(json.dumps({**summary, "status": "CLOSED_INFRASTRUCTURE_FAILURE", "files_written": True}, indent=2))
        return

    validate_s03_single_use_policy(
        output_root,
        execution_index=args.execution_index,
        ledger_status=ledger,
        record_id=request["record_id"],
    )
    previous_close_sha256 = None
    if args.execution_index:
        previous_close_sha256 = sha256(
            output_root / "_receipts" / f"{args.execution_index - 1:03d}.closed.json"
        )
    started = build_started_receipt(
        request=request,
        output_root=output_root,
        previous_close_sha256=previous_close_sha256,
        repository_root=ROOT,
    )
    write_started_receipt(output_root, started)
    try:
        prediction, outcome, artifacts = execute_s03_record_backend(
            request=request,
            schedule_row=row,
            manifest_row=manifest_row,
            identity=identity,
            record_dir=record_dir,
            repository_root=ROOT,
        )
        record = build_record_artifact(
            request=request,
            schedule_row=row,
            prediction=prediction,
            outcome=outcome,
            artifacts=artifacts,
            identity=identity,
            inference_executed=True,
        )
        record_path = record_dir / contract["output"]["record_artifact_name"]
    except Exception as exc:
        failure = {
            "stage": "BACKEND_OR_RECORD_CONSTRUCTION",
            "category": type(exc).__name__,
            "message": str(exc),
            "consumes_execution_index": True,
            "adjudication": "closed_without_rerun_or_replacement",
        }
        record = build_record_artifact(
            request=request,
            schedule_row=row,
            prediction=None,
            outcome=None,
            artifacts=existing_artifact_references(
                record_dir, repository_root=ROOT
            ),
            identity=identity,
            infrastructure_failure=failure,
            inference_executed=record_dir.exists(),
        )
        record_path = record_dir / contract["output"]["infrastructure_failure_artifact_name"]
    validate_s03_outcome_schema(record, repository_root=ROOT, identity=identity)
    _, close_path = write_record_and_close(
        output_root=output_root,
        record_path=record_path,
        record=record,
        started=started,
        repository_root=ROOT,
    )
    print(
        json.dumps(
            {
                **summary,
                "status": record["status"],
                "record_artifact": str(record_path),
                "close_receipt": str(close_path),
                "inference_executed": record["inference_executed"],
                "outcome_present": record["outcome_present"],
                "files_written": True,
            },
            indent=2,
        )
    )
    if record["status"] == "INFRASTRUCTURE_FAILURE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
