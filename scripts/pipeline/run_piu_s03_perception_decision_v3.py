#!/usr/bin/env python3
"""Execute one frozen S03 public-v3 record under an append-only lifecycle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.s03_execution import (  # noqa: E402
    build_record_artifact,
    build_started_receipt,
    canonical_sha256,
    execute_s03_record_backend,
    existing_artifact_references,
    sha256,
    validate_s03_outcome_schema,
    validate_s03_single_use_policy,
    write_record_and_close,
    write_started_receipt,
)
from piu.s03_v3_amendment import (  # noqa: E402
    CERTIFIED,
    EXECUTION_BLOCKED_INFRA,
    EXECUTION_COMPLETE_PENDING_CERTIFICATE,
    EXECUTION_VERSION,
    V3_OUTPUT_ROOT,
    V3_PLAN_PATH,
    validate_s03_v3_runner_preflight,
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-plan", type=Path, default=V3_PLAN_PATH)
    parser.add_argument("--execution-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=V3_OUTPUT_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--adjudicate-interrupted-as-infrastructure-failure", action="store_true")
    parser.add_argument("--allow-outcome-write", action="store_true")
    return parser.parse_args()


def _failure_message(exc: Exception) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        return json.dumps(
            {
                "exception": str(exc),
                "returncode": exc.returncode,
                "stdout": exc.stdout,
                "stderr": exc.stderr,
            },
            sort_keys=True,
        )
    return str(exc)


def main() -> None:
    args = _parse_args()
    plan_path = _resolve(args.execution_plan)
    output_root = _resolve(args.output_root)
    plan, identity, model, schedule, request, ledger, lifecycle = validate_s03_v3_runner_preflight(
        plan_path=plan_path,
        execution_index=args.execution_index,
        repository_root=ROOT,
    )
    canonical_output = _resolve(Path(identity["output"]["root"]))
    if output_root.resolve() != canonical_output.resolve():
        raise ValueError("S03 v3 output root must be the frozen canonical directory")
    if lifecycle in {EXECUTION_BLOCKED_INFRA, EXECUTION_COMPLETE_PENDING_CERTIFICATE, CERTIFIED}:
        raise ValueError(f"S03 v3 lifecycle {lifecycle} does not permit another record")
    summary = {
        "runner_status": identity["status"],
        "execution_version": EXECUTION_VERSION,
        "mode": "validate-only" if args.validate_only else "dry-run" if args.dry_run else "adjudicate" if args.adjudicate_interrupted_as_infrastructure_failure else "execute",
        "execution_index": args.execution_index,
        "record_id": request["record_id"],
        "request_sha256": canonical_sha256(request),
        "execution_plan_sha256": sha256(plan_path),
        "lifecycle_state": lifecycle,
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
        raise PermissionError("outcome-bearing S03 v3 operation requires --allow-outcome-write")

    schedule_row = schedule["records"][args.execution_index]
    manifest_path = _resolve(Path(plan["logical_manifest"]["path"]))
    manifest_row = json.loads(manifest_path.read_text())["records"][args.execution_index]
    record_dir = output_root / "records" / f"{args.execution_index:03d}_{request['record_id']}"
    started_path = output_root / "_receipts" / f"{args.execution_index:03d}.started.json"
    closed_path = output_root / "_receipts" / f"{args.execution_index:03d}.closed.json"
    if args.adjudicate_interrupted_as_infrastructure_failure:
        if ledger["in_flight"] != args.execution_index or not started_path.is_file() or closed_path.exists():
            raise ValueError("only the current unclosed S03 v3 index may be adjudicated")
        started = json.loads(started_path.read_text())
        record = build_record_artifact(
            request=request,
            schedule_row=schedule_row,
            prediction=None,
            outcome=None,
            artifacts=existing_artifact_references(record_dir, repository_root=ROOT),
            identity=model,
            infrastructure_failure={
                "stage": "INTERRUPTED_AFTER_STARTED_RECEIPT",
                "category": "OPERATOR_ADJUDICATED_INFRASTRUCTURE_FAILURE",
                "message": "started v3 execution did not produce a valid immutable record artifact",
                "consumes_execution_index": True,
                "adjudication": "closed_without_rerun_or_replacement",
            },
            inference_executed=False,
        )
        record_path = record_dir / "infrastructure_failure.json"
        validate_s03_outcome_schema(record, repository_root=ROOT, identity=model)
        write_record_and_close(output_root=output_root, record_path=record_path, record=record, started=started, repository_root=ROOT)
        print(json.dumps({**summary, "status": "CLOSED_INFRASTRUCTURE_FAILURE", "files_written": True}, indent=2))
        return

    validate_s03_single_use_policy(
        output_root,
        execution_index=args.execution_index,
        ledger_status=ledger,
        record_id=request["record_id"],
    )
    previous_close = None if args.execution_index == 0 else sha256(
        output_root / "_receipts" / f"{args.execution_index - 1:03d}.closed.json"
    )
    started = build_started_receipt(
        request=request,
        output_root=output_root,
        previous_close_sha256=previous_close,
        repository_root=ROOT,
    )
    write_started_receipt(output_root, started)
    try:
        prediction, outcome, artifacts = execute_s03_record_backend(
            request=request,
            schedule_row=schedule_row,
            manifest_row=manifest_row,
            identity=model,
            record_dir=record_dir,
            repository_root=ROOT,
        )
        record = build_record_artifact(
            request=request,
            schedule_row=schedule_row,
            prediction=prediction,
            outcome=outcome,
            artifacts=artifacts,
            identity=model,
            inference_executed=True,
        )
        record_path = record_dir / "record.json"
    except Exception as exc:
        record = build_record_artifact(
            request=request,
            schedule_row=schedule_row,
            prediction=None,
            outcome=None,
            artifacts=existing_artifact_references(record_dir, repository_root=ROOT),
            identity=model,
            infrastructure_failure={
                "stage": "BACKEND_OR_RECORD_CONSTRUCTION",
                "category": type(exc).__name__,
                "message": _failure_message(exc),
                "consumes_execution_index": True,
                "adjudication": "closed_without_rerun_or_replacement",
            },
            inference_executed=record_dir.exists(),
        )
        record_path = record_dir / "infrastructure_failure.json"
    validate_s03_outcome_schema(record, repository_root=ROOT, identity=model)
    _, close_path = write_record_and_close(
        output_root=output_root,
        record_path=record_path,
        record=record,
        started=started,
        repository_root=ROOT,
    )
    print(json.dumps({**summary, "status": record["status"], "record_artifact": str(record_path), "close_receipt": str(close_path), "inference_executed": record["inference_executed"], "outcome_present": record["outcome_present"], "files_written": True}, indent=2))
    if record["status"] == "INFRASTRUCTURE_FAILURE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
