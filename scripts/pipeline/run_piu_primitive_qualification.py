#!/usr/bin/env python3
"""Execute exactly the next frozen primitive-qualification attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from interactive_perception.policy_client import OpenPiWebsocketPolicy
from piu.policy_identity import load_checkpoint_identity, validate_server_metadata
from piu.primitive_registry import (
    load_primitive_qualification_execution_receipt,
    load_primitive_qualification_schedule,
    load_qualification_controller_decision,
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


def reference(path: Path) -> dict[str, str]:
    return {"path": portable(path), "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--execution-index", type=int, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--endpoint-timeout", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    schedule_path = resolve(args.schedule)
    schedule = load_primitive_qualification_schedule(
        schedule_path, repository_root=ROOT
    )
    entries = schedule["entries"]
    if (
        not isinstance(args.execution_index, int)
        or isinstance(args.execution_index, bool)
        or not 0 <= args.execution_index < len(entries)
    ):
        raise ValueError("qualification execution index is outside schedule")
    for index, prior in enumerate(entries[: args.execution_index]):
        receipt_path = resolve(Path(prior["expected_execution_receipt"]))
        if not receipt_path.is_file():
            raise ValueError("qualification attempts must follow frozen order")
        load_primitive_qualification_execution_receipt(
            receipt_path,
            schedule_path=schedule_path,
            schedule=schedule,
            execution_index=index,
            repository_root=ROOT,
        )
    if any(
        resolve(Path(row["expected_execution_receipt"])).exists()
        for row in entries[args.execution_index + 1 :]
    ):
        raise ValueError("qualification schedule contains a later completed attempt")
    entry = entries[args.execution_index]
    receipt_path = resolve(Path(entry["expected_execution_receipt"]))
    run_dir = receipt_path.parent
    if run_dir.exists():
        raise FileExistsError("qualification attempt is immutable and cannot rerun")
    controller_path = resolve(Path(entry["controller_report"]["path"]))
    decision = load_qualification_controller_decision(
        controller_path,
        candidate_id=str(entry["candidate_id"]),
        primitive=str(entry["primitive"]),
        initial_state_group=str(entry["initial_state_group"]),
    )
    baseline_path = resolve(Path(schedule["inputs"]["baseline_registry"]["path"]))
    scenario_path = resolve(Path(schedule["inputs"]["scenario_config"]["path"]))
    identity_path = resolve(Path(schedule["inputs"]["policy_identity"]["path"]))
    source_path = resolve(Path(entry["source_state"]["path"]))
    baseline = yaml.safe_load(baseline_path.read_text())
    primitive = str(entry["primitive"])
    semantic_path = run_dir / "semantic_report.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/pipeline/execute.py"),
        "--scenario-config",
        str(scenario_path),
        "--role",
        f"PIU_QUALIFY_{primitive}",
        "--prompt",
        decision["structured_subtask"],
        "--seed",
        str(entry["simulator_seed"]),
        "--steps",
        str(baseline["shared_contract"]["option_step_budgets"][primitive]),
        "--replan-steps",
        str(baseline["shared_contract"]["replanning_steps"]),
        "--report-schema",
        "v2",
        "--external-server",
        "--expected-policy-identity",
        str(identity_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--server-timeout",
        str(args.endpoint_timeout),
        "--initial-state",
        str(source_path),
        "--state-key",
        str(entry["state_key"]),
        "--assets",
        str(run_dir / "assets"),
        "--work",
        str(run_dir / "work"),
        "--output",
        str(semantic_path),
        "--final-state",
        str(run_dir / "final_state.npz"),
    ]
    if primitive in {"PICK", "PICK_TO_INSPECT"}:
        command.append("--preserve-grasp")
    plan = {
        "schema_version": "piu.primitive-qualification-execution-plan.v1",
        "execution_index": args.execution_index,
        "candidate_id": entry["candidate_id"],
        "primitive": primitive,
        "external_server_only": True,
        "qualification_schedule_authorizes_measurement_only": True,
        "paper_method_dispatch_authorized": False,
        "command": command,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if args.endpoint_timeout <= 0:
        raise ValueError("endpoint timeout must be positive")
    # Infrastructure availability and policy identity are checked before the
    # single-use ticket exists. A network setup failure is therefore retryable;
    # once STARTED is written, every terminal failure enters the denominator.
    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port)
    validate_server_metadata(
        policy.server_metadata, load_checkpoint_identity(identity_path)
    )
    run_dir.mkdir(parents=True)
    attempt_path = run_dir / "started.json"
    attempt = {
        "schema_version": "piu.primitive-qualification-attempt.v1",
        "status": "STARTED_SINGLE_USE",
        "outcomes_loaded": False,
        "execution_index": args.execution_index,
        "entry": entry,
        "schedule_sha256": sha256(schedule_path),
        "previous_receipt_sha256": (
            None
            if args.execution_index == 0
            else sha256(
                resolve(
                    Path(
                        entries[args.execution_index - 1][
                            "expected_execution_receipt"
                        ]
                    )
                )
            )
        ),
        "endpoint": {"host": args.host, "port": args.port},
    }
    with attempt_path.open("x") as handle:
        handle.write(json.dumps(attempt, indent=2) + "\n")
    completed = subprocess.run(command, cwd=ROOT, check=False)
    common = {
        "schema_version": "piu.primitive-qualification-execution.v1",
        "execution_index": args.execution_index,
        "initial_state_group": entry["initial_state_group"],
        "simulator_seed": entry["simulator_seed"],
        "candidate_id": entry["candidate_id"],
        "primitive": primitive,
        "context": entry["context"],
        "source_state_sha256": entry["source_state"]["sha256"],
        "controller_report_sha256": entry["controller_report"]["sha256"],
        "structured_subtask_sha256": entry["structured_subtask_sha256"],
        "schedule": reference(schedule_path),
        "attempt": reference(attempt_path),
    }
    if completed.returncode == 0:
        if not semantic_path.is_file():
            raise RuntimeError("qualification subprocess returned without a report")
        receipt = {
            **common,
            "status": "COMPLETED",
            "semantic_report": reference(semantic_path),
            "failure": None,
        }
    else:
        if semantic_path.exists():
            raise RuntimeError("failed qualification wrote an ambiguous semantic report")
        receipt = {
            **common,
            "status": "FAILED_ATTEMPT_COUNTED",
            "semantic_report": None,
            "failure": {
                "reason": "qualification execution subprocess returned nonzero",
                "returncode": completed.returncode,
            },
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
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
