#!/usr/bin/env python3
"""Execute the next single-use attempted-OPEN oracle/baseline formal group."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/infra"), str(ROOT / "src")]

from check_external_pi05 import validate_metadata, wait_for_endpoint
from piu.oracle_formal import (
    ARMS,
    RECEIPT_SCHEMA,
    artifact,
    load_oracle_formal_group_receipt,
    load_oracle_formal_schedule,
    oracle_formal_report_metrics,
    resolve,
    sha256,
    validate_oracle_formal_execution_report,
)
from piu.policy_identity import load_checkpoint_identity


def local(path: Path) -> Path:
    return resolve(path, repository_root=ROOT)


def load_endpoint(
    path: Path, *, host: str, port: int, identity: dict[str, Any]
) -> dict[str, Any]:
    value = json.loads(path.read_text())
    probe = value.get("action_probe")
    if (
        value.get("schema_version") != "piu.external-pi05-check.v1"
        or value.get("status") != "PASS"
        or value.get("endpoint") != {"host": host, "port": port}
        or not isinstance(probe, dict)
        or probe.get("finite") is not True
        or not isinstance(value.get("identity", {}).get("server_session_id"), str)
    ):
        raise ValueError("oracle formal execution requires an identified finite endpoint")
    validate_metadata(dict(value["identity"]), identity)
    return value


def command(
    *,
    entry: dict[str, Any],
    schedule: dict[str, Any],
    config: dict[str, Any],
    scenario_path: Path,
    baseline: dict[str, Any],
    identity_path: Path,
    host: str,
    port: int,
    timeout: float,
    arm: str,
) -> list[str]:
    group_root = local(Path(entry["expected_started_ticket"])).parent
    initial_state = (
        local(Path(entry["source_state"]["path"]))
        if arm == "source_open"
        else local(Path(entry["expected_post_open_state"]))
    )
    if arm == "source_open":
        run_root = group_root / "open_source"
        prompt = schedule["source_open_subtask"]
        role = "PIU_ORACLE_FORMAL_OPEN_SOURCE"
        steps = baseline["shared_contract"]["option_step_budgets"]["OPEN"]
    elif arm == "oracle_target_prompt":
        run_root = group_root / arm
        prompt = config["execution"]["prompt"]
        role = "PIU_ORACLE_FORMAL_ORACLE"
        steps = config["execution"]["steps"]
    elif arm == "raw_post_open_direct":
        run_root = group_root / arm
        scenario = yaml.safe_load(scenario_path.read_text())
        prompt = scenario["task"]["prompt"]
        role = "PIU_ORACLE_FORMAL_BASELINE"
        steps = config["execution"]["steps"]
    else:
        raise ValueError(f"unsupported oracle formal arm {arm!r}")
    result = [
        sys.executable,
        str(ROOT / "scripts/pipeline/execute.py"),
        "--scenario-config",
        str(scenario_path),
        "--role",
        role,
        "--prompt",
        str(prompt),
        "--seed",
        str(entry["simulator_seed"]),
        "--initial-state",
        str(initial_state),
        "--state-key",
        str(entry["state_key"]),
        "--steps",
        str(steps),
        "--replan-steps",
        str(baseline["shared_contract"]["replanning_steps"]),
        "--report-schema",
        "v2",
        "--external-server",
        "--expected-policy-identity",
        str(identity_path),
        "--host",
        host,
        "--port",
        str(port),
        "--server-timeout",
        str(timeout),
        "--assets",
        str(run_root / "assets"),
        "--work",
        str(run_root / "work"),
        "--output",
        str(run_root / "report.json"),
    ]
    if arm == "source_open":
        result.extend(
            ["--final-state", str(local(Path(entry["expected_post_open_state"])))]
        )
    if arm == "oracle_target_prompt":
        result.extend(
            [
                "--target-object",
                str(config["execution"]["target_object"]),
                "--oracle-target-visual-prompt",
                str(schedule["selected_style"]),
                "--oracle-minimum-visible-pixels",
                str(config["execution"]["target_presence_minimum_pixels"]),
            ]
        )
    return result


def maybe_artifact(path: Path) -> dict[str, str] | None:
    return artifact(path, repository_root=ROOT) if path.is_file() else None


def write_validated_receipt(
    value: dict[str, Any], *, output: Path, schedule_path: Path, schedule: dict[str, Any]
) -> None:
    pending: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
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
            execution_index=int(value["execution_index"]),
            repository_root=ROOT,
        )
        os.link(pending, output)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--execution-index", type=int, required=True)
    parser.add_argument("--endpoint-check", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--server-timeout", type=float, default=30.0)
    parser.add_argument(
        "--execution-location",
        choices=("external_simulator",),
        help="required live: local simulator rendering is outside the 1500 MiB contract",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.server_timeout <= 0.0:
        raise ValueError("oracle formal endpoint timeout must be positive")
    schedule_path = local(args.schedule)
    endpoint_path = local(args.endpoint_check)
    schedule = load_oracle_formal_schedule(schedule_path, repository_root=ROOT)
    if not 0 <= args.execution_index < len(schedule["entries"]):
        raise ValueError("oracle formal execution index is outside the schedule")
    entry = schedule["entries"][args.execution_index]
    for index in range(args.execution_index):
        previous = local(Path(schedule["entries"][index]["expected_group_receipt"]))
        if not previous.is_file():
            raise ValueError("oracle formal groups must execute in frozen order")
        load_oracle_formal_group_receipt(
            previous,
            schedule_path=schedule_path,
            schedule=schedule,
            execution_index=index,
            repository_root=ROOT,
        )
    for later in schedule["entries"][args.execution_index + 1 :]:
        if local(Path(later["expected_started_ticket"])).exists() or local(
            Path(later["expected_group_receipt"])
        ).exists():
            raise ValueError("oracle formal ledger contains an entry after a gap")
    config_path = local(Path(schedule["inputs"]["experiment"]["path"]))
    scenario_path = local(Path(schedule["inputs"]["scenario_config"]["path"]))
    baseline_path = local(Path(schedule["inputs"]["baseline_registry"]["path"]))
    identity_path = local(Path(schedule["inputs"]["policy_identity"]["path"]))
    config = yaml.safe_load(config_path.read_text())
    scenario = yaml.safe_load(scenario_path.read_text())
    baseline = yaml.safe_load(baseline_path.read_text())
    identity = load_checkpoint_identity(identity_path)
    endpoint = load_endpoint(
        endpoint_path, host=args.host, port=args.port, identity=identity
    )
    commands = {
        arm: command(
            entry=entry,
            schedule=schedule,
            config=config,
            scenario_path=scenario_path,
            baseline=baseline,
            identity_path=identity_path,
            host=args.host,
            port=args.port,
            timeout=args.server_timeout,
            arm=arm,
        )
        for arm in ("source_open", *entry["arm_order"])
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": "piu.oracle-formal-group-execution-plan.v1",
                    "execution_index": args.execution_index,
                    "initial_state_group": entry["initial_state_group"],
                    "arm_order": entry["arm_order"],
                    "external_server_only": True,
                    "commands": {
                        name: shlex.join(value) for name, value in commands.items()
                    },
                    "outputs_created": False,
                },
                indent=2,
            )
        )
        return
    if args.execution_location != "external_simulator":
        raise ValueError(
            "live oracle formal execution requires the allocated external simulator"
        )
    wait_for_endpoint(args.host, args.port, args.server_timeout)
    live = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/infra/check_external_pi05.py"),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--timeout",
            str(args.server_timeout),
            "--identity",
            str(identity_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    live_identity = json.loads(live.stdout)["identity"]
    session = endpoint["identity"]["server_session_id"]
    if live_identity.get("server_session_id") != session:
        raise ValueError("oracle formal policy server restarted after endpoint preflight")
    started_path = local(Path(entry["expected_started_ticket"]))
    group_root = started_path.parent
    if group_root.exists():
        raise FileExistsError("oracle formal group output is immutable")
    group_root.mkdir(parents=True)
    previous_digest = None
    if args.execution_index > 0:
        previous_digest = sha256(
            local(
                Path(
                    schedule["entries"][args.execution_index - 1][
                        "expected_group_receipt"
                    ]
                )
            )
        )
    started = {
        "schema_version": "piu.oracle-formal-group-start.v1",
        "status": "STARTED_SINGLE_USE",
        "claim_scope": "FORMAL_ORACLE_ATTEMPT_NO_OUTCOMES_LOADED",
        "outcomes_loaded": False,
        "execution_index": args.execution_index,
        "entry": entry,
        "schedule_sha256": sha256(schedule_path),
        "endpoint_check_sha256": sha256(endpoint_path),
        "previous_receipt_sha256": previous_digest,
    }
    with started_path.open("x") as handle:
        handle.write(json.dumps(started, indent=2) + "\n")
    statuses = {arm: "NOT_RUN_OPEN_FAILED" for arm in ARMS}
    errors: dict[str, str | None] = {"source_open": None, **{arm: None for arm in ARMS}}
    false_metrics = {
        "target_grasp_contact": False,
        "wrong_object_grasp_contact": False,
        "target_destination_final": False,
        "task_success": False,
        "target_maximum_lift_m": 0.0,
    }
    metrics = {arm: dict(false_metrics) for arm in ARMS}
    open_report = local(Path(entry["expected_open_report"]))
    post_open = local(Path(entry["expected_post_open_state"]))
    open_run = subprocess.run(commands["source_open"], cwd=ROOT, check=False)
    source_status = "FAILED"
    if open_run.returncode == 0 and open_report.is_file() and post_open.is_file():
        try:
            validate_oracle_formal_execution_report(
                open_report,
                entry=entry,
                arm="source_open",
                schedule=schedule,
                config=config,
                scenario=scenario,
                identity=identity,
                post_open_state_path=None,
                repository_root=ROOT,
            )
            source_status = "COMPLETE"
        except Exception as error:  # retain invalid external output as failure
            errors["source_open"] = f"{type(error).__name__}: {error}"
    else:
        errors["source_open"] = f"external executor return code {open_run.returncode}"
    if source_status == "COMPLETE":
        for arm in entry["arm_order"]:
            report_path = local(Path(entry["expected_arm_reports"][arm]))
            completed = subprocess.run(commands[arm], cwd=ROOT, check=False)
            statuses[arm] = "FAILED"
            if completed.returncode == 0 and report_path.is_file():
                try:
                    report, observed = validate_oracle_formal_execution_report(
                        report_path,
                        entry=entry,
                        arm=arm,
                        schedule=schedule,
                        config=config,
                        scenario=scenario,
                        identity=identity,
                        post_open_state_path=post_open,
                        repository_root=ROOT,
                    )
                    metrics[arm] = oracle_formal_report_metrics(report, config)
                    statuses[arm] = "COMPLETE"
                except Exception as error:  # invalid output stays in the denominator
                    errors[arm] = f"{type(error).__name__}: {error}"
            else:
                errors[arm] = f"external executor return code {completed.returncode}"
    receipt_path = local(Path(entry["expected_group_receipt"]))
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "CLOSED_SINGLE_USE",
        "claim_scope": "FORMAL_ORACLE_INTENTION_TO_TREAT",
        "execution_index": args.execution_index,
        "initial_state_group": entry["initial_state_group"],
        "schedule_sha256": sha256(schedule_path),
        "started_ticket_sha256": sha256(started_path),
        "endpoint_check": artifact(endpoint_path, repository_root=ROOT),
        "server_session_id": session,
        "source_open_status": source_status,
        "arm_status": statuses,
        "reports": {
            "source_open": maybe_artifact(open_report),
            **{
                arm: maybe_artifact(local(Path(entry["expected_arm_reports"][arm])))
                for arm in ARMS
            },
        },
        "post_open_state_sha256": sha256(post_open) if source_status == "COMPLETE" else None,
        "derived_outcomes": metrics,
        "errors": errors,
        "outcomes_entered_manually": False,
    }
    write_validated_receipt(
        receipt,
        output=receipt_path,
        schedule_path=schedule_path,
        schedule=schedule,
    )
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
