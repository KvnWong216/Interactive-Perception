#!/usr/bin/env python3
"""Run the B1 external prompted-VLM router in a qualified PIU closed loop."""

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

from piu.formal_attempt import artifact as formal_artifact
from piu.formal_attempt import validate_attempt_ticket
from piu.primitive_registry import load_qualified_executor_map


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def candidate_identity(path: Path, sample_id: str) -> tuple[str, str, tuple[str, ...]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    matches = [row for row in rows if row.get("sample_id") == sample_id]
    if len(matches) != 1:
        raise ValueError("B1 initial sample must select one candidate set")
    row = matches[0]
    if row.get("schema_version") != "piu.public-candidate-set.v1":
        raise ValueError("unsupported B1 public candidate-set schema")
    group = " ".join(str(row.get("initial_state_group", "")).split())
    split = str(row.get("split", ""))
    if not group or split not in {"development", "sealed_test"}:
        raise ValueError("B1 closed loop requires development or sealed_test")
    physical = tuple(
        str(candidate["candidate_id"])
        for candidate in row.get("candidates", ())
        if str(candidate.get("primitive", "")).upper()
        not in {"STOP", "REPORT_NOT_FOUND"}
    )
    if not physical or len(set(physical)) != len(physical):
        raise ValueError("B1 candidate set requires unique physical candidates")
    return group, split, physical


def qualifications(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    return load_qualified_executor_map(path, repository_root=ROOT)


def capture_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/data/capture_piu_initial_observation.py"),
        "--scenario-config",
        str(args.scenario_config),
        "--candidate-set",
        str(args.candidate_set),
        "--sample-id",
        args.initial_sample_id,
        "--seed",
        str(args.seed),
        "--output-dir",
        str(output_dir / "step000/capture"),
        "--external-simulator",
    ]
    if args.initial_state is not None:
        command.extend(
            ["--initial-state", str(args.initial_state), "--state-key", args.state_key]
        )
    return command


def router_command(
    args: argparse.Namespace,
    *,
    transition: Path,
    sample_id: str,
    split: str,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/pipeline/run_piu_prompted_vlm_router.py"),
        "--public-transition",
        str(transition),
        "--sample-id",
        sample_id,
        "--router-identity",
        str(args.router_identity),
        "--host",
        args.router_host,
        "--port",
        str(args.router_port),
        "--timeout",
        str(args.router_timeout),
        "--expected-split",
        split,
        "--output",
        str(output),
    ]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-config", type=Path, required=True)
    parser.add_argument(
        "--baseline-registry",
        type=Path,
        default=ROOT / "configs/experiments/piu_baselines_v1.yaml",
    )
    parser.add_argument("--candidate-set", type=Path, required=True)
    parser.add_argument("--initial-sample-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--state-key", default="state")
    parser.add_argument("--router-identity", type=Path, required=True)
    parser.add_argument("--router-host", required=True)
    parser.add_argument("--router-port", type=int, required=True)
    parser.add_argument("--router-timeout", type=float, default=30.0)
    parser.add_argument("--pi05-host", required=True)
    parser.add_argument("--pi05-port", type=int, default=8002)
    parser.add_argument("--qualification-map", type=Path)
    parser.add_argument("--maximum-decisions", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formal-attempt-ticket", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in (
        "scenario_config",
        "baseline_registry",
        "candidate_set",
        "initial_state",
        "router_identity",
        "qualification_map",
        "formal_attempt_ticket",
        "output_dir",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, resolve(value))
    if args.router_timeout <= 0:
        raise ValueError("B1 router timeout must be positive")
    for path in (
        args.scenario_config,
        args.baseline_registry,
        args.candidate_set,
        args.router_identity,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    registry = yaml.safe_load(args.baseline_registry.read_text())
    if registry.get("schema_version") != "piu.baseline-registry.v1":
        raise ValueError("unsupported B1 baseline registry")
    frozen_decisions = registry["shared_contract"].get(
        "maximum_controller_decisions"
    )
    if (
        not isinstance(frozen_decisions, int)
        or isinstance(frozen_decisions, bool)
        or frozen_decisions <= 0
    ):
        raise ValueError("B1 registry has an invalid controller decision cap")
    if args.maximum_decisions is None:
        args.maximum_decisions = frozen_decisions
    if args.maximum_decisions != frozen_decisions:
        raise ValueError("B1 decision cap differs from baseline registry")
    group, split, physical = candidate_identity(
        args.candidate_set, args.initial_sample_id
    )
    qualified = qualifications(args.qualification_map)
    capture = capture_command(args, args.output_dir)
    first_transition = args.output_dir / "step000/capture/public_transition.jsonl"
    first_controller = args.output_dir / "step000/controller.json"
    first_router = router_command(
        args,
        transition=first_transition,
        sample_id=args.initial_sample_id,
        split=split,
        output=first_controller,
    )
    missing = sorted(set(physical) - set(qualified))
    attempt = None
    if args.formal_attempt_ticket is not None:
        if args.dry_run:
            raise ValueError("B1 dry-run cannot consume a formal attempt ticket")
        if split != "sealed_test" or args.initial_state is None:
            raise ValueError(
                "B1 formal ticket requires sealed_test and an exact initial state"
            )
        validate_attempt_ticket(
            args.formal_attempt_ticket,
            repository_root=ROOT,
            method_id="B1",
            initial_state_group=group,
            simulator_seed=args.seed,
            source_state=args.initial_state,
            output_dir=args.output_dir,
            baseline_registry=args.baseline_registry,
            scenario_config=args.scenario_config,
        )
        attempt = formal_artifact(args.formal_attempt_ticket, repository_root=ROOT)
    elif split == "sealed_test" and not args.dry_run:
        raise ValueError("sealed B1 execution requires its ordered attempt ticket")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": "piu.prompted-vlm-closed-loop-plan.v1",
                    "method_id": "B1",
                    "initial_state_group": group,
                    "split": split,
                    "router_endpoint": f"{args.router_host}:{args.router_port}",
                    "pi05_endpoint": f"{args.pi05_host}:{args.pi05_port}",
                    "local_models_loaded": [],
                    "first_decision_commands": [capture, first_router],
                    "unqualified_physical_candidate_ids": missing,
                    "execution_ready": not missing,
                    "formal_attempt_ticket_required_for_execution": (
                        split == "sealed_test"
                    ),
                },
                indent=2,
            )
        )
        return
    if missing:
        raise ValueError(f"B1 candidate set contains unqualified actions {missing}")
    if args.output_dir.exists():
        raise FileExistsError("B1 closed-loop run directories are immutable")
    args.output_dir.mkdir(parents=True)
    run(capture)
    transition = first_transition
    current_state = args.output_dir / "step000/capture/initial_state.npz"
    execution_initial_state = current_state
    source_state = args.initial_state or execution_initial_state
    receipts = []
    status = "TIMEOUT"
    for step in range(args.maximum_decisions):
        step_dir = args.output_dir / f"step{step:03d}"
        if step:
            step_dir.mkdir()
        sample_id = (
            args.initial_sample_id
            if step == 0
            else f"{args.initial_sample_id}-step{step:03d}"
        )
        controller = step_dir / "controller.json"
        command = router_command(
            args,
            transition=transition,
            sample_id=sample_id,
            split=split,
            output=controller,
        )
        if split == "sealed_test":
            authorization = step_dir / "router_authorization.json"
            authorization.write_text(
                json.dumps(
                    {
                        "schema_version": "piu.prompted-vlm-sealed-authorization.v1",
                        "public_transition_sha256": sha256(transition),
                        "router_identity_sha256": sha256(args.router_identity),
                        "method_id": "B1",
                        "single_use_output": portable(controller),
                    },
                    indent=2,
                )
                + "\n"
            )
            command.extend(["--sealed-authorization", str(authorization)])
        run(command)
        report = json.loads(controller.read_text())
        decision = report["decisions"][0]
        kind = str(decision["decision_kind"])
        candidate_id = decision.get("selected_candidate_id")
        dispatch_dir = step_dir / "dispatch"
        dispatch = [
            sys.executable,
            str(ROOT / "scripts/pipeline/execute_piu_controller_decision.py"),
            "--controller-report",
            str(controller),
            "--sample-id",
            sample_id,
            "--scenario-config",
            str(args.scenario_config),
            "--initial-state",
            str(current_state),
            "--seed",
            str(args.seed),
            "--host",
            args.pi05_host,
            "--port",
            str(args.pi05_port),
            "--run-dir",
            str(dispatch_dir),
        ]
        if kind in {"EXECUTE", "INTERACT"}:
            if candidate_id not in qualified:
                raise ValueError(f"B1 selected unqualified candidate {candidate_id!r}")
            dispatch.extend(
                ["--primitive-qualification", str(qualified[candidate_id])]
            )
        run(dispatch)
        receipt = dispatch_dir / "dispatch.json"
        receipts.append({"path": portable(receipt), "sha256": sha256(receipt)})
        if kind not in {"EXECUTE", "INTERACT"}:
            status = "COMPLETE" if kind in {"STOP", "REPORT_NOT_FOUND"} else "ABSTAINED"
            break
        current_state = dispatch_dir / "final_state.npz"
        next_sample = f"{args.initial_sample_id}-step{step + 1:03d}"
        transition = dispatch_dir / "next_public_transition.jsonl"
        run(
            [
                sys.executable,
                str(ROOT / "scripts/data/export_piu_public_transition.py"),
                "--dispatch-receipt",
                str(receipt),
                "--sample-id",
                next_sample,
                "--initial-state-group",
                group,
                "--split",
                split,
                "--output",
                str(transition),
            ]
        )
    manifest = {
        "schema_version": "piu.closed-loop-run-manifest.v1",
        "method_id": "B1",
        "initial_state_group": group,
        "simulator_seed": args.seed,
        "split": split,
        "rollout_status": status,
        "baseline_registry": {
            "path": portable(args.baseline_registry),
            "sha256": sha256(args.baseline_registry),
        },
        "scenario_config": {
            "path": portable(args.scenario_config),
            "sha256": sha256(args.scenario_config),
        },
        "source_state": {
            "path": portable(source_state),
            "sha256": sha256(source_state),
            "state_key": args.state_key if args.initial_state is not None else "state",
        },
        "execution_initial_state": {
            "path": portable(execution_initial_state),
            "sha256": sha256(execution_initial_state),
            "state_key": "state",
        },
        "dispatch_receipts": receipts,
        "maximum_decisions": args.maximum_decisions,
        "external_pi05": f"{args.pi05_host}:{args.pi05_port}",
        "external_prompted_vlm": f"{args.router_host}:{args.router_port}",
        "router_identity": {
            "path": portable(args.router_identity),
            "sha256": sha256(args.router_identity),
        },
        "local_pi05_loaded": False,
        "paper_method_claim_allowed": False,
        "formal_attempt_ticket": attempt,
    }
    manifest_path = args.output_dir / "closed_loop_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
