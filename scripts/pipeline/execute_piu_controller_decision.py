#!/usr/bin/env python3
"""Execute one authorized PIU subtask through an external frozen pi0.5 server."""

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

from piu.contracts import assert_public_policy_value


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
    parser.add_argument("--controller-report", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--scenario-config", type=Path, required=True)
    parser.add_argument(
        "--baseline-registry",
        type=Path,
        default=ROOT / "configs/experiments/piu_baselines_v1.yaml",
    )
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument(
        "--primitive-qualification",
        type=Path,
        help="FORMALLY_QUALIFIED certificate; mandatory for a physical dispatch",
    )
    parser.add_argument("--state-key", default="state")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in (
        "controller_report",
        "scenario_config",
        "baseline_registry",
        "initial_state",
        "primitive_qualification",
        "run_dir",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, resolve(value))
    report = json.loads(args.controller_report.read_text())
    if report.get("schema_version") not in {
        "piu.calibrated-controller-report.v1",
        "piu.uncalibrated-ablation-controller-report.v1",
        "piu.prompted-vlm-router-report.v1",
    }:
        raise ValueError("unsupported PIU controller report")
    if report.get("evaluator_labels_loaded") is not False:
        raise ValueError("physical execution refuses a label-bearing controller report")
    matches = [
        row for row in report["decisions"] if row.get("sample_id") == args.sample_id
    ]
    if len(matches) != 1:
        raise ValueError("sample ID must select exactly one controller decision")
    decision = matches[0]
    kind = str(decision["decision_kind"])
    if kind not in {"EXECUTE", "INTERACT"}:
        result = {
            "schema_version": "piu.executor-dispatch.v1",
            "sample_id": args.sample_id,
            "method_id": report.get("method_id"),
            "initial_state_group": decision.get("initial_state_group"),
            "decision_kind": kind,
            "physical_action_dispatched": False,
            "reason": decision["reason"],
            "controller_report": {
                "path": portable(args.controller_report),
                "sha256": sha256(args.controller_report),
            },
            "evaluator_fields_copied": [],
        }
        if args.dry_run:
            print(json.dumps(result, indent=2))
            return
        args.run_dir.mkdir(parents=True, exist_ok=False)
        (args.run_dir / "dispatch.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    subtask = decision.get("structured_pi05_subtask")
    candidate_id = decision.get("selected_candidate_id")
    if not isinstance(subtask, str) or not subtask.strip() or not candidate_id:
        raise ValueError("authorized physical decision lacks a structured subtask")
    selected_candidate = decision.get("selected_candidate")
    public_candidates = decision.get("public_candidates")
    if not isinstance(selected_candidate, dict) or not isinstance(
        public_candidates, list
    ):
        raise TypeError("physical decision lacks public candidate payloads")
    assert_public_policy_value(selected_candidate, path="dispatch.selected_candidate")
    assert_public_policy_value(public_candidates, path="dispatch.public_candidates")
    matches = [
        row
        for row in public_candidates
        if row.get("candidate_id") == candidate_id
        and str(row.get("primitive", "")).upper()
        == str(decision.get("selected_candidate_primitive", "")).upper()
    ]
    if matches != [selected_candidate]:
        raise ValueError("selected candidate differs from the public candidate set")
    baseline = yaml.safe_load(args.baseline_registry.read_text())
    if baseline.get("schema_version") != "piu.baseline-registry.v1":
        raise ValueError("unsupported baseline/budget registry")
    primitive = " ".join(
        str(decision.get("selected_candidate_primitive", "")).split()
    ).upper()
    budgets = baseline["shared_contract"]["option_step_budgets"]
    policy_identity = resolve(
        Path(baseline["shared_contract"]["checkpoint_identity"])
    )
    if not policy_identity.is_file():
        raise FileNotFoundError(policy_identity)
    if primitive not in budgets:
        raise ValueError("selected primitive has no shared qualified step budget")
    qualification = None
    if args.primitive_qualification is not None:
        qualification = json.loads(args.primitive_qualification.read_text())
        if (
            qualification.get("schema_version")
            != "piu.primitive-qualification-certificate.v1"
        ):
            raise ValueError("unsupported primitive qualification certificate")
        if qualification.get("status") != "FORMALLY_QUALIFIED":
            raise ValueError("selected physical action is not formally qualified")
        if qualification.get("paper_method_action_authorized") is not True:
            raise ValueError(
                "primitive certificate does not authorize a paper-method action"
            )
        if str(qualification.get("candidate_id")) != str(candidate_id):
            raise ValueError(
                "primitive certificate candidate differs from controller decision"
            )
        if str(qualification.get("primitive", "")).upper() != primitive:
            raise ValueError(
                "primitive certificate action differs from controller decision"
            )
        result = qualification.get("result", {})
        if result.get("qualified") is not True:
            raise ValueError("primitive certificate result is not qualified")
        if result.get("trials") != len(qualification.get("initial_state_groups", [])):
            raise ValueError("primitive certificate denominator is incomplete")
    run_report = args.run_dir / "report.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/pipeline/execute.py"),
        "--scenario-config",
        str(args.scenario_config),
        "--role",
        f"PIU_{primitive}",
        "--prompt",
        subtask,
        "--seed",
        str(args.seed),
        "--steps",
        str(budgets[primitive]),
        "--replan-steps",
        str(baseline["shared_contract"]["replanning_steps"]),
        "--report-schema",
        "v2",
        "--external-server",
        "--expected-policy-identity",
        str(policy_identity),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--assets",
        str(args.run_dir / "assets"),
        "--work",
        str(args.run_dir / "work"),
        "--output",
        str(run_report),
        "--final-state",
        str(args.run_dir / "final_state.npz"),
    ]
    if args.initial_state is not None:
        if not args.initial_state.is_file():
            raise FileNotFoundError(args.initial_state)
        command.extend(
            ["--initial-state", str(args.initial_state), "--state-key", args.state_key]
        )
    if primitive in {"PICK", "PICK_TO_INSPECT"}:
        command.append("--preserve-grasp")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": "piu.executor-dispatch-plan.v1",
                    "sample_id": args.sample_id,
                    "candidate_id": candidate_id,
                    "primitive": primitive,
                    "external_server_only": True,
                    "primitive_formally_qualified": qualification is not None,
                    "primitive_qualification_sha256": (
                        sha256(args.primitive_qualification)
                        if args.primitive_qualification is not None
                        else None
                    ),
                    "physical_dispatch_allowed": qualification is not None,
                    "command": command,
                },
                indent=2,
            )
        )
        return
    if qualification is None:
        raise ValueError(
            "physical PIU dispatch requires a formally qualified primitive certificate"
        )
    if args.run_dir.exists():
        raise FileExistsError(f"PIU execution run is immutable: {args.run_dir}")
    subprocess.run(command, cwd=ROOT, check=True)
    execution = json.loads(run_report.read_text())
    if execution.get("controller", {}).get("online_oracle_inputs"):
        raise ValueError("PIU execution unexpectedly consumed online oracle inputs")
    scenario = yaml.safe_load(args.scenario_config.read_text())
    receipt = {
        "schema_version": "piu.executor-dispatch.v1",
        "sample_id": args.sample_id,
        "method_id": report.get("method_id"),
        "initial_state_group": decision.get("initial_state_group"),
        "decision_kind": kind,
        "candidate_id": candidate_id,
        "primitive": primitive,
        "selected_candidate": selected_candidate,
        "public_candidates": public_candidates,
        "task_prompt": scenario["task"]["prompt"],
        "physical_action_dispatched": True,
        "controller_report": {
            "path": portable(args.controller_report),
            "sha256": sha256(args.controller_report),
        },
        "primitive_qualification": {
            "path": portable(args.primitive_qualification),
            "sha256": sha256(args.primitive_qualification),
        },
        "execution_report": {
            "path": portable(run_report),
            "sha256": sha256(run_report),
        },
        "source_initial_state_transport": (
            {
                "path": portable(args.initial_state),
                "sha256": sha256(args.initial_state),
                "state_key": args.state_key,
            }
            if args.initial_state is not None
            else None
        ),
        "final_state_transport": {
            "path": portable(args.run_dir / "final_state.npz"),
            "sha256": sha256(args.run_dir / "final_state.npz"),
            "state_key": "state",
        },
        "evaluator_fields_copied": [],
    }
    (args.run_dir / "dispatch.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
