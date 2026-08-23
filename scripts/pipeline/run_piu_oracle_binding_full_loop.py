#!/usr/bin/env python3
"""Run B7 from the paired initial state with an audited dynamic target marker."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.policy_identity import load_checkpoint_identity, validate_server_metadata


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def selected_style(path: Path, *, oracle_config: Path) -> tuple[str, dict[str, Any]]:
    value = json.loads(path.read_text())
    if (
        value.get("schema_version")
        != "calibrated-interaction.oracle-target-prompt-result.v1"
        or value.get("formal_method_claim") is not False
        or value.get("online_oracle_input_count") != 2
    ):
        raise ValueError("unsupported B7 development style-selection artifact")
    experiment = value.get("experiment", {})
    if experiment.get("sha256") != sha256(oracle_config):
        raise ValueError("B7 style selection used another oracle protocol")
    style = value.get("screen", {}).get("selected_style")
    if style not in {"box", "point", "spotlight"}:
        raise ValueError("B7 requires one uniquely selected development style")
    return str(style), value


def evaluator_outcomes(report: dict[str, Any]) -> dict[str, bool | float | int]:
    evaluator = report["evaluator"]
    target = str(evaluator.get("target_object"))
    wrong_contact = any(
        name != target and int(row.get("grasp_contact_steps", 0)) > 0
        for name, row in evaluator.get("objects", {}).items()
    )
    return {
        "target_grasp_contact": bool(
            evaluator.get("target_grasp_contact_success", False)
        ),
        "wrong_object_grasp_contact": wrong_contact,
        "target_destination_final": bool(
            evaluator.get("target_in_destination_final", False)
        ),
        "task_success": bool(evaluator.get("task_success_final", False)),
        "abstention": False,
        "target_maximum_lift_m": float(
            evaluator.get("target_maximum_lift_m") or 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-config", type=Path, required=True)
    parser.add_argument(
        "--oracle-config",
        type=Path,
        default=ROOT
        / "configs/experiments/original_drawer_oracle_target_prompt_pilot_v2.yaml",
    )
    parser.add_argument("--style-selection", type=Path, required=True)
    parser.add_argument(
        "--baseline-registry",
        type=Path,
        default=ROOT / "configs/experiments/piu_baselines_v1.yaml",
    )
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--state-key", default="state")
    parser.add_argument("--initial-state-group", required=True)
    parser.add_argument(
        "--split", choices=("development", "sealed_test"), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--server-timeout", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in (
        "scenario_config",
        "oracle_config",
        "style_selection",
        "baseline_registry",
        "initial_state",
        "output_dir",
    ):
        setattr(args, name, resolve(getattr(args, name)))
    if not args.initial_state.is_file():
        raise FileNotFoundError(args.initial_state)
    if args.server_timeout <= 0:
        raise ValueError("server-timeout must be positive")
    group = " ".join(args.initial_state_group.split())
    if not group:
        raise ValueError("initial-state-group is required")
    registry = yaml.safe_load(args.baseline_registry.read_text())
    if registry.get("schema_version") != "piu.baseline-registry.v1":
        raise ValueError("unsupported PIU baseline registry")
    methods = {row["id"]: row for row in registry["methods"]}
    if methods.get("B7", {}).get("online_privileged_inputs") != [
        "target_instance_mask",
        "target_identity",
    ]:
        raise ValueError("B7 oracle-input registry changed")
    oracle = yaml.safe_load(args.oracle_config.read_text())
    if (
        oracle.get("schema_version")
        != "calibrated-interaction.oracle-target-prompt-pilot.v2"
        or resolve(Path(oracle["scenario_config"])) != args.scenario_config
    ):
        raise ValueError("B7 oracle/scenario protocol differs")
    style, selection = selected_style(
        args.style_selection, oracle_config=args.oracle_config
    )
    shared = registry["shared_contract"]
    execution = oracle["execution"]
    direct_budget = int(shared["option_step_budgets"]["DIRECT"])
    replan_steps = int(shared["replanning_steps"])
    if (
        int(execution["steps"]) != direct_budget
        or int(execution["replan_steps"]) != replan_steps
    ):
        raise ValueError("B7 oracle protocol differs from shared option budgets")
    identity = resolve(Path(shared["checkpoint_identity"]))
    if resolve(Path(oracle["resource_contract"]["checkpoint_identity"])) != identity:
        raise ValueError("B7 oracle and baseline checkpoint identities differ")
    report = args.output_dir / "report.json"
    final_state = args.output_dir / "final_state.npz"
    command = [
        sys.executable,
        str(ROOT / "scripts/pipeline/execute.py"),
        "--scenario-config",
        str(args.scenario_config),
        "--role",
        str(execution["role"]),
        "--prompt",
        str(execution["prompt"]),
        "--initial-state",
        str(args.initial_state),
        "--state-key",
        args.state_key,
        "--seed",
        str(args.seed),
        "--steps",
        str(direct_budget),
        "--replan-steps",
        str(replan_steps),
        "--report-schema",
        "v2",
        "--target-object",
        str(execution["target_object"]),
        "--external-server",
        "--expected-policy-identity",
        str(identity),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--server-timeout",
        str(args.server_timeout),
        "--oracle-target-visual-prompt",
        style,
        "--oracle-minimum-visible-pixels",
        str(execution["target_presence_minimum_pixels"]),
        "--oracle-target-allow-absent-until-visible",
        "--assets",
        str(args.output_dir / "assets"),
        "--work",
        str(args.output_dir / "work"),
        "--output",
        str(report),
        "--final-state",
        str(final_state),
    ]
    identity_check = [
        sys.executable,
        str(ROOT / "scripts/infra/check_external_pi05.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--timeout",
        str(args.server_timeout),
        "--identity",
        str(identity),
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": "piu.b7-oracle-binding-plan.v1",
                    "method_id": "B7",
                    "selected_style": style,
                    "style_selection_sha256": sha256(args.style_selection),
                    "external_pi05": f"{args.host}:{args.port}",
                    "local_pi05_loaded": False,
                    "identity_check": identity_check,
                    "command": command,
                },
                indent=2,
            )
        )
        return
    if args.output_dir.exists():
        raise FileExistsError("B7 run directories are immutable")
    subprocess.run(identity_check, cwd=ROOT, check=True)
    subprocess.run(command, cwd=ROOT, check=True)
    value = json.loads(report.read_text())
    controller = value.get("controller", {})
    if (
        value.get("schema_version") != "piu.semantic-option.v2"
        or value.get("claim_scope") != "EVALUATOR_ONLY_ORACLE_UPPER_BOUND"
        or len(controller.get("online_oracle_inputs", ())) != 2
    ):
        raise ValueError("B7 execution did not preserve the oracle contract")
    oracle_report = controller.get("oracle_visual_prompt", {})
    if (
        oracle_report.get("style") != style
        or oracle_report.get("allow_absent_until_visible") is not True
    ):
        raise ValueError("B7 dynamic oracle marker report differs")
    audits = oracle_report.get("policy_call_audit")
    if not isinstance(audits, list) or len(audits) != controller.get("policy_calls"):
        raise ValueError("B7 oracle policy-call audit is incomplete")
    validate_server_metadata(
        controller.get("server_metadata", {}), load_checkpoint_identity(identity)
    )
    source = controller.get("source_initial_state_transport", {})
    if source.get("sha256") != sha256(args.initial_state):
        raise ValueError("B7 execution did not start from the paired source state")
    action_path = resolve(Path(controller["action_history"]))
    actions = json.loads(action_path.read_text())
    if not isinstance(actions, list):
        raise TypeError("B7 low-level action history must be a list")
    outcomes = evaluator_outcomes(value)
    outcomes["interaction_count"] = 1
    outcomes["executed_steps"] = len(actions)
    activated = any(
        max(map(int, row["visible_pixels"].values()))
        >= int(execution["target_presence_minimum_pixels"])
        and any(int(count) > 0 for count in row["changed_pixels"].values())
        for row in audits
    )
    episode = {
        "schema_version": "piu.closed-loop-episode.v1",
        "claim_scope": "SEALED_ORACLE_UPPER_BOUND_EPISODE_NOT_PUBLIC_METHOD",
        "method_id": "B7",
        "initial_state_group": group,
        "split": args.split,
        "evidence_class": "oracle_upper_bound",
        "rollout_status": "COMPLETE" if outcomes["task_success"] else "FAILED",
        "source_state": {
            "path": portable(args.initial_state),
            "sha256": sha256(args.initial_state),
        },
        "policy_identity": {
            "path": portable(identity),
            "sha256": sha256(identity),
        },
        "public_action_history": {
            "path": portable(action_path),
            "sha256": sha256(action_path),
        },
        "outcomes": outcomes,
        "online_oracle_inputs": ["target_instance_mask", "target_identity"],
        "oracle_diagnostics": {
            "selected_style": style,
            "intervention_activated_at_least_once": activated,
            "policy_calls": len(audits),
            "paper_method_claim_allowed": False,
        },
        "inputs": {
            "execution_report": {"path": portable(report), "sha256": sha256(report)},
            "style_selection": {
                "path": portable(args.style_selection),
                "sha256": sha256(args.style_selection),
                "status": selection["status"],
            },
            "oracle_config": {
                "path": portable(args.oracle_config),
                "sha256": sha256(args.oracle_config),
            },
        },
    }
    episode_path = args.output_dir / "episode.json"
    episode_path.write_text(json.dumps(episode, indent=2, allow_nan=False) + "\n")
    print(json.dumps(episode, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
