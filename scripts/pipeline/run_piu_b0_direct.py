#!/usr/bin/env python3
"""Run the paired B0 frozen-pi0.5 direct baseline and export an episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


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
    parser.add_argument("--scenario-config", type=Path, required=True)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in ("scenario_config", "baseline_registry", "initial_state", "output_dir"):
        setattr(args, name, resolve(getattr(args, name)))
    if not args.initial_state.is_file():
        raise FileNotFoundError(args.initial_state)
    registry = yaml.safe_load(args.baseline_registry.read_text())
    if registry.get("schema_version") != "piu.baseline-registry.v1":
        raise ValueError("unsupported PIU baseline registry")
    direct_budget = int(registry["shared_contract"]["option_step_budgets"]["DIRECT"])
    replan_steps = int(registry["shared_contract"]["replanning_steps"])
    identity = resolve(Path(registry["shared_contract"]["checkpoint_identity"]))
    if not identity.is_file():
        raise FileNotFoundError(identity)
    report = args.output_dir / "report.json"
    final_state = args.output_dir / "final_state.npz"
    command = [
        sys.executable,
        str(ROOT / "scripts/pipeline/execute.py"),
        "--scenario-config",
        str(args.scenario_config),
        "--role",
        "DIRECT_ACT",
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
        "--external-server",
        "--expected-policy-identity",
        str(identity),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--assets",
        str(args.output_dir / "assets"),
        "--work",
        str(args.output_dir / "work"),
        "--output",
        str(report),
        "--final-state",
        str(final_state),
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": "piu.b0-direct-plan.v1",
                    "method_id": "B0",
                    "external_pi05": f"{args.host}:{args.port}",
                    "local_pi05_loaded": False,
                    "command": command,
                },
                indent=2,
            )
        )
        return
    if args.output_dir.exists():
        raise FileExistsError("B0 run directories are immutable")
    subprocess.run(command, cwd=ROOT, check=True)
    value = json.loads(report.read_text())
    if value.get("schema_version") != "piu.semantic-option.v2":
        raise ValueError("B0 execution did not use metric contract v2")
    if value.get("controller", {}).get("online_oracle_inputs") != []:
        raise ValueError("B0 execution consumed an oracle input")
    evaluator = value["evaluator"]
    target = str(evaluator.get("target_object"))
    wrong_contact = any(
        name != target and int(row.get("grasp_contact_steps", 0)) > 0
        for name, row in evaluator.get("objects", {}).items()
    )
    actions = resolve(Path(value["controller"]["action_history"]))
    if not actions.is_file():
        raise FileNotFoundError(actions)
    task_success = bool(evaluator.get("task_success_final", False))
    episode = {
        "schema_version": "piu.closed-loop-episode.v1",
        "claim_scope": "SEALED_PUBLIC_BASELINE_EPISODE_NOT_AGGREGATE_RESULT",
        "method_id": "B0",
        "initial_state_group": " ".join(args.initial_state_group.split()),
        "split": args.split,
        "evidence_class": "public_method",
        "rollout_status": "COMPLETE" if task_success else "FAILED",
        "source_state": {
            "path": portable(args.initial_state),
            "sha256": sha256(args.initial_state),
        },
        "policy_identity": {
            "path": portable(identity),
            "sha256": sha256(identity),
        },
        "public_action_history": {"path": portable(actions), "sha256": sha256(actions)},
        "outcomes": {
            "target_grasp_contact": bool(
                evaluator.get("target_grasp_contact_success", False)
            ),
            "wrong_object_grasp_contact": wrong_contact,
            "target_destination_final": bool(
                evaluator.get("target_in_destination_final", False)
            ),
            "task_success": task_success,
            "abstention": False,
            "target_maximum_lift_m": float(
                evaluator.get("target_maximum_lift_m") or 0.0
            ),
            "interaction_count": 1,
            "executed_steps": len(json.loads(actions.read_text())),
        },
        "online_oracle_inputs": [],
        "inputs": {
            "execution_report": {"path": portable(report), "sha256": sha256(report)}
        },
    }
    episode_path = args.output_dir / "episode.json"
    episode_path.write_text(json.dumps(episode, indent=2, allow_nan=False) + "\n")
    print(json.dumps(episode, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
