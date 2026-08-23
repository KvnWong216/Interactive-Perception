#!/usr/bin/env python3
"""Project the frozen one-step Heuristic V0 into one fair B2 episode."""

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

from piu.contracts import load_public_transitions, public_observation_sha256
from piu.policy_identity import load_checkpoint_identity, validate_server_metadata

FROZEN_COMMIT = "e7db12b7f35d9be416fc3ed57d36b12560e40cf0"

ACTION_CONTRACTS = {
    "ACT": ("DIRECT_ACT", "DIRECT", "ACT"),
    "OPEN_CONTAINER": ("OPEN_CONTAINER", "OPEN", "OBSERVE"),
    "MOVE_CLOSER": ("MOVE_CLOSER", None, "OBSERVE"),
    "NEXT_BEST_VIEW": ("NEXT_BEST_VIEW", None, "OBSERVE"),
    "REMOVE_OCCLUDER": ("REMOVE_OCCLUDER", None, "OBSERVE"),
    "STOP": ("STOP", None, "STOP"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def verified_artifact(value: dict[str, Any], *, name: str) -> Path:
    path = resolve(Path(value["path"]))
    if not path.is_file() or sha256(path) != value.get("sha256"):
        raise ValueError(f"B2 {name} differs from its content hash")
    return path


def evaluator_outcomes(report: dict[str, Any]) -> dict[str, bool | float | int]:
    evaluator = report["evaluator"]
    target = str(evaluator.get("target_object"))
    return {
        "target_grasp_contact": bool(
            evaluator.get("target_grasp_contact_success", False)
        ),
        "wrong_object_grasp_contact": any(
            name != target and int(row.get("grasp_contact_steps", 0)) > 0
            for name, row in evaluator.get("objects", {}).items()
        ),
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
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--scenario-config", type=Path, required=True)
    parser.add_argument(
        "--baseline-registry",
        type=Path,
        default=ROOT / "configs/experiments/piu_baselines_v1.yaml",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--server-timeout", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in ("attestation", "scenario_config", "baseline_registry", "output_dir"):
        setattr(args, name, resolve(getattr(args, name)))
    if args.server_timeout <= 0:
        raise ValueError("server-timeout must be positive")
    attestation = json.loads(args.attestation.read_text())
    if (
        attestation.get("schema_version")
        != "piu.heuristic-v0-inference-attestation.v1"
        or attestation.get("method_id") != "B2"
        or attestation.get("frozen_commit") != FROZEN_COMMIT
        or attestation.get("one_step_baseline") is not True
        or attestation.get("online_oracle_inputs") != []
        or attestation.get("paper_method_claim_allowed") is not False
    ):
        raise ValueError("unsupported frozen Heuristic V0 attestation")
    identities = attestation.get("model_identities")
    if not isinstance(identities, dict) or set(identities) != {
        "grounding_dino",
        "sam",
        "dinov2",
        "siglip",
        "qwen",
    }:
        raise ValueError("B2 attestation lacks every frozen model identity")
    if any(
        row.get("schema_version") != "piu.checkpoint-tree-sha256.v1"
        or len(str(row.get("sha256", ""))) != 64
        for row in identities.values()
    ):
        raise ValueError("B2 model identity is malformed")
    capture_path = verified_artifact(
        attestation["inputs"]["capture_report"], name="capture report"
    )
    capture = json.loads(capture_path.read_text())
    transition_path = verified_artifact(
        attestation["inputs"]["public_transition"], name="public transition"
    )
    if capture.get("public_transition", {}).get("sha256") != sha256(transition_path):
        raise ValueError("B2 attestation and capture transition differ")
    transitions = load_public_transitions(transition_path)
    if len(transitions) != 1:
        raise ValueError("B2 requires one captured public transition")
    transition = transitions[0]
    if transition.split.value not in {"development", "sealed_test"}:
        raise ValueError("B2 episodes are development or sealed-test only")
    decision_digest = public_observation_sha256(
        transition.observations["post_interaction"]
    )
    if attestation.get("decision_observation_sha256") != decision_digest:
        raise ValueError("B2 inference used another public decision observation")
    source_path = verified_artifact(attestation["source_state"], name="source state")
    capture_source = capture.get("source_state", {})
    if capture_source.get("sha256") != sha256(source_path):
        raise ValueError("B2 inference source state differs from initial capture")
    report_path = verified_artifact(
        attestation["inference_report"], name="inference report"
    )
    inference = json.loads(report_path.read_text())
    if (
        inference.get("schema_version")
        != "interaction-uncertainty.qwen-observation-pipeline.v0"
        or inference.get("prompt") != transition.prompt
        or inference.get("online_oracle_inputs") != []
    ):
        raise ValueError("B2 inference report violates the frozen public contract")
    selected = inference.get("selected_action", {})
    action = str(selected.get("action", ""))
    if action not in ACTION_CONTRACTS:
        raise ValueError("B2 inference selected an unknown frozen action")
    expected_primitive, shared_primitive, expected_mode = ACTION_CONTRACTS[action]
    contract = inference.get("execution_contract", {})
    if (
        contract.get("primitive") != expected_primitive
        or contract.get("mode") != expected_mode
        or contract.get("target_id") != selected.get("target_id")
    ):
        raise ValueError("B2 selected action and frozen execution contract differ")
    registry = yaml.safe_load(args.baseline_registry.read_text())
    if registry.get("schema_version") != "piu.baseline-registry.v1":
        raise ValueError("unsupported PIU baseline registry")
    shared = registry["shared_contract"]
    identity = resolve(Path(shared["checkpoint_identity"]))
    physical = shared_primitive is not None
    command = None
    identity_check = None
    execution_report = args.output_dir / "report.json"
    final_state = args.output_dir / "final_state.npz"
    if physical:
        subtask = contract.get("subtask")
        if not isinstance(subtask, str) or not subtask.strip():
            raise ValueError("B2 physical action lacks its frozen semantic subtask")
        budget = int(shared["option_step_budgets"][shared_primitive])
        command = [
            sys.executable,
            str(ROOT / "scripts/pipeline/execute.py"),
            "--scenario-config",
            str(args.scenario_config),
            "--role",
            f"B2_{expected_primitive}",
            "--prompt",
            subtask,
            "--initial-state",
            str(source_path),
            "--state-key",
            str(attestation["source_state"]["state_key"]),
            "--seed",
            str(args.seed),
            "--steps",
            str(budget),
            "--replan-steps",
            str(shared["replanning_steps"]),
            "--report-schema",
            "v2",
            "--external-server",
            "--expected-policy-identity",
            str(identity),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--server-timeout",
            str(args.server_timeout),
            "--assets",
            str(args.output_dir / "assets"),
            "--work",
            str(args.output_dir / "work"),
            "--output",
            str(execution_report),
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
                    "schema_version": "piu.heuristic-v0-once-plan.v1",
                    "method_id": "B2",
                    "frozen_commit": FROZEN_COMMIT,
                    "selected_action": action,
                    "physical_action_supported_by_shared_contract": physical,
                    "identity_check": identity_check,
                    "command": command,
                    "one_step_baseline": True,
                    "local_models_loaded": False,
                },
                indent=2,
            )
        )
        return
    if args.output_dir.exists():
        raise FileExistsError("B2 episode directories are immutable")
    args.output_dir.mkdir(parents=True)
    if physical:
        subprocess.run(identity_check, cwd=ROOT, check=True)
        subprocess.run(command, cwd=ROOT, check=True)
        execution = json.loads(execution_report.read_text())
        if (
            execution.get("schema_version") != "piu.semantic-option.v2"
            or execution.get("controller", {}).get("online_oracle_inputs") != []
        ):
            raise ValueError("B2 physical execution violates the public contract")
        validate_server_metadata(
            execution["controller"].get("server_metadata", {}),
            load_checkpoint_identity(identity),
        )
        action_path = resolve(Path(execution["controller"]["action_history"]))
        actions = json.loads(action_path.read_text())
        outcomes = evaluator_outcomes(execution)
        outcomes["interaction_count"] = 1
        outcomes["executed_steps"] = len(actions)
        status = "COMPLETE" if outcomes["task_success"] else "FAILED"
        execution_input = {
            "path": portable(execution_report),
            "sha256": sha256(execution_report),
        }
    else:
        action_path = args.output_dir / "public_action_history.json"
        action_path.write_text("[]\n")
        outcomes = {
            "target_grasp_contact": False,
            "wrong_object_grasp_contact": False,
            "target_destination_final": False,
            "task_success": False,
            "abstention": True,
            "target_maximum_lift_m": 0.0,
            "interaction_count": 0,
            "executed_steps": 0,
        }
        status = "ABSTAINED"
        execution_input = None
    episode = {
        "schema_version": "piu.closed-loop-episode.v1",
        "claim_scope": "FROZEN_ONE_STEP_ENGINEERING_BASELINE_EPISODE",
        "method_id": "B2",
        "initial_state_group": transition.initial_state_group,
        "split": transition.split.value,
        "evidence_class": "public_method",
        "rollout_status": status,
        "source_state": {"path": portable(source_path), "sha256": sha256(source_path)},
        "policy_identity": {"path": portable(identity), "sha256": sha256(identity)},
        "public_action_history": {
            "path": portable(action_path),
            "sha256": sha256(action_path),
        },
        "outcomes": outcomes,
        "online_oracle_inputs": [],
        "inputs": {
            "inference_attestation": {
                "path": portable(args.attestation),
                "sha256": sha256(args.attestation),
            },
            "inference_report": {
                "path": portable(report_path),
                "sha256": sha256(report_path),
            },
            "execution_report": execution_input,
        },
        "baseline_limitations": [
            "the frozen tag performs one decision only",
            "no post-action replanning is added by the comparison adapter",
            "unsupported frozen observation primitives count as abstention",
        ],
    }
    episode_path = args.output_dir / "episode.json"
    episode_path.write_text(json.dumps(episode, indent=2, allow_nan=False) + "\n")
    print(json.dumps(episode, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
