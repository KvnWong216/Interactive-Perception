#!/usr/bin/env python3
"""Aggregate a hash-chained PIU dispatch sequence into one closed-loop episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.formal_attempt import artifact, validate_attempt_ticket
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


def verified_artifact(value: dict[str, Any], *, name: str) -> Path:
    path = resolve(Path(value["path"]))
    if sha256(path) != value.get("sha256"):
        raise ValueError(f"closed-loop {name} differs from its content hash")
    return path


def same_opaque_state(
    left_path: Path,
    left_key: str,
    right_path: Path,
    right_key: str,
) -> bool:
    with np.load(left_path, allow_pickle=False) as left_store:
        left = np.asarray(left_store[left_key])
    with np.load(right_path, allow_pickle=False) as right_store:
        right = np.asarray(right_store[right_key])
    return left.shape == right.shape and np.array_equal(left, right)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--baseline-registry",
        type=Path,
        default=ROOT / "configs/experiments/piu_baselines_v1.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = resolve(args.manifest)
    registry_path = resolve(args.baseline_registry)
    output = resolve(args.output)
    history_output = output.with_suffix(".public_action_history.json")
    if output.exists() or history_output.exists():
        raise FileExistsError("closed-loop episode outputs are immutable")
    manifest = json.loads(manifest_path.read_text())
    registry = yaml.safe_load(registry_path.read_text())
    if registry.get("schema_version") != "piu.baseline-registry.v1":
        raise ValueError("unsupported closed-loop baseline registry")
    maximum_decisions = registry["shared_contract"].get(
        "maximum_controller_decisions"
    )
    if (
        not isinstance(maximum_decisions, int)
        or isinstance(maximum_decisions, bool)
        or maximum_decisions <= 0
    ):
        raise ValueError("closed-loop registry has an invalid decision cap")
    identity_path = resolve(
        Path(registry["shared_contract"]["checkpoint_identity"])
    )
    identity = load_checkpoint_identity(identity_path)
    if manifest.get("schema_version") != "piu.closed-loop-run-manifest.v1":
        raise ValueError("unsupported closed-loop run manifest")
    if manifest.get("maximum_decisions") != maximum_decisions:
        raise ValueError("closed-loop manifest differs from the shared decision cap")
    manifest_registry = verified_artifact(
        dict(manifest.get("baseline_registry", {})), name="baseline registry"
    )
    if sha256(manifest_registry) != sha256(registry_path):
        raise ValueError("closed-loop manifest used another baseline registry")
    scenario_path = verified_artifact(
        dict(manifest.get("scenario_config", {})), name="scenario config"
    )
    method_id = str(manifest.get("method_id", ""))
    group = " ".join(str(manifest.get("initial_state_group", "")).split())
    if method_id not in {"B1", "B3", "B4", "B5", "B8"} or not group:
        raise ValueError(
            "PIU closed-loop aggregator accepts public B1/B3/B4/B5/B8 runs"
        )
    split = str(manifest.get("split", ""))
    if split not in {"development", "sealed_test"}:
        raise ValueError("closed-loop aggregation requires development or sealed_test")
    simulator_seed = manifest.get("simulator_seed")
    if not isinstance(simulator_seed, int) or isinstance(simulator_seed, bool):
        raise TypeError("closed-loop manifest requires an integer simulator seed")
    source_state = dict(manifest.get("source_state", {}))
    source_path = verified_artifact(source_state, name="source state")
    execution_initial = dict(manifest.get("execution_initial_state", {}))
    execution_initial_path = verified_artifact(
        execution_initial, name="execution initial state"
    )
    if not same_opaque_state(
        source_path,
        str(source_state.get("state_key", "state")),
        execution_initial_path,
        str(execution_initial.get("state_key", "state")),
    ):
        raise ValueError("capture working state differs from scheduled source state")
    attempt = None
    attempt_value = manifest.get("formal_attempt_ticket")
    if split == "sealed_test":
        if not isinstance(attempt_value, dict):
            raise TypeError(
                "sealed closed-loop manifest lacks its formal attempt ticket"
            )
        ticket_path = verified_artifact(attempt_value, name="formal attempt ticket")
        validate_attempt_ticket(
            ticket_path,
            repository_root=ROOT,
            method_id=method_id,
            initial_state_group=group,
            simulator_seed=simulator_seed,
            source_state=source_path,
            output_dir=output.parent,
            baseline_registry=manifest_registry,
            scenario_config=scenario_path,
        )
        attempt = artifact(ticket_path, repository_root=ROOT)
    elif attempt_value is not None:
        raise ValueError("development episode cannot consume a formal attempt ticket")
    expected_state_hash = sha256(execution_initial_path)
    receipt_specs = manifest.get("dispatch_receipts")
    if not isinstance(receipt_specs, list) or not receipt_specs:
        raise ValueError("closed-loop manifest requires a nonempty dispatch sequence")
    if len(receipt_specs) > maximum_decisions:
        raise ValueError(
            "closed-loop dispatch sequence exceeds the shared decision cap"
        )
    physical_reports = []
    history_rows = []
    decision_kinds = []
    for index, spec in enumerate(receipt_specs):
        receipt_path = verified_artifact(spec, name=f"receipt {index}")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("schema_version") != "piu.executor-dispatch.v1":
            raise ValueError("closed-loop sequence contains a non-PIU receipt")
        if (
            receipt.get("method_id") != method_id
            or receipt.get("initial_state_group") != group
        ):
            raise ValueError("closed-loop receipt method/group differs")
        if receipt.get("evaluator_fields_copied") != []:
            raise ValueError("closed-loop receipt crossed the evaluator firewall")
        verified_artifact(receipt["controller_report"], name="controller report")
        kind = str(receipt.get("decision_kind", ""))
        decision_kinds.append(kind)
        if receipt.get("physical_action_dispatched") is not True:
            if index != len(receipt_specs) - 1:
                raise ValueError(
                    "non-physical controller decision must terminate the run"
                )
            continue
        initial = receipt.get("source_initial_state_transport")
        final = receipt.get("final_state_transport")
        if not isinstance(initial, dict) or not isinstance(final, dict):
            raise TypeError("physical receipt lacks state-chain provenance")
        initial_path = verified_artifact(initial, name="dispatch initial state")
        if sha256(initial_path) != expected_state_hash:
            raise ValueError(
                "dispatch sequence does not preserve the prior physical state"
            )
        final_path = verified_artifact(final, name="dispatch final state")
        expected_state_hash = sha256(final_path)
        execution_path = verified_artifact(
            receipt["execution_report"], name="execution report"
        )
        execution = json.loads(execution_path.read_text())
        if execution.get("schema_version") != "piu.semantic-option.v2":
            raise ValueError(
                "closed-loop physical execution must use metric contract v2"
            )
        if execution.get("controller", {}).get("online_oracle_inputs") != []:
            raise ValueError("public closed-loop execution consumed oracle inputs")
        expected_identity = execution["controller"].get(
            "expected_policy_identity", {}
        )
        if expected_identity.get("sha256") != sha256(identity_path):
            raise ValueError("closed-loop execution used another identity artifact")
        validate_server_metadata(
            execution["controller"].get("server_metadata", {}), identity
        )
        report_initial = execution["controller"].get("source_initial_state_transport")
        if not isinstance(report_initial, dict) or report_initial.get(
            "sha256"
        ) != sha256(initial_path):
            raise ValueError(
                "execution report and dispatch receipt start states differ"
            )
        action_path = resolve(Path(execution["controller"]["action_history"]))
        actions = json.loads(action_path.read_text())
        if not isinstance(actions, list):
            raise TypeError("public low-level action history must be a list")
        history_rows.append(
            {
                "dispatch_index": index,
                "decision_kind": kind,
                "candidate_id": receipt.get("candidate_id"),
                "primitive": receipt.get("primitive"),
                "action_history": {
                    "path": portable(action_path),
                    "sha256": sha256(action_path),
                    "steps": len(actions),
                },
            }
        )
        physical_reports.append(execution)
    status = str(manifest.get("rollout_status", ""))
    if status not in {"COMPLETE", "FAILED", "TIMEOUT", "ABSTAINED"}:
        raise ValueError("closed-loop rollout status is unsupported")
    if status == "ABSTAINED" and decision_kinds[-1] != "ABSTAIN":
        raise ValueError("ABSTAINED run must end in an ABSTAIN receipt")
    if status == "COMPLETE" and decision_kinds[-1] not in {
        "STOP",
        "REPORT_NOT_FOUND",
    }:
        raise ValueError("COMPLETE run needs an explicit terminal controller decision")
    evaluators = [report["evaluator"] for report in physical_reports]
    target_names = {str(evaluator.get("target_object")) for evaluator in evaluators}
    if len(target_names) > 1:
        raise ValueError("closed-loop evaluator target identity changed across steps")
    target_name = next(iter(target_names), None)
    target_contact = any(
        evaluator.get("target_grasp_contact_success") is True
        for evaluator in evaluators
    )
    wrong_contact = any(
        int(values.get("grasp_contact_steps", 0)) > 0
        for evaluator in evaluators
        for name, values in evaluator.get("objects", {}).items()
        if name != target_name
    )
    maximum_lift = max(
        (
            float(evaluator.get("target_maximum_lift_m") or 0.0)
            for evaluator in evaluators
        ),
        default=0.0,
    )
    final_evaluator = evaluators[-1] if evaluators else {}
    history = {
        "schema_version": "piu.closed-loop-public-action-history.v1",
        "method_id": method_id,
        "initial_state_group": group,
        "simulator_seed": simulator_seed,
        "dispatches": history_rows,
        "terminal_decision_kind": decision_kinds[-1],
    }
    history_output.parent.mkdir(parents=True, exist_ok=True)
    history_output.write_text(json.dumps(history, indent=2) + "\n")
    episode = {
        "schema_version": "piu.closed-loop-episode.v1",
        "claim_scope": (
            "SEALED_PUBLIC_METHOD_EPISODE_NOT_AGGREGATE_RESULT"
            if split == "sealed_test"
            else "DEVELOPMENT_PUBLIC_METHOD_EPISODE_NOT_FORMAL_EVIDENCE"
        ),
        "method_id": method_id,
        "initial_state_group": group,
        "simulator_seed": simulator_seed,
        "split": split,
        "evidence_class": "public_method",
        "rollout_status": status,
        "source_state": {
            "path": portable(source_path),
            "sha256": sha256(source_path),
        },
        "policy_identity": {
            "path": portable(identity_path),
            "sha256": sha256(identity_path),
        },
        "public_action_history": {
            "path": portable(history_output),
            "sha256": sha256(history_output),
        },
        "outcomes": {
            "target_grasp_contact": target_contact,
            "wrong_object_grasp_contact": wrong_contact,
            "target_destination_final": bool(
                final_evaluator.get("target_in_destination_final", False)
            ),
            "task_success": bool(final_evaluator.get("task_success_final", False)),
            "abstention": status == "ABSTAINED",
            "target_maximum_lift_m": maximum_lift,
            "interaction_count": len(physical_reports),
            "executed_steps": sum(
                row["action_history"]["steps"] for row in history_rows
            ),
        },
        "online_oracle_inputs": [],
        "formal_attempt_ticket": attempt,
        "inputs": {
            "manifest": {
                "path": portable(manifest_path),
                "sha256": sha256(manifest_path),
            },
            "dispatch_receipts": [
                {"path": portable(resolve(Path(row["path"]))), "sha256": row["sha256"]}
                for row in receipt_specs
            ],
        },
    }
    output.write_text(json.dumps(episode, indent=2, allow_nan=False) + "\n")
    print(json.dumps(episode, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
