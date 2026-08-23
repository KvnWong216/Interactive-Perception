#!/usr/bin/env python3
"""Replay a hash-bound B6 counterfactual tree as an oracle upper-bound episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.action_effect import load_effect_labels
from piu.contracts import load_public_transitions, public_observation_sha256
from piu.formal_attempt import artifact, validate_attempt_ticket
from piu.executor_bridge import SpatialReference, serialize_pi05_subtask
from piu.oracle_effect import decide_oracle_effect
from piu.policy_identity import load_checkpoint_identity, validate_server_metadata
from piu.primitive_registry import (
    load_primitive_qualification_certificate,
    validate_qualification_candidate_contract,
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


def verified(value: dict[str, Any], *, name: str) -> Path:
    path = resolve(Path(value["path"]))
    if not path.is_file() or sha256(path) != value.get("sha256"):
        raise ValueError(f"oracle-effect {name} differs from its content hash")
    return path


def one_transition(path: Path, sample_id: str):
    rows = [row for row in load_public_transitions(path) if row.sample_id == sample_id]
    if len(rows) != 1:
        raise ValueError("oracle-effect node must select one public transition")
    return rows[0]


def verify_authorization(path: Path, *, trace: Path, output_dir: Path) -> None:
    value = json.loads(path.read_text())
    if value.get("schema_version") != "piu.oracle-effect-trace-authorization.v1":
        raise ValueError("unsupported oracle-effect trace authorization")
    expected = {
        "trace_manifest_sha256": sha256(trace),
        "method_id": "B6",
        "single_use_output_dir": portable(output_dir),
    }
    for name, required in expected.items():
        if value.get(name) != required:
            raise ValueError(f"oracle-effect trace authorization differs at {name}")


def evaluator_outcomes(reports: list[dict[str, Any]]) -> dict[str, Any]:
    evaluators = [report["evaluator"] for report in reports]
    targets = {str(row.get("target_object")) for row in evaluators}
    if len(targets) > 1:
        raise ValueError("oracle-effect evaluator target changed along the trace")
    target = next(iter(targets), None)
    final = evaluators[-1] if evaluators else {}
    return {
        "target_grasp_contact": any(
            row.get("target_grasp_contact_success") is True for row in evaluators
        ),
        "wrong_object_grasp_contact": any(
            int(values.get("grasp_contact_steps", 0)) > 0
            for row in evaluators
            for name, values in row.get("objects", {}).items()
            if name != target
        ),
        "target_destination_final": bool(
            final.get("target_in_destination_final", False)
        ),
        "task_success": bool(final.get("task_success_final", False)),
        "abstention": False,
        "target_maximum_lift_m": max(
            (float(row.get("target_maximum_lift_m") or 0.0) for row in evaluators),
            default=0.0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument(
        "--baseline-registry",
        type=Path,
        default=ROOT / "configs/experiments/piu_baselines_v1.yaml",
    )
    parser.add_argument(
        "--scenario-config",
        type=Path,
        default=ROOT / "configs/scenarios/original_drawer.yaml",
    )
    parser.add_argument("--sealed-authorization", type=Path)
    parser.add_argument("--formal-attempt-ticket", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    trace_path = resolve(args.trace_manifest)
    registry_path = resolve(args.baseline_registry)
    scenario_path = resolve(args.scenario_config)
    output_dir = resolve(args.output_dir)
    authorization_path = (
        None
        if args.sealed_authorization is None
        else resolve(args.sealed_authorization)
    )
    ticket_path = (
        None
        if args.formal_attempt_ticket is None
        else resolve(args.formal_attempt_ticket)
    )
    trace = json.loads(trace_path.read_text())
    registry = yaml.safe_load(registry_path.read_text())
    if registry.get("schema_version") != "piu.baseline-registry.v1":
        raise ValueError("unsupported oracle-effect baseline registry")
    if not scenario_path.is_file():
        raise FileNotFoundError(scenario_path)
    maximum_decisions = registry["shared_contract"].get(
        "maximum_controller_decisions"
    )
    if (
        not isinstance(maximum_decisions, int)
        or isinstance(maximum_decisions, bool)
        or maximum_decisions <= 0
    ):
        raise ValueError("oracle-effect registry has an invalid decision cap")
    identity_path = resolve(
        Path(registry["shared_contract"]["checkpoint_identity"])
    )
    identity = load_checkpoint_identity(identity_path)
    if trace.get("schema_version") != "piu.oracle-effect-trace-manifest.v1":
        raise ValueError("unsupported oracle-effect trace manifest")
    if trace.get("method_id") != "B6":
        raise ValueError("oracle-effect trace method must be B6")
    split = str(trace.get("split", ""))
    if split not in {"development", "sealed_test"}:
        raise ValueError("oracle-effect trace split is unsupported")
    if split == "sealed_test":
        if authorization_path is None:
            raise ValueError("sealed oracle-effect trace requires authorization")
        if ticket_path is None:
            raise ValueError("sealed oracle-effect trace requires its ordered ticket")
        verify_authorization(
            authorization_path, trace=trace_path, output_dir=output_dir
        )
    elif authorization_path is not None or ticket_path is not None:
        raise ValueError(
            "development oracle-effect trace cannot use sealed authorization/ticket"
        )
    group = " ".join(str(trace.get("initial_state_group", "")).split())
    if not group:
        raise ValueError("oracle-effect trace group is required")
    simulator_seed = trace.get("simulator_seed")
    if not isinstance(simulator_seed, int) or isinstance(simulator_seed, bool):
        raise TypeError("oracle-effect trace requires an integer simulator seed")
    source_state = verified(dict(trace.get("source_state", {})), name="source state")
    attempt = None
    if ticket_path is not None:
        validate_attempt_ticket(
            ticket_path,
            repository_root=ROOT,
            method_id="B6",
            initial_state_group=group,
            simulator_seed=simulator_seed,
            source_state=source_state,
            output_dir=output_dir,
            baseline_registry=registry_path,
            scenario_config=scenario_path,
        )
        attempt = artifact(ticket_path, repository_root=ROOT)
    expected_state_sha256 = sha256(source_state)
    nodes = trace.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("oracle-effect trace requires an ordered nonempty node list")
    if len(nodes) > maximum_decisions:
        raise ValueError("oracle-effect trace exceeds the shared decision cap")
    decisions = []
    reports = []
    histories = []
    expected_observation_sha256: str | None = None
    status = "TIMEOUT"
    for step, specification in enumerate(nodes):
        if not isinstance(specification, dict):
            raise TypeError("oracle-effect node specification must be an object")
        transition_path = verified(
            dict(specification.get("decision_transition", {})),
            name=f"node {step} decision transition",
        )
        labels_path = verified(
            dict(specification.get("effect_labels", {})),
            name=f"node {step} effect labels",
        )
        branches_path = verified(
            dict(specification.get("branch_manifest", {})),
            name=f"node {step} branch manifest",
        )
        sample_id = " ".join(str(specification.get("decision_sample_id", "")).split())
        decision_state = one_transition(transition_path, sample_id)
        if (
            decision_state.initial_state_group != group
            or decision_state.split.value != split
        ):
            raise ValueError("oracle-effect node group/split differs from trace")
        decision_digest = public_observation_sha256(
            decision_state.observations["post_interaction"]
        )
        if (
            expected_observation_sha256 is not None
            and decision_digest != expected_observation_sha256
        ):
            raise ValueError("oracle-effect selected branch does not reach next node")
        labels = [
            row for row in load_effect_labels(labels_path) if row.sample_id == sample_id
        ]
        oracle = decide_oracle_effect(labels, decision_state.candidate_actions)
        branch_manifest = json.loads(branches_path.read_text())
        if (
            branch_manifest.get("schema_version")
            != "piu.counterfactual-branch-manifest.v1"
            or branch_manifest.get("decision_sample_id") != sample_id
            or branch_manifest.get("initial_state_group") != group
            or branch_manifest.get("split") != split
            or branch_manifest.get("decision_observation_sha256") != decision_digest
        ):
            raise ValueError("oracle-effect branch manifest differs from decision node")
        branch_source = verified(
            dict(branch_manifest.get("source_state", {})),
            name=f"node {step} source state",
        )
        if sha256(branch_source) != expected_state_sha256:
            raise ValueError("oracle-effect physical state chain is broken")
        branch_rows = [
            row
            for row in branch_manifest.get("branches", ())
            if row.get("candidate_id") == oracle.candidate_id
        ]
        if len(branch_rows) != 1:
            raise ValueError("oracle-effect selected branch is absent or duplicated")
        branch = branch_rows[0]
        if branch.get("eligible_for_execution") is not True:
            raise ValueError("oracle-effect selected branch is context-ineligible")
        decisions.append(
            {
                "step": step,
                "sample_id": sample_id,
                "decision_kind": oracle.kind.value,
                "candidate_id": oracle.candidate_id,
                "primitive": oracle.primitive,
                "decision_observation_sha256": decision_digest,
                "effect_labels": {
                    "path": portable(labels_path),
                    "sha256": sha256(labels_path),
                },
            }
        )
        if oracle.kind.value in {"STOP", "REPORT_NOT_FOUND"}:
            if branch.get("outcome_transition") is not None:
                raise ValueError("oracle-effect terminal branch must be exact null")
            if step != len(nodes) - 1:
                raise ValueError("oracle-effect trace contains nodes after termination")
            status = "COMPLETE"
            break
        report_path = verified(
            dict(branch.get("execution_report", {})),
            name=f"node {step} execution report",
        )
        final_state = verified(
            dict(branch.get("final_state", {})), name=f"node {step} final state"
        )
        qualification = verified(
            dict(branch.get("qualification", {})),
            name=f"node {step} qualification",
        )
        certificate = load_primitive_qualification_certificate(
            qualification, repository_root=ROOT
        )
        execution_plan_path = verified(
            dict(branch_manifest.get("execution_plan", {})),
            name=f"node {step} execution plan",
        )
        execution_plan = json.loads(execution_plan_path.read_text())
        planned_rows = [
            row
            for row in execution_plan.get("candidates", ())
            if row.get("candidate_id") == oracle.candidate_id
        ]
        candidate_rows = [
            row
            for row in decision_state.candidate_actions
            if row.get("candidate_id") == oracle.candidate_id
        ]
        if len(planned_rows) != 1 or len(candidate_rows) != 1:
            raise ValueError("oracle-effect qualification candidate is ambiguous")
        reference_rows = planned_rows[0].get("spatial_references", ())
        references = tuple(
            SpatialReference(
                camera=str(row["camera"]),
                selected_patch_indices=tuple(row["selected_patch_indices"]),
                x_interval=tuple(row["x_interval"]),
                y_interval=tuple(row["y_interval"]),
            )
            for row in reference_rows
        )
        validate_qualification_candidate_contract(
            certificate,
            candidate=candidate_rows[0],
            spatial_reference_mode=(
                "calibrated_current_frame_boxes" if references else "none"
            ),
        )
        if (
            certificate.get("status") != "FORMALLY_QUALIFIED"
            or certificate.get("paper_method_action_authorized") is not True
            or certificate.get("candidate_id") != oracle.candidate_id
            or str(certificate.get("primitive", "")).upper() != oracle.primitive
        ):
            raise ValueError("oracle-effect selected branch is not formally qualified")
        report = json.loads(report_path.read_text())
        if report.get("schema_version") != "piu.semantic-option.v2":
            raise ValueError("oracle-effect branch lacks metric contract v2")
        if report.get("controller", {}).get("online_oracle_inputs") != []:
            raise ValueError("oracle-effect branch execution consumed oracle inputs")
        expected_subtask = serialize_pi05_subtask(
            candidate_rows[0],
            spatial_references=(
                references if oracle.primitive in {"PICK", "DIRECT"} else ()
            ),
        )
        if (
            planned_rows[0].get("structured_pi05_subtask") != expected_subtask
            or report.get("prompt") != expected_subtask
        ):
            raise ValueError("oracle-effect execution differs from qualified subtask")
        controller = report["controller"]
        reported_identity = controller.get("expected_policy_identity", {})
        if (
            reported_identity.get("sha256") != sha256(identity_path)
        ):
            raise ValueError("oracle-effect branch used another frozen policy")
        validate_server_metadata(controller.get("server_metadata", {}), identity)
        action_path = resolve(Path(report["controller"]["action_history"]))
        if not action_path.is_file():
            raise FileNotFoundError(action_path)
        outcome_path = verified(
            dict(branch.get("outcome_transition", {})),
            name=f"node {step} outcome transition",
        )
        outcome_sample = str(branch.get("outcome_sample_id", ""))
        outcome = one_transition(outcome_path, outcome_sample)
        if public_observation_sha256(
            outcome.observations["pre_interaction"]
        ) != decision_digest:
            raise ValueError("oracle-effect branch did not fork from decision RGB")
        expected_observation_sha256 = public_observation_sha256(
            outcome.observations["post_interaction"]
        )
        expected_state_sha256 = sha256(final_state)
        reports.append(report)
        histories.append(
            {
                "step": step,
                "candidate_id": oracle.candidate_id,
                "primitive": oracle.primitive,
                "action_history": {
                    "path": portable(action_path),
                    "sha256": sha256(action_path),
                    "steps": len(json.loads(action_path.read_text())),
                },
                "execution_report": {
                    "path": portable(report_path),
                    "sha256": sha256(report_path),
                },
            }
        )
    outcomes = evaluator_outcomes(reports)
    outcomes["interaction_count"] = len(reports)
    outcomes["executed_steps"] = sum(
        row["action_history"]["steps"] for row in histories
    )
    history = {
        "schema_version": "piu.oracle-effect-public-action-history.v1",
        "method_id": "B6",
        "initial_state_group": group,
        "oracle_decisions": decisions,
        "selected_public_executions": histories,
        "terminal_status": status,
    }
    if output_dir.exists():
        raise FileExistsError("oracle-effect trace outputs are immutable")
    output_dir.mkdir(parents=True)
    history_path = output_dir / "public_action_history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n")
    episode = {
        "schema_version": "piu.closed-loop-episode.v1",
        "claim_scope": "EVALUATOR_ONLY_ORACLE_UPPER_BOUND_EPISODE",
        "method_id": "B6",
        "initial_state_group": group,
        "simulator_seed": simulator_seed,
        "split": split,
        "evidence_class": "oracle_upper_bound",
        "rollout_status": status,
        "source_state": {
            "path": portable(source_state),
            "sha256": sha256(source_state),
        },
        "policy_identity": {
            "path": portable(identity_path),
            "sha256": sha256(identity_path),
        },
        "public_action_history": {
            "path": portable(history_path),
            "sha256": sha256(history_path),
        },
        "outcomes": outcomes,
        "online_oracle_inputs": ["executed_candidate_effect_labels"],
        "formal_attempt_ticket": attempt,
        "inputs": {
            "trace_manifest": {
                "path": portable(trace_path),
                "sha256": sha256(trace_path),
            }
        },
        "eligible_for_main_method_comparison": False,
        "paper_method_claim_allowed": False,
    }
    episode_path = output_dir / "episode.json"
    episode_path.write_text(json.dumps(episode, indent=2, allow_nan=False) + "\n")
    print(json.dumps(episode, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
