"""Builders for fully provenance-valid PIU test artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from piu.executor_bridge import SpatialReference, serialize_pi05_subtask
from piu.policy_identity import expected_server_metadata, load_checkpoint_identity
from piu.primitive_registry import (
    allocate_episode_primitive_risk,
    load_primitive_qualification_certificate,
    smallest_binomial_design,
)

ROOT = Path(__file__).resolve().parents[1]


def _reference(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _candidate(candidate_id: str, primitive: str) -> dict[str, str]:
    if primitive == "OPEN":
        return {
            "candidate_id": candidate_id,
            "primitive": primitive,
            "target": "middle drawer",
        }
    if primitive == "PLACE":
        return {
            "candidate_id": candidate_id,
            "primitive": primitive,
            "target": "butter",
            "reference": "basket",
        }
    return {
        "candidate_id": candidate_id,
        "primitive": primitive,
        "target": "butter",
    }


def write_formal_primitive_certificate(
    directory: Path,
    *,
    candidate_id: str,
    primitive: str,
    context: str = "synthetic_fixture_context",
) -> Path:
    """Write an exact, input-bound all-success certificate for contract tests."""

    directory.mkdir(parents=True, exist_ok=True)
    primitive = primitive.upper()
    allocation_path = ROOT / "configs/experiments/piu_executor_risk_allocation_v1.yaml"
    baseline_path = ROOT / "configs/experiments/piu_baselines_v1.yaml"
    protocol_path = ROOT / "configs/experiments/piu_primitive_registry_v2.yaml"
    identity_path = ROOT / "results/diagnostics/pi05_libero_checkpoint_identity_v1.json"
    baseline = yaml.safe_load(baseline_path.read_text())
    protocol = yaml.safe_load(protocol_path.read_text())
    budget_path = directory / f"{candidate_id}_external_budget.yaml"
    budget_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.external-execution-risk-budget.v1",
                "status": "FROZEN_BEFORE_PRIMITIVE_QUALIFICATION_OUTCOMES",
                "maximum_episode_probability_of_any_primitive_failure": 0.999,
                "design_alternative_per_dispatch_success_probability": 1.0,
                "authority": "synthetic regression fixture",
                "rationale": "exercise certificate provenance only",
                "outcomes_loaded": False,
            },
            sort_keys=False,
        )
    )
    allocation = allocate_episode_primitive_risk(
        maximum_episode_failure_probability=0.999,
        maximum_physical_dispatches=baseline["shared_contract"][
            "maximum_controller_decisions"
        ],
    )
    risk_path = directory / f"{candidate_id}_risk.json"
    risk = {
        "schema_version": "piu.primitive-risk-contract.v1",
        "status": "FROZEN_BEFORE_PRIMITIVE_QUALIFICATION_OUTCOMES",
        "claim_scope": "EXECUTOR_RELIABILITY_ONLY_NOT_TASK_SUCCESS",
        "primitive": primitive,
        "context": context,
        "candidate_id": candidate_id,
        "minimum_reliable_rate": allocation["minimum_reliable_rate"],
        "minimum_reliable_rate_provenance": (
            "derived_union_bound_from_external_episode_budget"
        ),
        "alpha": float(protocol["formal_qualification"]["alpha"]),
        "target_power": float(protocol["formal_qualification"]["target_power"]),
        "design_alternative_success_probability": 1.0,
        "design_alternative_provenance": "external_task_owner_contract",
        "retrospective_pilot_used_for_effect_size": False,
        "risk_allocation": allocation,
        "external_authority": "synthetic regression fixture",
        "external_rationale": "exercise certificate provenance only",
        "inputs": {
            "allocation_config": _reference(allocation_path),
            "baseline_registry": _reference(baseline_path),
            "primitive_registry_protocol": _reference(protocol_path),
            "external_budget": _reference(budget_path),
        },
        "outcomes_loaded": False,
        "paper_method_claim_allowed": False,
    }
    risk_path.write_text(json.dumps(risk, indent=2) + "\n")
    registry_path = ROOT / "results/method/piu_primitive_reliability_registry_v2.json"
    design = smallest_binomial_design(
        null_success_probability=allocation["minimum_reliable_rate"],
        alternative_success_probability=1.0,
        alpha=risk["alpha"],
        target_power=risk["target_power"],
        search_limit=1000,
    )
    assert design is not None
    plan_path = directory / f"{candidate_id}_plan.json"
    risk_summary = {
        **_reference(risk_path),
        "minimum_reliable_rate": allocation["minimum_reliable_rate"],
        "provenance": "derived_union_bound_from_external_episode_budget",
        "external_budget": risk["inputs"]["external_budget"],
        "risk_allocation": allocation,
    }
    plan = {
        "schema_version": "piu.primitive-qualification-plan.v1",
        "status": "PROSPECTIVE_GROUP_COUNT_FROZEN",
        "claim_scope": "DESIGN_ONLY_NO_FORMAL_OUTCOME_DATA",
        "primitive": primitive,
        "context": context,
        "candidate_id": candidate_id,
        "registry": _reference(registry_path),
        "risk_contract": risk_summary,
        "retrospective_registry_role": "diagnostic_and_seed_exclusion_only",
        "retrospective_pilot_used_for_effect_size": False,
        "alternative_success_probability": 1.0,
        "alternative_success_probability_provenance": (
            "external_task_owner_contract"
        ),
        "alpha": risk["alpha"],
        "target_power": risk["target_power"],
        "design": design,
        "search_limit": 1000,
        "test": "exact_one_sided_binomial",
    }
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")

    split_path = directory / f"{candidate_id}_split.yaml"
    roles = (
        "train",
        "development",
        "calibration_temperature",
        "calibration_conformal",
        "sealed_test",
    )
    assignments = [
        {
            "initial_state_group": f"core-{role}",
            "seed": 20000 + index,
            "split_role": role,
        }
        for index, role in enumerate(roles)
    ]
    groups = [
        f"{candidate_id}-formal-{index:03d}"
        for index in range(int(design["trials"]))
    ]
    assignments.extend(
        {
            "initial_state_group": group,
            "seed": 30000 + index,
            "split_role": "primitive_qualification",
        }
        for index, group in enumerate(groups)
    )
    split_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.group-split-manifest.v1",
                "status": "FROZEN_BEFORE_COLLECTION",
                "allocation_method": "prospective_without_outcome_access",
                "scenario": "original_cluttered_drawer",
                "assignments": assignments,
            },
            sort_keys=False,
        )
    )
    candidate = _candidate(candidate_id, primitive)
    state_paths = {}
    controller_paths = {}
    for index, group in enumerate(groups):
        state_path = directory / f"{group}.npz"
        np.savez_compressed(state_path, state=np.asarray([index, index + 0.5]))
        state_paths[group] = state_path
        references = (
            (
                SpatialReference(
                    camera="agentview",
                    selected_patch_indices=(index,),
                    x_interval=(0.1, 0.2),
                    y_interval=(0.3, 0.4),
                ),
            )
            if primitive in {"PICK", "DIRECT"}
            else ()
        )
        subtask = serialize_pi05_subtask(candidate, spatial_references=references)
        controller_path = directory / f"{group}_controller.json"
        controller_path.write_text(
            json.dumps(
                {
                    "schema_version": "piu.calibrated-controller-report.v1",
                    "evaluator_labels_loaded": False,
                    "decisions": [
                        {
                            "sample_id": group,
                            "initial_state_group": group,
                            "decision_kind": (
                                "INTERACT" if primitive == "OPEN" else "EXECUTE"
                            ),
                            "selected_candidate_id": candidate_id,
                            "selected_candidate_primitive": primitive,
                            "selected_candidate": candidate,
                            "public_candidates": [candidate],
                            "structured_pi05_subtask": subtask,
                            "spatial_references": [
                                {
                                    "camera": item.camera,
                                    "selected_patch_indices": list(
                                        item.selected_patch_indices
                                    ),
                                    "x_interval": list(item.x_interval),
                                    "y_interval": list(item.y_interval),
                                }
                                for item in references
                            ],
                        }
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        controller_paths[group] = controller_path
    run_root = directory / f"{candidate_id}_qualification_runs"
    schedule_path = directory / f"{candidate_id}_schedule.json"
    schedule_command = [
        sys.executable,
        str(ROOT / "scripts/evaluation/build_piu_primitive_qualification_schedule.py"),
        "--plan",
        str(plan_path),
        "--split-manifest",
        str(split_path),
        "--run-root",
        str(run_root),
        "--output",
        str(schedule_path),
    ]
    for group in groups:
        schedule_command.extend(["--state", group, str(state_paths[group])])
        schedule_command.extend(
            ["--controller-report", group, str(controller_paths[group])]
        )
    subprocess.run(
        schedule_command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    schedule = json.loads(schedule_path.read_text())
    identity = load_checkpoint_identity(identity_path)
    previous_receipt_sha256 = None
    for entry in schedule["entries"]:
        receipt_path = Path(entry["expected_execution_receipt"])
        if not receipt_path.is_absolute():
            receipt_path = ROOT / receipt_path
        receipt_path.parent.mkdir(parents=True)
        attempt_path = receipt_path.parent / "started.json"
        attempt_path.write_text(
            json.dumps(
                {
                    "schema_version": "piu.primitive-qualification-attempt.v1",
                    "status": "STARTED_SINGLE_USE",
                    "outcomes_loaded": False,
                    "execution_index": entry["execution_index"],
                    "entry": entry,
                    "schedule_sha256": hashlib.sha256(
                        schedule_path.read_bytes()
                    ).hexdigest(),
                    "previous_receipt_sha256": previous_receipt_sha256,
                },
                indent=2,
            )
            + "\n"
        )
        controller_path = Path(entry["controller_report"]["path"])
        if not controller_path.is_absolute():
            controller_path = ROOT / controller_path
        controller = json.loads(controller_path.read_text())["decisions"][0]
        semantic_path = receipt_path.parent / "semantic_report.json"
        evaluator = {
            "target_object": "butter_1",
            "target_grasp_contact_success": True,
            "target_in_destination_final": True,
            "task_success_final": True,
            "objects": {"butter_1": {"grasp_contact_steps": 1}},
            "joints": {
                "wooden_cabinet_1_middle_level": {"minimum": -0.15}
            },
        }
        semantic_path.write_text(
            json.dumps(
                {
                    "schema_version": "piu.semantic-option.v2",
                    "claim_scope": "PUBLIC_INPUT_EXECUTION",
                    "scenario": baseline["scenario"],
                    "seed": entry["simulator_seed"],
                    "role": f"PIU_QUALIFY_{primitive}",
                    "prompt": controller["structured_pi05_subtask"],
                    "controller": {
                        "server_mode": "external",
                        "server_metadata": expected_server_metadata(identity),
                        "online_oracle_inputs": [],
                        "subtask_steps": baseline["shared_contract"][
                            "option_step_budgets"
                        ][primitive],
                        "expected_policy_identity": _reference(identity_path),
                        "source_initial_state_transport": {
                            "path": entry["source_state"]["path"],
                            "sha256": entry["source_state"]["sha256"],
                            "state_key": entry["state_key"],
                        },
                    },
                    "evaluator": evaluator,
                },
                indent=2,
            )
            + "\n"
        )
        receipt_path.write_text(
            json.dumps(
                {
                    "schema_version": "piu.primitive-qualification-execution.v1",
                    "status": "COMPLETED",
                    "execution_index": entry["execution_index"],
                    "initial_state_group": entry["initial_state_group"],
                    "simulator_seed": entry["simulator_seed"],
                    "candidate_id": candidate_id,
                    "primitive": primitive,
                    "context": context,
                    "source_state_sha256": entry["source_state"]["sha256"],
                    "controller_report_sha256": entry["controller_report"]["sha256"],
                    "structured_subtask_sha256": entry[
                        "structured_subtask_sha256"
                    ],
                    "schedule": _reference(schedule_path),
                    "attempt": _reference(attempt_path),
                    "semantic_report": _reference(semantic_path),
                    "failure": None,
                },
                indent=2,
            )
            + "\n"
        )
        previous_receipt_sha256 = hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest()
    outcomes_path = directory / f"{candidate_id}_outcomes.jsonl"
    certificate_path = directory / f"{candidate_id}_certificate.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/evaluate_piu_primitive_qualification.py"),
            "--plan",
            str(plan_path),
            "--schedule",
            str(schedule_path),
            "--outcomes-output",
            str(outcomes_path),
            "--output",
            str(certificate_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    load_primitive_qualification_certificate(
        certificate_path, repository_root=ROOT
    )
    return certificate_path
