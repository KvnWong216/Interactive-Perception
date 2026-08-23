#!/usr/bin/env python3
"""Freeze a binder-grounded PICK/PLACE/DIRECT executor stimulus without effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.contracts import assert_public_policy_value, load_public_transitions
from piu.primitive_registry import (
    load_primitive_qualification_plan,
    load_qualification_controller_decision,
)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(path: Path) -> dict[str, str]:
    return {"path": portable(path), "sha256": sha256(path)}


def verified_reference(value: dict, *, name: str) -> Path:
    path = resolve(Path(str(value.get("path", ""))))
    if not path.is_file() or sha256(path) != value.get("sha256"):
        raise ValueError(f"{name} differs from its content hash")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan_path = resolve(args.plan)
    execution_plan_path = resolve(args.execution_plan)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("binding qualification stimuli are immutable")
    plan = load_primitive_qualification_plan(plan_path, repository_root=ROOT)
    primitive = str(plan["primitive"]).upper()
    if primitive not in {"PICK", "PLACE", "DIRECT"}:
        raise ValueError("binder-grounded qualification is for task primitives only")
    execution_plan = json.loads(execution_plan_path.read_text())
    if (
        execution_plan.get("schema_version")
        != "piu.counterfactual-execution-plan.v1"
        or execution_plan.get("split") != "primitive_qualification"
        or execution_plan.get("split_role") != "primitive_qualification"
        or execution_plan.get("public_inputs_only") is not True
        or execution_plan.get("online_oracle_inputs") != []
        or execution_plan.get("paper_method_claim_allowed") is not False
    ):
        raise ValueError("qualification execution plan is not public binder-only")
    public_path = verified_reference(
        execution_plan.get("inputs", {}).get("public_transition", {}),
        name="qualification public transition",
    )
    transitions = [
        row
        for row in load_public_transitions(public_path)
        if row.sample_id == execution_plan.get("decision_sample_id")
    ]
    if len(transitions) != 1:
        raise ValueError("qualification execution plan lacks one public transition")
    transition = transitions[0]
    if transition.initial_state_group != execution_plan.get("initial_state_group"):
        raise ValueError("qualification transition group differs")
    candidates = [dict(row) for row in transition.candidate_actions]
    assert_public_policy_value(candidates, path="binding_qualification.candidates")
    selected = [
        row
        for row in candidates
        if row.get("candidate_id") == plan["candidate_id"]
        and str(row.get("primitive", "")).upper() == primitive
    ]
    planned = [
        row
        for row in execution_plan.get("candidates", ())
        if isinstance(row, dict) and row.get("candidate_id") == plan["candidate_id"]
    ]
    if len(selected) != 1 or len(planned) != 1:
        raise ValueError("qualification candidate is absent or duplicated")
    planned_candidate = planned[0]
    references = planned_candidate.get("spatial_references")
    if (
        planned_candidate.get("eligible_for_execution") is not True
        or not planned_candidate.get("structured_pi05_subtask")
        or not isinstance(references, list)
        or (primitive in {"PICK", "DIRECT"} and not references)
        or (primitive == "PLACE" and references)
    ):
        raise ValueError("candidate lacks its calibrated binder execution preconditions")
    result = {
        "schema_version": "piu.binding-qualification-stimulus.v1",
        "status": "FROZEN_BEFORE_PRIMITIVE_QUALIFICATION_OUTCOMES",
        "claim_scope": "BINDER_GROUNDED_EXECUTOR_STIMULUS_NOT_METHOD_SELECTION",
        "outcomes_loaded": False,
        "selection_source": "preregistered_candidate_with_calibrated_binder_eligibility",
        "candidate_choice_outcome_dependent": False,
        "paper_method_selection_claim_allowed": False,
        "trained_binder_loaded": True,
        "binder_calibration_loaded": True,
        "effect_model_loaded": False,
        "evaluator_labels_loaded": False,
        "online_oracle_inputs": [],
        "inputs": {
            "plan": reference(plan_path),
            "execution_plan": reference(execution_plan_path),
        },
        "decisions": [
            {
                "sample_id": transition.sample_id,
                "initial_state_group": transition.initial_state_group,
                "decision_kind": "EXECUTE",
                "selected_candidate_id": plan["candidate_id"],
                "selected_candidate_primitive": primitive,
                "selected_candidate": selected[0],
                "public_candidates": candidates,
                "reason": "predeclared executor stimulus with calibrated binder eligibility",
                "structured_pi05_subtask": planned_candidate[
                    "structured_pi05_subtask"
                ],
                "spatial_references": references,
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    pending: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
            pending = Path(handle.name)
        load_qualification_controller_decision(
            pending,
            candidate_id=str(plan["candidate_id"]),
            primitive=primitive,
            initial_state_group=transition.initial_state_group,
            repository_root=ROOT,
        )
        os.replace(pending, output)
        pending = None
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
