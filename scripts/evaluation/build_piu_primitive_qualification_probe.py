#!/usr/bin/env python3
"""Freeze one public, model-free OPEN stimulus for executor qualification."""

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

from piu.contracts import assert_public_policy_value
from piu.executor_bridge import serialize_pi05_subtask
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


def reference(path: Path) -> dict[str, str]:
    return {
        "path": portable(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidate-set", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--initial-state-group", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan_path = resolve(args.plan)
    candidate_set_path = resolve(args.candidate_set)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("primitive qualification probes are immutable")
    plan = load_primitive_qualification_plan(plan_path, repository_root=ROOT)
    primitive = str(plan["primitive"]).upper()
    if primitive != "OPEN":
        raise ValueError(
            "model-free qualification probes are OPEN-only; PICK/DIRECT require "
            "calibrated current-frame spatial references"
        )
    sample_id = " ".join(args.sample_id.split())
    group = " ".join(args.initial_state_group.split())
    if not sample_id or not group:
        raise ValueError("qualification probe sample and group must be nonempty")
    rows = [
        json.loads(line)
        for line in candidate_set_path.read_text().splitlines()
        if line
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError("qualification probe candidate-set rows must be objects")
    matches = [
        row
        for row in rows
        if row.get("schema_version") == "piu.public-candidate-set.v1"
        and row.get("sample_id") == sample_id
        and row.get("initial_state_group") == group
    ]
    if len(matches) != 1:
        raise ValueError("qualification probe requires one public candidate-set row")
    source = matches[0]
    if (
        source.get("public_inputs_only") is not True
        or source.get("online_oracle_inputs") != []
    ):
        raise ValueError("qualification probe candidate set is not public-only")
    candidates = source.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise TypeError("qualification probe requires a nonempty candidate list")
    assert_public_policy_value(candidates, path="qualification_probe.candidates")
    selected = [
        row
        for row in candidates
        if isinstance(row, dict)
        and row.get("candidate_id") == plan["candidate_id"]
        and str(row.get("primitive", "")).upper() == primitive
    ]
    if len(selected) != 1:
        raise ValueError("frozen plan candidate is absent or duplicated")
    candidate = selected[0]
    subtask = serialize_pi05_subtask(candidate, spatial_references=())
    if not subtask:
        raise ValueError("OPEN qualification probe lacks a serialized subtask")
    result = {
        "schema_version": "piu.primitive-qualification-probe.v1",
        "status": "FROZEN_BEFORE_PRIMITIVE_QUALIFICATION_OUTCOMES",
        "claim_scope": "EXECUTOR_STIMULUS_ONLY_NOT_METHOD_SELECTION",
        "outcomes_loaded": False,
        "selection_source": "preregistered_executor_probe_not_method_decision",
        "candidate_choice_outcome_dependent": False,
        "paper_method_selection_claim_allowed": False,
        "trained_model_loaded": False,
        "calibration_loaded": False,
        "evaluator_labels_loaded": False,
        "online_oracle_inputs": [],
        "inputs": {
            "plan": reference(plan_path),
            "candidate_set": reference(candidate_set_path),
        },
        "decisions": [
            {
                "sample_id": sample_id,
                "initial_state_group": group,
                "decision_kind": "INTERACT",
                "selected_candidate_id": plan["candidate_id"],
                "selected_candidate_primitive": primitive,
                "selected_candidate": candidate,
                "public_candidates": candidates,
                "reason": "predeclared executor-qualification stimulus",
                "structured_pi05_subtask": subtask,
                "spatial_references": [],
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
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
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
            pending = Path(handle.name)
        load_qualification_controller_decision(
            pending,
            candidate_id=str(plan["candidate_id"]),
            primitive=primitive,
            initial_state_group=group,
            repository_root=ROOT,
        )
        os.link(pending, output)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
    print(json.dumps({"output": portable(output), **result}, indent=2))


if __name__ == "__main__":
    main()
