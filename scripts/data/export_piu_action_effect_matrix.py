#!/usr/bin/env python3
"""Export one causally aligned PIU candidate-effect label matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.action_effect import EFFECT_FACTORS, EffectLabel
from piu.contracts import load_public_transitions, public_observation_sha256


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def artifact(value: dict[str, Any], *, name: str) -> Path:
    path = resolve(Path(value["path"]))
    if not path.is_file() or sha256(path) != value.get("sha256"):
        raise ValueError(f"{name} differs from its content hash")
    return path


def one_transition(path: Path, *, sample_id: str):
    rows = [row for row in load_public_transitions(path) if row.sample_id == sample_id]
    if len(rows) != 1:
        raise ValueError("sample ID must select exactly one public transition")
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-transition", type=Path, required=True)
    parser.add_argument("--decision-sample-id", required=True)
    parser.add_argument("--branch-manifest", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("decision_transition", "branch_manifest", "annotation", "output"):
        setattr(args, name, resolve(getattr(args, name)))
    report_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or report_path.exists():
        raise FileExistsError("action-effect matrices are immutable")
    decision = one_transition(
        args.decision_transition, sample_id=args.decision_sample_id
    )
    decision_digest = public_observation_sha256(
        decision.observations["post_interaction"]
    )
    candidates = {
        str(row["candidate_id"]): str(row["primitive"]).upper()
        for row in decision.candidate_actions
    }
    if len(candidates) != len(decision.candidate_actions):
        raise ValueError("decision candidate IDs are not unique")

    manifest = json.loads(args.branch_manifest.read_text())
    if manifest.get("schema_version") != "piu.counterfactual-branch-manifest.v1":
        raise ValueError("unsupported counterfactual branch manifest")
    if (
        manifest.get("decision_sample_id") != decision.sample_id
        or manifest.get("decision_observation_sha256") != decision_digest
    ):
        raise ValueError("branch manifest is not bound to the decision state")
    branch_rows = manifest.get("branches")
    if not isinstance(branch_rows, list):
        raise TypeError("counterfactual branches must be a list")
    branches = {str(row.get("candidate_id")): row for row in branch_rows}
    if set(branches) != set(candidates) or len(branches) != len(branch_rows):
        raise ValueError("branch manifest must cover the exact public candidate set")

    annotation = json.loads(args.annotation.read_text())
    if annotation.get("schema_version") != "piu.counterfactual-effect-annotation.v1":
        raise ValueError("unsupported counterfactual annotation schema")
    if (
        annotation.get("decision_sample_id") != decision.sample_id
        or annotation.get("decision_observation_sha256") != decision_digest
    ):
        raise ValueError("effect annotation is not bound to the decision state")
    if (
        annotation.get("annotator_blinded_to_model") is not True
        or not str(annotation.get("annotation_protocol", "")).strip()
    ):
        raise ValueError("effect annotations require a declared blinded protocol")
    annotation_rows = annotation.get("candidates")
    if not isinstance(annotation_rows, list):
        raise TypeError("effect annotation candidates must be a list")
    annotations = {str(row.get("candidate_id")): row for row in annotation_rows}
    if set(annotations) != set(candidates) or len(annotations) != len(annotation_rows):
        raise ValueError("effect annotation must cover the exact candidate set")
    if sum(row.get("selection_correct") is True for row in annotation_rows) != 1:
        raise ValueError("effect annotation requires exactly one correct route")

    rows = []
    outcome_sources = {}
    for candidate_id, primitive in candidates.items():
        branch = branches[candidate_id]
        if str(branch.get("primitive", "")).upper() != primitive:
            raise ValueError("branch primitive differs from public candidate")
        label_source = annotations[candidate_id]
        if str(label_source.get("primitive", "")).upper() != primitive:
            raise ValueError("annotation primitive differs from public candidate")
        factors = dict(label_source.get("factors", {}))
        if set(factors) != set(EFFECT_FACTORS):
            raise ValueError("effect annotation factor family differs")
        terminal = primitive in {"STOP", "REPORT_NOT_FOUND"}
        eligible = branch.get("eligible_for_execution")
        if not isinstance(eligible, bool):
            raise TypeError("branch execution eligibility must be boolean")
        if terminal:
            if eligible is not True:
                raise ValueError("terminal exact-null candidates must be eligible")
            if branch.get("outcome_transition") is not None:
                raise ValueError("nonphysical terminal branch must be an exact null")
            outcome_hashes = {"pre": decision_digest, "post": decision_digest}
            outcome_sources[candidate_id] = None
            executed = False
            exact_null = True
        elif not eligible:
            if branch.get("outcome_transition") is not None:
                raise ValueError("ineligible physical branch cannot carry an outcome")
            if any(value is not None for value in factors.values()):
                raise ValueError("ineligible physical branch factors must all be null")
            outcome_hashes = {"pre": decision_digest, "post": decision_digest}
            outcome_sources[candidate_id] = None
            executed = False
            exact_null = False
        else:
            specification = branch.get("outcome_transition")
            if not isinstance(specification, dict):
                raise TypeError("physical candidate branch lacks an outcome transition")
            outcome_path = artifact(
                specification, name=f"{candidate_id} outcome transition"
            )
            outcome_sample = str(branch.get("outcome_sample_id", ""))
            outcome = one_transition(outcome_path, sample_id=outcome_sample)
            if (
                outcome.initial_state_group != decision.initial_state_group
                or outcome.split is not decision.split
            ):
                raise ValueError("candidate outcome group/split differs from decision")
            executed = outcome.public_action_history.get("last_executed_candidate")
            if not isinstance(executed, dict) or (
                str(executed.get("candidate_id")) != candidate_id
                or str(executed.get("primitive", "")).upper() != primitive
            ):
                raise ValueError("candidate outcome history differs from its branch")
            outcome_hashes = {
                "pre": public_observation_sha256(
                    outcome.observations["pre_interaction"]
                ),
                "post": public_observation_sha256(
                    outcome.observations["post_interaction"]
                ),
            }
            if outcome_hashes["pre"] != decision_digest:
                raise ValueError("candidate outcome does not fork from decision state")
            outcome_sources[candidate_id] = {
                "path": portable(outcome_path),
                "sha256": sha256(outcome_path),
                "sample_id": outcome_sample,
            }
            executed = True
            exact_null = False
        value = {
            "schema_version": "piu.action-effect-label.v1",
            "sample_id": decision.sample_id,
            "initial_state_group": decision.initial_state_group,
            "split": decision.split.value,
            "candidate_id": candidate_id,
            "candidate_primitive": primitive,
            "decision_observation_sha256": decision_digest,
            "outcome_observation_sha256": outcome_hashes,
            "selection_correct": label_source.get("selection_correct"),
            "eligible_for_execution": eligible,
            "executed": executed,
            "exact_null_transition": exact_null,
            "factors": factors,
            "simulator_teacher_only": True,
            "provenance": {
                "annotation": {
                    "path": portable(args.annotation),
                    "sha256": sha256(args.annotation),
                },
                "outcome_transition": outcome_sources[candidate_id],
            },
        }
        EffectLabel.from_mapping(value)
        rows.append(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    report = {
        "schema_version": "piu.action-effect-label-manifest.v1",
        "claim_scope": "EVALUATOR_ONLY_EXECUTED_COUNTERFACTUAL_SUPERVISION",
        "decision_sample_id": decision.sample_id,
        "initial_state_group": decision.initial_state_group,
        "split": decision.split.value,
        "candidates": len(rows),
        "decision_observation_sha256": decision_digest,
        "inputs": {
            "decision_transition": {
                "path": portable(args.decision_transition),
                "sha256": sha256(args.decision_transition),
            },
            "branch_manifest": {
                "path": portable(args.branch_manifest),
                "sha256": sha256(args.branch_manifest),
            },
            "annotation": {
                "path": portable(args.annotation),
                "sha256": sha256(args.annotation),
            },
            "outcome_transitions": outcome_sources,
        },
        "exact_candidate_matrix": True,
        "post_outcomes_excluded_from_decision_features": True,
        "paper_method_claim_allowed": False,
        "labels": {"path": portable(args.output), "sha256": sha256(args.output)},
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
