#!/usr/bin/env python3
"""Run one explicit B6 evaluator-only oracle-effect decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.action_effect import load_effect_labels
from piu.contracts import load_public_transitions
from piu.executor_bridge import serialize_pi05_subtask
from piu.oracle_effect import decide_oracle_effect


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def verify_sealed(path: Path, *, public: Path, labels: Path, output: Path) -> None:
    value = json.loads(path.read_text())
    if value.get("schema_version") != "piu.oracle-effect-sealed-authorization.v1":
        raise ValueError("unsupported oracle-effect sealed authorization")
    expected = {
        "public_transition_sha256": sha256(public),
        "effect_label_sha256": sha256(labels),
        "method_id": "B6",
        "single_use_output": portable(output),
    }
    for name, required in expected.items():
        if value.get(name) != required:
            raise ValueError(f"oracle-effect authorization differs at {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-transition", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--effect-labels", type=Path, required=True)
    parser.add_argument(
        "--expected-split", choices=("development", "sealed_test"), required=True
    )
    parser.add_argument("--sealed-authorization", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "public_transition",
        "effect_labels",
        "sealed_authorization",
        "output",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, resolve(value))
    if args.output.exists():
        raise FileExistsError("oracle-effect controller reports are immutable")
    if args.expected_split == "sealed_test":
        if args.sealed_authorization is None:
            raise ValueError("sealed oracle-effect inference requires authorization")
        verify_sealed(
            args.sealed_authorization,
            public=args.public_transition,
            labels=args.effect_labels,
            output=args.output,
        )
    elif args.sealed_authorization is not None:
        raise ValueError(
            "development oracle-effect inference cannot use sealed authorization"
        )
    rows = [
        row
        for row in load_public_transitions(args.public_transition)
        if row.sample_id == args.sample_id
    ]
    if len(rows) != 1:
        raise ValueError("sample ID must select one public transition")
    public = rows[0]
    if public.split.value != args.expected_split:
        raise ValueError("oracle-effect public split differs")
    labels = [
        label
        for label in load_effect_labels(args.effect_labels)
        if label.sample_id == args.sample_id
    ]
    decision = decide_oracle_effect(labels, public.candidate_actions)
    selected = next(
        row
        for row in public.candidate_actions
        if str(row["candidate_id"]) == decision.candidate_id
    )
    structured = serialize_pi05_subtask(selected, spatial_references=())
    result = {
        "schema_version": "piu.oracle-effect-controller-report.v1",
        "claim_scope": "EVALUATOR_ONLY_ORACLE_UPPER_BOUND",
        "method_id": "B6",
        "split": public.split.value,
        "initial_state_groups": [public.initial_state_group],
        "evaluator_labels_loaded": True,
        "online_oracle_inputs": ["executed_candidate_effect_labels"],
        "decisions": [
            {
                "sample_id": public.sample_id,
                "initial_state_group": public.initial_state_group,
                "decision_kind": decision.kind.value,
                "selected_candidate_id": decision.candidate_id,
                "selected_candidate_primitive": decision.primitive,
                "selected_candidate": dict(selected),
                "public_candidates": [dict(row) for row in public.candidate_actions],
                "structured_pi05_subtask": structured,
                "reason": "unique evaluator-correct branch in exact effect matrix",
            }
        ],
        "inputs": {
            "public_transition": {
                "path": portable(args.public_transition),
                "sha256": sha256(args.public_transition),
            },
            "effect_labels": {
                "path": portable(args.effect_labels),
                "sha256": sha256(args.effect_labels),
            },
        },
        "eligible_for_main_method_comparison": False,
        "paper_method_claim_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
