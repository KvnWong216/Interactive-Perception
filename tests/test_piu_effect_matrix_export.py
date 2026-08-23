from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from piu.contracts import public_observation_sha256

ROOT = Path(__file__).resolve().parents[1]


def _transition(*, sample: str, pre: dict, post: dict, initial: bool) -> dict:
    candidates = [
        {"candidate_id": "open_drawer", "primitive": "OPEN"},
        {"candidate_id": "place", "primitive": "PLACE"},
        {"candidate_id": "stop", "primitive": "STOP"},
    ]
    return {
        "schema_version": "piu.public-transition.v1",
        "sample_id": sample,
        "initial_state_group": "group",
        "split": "train",
        "prompt": "pick butter",
        "observations": {"pre_interaction": pre, "post_interaction": post},
        "public_action_history": (
            {"initial_observation": True, "last_executed_candidate": None}
            if initial
            else {
                "last_executed_candidate": {
                    "candidate_id": "open_drawer",
                    "primitive": "OPEN",
                }
            }
        ),
        "candidate_actions": candidates,
        "online_oracle_inputs": [],
    }


def test_effect_matrix_export_enforces_decision_state_fork(tmp_path: Path) -> None:
    before = {
        "images": {"agentview": {"sha256": "a" * 64}},
        "public_robot_state": [0.0],
    }
    after = {
        "images": {"agentview": {"sha256": "b" * 64}},
        "public_robot_state": [1.0],
    }
    decision_path = tmp_path / "decision.jsonl"
    decision_path.write_text(
        json.dumps(
            _transition(sample="decision", pre=before, post=before, initial=True)
        )
        + "\n"
    )
    outcome_path = tmp_path / "outcome.jsonl"
    outcome_path.write_text(
        json.dumps(_transition(sample="outcome", pre=before, post=after, initial=False))
        + "\n"
    )
    decision_digest = public_observation_sha256(before)
    branch_path = tmp_path / "branches.json"
    branch_path.write_text(
        json.dumps(
            {
                "schema_version": "piu.counterfactual-branch-manifest.v1",
                "decision_sample_id": "decision",
                "decision_observation_sha256": decision_digest,
                "branches": [
                    {
                        "candidate_id": "open_drawer",
                        "primitive": "OPEN",
                        "eligible_for_execution": True,
                        "outcome_sample_id": "outcome",
                        "outcome_transition": {
                            "path": str(outcome_path),
                            "sha256": hashlib.sha256(
                                outcome_path.read_bytes()
                            ).hexdigest(),
                        },
                    },
                    {
                        "candidate_id": "place",
                        "primitive": "PLACE",
                        "eligible_for_execution": False,
                        "eligibility_reason": "not holding requested target",
                        "outcome_transition": None,
                    },
                    {
                        "candidate_id": "stop",
                        "primitive": "STOP",
                        "eligible_for_execution": True,
                        "outcome_transition": None,
                    },
                ],
            }
        )
    )
    false_factors = {
        "execution_succeeded": None,
        "task_progress_succeeded": False,
        "task_relevant_change": False,
        "target_revealed": False,
        "identity_resolved_post": None,
        "candidate_rejected": False,
        "region_confirmed_empty": False,
        "task_information_sufficient_post": None,
    }
    annotation_path = tmp_path / "annotation.json"
    annotation_path.write_text(
        json.dumps(
            {
                "schema_version": "piu.counterfactual-effect-annotation.v1",
                "decision_sample_id": "decision",
                "decision_observation_sha256": decision_digest,
                "annotator_blinded_to_model": True,
                "annotation_protocol": "frozen evaluator protocol v1",
                "candidates": [
                    {
                        "candidate_id": "open_drawer",
                        "primitive": "OPEN",
                        "selection_correct": False,
                        "factors": {
                            **false_factors,
                            "execution_succeeded": True,
                            "task_relevant_change": True,
                            "target_revealed": True,
                        },
                    },
                    {
                        "candidate_id": "place",
                        "primitive": "PLACE",
                        "selection_correct": True,
                        "factors": {name: None for name in false_factors},
                    },
                    {
                        "candidate_id": "stop",
                        "primitive": "STOP",
                        "selection_correct": False,
                        "factors": false_factors,
                    },
                ],
            }
        )
    )
    output = tmp_path / "labels.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/data/export_piu_action_effect_matrix.py"),
            "--decision-transition",
            str(decision_path),
            "--decision-sample-id",
            "decision",
            "--branch-manifest",
            str(branch_path),
            "--annotation",
            str(annotation_path),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    labels = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(labels) == 3
    by_id = {row["candidate_id"]: row for row in labels}
    assert by_id["open_drawer"]["outcome_observation_sha256"]["pre"] == decision_digest
    assert by_id["place"]["eligible_for_execution"] is False
    assert by_id["place"]["selection_correct"] is True
    assert by_id["place"]["executed"] is False
    assert by_id["place"]["exact_null_transition"] is False
    assert all(value is None for value in by_id["place"]["factors"].values())
    assert by_id["stop"]["exact_null_transition"] is True
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["exact_candidate_matrix"] is True
