from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from piu.action_effect import EffectLabel
from piu.contracts import public_observation_sha256
from piu.oracle_effect import decide_oracle_effect

ROOT = Path(__file__).resolve().parents[1]


def _label(
    candidate: str,
    primitive: str,
    *,
    correct: bool,
    sample: str = "sample",
    group: str = "group",
    split: str = "development",
    decision_digest: str = "a" * 64,
    outcome_digest: str = "a" * 64,
) -> dict:
    terminal = primitive in {"STOP", "REPORT_NOT_FOUND"}
    return {
        "schema_version": "piu.action-effect-label.v1",
        "sample_id": sample,
        "initial_state_group": group,
        "split": split,
        "candidate_id": candidate,
        "candidate_primitive": primitive,
        "decision_observation_sha256": decision_digest,
        "outcome_observation_sha256": {
            "pre": decision_digest,
            "post": decision_digest if terminal else outcome_digest,
        },
        "selection_correct": correct,
        "executed": not terminal,
        "exact_null_transition": terminal,
        "factors": {
            "execution_succeeded": None if terminal else True,
            "task_progress_succeeded": False,
            "task_relevant_change": not terminal,
            "target_revealed": False,
            "identity_resolved_post": None,
            "candidate_rejected": False,
            "region_confirmed_empty": False,
            "task_information_sufficient_post": None,
        },
        "simulator_teacher_only": True,
    }


def test_oracle_effect_is_explicitly_separate_from_public_controller(
    tmp_path: Path,
) -> None:
    candidates = [
        {
            "candidate_id": "open_drawer",
            "primitive": "OPEN",
            "target": "middle drawer",
        },
        {"candidate_id": "stop", "primitive": "STOP", "target": "task"},
    ]
    labels = [
        EffectLabel.from_mapping(_label("open_drawer", "OPEN", correct=True)),
        EffectLabel.from_mapping(_label("stop", "STOP", correct=False)),
    ]
    decision = decide_oracle_effect(labels, candidates)
    assert decision.candidate_id == "open_drawer"

    observation = {
        "images": {"agentview": {"sha256": "a" * 64}},
        "public_robot_state": [0.0],
    }
    public = tmp_path / "public.jsonl"
    public.write_text(
        json.dumps(
            {
                "schema_version": "piu.public-transition.v1",
                "sample_id": "sample",
                "initial_state_group": "group",
                "split": "development",
                "prompt": "pick butter",
                "observations": {
                    "pre_interaction": observation,
                    "post_interaction": observation,
                },
                "public_action_history": {
                    "initial_observation": True,
                    "last_executed_candidate": None,
                },
                "candidate_actions": candidates,
                "online_oracle_inputs": [],
            }
        )
        + "\n"
    )
    label_path = tmp_path / "labels.jsonl"
    label_path.write_text(
        "".join(
            json.dumps(value) + "\n"
            for value in (
                _label("open_drawer", "OPEN", correct=True),
                _label("stop", "STOP", correct=False),
            )
        )
    )
    output = tmp_path / "controller.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/pipeline/run_piu_oracle_effect_controller.py"),
            "--public-transition",
            str(public),
            "--sample-id",
            "sample",
            "--effect-labels",
            str(label_path),
            "--expected-split",
            "development",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["method_id"] == "B6"
    assert report["evaluator_labels_loaded"] is True
    assert report["online_oracle_inputs"] == ["executed_candidate_effect_labels"]
    assert report["eligible_for_main_method_comparison"] is False


def _artifact(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _public_row(
    *, sample: str, observation: dict, candidates: list[dict], initial: bool
) -> dict:
    return {
        "schema_version": "piu.public-transition.v1",
        "sample_id": sample,
        "initial_state_group": "group",
        "split": "development",
        "prompt": "pick butter",
        "observations": {
            "pre_interaction": observation,
            "post_interaction": observation,
        },
        "public_action_history": {
            "initial_observation": initial,
            "last_executed_candidate": (
                None if initial else candidates[0]
            ),
        },
        "candidate_actions": candidates,
        "online_oracle_inputs": [],
    }


def test_oracle_effect_trace_follows_only_the_selected_real_branch(
    tmp_path: Path,
) -> None:
    candidates = [
        {
            "candidate_id": "open_drawer",
            "primitive": "OPEN",
            "target": "middle drawer",
        },
        {"candidate_id": "stop", "primitive": "STOP", "target": "task"},
    ]
    observation_a = {
        "images": {"agentview": {"sha256": "a" * 64}},
        "public_robot_state": [0.0],
    }
    observation_b = {
        "images": {"agentview": {"sha256": "b" * 64}},
        "public_robot_state": [1.0],
    }
    digest_a = public_observation_sha256(observation_a)
    digest_b = public_observation_sha256(observation_b)
    state_a = tmp_path / "state_a.npz"
    state_b = tmp_path / "state_b.npz"
    state_a.write_bytes(b"opaque-a")
    state_b.write_bytes(b"opaque-b")
    decision_a = tmp_path / "decision_a.jsonl"
    decision_b = tmp_path / "decision_b.jsonl"
    outcome_a = tmp_path / "outcome_a.jsonl"
    decision_a.write_text(
        json.dumps(
            _public_row(
                sample="node-a",
                observation=observation_a,
                candidates=candidates,
                initial=True,
            )
        )
        + "\n"
    )
    outcome_row = _public_row(
        sample="outcome-a",
        observation=observation_b,
        candidates=candidates,
        initial=False,
    )
    outcome_row["observations"]["pre_interaction"] = observation_a
    outcome_a.write_text(json.dumps(outcome_row) + "\n")
    decision_b.write_text(
        json.dumps(
            _public_row(
                sample="node-b",
                observation=observation_b,
                candidates=candidates,
                initial=False,
            )
        )
        + "\n"
    )
    labels_a = tmp_path / "labels_a.jsonl"
    labels_b = tmp_path / "labels_b.jsonl"
    labels_a.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                _label(
                    "open_drawer",
                    "OPEN",
                    correct=True,
                    sample="node-a",
                    decision_digest=digest_a,
                    outcome_digest=digest_b,
                ),
                _label(
                    "stop",
                    "STOP",
                    correct=False,
                    sample="node-a",
                    decision_digest=digest_a,
                ),
            )
        )
    )
    labels_b.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                _label(
                    "open_drawer",
                    "OPEN",
                    correct=False,
                    sample="node-b",
                    decision_digest=digest_b,
                    outcome_digest="c" * 64,
                ),
                _label(
                    "stop",
                    "STOP",
                    correct=True,
                    sample="node-b",
                    decision_digest=digest_b,
                ),
            )
        )
    )
    actions = tmp_path / "actions.json"
    actions.write_text("[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]\n")
    identity_path = ROOT / "results/diagnostics/pi05_libero_checkpoint_identity_v1.json"
    identity = json.loads(identity_path.read_text())
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "piu.semantic-option.v2",
                "controller": {
                    "online_oracle_inputs": [],
                    "action_history": str(actions),
                    "expected_policy_identity": {
                        "path": str(identity_path),
                        "sha256": hashlib.sha256(
                            identity_path.read_bytes()
                        ).hexdigest(),
                    },
                    "server_metadata": {
                        "schema_version": "piu.identified-pi05-server.v1",
                        "policy_config": identity["policy_config"],
                        "environment": "LIBERO",
                        "checkpoint": identity["checkpoint"],
                    },
                },
                "evaluator": {
                    "target_object": "butter_1",
                    "target_grasp_contact_success": True,
                    "target_in_destination_final": True,
                    "task_success_final": True,
                    "target_maximum_lift_m": 0.12,
                    "objects": {
                        "butter_1": {"grasp_contact_steps": 2},
                        "cream_cheese_1": {"grasp_contact_steps": 0},
                    },
                },
            }
        )
    )
    certificate = tmp_path / "certificate.json"
    certificate.write_text(
        json.dumps(
            {
                "schema_version": "piu.primitive-qualification-certificate.v1",
                "status": "FORMALLY_QUALIFIED",
                "paper_method_action_authorized": True,
                "candidate_id": "open_drawer",
                "primitive": "OPEN",
            }
        )
    )
    branch_a = tmp_path / "branches_a.json"
    branch_b = tmp_path / "branches_b.json"
    branch_a.write_text(
        json.dumps(
            {
                "schema_version": "piu.counterfactual-branch-manifest.v1",
                "decision_sample_id": "node-a",
                "initial_state_group": "group",
                "split": "development",
                "decision_observation_sha256": digest_a,
                "source_state": _artifact(state_a),
                "branches": [
                    {
                        "candidate_id": "open_drawer",
                        "primitive": "OPEN",
                        "eligible_for_execution": True,
                        "outcome_sample_id": "outcome-a",
                        "outcome_transition": _artifact(outcome_a),
                        "execution_report": _artifact(report),
                        "final_state": _artifact(state_b),
                        "qualification": _artifact(certificate),
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
    branch_b.write_text(
        json.dumps(
            {
                "schema_version": "piu.counterfactual-branch-manifest.v1",
                "decision_sample_id": "node-b",
                "initial_state_group": "group",
                "split": "development",
                "decision_observation_sha256": digest_b,
                "source_state": _artifact(state_b),
                "branches": [
                    {
                        "candidate_id": "open_drawer",
                        "primitive": "OPEN",
                        "eligible_for_execution": True,
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
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "schema_version": "piu.oracle-effect-trace-manifest.v1",
                "method_id": "B6",
                "split": "development",
                "initial_state_group": "group",
                "simulator_seed": 17,
                "source_state": _artifact(state_a),
                "nodes": [
                    {
                        "decision_transition": _artifact(decision_a),
                        "decision_sample_id": "node-a",
                        "effect_labels": _artifact(labels_a),
                        "branch_manifest": _artifact(branch_a),
                    },
                    {
                        "decision_transition": _artifact(decision_b),
                        "decision_sample_id": "node-b",
                        "effect_labels": _artifact(labels_b),
                        "branch_manifest": _artifact(branch_b),
                    },
                ],
            }
        )
    )
    output_dir = tmp_path / "oracle_trace"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/pipeline/replay_piu_oracle_effect_trace.py"),
            "--trace-manifest",
            str(trace),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    episode = json.loads((output_dir / "episode.json").read_text())
    assert episode["rollout_status"] == "COMPLETE"
    assert episode["outcomes"]["interaction_count"] == 1
    assert episode["outcomes"]["task_success"] is True
    assert episode["online_oracle_inputs"] == [
        "executed_candidate_effect_labels"
    ]

    state_c = tmp_path / "state_c.npz"
    state_c.write_bytes(b"unrelated-opaque-state")
    broken_branch = json.loads(branch_b.read_text())
    broken_branch["source_state"] = _artifact(state_c)
    branch_b.write_text(json.dumps(broken_branch))
    broken_trace = json.loads(trace.read_text())
    broken_trace["nodes"][1]["branch_manifest"] = _artifact(branch_b)
    trace.write_text(json.dumps(broken_trace))
    with pytest.raises(subprocess.CalledProcessError) as error:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/pipeline/replay_piu_oracle_effect_trace.py"),
                "--trace-manifest",
                str(trace),
                "--output-dir",
                str(tmp_path / "broken_trace"),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    assert "physical state chain is broken" in error.value.stderr
