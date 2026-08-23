from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from piu.contracts import public_observation_sha256

ROOT = Path(__file__).resolve().parents[1]


def test_counterfactual_collection_plan_requires_qualified_physical_branches(
    tmp_path: Path,
) -> None:
    observation = {
        "images": {"agentview": {"sha256": "a" * 64}},
        "public_robot_state": [0.0],
    }
    decision = tmp_path / "decision.jsonl"
    decision.write_text(
        json.dumps(
            {
                "schema_version": "piu.public-transition.v1",
                "sample_id": "decision",
                "initial_state_group": "group",
                "split": "train",
                "prompt": "pick butter",
                "observations": {
                    "pre_interaction": observation,
                    "post_interaction": observation,
                },
                "public_action_history": {
                    "initial_observation": True,
                    "last_executed_candidate": None,
                },
                "candidate_actions": [
                    {
                        "candidate_id": "open_drawer",
                        "primitive": "OPEN",
                        "target": "middle drawer",
                    },
                    {"candidate_id": "stop", "primitive": "STOP", "target": "task"},
                ],
                "online_oracle_inputs": [],
            }
        )
        + "\n"
    )
    execution_plan = tmp_path / "execution_plan.json"
    execution_plan.write_text(
        json.dumps(
            {
                "schema_version": "piu.counterfactual-execution-plan.v1",
                "decision_sample_id": "decision",
                "initial_state_group": "group",
                "split": "train",
                "decision_observation_sha256": public_observation_sha256(
                    observation
                ),
                "public_inputs_only": True,
                "online_oracle_inputs": [],
                "inputs": {
                    "public_transition": {
                        "path": str(decision),
                        "sha256": hashlib.sha256(decision.read_bytes()).hexdigest(),
                    }
                },
                "candidates": [
                    {
                        "candidate_id": "open_drawer",
                        "primitive": "OPEN",
                        "eligible_for_execution": True,
                        "eligibility_reason": "fixture public precondition",
                        "structured_pi05_subtask": "Open the middle drawer.",
                        "spatial_references": [],
                    },
                    {
                        "candidate_id": "stop",
                        "primitive": "STOP",
                        "eligible_for_execution": True,
                        "eligibility_reason": "terminal exact null",
                        "structured_pi05_subtask": None,
                        "spatial_references": [],
                    },
                ],
            }
        )
    )
    certificate = tmp_path / "open_certificate.json"
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
    qualification_map = tmp_path / "qualification_map.json"
    qualification_map.write_text(
        json.dumps(
            {
                "schema_version": "piu.qualified-executor-map.v1",
                "candidates": {
                    "open_drawer": {
                        "path": str(certificate),
                        "sha256": hashlib.sha256(certificate.read_bytes()).hexdigest(),
                    }
                },
            }
        )
    )
    state = tmp_path / "state.npz"
    state.write_bytes(b"opaque")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/data/collect_piu_counterfactual_branches.py"),
            "--decision-transition",
            str(decision),
            "--decision-sample-id",
            "decision",
            "--execution-plan",
            str(execution_plan),
            "--scenario-config",
            str(ROOT / "configs/scenarios/original_drawer.yaml"),
            "--qualification-map",
            str(qualification_map),
            "--source-state",
            str(state),
            "--seed",
            "9",
            "--host",
            "pi05.example.internal",
            "--output-dir",
            str(tmp_path / "branches"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)
    assert [row["candidate_id"] for row in plan["physical_branches"]] == ["open_drawer"]
    assert plan["null_branches"] == ["stop"]
    assert plan["local_pi05_loaded"] is False
