from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_initial_label_export_dry_run_verifies_hashed_state_pair(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state.npz"
    state.write_bytes(b"opaque-test-state")
    state_hash = hashlib.sha256(state.read_bytes()).hexdigest()
    capture = tmp_path / "capture.json"
    capture.write_text(
        json.dumps(
            {
                "schema_version": "piu.initial-observation-capture.v1",
                "sample_id": "initial",
                "source_state": {
                    "path": str(state),
                    "sha256": state_hash,
                    "state_key": "state",
                },
            }
        )
    )
    observation = {
        "images": {"agentview": {"sha256": "a" * 64}},
        "public_robot_state": [0.0],
    }
    public = tmp_path / "public.jsonl"
    public.write_text(
        json.dumps(
            {
                "schema_version": "piu.public-transition.v1",
                "sample_id": "initial",
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
                    {"candidate_id": "open_drawer", "primitive": "OPEN"}
                ],
                "online_oracle_inputs": [],
            }
        )
        + "\n"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/data/export_piu_post_observation_labels.py"),
            "--public-transition",
            str(public),
            "--sample-id",
            "initial",
            "--scenario-config",
            str(ROOT / "configs/scenarios/original_drawer.yaml"),
            "--capture-report",
            str(capture),
            "--output",
            str(tmp_path / "labels.jsonl"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)
    assert plan["state_pair_verified"] is True
    assert plan["executed_action"] == "INITIAL_OBSERVATION"
    assert plan["task_sufficiency_annotation"] is False
    assert plan["external_simulator_required"] is True
