from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def test_closed_loop_episode_requires_and_aggregates_a_state_hash_chain(
    tmp_path: Path,
) -> None:
    initial = tmp_path / "initial.npz"
    final = tmp_path / "final.npz"
    initial.write_bytes(b"initial")
    final.write_bytes(b"final")
    actions = tmp_path / "actions.json"
    actions.write_text("[[0,0,0,0,0,0,0],[1,0,0,0,0,0,0]]\n")
    controller = tmp_path / "controller.json"
    controller.write_text("{}\n")
    identity_path = ROOT / "results/diagnostics/pi05_libero_checkpoint_identity_v1.json"
    identity = json.loads(identity_path.read_text())
    execution = tmp_path / "execution.json"
    execution.write_text(
        json.dumps(
            {
                "schema_version": "piu.semantic-option.v2",
                "controller": {
                    "online_oracle_inputs": [],
                    "source_initial_state_transport": _artifact(initial),
                    "action_history": str(actions),
                    "expected_policy_identity": _artifact(identity_path),
                    "server_metadata": {
                        "schema_version": "piu.identified-pi05-server.v1",
                        "policy_config": identity["policy_config"],
                        "environment": "LIBERO",
                        "checkpoint": identity["checkpoint"],
                        "capabilities": ["action_chunks", "spatial_prefix_v1"],
                    },
                },
                "evaluator": {
                    "target_object": "butter_1",
                    "target_grasp_contact_success": True,
                    "target_maximum_lift_m": 0.04,
                    "target_in_destination_final": True,
                    "task_success_final": True,
                    "objects": {
                        "butter_1": {"grasp_contact_steps": 1},
                        "cream_cheese_1": {"grasp_contact_steps": 0},
                    },
                },
            }
        )
    )
    physical = tmp_path / "physical_receipt.json"
    physical.write_text(
        json.dumps(
            {
                "schema_version": "piu.executor-dispatch.v1",
                "method_id": "B8",
                "initial_state_group": "group",
                "decision_kind": "EXECUTE",
                "physical_action_dispatched": True,
                "candidate_id": "pick_butter",
                "primitive": "PICK",
                "controller_report": _artifact(controller),
                "source_initial_state_transport": _artifact(initial),
                "final_state_transport": _artifact(final),
                "execution_report": _artifact(execution),
                "evaluator_fields_copied": [],
            }
        )
    )
    terminal = tmp_path / "terminal_receipt.json"
    terminal.write_text(
        json.dumps(
            {
                "schema_version": "piu.executor-dispatch.v1",
                "method_id": "B8",
                "initial_state_group": "group",
                "decision_kind": "STOP",
                "physical_action_dispatched": False,
                "controller_report": _artifact(controller),
                "evaluator_fields_copied": [],
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "piu.closed-loop-run-manifest.v1",
                "method_id": "B8",
                "initial_state_group": "group",
                "split": "sealed_test",
                "rollout_status": "COMPLETE",
                "source_state": _artifact(initial),
                "dispatch_receipts": [_artifact(physical), _artifact(terminal)],
            }
        )
    )
    output = tmp_path / "episode.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/aggregate_piu_closed_loop_episode.py"),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["outcomes"]["target_grasp_contact"] is True
    assert report["outcomes"]["target_destination_final"] is True
    assert report["outcomes"]["interaction_count"] == 1
    assert report["outcomes"]["executed_steps"] == 2
    assert report["online_oracle_inputs"] == []
    assert report["policy_identity"]["sha256"] == _sha256(identity_path)
