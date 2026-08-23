from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def test_closed_loop_episode_requires_and_aggregates_a_state_hash_chain(
    tmp_path: Path,
) -> None:
    initial = tmp_path / "initial.npz"
    execution_initial = tmp_path / "execution_initial.npz"
    final = tmp_path / "final.npz"
    np.savez_compressed(initial, state=np.asarray([1.0], dtype=np.float64))
    np.savez_compressed(
        execution_initial, state=np.asarray([1.0], dtype=np.float64)
    )
    np.savez_compressed(final, state=np.asarray([2.0], dtype=np.float64))
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
                    "source_initial_state_transport": _artifact(execution_initial),
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
                "source_initial_state_transport": _artifact(execution_initial),
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
    repro_lock = ROOT / "results/diagnostics/piu_offline_repro_preflight_v1.json"
    baseline_registry = ROOT / "configs/experiments/piu_baselines_v1.yaml"
    scenario_config = ROOT / "configs/scenarios/original_drawer.yaml"
    schedule = tmp_path / "schedule.json"
    schedule.write_text(
        json.dumps(
            {
                "schema_version": "piu.formal-execution-schedule.v1",
                "status": "FROZEN_BEFORE_FORMAL_OUTCOME_COLLECTION",
                "outcomes_loaded": False,
                "inputs": {
                    "offline_repro_lock": {
                        **_artifact(repro_lock),
                        "manifest_sha256": _sha256(
                            ROOT / "configs/experiments/piu_offline_repro_v1.yaml"
                        ),
                    },
                    "policy_identity": _artifact(identity_path),
                    "baseline_registry": _artifact(baseline_registry),
                    "scenario_config": _artifact(scenario_config),
                },
                "shared_execution_contract": {
                    "maximum_controller_decisions": 8,
                    "interpretation": (
                        "resource_cap_not_learned_decision_threshold"
                    ),
                },
                "entries": [
                    {
                        "execution_index": 0,
                        "initial_state_group": "group",
                        "simulator_seed": 17,
                        "method_id": "B8",
                        "state_key": "state",
                        "source_state": _artifact(initial),
                    }
                ],
            }
        )
    )
    run_dir = tmp_path / "sealed_run"
    ledger = tmp_path / "ledger"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/begin_piu_formal_attempt.py"),
            "--schedule",
            str(schedule),
            "--ledger-dir",
            str(ledger),
            "--run-output-dir",
            str(run_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    ticket = ledger / "00000.started.json"
    premature = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/begin_piu_formal_attempt.py"),
            "--schedule",
            str(schedule),
            "--ledger-dir",
            str(ledger),
            "--run-output-dir",
            str(tmp_path / "another_run"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert premature.returncode != 0
    assert "still open" in premature.stderr
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "piu.closed-loop-run-manifest.v1",
                "method_id": "B8",
                "initial_state_group": "group",
                "simulator_seed": 17,
                "split": "sealed_test",
                "rollout_status": "COMPLETE",
                "maximum_decisions": 8,
                "baseline_registry": _artifact(baseline_registry),
                "scenario_config": _artifact(scenario_config),
                "source_state": {**_artifact(initial), "state_key": "state"},
                "execution_initial_state": {
                    **_artifact(execution_initial),
                    "state_key": "state",
                },
                "dispatch_receipts": [_artifact(physical), _artifact(terminal)],
                "formal_attempt_ticket": _artifact(ticket),
            }
        )
    )
    output = run_dir / "episode.json"
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
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/close_piu_formal_attempt.py"),
            "--ticket",
            str(ticket),
            "--episode",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    development_manifest = json.loads(manifest.read_text())
    development_manifest["split"] = "development"
    development_manifest["formal_attempt_ticket"] = None
    development_path = tmp_path / "development_manifest.json"
    development_path.write_text(json.dumps(development_manifest))
    development_output = tmp_path / "development_run/episode.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/aggregate_piu_closed_loop_episode.py"),
            "--manifest",
            str(development_path),
            "--output",
            str(development_output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    development = json.loads(development_output.read_text())
    assert development["split"] == "development"
    assert development["simulator_seed"] == 17
    assert development["claim_scope"] == (
        "DEVELOPMENT_PUBLIC_METHOD_EPISODE_NOT_FORMAL_EVIDENCE"
    )
