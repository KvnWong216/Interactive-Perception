from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from piu.primitive_registry import (
    _reject_embedded_qualification_results,
    load_qualification_controller_decision,
)

ROOT = Path(__file__).resolve().parents[1]


def _reference(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_repository_allocation_freezes_exact_post_snapshot_block(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.json"
    split = tmp_path / "split.json"
    contexts = tmp_path / "contexts.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "scripts/evaluation/build_piu_primitive_qualification_input_allocation.py"
            ),
            "--config",
            str(
                ROOT
                / "configs/experiments/piu_open_primitive_qualification_input_allocation_v1.yaml"
            ),
            "--plan",
            str(ROOT / "results/method/piu_open_primitive_qualification_plan_v1.json"),
            "--seed-inventory-output",
            str(inventory),
            "--split-output",
            str(split),
            "--contexts-output",
            str(contexts),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    seed_audit = json.loads(inventory.read_text())
    allocation = json.loads(split.read_text())
    rows = allocation["assignments"]
    assert seed_audit["snapshot_commit"] == (
        "cd1d6ba1adade1bef8d2620b723b5ab1cf544c5f"
    )
    assert seed_audit["maximum_observed_seed"] == 26081811
    assert len(rows) == 124
    assert [row["seed"] for row in rows] == list(range(26081812, 26081936))
    assert len({row["initial_state_group"] for row in rows}) == 124
    assert allocation["allocation"]["outcomes_loaded"] is False
    assert allocation["allocation"]["rollout_executed"] is False
    assert len(contexts.read_text().splitlines()) == 124


def test_v2_open_probe_binds_one_public_reset_capture(tmp_path: Path) -> None:
    group = "qualification-group"
    sample = f"{group}::initial"
    candidate = {
        "candidate_id": "open_middle_drawer",
        "primitive": "OPEN",
        "target": "middle drawer",
        "purpose": "inspect inside",
        "required_capability": "OPEN",
    }
    candidate_set = tmp_path / "candidates.jsonl"
    candidate_set.write_text(
        json.dumps(
            {
                "schema_version": "piu.public-candidate-set.v1",
                "sample_id": sample,
                "initial_state_group": group,
                "split": "primitive_qualification",
                "public_inputs_only": True,
                "online_oracle_inputs": [],
                "candidates": [candidate],
            }
        )
        + "\n"
    )
    state = tmp_path / "state.npz"
    np.savez_compressed(state, state=np.asarray([1.0, 2.0]))
    observation = {
        "images": {"agentview": {"sha256": "a" * 64}},
        "public_robot_state": [0.0],
    }
    public = tmp_path / "public.jsonl"
    public.write_text(
        json.dumps(
            {
                "schema_version": "piu.public-transition.v1",
                "sample_id": sample,
                "initial_state_group": group,
                "split": "primitive_qualification",
                "prompt": "Place the butter in the basket",
                "observations": {
                    "pre_interaction": observation,
                    "post_interaction": observation,
                },
                "public_action_history": {
                    "initial_observation": True,
                    "last_executed_candidate": None,
                    "events": [],
                },
                "candidate_actions": [candidate],
                "online_oracle_inputs": [],
            }
        )
        + "\n"
    )
    capture = tmp_path / "capture.json"
    capture.write_text(
        json.dumps(
            {
                "schema_version": "piu.initial-observation-capture.v1",
                "sample_id": sample,
                "initial_state_group": group,
                "seed": 26081812,
                "simulator_steps_executed": 0,
                "rollout_executed": False,
                "outcomes_loaded": False,
                "pre_outcome_only": True,
                "state_reload_validated": True,
                "evaluator_fields_copied": [],
                "online_oracle_inputs": [],
                "local_pi05_loaded": False,
                "source_state": {**_reference(state), "state_key": "state"},
                "public_transition": _reference(public),
            }
        )
        + "\n"
    )
    output = tmp_path / "controller.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/build_piu_primitive_qualification_probe.py"),
            "--plan",
            str(ROOT / "results/method/piu_open_primitive_qualification_plan_v1.json"),
            "--candidate-set",
            str(candidate_set),
            "--sample-id",
            sample,
            "--initial-state-group",
            group,
            "--capture-report",
            str(capture),
            "--public-transition",
            str(public),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = load_qualification_controller_decision(
        output,
        candidate_id="open_middle_drawer",
        primitive="OPEN",
        initial_state_group=group,
        repository_root=ROOT,
    )
    assert loaded["controller_schema_version"] == (
        "piu.primitive-qualification-probe.v2"
    )
    assert loaded["source_state_sha256"] == _reference(state)["sha256"]
    assert loaded["simulator_seed"] == 26081812


def test_schedule_result_field_firewall_is_closed() -> None:
    _reject_embedded_qualification_results(
        {
            "outcomes_loaded": False,
            "outcome_loaded": False,
            "pre_outcome_only": True,
        },
        path="schedule",
    )
    for name in ("outcome", "success", "failure", "reward", "contact", "revealed", "empty"):
        with pytest.raises(ValueError, match="embeds result field"):
            _reject_embedded_qualification_results(
                {name: False}, path="schedule"
            )
