from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from piu.formal_design import (
    paired_risk_difference_interval,
    prospective_paired_design,
)
from piu.formal_states import validate_state_archive

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/piu_formal_analysis_v1.yaml"
REGISTRY = ROOT / "configs/experiments/piu_baselines_v1.yaml"
IDENTITY = ROOT / "results/diagnostics/pi05_libero_checkpoint_identity_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _episode(
    tmp_path: Path, *, method: str, group: str, seed: int, success: bool
) -> Path:
    source = tmp_path / f"{group}.npz"
    if not source.exists():
        source.write_bytes(group.encode())
    history = tmp_path / f"{group}-{method}-history.json"
    history.write_text("[]\n")
    path = tmp_path / f"{group}-{method}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "piu.closed-loop-episode.v1",
                "method_id": method,
                "initial_state_group": group,
                "simulator_seed": seed,
                "split": "development",
                "evidence_class": "public_method",
                "rollout_status": "COMPLETE" if success else "FAILED",
                "source_state": _artifact(source),
                "policy_identity": _artifact(IDENTITY),
                "public_action_history": _artifact(history),
                "outcomes": {
                    "target_grasp_contact": success,
                    "task_success": success,
                },
                "online_oracle_inputs": [],
            }
        )
    )
    return path


def test_conservative_design_blocks_five_perfect_pairs_but_sizes_ten() -> None:
    five = prospective_paired_design(
        [True] * 5,
        [False] * 5,
        alpha=0.05,
        target_power=0.80,
        report_confidence=0.95,
        design_joint_confidence=0.95,
        search_limit=200,
    )
    assert five["status"] == (
        "PILOT_TOO_UNCERTAIN_FOR_CONSERVATIVE_DIRECTIONAL_PLAN"
    )
    ten = prospective_paired_design(
        [True] * 10,
        [False] * 10,
        alpha=0.05,
        target_power=0.80,
        report_confidence=0.95,
        design_joint_confidence=0.95,
        search_limit=200,
    )
    assert ten["status"] == "PROSPECTIVE_GROUP_COUNT_FROZEN"
    assert ten["design"]["prospective_group_count"] == 81
    interval = paired_risk_difference_interval(
        treatment_only=10,
        comparator_only=0,
        trials=10,
        confidence=0.95,
    )
    assert interval[0] > 0.0
    assert interval[1] == 1.0


def test_development_episode_arm_requires_exact_frozen_groups(tmp_path: Path) -> None:
    episodes = [
        _episode(
            tmp_path,
            method="B2",
            group=f"development-{index}",
            seed=100 + index,
            success=False,
        )
        for index in range(2)
    ]
    split = tmp_path / "development_split.json"
    split.write_text(
        json.dumps(
            {
                "schema_version": "piu.group-split-manifest.v1",
                "status": "FROZEN_BEFORE_COLLECTION",
                "allocation_method": "prospective_without_outcome_access",
                "scenario": "fixed drawer",
                "required_roles": ["development"],
                "assignments": [
                    {
                        "initial_state_group": f"development-{index}",
                        "seed": 100 + index,
                        "split_role": "development",
                    }
                    for index in range(2)
                ],
            }
        )
    )
    output = tmp_path / "b2.jsonl"
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluation/assemble_piu_development_episode_arm.py"),
        "--episode",
        *(str(path) for path in episodes),
        "--method-id",
        "B2",
        "--split-manifest",
        str(split),
        "--output",
        str(output),
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert {row["initial_state_group"] for row in rows} == {
        "development-0",
        "development-1",
    }
    rejected = subprocess.run(
        [*command[:3], str(episodes[0]), *command[5:-1], str(tmp_path / "bad.jsonl")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "complete frozen allocation" in rejected.stderr


def test_formal_state_archive_rejects_extra_keys_and_nonfinite_values(
    tmp_path: Path,
) -> None:
    extra = tmp_path / "extra.npz"
    np.savez(extra, state=np.asarray([1.0]), hidden=np.asarray([2.0]))
    with pytest.raises(ValueError, match="exactly key"):
        validate_state_archive(extra, state_key="state")
    nonfinite = tmp_path / "nonfinite.npz"
    np.savez(nonfinite, state=np.asarray([np.nan]))
    with pytest.raises(ValueError, match="nonfinite"):
        validate_state_archive(nonfinite, state_key="state")


def test_pilot_plan_and_schedule_are_paired_disjoint_and_hash_bound(
    tmp_path: Path,
) -> None:
    treatment = []
    comparator = []
    for index in range(10):
        group = f"pilot-{index:02d}"
        treatment.append(
            _episode(
                tmp_path,
                method="B8",
                group=group,
                seed=100 + index,
                success=True,
            )
        )
        comparator.append(
            _episode(
                tmp_path,
                method="B0",
                group=group,
                seed=100 + index,
                success=False,
            )
        )
    plan = tmp_path / "plan.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/plan_piu_formal_paired_test.py"),
            "--treatment-episodes",
            *(str(path) for path in treatment),
            "--comparator-episodes",
            *(str(path) for path in comparator),
            "--output",
            str(plan),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    planned = json.loads(plan.read_text())
    assert planned["status"] == "PROSPECTIVE_GROUP_COUNT_FROZEN"
    assert planned["design"]["prospective_group_count"] == 81
    assert planned["pilot"]["excluded_from_formal_analysis"] is True

    assignments = [
        {
            "initial_state_group": role,
            "seed": index,
            "split_role": role,
        }
        for index, role in enumerate(
            ("train", "development", "calibration_temperature", "calibration_conformal")
        )
    ]
    assignments.extend(
        {
            "initial_state_group": f"formal-{index:03d}",
            "seed": 1000 + index,
            "split_role": "sealed_test",
        }
        for index in range(81)
    )
    split = tmp_path / "split.yaml"
    split.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.group-split-manifest.v1",
                "status": "FROZEN_BEFORE_COLLECTION",
                "allocation_method": "prospective_without_outcome_access",
                "scenario": "fixed drawer",
                "assignments": assignments,
            }
        )
    )
    state_arguments = []
    for index in range(81):
        group = f"formal-{index:03d}"
        state = tmp_path / f"{group}.npz"
        np.savez_compressed(state, state=np.asarray([index], dtype=np.float32))
        state_arguments.extend(["--state", group, str(state)])
    initial_states = tmp_path / "initial_states.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/build_piu_formal_initial_states.py"),
            "--split-manifest",
            str(split),
            *state_arguments,
            "--output",
            str(initial_states),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    schedule = tmp_path / "schedule.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/build_piu_formal_schedule.py"),
            "--formal-plan",
            str(plan),
            "--split-manifest",
            str(split),
            "--initial-state-manifest",
            str(initial_states),
            "--output",
            str(schedule),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    frozen = json.loads(schedule.read_text())
    assert frozen["outcomes_loaded"] is False
    assert len(frozen["entries"]) == 81 * 9
    assert {row["method_id"] for row in frozen["entries"]} == {
        f"B{index}" for index in range(9)
    }
    assert len({row["execution_index"] for row in frozen["entries"]}) == 81 * 9
    source_by_group = {
        row["initial_state_group"]: row["source_state"]["sha256"]
        for row in frozen["entries"]
    }
    assert len(source_by_group) == 81
    assert len(set(source_by_group.values())) == 81


def test_schedule_rejects_reused_pilot_group(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": "piu.formal-paired-test-plan.v1",
                "status": "PROSPECTIVE_GROUP_COUNT_FROZEN",
                "config": {"sha256": _sha256(CONFIG)},
                "baseline_registry": {"sha256": _sha256(REGISTRY)},
                "offline_repro_lock": {
                    "sha256": _sha256(
                        ROOT
                        / "results/diagnostics/piu_offline_repro_preflight_v1.json"
                    )
                },
                "comparison": {
                    "treatment": "B8",
                    "comparator": "B0",
                    "outcome": "target_grasp_contact",
                },
                "pilot": {
                    "groups": ["reused"],
                    "policy_identity_sha256": _sha256(IDENTITY),
                },
                "design": {"prospective_group_count": 1},
            }
        )
    )
    split = tmp_path / "split.yaml"
    roles = (
        "train",
        "development",
        "calibration_temperature",
        "calibration_conformal",
        "sealed_test",
    )
    split.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.group-split-manifest.v1",
                "status": "FROZEN_BEFORE_COLLECTION",
                "allocation_method": "prospective_without_outcome_access",
                "scenario": "fixed drawer",
                "assignments": [
                    {
                        "initial_state_group": "reused" if index == 4 else role,
                        "seed": index,
                        "split_role": role,
                    }
                    for index, role in enumerate(roles)
                ],
            }
        )
    )
    state = tmp_path / "reused.npz"
    np.savez_compressed(state, state=np.asarray([1.0], dtype=np.float32))
    initial_states = tmp_path / "initial_states.json"
    initial_states.write_text(
        json.dumps(
            {
                "schema_version": "piu.formal-initial-state-manifest.v1",
                "status": "FROZEN_BEFORE_FORMAL_OUTCOME_COLLECTION",
                "outcomes_loaded": False,
                "split_manifest": {"sha256": _sha256(split)},
                "offline_repro_lock": {
                    "sha256": _sha256(
                        ROOT
                        / "results/diagnostics/piu_offline_repro_preflight_v1.json"
                    )
                },
                "states": [
                    {
                        "initial_state_group": "reused",
                        "simulator_seed": 4,
                        "state_key": "state",
                        "source_state": _artifact(state),
                    }
                ],
            }
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/build_piu_formal_schedule.py"),
            "--formal-plan",
            str(plan),
            "--split-manifest",
            str(split),
            "--initial-state-manifest",
            str(initial_states),
            "--output",
            str(tmp_path / "schedule.json"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "reuses a development pilot group" in completed.stderr
