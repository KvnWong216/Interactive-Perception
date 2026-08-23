from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from piu.candidate_generator import PublicAffordanceEntity, generate_candidates

ROOT = Path(__file__).resolve().parents[1]


def test_generator_expands_every_public_affordance_without_correct_container() -> None:
    entities = [
        PublicAffordanceEntity.from_mapping(
            {
                "entity_id": "middle_drawer",
                "description": "middle drawer",
                "capabilities": ["OPEN"],
            }
        ),
        PublicAffordanceEntity.from_mapping(
            {
                "entity_id": "front_box",
                "description": "box in front",
                "capabilities": ["REMOVE", "ROTATE"],
            }
        ),
    ]
    candidates = generate_candidates(
        task_target_description="butter",
        destination_description="basket",
        entities=entities,
    )
    assert [row["primitive"] for row in candidates] == [
        "PICK",
        "PLACE",
        "OPEN",
        "REMOVE",
        "ROTATE",
        "REPORT_NOT_FOUND",
        "STOP",
    ]
    assert all("container" not in row for row in candidates)


def test_generator_always_retains_pick_and_place_public_superset() -> None:
    candidates = generate_candidates(
        task_target_description="cream cheese",
        destination_description="basket",
        entities=[],
    )
    assert [row["primitive"] for row in candidates[:2]] == ["PICK", "PLACE"]


def test_candidate_context_rejects_injected_holding_belief(tmp_path: Path) -> None:
    context = tmp_path / "context.jsonl"
    context.write_text(
        json.dumps(
            {
                "schema_version": "piu.public-candidate-context.v1",
                "sample_id": "sample",
                "initial_state_group": "group",
                "split": "train",
                "split_role": "train",
                "public_inputs_only": True,
                "online_oracle_inputs": [],
                "task": {
                    "target_description": "butter",
                    "destination_description": "basket",
                },
                "holding_requested_target_set": [True],
                "public_affordance_entities": [],
            }
        )
        + "\n"
    )
    split_manifest = tmp_path / "splits.yaml"
    split_manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.group-split-manifest.v1",
                "status": "FROZEN_BEFORE_COLLECTION",
                "allocation_method": "prospective_without_outcome_access",
                "scenario": "fixed_drawer",
                "assignments": [
                    {
                        "initial_state_group": group,
                        "seed": index,
                        "split_role": role,
                    }
                    for index, (group, role) in enumerate(
                        (
                            ("group", "train"),
                            ("development_group", "development"),
                            ("temperature_group", "calibration_temperature"),
                            ("conformal_group", "calibration_conformal"),
                            ("sealed_group", "sealed_test"),
                        )
                    )
                ],
            }
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/data/build_piu_candidate_sets.py"),
            "--contexts",
            str(context),
            "--split-manifest",
            str(split_manifest),
            "--output",
            str(tmp_path / "candidates.jsonl"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "cannot inject holding belief" in completed.stderr


def test_privileged_affordance_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="evaluator-only"):
        PublicAffordanceEntity.from_mapping(
            {
                "entity_id": "drawer",
                "description": "drawer",
                "capabilities": ["OPEN"],
                "target_pose": [0.0, 0.0, 0.0],
            }
        )


def test_candidate_set_cli_and_initial_capture_dry_run(tmp_path: Path) -> None:
    split_manifest = tmp_path / "splits.yaml"
    split_manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.group-split-manifest.v1",
                "status": "FROZEN_BEFORE_COLLECTION",
                "allocation_method": "prospective_without_outcome_access",
                "scenario": "fixed_drawer",
                "assignments": [
                    {
                        "initial_state_group": group,
                        "seed": index,
                        "split_role": role,
                    }
                    for index, (group, role) in enumerate(
                        (
                            ("group", "train"),
                            ("development_group", "development"),
                            ("temperature_group", "calibration_temperature"),
                            ("conformal_group", "calibration_conformal"),
                            ("sealed_group", "sealed_test"),
                        )
                    )
                ],
            }
        )
    )
    context = tmp_path / "context.jsonl"
    context.write_text(
        json.dumps(
            {
                "schema_version": "piu.public-candidate-context.v1",
                "sample_id": "initial",
                "initial_state_group": "group",
                "split": "train",
                "split_role": "train",
                "public_inputs_only": True,
                "online_oracle_inputs": [],
                "task": {
                    "target_description": "requested butter",
                    "destination_description": "basket",
                },
                "public_affordance_entities": [
                    {
                        "entity_id": "middle_drawer",
                        "description": "middle drawer",
                        "capabilities": ["OPEN"],
                    }
                ],
            }
        )
        + "\n"
    )
    candidates = tmp_path / "candidates.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/data/build_piu_candidate_sets.py"),
            "--contexts",
            str(context),
            "--split-manifest",
            str(split_manifest),
            "--output",
            str(candidates),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    row = json.loads(candidates.read_text())
    assert [item["primitive"] for item in row["candidates"]] == [
        "PICK",
        "PLACE",
        "OPEN",
        "REPORT_NOT_FOUND",
        "STOP",
    ]
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/data/capture_piu_initial_observation.py"),
            "--scenario-config",
            str(ROOT / "configs/scenarios/original_drawer.yaml"),
            "--candidate-set",
            str(candidates),
            "--sample-id",
            "initial",
            "--seed",
            "1",
            "--output-dir",
            str(tmp_path / "capture"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)
    assert plan["pi05_loaded"] is False
    assert plan["external_simulator_required"] is True
