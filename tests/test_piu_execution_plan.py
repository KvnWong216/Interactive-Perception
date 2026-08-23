from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from piu.binding_calibration import MondrianBinaryLAC
from piu.execution_plan import PublicExecutionContext, candidate_eligibility

ROOT = Path(__file__).resolve().parents[1]


def _context(
    *, sufficient=(False,), present=(True,), holding=(False,), spatial=True
) -> PublicExecutionContext:
    return PublicExecutionContext.create(
        task_sufficiency=sufficient,
        target_presence=present,
        holding_requested_target=holding,
        spatial_reference_available=spatial,
    )


def test_candidate_execution_preconditions_match_public_controller_semantics() -> None:
    assert candidate_eligibility(
        {"primitive": "OPEN"}, _context(sufficient=(False,))
    ).eligible
    assert not candidate_eligibility(
        {"primitive": "OPEN"}, _context(sufficient=(True,))
    ).eligible
    assert candidate_eligibility(
        {"primitive": "PICK"}, _context(sufficient=(True,))
    ).eligible
    assert not candidate_eligibility(
        {"primitive": "PICK"}, _context(sufficient=(True,), spatial=False)
    ).eligible
    assert candidate_eligibility(
        {"primitive": "PLACE"}, _context(sufficient=(True,), holding=(True,))
    ).eligible
    assert not candidate_eligibility(
        {"primitive": "PLACE"}, _context(sufficient=(True,), holding=(False,))
    ).eligible


def test_terminal_candidate_is_exact_null_not_a_physical_precondition() -> None:
    result = candidate_eligibility({"primitive": "STOP"}, _context())
    assert result.eligible
    assert "exact-null" in result.reason


def test_execution_plan_masks_task_superset_outside_calibrated_context(
    tmp_path: Path,
) -> None:
    observation = {
        "images": {"agentview": {"sha256": "a" * 64}},
        "public_robot_state": [0.0],
    }
    transition = tmp_path / "public.jsonl"
    transition.write_text(
        json.dumps(
            {
                "schema_version": "piu.public-transition.v1",
                "sample_id": "sample",
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
                        "candidate_id": "open",
                        "primitive": "OPEN",
                        "target": "middle drawer",
                    },
                    {
                        "candidate_id": "pick",
                        "primitive": "PICK",
                        "target": "requested butter",
                    },
                    {
                        "candidate_id": "place",
                        "primitive": "PLACE",
                        "target": "requested butter",
                        "reference": "basket",
                    },
                    {"candidate_id": "stop", "primitive": "STOP", "target": "task"},
                ],
                "online_oracle_inputs": [],
            }
        )
        + "\n"
    )
    feature_report = tmp_path / "features.json"
    feature_report.write_text(
        json.dumps(
            {
                "schema_version": "piu.spatial-prefix-features.v1",
                "layout": {"camera_names": ["agentview"]},
            }
        )
    )
    checkpoint_digest = "c" * 64
    predictions = tmp_path / "binding.npz"
    np.savez_compressed(
        predictions,
        sample_id=np.asarray(["sample"]),
        initial_state_group=np.asarray(["group"]),
        split=np.asarray(["train"]),
        image_valid_mask=np.asarray([[True, True, True, True]]),
        patch_xy=np.asarray(
            [[[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]]
        ),
        camera_id=np.zeros((1, 4), dtype=np.int64),
        temporal_id=np.zeros((1, 4), dtype=np.int64),
        spatial_logits=np.asarray([[10.0, -10.0, -10.0, -10.0]]),
        target_present_logit=np.asarray([10.0]),
        task_sufficiency_logit=np.asarray([-10.0]),
        holding_requested_target_logit=np.asarray([-10.0]),
        region_confirmed_empty_logit=np.asarray([-10.0]),
        task_complete_logit=np.asarray([-10.0]),
    )
    binding_report = tmp_path / "binding.json"
    binding_report.write_text(
        json.dumps(
            {
                "schema_version": "piu.target-binder-online-predictions.v1",
                "inputs": {
                    "checkpoint": {"sha256": checkpoint_digest},
                    "feature_report": {
                        "sha256": hashlib.sha256(
                            feature_report.read_bytes()
                        ).hexdigest()
                    },
                },
                "output": {
                    "sha256": hashlib.sha256(predictions.read_bytes()).hexdigest()
                },
            }
        )
    )
    calibrator = MondrianBinaryLAC.fit(
        np.asarray([0.01, 0.02, 0.03, 0.97, 0.98, 0.99]),
        np.asarray([0, 0, 0, 1, 1, 1]),
        alpha=0.5,
    ).to_dict()
    binary = {
        "status": "SUPPORTED",
        "temperature": 1.0,
        "conformal": {"0.5": {"calibrator": calibrator}},
    }
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema_version": "piu.target-binder-calibration.v1",
                "checkpoint_sha256": checkpoint_digest,
                "risk_contract": {
                    "primary_alpha": 0.5,
                    "reported_alpha": [0.5],
                },
                "spatial": {
                    "temperature": 1.0,
                    "conformal": {
                        "0.5": {
                            "calibrator": {
                                "method": "target_intersection_split_conformal",
                                "quantile": 0.5,
                            }
                        }
                    },
                },
                "target_presence": binary,
                "task_sufficiency": binary,
                "holding_requested_target": binary,
                "region_confirmed_empty": binary,
                "task_complete": binary,
            }
        )
    )
    split_manifest = tmp_path / "split.yaml"
    split_manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.group-split-manifest.v1",
                "status": "FROZEN_BEFORE_COLLECTION",
                "allocation_method": "prospective_without_outcome_access",
                "scenario": "fixed drawer",
                "assignments": [
                    {
                        "initial_state_group": name,
                        "seed": 100 + index,
                        "split_role": role,
                    }
                    for index, (name, role) in enumerate(
                        (
                            ("group", "train"),
                            ("development", "development"),
                            ("temperature", "calibration_temperature"),
                            ("conformal", "calibration_conformal"),
                            ("sealed", "sealed_test"),
                        )
                    )
                ],
            },
            sort_keys=False,
        )
    )
    output = tmp_path / "execution_plan.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/data/build_piu_counterfactual_execution_plan.py"),
            "--public-transition",
            str(transition),
            "--sample-id",
            "sample",
            "--binding-predictions",
            str(predictions),
            "--binding-report",
            str(binding_report),
            "--binder-calibration",
            str(calibration),
            "--feature-report",
            str(feature_report),
            "--split-manifest",
            str(split_manifest),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(output.read_text())
    by_id = {row["candidate_id"]: row for row in plan["candidates"]}
    assert by_id["open"]["eligible_for_execution"] is True
    assert by_id["open"]["structured_pi05_subtask"] == "Open the middle drawer."
    assert by_id["pick"]["eligible_for_execution"] is False
    assert by_id["place"]["eligible_for_execution"] is False
    assert by_id["stop"]["eligible_for_execution"] is True
    assert plan["public_inputs_only"] is True
