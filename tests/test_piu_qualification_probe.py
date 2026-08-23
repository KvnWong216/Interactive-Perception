from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
import numpy as np

from piu.binding_calibration import MondrianBinaryLAC
from piu.contracts import load_public_transitions, public_observation_sha256
from piu.execution_plan import calibrated_candidate_plan
from piu.primitive_registry import load_qualification_controller_decision
from piu_test_artifacts import write_formal_primitive_certificate

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/evaluation/build_piu_primitive_qualification_probe.py"
SCHEDULE_BUILDER = (
    ROOT / "scripts/evaluation/build_piu_primitive_qualification_schedule.py"
)
BINDING_BUILDER = (
    ROOT / "scripts/evaluation/build_piu_binding_qualification_stimulus.py"
)


def _reference(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _qualification_group(directory: Path, candidate_id: str) -> str:
    split = yaml.safe_load((directory / f"{candidate_id}_split.yaml").read_text())
    return next(
        row["initial_state_group"]
        for row in split["assignments"]
        if row["split_role"] == "primitive_qualification"
    )


def _qualification_groups(directory: Path, candidate_id: str) -> list[str]:
    split = yaml.safe_load((directory / f"{candidate_id}_split.yaml").read_text())
    return [
        row["initial_state_group"]
        for row in split["assignments"]
        if row["split_role"] == "primitive_qualification"
    ]


def test_open_probe_breaks_pretraining_cycle_without_method_claim(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "qualification"
    write_formal_primitive_certificate(
        fixture, candidate_id="open_middle_drawer", primitive="OPEN"
    )
    plan = fixture / "open_middle_drawer_plan.json"
    groups = _qualification_groups(fixture, "open_middle_drawer")
    group = groups[0]
    candidate = {
        "candidate_id": "open_middle_drawer",
        "primitive": "OPEN",
        "target": "middle drawer",
        "purpose": "inspect inside",
        "required_capability": "OPEN",
    }
    candidate_set = tmp_path / "candidates.jsonl"
    candidate_set.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": "piu.public-candidate-set.v1",
                    "sample_id": f"open-probe::{item}",
                    "initial_state_group": item,
                    "split": "primitive_qualification",
                    "public_inputs_only": True,
                    "online_oracle_inputs": [],
                    "candidates": [candidate],
                }
            )
            + "\n"
            for item in groups
        )
    )
    output = tmp_path / "probe.json"
    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--plan",
            str(plan),
            "--candidate-set",
            str(candidate_set),
            "--sample-id",
            f"open-probe::{group}",
            "--initial-state-group",
            group,
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["selection_source"] == (
        "preregistered_executor_probe_not_method_decision"
    )
    assert report["trained_model_loaded"] is False
    assert report["calibration_loaded"] is False
    assert report["paper_method_selection_claim_allowed"] is False
    assert report["online_oracle_inputs"] == []
    loaded = load_qualification_controller_decision(
        output,
        candidate_id="open_middle_drawer",
        primitive="OPEN",
        initial_state_group=group,
        repository_root=ROOT,
    )
    assert loaded["spatial_reference_mode"] == "none"
    assert loaded["structured_subtask"] == "Open the middle drawer."

    widened = json.loads(output.read_text())
    widened["decisions"].append(dict(widened["decisions"][0]))
    widened_path = tmp_path / "widened_probe.json"
    widened_path.write_text(json.dumps(widened, indent=2) + "\n")
    with pytest.raises(ValueError, match="one decision"):
        load_qualification_controller_decision(
            widened_path,
            candidate_id="open_middle_drawer",
            primitive="OPEN",
            initial_state_group=group,
            repository_root=ROOT,
        )

    report_paths = {group: output}
    for item in groups[1:]:
        copied = json.loads(output.read_text())
        copied["decisions"][0]["sample_id"] = f"open-probe::{item}"
        copied["decisions"][0]["initial_state_group"] = item
        path = tmp_path / f"{item}_probe.json"
        path.write_text(json.dumps(copied, indent=2) + "\n")
        report_paths[item] = path
    schedule = tmp_path / "probe_schedule.json"
    command = [
        sys.executable,
        str(SCHEDULE_BUILDER),
        "--plan",
        str(plan),
        "--split-manifest",
        str(fixture / "open_middle_drawer_split.yaml"),
        "--run-root",
        str(tmp_path / "probe_runs"),
        "--output",
        str(schedule),
    ]
    for item in groups:
        command.extend(
            [
                "--state",
                item,
                str(fixture / f"{item}.npz"),
                "--controller-report",
                item,
                str(report_paths[item]),
            ]
        )
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    frozen = json.loads(schedule.read_text())
    assert len(frozen["entries"]) == len(groups)
    assert frozen["candidate_contract"]["spatial_reference_mode"] == "none"

    candidate_set.write_text(candidate_set.read_text() + "\n")
    with pytest.raises(ValueError, match="artifact differs"):
        load_qualification_controller_decision(
            output,
            candidate_id="open_middle_drawer",
            primitive="OPEN",
            initial_state_group=group,
            repository_root=ROOT,
        )


def test_model_free_probe_cannot_bypass_pick_spatial_binding(tmp_path: Path) -> None:
    fixture = tmp_path / "qualification"
    write_formal_primitive_certificate(
        fixture, candidate_id="pick_butter", primitive="PICK"
    )
    group = _qualification_group(fixture, "pick_butter")
    candidate_set = tmp_path / "candidates.jsonl"
    candidate_set.write_text(
        json.dumps(
            {
                "schema_version": "piu.public-candidate-set.v1",
                "sample_id": "pick-probe",
                "initial_state_group": group,
                "split": "primitive_qualification",
                "public_inputs_only": True,
                "online_oracle_inputs": [],
                "candidates": [
                    {
                        "candidate_id": "pick_butter",
                        "primitive": "PICK",
                        "target": "butter",
                    }
                ],
            }
        )
        + "\n"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--plan",
            str(fixture / "pick_butter_plan.json"),
            "--candidate-set",
            str(candidate_set),
            "--sample-id",
            "pick-probe",
            "--initial-state-group",
            group,
            "--output",
            str(tmp_path / "probe.json"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "OPEN-only" in completed.stderr
    assert not (tmp_path / "probe.json").exists()


def test_pick_qualification_uses_recomputed_calibrated_binder_boxes(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "qualification"
    write_formal_primitive_certificate(
        fixture, candidate_id="pick_butter", primitive="PICK"
    )
    group = _qualification_group(fixture, "pick_butter")
    plan_path = fixture / "pick_butter_plan.json"
    candidate = json.loads((fixture / f"{group}_controller.json").read_text())[
        "decisions"
    ][0]["selected_candidate"]
    observation = {
        "images": {"agentview": {"sha256": "a" * 64}},
        "public_robot_state": [0.0],
    }
    public_path = tmp_path / "public.jsonl"
    public_path.write_text(
        json.dumps(
            {
                "schema_version": "piu.public-transition.v1",
                "sample_id": "pick-qualification",
                "initial_state_group": group,
                "split": "primitive_qualification",
                "prompt": "pick butter",
                "observations": {
                    "pre_interaction": observation,
                    "post_interaction": observation,
                },
                "public_action_history": {
                    "initial_observation": True,
                    "last_executed_candidate": None,
                },
                "candidate_actions": [candidate],
                "online_oracle_inputs": [],
            }
        )
        + "\n"
    )
    feature_report_path = tmp_path / "features.json"
    feature_report_path.write_text(
        json.dumps(
            {
                "schema_version": "piu.spatial-prefix-features.v1",
                "layout": {"camera_names": ["agentview"]},
            }
        )
    )
    predictions_path = tmp_path / "binding.npz"
    np.savez_compressed(
        predictions_path,
        sample_id=np.asarray(["pick-qualification"]),
        initial_state_group=np.asarray([group]),
        split=np.asarray(["primitive_qualification"]),
        image_valid_mask=np.asarray([[True, True, True, True]]),
        patch_xy=np.asarray(
            [[[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]]
        ),
        camera_id=np.zeros((1, 4), dtype=np.int64),
        temporal_id=np.zeros((1, 4), dtype=np.int64),
        spatial_logits=np.asarray([[10.0, -10.0, -10.0, -10.0]]),
        target_present_logit=np.asarray([10.0]),
        task_sufficiency_logit=np.asarray([10.0]),
        holding_requested_target_logit=np.asarray([-10.0]),
        region_confirmed_empty_logit=np.asarray([-10.0]),
        task_complete_logit=np.asarray([-10.0]),
    )
    checkpoint_digest = "c" * 64
    binding_report_path = tmp_path / "binding.json"
    binding_report_path.write_text(
        json.dumps(
            {
                "schema_version": "piu.target-binder-online-predictions.v1",
                "inputs": {
                    "checkpoint": {"sha256": checkpoint_digest},
                    "feature_report": {
                        "sha256": hashlib.sha256(
                            feature_report_path.read_bytes()
                        ).hexdigest()
                    },
                },
                "output": {
                    "sha256": hashlib.sha256(
                        predictions_path.read_bytes()
                    ).hexdigest()
                },
            }
        )
    )
    fitted = MondrianBinaryLAC.fit(
        np.asarray([0.01, 0.02, 0.03, 0.97, 0.98, 0.99]),
        np.asarray([0, 0, 0, 1, 1, 1]),
        alpha=0.5,
    ).to_dict()
    binary = {
        "status": "SUPPORTED",
        "temperature": 1.0,
        "conformal": {"0.5": {"calibrator": fitted}},
    }
    calibration_path = tmp_path / "calibration.json"
    calibration = {
        "schema_version": "piu.target-binder-calibration.v1",
        "checkpoint_sha256": checkpoint_digest,
        "risk_contract": {"primary_alpha": 0.5, "reported_alpha": [0.5]},
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
    calibration_path.write_text(json.dumps(calibration))
    with np.load(predictions_path) as store:
        binding = {name: np.asarray(store[name]) for name in store.files}
    transition = load_public_transitions(public_path)[0]
    planned = calibrated_candidate_plan(
        transition=transition,
        binding=binding,
        calibration=calibration,
        feature_report=json.loads(feature_report_path.read_text()),
        sample_index=0,
        alpha=0.5,
    )
    execution_plan_path = tmp_path / "execution_plan.json"
    execution_plan_path.write_text(
        json.dumps(
            {
                "schema_version": "piu.counterfactual-execution-plan.v1",
                "decision_sample_id": transition.sample_id,
                "initial_state_group": group,
                "split": "primitive_qualification",
                "split_role": "primitive_qualification",
                "decision_observation_sha256": public_observation_sha256(
                    transition.observations["post_interaction"]
                ),
                **planned,
                "public_inputs_only": True,
                "online_oracle_inputs": [],
                "inputs": {
                    "public_transition": _reference(public_path),
                    "binding_predictions": _reference(predictions_path),
                    "binding_report": _reference(binding_report_path),
                    "binder_calibration": _reference(calibration_path),
                    "feature_report": _reference(feature_report_path),
                    "split_manifest": _reference(
                        fixture / "pick_butter_split.yaml"
                    ),
                },
                "paper_method_claim_allowed": False,
            }
        )
    )
    output = tmp_path / "binding_stimulus.json"
    subprocess.run(
        [
            sys.executable,
            str(BINDING_BUILDER),
            "--plan",
            str(plan_path),
            "--execution-plan",
            str(execution_plan_path),
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
        candidate_id="pick_butter",
        primitive="PICK",
        initial_state_group=group,
        repository_root=ROOT,
    )
    assert loaded["spatial_reference_mode"] == "calibrated_current_frame_boxes"
    tampered = json.loads(execution_plan_path.read_text())
    tampered["candidates"][0]["spatial_references"][0]["x_interval"] = [0.0, 1.0]
    execution_plan_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="artifact differs"):
        load_qualification_controller_decision(
            output,
            candidate_id="pick_butter",
            primitive="PICK",
            initial_state_group=group,
            repository_root=ROOT,
        )
