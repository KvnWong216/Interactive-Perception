from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from piu.contracts import (
    EvaluatorSidecar,
    PublicTransition,
    Split,
    assert_public_policy_value,
    load_evaluator_sidecars,
    load_public_transitions,
    validate_group_splits,
    validate_public_sidecar_pair,
)
from piu.evaluation import aggregate_stage_evidence
from piu.spatial_prefix import PrefixLayout, validate_feature_arrays

ROOT = Path(__file__).resolve().parents[1]


def _public(*, group: str = "g1", split: str = "development") -> dict:
    observation = {
        "images": {
            "agentview": {"path": "agent.png", "sha256": "a" * 64},
            "wrist": {"path": "wrist.png", "sha256": "b" * 64},
        },
        "public_robot_state": [0.0] * 8,
    }
    return {
        "schema_version": "piu.public-transition.v1",
        "sample_id": f"{group}_sample",
        "initial_state_group": group,
        "split": split,
        "prompt": "Place the butter in the basket",
        "observations": {
            "pre_interaction": observation,
            "post_interaction": observation,
        },
        "public_action_history": {"path": "history.json", "sha256": "c" * 64},
        "candidate_actions": [
            {"candidate_id": "open", "primitive": "OPEN", "target": "drawer"}
        ],
        "online_oracle_inputs": [],
    }


def _sidecar(
    *,
    group: str = "g1",
    split: str = "development",
    acquired: bool = True,
    contacted: bool = False,
) -> dict:
    return {
        "schema_version": "piu.evaluator-sidecar.v1",
        "sample_id": f"{group}_sample",
        "initial_state_group": group,
        "split": split,
        "sufficiency_decision_correct": None,
        "interaction_selection_correct": None,
        "primitive_execution_success": True,
        "target_visible_pixels_pre": {"agentview": 0, "wrist": 0},
        "target_visible_pixels_post": {
            "agentview": 12 if acquired else 0,
            "wrist": 0,
        },
        "information_acquired": acquired,
        "target_identity_resolved": None,
        "target_grasp_contact": contacted,
        "wrong_object_grasp_contact": False,
        "target_maximum_lift_m": 0.0,
        "target_destination_final": False,
        "task_success": False,
        "provenance": {"claim_scope": "TEST"},
    }


def test_public_sidecar_firewall_and_group_split() -> None:
    public = PublicTransition.from_mapping(_public())
    sidecar = EvaluatorSidecar.from_mapping(_sidecar())
    validate_public_sidecar_pair(public, sidecar)
    assert "provenance" not in public.policy_input()
    with pytest.raises(ValueError, match="evaluator-only"):
        assert_public_policy_value({"target_pose": [0, 0, 0]})
    with pytest.raises(ValueError, match="group leakage"):
        validate_group_splits(
            [
                public,
                PublicTransition.from_mapping(
                    _public(group="g1", split="calibration")
                ),
            ]
        )


def test_acquisition_requires_zero_pre_and_nonempty_post_mask() -> None:
    invalid = _sidecar(acquired=True)
    invalid["target_visible_pixels_pre"] = {"agentview": 1}
    with pytest.raises(ValueError, match="information_acquired"):
        EvaluatorSidecar.from_mapping(invalid)


def test_stage_evaluator_keeps_missing_labels_unsupported() -> None:
    rows = [
        EvaluatorSidecar.from_mapping(_sidecar(group="g1", contacted=False)),
        EvaluatorSidecar.from_mapping(_sidecar(group="g2", contacted=True)),
        EvaluatorSidecar.from_mapping(_sidecar(group="g3", acquired=False)),
    ]
    report = aggregate_stage_evidence(rows)
    assert report["stages"]["L0_information_sufficiency"]["unsupported"] == 3
    assert report["stages"]["L3_information_acquisition"]["successes"] == 2
    conditioned = report["acquisition_to_utilization"][
        "target_contact_after_acquisition"
    ]
    assert conditioned == {
        "successes": 1,
        "trials": 2,
        "rate": 0.5,
        "wilson_95": conditioned["wilson_95"],
    }


def test_prefix_layout_retains_exact_spans_and_row_major_coordinates() -> None:
    layout = PrefixLayout.from_counts({"agentview": 4, "wrist": 9})
    assert layout.spans() == {"agentview": (0, 4), "wrist": (4, 13)}
    camera_id, xy = layout.patch_metadata()
    assert camera_id.tolist() == [0] * 4 + [1] * 9
    np.testing.assert_allclose(xy[0], [0.25, 0.25])
    np.testing.assert_allclose(xy[3], [0.75, 0.75])
    with pytest.raises(ValueError, match="non-square"):
        PrefixLayout.from_counts({"camera": 6}).patch_metadata()


def test_spatial_prefix_cache_contract() -> None:
    arrays = {
        "image_tokens": np.zeros((2, 2, 8, 16), dtype=np.float16),
        "image_valid_mask": np.ones((2, 2, 8), dtype=bool),
        "prompt_tokens": np.zeros((2, 2, 5, 16), dtype=np.float16),
        "prompt_valid_mask": np.ones((2, 2, 5), dtype=bool),
        "patch_xy": np.zeros((8, 2), dtype=np.float32),
        "camera_id": np.zeros(8, dtype=np.int16),
        "sample_id": np.asarray(["a", "b"]),
        "initial_state_group": np.asarray(["g1", "g2"]),
        "split": np.asarray(["train", "development"]),
    }
    validate_feature_arrays(arrays)
    arrays["prompt_valid_mask"] = np.ones((2, 2, 4), dtype=bool)
    with pytest.raises(ValueError, match="prompt_valid_mask"):
        validate_feature_arrays(arrays)


def test_sprint_config_defers_modules_and_has_no_small_sample_gate() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/experiments/piu_drawer_binding_sprint_v1.yaml").read_text()
    )
    assert config["oracle_gate"]["arbitrary_small_sample_pass_rule"] is None
    assert config["scope"]["scenario_expansion_allowed_before_gate"] is False
    assert "conformal action controller" in config["deferred_until_binding_gate"]
    assert config["resource_contract"]["local_gpu_memory_mib_max"] == 1500


def test_retrospective_dataset_rebuild_and_evaluation(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/data/build_piu_drawer_sprint_dataset.py"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    public = load_public_transitions(output_dir / "public_transitions.jsonl")
    sidecars = load_evaluator_sidecars(output_dir / "evaluator_sidecars.jsonl")
    assert len(public) == len(sidecars) == 10
    assert {row.split for row in public} == {Split.RETROSPECTIVE_DEVELOPMENT}
    report_path = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/evaluate_piu_stages.py"),
            "--public",
            str(output_dir / "public_transitions.jsonl"),
            "--evaluator",
            str(output_dir / "evaluator_sidecars.jsonl"),
            "--output",
            str(report_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(report_path.read_text())
    assert report["stages"]["L2_primitive_execution"]["successes"] == 9
    assert report["stages"]["L3_information_acquisition"]["successes"] == 8
    assert report["acquisition_to_utilization"][
        "target_contact_after_acquisition"
    ]["successes"] == 0


def test_prefix_extractor_dry_run_does_not_load_checkpoint_or_gpu() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/data/extract_piu_spatial_prefix_features.py"),
            "--public",
            "data/piu/drawer_binding_sprint_v1/public_transitions.jsonl",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["checkpoint_loaded"] is False
    assert report["gpu_used"] is False
