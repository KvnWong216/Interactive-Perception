from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from piu.binding_data import (
    BindingLabel,
    join_binding_features,
    mask_to_patch_coverage,
    rle_decode,
    rle_encode,
)
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
from piu.spatial_prefix import (
    PrefixLayout,
    libero_camera_to_label_view,
    validate_feature_arrays,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    assert libero_camera_to_label_view(
        ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    ) == {
        "base_0_rgb": "agentview",
        "left_wrist_0_rgb": "robot0_eye_in_hand",
        "right_wrist_0_rgb": None,
    }


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


def test_binding_rle_patch_alignment_and_join() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    encoded = rle_encode(mask)
    np.testing.assert_array_equal(rle_decode(encoded), mask)
    np.testing.assert_allclose(mask_to_patch_coverage(mask, grid_side=2), [1, 0, 0, 0])
    empty = rle_encode(np.zeros((4, 4), dtype=bool))
    label = BindingLabel.from_mapping(
        {
            "schema_version": "piu.binding-label.v1",
            "sample_id": "sample",
            "initial_state_group": "group",
            "split": "train",
            "target_mask_policy_resolution_rle": {
                "pre_interaction": {
                    "agentview": empty,
                    "robot0_eye_in_hand": empty,
                },
                "post_interaction": {
                    "agentview": encoded,
                    "robot0_eye_in_hand": empty,
                },
            },
            "target_present_post": True,
            "task_sufficient_post": None,
            "executed_action": "OPEN",
            "simulator_teacher_only": True,
        }
    )
    arrays = {
        "image_tokens": np.zeros((1, 2, 12, 8), dtype=np.float16),
        "image_valid_mask": np.ones((1, 2, 12), dtype=bool),
        "prompt_tokens": np.zeros((1, 2, 3, 8), dtype=np.float16),
        "prompt_valid_mask": np.ones((1, 2, 3), dtype=bool),
        "patch_xy": np.zeros((12, 2), dtype=np.float32),
        "camera_id": np.repeat(np.arange(3), 4),
        "sample_id": np.asarray(["sample"]),
        "initial_state_group": np.asarray(["group"]),
        "split": np.asarray(["train"]),
    }
    report = {
        "schema_version": "piu.spatial-prefix-features.v1",
        "layout": {
            "camera_names": [
                "base_0_rgb",
                "left_wrist_0_rgb",
                "right_wrist_0_rgb",
            ],
            "tokens_per_camera": [4, 4, 4],
            "camera_to_label_view": {
                "base_0_rgb": "agentview",
                "left_wrist_0_rgb": "robot0_eye_in_hand",
                "right_wrist_0_rgb": None,
            },
        },
    }
    joined = join_binding_features(
        feature_arrays=arrays,
        feature_report=report,
        labels=[label],
        action_vocabulary=("DIRECT", "OPEN"),
    )
    assert joined.image_tokens.shape == (1, 24, 8)
    assert joined.prompt_tokens.shape == (1, 6, 8)
    np.testing.assert_allclose(joined.patch_target[0, 12:16], [1, 0, 0, 0])
    assert joined.executed_action_id.tolist() == [1]
    assert joined.task_sufficient_mask.tolist() == [False]


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


def test_binding_label_exporter_dry_run_does_not_start_simulator() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/data/export_piu_binding_labels.py"),
            "--public",
            "data/piu/drawer_binding_sprint_v1/public_transitions.jsonl",
            "--evaluator",
            "data/piu/drawer_binding_sprint_v1/evaluator_sidecars.jsonl",
            "--scenario-config",
            "configs/scenarios/original_drawer.yaml",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["source_states_verified"] == 10
    assert report["simulator_started"] is False
    assert report["gpu_used"] is False


def _write_synthetic_binding_bundle(
    directory: Path,
    *,
    split: str,
    count: int,
    seed: int,
    name: str | None = None,
) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(seed)
    stem = split if name is None else name
    sample_ids = [f"{stem}_{index}" for index in range(count)]
    groups = [f"{stem}_group_{index}" for index in range(count)]
    image_tokens = rng.normal(size=(count, 2, 12, 8)).astype(np.float16)
    image_valid_mask = np.ones((count, 2, 12), dtype=bool)
    image_valid_mask[:, :, 8:12] = False
    prompt_tokens = rng.normal(size=(count, 2, 3, 8)).astype(np.float16)
    arrays = {
        "image_tokens": image_tokens,
        "image_valid_mask": image_valid_mask,
        "prompt_tokens": prompt_tokens,
        "prompt_valid_mask": np.ones((count, 2, 3), dtype=bool),
        "patch_xy": np.tile(
            np.asarray(
                [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]],
                dtype=np.float32,
            ),
            (3, 1),
        ),
        "camera_id": np.repeat(np.arange(3, dtype=np.int16), 4),
        "sample_id": np.asarray(sample_ids),
        "initial_state_group": np.asarray(groups),
        "split": np.asarray([split] * count),
    }
    feature_path = directory / f"{stem}_features.npz"
    np.savez_compressed(feature_path, **arrays)
    report_path = directory / f"{stem}_features.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "piu.spatial-prefix-features.v1",
                "output": {
                    "path": str(feature_path),
                    "sha256": _sha256(feature_path),
                },
                "layout": {
                    "camera_names": [
                        "base_0_rgb",
                        "left_wrist_0_rgb",
                        "right_wrist_0_rgb",
                    ],
                    "tokens_per_camera": [4, 4, 4],
                    "camera_to_label_view": {
                        "base_0_rgb": "agentview",
                        "left_wrist_0_rgb": "robot0_eye_in_hand",
                        "right_wrist_0_rgb": None,
                    },
                },
            }
        )
    )
    label_path = directory / f"{stem}_labels.jsonl"
    empty = rle_encode(np.zeros((4, 4), dtype=bool))
    rows = []
    for index, (sample_id, group) in enumerate(
        zip(sample_ids, groups, strict=True)
    ):
        present = index % 2 == 0
        target = np.zeros((4, 4), dtype=bool)
        if present:
            target[:2, :2] = True
        rows.append(
            {
                "schema_version": "piu.binding-label.v1",
                "sample_id": sample_id,
                "initial_state_group": group,
                "split": split,
                "target_mask_policy_resolution_rle": {
                    "pre_interaction": {
                        "agentview": empty,
                        "robot0_eye_in_hand": empty,
                    },
                    "post_interaction": {
                        "agentview": rle_encode(target),
                        "robot0_eye_in_hand": empty,
                    },
                },
                "target_present_post": present,
                "task_sufficient_post": None,
                "executed_action": "OPEN",
                "simulator_teacher_only": True,
            }
        )
    label_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return feature_path, report_path, label_path


def test_cpu_binding_trainer_is_split_safe_and_writes_frozen_artifacts(
    tmp_path: Path,
) -> None:
    train = _write_synthetic_binding_bundle(
        tmp_path, split="train", count=4, seed=7
    )
    development = _write_synthetic_binding_bundle(
        tmp_path, split="development", count=4, seed=11
    )
    config = {
        "schema_version": "piu.binding-adapter-experiment.v1",
        "inputs": {"action_vocabulary": ["DIRECT", "OPEN"]},
        "model_search": {
            "model_width": [8],
            "num_heads": [2],
            "dropout": [0.0],
            "learning_rate": [0.001],
            "seeds": [13],
            "epochs": 2,
            "batch_size": 2,
            "maximum_parameter_count": 100000,
            "selection_order": [
                "minimize_development_spatial_nll",
                "minimize_development_presence_brier",
                "minimize_parameter_count",
            ],
        },
        "objectives": {"manual_loss_weights": None},
        "compute": {"device": "cpu", "torch_threads": 1},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    checkpoint_path = tmp_path / "binder.pt"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/training/train_piu_target_binding.py"),
            "--config",
            str(config_path),
            "--train-features",
            str(train[0]),
            "--train-feature-report",
            str(train[1]),
            "--train-labels",
            str(train[2]),
            "--development-features",
            str(development[0]),
            "--development-feature-report",
            str(development[1]),
            "--development-labels",
            str(development[2]),
            "--output",
            str(checkpoint_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(checkpoint_path.with_suffix(".json").read_text())
    assert report["claim_scope"] == "DEVELOPMENT_MODEL_SELECTION_NOT_TEST_EVIDENCE"
    assert report["train_groups"] == report["development_groups"] == 4
    assert report["cuda_visible_to_trainer"] is False
    assert report["sealed_test_loaded"] is False
    assert report["calibration_loaded"] is False
    assert len(report["trials"]) == 1
    assert checkpoint_path.is_file()
    with np.load(checkpoint_path.with_suffix(".development_predictions.npz")) as data:
        assert data["spatial_attention"].shape == (4, 24)
        assert data["task_sufficient_mask"].tolist() == [False] * 4

    calibration_roles = {}
    for role, seed in (("temperature", 17), ("conformal", 19)):
        bundle = _write_synthetic_binding_bundle(
            tmp_path,
            split="calibration",
            count=4,
            seed=seed,
            name=f"calibration_{role}",
        )
        output = tmp_path / f"{role}_predictions.npz"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/evaluation/predict_piu_target_binding.py"),
                "--checkpoint",
                str(checkpoint_path),
                "--training-report",
                str(checkpoint_path.with_suffix(".json")),
                "--features",
                str(bundle[0]),
                "--feature-report",
                str(bundle[1]),
                "--labels",
                str(bundle[2]),
                "--expected-split",
                "calibration",
                "--calibration-role",
                role,
                "--output",
                str(output),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        calibration_roles[role] = (output, output.with_suffix(".json"))
    calibration_output = tmp_path / "calibrator.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/calibration/calibrate_piu_target_binding.py"),
            "--config",
            str(ROOT / "configs/experiments/piu_binding_calibration_v1.yaml"),
            "--temperature-predictions",
            str(calibration_roles["temperature"][0]),
            "--temperature-report",
            str(calibration_roles["temperature"][1]),
            "--conformal-predictions",
            str(calibration_roles["conformal"][0]),
            "--conformal-report",
            str(calibration_roles["conformal"][1]),
            "--output",
            str(calibration_output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    calibration = json.loads(calibration_output.read_text())
    assert calibration["temperature_conformal_groups_disjoint"] is True
    assert calibration["sealed_test_loaded"] is False
    assert calibration["spatial"]["status"] == "SUPPORTED"
    assert calibration["target_presence"]["status"] == "SUPPORTED"
    assert calibration["task_sufficiency"]["status"] == "UNSUPPORTED"

    sealed_bundle = _write_synthetic_binding_bundle(
        tmp_path,
        split="sealed_test",
        count=4,
        seed=23,
    )
    sealed_predictions = tmp_path / "sealed_predictions.npz"
    authorization_path = tmp_path / "sealed_authorization.json"
    authorization_path.write_text(
        json.dumps(
            {
                "schema_version": "piu.sealed-test-authorization.v1",
                "checkpoint_sha256": _sha256(checkpoint_path),
                "feature_sha256": _sha256(sealed_bundle[0]),
                "label_sha256": _sha256(sealed_bundle[2]),
                "single_use_output": str(sealed_predictions),
            }
        )
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/predict_piu_target_binding.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--training-report",
            str(checkpoint_path.with_suffix(".json")),
            "--features",
            str(sealed_bundle[0]),
            "--feature-report",
            str(sealed_bundle[1]),
            "--labels",
            str(sealed_bundle[2]),
            "--expected-split",
            "sealed_test",
            "--sealed-authorization",
            str(authorization_path),
            "--output",
            str(sealed_predictions),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    sealed_result_path = tmp_path / "sealed_result.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/evaluate_piu_target_binding.py"),
            "--predictions",
            str(sealed_predictions),
            "--prediction-report",
            str(sealed_predictions.with_suffix(".json")),
            "--calibration",
            str(calibration_output),
            "--output",
            str(sealed_result_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    sealed_result = json.loads(sealed_result_path.read_text())
    assert sealed_result["sealed_test_opened"] is True
    assert sealed_result["spatial"]["proper_scores"]["samples"] == 2
    assert sealed_result["paper_method_claim_allowed"] is False
