from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from piu.action_effect import (
    EFFECT_FACTORS,
    CandidateConditionedEffectPredictor,
    EffectLabel,
    LearnedEffectObjective,
    build_effect_inputs,
    effect_objectives,
    join_effect_features,
)
from piu.effect_training import EffectHyperparameters, train_effect_predictor

ROOT = Path(__file__).resolve().parents[1]


def _effect_label(
    *,
    sample: str,
    group: str,
    candidate: str,
    primitive: str,
    split: str = "train",
) -> dict:
    is_stop = primitive in {"STOP", "REPORT_NOT_FOUND"}
    return {
        "schema_version": "piu.action-effect-label.v1",
        "sample_id": sample,
        "initial_state_group": group,
        "split": split,
        "candidate_id": candidate,
        "candidate_primitive": primitive,
        "decision_observation_sha256": "a" * 64,
        "outcome_observation_sha256": {"pre": "a" * 64, "post": "a" * 64},
        "selection_correct": primitive == "OPEN",
        "executed": not is_stop,
        "exact_null_transition": is_stop,
        "factors": {
            "execution_succeeded": None if is_stop else True,
            "task_progress_succeeded": False,
            "task_relevant_change": not is_stop,
            "target_revealed": not is_stop,
            "identity_resolved_post": False,
            "candidate_rejected": False,
            "region_confirmed_empty": False,
            "task_information_sufficient_post": False,
        },
        "simulator_teacher_only": True,
    }


def _effect_fixture(*, split: str = "train", stem: str = ""):
    count = 2
    sample_ids = [f"{stem}sample_0", f"{stem}sample_1"]
    groups = [f"{stem}group_0", f"{stem}group_1"]
    arrays = {
        "image_tokens": np.zeros((count, 2, 4, 8), dtype=np.float16),
        "image_valid_mask": np.ones((count, 2, 4), dtype=bool),
        "prompt_tokens": np.zeros((count, 2, 3, 8), dtype=np.float16),
        "prompt_valid_mask": np.ones((count, 2, 3), dtype=bool),
        "patch_xy": np.asarray(
            [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]],
            dtype=np.float32,
        ),
        "camera_id": np.zeros(4, dtype=np.int16),
        "sample_id": np.asarray(sample_ids),
        "initial_state_group": np.asarray(groups),
        "split": np.asarray([split, split]),
        "executed_action": np.asarray(["OPEN", "OPEN"]),
        "decision_observation_sha256": np.asarray(["a" * 64, "a" * 64]),
        "candidate_prompt_tokens": np.zeros((count, 2, 2, 3, 8), dtype=np.float16),
        "candidate_prompt_valid_mask": np.ones((count, 2, 2, 3), dtype=bool),
        "candidate_valid_mask": np.ones((count, 2), dtype=bool),
        "candidate_id": np.asarray([["open", "stop"], ["open", "stop"]]),
        "candidate_primitive": np.asarray([["OPEN", "STOP"], ["OPEN", "STOP"]]),
        "candidate_payload": np.asarray(
            [
                [
                    '{"candidate_id":"open","primitive":"OPEN","target":"middle drawer"}',
                    '{"candidate_id":"stop","primitive":"STOP","target":"task"}',
                ],
                [
                    '{"candidate_id":"open","primitive":"OPEN","target":"middle drawer"}',
                    '{"candidate_id":"stop","primitive":"STOP","target":"task"}',
                ],
            ]
        ),
    }
    binding = {
        "sample_id": arrays["sample_id"],
        "initial_state_group": arrays["initial_state_group"],
        "split": arrays["split"],
        "image_valid_mask": np.ones((count, 8), dtype=bool),
        "patch_xy": np.tile(arrays["patch_xy"], (count, 2, 1)),
        "camera_id": np.zeros((count, 8), dtype=np.int64),
        "temporal_id": np.tile(
            np.asarray([0] * 4 + [1] * 4, dtype=np.int64), (count, 1)
        ),
        "spatial_logits": np.zeros((count, 8), dtype=np.float32),
        "target_token": np.zeros((count, 8), dtype=np.float32),
        "target_present_logit": np.zeros(count, dtype=np.float32),
        "task_sufficiency_logit": np.zeros(count, dtype=np.float32),
        "holding_requested_target_logit": np.zeros(count, dtype=np.float32),
        "region_confirmed_empty_logit": np.zeros(count, dtype=np.float32),
        "task_complete_logit": np.zeros(count, dtype=np.float32),
    }
    labels = [
        EffectLabel.from_mapping(
            _effect_label(
                sample=sample_ids[sample_index],
                group=groups[sample_index],
                candidate=candidate,
                primitive=primitive,
                split=split,
            )
        )
        for sample_index in range(count)
        for candidate, primitive in (("open", "OPEN"), ("stop", "STOP"))
    ]
    return arrays, binding, labels


def test_exact_null_effect_contract_rejects_observation_drift() -> None:
    value = _effect_label(
        sample="sample", group="group", candidate="stop", primitive="STOP"
    )
    EffectLabel.from_mapping(value)
    value["outcome_observation_sha256"]["post"] = "b" * 64
    with pytest.raises(ValueError, match="preserve public observation"):
        EffectLabel.from_mapping(value)


def test_report_not_found_is_an_exact_nonphysical_null_transition() -> None:
    value = _effect_label(
        sample="sample",
        group="group",
        candidate="report_not_found",
        primitive="REPORT_NOT_FOUND",
    )
    label = EffectLabel.from_mapping(value)
    assert label.exact_null_transition is True
    assert label.executed is False


def test_context_ineligible_candidate_is_retained_with_all_effects_masked() -> None:
    value = _effect_label(
        sample="sample", group="group", candidate="place", primitive="PLACE"
    )
    value.update(
        {
            "selection_correct": True,
            "eligible_for_execution": False,
            "executed": False,
            "exact_null_transition": False,
            "factors": {name: None for name in EFFECT_FACTORS},
        }
    )
    label = EffectLabel.from_mapping(value)
    assert label.eligible_for_execution is False
    assert label.executed is False
    assert label.selection_correct is True
    value["factors"]["execution_succeeded"] = False
    with pytest.raises(ValueError, match="effects must all be null"):
        EffectLabel.from_mapping(value)


@pytest.mark.parametrize(
    "target_name",
    [
        "patch_target",
        "target_present",
        "task_sufficient",
        "task_sufficient_mask",
        "holding_requested_target",
        "holding_requested_target_mask",
        "region_confirmed_empty",
        "region_confirmed_empty_mask",
        "task_complete",
        "task_complete_mask",
        "route_target",
        "effect_target",
        "effect_support_mask",
        "selection_correct",
    ],
)
def test_online_effect_inputs_reject_every_evaluator_target(target_name: str) -> None:
    arrays, binding, _ = _effect_fixture()
    binding[target_name] = np.zeros(2, dtype=np.float32)
    with pytest.raises(ValueError, match="evaluator targets"):
        build_effect_inputs(
            feature_arrays=arrays,
            binding_predictions=binding,
            action_vocabulary=["OPEN", "STOP"],
        )


def test_effect_join_retains_candidates_factors_and_support_masks() -> None:
    arrays, binding, labels = _effect_fixture()
    joined = join_effect_features(
        feature_arrays=arrays,
        binding_predictions=binding,
        labels=labels,
        action_vocabulary=("OPEN", "STOP"),
    )
    assert joined.candidate_prompt_tokens.shape == (2, 2, 6, 8)
    assert joined.effect_target.shape == (2, 2, len(EFFECT_FACTORS))
    assert joined.effect_support_mask[:, 1, 0].tolist() == [False, False]
    assert joined.executed_mask.tolist() == [[True, False], [True, False]]


def test_effect_join_rejects_outcomes_from_a_different_decision_state() -> None:
    arrays, binding, labels = _effect_fixture()
    value = _effect_label(
        sample="sample_0", group="group_0", candidate="open", primitive="OPEN"
    )
    value["decision_observation_sha256"] = "b" * 64
    value["outcome_observation_sha256"]["pre"] = "b" * 64
    labels[0] = EffectLabel.from_mapping(value)
    with pytest.raises(ValueError, match="feature decision state"):
        join_effect_features(
            feature_arrays=arrays,
            binding_predictions=binding,
            labels=labels,
            action_vocabulary=("OPEN", "STOP"),
        )


def test_online_effect_input_builder_accepts_no_evaluator_labels() -> None:
    arrays, binding, _labels = _effect_fixture()
    joined = build_effect_inputs(
        feature_arrays=arrays,
        binding_predictions=binding,
        action_vocabulary=("OPEN", "STOP"),
    )
    assert joined.candidate_id.tolist() == [["open", "stop"], ["open", "stop"]]
    assert not hasattr(joined, "route_target")
    assert not hasattr(joined, "effect_target")
    binding["patch_target"] = np.zeros((2, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="evaluator targets"):
        build_effect_inputs(
            feature_arrays=arrays,
            binding_predictions=binding,
            action_vocabulary=("OPEN", "STOP"),
        )


def test_candidate_effect_predictor_and_learned_scales_are_finite() -> None:
    torch = pytest.importorskip("torch")
    arrays, binding, labels = _effect_fixture()
    joined = join_effect_features(
        feature_arrays=arrays,
        binding_predictions=binding,
        labels=labels,
        action_vocabulary=("OPEN", "STOP"),
    )
    model = CandidateConditionedEffectPredictor(
        belief_width=8,
        vlm_width=8,
        model_width=8,
        num_heads=2,
        maximum_action_types=2,
    )
    outputs = model(
        torch.as_tensor(joined.belief_token),
        torch.as_tensor(joined.candidate_prompt_tokens),
        candidate_prompt_valid_mask=torch.as_tensor(joined.candidate_prompt_valid_mask),
        candidate_valid_mask=torch.as_tensor(joined.candidate_valid_mask),
        candidate_action_ids=torch.as_tensor(joined.candidate_action_id),
    )
    losses = effect_objectives(
        outputs,
        route_target_distribution=torch.as_tensor(joined.route_target),
        factor_target=torch.as_tensor(joined.effect_target),
        factor_support_mask=torch.as_tensor(joined.effect_support_mask),
        candidate_valid_mask=torch.as_tensor(joined.candidate_valid_mask),
    )
    supported = {
        name: bool(joined.effect_support_mask[:, :, index].any())
        for index, name in enumerate(EFFECT_FACTORS)
    }
    supported["route_selection"] = True
    objective = LearnedEffectObjective()
    combined = objective(losses, supported=supported)
    combined["loss"].backward()
    assert torch.isfinite(combined["loss"])
    assert model.factor_head.weight.grad is not None
    assert model.route_head.in_features == model.joint[1].out_features + len(
        EFFECT_FACTORS
    )


def test_route_only_bridge_is_exactly_factor_independent() -> None:
    torch = pytest.importorskip("torch")
    arrays, binding, labels = _effect_fixture()
    joined = join_effect_features(
        feature_arrays=arrays,
        binding_predictions=binding,
        labels=labels,
        action_vocabulary=("OPEN", "STOP"),
    )
    model = CandidateConditionedEffectPredictor(
        belief_width=8,
        vlm_width=8,
        model_width=8,
        num_heads=2,
        maximum_action_types=2,
    )
    arguments = {
        "candidate_prompt_valid_mask": torch.as_tensor(
            joined.candidate_prompt_valid_mask
        ),
        "candidate_valid_mask": torch.as_tensor(joined.candidate_valid_mask),
        "candidate_action_ids": torch.as_tensor(joined.candidate_action_id),
        "effect_backprop_to_shared": False,
        "route_use_predicted_effects": False,
    }
    before = model(
        torch.as_tensor(joined.belief_token),
        torch.as_tensor(joined.candidate_prompt_tokens),
        **arguments,
    )["route_logits"]
    with torch.no_grad():
        model.factor_head.weight.fill_(100.0)
        model.factor_head.bias.fill_(-100.0)
    after = model(
        torch.as_tensor(joined.belief_token),
        torch.as_tensor(joined.candidate_prompt_tokens),
        **arguments,
    )["route_logits"]
    assert torch.equal(before, after)
    outputs = model(
        torch.as_tensor(joined.belief_token),
        torch.as_tensor(joined.candidate_prompt_tokens),
        **arguments,
    )
    losses = effect_objectives(
        outputs,
        route_target_distribution=torch.as_tensor(joined.route_target),
        factor_target=torch.as_tensor(joined.effect_target),
        factor_support_mask=torch.as_tensor(joined.effect_support_mask),
        candidate_valid_mask=torch.as_tensor(joined.candidate_valid_mask),
    )
    supported = {name: False for name in EFFECT_FACTORS}
    supported["route_selection"] = True
    LearnedEffectObjective()(losses, supported=supported)["loss"].backward()
    assert model.factor_head.weight.grad is None


@pytest.mark.parametrize(
    ("effect_backprop_to_shared", "shared_gradient_expected"),
    [(False, False), (True, True)],
)
def test_effect_gradient_ablation_reaches_shared_representation_only_when_declared(
    effect_backprop_to_shared: bool, shared_gradient_expected: bool
) -> None:
    torch = pytest.importorskip("torch")
    arrays, binding, labels = _effect_fixture()
    joined = join_effect_features(
        feature_arrays=arrays,
        binding_predictions=binding,
        labels=labels,
        action_vocabulary=("OPEN", "STOP"),
    )
    model = CandidateConditionedEffectPredictor(
        belief_width=8,
        vlm_width=8,
        model_width=8,
        num_heads=2,
        maximum_action_types=2,
    )
    outputs = model(
        torch.as_tensor(joined.belief_token),
        torch.as_tensor(joined.candidate_prompt_tokens),
        candidate_prompt_valid_mask=torch.as_tensor(joined.candidate_prompt_valid_mask),
        candidate_valid_mask=torch.as_tensor(joined.candidate_valid_mask),
        candidate_action_ids=torch.as_tensor(joined.candidate_action_id),
        effect_backprop_to_shared=effect_backprop_to_shared,
        route_use_predicted_effects=True,
    )
    losses = effect_objectives(
        outputs,
        route_target_distribution=torch.as_tensor(joined.route_target),
        factor_target=torch.as_tensor(joined.effect_target),
        factor_support_mask=torch.as_tensor(joined.effect_support_mask),
        candidate_valid_mask=torch.as_tensor(joined.candidate_valid_mask),
    )
    supported = {name: False for name in EFFECT_FACTORS}
    supported["route_selection"] = False
    supported["execution_succeeded"] = True
    LearnedEffectObjective()(losses, supported=supported)["loss"].backward()
    assert model.factor_head.weight.grad is not None
    assert (model.joint[1].weight.grad is not None) is shared_gradient_expected


@pytest.mark.parametrize(
    "variant", ["route_only", "stop_gradient_effect", "joint_effect"]
)
def test_effect_training_variants_are_explicit_and_cpu_finite(variant: str) -> None:
    pytest.importorskip("torch")
    arrays, binding, labels = _effect_fixture()
    joined = join_effect_features(
        feature_arrays=arrays,
        binding_predictions=binding,
        labels=labels,
        action_vocabulary=("OPEN", "STOP"),
    )
    model = CandidateConditionedEffectPredictor(
        belief_width=8,
        vlm_width=8,
        model_width=8,
        num_heads=2,
        maximum_action_types=2,
    )
    result = train_effect_predictor(
        model=model,
        objective=LearnedEffectObjective(),
        train=joined,
        development=joined,
        hyperparameters=EffectHyperparameters(
            model_width=8,
            num_heads=2,
            dropout=0.0,
            learning_rate=0.001,
            epochs=1,
            batch_size=2,
            seed=7,
        ),
        variant=variant,
    )
    assert result["variant"] == variant
    assert np.isfinite(result["development_metrics"]["route_nll"])


def _write_effect_bundle(
    directory: Path, *, split: str, stem: str
) -> tuple[Path, Path, Path, Path, Path]:
    arrays, binding, _ = _effect_fixture(split=split, stem=stem)
    feature_path = directory / f"{stem}features.npz"
    np.savez_compressed(feature_path, **arrays)
    feature_report = directory / f"{stem}features.json"
    feature_report.write_text(
        json.dumps(
            {
                "schema_version": "piu.spatial-prefix-features.v1",
                "output": {
                    "path": str(feature_path),
                    "sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
                },
                "layout": {
                    "camera_names": ["agentview"],
                    "tokens_per_camera": [4],
                },
            }
        )
    )
    binding_path = directory / f"{stem}binding.npz"
    np.savez_compressed(binding_path, **binding)
    binding_report = directory / f"{stem}binding.json"
    binding_report.write_text(
        json.dumps(
            {
                "schema_version": "piu.target-binder-online-predictions.v1",
                "output": {
                    "path": str(binding_path),
                    "sha256": hashlib.sha256(binding_path.read_bytes()).hexdigest(),
                },
                "inputs": {"checkpoint": {"sha256": "b" * 64}},
            }
        )
    )
    label_path = directory / f"{stem}effects.jsonl"
    rows = [
        _effect_label(
            sample=str(arrays["sample_id"][sample_index]),
            group=str(arrays["initial_state_group"][sample_index]),
            candidate=candidate,
            primitive=primitive,
            split=split,
        )
        for sample_index in range(2)
        for candidate, primitive in (("open", "OPEN"), ("stop", "STOP"))
    ]
    label_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return feature_path, feature_report, binding_path, binding_report, label_path


def test_action_effect_training_cli_retains_all_declared_ablations(
    tmp_path: Path,
) -> None:
    train = _write_effect_bundle(tmp_path, split="train", stem="train_")
    development = _write_effect_bundle(
        tmp_path, split="development", stem="development_"
    )
    config = {
        "schema_version": "piu.action-effect-experiment.v1",
        "inputs": {"action_vocabulary": ["OPEN", "STOP"]},
        "variants": ["route_only", "stop_gradient_effect", "joint_effect"],
        "model_search": {
            "model_width": [8],
            "num_heads": [2],
            "dropout": [0.0],
            "learning_rate": [0.001],
            "seeds": [7],
            "epochs": 1,
            "batch_size": 2,
            "maximum_parameter_count": 100000,
        },
        "objectives": {"manual_loss_weights": None},
        "compute": {"device": "cpu", "torch_threads": 1},
    }
    config_path = tmp_path / "effect_config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output_dir = tmp_path / "effect_output"
    command = [
        sys.executable,
        str(ROOT / "scripts/training/train_piu_action_effect.py"),
        "--config",
        str(config_path),
    ]
    for name, bundle in (("train", train), ("development", development)):
        command.extend(
            [
                f"--{name}-features",
                str(bundle[0]),
                f"--{name}-feature-report",
                str(bundle[1]),
                f"--{name}-binding-predictions",
                str(bundle[2]),
                f"--{name}-binding-report",
                str(bundle[3]),
                f"--{name}-labels",
                str(bundle[4]),
            ]
        )
    command.extend(["--output-dir", str(output_dir)])
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((output_dir / "training_report.json").read_text())
    assert set(report["variants"]) == {
        "route_only",
        "stop_gradient_effect",
        "joint_effect",
    }
    assert report["cuda_visible_to_trainer"] is False
    assert report["calibration_loaded"] is False
    assert report["sealed_test_loaded"] is False

    for method_id, variant, effect_channel in (
        ("B3", "route_only", False),
        ("B4", "joint_effect", True),
    ):
        ablation_output = tmp_path / f"{method_id}_development_controller.json"
        subprocess.run(
            [
                sys.executable,
                str(
                    ROOT
                    / "scripts/pipeline/run_piu_uncalibrated_ablation_controller.py"
                ),
                "--method-id",
                method_id,
                "--checkpoint",
                str(output_dir / f"{variant}.pt"),
                "--training-report",
                str(output_dir / "training_report.json"),
                "--features",
                str(development[0]),
                "--feature-report",
                str(development[1]),
                "--binding-predictions",
                str(development[2]),
                "--binding-report",
                str(development[3]),
                "--expected-split",
                "development",
                "--output",
                str(ablation_output),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        ablation = json.loads(ablation_output.read_text())
        assert ablation["method_id"] == method_id
        assert ablation["effect_channel_used"] is effect_channel
        assert ablation["evaluator_labels_loaded"] is False

    calibration_roles = {}
    for role in ("temperature", "conformal"):
        bundle = _write_effect_bundle(
            tmp_path,
            split="calibration",
            stem=f"calibration_{role}_",
        )
        predictions = tmp_path / f"effect_{role}.npz"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/evaluation/predict_piu_action_effect.py"),
                "--checkpoint",
                str(output_dir / "joint_effect.pt"),
                "--training-report",
                str(output_dir / "training_report.json"),
                "--features",
                str(bundle[0]),
                "--feature-report",
                str(bundle[1]),
                "--binding-predictions",
                str(bundle[2]),
                "--binding-report",
                str(bundle[3]),
                "--labels",
                str(bundle[4]),
                "--expected-split",
                "calibration",
                "--calibration-role",
                role,
                "--output",
                str(predictions),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        calibration_roles[role] = (predictions, predictions.with_suffix(".json"))
    calibration_path = tmp_path / "effect_calibration.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/calibration/calibrate_piu_action_effect.py"),
            "--config",
            str(ROOT / "configs/experiments/piu_action_effect_calibration_v1.yaml"),
            "--temperature-predictions",
            str(calibration_roles["temperature"][0]),
            "--temperature-report",
            str(calibration_roles["temperature"][1]),
            "--conformal-predictions",
            str(calibration_roles["conformal"][0]),
            "--conformal-report",
            str(calibration_roles["conformal"][1]),
            "--output",
            str(calibration_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    calibration = json.loads(calibration_path.read_text())
    assert calibration["variant"] == "joint_effect"
    assert calibration["sealed_test_loaded"] is False
    assert calibration["manual_confidence_thresholds"] is None

    sealed_bundle = _write_effect_bundle(tmp_path, split="sealed_test", stem="sealed_")
    sealed_predictions = tmp_path / "effect_sealed.npz"
    authorization = tmp_path / "effect_sealed_authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema_version": "piu.action-effect-sealed-authorization.v1",
                "checkpoint_sha256": hashlib.sha256(
                    (output_dir / "joint_effect.pt").read_bytes()
                ).hexdigest(),
                "feature_sha256": hashlib.sha256(
                    sealed_bundle[0].read_bytes()
                ).hexdigest(),
                "binding_prediction_sha256": hashlib.sha256(
                    sealed_bundle[2].read_bytes()
                ).hexdigest(),
                "effect_label_sha256": hashlib.sha256(
                    sealed_bundle[4].read_bytes()
                ).hexdigest(),
                "single_use_output": str(sealed_predictions),
            }
        )
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/predict_piu_action_effect.py"),
            "--checkpoint",
            str(output_dir / "joint_effect.pt"),
            "--training-report",
            str(output_dir / "training_report.json"),
            "--features",
            str(sealed_bundle[0]),
            "--feature-report",
            str(sealed_bundle[1]),
            "--binding-predictions",
            str(sealed_bundle[2]),
            "--binding-report",
            str(sealed_bundle[3]),
            "--labels",
            str(sealed_bundle[4]),
            "--expected-split",
            "sealed_test",
            "--sealed-authorization",
            str(authorization),
            "--output",
            str(sealed_predictions),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    sealed_result = tmp_path / "effect_sealed_result.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/evaluate_piu_action_effect.py"),
            "--predictions",
            str(sealed_predictions),
            "--prediction-report",
            str(sealed_predictions.with_suffix(".json")),
            "--calibration",
            str(calibration_path),
            "--output",
            str(sealed_result),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    sealed = json.loads(sealed_result.read_text())
    assert sealed["sealed_test_opened"] is True
    assert sealed["paper_method_claim_allowed"] is False

    binder_calibration_path = tmp_path / "binder_calibration.json"
    binder_calibration_path.write_text(
        json.dumps(
            {
                "schema_version": "piu.target-binder-calibration.v1",
                "checkpoint_sha256": "b" * 64,
                "primary_alpha": 0.1,
                "risk_contract": {"reported_alpha": [0.05, 0.1, 0.2]},
                "initial_state_groups": {
                    "temperature": ["binder_temperature"],
                    "conformal": ["binder_conformal"],
                },
                "spatial": {
                    "temperature": 1.0,
                    "conformal": {
                        str(alpha): {
                            "calibrator": {
                                "method": "target_intersection_split_conformal",
                                "quantile": 1.0,
                            }
                        }
                        for alpha in (0.05, 0.1, 0.2)
                    },
                },
                "target_presence": {"status": "UNSUPPORTED"},
                "task_sufficiency": {"status": "UNSUPPORTED"},
                "holding_requested_target": {"status": "UNSUPPORTED"},
                "region_confirmed_empty": {"status": "UNSUPPORTED"},
                "task_complete": {"status": "UNSUPPORTED"},
            }
        )
    )
    public_state_path = tmp_path / "sealed_public_state_sets.jsonl"
    public_state_path.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": "piu.controller-public-state-sets.v1",
                    "sample_id": str(sample_id),
                    "public_inputs_only": True,
                    "online_oracle_inputs": [],
                    "state_sets": {
                        "search_coverage_sufficient": [False],
                    },
                }
            )
            + "\n"
            for sample_id in np.load(sealed_bundle[0])["sample_id"]
        )
    )
    controller_output = tmp_path / "sealed_controller.json"
    controller_authorization = tmp_path / "controller_authorization.json"
    controller_authorization.write_text(
        json.dumps(
            {
                "schema_version": "piu.controller-sealed-authorization.v1",
                "checkpoint_sha256": hashlib.sha256(
                    (output_dir / "joint_effect.pt").read_bytes()
                ).hexdigest(),
                "feature_sha256": hashlib.sha256(
                    sealed_bundle[0].read_bytes()
                ).hexdigest(),
                "binding_prediction_sha256": hashlib.sha256(
                    sealed_bundle[2].read_bytes()
                ).hexdigest(),
                "binder_calibration_sha256": hashlib.sha256(
                    binder_calibration_path.read_bytes()
                ).hexdigest(),
                "effect_calibration_sha256": hashlib.sha256(
                    calibration_path.read_bytes()
                ).hexdigest(),
                "public_state_sets_sha256": hashlib.sha256(
                    public_state_path.read_bytes()
                ).hexdigest(),
                "method_id": "B8",
                "single_use_output": str(controller_output),
            }
        )
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/pipeline/run_piu_calibrated_controller.py"),
            "--checkpoint",
            str(output_dir / "joint_effect.pt"),
            "--training-report",
            str(output_dir / "training_report.json"),
            "--features",
            str(sealed_bundle[0]),
            "--feature-report",
            str(sealed_bundle[1]),
            "--binding-predictions",
            str(sealed_bundle[2]),
            "--binding-report",
            str(sealed_bundle[3]),
            "--binder-calibration",
            str(binder_calibration_path),
            "--effect-calibration",
            str(calibration_path),
            "--public-state-sets",
            str(public_state_path),
            "--expected-split",
            "sealed_test",
            "--sealed-authorization",
            str(controller_authorization),
            "--output",
            str(controller_output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    controller = json.loads(controller_output.read_text())
    assert controller["evaluator_labels_loaded"] is False
    assert "labels" not in controller["inputs"]
    assert {row["decision_kind"] for row in controller["decisions"]} == {"ABSTAIN"}
