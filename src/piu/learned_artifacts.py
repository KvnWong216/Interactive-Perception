"""Semantic verification for CPU-learned PIU artifacts.

The empirical DAG must not accept a JSON report merely because its referenced
files exist.  These validators reload frozen inputs and checkpoints on CPU,
replay inference, and recompute proper-score and calibration artifacts.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .ablations import apply_binding_ablation
from .action_effect import (
    CandidateConditionedEffectPredictor,
    LearnedEffectObjective,
    join_effect_features,
    load_effect_labels,
)
from .binding_calibration import fit_binding_calibration
from .binding_data import build_binding_inputs, join_binding_features, load_binding_labels
from .binding_training import probability_metrics, tensor_batch as binder_tensor_batch
from .effect_calibration import fit_effect_calibration
from .effect_training import effect_probability_metrics, tensor_batch as effect_tensor_batch
from .target_binding import LearnedMultiTaskObjective, PromptConditionedTargetBinder


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(path: str | Path, *, repository_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repository_root / value


def _reference_path(
    reference: Any, *, repository_root: Path, name: str
) -> Path:
    if not isinstance(reference, Mapping):
        raise TypeError(f"{name} must be an artifact reference")
    path = _resolve(str(reference.get("path", "")), repository_root=repository_root)
    if not path.is_file() or reference.get("sha256") != _sha256(path):
        raise ValueError(f"{name} differs from its content hash")
    return path


def _load_json(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise ValueError(f"unsupported {schema} artifact")
    return value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as store:
        return {name: np.asarray(store[name]) for name in store.files}


def _load_referenced_npz(
    reference: Any, *, repository_root: Path, name: str
) -> tuple[Path, dict[str, np.ndarray]]:
    path = _reference_path(reference, repository_root=repository_root, name=name)
    return path, _load_npz(path)


def _load_checkpoint(
    reference: Any,
    *,
    repository_root: Path,
    name: str,
    schema: str,
) -> tuple[Path, dict[str, Any]]:
    path = _reference_path(reference, repository_root=repository_root, name=name)
    import torch

    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise ValueError(f"unsupported {name} checkpoint")
    for state_name in ("model_state", "objective_state"):
        state = value.get(state_name)
        if not isinstance(state, Mapping) or not state:
            raise ValueError(f"{name} lacks {state_name}")
        if any(
            not hasattr(tensor, "isfinite") or not bool(tensor.isfinite().all())
            for tensor in state.values()
        ):
            raise ValueError(f"{name} contains non-finite or non-tensor state")
    return path, value


def _same_array(observed: Any, expected: Any, *, name: str) -> None:
    left = np.asarray(observed)
    right = np.asarray(expected)
    if left.shape != right.shape:
        raise ValueError(f"{name} shape differs from exact replay")
    if left.dtype.kind in "fc" or right.dtype.kind in "fc":
        if not np.allclose(left, right, rtol=1e-5, atol=1e-6, equal_nan=True):
            raise ValueError(f"{name} values differ from exact replay")
    elif not np.array_equal(left, right):
        raise ValueError(f"{name} values differ from exact replay")


def _same_json(observed: Any, expected: Any, *, name: str) -> None:
    if isinstance(observed, Mapping) and isinstance(expected, Mapping):
        if set(observed) != set(expected):
            raise ValueError(f"{name} fields differ from exact recomputation")
        for key in expected:
            _same_json(observed[key], expected[key], name=f"{name}.{key}")
        return
    if (
        isinstance(observed, Sequence)
        and not isinstance(observed, (str, bytes))
        and isinstance(expected, Sequence)
        and not isinstance(expected, (str, bytes))
    ):
        if len(observed) != len(expected):
            raise ValueError(f"{name} length differs from exact recomputation")
        for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
            _same_json(left, right, name=f"{name}[{index}]")
        return
    if isinstance(observed, (int, float)) and isinstance(expected, (int, float)):
        if isinstance(observed, bool) or isinstance(expected, bool):
            if observed is not expected:
                raise ValueError(f"{name} differs from exact recomputation")
        elif not math.isclose(float(observed), float(expected), rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"{name} differs from exact recomputation")
        return
    if observed != expected:
        raise ValueError(f"{name} differs from exact recomputation")


def _input_path(
    report: Mapping[str, Any], key: str, *, repository_root: Path
) -> Path:
    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping) or key not in inputs:
        raise ValueError(f"report lacks input {key}")
    return _reference_path(
        inputs[key], repository_root=repository_root, name=f"input {key}"
    )


def _reported_npz(
    report: Mapping[str, Any],
    data_key: str,
    report_key: str,
    schema: str,
    *,
    repository_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data_path = _input_path(report, data_key, repository_root=repository_root)
    report_path = _input_path(report, report_key, repository_root=repository_root)
    source_report = _load_json(report_path, schema)
    output = source_report.get("output")
    if not isinstance(output, Mapping) or output.get("sha256") != _sha256(data_path):
        raise ValueError(f"{data_key} differs from its source report")
    return _load_npz(data_path), source_report


def _binder_bundle(
    report: Mapping[str, Any], prefix: str, *, repository_root: Path
):
    feature_arrays, feature_report = _reported_npz(
        report,
        f"{prefix}_features",
        f"{prefix}_feature_report",
        "piu.spatial-prefix-features.v1",
        repository_root=repository_root,
    )
    labels_path = _input_path(report, f"{prefix}_labels", repository_root=repository_root)
    action_vocabulary = [str(value) for value in yaml.safe_load(
        _input_path(report, "config", repository_root=repository_root).read_text()
    )["inputs"]["action_vocabulary"]]
    return join_binding_features(
        feature_arrays=feature_arrays,
        feature_report=feature_report,
        labels=load_binding_labels(labels_path),
        action_vocabulary=action_vocabulary,
    )


def _binder_model(checkpoint: Mapping[str, Any]):
    model = PromptConditionedTargetBinder(**checkpoint["model"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.cpu().eval()
    objective = LearnedMultiTaskObjective()
    objective.load_state_dict(checkpoint["objective_state"], strict=True)
    objective.cpu().eval()
    return model, objective


def _binder_forward(model: Any, bundle: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    batch = binder_tensor_batch(bundle)
    with torch.no_grad():
        outputs = model(
            batch["image_tokens"],
            batch["prompt_tokens"],
            patch_xy=batch["patch_xy"],
            camera_ids=batch["camera_ids"],
            temporal_ids=batch["temporal_ids"],
            executed_action_ids=batch["executed_action_ids"],
            image_valid_mask=batch["image_valid_mask"],
            prompt_valid_mask=batch["prompt_valid_mask"],
        )
    return outputs, batch


_BINDER_TRAIN_ARRAYS = {
    "sample_id",
    "initial_state_group",
    "image_valid_mask",
    "spatial_logits",
    "spatial_attention",
    "target_token",
    "target_present_logit",
    "task_sufficiency_logit",
    "holding_requested_target_logit",
    "region_confirmed_empty_logit",
    "task_complete_logit",
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
}


_BINDER_PREDICTION_ARRAYS = _BINDER_TRAIN_ARRAYS - {"spatial_attention"} | {"split"}


def _validate_binder_arrays(
    arrays: Mapping[str, Any], bundle: Any, outputs: Mapping[str, Any], *, training: bool
) -> dict[str, Any]:
    required = _BINDER_TRAIN_ARRAYS if training else _BINDER_PREDICTION_ARRAYS
    if set(arrays) != required:
        raise ValueError(f"binder prediction arrays differ: {sorted(set(arrays) ^ required)}")
    source = {
        "sample_id": bundle.sample_id,
        "initial_state_group": bundle.initial_state_group,
        "image_valid_mask": bundle.image_valid_mask,
        "patch_target": bundle.patch_target,
        "target_present": bundle.target_present,
        "task_sufficient": bundle.task_sufficient,
        "task_sufficient_mask": bundle.task_sufficient_mask,
        "holding_requested_target": bundle.holding_requested_target,
        "holding_requested_target_mask": bundle.holding_requested_target_mask,
        "region_confirmed_empty": bundle.region_confirmed_empty,
        "region_confirmed_empty_mask": bundle.region_confirmed_empty_mask,
        "task_complete": bundle.task_complete,
        "task_complete_mask": bundle.task_complete_mask,
    }
    if not training:
        source["split"] = [value.value for value in bundle.split]
    for name, expected in source.items():
        _same_array(arrays[name], expected, name=f"binder predictions.{name}")
    output_names = {
        "spatial_logits",
        "target_token",
        "target_present_logit",
        "task_sufficiency_logit",
        "holding_requested_target_logit",
        "region_confirmed_empty_logit",
        "task_complete_logit",
    }
    if training:
        output_names.add("spatial_attention")
    for name in output_names:
        _same_array(
            arrays[name], outputs[name].detach().cpu().numpy(), name=f"binder predictions.{name}"
        )
    return source


def _expected_search(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    search = config["model_search"]
    return [
        {
            "model_width": int(width),
            "num_heads": int(heads),
            "dropout": float(dropout),
            "learning_rate": float(rate),
            "epochs": int(search["epochs"]),
            "batch_size": int(search["batch_size"]),
            "seed": int(seed),
        }
        for width, heads, dropout, rate, seed in itertools.product(
            search["model_width"],
            search["num_heads"],
            search["dropout"],
            search["learning_rate"],
            search["seeds"],
        )
    ]


def _validate_trial_history(
    trial: Mapping[str, Any], *, effect: bool, route_only: bool = False
) -> None:
    hyperparameters = trial["hyperparameters"]
    history = trial.get("history")
    if not isinstance(history, list) or len(history) != int(hyperparameters["epochs"]):
        raise ValueError("training history differs from the declared epoch budget")
    ranks = []
    for index, row in enumerate(history):
        if row.get("epoch") != index + 1 or not math.isfinite(float(row["mean_train_loss"])):
            raise ValueError("training history epoch/loss is malformed")
        metrics = row["development"]
        if effect:
            factor = metrics["macro_supported_factor_brier"]
            factor_rank = (
                float("inf") if factor is None or route_only else float(factor)
            )
            ranks.append((float(metrics["route_nll"]), factor_rank, index + 1))
        else:
            ranks.append(
                (
                    float(metrics["spatial_nll"]),
                    float(metrics["presence_brier"]),
                    index + 1,
                )
            )
    best = min(ranks)
    if trial.get("best_epoch") != best[-1]:
        raise ValueError("reported best epoch differs from history")
    expected_metrics = history[best[-1] - 1]["development"]
    _same_json(
        trial.get("development_metrics"),
        expected_metrics,
        name="selected development metrics",
    )


def validate_binder_training_report(
    path: Path, *, repository_root: Path
) -> dict[str, Any]:
    report = _load_json(path, "piu.target-binder-training.v1")
    config_path = _reference_path(
        report.get("config"), repository_root=repository_root, name="binder config"
    )
    config = yaml.safe_load(config_path.read_text())
    if (
        config.get("schema_version") != "piu.binding-adapter-experiment.v1"
        or config["compute"]["device"] != "cpu"
        or config["objectives"].get("manual_loss_weights") is not None
    ):
        raise ValueError("binder training config violates the frozen CPU contract")
    inputs = dict(report.get("inputs", {}))
    expected_inputs = {
        f"{prefix}_{name}"
        for prefix in ("train", "development")
        for name in ("features", "feature_report", "labels")
    }
    if set(inputs) != expected_inputs:
        raise ValueError("binder trainer inputs are not field-closed")
    # _binder_bundle expects the canonical config through the report input map.
    report_with_config = {**report, "inputs": {**inputs, "config": report["config"]}}
    train = _binder_bundle(report_with_config, "train", repository_root=repository_root)
    development = _binder_bundle(
        report_with_config, "development", repository_root=repository_root
    )
    groups = {
        "train": sorted(set(train.initial_state_group)),
        "development": sorted(set(development.initial_state_group)),
    }
    if set(groups["train"]) & set(groups["development"]):
        raise ValueError("binder train/development groups overlap")
    if report.get("initial_state_groups") != groups or report.get("train_groups") != len(
        groups["train"]
    ) or report.get("development_groups") != len(groups["development"]):
        raise ValueError("binder report groups differ from source data")
    expected_search = _expected_search(config)
    trials = report.get("trials")
    if not isinstance(trials, list) or len(trials) != len(expected_search):
        raise ValueError("binder report does not retain the complete search")
    for index, (trial, expected) in enumerate(zip(trials, expected_search, strict=True)):
        if trial.get("trial") != index or trial.get("hyperparameters") != expected:
            raise ValueError("binder trial grid differs from the frozen search")
        if int(trial["parameter_count"]) > int(config["model_search"]["maximum_parameter_count"]):
            raise ValueError("binder trial exceeds the parameter cap")
        _validate_trial_history(trial, effect=False)
    selected = min(
        trials,
        key=lambda row: (
            float(row["development_metrics"]["spatial_nll"]),
            float(row["development_metrics"]["presence_brier"]),
            int(row["parameter_count"]),
            int(row["trial"]),
        ),
    )
    if report.get("selected_trial") != selected["trial"] or report.get(
        "selection_order"
    ) != config["model_search"]["selection_order"]:
        raise ValueError("binder selected trial differs from the frozen order")
    declared = [str(value) for value in config.get("development_ablations", ["full"])]
    ablations = report.get("development_ablations")
    if not isinstance(ablations, Mapping) or list(ablations) != declared:
        raise ValueError("binder ablation set differs from the frozen config")
    no_history_id = None
    vocabulary = [str(value) for value in config["inputs"]["action_vocabulary"]]
    if "no_action_history" in declared:
        normalized = [value.upper() for value in vocabulary]
        if normalized.count("NO_HISTORY") != 1:
            raise ValueError("binder no-history ablation lacks its null token")
        no_history_id = normalized.index("NO_HISTORY")
    for name in declared:
        section = ablations[name]
        _validate_trial_history(
            {
                "hyperparameters": selected["hyperparameters"],
                "history": section.get("history"),
                "best_epoch": section.get("best_epoch"),
                "development_metrics": section.get("development_metrics"),
            },
            effect=False,
        )
        _, checkpoint = _load_checkpoint(
            section.get("checkpoint"),
            repository_root=repository_root,
            name=f"binder {name}",
            schema="piu.target-binder-checkpoint.v1",
        )
        expected_checkpoint_fields = {
            "schema_version",
            "model",
            "action_vocabulary",
            "model_state",
            "objective_state",
        } | ({"development_ablation"} if name != "full" else set())
        if set(checkpoint) != expected_checkpoint_fields:
            raise ValueError(f"binder {name} checkpoint fields are not closed")
        if checkpoint.get("action_vocabulary") != vocabulary or (
            name != "full" and checkpoint.get("development_ablation") != name
        ):
            raise ValueError(f"binder {name} checkpoint identity differs")
        model, objective = _binder_model(checkpoint)
        expected_model = {
            "vlm_width": int(development.image_tokens.shape[-1]),
            "model_width": int(selected["hyperparameters"]["model_width"]),
            "num_heads": int(selected["hyperparameters"]["num_heads"]),
            "maximum_cameras": int(max(train.camera_id.max(), development.camera_id.max())) + 1,
            "maximum_time_steps": int(
                max(train.temporal_id.max(), development.temporal_id.max())
            )
            + 1,
            "maximum_action_types": len(vocabulary),
            "dropout": float(selected["hyperparameters"]["dropout"]),
        }
        if checkpoint.get("model") != expected_model:
            raise ValueError(f"binder {name} model config differs from selected trial")
        parameter_count = sum(
            parameter.numel()
            for parameter in itertools.chain(model.parameters(), objective.parameters())
        )
        if name == "full" and parameter_count != selected["parameter_count"]:
            raise ValueError("binder checkpoint parameter count differs from selection")
        ablated = (
            development
            if name == "full"
            else apply_binding_ablation(
                development,
                name=name,
                seed=int(selected["hyperparameters"]["seed"]),
                no_history_action_id=no_history_id,
            )
        )
        outputs, batch = _binder_forward(model, ablated)
        _, predictions = _load_referenced_npz(
            section.get("development_predictions"),
            repository_root=repository_root,
            name=f"binder {name} development predictions",
        )
        _validate_binder_arrays(predictions, ablated, outputs, training=True)
        metrics = probability_metrics(outputs, batch)
        expected_metrics = {key: value for key, value in metrics.items() if key != "raw"}
        _same_json(section.get("development_metrics"), expected_metrics, name=f"binder {name} metrics")
        if section.get("best_epoch") is None:
            raise ValueError(f"binder {name} lacks a selected epoch")
    if report.get("checkpoint") != ablations["full"]["checkpoint"] or report.get(
        "development_predictions"
    ) != ablations["full"]["development_predictions"]:
        raise ValueError("binder primary artifacts differ from the full ablation")
    _same_json(
        selected["development_metrics"],
        ablations["full"]["development_metrics"],
        name="binder selected trial metrics",
    )
    return report


def validate_binder_prediction_report(
    path: Path, *, repository_root: Path
) -> dict[str, Any]:
    report = _load_json(path, "piu.target-binder-predictions.v1")
    if set(report.get("inputs", {})) != {
        "checkpoint",
        "training_report",
        "features",
        "feature_report",
        "labels",
    }:
        raise ValueError("binder prediction inputs are not field-closed")
    training_path = _input_path(report, "training_report", repository_root=repository_root)
    training = validate_binder_training_report(training_path, repository_root=repository_root)
    checkpoint_path = _input_path(report, "checkpoint", repository_root=repository_root)
    if _sha256(checkpoint_path) != training["checkpoint"]["sha256"]:
        raise ValueError("binder prediction uses another checkpoint")
    _, checkpoint = _load_checkpoint(
        report["inputs"]["checkpoint"],
        repository_root=repository_root,
        name="binder prediction",
        schema="piu.target-binder-checkpoint.v1",
    )
    feature_arrays, feature_report = _reported_npz(
        report,
        "features",
        "feature_report",
        "piu.spatial-prefix-features.v1",
        repository_root=repository_root,
    )
    labels_path = _input_path(report, "labels", repository_root=repository_root)
    bundle = join_binding_features(
        feature_arrays=feature_arrays,
        feature_report=feature_report,
        labels=load_binding_labels(labels_path),
        action_vocabulary=checkpoint["action_vocabulary"],
    )
    model, _ = _binder_model(checkpoint)
    outputs, _ = _binder_forward(model, bundle)
    _, arrays = _load_referenced_npz(
        report.get("output"), repository_root=repository_root, name="binder predictions"
    )
    _validate_binder_arrays(arrays, bundle, outputs, training=False)
    groups = sorted(set(bundle.initial_state_group))
    if report.get("initial_state_groups") != groups or report.get("samples") != len(bundle.sample_id):
        raise ValueError("binder prediction report differs from its rows")
    if set(arrays["split"].astype(str)) != {str(report.get("split"))}:
        raise ValueError("binder prediction split differs from its report")
    return report


_BINDER_ONLINE_ARRAYS = {
    "sample_id",
    "initial_state_group",
    "split",
    "image_valid_mask",
    "patch_xy",
    "camera_id",
    "temporal_id",
    "spatial_logits",
    "target_token",
    "target_present_logit",
    "task_sufficiency_logit",
    "holding_requested_target_logit",
    "region_confirmed_empty_logit",
    "task_complete_logit",
}


def validate_binder_online_prediction_report(
    path: Path, *, repository_root: Path
) -> dict[str, Any]:
    report = _load_json(path, "piu.target-binder-online-predictions.v1")
    if set(report.get("inputs", {})) != {
        "checkpoint",
        "training_report",
        "features",
        "feature_report",
    }:
        raise ValueError("online binder inputs are not field-closed")
    training_path = _input_path(report, "training_report", repository_root=repository_root)
    training = validate_binder_training_report(training_path, repository_root=repository_root)
    _, checkpoint = _load_checkpoint(
        report["inputs"]["checkpoint"],
        repository_root=repository_root,
        name="online binder",
        schema="piu.target-binder-checkpoint.v1",
    )
    if report["inputs"]["checkpoint"]["sha256"] != training["checkpoint"]["sha256"]:
        raise ValueError("online binder uses another checkpoint")
    feature_arrays, feature_report = _reported_npz(
        report,
        "features",
        "feature_report",
        "piu.spatial-prefix-features.v1",
        repository_root=repository_root,
    )
    values = build_binding_inputs(
        feature_arrays=feature_arrays,
        feature_report=feature_report,
        action_vocabulary=checkpoint["action_vocabulary"],
    )
    model, _ = _binder_model(checkpoint)
    import torch

    with torch.no_grad():
        outputs = model(
            torch.as_tensor(values.image_tokens),
            torch.as_tensor(values.prompt_tokens),
            patch_xy=torch.as_tensor(values.patch_xy),
            camera_ids=torch.as_tensor(values.camera_id),
            temporal_ids=torch.as_tensor(values.temporal_id),
            executed_action_ids=torch.as_tensor(values.executed_action_id),
            image_valid_mask=torch.as_tensor(values.image_valid_mask),
            prompt_valid_mask=torch.as_tensor(values.prompt_valid_mask),
        )
    _, arrays = _load_referenced_npz(
        report.get("output"), repository_root=repository_root, name="online binder predictions"
    )
    if set(arrays) != _BINDER_ONLINE_ARRAYS:
        raise ValueError(f"online binder arrays differ: {sorted(set(arrays) ^ _BINDER_ONLINE_ARRAYS)}")
    source = {
        "sample_id": values.sample_id,
        "initial_state_group": values.initial_state_group,
        "split": [value.value for value in values.split],
        "image_valid_mask": values.image_valid_mask,
        "patch_xy": values.patch_xy,
        "camera_id": values.camera_id,
        "temporal_id": values.temporal_id,
    }
    for name, expected in source.items():
        _same_array(arrays[name], expected, name=f"online binder.{name}")
    for name in _BINDER_ONLINE_ARRAYS - set(source):
        _same_array(arrays[name], outputs[name].detach().cpu().numpy(), name=f"online binder.{name}")
    if report.get("initial_state_groups") != sorted(set(values.initial_state_group)):
        raise ValueError("online binder groups differ from its source")
    return report


def _effect_bundle(report: Mapping[str, Any], prefix: str, *, repository_root: Path):
    feature_values, _ = _reported_npz(
        report,
        f"{prefix}_features",
        f"{prefix}_feature_report",
        "piu.spatial-prefix-features.v1",
        repository_root=repository_root,
    )
    binding_values, _ = _reported_npz(
        report,
        f"{prefix}_binding_predictions",
        f"{prefix}_binding_report",
        "piu.target-binder-online-predictions.v1",
        repository_root=repository_root,
    )
    labels = load_effect_labels(
        _input_path(report, f"{prefix}_labels", repository_root=repository_root)
    )
    config = yaml.safe_load(_input_path(report, "config", repository_root=repository_root).read_text())
    return join_effect_features(
        feature_arrays=feature_values,
        binding_predictions=binding_values,
        labels=labels,
        action_vocabulary=[str(value) for value in config["inputs"]["action_vocabulary"]],
    )


def _effect_model(checkpoint: Mapping[str, Any]):
    model = CandidateConditionedEffectPredictor(**checkpoint["model"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.cpu().eval()
    objective = LearnedEffectObjective()
    objective.load_state_dict(checkpoint["objective_state"], strict=True)
    objective.cpu().eval()
    return model, objective


def _effect_forward(model: Any, bundle: Any, variant: str):
    import torch

    batch = effect_tensor_batch(bundle)
    with torch.no_grad():
        outputs = model(
            batch["belief_token"],
            batch["candidate_prompt_tokens"],
            candidate_prompt_valid_mask=batch["candidate_prompt_valid_mask"],
            candidate_valid_mask=batch["candidate_valid_mask"],
            candidate_action_ids=batch["candidate_action_id"],
            effect_backprop_to_shared=variant == "joint_effect",
            route_use_predicted_effects=variant != "route_only",
        )
    return outputs, batch


_EFFECT_TRAIN_ARRAYS = {
    "sample_id",
    "initial_state_group",
    "split",
    "candidate_valid_mask",
    "route_logits",
    "route_target",
    "factor_logits",
    "factor_target",
    "factor_support_mask",
}
_EFFECT_PREDICTION_ARRAYS = _EFFECT_TRAIN_ARRAYS | {
    "candidate_id",
    "candidate_primitive",
    "candidate_payload",
}


def _validate_effect_arrays(
    arrays: Mapping[str, Any], bundle: Any, outputs: Mapping[str, Any], *, training: bool
) -> None:
    required = _EFFECT_TRAIN_ARRAYS if training else _EFFECT_PREDICTION_ARRAYS
    if set(arrays) != required:
        raise ValueError(f"effect prediction arrays differ: {sorted(set(arrays) ^ required)}")
    source = {
        "sample_id": bundle.sample_id,
        "initial_state_group": bundle.initial_state_group,
        "split": [value.value for value in bundle.split],
        "candidate_valid_mask": bundle.candidate_valid_mask,
        "route_target": bundle.route_target,
        "factor_target": bundle.effect_target,
        "factor_support_mask": bundle.effect_support_mask,
    }
    if not training:
        source.update(
            candidate_id=bundle.candidate_id,
            candidate_primitive=bundle.candidate_primitive,
            candidate_payload=bundle.candidate_payload,
        )
    for name, expected in source.items():
        _same_array(arrays[name], expected, name=f"effect predictions.{name}")
    for name in ("route_logits", "factor_logits"):
        _same_array(arrays[name], outputs[name].detach().cpu().numpy(), name=f"effect predictions.{name}")


def validate_effect_training_report(
    path: Path, *, repository_root: Path
) -> dict[str, Any]:
    report = _load_json(path, "piu.action-effect-training.v1")
    expected_inputs = {"config"} | {
        f"{prefix}_{name}"
        for prefix in ("train", "development")
        for name in (
            "features",
            "feature_report",
            "binding_predictions",
            "binding_report",
            "labels",
        )
    }
    if set(report.get("inputs", {})) != expected_inputs:
        raise ValueError("effect trainer inputs are not field-closed")
    config_path = _input_path(report, "config", repository_root=repository_root)
    config = yaml.safe_load(config_path.read_text())
    if (
        config.get("schema_version") != "piu.action-effect-experiment.v1"
        or config["compute"]["device"] != "cpu"
        or config["objectives"].get("manual_loss_weights") is not None
    ):
        raise ValueError("effect training config violates the frozen CPU contract")
    train = _effect_bundle(report, "train", repository_root=repository_root)
    development = _effect_bundle(report, "development", repository_root=repository_root)
    groups = {
        "train": sorted(set(train.initial_state_group)),
        "development": sorted(set(development.initial_state_group)),
    }
    if set(groups["train"]) & set(groups["development"]) or report.get("initial_state_groups") != groups:
        raise ValueError("effect training groups differ from source data")
    variants = report.get("variants")
    declared = [str(value) for value in config["variants"]]
    if not isinstance(variants, Mapping) or list(variants) != declared:
        raise ValueError("effect variants differ from the frozen config")
    expected_search = _expected_search(config)
    for variant in declared:
        section = variants[variant]
        trials = section.get("trials")
        if not isinstance(trials, list) or len(trials) != len(expected_search):
            raise ValueError(f"effect {variant} does not retain the complete search")
        for index, (trial, expected) in enumerate(zip(trials, expected_search, strict=True)):
            if trial.get("trial") != index or trial.get("hyperparameters") != expected:
                raise ValueError(f"effect {variant} trial grid differs")
            if int(trial["parameter_count"]) > int(config["model_search"]["maximum_parameter_count"]):
                raise ValueError(f"effect {variant} exceeds the parameter cap")
            _validate_trial_history(
                trial, effect=True, route_only=variant == "route_only"
            )
        def rank(row: Mapping[str, Any]) -> tuple[float, float, int, int]:
            factor = row["development_metrics"]["macro_supported_factor_brier"]
            return (
                float(row["development_metrics"]["route_nll"]),
                float("inf") if factor is None or variant == "route_only" else float(factor),
                int(row["parameter_count"]),
                int(row["trial"]),
            )
        selected = min(trials, key=rank)
        if section.get("selected_trial") != selected["trial"]:
            raise ValueError(f"effect {variant} selected trial differs")
        _, checkpoint = _load_checkpoint(
            section.get("checkpoint"),
            repository_root=repository_root,
            name=f"effect {variant}",
            schema="piu.action-effect-checkpoint.v1",
        )
        if set(checkpoint) != {
            "schema_version",
            "variant",
            "model",
            "action_vocabulary",
            "model_state",
            "objective_state",
        }:
            raise ValueError(f"effect {variant} checkpoint fields are not closed")
        if checkpoint.get("variant") != variant or checkpoint.get("action_vocabulary") != [
            str(value) for value in config["inputs"]["action_vocabulary"]
        ]:
            raise ValueError(f"effect {variant} checkpoint identity differs")
        model, objective = _effect_model(checkpoint)
        expected_model = {
            "belief_width": int(train.belief_token.shape[-1]),
            "vlm_width": int(train.candidate_prompt_tokens.shape[-1]),
            "maximum_action_types": len(config["inputs"]["action_vocabulary"]),
            "model_width": int(selected["hyperparameters"]["model_width"]),
            "num_heads": int(selected["hyperparameters"]["num_heads"]),
            "dropout": float(selected["hyperparameters"]["dropout"]),
        }
        if checkpoint.get("model") != expected_model:
            raise ValueError(f"effect {variant} model config differs from selected trial")
        parameter_count = sum(
            parameter.numel()
            for parameter in itertools.chain(model.parameters(), objective.parameters())
        )
        if parameter_count != selected["parameter_count"]:
            raise ValueError(f"effect {variant} parameter count differs")
        outputs, batch = _effect_forward(model, development, variant)
        _, arrays = _load_referenced_npz(
            section.get("development_predictions"),
            repository_root=repository_root,
            name=f"effect {variant} development predictions",
        )
        _validate_effect_arrays(arrays, development, outputs, training=True)
        metrics = effect_probability_metrics(outputs, batch)
        expected_metrics = {key: value for key, value in metrics.items() if key != "raw"}
        _same_json(selected["development_metrics"], expected_metrics, name=f"effect {variant} metrics")
    if {"route_only", "joint_effect"} <= set(variants):
        expected_difference = float(
            variants["joint_effect"]["trials"][variants["joint_effect"]["selected_trial"]][
                "development_metrics"
            ]["route_nll"]
        ) - float(
            variants["route_only"]["trials"][variants["route_only"]["selected_trial"]][
                "development_metrics"
            ]["route_nll"]
        )
        if not math.isclose(
            float(report["development_effect_ablation"]["joint_minus_route_only_route_nll"]),
            expected_difference,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError("effect ablation comparison differs from selected models")
    return report


def validate_effect_prediction_report(
    path: Path, *, repository_root: Path
) -> dict[str, Any]:
    report = _load_json(path, "piu.action-effect-predictions.v1")
    if set(report.get("inputs", {})) != {
        "checkpoint",
        "training_report",
        "features",
        "feature_report",
        "binding_predictions",
        "binding_report",
        "labels",
    }:
        raise ValueError("effect prediction inputs are not field-closed")
    training_path = _input_path(report, "training_report", repository_root=repository_root)
    training = validate_effect_training_report(training_path, repository_root=repository_root)
    variant = str(report.get("variant"))
    if variant not in training["variants"]:
        raise ValueError("effect prediction variant is not frozen")
    _, checkpoint = _load_checkpoint(
        report["inputs"]["checkpoint"],
        repository_root=repository_root,
        name="effect prediction",
        schema="piu.action-effect-checkpoint.v1",
    )
    if report["inputs"]["checkpoint"]["sha256"] != training["variants"][variant]["checkpoint"]["sha256"]:
        raise ValueError("effect prediction uses another checkpoint")
    feature_values, _ = _reported_npz(
        report,
        "features",
        "feature_report",
        "piu.spatial-prefix-features.v1",
        repository_root=repository_root,
    )
    binding_values, _ = _reported_npz(
        report,
        "binding_predictions",
        "binding_report",
        "piu.target-binder-online-predictions.v1",
        repository_root=repository_root,
    )
    bundle = join_effect_features(
        feature_arrays=feature_values,
        binding_predictions=binding_values,
        labels=load_effect_labels(_input_path(report, "labels", repository_root=repository_root)),
        action_vocabulary=checkpoint["action_vocabulary"],
    )
    model, _ = _effect_model(checkpoint)
    outputs, _ = _effect_forward(model, bundle, variant)
    _, arrays = _load_referenced_npz(
        report.get("output"), repository_root=repository_root, name="effect predictions"
    )
    _validate_effect_arrays(arrays, bundle, outputs, training=False)
    if report.get("initial_state_groups") != sorted(set(bundle.initial_state_group)):
        raise ValueError("effect prediction groups differ from its source")
    if set(arrays["split"].astype(str)) != {str(report.get("split"))}:
        raise ValueError("effect prediction split differs from its report")
    return report


def validate_binder_calibration_artifact(
    path: Path, *, repository_root: Path
) -> dict[str, Any]:
    value = _load_json(path, "piu.target-binder-calibration.v1")
    if set(value.get("inputs", {})) != {
        "temperature_predictions",
        "temperature_report",
        "conformal_predictions",
        "conformal_report",
    }:
        raise ValueError("binder calibration inputs are not field-closed")
    config_path = _reference_path(
        value.get("config"), repository_root=repository_root, name="binder calibration config"
    )
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != "piu.binding-calibration-experiment.v1":
        raise ValueError("unsupported binder calibration config")
    input_paths = {
        name: _input_path(value, name, repository_root=repository_root)
        for name in (
            "temperature_predictions",
            "temperature_report",
            "conformal_predictions",
            "conformal_report",
        )
    }
    temperature_report = validate_binder_prediction_report(
        input_paths["temperature_report"], repository_root=repository_root
    )
    conformal_report = validate_binder_prediction_report(
        input_paths["conformal_report"], repository_root=repository_root
    )
    temperature = _load_npz(input_paths["temperature_predictions"])
    conformal = _load_npz(input_paths["conformal_predictions"])
    if temperature_report["output"]["sha256"] != _sha256(input_paths["temperature_predictions"]) or conformal_report[
        "output"
    ]["sha256"] != _sha256(input_paths["conformal_predictions"]):
        raise ValueError("binder calibration predictions differ from reports")
    temperature_groups = set(temperature["initial_state_group"].astype(str))
    conformal_groups = set(conformal["initial_state_group"].astype(str))
    if temperature_groups & conformal_groups:
        raise ValueError("binder calibration roles overlap")
    risks = [float(item) for item in config["risk_contract"]["reported_alpha"]]
    fitted = fit_binding_calibration(
        temperature_values=temperature, conformal_values=conformal, alphas=risks
    )
    for name, expected in fitted.items():
        _same_json(value.get(name), expected, name=f"binder calibration.{name}")
    expected_groups = {
        "temperature": sorted(temperature_groups),
        "conformal": sorted(conformal_groups),
    }
    if (
        value.get("initial_state_groups") != expected_groups
        or value.get("checkpoint_sha256") != temperature_report["inputs"]["checkpoint"]["sha256"]
        or value.get("checkpoint_sha256") != conformal_report["inputs"]["checkpoint"]["sha256"]
        or value.get("risk_contract") != config["risk_contract"]
        or float(value.get("primary_alpha")) != float(config["risk_contract"]["primary_alpha"])
    ):
        raise ValueError("binder calibration provenance differs from exact inputs")
    return value


def validate_effect_calibration_artifact(
    path: Path, *, repository_root: Path
) -> dict[str, Any]:
    value = _load_json(path, "piu.action-effect-calibration.v1")
    if set(value.get("inputs", {})) != {
        "temperature_predictions",
        "temperature_report",
        "conformal_predictions",
        "conformal_report",
    }:
        raise ValueError("effect calibration inputs are not field-closed")
    config_path = _reference_path(
        value.get("config"), repository_root=repository_root, name="effect calibration config"
    )
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != "piu.action-effect-calibration-experiment.v1":
        raise ValueError("unsupported effect calibration config")
    paths = {
        name: _input_path(value, name, repository_root=repository_root)
        for name in (
            "temperature_predictions",
            "temperature_report",
            "conformal_predictions",
            "conformal_report",
        )
    }
    temperature_report = validate_effect_prediction_report(
        paths["temperature_report"], repository_root=repository_root
    )
    conformal_report = validate_effect_prediction_report(
        paths["conformal_report"], repository_root=repository_root
    )
    temperature = _load_npz(paths["temperature_predictions"])
    conformal = _load_npz(paths["conformal_predictions"])
    if temperature_report["output"]["sha256"] != _sha256(paths["temperature_predictions"]) or conformal_report[
        "output"
    ]["sha256"] != _sha256(paths["conformal_predictions"]):
        raise ValueError("effect calibration predictions differ from reports")
    groups = {
        "temperature": sorted(set(temperature["initial_state_group"].astype(str))),
        "conformal": sorted(set(conformal["initial_state_group"].astype(str))),
    }
    if set(groups["temperature"]) & set(groups["conformal"]):
        raise ValueError("effect calibration roles overlap")
    risks = [float(item) for item in config["risk_contract"]["reported_alpha"]]
    fitted = fit_effect_calibration(
        temperature_values=temperature, conformal_values=conformal, alphas=risks
    )
    for name, expected in fitted.items():
        _same_json(value.get(name), expected, name=f"effect calibration.{name}")
    if (
        value.get("initial_state_groups") != groups
        or value.get("variant") != temperature_report["variant"]
        or value.get("variant") != conformal_report["variant"]
        or value.get("checkpoint_sha256") != temperature_report["inputs"]["checkpoint"]["sha256"]
        or value.get("checkpoint_sha256") != conformal_report["inputs"]["checkpoint"]["sha256"]
        or float(value.get("primary_alpha")) != float(config["risk_contract"]["primary_alpha"])
        or value.get("controller_contract") != config["controller_contract"]
    ):
        raise ValueError("effect calibration provenance differs from exact inputs")
    return value
