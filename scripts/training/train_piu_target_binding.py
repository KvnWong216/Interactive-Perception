#!/usr/bin/env python3
"""Search and freeze the lightweight PIU spatial binder on CPU."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

# This trainer operates only on already-extracted frozen features.  Override,
# rather than inherit, a caller's CUDA visibility so an accidental device move
# cannot compete with the externally managed GPU processes.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.binding_data import join_binding_features, load_binding_labels
from piu.binding_training import BinderHyperparameters, train_binder
from piu.contracts import Split
from piu.target_binding import (
    LearnedMultiTaskObjective,
    PromptConditionedTargetBinder,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def load_bundle(
    *,
    feature_path: Path,
    report_path: Path,
    label_path: Path,
    action_vocabulary: list[str],
):
    report = json.loads(report_path.read_text())
    if report.get("schema_version") != "piu.spatial-prefix-features.v1":
        raise ValueError("unsupported feature report")
    output = report.get("output", {})
    if sha256(feature_path) != output.get("sha256"):
        raise ValueError("feature cache hash differs from its report")
    with np.load(feature_path) as store:
        arrays = {name: np.asarray(store[name]) for name in store.files}
    labels = load_binding_labels(label_path)
    return join_binding_features(
        feature_arrays=arrays,
        feature_report=report,
        labels=labels,
        action_vocabulary=action_vocabulary,
    )


def require_only_split(values, expected: Split, *, name: str) -> None:
    observed = set(values.split)
    if observed != {expected}:
        raise ValueError(
            f"{name} must contain only {expected.value}, got "
            f"{sorted(value.value for value in observed)}"
        )


def expand_search(config: dict[str, Any]) -> list[BinderHyperparameters]:
    search = config["model_search"]
    return [
        BinderHyperparameters(
            model_width=int(width),
            num_heads=int(heads),
            dropout=float(dropout),
            learning_rate=float(rate),
            epochs=int(search["epochs"]),
            batch_size=int(search["batch_size"]),
            seed=int(seed),
        )
        for width, heads, dropout, rate, seed in itertools.product(
            search["model_width"],
            search["num_heads"],
            search["dropout"],
            search["learning_rate"],
            search["seeds"],
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--train-feature-report", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--development-features", type=Path, required=True)
    parser.add_argument("--development-feature-report", type=Path, required=True)
    parser.add_argument("--development-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "config",
        "train_features",
        "train_feature_report",
        "train_labels",
        "development_features",
        "development_feature_report",
        "development_labels",
        "output",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    report_path = args.output.with_suffix(".json")
    predictions_path = args.output.with_suffix(".development_predictions.npz")
    for path in (args.output, report_path, predictions_path):
        if path.exists():
            raise FileExistsError(f"training outputs are immutable: {path}")
    config = yaml.safe_load(args.config.read_text())
    if config.get("schema_version") != "piu.binding-adapter-experiment.v1":
        raise ValueError("unsupported binding experiment config")
    if config["compute"]["device"] != "cpu":
        raise ValueError("local binding trainer must use CPU")
    if config["objectives"].get("manual_loss_weights") is not None:
        raise ValueError("manual loss weights are prohibited")
    action_vocabulary = [str(value) for value in config["inputs"]["action_vocabulary"]]
    train = load_bundle(
        feature_path=args.train_features,
        report_path=args.train_feature_report,
        label_path=args.train_labels,
        action_vocabulary=action_vocabulary,
    )
    development = load_bundle(
        feature_path=args.development_features,
        report_path=args.development_feature_report,
        label_path=args.development_labels,
        action_vocabulary=action_vocabulary,
    )
    require_only_split(train, Split.TRAIN, name="training data")
    require_only_split(development, Split.DEVELOPMENT, name="development data")
    if set(train.initial_state_group) & set(development.initial_state_group):
        raise ValueError("train/development initial-state groups overlap")

    import torch

    if torch.cuda.is_available():
        raise RuntimeError("CUDA remained visible after CPU trainer isolation")
    torch.set_num_threads(int(config["compute"]["torch_threads"]))
    trials = []
    best_rank = None
    best_result = None
    best_hyperparameters = None
    input_width = int(train.image_tokens.shape[-1])
    maximum_cameras = int(max(train.camera_id.max(), development.camera_id.max())) + 1
    maximum_time_steps = (
        int(max(train.temporal_id.max(), development.temporal_id.max())) + 1
    )
    for trial_index, hyperparameters in enumerate(expand_search(config)):
        torch.manual_seed(hyperparameters.seed)
        model = PromptConditionedTargetBinder(
            vlm_width=input_width,
            model_width=hyperparameters.model_width,
            num_heads=hyperparameters.num_heads,
            maximum_cameras=maximum_cameras,
            maximum_time_steps=maximum_time_steps,
            maximum_action_types=len(action_vocabulary),
            dropout=hyperparameters.dropout,
        )
        objective = LearnedMultiTaskObjective()
        parameter_count = sum(
            parameter.numel()
            for parameter in itertools.chain(model.parameters(), objective.parameters())
        )
        if parameter_count > int(config["model_search"]["maximum_parameter_count"]):
            raise ValueError("binder trial exceeds the declared parameter budget")
        result = train_binder(
            model=model,
            objective=objective,
            train=train,
            development=development,
            hyperparameters=hyperparameters,
        )
        metrics = result["development_metrics"]
        rank = (
            float(metrics["spatial_nll"]),
            float(metrics["presence_brier"]),
            parameter_count,
            trial_index,
        )
        trials.append(
            {
                "trial": trial_index,
                "hyperparameters": hyperparameters.__dict__,
                "parameter_count": parameter_count,
                "best_epoch": result["best_epoch"],
                "development_metrics": metrics,
                "history": result["history"],
            }
        )
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_result = result
            best_hyperparameters = hyperparameters
    if best_result is None or best_hyperparameters is None:
        raise RuntimeError("model search produced no trial")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": "piu.target-binder-checkpoint.v1",
        "model": {
            "vlm_width": input_width,
            "model_width": best_hyperparameters.model_width,
            "num_heads": best_hyperparameters.num_heads,
            "maximum_cameras": maximum_cameras,
            "maximum_time_steps": maximum_time_steps,
            "maximum_action_types": len(action_vocabulary),
            "dropout": best_hyperparameters.dropout,
        },
        "action_vocabulary": action_vocabulary,
        "model_state": best_result["model_state"],
        "objective_state": best_result["objective_state"],
    }
    torch.save(checkpoint, args.output)
    raw = best_result["raw_development_predictions"]
    np.savez_compressed(
        predictions_path,
        sample_id=np.asarray(development.sample_id),
        initial_state_group=np.asarray(development.initial_state_group),
        image_valid_mask=development.image_valid_mask,
        spatial_logits=raw["spatial_logits"],
        spatial_attention=raw["spatial_attention"],
        target_present_logit=raw["target_present_logit"],
        task_sufficiency_logit=raw["task_sufficiency_logit"],
        patch_target=development.patch_target,
        target_present=development.target_present,
        task_sufficient=development.task_sufficient,
        task_sufficient_mask=development.task_sufficient_mask,
    )
    report = {
        "schema_version": "piu.target-binder-training.v1",
        "claim_scope": "DEVELOPMENT_MODEL_SELECTION_NOT_TEST_EVIDENCE",
        "config": {"path": portable(args.config), "sha256": sha256(args.config)},
        "inputs": {
            name: {"path": portable(path), "sha256": sha256(path)}
            for name, path in {
                "train_features": args.train_features,
                "train_feature_report": args.train_feature_report,
                "train_labels": args.train_labels,
                "development_features": args.development_features,
                "development_feature_report": args.development_feature_report,
                "development_labels": args.development_labels,
            }.items()
        },
        "train_groups": len(set(train.initial_state_group)),
        "development_groups": len(set(development.initial_state_group)),
        "initial_state_groups": {
            "train": sorted(set(train.initial_state_group)),
            "development": sorted(set(development.initial_state_group)),
        },
        "selection_order": config["model_search"]["selection_order"],
        "trials": trials,
        "selected_trial": int(best_rank[-1]),
        "checkpoint": {"path": portable(args.output), "sha256": sha256(args.output)},
        "development_predictions": {
            "path": portable(predictions_path),
            "sha256": sha256(predictions_path),
        },
        "cuda_visible_to_trainer": torch.cuda.is_available(),
        "sealed_test_loaded": False,
        "calibration_loaded": False,
        "paper_method_claim_allowed": False,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
