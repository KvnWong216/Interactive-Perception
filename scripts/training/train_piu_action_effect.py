#!/usr/bin/env python3
"""Train and retain route/effect ablations on CPU model-selection groups."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.action_effect import (
    CandidateConditionedEffectPredictor,
    LearnedEffectObjective,
    join_effect_features,
    load_effect_labels,
)
from piu.contracts import Split
from piu.effect_training import EffectHyperparameters, train_effect_predictor


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_npz_with_report(
    path: Path, report_path: Path, *, report_schema: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    report = json.loads(report_path.read_text())
    if report.get("schema_version") != report_schema:
        raise ValueError(f"unsupported input report {report_schema}")
    if sha256(path) != report["output"]["sha256"]:
        raise ValueError("input artifact differs from its report")
    with np.load(path) as store:
        return {name: np.asarray(store[name]) for name in store.files}, report


def load_bundle(
    *,
    features: Path,
    feature_report: Path,
    binding_predictions: Path,
    binding_report: Path,
    labels: Path,
    action_vocabulary: list[str],
):
    feature_values, _ = load_npz_with_report(
        features,
        feature_report,
        report_schema="piu.spatial-prefix-features.v1",
    )
    binding_values, _ = load_npz_with_report(
        binding_predictions,
        binding_report,
        report_schema="piu.target-binder-online-predictions.v1",
    )
    return join_effect_features(
        feature_arrays=feature_values,
        binding_predictions=binding_values,
        labels=load_effect_labels(labels),
        action_vocabulary=action_vocabulary,
    )


def expand_search(config: dict[str, Any]) -> list[EffectHyperparameters]:
    search = config["model_search"]
    return [
        EffectHyperparameters(
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
    for prefix in ("train", "development"):
        parser.add_argument(f"--{prefix}-features", type=Path, required=True)
        parser.add_argument(f"--{prefix}-feature-report", type=Path, required=True)
        parser.add_argument(
            f"--{prefix}-binding-predictions", type=Path, required=True
        )
        parser.add_argument(f"--{prefix}-binding-report", type=Path, required=True)
        parser.add_argument(f"--{prefix}-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    path_names = ["config", "output_dir"] + [
        f"{prefix}_{suffix}"
        for prefix in ("train", "development")
        for suffix in (
            "features",
            "feature_report",
            "binding_predictions",
            "binding_report",
            "labels",
        )
    ]
    for name in path_names:
        setattr(args, name, resolve(getattr(args, name)))
    if args.output_dir.exists():
        raise FileExistsError("action-effect training output directory is immutable")
    config = yaml.safe_load(args.config.read_text())
    if config.get("schema_version") != "piu.action-effect-experiment.v1":
        raise ValueError("unsupported action-effect training config")
    if config["compute"]["device"] != "cpu":
        raise ValueError("local effect trainer must use CPU")
    if config["objectives"].get("manual_loss_weights") is not None:
        raise ValueError("manual action-effect loss weights are prohibited")
    action_vocabulary = [str(value) for value in config["inputs"]["action_vocabulary"]]
    train = load_bundle(
        features=args.train_features,
        feature_report=args.train_feature_report,
        binding_predictions=args.train_binding_predictions,
        binding_report=args.train_binding_report,
        labels=args.train_labels,
        action_vocabulary=action_vocabulary,
    )
    development = load_bundle(
        features=args.development_features,
        feature_report=args.development_feature_report,
        binding_predictions=args.development_binding_predictions,
        binding_report=args.development_binding_report,
        labels=args.development_labels,
        action_vocabulary=action_vocabulary,
    )
    if set(train.split) != {Split.TRAIN} or set(development.split) != {
        Split.DEVELOPMENT
    }:
        raise ValueError("effect trainer accepts only train and development splits")
    if set(train.initial_state_group) & set(development.initial_state_group):
        raise ValueError("effect train/development groups overlap")

    import torch

    if torch.cuda.is_available():
        raise RuntimeError("CUDA remained visible after effect trainer isolation")
    torch.set_num_threads(int(config["compute"]["torch_threads"]))
    input_model = {
        "belief_width": int(train.belief_token.shape[-1]),
        "vlm_width": int(train.candidate_prompt_tokens.shape[-1]),
        "maximum_action_types": len(action_vocabulary),
    }
    variants = {}
    hyperparameters = expand_search(config)
    for variant in config["variants"]:
        trials = []
        best_rank = None
        best = None
        best_parameters = None
        for trial_index, parameters in enumerate(hyperparameters):
            torch.manual_seed(parameters.seed)
            model_config = {
                **input_model,
                "model_width": parameters.model_width,
                "num_heads": parameters.num_heads,
                "dropout": parameters.dropout,
            }
            model = CandidateConditionedEffectPredictor(**model_config)
            objective = LearnedEffectObjective()
            parameter_count = sum(
                parameter.numel()
                for parameter in itertools.chain(
                    model.parameters(), objective.parameters()
                )
            )
            if parameter_count > int(config["model_search"]["maximum_parameter_count"]):
                raise ValueError("effect trial exceeds declared parameter budget")
            result = train_effect_predictor(
                model=model,
                objective=objective,
                train=train,
                development=development,
                hyperparameters=parameters,
                variant=str(variant),
            )
            metrics = result["development_metrics"]
            factor_brier = metrics["macro_supported_factor_brier"]
            rank = (
                float(metrics["route_nll"]),
                (
                    float("inf")
                    if factor_brier is None or variant == "route_only"
                    else float(factor_brier)
                ),
                parameter_count,
                trial_index,
            )
            trials.append(
                {
                    "trial": trial_index,
                    "hyperparameters": parameters.__dict__,
                    "parameter_count": parameter_count,
                    "best_epoch": result["best_epoch"],
                    "development_metrics": metrics,
                    "history": result["history"],
                }
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best = result
                best_parameters = parameters
        if best is None or best_parameters is None or best_rank is None:
            raise RuntimeError("effect model search produced no trial")
        variants[str(variant)] = {
            "trials": trials,
            "selected_trial": int(best_rank[-1]),
            "selected_hyperparameters": best_parameters,
            "result": best,
        }
    args.output_dir.mkdir(parents=True)
    report_variants = {}
    for variant, selected in variants.items():
        parameters = selected["selected_hyperparameters"]
        checkpoint_path = args.output_dir / f"{variant}.pt"
        predictions_path = args.output_dir / f"{variant}_development_predictions.npz"
        checkpoint = {
            "schema_version": "piu.action-effect-checkpoint.v1",
            "variant": variant,
            "model": {
                **input_model,
                "model_width": parameters.model_width,
                "num_heads": parameters.num_heads,
                "dropout": parameters.dropout,
            },
            "action_vocabulary": action_vocabulary,
            "model_state": selected["result"]["model_state"],
            "objective_state": selected["result"]["objective_state"],
        }
        torch.save(checkpoint, checkpoint_path)
        raw = selected["result"]["raw_development_predictions"]
        np.savez_compressed(
            predictions_path,
            sample_id=np.asarray(development.sample_id),
            initial_state_group=np.asarray(development.initial_state_group),
            split=np.asarray([value.value for value in development.split]),
            candidate_valid_mask=development.candidate_valid_mask,
            route_logits=raw["route_logits"],
            route_target=development.route_target,
            factor_logits=raw["factor_logits"],
            factor_target=development.effect_target,
            factor_support_mask=development.effect_support_mask,
        )
        report_variants[variant] = {
            "trials": selected["trials"],
            "selected_trial": selected["selected_trial"],
            "checkpoint": {
                "path": portable(checkpoint_path),
                "sha256": sha256(checkpoint_path),
            },
            "development_predictions": {
                "path": portable(predictions_path),
                "sha256": sha256(predictions_path),
            },
        }
    comparison = None
    if {"route_only", "joint_effect"} <= set(report_variants):
        route_only = variants["route_only"]["result"]["development_metrics"]
        joint = variants["joint_effect"]["result"]["development_metrics"]
        comparison = {
            "joint_minus_route_only_route_nll": float(joint["route_nll"])
            - float(route_only["route_nll"]),
            "positive_effect_claim_allowed": False,
            "reason": "development ablation requires later sealed confirmation",
        }
    input_paths = {
        name: getattr(args, name)
        for name in path_names
        if name not in {"output_dir"}
    }
    report = {
        "schema_version": "piu.action-effect-training.v1",
        "claim_scope": "DEVELOPMENT_ABLATION_NOT_TEST_EVIDENCE",
        "inputs": {
            name: {"path": portable(path), "sha256": sha256(path)}
            for name, path in input_paths.items()
        },
        "initial_state_groups": {
            "train": sorted(set(train.initial_state_group)),
            "development": sorted(set(development.initial_state_group)),
        },
        "variants": report_variants,
        "development_effect_ablation": comparison,
        "cuda_visible_to_trainer": torch.cuda.is_available(),
        "calibration_loaded": False,
        "sealed_test_loaded": False,
        "paper_method_claim_allowed": False,
    }
    report_path = args.output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
