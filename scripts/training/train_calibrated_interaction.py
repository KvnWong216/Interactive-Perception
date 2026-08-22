#!/usr/bin/env python3
"""Train B6/B7 heads, calibrate on an isolated split, and score held-out data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from calibrated_interaction.calibration import (
    BinaryEffectCalibration,
    LACCalibrator,
    TemperatureScaler,
)
from calibrated_interaction.contracts import (
    CandidateAction,
    EffectFactor,
    validate_candidate_set,
)
from calibrated_interaction.controller import (
    CalibratedSelector,
    DecisionKind,
    ModelPrediction,
)
from calibrated_interaction.model import (
    CandidateInteractionDecoder,
    interaction_loss,
)

ROUTE_MAP = {
    "DIRECT_ACT": "direct_requested_to_basket",
    "OPEN_TO_INSPECT": "open_middle_drawer",
    "ABSTAIN": "stop_unsupported",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def keyed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["sample_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate sample_id")
    return result


def load_candidates(path: Path) -> tuple[CandidateAction, ...]:
    value = yaml.safe_load(path.read_text())
    return validate_candidate_set(
        [CandidateAction.from_mapping(row) for row in value["candidates"]]
    )


def load_effect_proxy(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if value.get("schema_version") != (
        "calibrated-interaction.effect-supervision-proxy.v1"
    ):
        raise ValueError("unsupported effect proxy schema")
    for source in value["sources"].values():
        report = ROOT / source["report"]
        if not report.is_file():
            raise FileNotFoundError(report)
    return value


def load_split(
    snapshots_path: Path,
    labels_path: Path,
    features_path: Path,
    candidates: tuple[CandidateAction, ...],
    effect_proxy: dict[str, Any],
) -> dict[str, Any]:
    snapshots = keyed(read_jsonl(snapshots_path))
    labels = keyed(read_jsonl(labels_path))
    if set(snapshots) != set(labels):
        raise ValueError("snapshot and label IDs differ")
    store = np.load(features_path)
    identifiers = store["sample_id"].astype(str).tolist()
    if set(identifiers) != set(snapshots) or len(identifiers) != len(snapshots):
        raise ValueError("feature and snapshot IDs differ")
    candidate_ids = tuple(store["candidate_id"].astype(str).tolist())
    expected_ids = tuple(candidate.candidate_id for candidate in candidates)
    if candidate_ids != expected_ids:
        raise ValueError("feature candidate order differs from candidate config")

    context = np.asarray(store["context_tokens"], dtype=np.float32)
    candidate_values = np.asarray(store["candidate_tokens"], dtype=np.float32)
    if context.shape != (len(identifiers), 4, 2048):
        raise ValueError("unexpected context feature shape")
    if candidate_values.shape != (len(identifiers), len(candidates), 2048):
        raise ValueError("unexpected candidate feature shape")

    route = np.asarray(
        [
            candidate_ids.index(ROUTE_MAP[str(labels[sample_id]["preferred_action"])])
            for sample_id in identifiers
        ],
        dtype=np.int64,
    )
    effect_labels = np.zeros(
        (len(identifiers), len(candidates), len(EffectFactor)), dtype=np.float32
    )
    effect_mask = np.zeros_like(effect_labels, dtype=bool)
    for row_index, sample_id in enumerate(identifiers):
        location = str(labels[sample_id]["target_location"])
        for candidate_id, values in effect_proxy["labels"].get(location, {}).items():
            candidate_index = candidate_ids.index(candidate_id)
            for factor_index, factor in enumerate(EffectFactor):
                if factor.value in values:
                    effect_labels[row_index, candidate_index, factor_index] = float(
                        values[factor.value]
                    )
                    effect_mask[row_index, candidate_index, factor_index] = True
    return {
        "ids": identifiers,
        "seed": np.asarray(store["seed"], dtype=np.int64),
        "context": context,
        "candidates": candidate_values,
        "route": route,
        "effects": effect_labels,
        "effect_mask": effect_mask,
        "candidate_ids": candidate_ids,
        "snapshots": snapshots,
        "labels": labels,
    }


def tensors(split: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "context": torch.from_numpy(split["context"]),
        "candidates": torch.from_numpy(split["candidates"]),
        "route": torch.from_numpy(split["route"]),
        "effects": torch.from_numpy(split["effects"]),
        "effect_mask": torch.from_numpy(split["effect_mask"]),
    }


def forward_numpy(
    model: CandidateInteractionDecoder, split: dict[str, Any]
) -> dict[str, np.ndarray]:
    values = tensors(split)
    model.eval()
    with torch.no_grad():
        output = model(values["context"], values["candidates"])
    return {key: value.detach().cpu().numpy() for key, value in output.items()}


def effect_bce(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    selected_logits = logits[mask]
    selected_labels = labels[mask]
    if not len(selected_logits):
        return float("nan")
    losses = np.maximum(selected_logits, 0) - selected_logits * selected_labels
    losses += np.log1p(np.exp(-np.abs(selected_logits)))
    return float(losses.mean())


def macro_f1(predicted: np.ndarray, truth: np.ndarray, classes: int) -> float:
    scores = []
    for index in range(classes):
        tp = int(np.sum((predicted == index) & (truth == index)))
        fp = int(np.sum((predicted == index) & (truth != index)))
        fn = int(np.sum((predicted != index) & (truth == index)))
        if tp + fp + fn == 0:
            continue
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(scores)) if scores else 0.0


def expected_calibration_error(
    probabilities: np.ndarray, truth: np.ndarray, bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == truth
    result = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            result += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return result


def route_metrics(logits: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    one_hot = np.eye(probabilities.shape[1])[truth]
    direct = 0
    return {
        "accuracy": float(np.mean(predicted == truth)),
        "macro_f1": macro_f1(predicted, truth, probabilities.shape[1]),
        "nll": float(
            -np.log(
                np.clip(probabilities[np.arange(len(truth)), truth], 1e-12, 1)
            ).mean()
        ),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": expected_calibration_error(probabilities, truth),
        "false_direct_rate": float(np.mean((predicted == direct) & (truth != direct))),
    }


def paired_route_accuracy(
    logits: np.ndarray, truth: np.ndarray, seeds: np.ndarray
) -> float:
    """Require both prompt-counterfactual routes for a scene seed to be right."""

    predicted = logits.argmax(axis=1)
    paired = []
    for seed in sorted(set(seeds.tolist())):
        indices = np.flatnonzero(seeds == seed)
        if len(indices) != 2:
            raise ValueError(f"seed {seed} does not contain exactly one prompt pair")
        paired.append(bool(np.all(predicted[indices] == truth[indices])))
    return float(np.mean(paired))


def paired_feature_ablation(split: dict[str, Any], *, mode: str) -> dict[str, Any]:
    """Alter only prompt-conditioned features inside identical-RGB seed pairs."""

    if mode not in {"pair_average", "prompt_swap"}:
        raise ValueError(f"unsupported paired feature ablation: {mode}")
    result = dict(split)
    result["context"] = split["context"].copy()
    result["candidates"] = split["candidates"].copy()
    for seed in sorted(set(split["seed"].tolist())):
        indices = np.flatnonzero(split["seed"] == seed)
        if len(indices) != 2:
            raise ValueError(f"seed {seed} does not contain exactly one prompt pair")
        if mode == "pair_average":
            result["context"][indices] = split["context"][indices].mean(
                axis=0, keepdims=True
            )
            result["candidates"][indices] = split["candidates"][indices].mean(
                axis=0, keepdims=True
            )
        else:
            result["context"][indices] = split["context"][indices[::-1]]
            result["candidates"][indices] = split["candidates"][indices[::-1]]
    return result


def route_ablation_metrics(
    model: CandidateInteractionDecoder, split: dict[str, Any]
) -> dict[str, dict[str, float]]:
    results = {}
    for name, value in (
        ("unaltered", split),
        ("prompt_pair_average", paired_feature_ablation(split, mode="pair_average")),
        ("prompt_swap", paired_feature_ablation(split, mode="prompt_swap")),
    ):
        output = forward_numpy(model, value)
        metrics = route_metrics(output["route_logits"], value["route"])
        metrics["paired_seed_accuracy"] = paired_route_accuracy(
            output["route_logits"], value["route"], value["seed"]
        )
        results[name] = metrics
    return results


def shared_parameters(model: CandidateInteractionDecoder) -> list[nn.Parameter]:
    modules = (
        model.context_projection,
        model.candidate_projection,
        model.cross_attention,
        model.attention_norm,
        model.feed_forward,
        model.output_norm,
    )
    return [parameter for module in modules for parameter in module.parameters()]


def gradient_matched_weight(
    model: CandidateInteractionDecoder, split: dict[str, Any]
) -> tuple[float, dict[str, float]]:
    value = tensors(split)
    output = model(value["context"], value["candidates"])
    losses = interaction_loss(
        output,
        route_labels=value["route"],
        effect_labels=value["effects"],
        effect_mask=value["effect_mask"],
        effect_weight=1.0,
    )
    parameters = shared_parameters(model)
    route_gradients = torch.autograd.grad(
        losses["route_loss"], parameters, retain_graph=True, allow_unused=True
    )
    effect_gradients = torch.autograd.grad(
        losses["effect_loss"], parameters, allow_unused=True
    )

    def norm(values: tuple[torch.Tensor | None, ...]) -> float:
        return math.sqrt(
            sum(
                float(value.detach().square().sum())
                for value in values
                if value is not None
            )
        )

    route_norm = norm(route_gradients)
    effect_norm = norm(effect_gradients)
    raw = route_norm / max(effect_norm, 1e-12)
    weight = float(np.clip(raw, 0.05, 5.0))
    return weight, {
        "route_gradient_norm": route_norm,
        "effect_gradient_norm": effect_norm,
        "unclipped_ratio": raw,
        "selected_weight": weight,
    }


def fit(
    train: dict[str, Any],
    development: dict[str, Any],
    *,
    seed: int,
    effect_supervision: bool,
    epochs: int,
) -> tuple[CandidateInteractionDecoder, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CandidateInteractionDecoder(
        vlm_width=2048,
        model_width=128,
        num_heads=4,
        effect_factors=len(EffectFactor),
        dropout=0.1,
    )
    if effect_supervision:
        effect_weight, gradient_report = gradient_matched_weight(model, train)
    else:
        effect_weight = 0.0
        gradient_report = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    train_values = tensors(train)
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    best_epoch = 0
    patience = 80
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_values["context"], train_values["candidates"])
        losses = interaction_loss(
            output,
            route_labels=train_values["route"],
            effect_labels=train_values["effects"],
            effect_mask=train_values["effect_mask"],
            effect_weight=effect_weight,
        )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch % 5 == 0 or epoch == epochs - 1:
            dev_output = forward_numpy(model, development)
            dev_route = route_metrics(dev_output["route_logits"], development["route"])
            dev_effect = effect_bce(
                dev_output["effect_logits"],
                development["effects"],
                development["effect_mask"],
            )
            score = dev_route["nll"] + (0.1 * dev_effect if effect_supervision else 0.0)
            if score < best_score - 1e-6:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            if epoch - best_epoch >= patience:
                break
    model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "effect_supervision": effect_supervision,
        "effect_weight": effect_weight,
        "gradient_match": gradient_report,
        "best_epoch": best_epoch,
        "epochs_ran": epoch + 1,
        "development_objective": best_score,
    }


def calibrated_metrics(
    route_logits: np.ndarray,
    effect_logits: np.ndarray,
    split: dict[str, Any],
    *,
    scaler: TemperatureScaler,
    route_calibrator: LACCalibrator,
    effect_calibrator: BinaryEffectCalibration,
    candidates: tuple[CandidateAction, ...],
) -> dict[str, Any]:
    route_probabilities = scaler.probabilities(route_logits)
    effect_probabilities = 1.0 / (1.0 + np.exp(-effect_logits))
    route_sets = [route_calibrator.predict(row) for row in route_probabilities]
    truth_names = [split["candidate_ids"][index] for index in split["route"]]
    covered = [
        truth in values for truth, values in zip(truth_names, route_sets, strict=True)
    ]
    singleton_correct = [
        values == (truth,)
        for truth, values in zip(truth_names, route_sets, strict=True)
    ]
    decisions = []
    selector = CalibratedSelector(route_calibrator, effect_calibrator)
    for index in range(len(route_logits)):
        prediction = ModelPrediction(
            candidate_ids=split["candidate_ids"],
            route_probabilities={
                candidate_id: float(route_probabilities[index, candidate_index])
                for candidate_index, candidate_id in enumerate(split["candidate_ids"])
            },
            effect_positive_probabilities={
                candidate_id: {
                    factor: float(
                        effect_probabilities[index, candidate_index, factor_index]
                    )
                    for factor_index, factor in enumerate(EffectFactor)
                }
                for candidate_index, candidate_id in enumerate(split["candidate_ids"])
            },
        )
        decisions.append(selector.select(candidates, prediction))
    action_correct = [
        decision.candidate_id == truth
        for decision, truth in zip(decisions, truth_names, strict=True)
    ]
    wrong_execute = [
        decision.kind is DecisionKind.EXECUTE and decision.candidate_id != truth
        for decision, truth in zip(decisions, truth_names, strict=True)
    ]
    paired = {}
    for seed in sorted(set(split["seed"].tolist())):
        indices = np.flatnonzero(split["seed"] == seed)
        paired[str(seed)] = bool(all(action_correct[index] for index in indices))
    return {
        "coverage": float(np.mean(covered)),
        "mean_set_size": float(np.mean([len(values) for values in route_sets])),
        "singleton_precision": float(
            np.mean(
                [
                    singleton_correct[index]
                    for index, values in enumerate(route_sets)
                    if len(values) == 1
                ]
            )
        )
        if any(len(values) == 1 for values in route_sets)
        else None,
        "abstention_rate": float(
            np.mean([decision.kind is DecisionKind.ABSTAIN for decision in decisions])
        ),
        "correct_execute_rate": float(np.mean(action_correct)),
        "wrong_execute_rate": float(np.mean(wrong_execute)),
        "decision_counts": dict(Counter(decision.kind.value for decision in decisions)),
        "paired_seed_accuracy": float(np.mean(list(paired.values()))),
        "paired_seed_results": paired,
        "route_sets": [list(values) for values in route_sets],
        "decisions": [decision.to_dict() for decision in decisions],
    }


def summarize(values: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": float(np.mean([value[key] for value in values])),
            "std": float(np.std([value[key] for value in values])),
        }
        for key in values[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--effect-proxy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    for name in (
        "data_root",
        "feature_root",
        "candidates",
        "effect_proxy",
        "output",
        "model",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output.exists() or args.model.exists():
        raise FileExistsError("training outputs are immutable")
    torch.set_num_threads(8)
    candidates = load_candidates(args.candidates)
    effect_proxy = load_effect_proxy(args.effect_proxy)
    splits = {
        split: load_split(
            args.data_root / f"{split}.jsonl",
            args.data_root / f"{split}_labels.jsonl",
            args.feature_root / split / "shared_vlm_features.npz",
            candidates,
            effect_proxy,
        )
        for split in ("train", "development", "calibration", "test")
    }
    seed_owners: dict[int, str] = {}
    for split, value in splits.items():
        for seed in set(value["seed"].tolist()):
            if seed in seed_owners:
                raise ValueError(f"seed {seed} leaks across splits")
            seed_owners[seed] = split

    seeds = (20260822, 20260823, 20260824)
    variants = {}
    selected_models = {}
    for name, effect_supervision in (
        ("B6_route_only", False),
        ("B7_effect_route", True),
    ):
        runs = []
        models = []
        for seed in seeds:
            model, training = fit(
                splits["train"],
                splits["development"],
                seed=seed,
                effect_supervision=effect_supervision,
                epochs=args.epochs,
            )
            dev_output = forward_numpy(model, splits["development"])
            test_output = forward_numpy(model, splits["test"])
            training["development"] = route_metrics(
                dev_output["route_logits"], splits["development"]["route"]
            )
            training["test"] = route_metrics(
                test_output["route_logits"], splits["test"]["route"]
            )
            training["test_effect_bce_proxy"] = effect_bce(
                test_output["effect_logits"],
                splits["test"]["effects"],
                splits["test"]["effect_mask"],
            )
            runs.append(training)
            models.append(model)
        selected = min(
            range(len(runs)), key=lambda index: runs[index]["development_objective"]
        )
        selected_models[name] = models[selected]
        variants[name] = {
            "runs": runs,
            "selected_seed": runs[selected]["seed"],
            "test_summary": summarize([run["test"] for run in runs]),
        }

    deployed = selected_models["B7_effect_route"]
    calibration_output = forward_numpy(deployed, splits["calibration"])
    route_scaler = TemperatureScaler.fit(
        calibration_output["route_logits"], splits["calibration"]["route"]
    )
    calibration_probabilities = route_scaler.probabilities(
        calibration_output["route_logits"]
    )
    route_calibrator = LACCalibrator.fit(
        calibration_probabilities,
        splits["calibration"]["route"],
        labels=splits["calibration"]["candidate_ids"],
        alpha=args.alpha,
        split_id="original_drawer_calibration_v1",
    )
    effect_probabilities = 1.0 / (1.0 + np.exp(-calibration_output["effect_logits"]))
    observed_candidates = splits["calibration"]["effect_mask"].all(axis=2)
    effect_calibrator = BinaryEffectCalibration.fit(
        effect_probabilities[observed_candidates],
        splits["calibration"]["effects"][observed_candidates].astype(np.int64),
        joint_alpha=args.alpha,
        split_id="original_drawer_calibration_proxy_v1",
        decision_factors=(
            EffectFactor.EXECUTION_SUCCEEDED,
            EffectFactor.AMBIGUITY_REDUCED,
        ),
    )
    test_output = forward_numpy(deployed, splits["test"])
    calibrated = calibrated_metrics(
        test_output["route_logits"],
        test_output["effect_logits"],
        splits["test"],
        scaler=route_scaler,
        route_calibrator=route_calibrator,
        effect_calibrator=effect_calibrator,
        candidates=candidates,
    )
    ablations = {
        name: route_ablation_metrics(model, splits["test"])
        for name, model in selected_models.items()
    }

    args.model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": deployed.state_dict(),
            "model_config": {
                "vlm_width": 2048,
                "model_width": 128,
                "num_heads": 4,
                "effect_factors": len(EffectFactor),
                "dropout": 0.1,
            },
            "candidate_ids": list(splits["train"]["candidate_ids"]),
            "route_temperature": route_scaler.temperature,
            "route_calibration": route_calibrator.to_dict(),
            "effect_calibration": effect_calibrator.to_dict(),
        },
        args.model,
    )
    report = {
        "schema_version": "calibrated-interaction.original-drawer-pilot.v1",
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "decision": "PILOT_ONLY_NOT_METHOD_EVIDENCE",
        "reason": (
            "route splits and shared-VLM features are real, but effect labels "
            "are repeated capability-level seed-1399 proxies rather than "
            "per-seed executed counterfactual outcomes"
        ),
        "samples": {name: len(value["ids"]) for name, value in splits.items()},
        "seed_groups_disjoint": True,
        "same_rgb_prompt_counterfactual": True,
        "variants": variants,
        "held_out_route_feature_ablations": {
            "description": (
                "Within each same-RGB seed pair, pair_average removes the "
                "task-specific contrast from the cached shared-VLM features; "
                "prompt_swap exchanges those features while preserving labels."
            ),
            "limitation": (
                "This is a feature-level causal diagnostic, not a replacement "
                "for a fresh no-prompt VLM encoding."
            ),
            "variants": ablations,
        },
        "calibration": {
            "temperature": route_scaler.temperature,
            "route": route_calibrator.to_dict(),
            "effects": effect_calibrator.to_dict(),
        },
        "held_out_test_calibrated": calibrated,
        "sources": {
            "candidate_set": artifact(args.candidates),
            "effect_proxy": {
                **artifact(args.effect_proxy),
                "claim_scope": effect_proxy["claim_scope"],
            },
            "model": artifact(args.model),
            "code": {
                "trainer": artifact(Path(__file__).resolve()),
                "decoder": artifact(ROOT / "src/calibrated_interaction/model.py"),
                "calibration": artifact(
                    ROOT / "src/calibrated_interaction/calibration.py"
                ),
                "contracts": artifact(ROOT / "src/calibrated_interaction/contracts.py"),
                "controller": artifact(
                    ROOT / "src/calibrated_interaction/controller.py"
                ),
            },
            "splits": {
                split: {
                    "public_index": artifact(args.data_root / f"{split}.jsonl"),
                    "label_index": artifact(args.data_root / f"{split}_labels.jsonl"),
                    "features": artifact(
                        args.feature_root / split / "shared_vlm_features.npz"
                    ),
                    "feature_manifest": artifact(
                        args.feature_root / split / "shared_vlm_features.json"
                    ),
                }
                for split in splits
            },
        },
        "online_oracle_inputs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    calibrated_summary = {
        key: value
        for key, value in calibrated.items()
        if key not in {"decisions", "route_sets", "paired_seed_results"}
    }
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "variants": {
                    name: value["test_summary"] for name, value in variants.items()
                },
                "calibrated": calibrated_summary,
                "route_feature_ablations": ablations,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
