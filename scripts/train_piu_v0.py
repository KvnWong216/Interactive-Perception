#!/usr/bin/env python3
"""Train PIU V0 belief/effect/ranking heads on frozen public VLM features.

This is a prototype/development run, not a clean validation.  It uses only
policy-visible frozen prefix features as model inputs.  Simulator-derived
drawer/visibility facts are read solely while building offline labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interaction_uncertainty.sidecar import (  # noqa: E402
    ACTION_LABELS,
    LOCATION_LABELS,
    OUTCOME_LABELS,
    build_torch_model,
    fixed_project,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    if scores.size == 0:
        return -1.0
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    rank = min(ordered.size, math.ceil((ordered.size + 1) * (1.0 - alpha)))
    return float(ordered[rank - 1])


def mondrian_thresholds(
    probabilities: np.ndarray,
    truth: np.ndarray,
    labels: tuple[str, ...],
    alpha: float,
) -> dict[str, float]:
    return {
        label: conformal_quantile(
            1.0 - probabilities[truth == index, index], alpha
        )
        for index, label in enumerate(labels)
    }


def prediction_sets(
    probabilities: np.ndarray,
    labels: tuple[str, ...],
    thresholds: dict[str, float],
) -> list[tuple[str, ...]]:
    eligible = tuple(label for label in labels if thresholds[label] >= 0.0)
    results = []
    for row in probabilities:
        selected = tuple(
            label
            for index, label in enumerate(labels)
            if label in eligible and 1.0 - float(row[index]) <= thresholds[label]
        )
        results.append(selected or eligible)
    return results


def metrics(
    probabilities: np.ndarray,
    truth: np.ndarray,
    labels: tuple[str, ...],
    thresholds: dict[str, float],
) -> dict[str, object]:
    predicted = probabilities.argmax(axis=1)
    sets = prediction_sets(probabilities, labels, thresholds)
    coverage = np.asarray(
        [labels[int(label)] in result for label, result in zip(truth, sets, strict=True)]
    )
    singleton = np.asarray(
        [result == (labels[int(label)],) for label, result in zip(truth, sets, strict=True)]
    )
    return {
        "samples": int(truth.size),
        "accuracy": float(np.mean(predicted == truth)),
        "coverage": float(np.mean(coverage)),
        "singleton_correct": float(np.mean(singleton)),
        "mean_set_size": float(np.mean([len(item) for item in sets])),
        "per_class": {
            label: {
                "samples": int(np.sum(truth == index)),
                "accuracy": float(np.mean(predicted[truth == index] == index))
                if np.any(truth == index)
                else None,
                "coverage": float(np.mean(coverage[truth == index]))
                if np.any(truth == index)
                else None,
            }
            for index, label in enumerate(labels)
        },
    }


def v9_outcome(row: dict) -> str:
    evaluator = row["evaluator_only"]
    target_maximum = max(
        max(point["target_pixels"].values())
        for point in evaluator["visibility_history"]
    )
    opened = bool(evaluator["drawer_opened"])
    returned = row["return_status"]["phase"] == "COMPLETE"
    if not row["full_executor"] or not opened or not returned:
        return "FAILED"
    if target_maximum >= 256:
        return "REVEALED"
    if row["intended_outcome"] == "EMPTY":
        counterfactual_maximum = max(
            max(point["target_pixels"].values())
            for point in evaluator["counterfactual_visibility_history"]
        )
        if counterfactual_maximum >= 256:
            return "EMPTY"
    return "FAILED"


def class_weights(labels: np.ndarray, count: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=count).astype(np.float32)
    weights = np.zeros(count, dtype=np.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.from_numpy(weights)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--belief-embeddings",
        type=Path,
        default=ROOT / "outputs/t01_prompt_state_v1/pi05_prefix_embeddings.npz",
    )
    parser.add_argument(
        "--effect-embeddings",
        type=Path,
        default=ROOT / "outputs/t01_open_and_observe_effect_v3/pi05_temporal_embeddings_v5.npz",
    )
    parser.add_argument(
        "--effect-data",
        type=Path,
        default=ROOT / "data/calibration/t01_open_and_observe_effect_v3.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/models/piu_v0_sidecar.pt",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results/training/piu_v0_training.json",
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=170817)
    args = parser.parse_args()
    for name in (
        "belief_embeddings",
        "effect_embeddings",
        "effect_data",
        "output",
        "report",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output.exists() or args.report.exists():
        raise FileExistsError("PIU training artifacts are immutable; choose a new output")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)

    belief_data = np.load(args.belief_embeddings)
    belief_raw = np.asarray(belief_data["features"], dtype=np.float32)
    belief_condition = np.asarray(belief_data["condition"], dtype=str)
    belief_seed = np.asarray(belief_data["seed"], dtype=np.int64)
    belief_split = np.asarray(belief_data["split"], dtype=str)
    location_by_condition = {
        "closed_hidden_butter": "middle_drawer",
        "closed_visible_cream_cheese": "visible_workspace",
        "open_visible_butter": "visible_workspace",
    }
    belief_truth = np.asarray(
        [LOCATION_LABELS.index(location_by_condition[item]) for item in belief_condition],
        dtype=np.int64,
    )
    rank_truth = np.asarray(
        [
            ACTION_LABELS.index("OPEN_TO_INSPECT")
            if item == "closed_hidden_butter"
            else ACTION_LABELS.index("DIRECT_ACT")
            for item in belief_condition
        ],
        dtype=np.int64,
    )
    belief_train = belief_split == "prototype_train"
    belief_calibration = (belief_seed >= 240) & (belief_seed <= 259)
    belief_diagnostic = belief_split == "heldout_validation"

    effect_data = np.load(args.effect_embeddings)
    history_raw = np.asarray(effect_data["history_features"], dtype=np.float32)
    robot_history = np.asarray(effect_data["robot_state_history"], dtype=np.float32)
    effect_seed = np.asarray(effect_data["seed"], dtype=np.int64)
    effect_split = np.asarray(effect_data["split"], dtype=str)
    rows = [json.loads(line) for line in args.effect_data.read_text().splitlines() if line]
    if len(rows) != history_raw.shape[0]:
        raise ValueError("effect rows and embeddings are not aligned")
    row_keys = [(row["regime"], int(row["seed"])) for row in rows]
    embedding_keys = list(zip(effect_data["regime"].astype(str), effect_seed.tolist(), strict=True))
    if row_keys != embedding_keys:
        raise ValueError("effect row order differs from frozen embeddings")
    relabeled = np.asarray([v9_outcome(row) for row in rows])
    effect_truth = np.asarray([OUTCOME_LABELS.index(item) for item in relabeled], dtype=np.int64)
    future_by_outcome = {
        "FAILED": "middle_drawer",
        "REVEALED": "visible_workspace",
        "EMPTY": "other_unsearched_region",
    }
    future_truth = np.asarray(
        [LOCATION_LABELS.index(future_by_outcome[item]) for item in relabeled], dtype=np.int64
    )
    progress_truth = np.asarray(
        [{"FAILED": 0.0, "REVEALED": 1.0, "EMPTY": 0.4}[item] for item in relabeled],
        dtype=np.float32,
    )
    effect_train = effect_split == "prototype_train"
    effect_calibration = effect_split == "conformal_calibration"
    effect_diagnostic = effect_split == "heldout_development"

    config = {
        "schema_version": "interaction-uncertainty.piu-sidecar-config.v0",
        "projected_dimension": 256,
        "hidden_dimension": 128,
        "action_context_dimension": len(ACTION_LABELS) + 1,
        "global_projection_seed": 171101,
        "spatial_projection_seed": 171102,
        "location_labels": list(LOCATION_LABELS),
        "outcome_labels": list(OUTCOME_LABELS),
        "action_labels": list(ACTION_LABELS),
    }
    belief_projected = fixed_project(
        belief_raw,
        output_dimension=config["projected_dimension"],
        seed=config["global_projection_seed"],
    )
    history_projected = fixed_project(
        history_raw,
        output_dimension=config["projected_dimension"],
        seed=config["spatial_projection_seed"],
    )
    action = np.zeros((len(rows), config["action_context_dimension"]), dtype=np.float32)
    action[:, ACTION_LABELS.index("OPEN_TO_INSPECT")] = 1.0
    action[:, len(ACTION_LABELS)] = np.asarray(
        [1.0 if row["full_executor"] else 25.0 / 300.0 for row in rows],
        dtype=np.float32,
    )
    outcome_one_hot = np.eye(len(OUTCOME_LABELS), dtype=np.float32)[effect_truth]

    tensors = {
        "belief": torch.from_numpy(belief_projected),
        "belief_truth": torch.from_numpy(belief_truth),
        "rank_truth": torch.from_numpy(rank_truth),
        "history": torch.from_numpy(history_projected),
        "robot": torch.from_numpy(robot_history),
        "action": torch.from_numpy(action),
        "effect_truth": torch.from_numpy(effect_truth),
        "future_truth": torch.from_numpy(future_truth),
        "outcome_one_hot": torch.from_numpy(outcome_one_hot),
        "progress_truth": torch.from_numpy(progress_truth),
    }
    model = build_torch_model(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    belief_weight = class_weights(belief_truth[belief_train], len(LOCATION_LABELS))
    rank_weight = class_weights(rank_truth[belief_train], len(ACTION_LABELS))
    effect_weight = class_weights(effect_truth[effect_train], len(OUTCOME_LABELS))
    future_weight = class_weights(future_truth[effect_train], len(LOCATION_LABELS))
    bce = nn.BCEWithLogitsLoss()
    train_history = []
    belief_train_t = torch.from_numpy(belief_train)
    effect_train_t = torch.from_numpy(effect_train)
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        belief_logits = model.belief_logits(tensors["belief"][belief_train_t])
        rank_logits = model.rank_logits(tensors["belief"][belief_train_t])
        before = tensors["history"][effect_train_t, 0]
        forecast = model.forecast_values(before, tensors["action"][effect_train_t])
        outcome_logits = model.outcome_logits(
            tensors["history"][effect_train_t], tensors["robot"][effect_train_t]
        )
        future_logits = model.future_logits(
            before,
            tensors["action"][effect_train_t],
            tensors["outcome_one_hot"][effect_train_t],
        )
        loss_parts = {
            "belief": nn.functional.cross_entropy(
                belief_logits, tensors["belief_truth"][belief_train_t], weight=belief_weight
            ),
            "rank": nn.functional.cross_entropy(
                rank_logits, tensors["rank_truth"][belief_train_t], weight=rank_weight
            ),
            "forecast": nn.functional.cross_entropy(
                forecast[:, : len(OUTCOME_LABELS)],
                tensors["effect_truth"][effect_train_t],
                weight=effect_weight,
            ),
            "outcome": nn.functional.cross_entropy(
                outcome_logits, tensors["effect_truth"][effect_train_t], weight=effect_weight
            ),
            "future": nn.functional.cross_entropy(
                future_logits, tensors["future_truth"][effect_train_t], weight=future_weight
            ),
            "progress": bce(forecast[:, -1], tensors["progress_truth"][effect_train_t]),
        }
        loss = sum(loss_parts.values())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch in {0, args.epochs - 1} or (epoch + 1) % 25 == 0:
            train_history.append(
                {"epoch": epoch + 1, "total": float(loss.item()), **{key: float(value.item()) for key, value in loss_parts.items()}}
            )

    model.eval()
    with torch.no_grad():
        belief_probabilities = torch.softmax(model.belief_logits(tensors["belief"]), dim=-1).numpy()
        outcome_probabilities = torch.softmax(
            model.outcome_logits(tensors["history"], tensors["robot"]), dim=-1
        ).numpy()
        forecast_probabilities = torch.softmax(
            model.forecast_values(tensors["history"][:, 0], tensors["action"])[
                :, : len(OUTCOME_LABELS)
            ],
            dim=-1,
        ).numpy()
        rank_prediction = model.rank_logits(tensors["belief"]).argmax(dim=-1).numpy()
    belief_thresholds = mondrian_thresholds(
        belief_probabilities[belief_calibration],
        belief_truth[belief_calibration],
        LOCATION_LABELS,
        args.alpha,
    )
    outcome_thresholds = mondrian_thresholds(
        outcome_probabilities[effect_calibration],
        effect_truth[effect_calibration],
        OUTCOME_LABELS,
        args.alpha,
    )
    metadata = {
        "schema_version": "interaction-uncertainty.piu-sidecar.v0",
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "encoder": "frozen pi05_libero PaliGemma prefix",
        "encoder_inputs": ["stock agentview RGB", "stock wrist RGB", "public robot state", "prompt", "six-point public history"],
        "online_oracle_inputs": [],
        "offline_label_inputs": ["post-termination segmentation visibility", "post-termination drawer joint", "registered scenario regime"],
        "claim_status": "prototype/development-only; not clean validation",
        "data": {
            "belief_embeddings": str(args.belief_embeddings.relative_to(ROOT)),
            "belief_sha256": digest(args.belief_embeddings),
            "effect_embeddings": str(args.effect_embeddings.relative_to(ROOT)),
            "effect_embeddings_sha256": digest(args.effect_embeddings),
            "effect_data": str(args.effect_data.relative_to(ROOT)),
            "effect_data_sha256": digest(args.effect_data),
        },
        "split_contract": {
            "belief_train": "220-239",
            "belief_conformal": "240-259",
            "belief_diagnostic": "260-269",
            "effect_train": "600-619",
            "effect_conformal": "620-652",
            "effect_diagnostic_contaminated": "653-659",
            "clean_extension_untouched": "660-699",
            "sealed_audit_untouched": "900-999",
        },
        "conformal": {"alpha": args.alpha, "belief": belief_thresholds, "outcome": outcome_thresholds},
        "unsupported_initial_location_classes": [
            label for label, threshold in belief_thresholds.items() if threshold < 0.0
        ],
    }
    report = {
        **metadata,
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "loss_history": train_history,
            "train_label_counts": {
                "belief": Counter(LOCATION_LABELS[index] for index in belief_truth[belief_train]),
                "outcome_v9": Counter(OUTCOME_LABELS[index] for index in effect_truth[effect_train]),
            },
        },
        "diagnostics": {
            "belief_260_269": metrics(
                belief_probabilities[belief_diagnostic],
                belief_truth[belief_diagnostic],
                LOCATION_LABELS,
                belief_thresholds,
            ),
            "outcome_653_659_contaminated": metrics(
                outcome_probabilities[effect_diagnostic],
                effect_truth[effect_diagnostic],
                OUTCOME_LABELS,
                outcome_thresholds,
            ),
            "forecast_argmax_653_659_contaminated": {
                "samples": int(effect_diagnostic.sum()),
                "accuracy": float(
                    np.mean(
                        forecast_probabilities[effect_diagnostic].argmax(axis=1)
                        == effect_truth[effect_diagnostic]
                    )
                ),
            },
            "rank_260_269": {
                "samples": int(belief_diagnostic.sum()),
                "accuracy": float(np.mean(rank_prediction[belief_diagnostic] == rank_truth[belief_diagnostic])),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "metadata": metadata, "state_dict": model.state_dict()}, args.output)
    report["model"] = str(args.output.relative_to(ROOT))
    report["model_sha256"] = digest(args.output)
    args.report.write_text(json.dumps(report, indent=2, default=dict) + "\n")
    print(json.dumps({"model": report["model"], "diagnostics": report["diagnostics"]}, indent=2))


if __name__ == "__main__":
    main()
