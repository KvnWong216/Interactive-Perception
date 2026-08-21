#!/usr/bin/env python3
"""Train and freeze the object-level PIU belief/effect sidecar on train+cal only."""

from __future__ import annotations

import argparse
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
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from interaction_uncertainty.object_sidecar import (  # noqa: E402
    ACTION_LABELS_V1,
    LOCATION_LABELS_V1,
    SEMANTIC_EFFECT_LABELS_V1,
    build_object_torch_model,
    semantic_effect_teacher,
)
from interaction_uncertainty.sidecar import fixed_project  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def keyed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["sample_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate sample_id")
    return result


def rle_decode(value: dict[str, Any]) -> np.ndarray:
    values = []
    current = int(value.get("starts_with", 0))
    for length in value["counts"]:
        values.extend([current] * int(length))
        current = 1 - current
    return np.asarray(values, dtype=bool).reshape(value["size"])


def overlap_fraction(left: np.ndarray, right: np.ndarray) -> float:
    denominator = int(right.sum())
    if denominator == 0:
        return 0.0
    return float(np.logical_and(left, right).sum() / denominator)


def tokens(value: str) -> set[str]:
    return {
        token
        for token in value.lower().replace("_", " ").split()
        if len(token) > 2 and token not in {"the", "with", "into", "layer"}
    }


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    if ordered.size == 0:
        raise ValueError("each calibration class must be represented")
    rank = min(ordered.size, math.ceil((ordered.size + 1) * (1.0 - alpha)))
    return float(ordered[rank - 1])


def mondrian_thresholds(
    probabilities: np.ndarray,
    truth: np.ndarray,
    labels: tuple[str, ...],
    alpha: float,
) -> dict[str, float]:
    return {
        label: conformal_quantile(1.0 - probabilities[truth == index, index], alpha)
        for index, label in enumerate(labels)
    }


def prediction_sets(
    probabilities: np.ndarray,
    labels: tuple[str, ...],
    thresholds: dict[str, float],
) -> list[tuple[str, ...]]:
    results = []
    for row in probabilities:
        selected = tuple(
            label
            for index, label in enumerate(labels)
            if 1.0 - float(row[index]) <= thresholds[label]
        )
        results.append(selected or labels)
    return results


def entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-9, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=-1) / math.log(clipped.shape[-1])


def probability_metrics(
    probabilities: np.ndarray,
    truth: np.ndarray,
    labels: tuple[str, ...],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    sets = prediction_sets(probabilities, labels, thresholds)
    predicted = probabilities.argmax(axis=-1)
    truth_one_hot = np.eye(len(labels), dtype=np.float64)[truth]
    coverage = np.asarray(
        [labels[int(index)] in result for index, result in zip(truth, sets, strict=True)]
    )
    singleton = np.asarray(
        [result == (labels[int(index)],) for index, result in zip(truth, sets, strict=True)]
    )
    return {
        "samples": int(len(truth)),
        "accuracy": float(np.mean(predicted == truth)),
        "brier": float(np.mean(np.sum((probabilities - truth_one_hot) ** 2, axis=-1))),
        "nll": float(-np.mean(np.log(np.clip(probabilities[np.arange(len(truth)), truth], 1e-9, 1.0)))),
        "coverage": float(np.mean(coverage)),
        "singleton_correct": float(np.mean(singleton)),
        "mean_prediction_set_size": float(np.mean([len(value) for value in sets])),
        "per_class": {
            label: {
                "samples": int(np.sum(truth == index)),
                "accuracy": float(np.mean(predicted[truth == index] == index)),
                "coverage": float(np.mean(coverage[truth == index])),
                "singleton_correct": float(np.mean(singleton[truth == index])),
            }
            for index, label in enumerate(labels)
        },
    }


def load_split(
    *,
    snapshots_path: Path,
    labels_path: Path,
    scene_path: Path,
    object_features_path: Path,
    prefix_features_path: Path,
    prefix_projection_dimension: int,
    prefix_projection_seed: int,
) -> dict[str, Any]:
    snapshots = keyed(read_jsonl(snapshots_path))
    labels = keyed(read_jsonl(labels_path))
    scenes = keyed(read_jsonl(scene_path))
    prefix_store = np.load(prefix_features_path)
    prefix_ids = prefix_store["sample_id"].astype(str).tolist()
    prefix_by_id = {
        sample_id: np.asarray(prefix_store["features"][index], dtype=np.float32)
        for index, sample_id in enumerate(prefix_ids)
    }
    object_store = np.load(object_features_path)
    object_matrix = np.asarray(object_store["features"], dtype=np.float32)
    identifiers = list(snapshots)
    if set(identifiers) != set(labels) or set(identifiers) != set(scenes) or set(identifiers) != set(prefix_by_id):
        raise ValueError("snapshot, label, scene, and prefix sample IDs differ")

    prefix = fixed_project(
        np.stack([prefix_by_id[sample_id] for sample_id in identifiers]),
        output_dimension=prefix_projection_dimension,
        seed=prefix_projection_seed,
    ).astype(np.float32)
    node_rows: list[np.ndarray] = []
    node_truth_rows: list[np.ndarray] = []
    object_ids: list[list[str]] = []
    target_covered: list[bool] = []
    for sample_id in identifiers:
        snapshot = snapshots[sample_id]
        label = labels[sample_id]
        scene = scenes[sample_id]
        teacher_interaction_target = str(label.get("interaction_target") or "").strip().lower()
        policy_queries = {str(value).strip().lower() for value in snapshot["visual_queries"]}
        frontend_queries = {str(value).strip().lower() for value in scene["visual_queries"]}
        if teacher_interaction_target and (
            teacher_interaction_target in policy_queries
            or teacher_interaction_target in frontend_queries
        ):
            raise RuntimeError(
                f"teacher interaction_target leaked into policy/frontend queries: {sample_id}"
            )
        target_masks = {
            "agentview": rle_decode(label["target_mask_policy_resolution_rle"]["agentview"]),
            "wrist": rle_decode(
                label["target_mask_policy_resolution_rle"]["robot0_eye_in_hand"]
            ),
        }
        target_terms = tokens(str(snapshot["target"]))
        # This is teacher-only node supervision.  It must never be copied into
        # the node input features or policy-side open-vocabulary queries.
        interaction_terms = tokens(teacher_interaction_target)
        nodes = []
        relevances = []
        ids = []
        for node in scene["objects"]:
            feature = object_matrix[int(node["feature_row"])]
            box = np.asarray(node["bbox_xyxy"], dtype=np.float32) / 224.0
            label_terms = set().union(
                *(tokens(value) for value in node["label_candidates"])
            )
            target_label_match = float(bool(target_terms & label_terms))
            metadata = np.asarray(
                [
                    float(node["grounding_score"]),
                    float(node["mask_score"]),
                    min(1.0, float(node["visible_area"]) / (224.0 * 224.0)),
                    *box.tolist(),
                    float(node["view"] == "agentview"),
                    float(node["view"] == "wrist"),
                    target_label_match,
                ],
                dtype=np.float32,
            )
            nodes.append(np.concatenate((feature, metadata)))
            if label["target_location"] == "visible_workspace":
                node_mask = rle_decode(node["mask_rle"])
                relevance = overlap_fraction(node_mask, target_masks[str(node["view"])])
                relevances.append(float(relevance >= 0.25))
            else:
                relevances.append(float(bool(interaction_terms & label_terms)))
            ids.append(str(node["object_id"]))
        if not nodes:
            raise ValueError(f"scene packet has no objects: {sample_id}")
        truth = np.asarray(relevances, dtype=np.float32)
        node_rows.append(np.stack(nodes))
        node_truth_rows.append(truth)
        object_ids.append(ids)
        target_covered.append(bool(truth.any()))

    maximum_nodes = max(len(nodes) for nodes in node_rows)
    node_dimension = node_rows[0].shape[-1]
    node_values = np.zeros((len(identifiers), maximum_nodes, node_dimension), dtype=np.float32)
    node_mask = np.zeros((len(identifiers), maximum_nodes), dtype=bool)
    node_truth = np.zeros((len(identifiers), maximum_nodes), dtype=np.float32)
    for index, (nodes, truth) in enumerate(zip(node_rows, node_truth_rows, strict=True)):
        node_values[index, : len(nodes)] = nodes
        node_mask[index, : len(nodes)] = True
        node_truth[index, : len(nodes)] = truth
    location_truth = np.asarray(
        [LOCATION_LABELS_V1.index(str(labels[sample_id]["target_location"])) for sample_id in identifiers],
        dtype=np.int64,
    )
    action_truth = np.asarray(
        [ACTION_LABELS_V1.index(str(labels[sample_id]["preferred_action"])) for sample_id in identifiers],
        dtype=np.int64,
    )
    return {
        "ids": identifiers,
        "prefix": prefix,
        "nodes": node_values,
        "node_mask": node_mask,
        "node_truth": node_truth,
        "object_ids": object_ids,
        "target_covered": np.asarray(target_covered),
        "location_truth": location_truth,
        "action_truth": action_truth,
        "paths": {
            "snapshots": snapshots_path,
            "labels": labels_path,
            "scenes": scene_path,
            "object_features": object_features_path,
            "prefix_features": prefix_features_path,
        },
    }


def combine_splits(values: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate development-only training splits with feature-row padding."""

    if not values:
        raise ValueError("at least one training split is required")
    node_dimensions = {value["nodes"].shape[-1] for value in values}
    if len(node_dimensions) != 1:
        raise ValueError("training splits have different node feature dimensions")
    node_dimension = node_dimensions.pop()
    maximum_nodes = max(value["nodes"].shape[1] for value in values)
    padded_nodes = []
    padded_masks = []
    padded_truth = []
    for value in values:
        count, old_nodes, _ = value["nodes"].shape
        nodes = np.zeros((count, maximum_nodes, node_dimension), dtype=np.float32)
        mask = np.zeros((count, maximum_nodes), dtype=bool)
        truth = np.zeros((count, maximum_nodes), dtype=np.float32)
        nodes[:, :old_nodes] = value["nodes"]
        mask[:, :old_nodes] = value["node_mask"]
        truth[:, :old_nodes] = value["node_truth"]
        padded_nodes.append(nodes)
        padded_masks.append(mask)
        padded_truth.append(truth)
    paths = {}
    for index, value in enumerate(values):
        prefix = "primary" if index == 0 else f"extra_{index}"
        for name, path in value["paths"].items():
            paths[f"{prefix}_{name}"] = path
    return {
        "ids": [sample_id for value in values for sample_id in value["ids"]],
        "prefix": np.concatenate([value["prefix"] for value in values], axis=0),
        "nodes": np.concatenate(padded_nodes, axis=0),
        "node_mask": np.concatenate(padded_masks, axis=0),
        "node_truth": np.concatenate(padded_truth, axis=0),
        "object_ids": [object_id for value in values for object_id in value["object_ids"]],
        "target_covered": np.concatenate([value["target_covered"] for value in values]),
        "location_truth": np.concatenate([value["location_truth"] for value in values]),
        "action_truth": np.concatenate([value["action_truth"] for value in values]),
        "paths": paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "calibration"):
        parser.add_argument(f"--{split}-snapshots", type=Path, required=True)
        parser.add_argument(f"--{split}-labels", type=Path, required=True)
        parser.add_argument(f"--{split}-scenes", type=Path, required=True)
        parser.add_argument(f"--{split}-object-features", type=Path, required=True)
        parser.add_argument(f"--{split}-prefix-features", type=Path, required=True)
    parser.add_argument("--extra-train-snapshots", type=Path)
    parser.add_argument("--extra-train-labels", type=Path)
    parser.add_argument("--extra-train-scenes", type=Path)
    parser.add_argument("--extra-train-object-features", type=Path)
    parser.add_argument("--extra-train-prefix-features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=26081801)
    args = parser.parse_args()
    path_names = [
        f"{split}_{kind}"
        for split in ("train", "calibration")
        for kind in ("snapshots", "labels", "scenes", "object_features", "prefix_features")
    ] + ["output", "report"]
    extra_names = [
        "extra_train_snapshots",
        "extra_train_labels",
        "extra_train_scenes",
        "extra_train_object_features",
        "extra_train_prefix_features",
    ]
    provided_extra = [getattr(args, name) is not None for name in extra_names]
    if any(provided_extra) and not all(provided_extra):
        raise ValueError("all five extra-training paths must be provided together")
    path_names += [name for name in extra_names if getattr(args, name) is not None]
    for name in path_names:
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output.exists() or args.report.exists():
        raise FileExistsError("model/report outputs are immutable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    projection_dimension = 256
    projection_seed = 26081811
    train = load_split(
        snapshots_path=args.train_snapshots,
        labels_path=args.train_labels,
        scene_path=args.train_scenes,
        object_features_path=args.train_object_features,
        prefix_features_path=args.train_prefix_features,
        prefix_projection_dimension=projection_dimension,
        prefix_projection_seed=projection_seed,
    )
    if all(provided_extra):
        extra_train = load_split(
            snapshots_path=args.extra_train_snapshots,
            labels_path=args.extra_train_labels,
            scene_path=args.extra_train_scenes,
            object_features_path=args.extra_train_object_features,
            prefix_features_path=args.extra_train_prefix_features,
            prefix_projection_dimension=projection_dimension,
            prefix_projection_seed=projection_seed,
        )
        train = combine_splits([train, extra_train])
    calibration = load_split(
        snapshots_path=args.calibration_snapshots,
        labels_path=args.calibration_labels,
        scene_path=args.calibration_scenes,
        object_features_path=args.calibration_object_features,
        prefix_features_path=args.calibration_prefix_features,
        prefix_projection_dimension=projection_dimension,
        prefix_projection_seed=projection_seed,
    )
    if not train["target_covered"].all() or not calibration["target_covered"].all():
        raise RuntimeError("object frontend failed to propose a supervised target/container node")
    maximum_nodes = max(train["nodes"].shape[1], calibration["nodes"].shape[1])
    node_dimension = train["nodes"].shape[-1]

    def repad(split):
        if split["nodes"].shape[1] == maximum_nodes:
            return
        count = split["nodes"].shape[0]
        padded = np.zeros((count, maximum_nodes, node_dimension), dtype=np.float32)
        mask = np.zeros((count, maximum_nodes), dtype=bool)
        truth = np.zeros((count, maximum_nodes), dtype=np.float32)
        old = split["nodes"].shape[1]
        padded[:, :old] = split["nodes"]
        mask[:, :old] = split["node_mask"]
        truth[:, :old] = split["node_truth"]
        split["nodes"], split["node_mask"], split["node_truth"] = padded, mask, truth

    repad(train)
    repad(calibration)
    config = {
        "schema_version": "interaction-uncertainty.piu-object-sidecar-config.v1",
        "prefix_projected_dimension": projection_dimension,
        "prefix_projection_seed": projection_seed,
        "node_input_dimension": node_dimension,
        "hidden_dimension": 128,
        "location_labels": list(LOCATION_LABELS_V1),
        "action_labels": list(ACTION_LABELS_V1),
        "semantic_effect_labels": list(SEMANTIC_EFFECT_LABELS_V1),
        "utility": {
            "eta_information": 1.0,
            "eta_task": 0.5,
            "cost": {"DIRECT_ACT": 0.10, "OPEN_TO_INSPECT": 0.18},
            "execution_lower_bound": {"DIRECT_ACT": 0.970, "OPEN_TO_INSPECT": 0.924},
        },
    }
    model = build_object_torch_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    prefix = torch.from_numpy(train["prefix"])
    nodes = torch.from_numpy(train["nodes"])
    node_mask = torch.from_numpy(train["node_mask"])
    node_truth = torch.from_numpy(train["node_truth"])
    location_truth = torch.from_numpy(train["location_truth"])
    action_truth = torch.from_numpy(train["action_truth"])
    valid_relevance = node_mask
    positive = float(node_truth[valid_relevance].sum())
    negative = float(valid_relevance.sum()) - positive
    pos_weight = torch.tensor(max(1.0, negative / max(positive, 1.0)))
    history = []
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        location_logits, action_logits, relevance_logits, _, fused = model.state_logits(prefix, nodes, node_mask)
        fused_actions = fused.repeat_interleave(len(ACTION_LABELS_V1), dim=0)
        action_indices = torch.arange(len(ACTION_LABELS_V1)).repeat(len(train["ids"]))
        action_one_hot = nn.functional.one_hot(action_indices, len(ACTION_LABELS_V1)).float()
        effect_logits, future_logits = model.effect_logits(fused_actions, action_one_hot)
        effect_truth = []
        future_truth = []
        for location_index in train["location_truth"]:
            location = LOCATION_LABELS_V1[int(location_index)]
            for action in ACTION_LABELS_V1:
                effect, future = semantic_effect_teacher(location, action)
                effect_truth.append(SEMANTIC_EFFECT_LABELS_V1.index(effect))
                future_truth.append(LOCATION_LABELS_V1.index(future))
        losses = {
            "location": nn.functional.cross_entropy(location_logits, location_truth),
            "rank": nn.functional.cross_entropy(action_logits, action_truth),
            "node": nn.functional.binary_cross_entropy_with_logits(
                relevance_logits[valid_relevance], node_truth[valid_relevance], pos_weight=pos_weight
            ),
            "effect": nn.functional.cross_entropy(effect_logits, torch.tensor(effect_truth)),
            "future": nn.functional.cross_entropy(future_logits, torch.tensor(future_truth)),
        }
        loss = losses["location"] + losses["rank"] + 0.5 * losses["node"] + 0.5 * losses["effect"] + 0.5 * losses["future"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch in {0, args.epochs - 1} or (epoch + 1) % 50 == 0:
            history.append({"epoch": epoch + 1, "total": float(loss.item()), **{key: float(value.item()) for key, value in losses.items()}})

    def infer(split):
        model.eval()
        with torch.no_grad():
            prefix_t = torch.from_numpy(split["prefix"])
            nodes_t = torch.from_numpy(split["nodes"])
            mask_t = torch.from_numpy(split["node_mask"])
            location_logits, action_logits, relevance, attention, fused = model.state_logits(prefix_t, nodes_t, mask_t)
            location_probability = torch.softmax(location_logits, dim=-1).numpy()
            action_probability = torch.softmax(action_logits, dim=-1).numpy()
            action_indices = torch.arange(len(ACTION_LABELS_V1)).repeat(len(split["ids"]))
            action_one_hot = nn.functional.one_hot(action_indices, len(ACTION_LABELS_V1)).float()
            effect_logits, future_logits = model.effect_logits(
                fused.repeat_interleave(len(ACTION_LABELS_V1), dim=0), action_one_hot
            )
            effect_probability = torch.softmax(effect_logits, dim=-1).reshape(len(split["ids"]), len(ACTION_LABELS_V1), -1).numpy()
            future_probability = torch.softmax(future_logits, dim=-1).reshape(len(split["ids"]), len(ACTION_LABELS_V1), -1).numpy()
        node_hit = []
        for index in range(len(split["ids"])):
            valid = np.flatnonzero(split["node_mask"][index])
            selected = valid[int(np.argmax(relevance[index, valid].numpy()))]
            node_hit.append(bool(split["node_truth"][index, selected]))
        current_entropy = entropy(location_probability)
        future_entropy = entropy(future_probability)
        information_gain = current_entropy[:, None] - future_entropy
        task_progress = effect_probability[:, :, SEMANTIC_EFFECT_LABELS_V1.index("TASK_PROGRESS")]
        task_progress += effect_probability[:, :, SEMANTIC_EFFECT_LABELS_V1.index("TARGET_REVEALED")]
        reliability = np.asarray([config["utility"]["execution_lower_bound"][action] for action in ACTION_LABELS_V1])
        cost = np.asarray([config["utility"]["cost"][action] for action in ACTION_LABELS_V1])
        utility = reliability[None, :] * (information_gain + config["utility"]["eta_task"] * task_progress) - cost[None, :]
        selected = utility.argmax(axis=-1)
        return {
            "location_probability": location_probability,
            "action_probability": action_probability,
            "effect_probability": effect_probability,
            "future_probability": future_probability,
            "node_hit_at_1": float(np.mean(node_hit)),
            "optimizer_selected": selected,
            "optimizer_accuracy": float(np.mean(selected == split["action_truth"])),
            "rank_accuracy": float(np.mean(action_probability.argmax(axis=-1) == split["action_truth"])),
            "information_gain": information_gain,
            "utility": utility,
        }

    train_output = infer(train)
    calibration_output = infer(calibration)
    thresholds = mondrian_thresholds(
        calibration_output["location_probability"], calibration["location_truth"], LOCATION_LABELS_V1, args.alpha
    )
    action_thresholds = mondrian_thresholds(
        calibration_output["action_probability"], calibration["action_truth"], ACTION_LABELS_V1, args.alpha
    )
    metadata = {
        "schema_version": "interaction-uncertainty.piu-object-sidecar.v1",
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "claim_status": "frozen before scene-disjoint clean development",
        "config": config,
        "conformal": {"alpha": args.alpha, "location": thresholds, "action": action_thresholds},
        "online_inputs": ["frozen pi05 prefix", "Grounding DINO boxes", "SAM masks", "DINOv2 mask-pooled features", "public geometry metadata"],
        "online_oracle_inputs": [],
        "semantic_effect_note": "counterfactual semantic effect; physical execution risk enters utility separately",
        "sources": {
            split: {name: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for name, path in values["paths"].items()}
            for split, values in (("train", train), ("calibration", calibration))
        },
    }
    report = {
        **metadata,
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "loss_history": history,
            "class_counts": {
                "location": dict(Counter(LOCATION_LABELS_V1[index] for index in train["location_truth"])),
                "action": dict(Counter(ACTION_LABELS_V1[index] for index in train["action_truth"])),
            },
        },
        "train_metrics": {
            "location": probability_metrics(train_output["location_probability"], train["location_truth"], LOCATION_LABELS_V1, thresholds),
            "node_hit_at_1": train_output["node_hit_at_1"],
            "rank_accuracy": train_output["rank_accuracy"],
            "optimizer_accuracy": train_output["optimizer_accuracy"],
        },
        "calibration_metrics": {
            "location": probability_metrics(calibration_output["location_probability"], calibration["location_truth"], LOCATION_LABELS_V1, thresholds),
            "action": probability_metrics(calibration_output["action_probability"], calibration["action_truth"], ACTION_LABELS_V1, action_thresholds),
            "node_hit_at_1": calibration_output["node_hit_at_1"],
            "rank_accuracy": calibration_output["rank_accuracy"],
            "optimizer_accuracy": calibration_output["optimizer_accuracy"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "metadata": metadata, "state_dict": model.state_dict()}, args.output)
    report["model"] = str(args.output.relative_to(ROOT))
    report["model_sha256"] = digest(args.output)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"model": report["model"], "calibration_metrics": report["calibration_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
