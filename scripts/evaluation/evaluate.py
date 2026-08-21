#!/usr/bin/env python3
"""Evaluate the frozen object-level PIU sidecar on scene-disjoint clean data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/training"), str(ROOT / "src")]
from interaction_uncertainty.object_sidecar import (  # noqa: E402
    ACTION_LABELS_V1,
    LOCATION_LABELS_V1,
    SEMANTIC_EFFECT_LABELS_V1,
    build_object_torch_model,
    semantic_effect_teacher,
)
from train import (  # noqa: E402
    entropy,
    load_split,
    prediction_sets,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def lower_bound(successes: int, trials: int, confidence: float = 0.95) -> float:
    if trials <= 0:
        return 0.0
    if successes <= 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence, successes, trials - successes + 1))


def class_rates(values: np.ndarray, truth: np.ndarray, labels: tuple[str, ...]) -> dict[str, Any]:
    result = {}
    for index, label in enumerate(labels):
        selected = values[truth == index]
        successes = int(selected.sum())
        trials = int(selected.size)
        result[label] = {
            "successes": successes,
            "trials": trials,
            "rate": float(successes / trials),
            "one_sided_95_lower": lower_bound(successes, trials),
            "meets_development_0.80": lower_bound(successes, trials) >= 0.80,
            "meets_original_0.90": lower_bound(successes, trials) >= 0.90,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--object-features", type=Path, required=True)
    parser.add_argument("--prefix-features", type=Path, required=True)
    parser.add_argument("--frontend-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "gates",
        "model",
        "snapshots",
        "labels",
        "scenes",
        "object_features",
        "prefix_features",
        "frontend_audit",
        "output",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output.exists():
        raise FileExistsError(f"immutable clean evaluation exists: {args.output}")
    gate_spec = yaml.safe_load(args.gates.read_text())
    if digest(args.model) != gate_spec["model"]["sha256"]:
        raise ValueError("frozen model hash differs from clean gate specification")

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    metadata = checkpoint["metadata"]
    split = load_split(
        snapshots_path=args.snapshots,
        labels_path=args.labels,
        scene_path=args.scenes,
        object_features_path=args.object_features,
        prefix_features_path=args.prefix_features,
        prefix_projection_dimension=int(config["prefix_projected_dimension"]),
        prefix_projection_seed=int(config["prefix_projection_seed"]),
    )
    model = build_object_torch_model(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    def infer(prefix_values: np.ndarray, node_values: np.ndarray) -> dict[str, Any]:
        with torch.no_grad():
            prefix = torch.from_numpy(prefix_values)
            nodes = torch.from_numpy(node_values)
            node_mask = torch.from_numpy(split["node_mask"])
            location_logits, action_logits, relevance, attention, fused = model.state_logits(
                prefix, nodes, node_mask
            )
            location_probability = torch.softmax(location_logits, dim=-1).numpy()
            action_probability = torch.softmax(action_logits, dim=-1).numpy()
            action_indices = torch.arange(len(ACTION_LABELS_V1)).repeat(len(split["ids"]))
            one_hot = torch.nn.functional.one_hot(
                action_indices, len(ACTION_LABELS_V1)
            ).float()
            effect_logits, future_logits = model.effect_logits(
                fused.repeat_interleave(len(ACTION_LABELS_V1), dim=0), one_hot
            )
            effect_probability = torch.softmax(effect_logits, dim=-1).reshape(
                len(split["ids"]), len(ACTION_LABELS_V1), -1
            ).numpy()
            future_probability = torch.softmax(future_logits, dim=-1).reshape(
                len(split["ids"]), len(ACTION_LABELS_V1), -1
            ).numpy()
        current_entropy = entropy(location_probability)
        future_entropy = entropy(future_probability)
        information_gain = current_entropy[:, None] - future_entropy
        task_progress = effect_probability[
            :, :, SEMANTIC_EFFECT_LABELS_V1.index("TASK_PROGRESS")
        ] + effect_probability[:, :, SEMANTIC_EFFECT_LABELS_V1.index("TARGET_REVEALED")]
        reliability = np.asarray(
            [config["utility"]["execution_lower_bound"][action] for action in ACTION_LABELS_V1]
        )
        cost = np.asarray([config["utility"]["cost"][action] for action in ACTION_LABELS_V1])
        utility = reliability[None, :] * (
            config["utility"]["eta_information"] * information_gain
            + config["utility"]["eta_task"] * task_progress
        ) - cost[None, :]
        return {
            "location_probability": location_probability,
            "action_probability": action_probability,
            "effect_probability": effect_probability,
            "future_probability": future_probability,
            "relevance": relevance.numpy(),
            "attention": attention.numpy(),
            "entropy": current_entropy,
            "information_gain": information_gain,
            "task_progress": task_progress,
            "utility": utility,
            "optimizer_selected": utility.argmax(axis=-1),
        }

    output = infer(split["prefix"], split["nodes"])
    prefix_zeroed = infer(np.zeros_like(split["prefix"]), split["nodes"])
    nodes_zeroed = infer(split["prefix"], np.zeros_like(split["nodes"]))
    location_sets = prediction_sets(
        output["location_probability"],
        LOCATION_LABELS_V1,
        metadata["conformal"]["location"],
    )
    action_sets = prediction_sets(
        output["action_probability"],
        ACTION_LABELS_V1,
        metadata["conformal"]["action"],
    )
    location_truth = split["location_truth"]
    action_truth = split["action_truth"]
    location_retained = np.asarray(
        [LOCATION_LABELS_V1[int(value)] in selected for value, selected in zip(location_truth, location_sets, strict=True)]
    )
    action_retained = np.asarray(
        [ACTION_LABELS_V1[int(value)] in selected for value, selected in zip(action_truth, action_sets, strict=True)]
    )
    deterministic_action = {
        "visible_workspace": "DIRECT_ACT",
        "closed_container": "OPEN_TO_INSPECT",
    }
    route = []
    correct_singleton = []
    false_singleton = []
    node_hits = []
    traces = []
    snapshots = {row["sample_id"]: row for row in read_jsonl(args.snapshots)}
    scenes = {row["sample_id"]: row for row in read_jsonl(args.scenes)}
    for index, sample_id in enumerate(split["ids"]):
        optimizer_action = ACTION_LABELS_V1[int(output["optimizer_selected"][index])]
        if (
            len(location_sets[index]) == 1
            and len(action_sets[index]) == 1
            and deterministic_action[location_sets[index][0]] == action_sets[index][0]
            and action_sets[index][0] == optimizer_action
        ):
            selected_route = optimizer_action
        else:
            selected_route = "SAFE_STOP"
        truth_action = ACTION_LABELS_V1[int(action_truth[index])]
        route.append(selected_route)
        correct_singleton.append(selected_route == truth_action)
        false_singleton.append(selected_route not in {truth_action, "SAFE_STOP"})
        valid = np.flatnonzero(split["node_mask"][index])
        relevance_values = output["relevance"][index, valid]
        order = valid[np.argsort(relevance_values)[::-1]]
        selected_node = int(order[0])
        node_hit = bool(split["node_truth"][index, selected_node])
        node_hits.append(node_hit)
        object_nodes = scenes[sample_id]["objects"]
        top_nodes = []
        for node_index in order[:5]:
            node = object_nodes[int(node_index)]
            top_nodes.append(
                {
                    "object_id": node["object_id"],
                    "view": node["view"],
                    "labels": node["label_candidates"],
                    "bbox_xyxy": node["bbox_xyxy"],
                    "relevance": float(output["relevance"][index, node_index]),
                    "attention": float(output["attention"][index, node_index]),
                    "teacher_relevant_evaluator_only": bool(split["node_truth"][index, node_index]),
                }
            )
        effects = {}
        for action_index, action in enumerate(ACTION_LABELS_V1):
            effects[action] = {
                "outcome_probability": {
                    label: float(output["effect_probability"][index, action_index, value])
                    for value, label in enumerate(SEMANTIC_EFFECT_LABELS_V1)
                },
                "future_location_probability": {
                    label: float(output["future_probability"][index, action_index, value])
                    for value, label in enumerate(LOCATION_LABELS_V1)
                },
                "expected_information_gain": float(output["information_gain"][index, action_index]),
                "expected_task_progress": float(output["task_progress"][index, action_index]),
                "utility": float(output["utility"][index, action_index]),
            }
        traces.append(
            {
                "sample_id": sample_id,
                "scenario_id": snapshots[sample_id]["scenario_id"],
                "prompt": snapshots[sample_id]["prompt"],
                "location_truth_evaluator_only": LOCATION_LABELS_V1[int(location_truth[index])],
                "action_truth_evaluator_only": truth_action,
                "location_probability": {
                    label: float(output["location_probability"][index, value])
                    for value, label in enumerate(LOCATION_LABELS_V1)
                },
                "location_conformal_set": list(location_sets[index]),
                "task_uncertainty": float(output["entropy"][index]),
                "action_rank_probability": {
                    label: float(output["action_probability"][index, value])
                    for value, label in enumerate(ACTION_LABELS_V1)
                },
                "action_conformal_set": list(action_sets[index]),
                "optimizer_action": optimizer_action,
                "online_route": selected_route,
                "node_hit_at_1_evaluator_only": node_hit,
                "top_uncertain_nodes": top_nodes,
                "candidate_effects": effects,
                "online_oracle_reads": 0,
            }
        )

    correct_singleton_array = np.asarray(correct_singleton)
    false_singleton_array = np.asarray(false_singleton)
    effect_correct = defaultdict(list)
    future_correct = defaultdict(list)
    for sample_index, location_index in enumerate(location_truth):
        location = LOCATION_LABELS_V1[int(location_index)]
        for action_index, action in enumerate(ACTION_LABELS_V1):
            effect_truth, future_truth = semantic_effect_teacher(location, action)
            effect_predicted = SEMANTIC_EFFECT_LABELS_V1[
                int(output["effect_probability"][sample_index, action_index].argmax())
            ]
            future_predicted = LOCATION_LABELS_V1[
                int(output["future_probability"][sample_index, action_index].argmax())
            ]
            effect_correct[effect_truth].append(effect_predicted == effect_truth)
            future_correct[future_truth].append(future_predicted == future_truth)

    frontend = json.loads(args.frontend_audit.read_text())
    development_standard = float(
        gate_spec["gates"]["correct_singleton_route_one_sided_95_lower_each_class"]
    )
    correct_route_by_class = class_rates(
        correct_singleton_array, action_truth, ACTION_LABELS_V1
    )
    location_retention_by_class = class_rates(
        location_retained, location_truth, LOCATION_LABELS_V1
    )
    action_retention_by_class = class_rates(action_retained, action_truth, ACTION_LABELS_V1)
    effect_metrics = {
        label: {"correct": int(sum(values)), "trials": len(values), "accuracy": float(np.mean(values))}
        for label, values in effect_correct.items()
    }
    future_metrics = {
        label: {"correct": int(sum(values)), "trials": len(values), "accuracy": float(np.mean(values))}
        for label, values in future_correct.items()
    }
    gates = {
        "frontend_truth_node_proposal_rate": frontend["truth_node_proposal_rate"] >= gate_spec["gates"]["frontend_truth_node_proposal_rate"],
        "frontend_query_leaks": frontend["query_leaks"] == gate_spec["gates"]["frontend_query_leaks"],
        "location_truth_retention_each_class": min(value["rate"] for value in location_retention_by_class.values()) >= gate_spec["gates"]["location_truth_retention_each_class"],
        "action_truth_retention_each_class": min(value["rate"] for value in action_retention_by_class.values()) >= gate_spec["gates"]["action_truth_retention_each_class"],
        "correct_singleton_route_lower_each_class": min(value["one_sided_95_lower"] for value in correct_route_by_class.values()) >= development_standard,
        "false_singleton_routes": int(false_singleton_array.sum()) == gate_spec["gates"]["false_singleton_routes"],
        "mean_location_prediction_set_size": float(np.mean([len(value) for value in location_sets])) <= gate_spec["gates"]["mean_location_prediction_set_size_max"],
        "mean_action_prediction_set_size": float(np.mean([len(value) for value in action_sets])) <= gate_spec["gates"]["mean_action_prediction_set_size_max"],
        "semantic_effect_accuracy_each_class": min(value["accuracy"] for value in effect_metrics.values()) >= gate_spec["gates"]["semantic_effect_accuracy_each_class"],
        "future_belief_accuracy_each_class": min(value["accuracy"] for value in future_metrics.values()) >= gate_spec["gates"]["future_belief_accuracy_each_class"],
        "node_hit_at_1": float(np.mean(node_hits)) >= gate_spec["gates"]["node_hit_at_1_min"],
        "online_oracle_read_count": sum(trace["online_oracle_reads"] for trace in traces) == gate_spec["gates"]["online_oracle_read_count"],
    }
    rank_argmax = output["action_probability"].argmax(axis=-1)
    optimizer = output["optimizer_selected"]
    report = {
        "schema_version": "interaction-uncertainty.piu-object-clean-evaluation.v1",
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "decision": "GO" if all(gates.values()) else "NOT-GO",
        "gates": gates,
        "gate_spec": {"path": str(args.gates.relative_to(ROOT)), "sha256": digest(args.gates)},
        "model": {"path": str(args.model.relative_to(ROOT)), "sha256": digest(args.model)},
        "sources": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
            for name, path in (
                ("snapshots", args.snapshots),
                ("labels", args.labels),
                ("scenes", args.scenes),
                ("object_features", args.object_features),
                ("prefix_features", args.prefix_features),
                ("frontend_audit", args.frontend_audit),
            )
        },
        "samples": len(split["ids"]),
        "class_counts": dict(Counter(ACTION_LABELS_V1[int(value)] for value in action_truth)),
        "location_truth_retention": location_retention_by_class,
        "action_truth_retention": action_retention_by_class,
        "correct_singleton_route": correct_route_by_class,
        "false_singleton_routes": int(false_singleton_array.sum()),
        "safe_stop": {"count": route.count("SAFE_STOP"), "rate": route.count("SAFE_STOP") / len(route)},
        "mean_prediction_set_size": {
            "location": float(np.mean([len(value) for value in location_sets])),
            "action": float(np.mean([len(value) for value in action_sets])),
        },
        "node_hit_at_1": float(np.mean(node_hits)),
        "semantic_effect": effect_metrics,
        "future_belief": future_metrics,
        "rank_head_argmax_accuracy": float(np.mean(rank_argmax == action_truth)),
        "explicit_optimizer_argmax_accuracy": float(np.mean(optimizer == action_truth)),
        "unnecessary_interaction_rate": float(
            np.mean([selected == "OPEN_TO_INSPECT" for selected, truth in zip(route, action_truth, strict=True) if ACTION_LABELS_V1[int(truth)] == "DIRECT_ACT"])
        ),
        "baselines": {
            "always_DIRECT_ACT": float(np.mean(action_truth == ACTION_LABELS_V1.index("DIRECT_ACT"))),
            "always_OPEN_TO_INSPECT": float(np.mean(action_truth == ACTION_LABELS_V1.index("OPEN_TO_INSPECT"))),
            "prefix_zeroed_optimizer": float(np.mean(prefix_zeroed["optimizer_selected"] == action_truth)),
            "node_zeroed_optimizer": float(np.mean(nodes_zeroed["optimizer_selected"] == action_truth)),
        },
        "original_0.90_standard_met_each_class": all(value["meets_original_0.90"] for value in correct_route_by_class.values()),
        "online_oracle_read_count": 0,
        "partially_passed_is_not_go": True,
        "traces": traces,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("decision", "gates", "correct_singleton_route", "safe_stop", "node_hit_at_1", "semantic_effect", "baselines")}, indent=2))


if __name__ == "__main__":
    main()
