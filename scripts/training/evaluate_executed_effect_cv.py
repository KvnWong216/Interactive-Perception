#!/usr/bin/env python3
"""Evaluate real effect supervision with grouped CPU-only cross-validation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from calibrated_interaction.contracts import CandidateAction, EffectFactor
from calibrated_interaction.cpu_baseline import (
    candidate_token,
    context_tokens,
)


def load_pilot_module() -> Any:
    path = ROOT / "scripts/training/train_calibrated_interaction.py"
    spec = importlib.util.spec_from_file_location("calibrated_pilot_trainer", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve().relative_to(ROOT)), "sha256": sha256(path)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_candidates(path: Path) -> tuple[CandidateAction, ...]:
    value = yaml.safe_load(path.read_text())
    return tuple(CandidateAction.from_mapping(row) for row in value["candidates"])


def load_context_dataset(
    dataset_path: Path,
    candidates: tuple[CandidateAction, ...],
    unsupported_factors: set[EffectFactor],
) -> dict[str, Any]:
    rows = read_jsonl(dataset_path)
    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["initial_state_id"], row["prompt"])
        groups.setdefault(key, []).append(row)
    values: dict[str, list[Any]] = {
        key: []
        for key in ("ids", "seed", "context", "candidates", "route", "effects", "effect_mask")
    }
    fixed_candidate_tokens = np.stack(
        [candidate_token(candidate.to_dict()) for candidate in candidates]
    )
    for (state_id, prompt), branches in sorted(groups.items()):
        branch_by_candidate = {row["executed_candidate"]: row for row in branches}
        if tuple(branch_by_candidate) != candidate_ids:
            raise ValueError(f"{state_id}: candidate branch order/content mismatch")
        routes = {row["route_label"] for row in branches}
        frames = {tuple(row["observation_frames"]) for row in branches}
        if len(routes) != 1 or len(frames) != 1:
            raise ValueError(f"{state_id}: branch policy inputs or routes differ")
        frame_pair = next(iter(frames))
        if len(frame_pair) != 2:
            raise ValueError("CPU baseline requires agentview and wrist")
        effect = np.zeros((len(candidates), len(EffectFactor)), dtype=np.float32)
        mask = np.ones_like(effect, dtype=bool)
        for candidate_index, candidate_id in enumerate(candidate_ids):
            branch = branch_by_candidate[candidate_id]
            for factor_index, factor in enumerate(EffectFactor):
                effect[candidate_index, factor_index] = float(
                    branch["effect_labels"][factor.value]
                )
                if factor in unsupported_factors:
                    mask[candidate_index, factor_index] = False
        seed = int(branches[0]["privileged_metadata_for_evaluation_only"]["seed"])
        values["ids"].append(f"{state_id}:{hashlib.sha256(prompt.encode()).hexdigest()[:8]}")
        values["seed"].append(seed)
        values["context"].append(
            context_tokens(
                prompt=prompt,
                agentview=ROOT / frame_pair[0],
                wrist=ROOT / frame_pair[1],
            )
        )
        values["candidates"].append(fixed_candidate_tokens)
        values["route"].append(candidate_ids.index(next(iter(routes))))
        values["effects"].append(effect)
        values["effect_mask"].append(mask)
    return {
        "ids": values["ids"],
        "seed": np.asarray(values["seed"], dtype=np.int64),
        "context": np.stack(values["context"]).astype(np.float32),
        "candidates": np.stack(values["candidates"]).astype(np.float32),
        "route": np.asarray(values["route"], dtype=np.int64),
        "effects": np.stack(values["effects"]).astype(np.float32),
        "effect_mask": np.stack(values["effect_mask"]).astype(bool),
        "candidate_ids": candidate_ids,
    }


def subset(dataset: dict[str, Any], seeds: set[int]) -> dict[str, Any]:
    selected = np.asarray([int(seed) in seeds for seed in dataset["seed"]])
    return {
        key: (
            value[selected]
            if isinstance(value, np.ndarray) and value.shape[:1] == selected.shape
            else [item for item, keep in zip(value, selected, strict=True) if keep]
            if key == "ids"
            else value
        )
        for key, value in dataset.items()
    }


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def effect_metrics(logits: np.ndarray, split: dict[str, Any], pilot: Any) -> dict[str, Any]:
    probabilities = sigmoid(logits)
    labels = split["effects"]
    mask = split["effect_mask"]
    predicted = probabilities >= 0.5
    selected = mask
    result: dict[str, Any] = {
        "bce": pilot.effect_bce(logits, labels, mask),
        "micro_accuracy": float(np.mean(predicted[selected] == labels[selected])),
        "micro_brier": float(np.mean((probabilities[selected] - labels[selected]) ** 2)),
    }
    exact = []
    for sample_index in range(len(labels)):
        for candidate_index in range(labels.shape[1]):
            current = mask[sample_index, candidate_index]
            exact.append(
                bool(
                    np.all(
                        predicted[sample_index, candidate_index, current]
                        == labels[sample_index, candidate_index, current]
                    )
                )
            )
    result["candidate_factor_exact_match"] = float(np.mean(exact))
    per_factor = {}
    for factor_index, factor in enumerate(EffectFactor):
        current = mask[:, :, factor_index]
        if not current.any():
            per_factor[factor.value] = {"supported": False}
            continue
        truth = labels[:, :, factor_index][current].astype(bool)
        guess = predicted[:, :, factor_index][current]
        positive = truth
        negative = ~truth
        balanced = (
            0.5
            * (
                float(np.mean(guess[positive]))
                + float(np.mean(~guess[negative]))
            )
            if positive.any() and negative.any()
            else None
        )
        per_factor[factor.value] = {
            "supported": True,
            "positives": int(positive.sum()),
            "negatives": int(negative.sum()),
            "accuracy": float(np.mean(guess == truth)),
            "balanced_accuracy": balanced,
            "brier": float(
                np.mean((probabilities[:, :, factor_index][current] - truth) ** 2)
            ),
        }
    result["per_factor"] = per_factor
    return result


def prompt_ablation(split: dict[str, Any], *, mode: str) -> dict[str, Any]:
    result = copy.deepcopy(split)
    if mode == "remove":
        result["context"][:, 2] = 0.0
        result["context"][:, 3] = 0.5 * (
            result["context"][:, 0] + result["context"][:, 1]
        )
    elif mode == "swap":
        for seed in sorted(set(result["seed"].tolist())):
            indices = np.flatnonzero(result["seed"] == seed)
            if len(indices) != 2:
                raise ValueError(f"seed {seed}: expected prompt pair")
            result["context"][indices, 2:] = split["context"][indices[::-1], 2:]
    else:
        raise ValueError(mode)
    return result


def mean_std(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
        }
        for key in rows[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name in ("dataset", "manifest", "candidates", "output", "model"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if not args.force:
        for path in (args.output, args.model):
            if path.exists():
                raise FileExistsError(path)
    manifest = json.loads(args.manifest.read_text())
    if sha256(args.dataset) != manifest["dataset"]["sha256"]:
        raise ValueError("dataset manifest hash mismatch")
    unsupported = {
        EffectFactor(value) for value in manifest["constant_unsupported_factors"]
    }
    candidates = load_candidates(args.candidates)
    dataset = load_context_dataset(args.dataset, candidates, unsupported)
    seeds = sorted(set(dataset["seed"].tolist()))
    if len(seeds) != 10:
        raise ValueError("frozen development CV requires exactly ten seed groups")
    folds = []
    init_seeds = (20260822, 20260823, 20260824)
    pilot = load_pilot_module()
    torch.set_num_threads(8)
    selected_models = []
    for fold_index in range(5):
        test_seeds = set(seeds[2 * fold_index : 2 * fold_index + 2])
        validation_start = (2 * (fold_index + 1)) % len(seeds)
        validation_seeds = {
            seeds[validation_start],
            seeds[(validation_start + 1) % len(seeds)],
        }
        train_seeds = set(seeds) - test_seeds - validation_seeds
        splits = {
            "train": subset(dataset, train_seeds),
            "validation": subset(dataset, validation_seeds),
            "test": subset(dataset, test_seeds),
        }
        variants = {}
        fold_models = {}
        for name, supervised in (
            ("B6_route_only", False),
            ("B7_executed_effect_route", True),
        ):
            runs = []
            models = []
            for init_seed in init_seeds:
                model, training = pilot.fit(
                    splits["train"],
                    splits["validation"],
                    seed=init_seed,
                    effect_supervision=supervised,
                    epochs=args.epochs,
                )
                output = pilot.forward_numpy(model, splits["test"])
                training["test_route"] = pilot.route_metrics(
                    output["route_logits"], splits["test"]["route"]
                )
                if supervised:
                    training["test_effect"] = effect_metrics(
                        output["effect_logits"], splits["test"], pilot
                    )
                runs.append(training)
                models.append(model)
            selected = min(
                range(len(runs)),
                key=lambda index: runs[index]["development_objective"],
            )
            model = models[selected]
            fold_models[name] = model
            ablations = {}
            for ablation_name, ablated in (
                ("unaltered", splits["test"]),
                ("prompt_removed", prompt_ablation(splits["test"], mode="remove")),
                ("prompt_swapped", prompt_ablation(splits["test"], mode="swap")),
            ):
                output = pilot.forward_numpy(model, ablated)
                ablations[ablation_name] = pilot.route_metrics(
                    output["route_logits"], ablated["route"]
                )
            variants[name] = {
                "runs": runs,
                "selected_initialization": runs[selected]["seed"],
                "selected_test_route": runs[selected]["test_route"],
                "selected_test_effect": runs[selected].get("test_effect"),
                "prompt_ablations": ablations,
            }
        selected_models.append(fold_models["B7_executed_effect_route"].state_dict())
        folds.append(
            {
                "fold": fold_index,
                "train_seeds": sorted(train_seeds),
                "validation_seeds": sorted(validation_seeds),
                "test_seeds": sorted(test_seeds),
                "variants": variants,
            }
        )
    summaries = {}
    for name in ("B6_route_only", "B7_executed_effect_route"):
        route_rows = [fold["variants"][name]["selected_test_route"] for fold in folds]
        summaries[name] = {
            "route": mean_std(route_rows),
            "prompt_ablation_accuracy": {
                ablation: {
                    "mean": float(
                        np.mean(
                            [
                                fold["variants"][name]["prompt_ablations"][ablation][
                                    "accuracy"
                                ]
                                for fold in folds
                            ]
                        )
                    )
                }
                for ablation in ("unaltered", "prompt_removed", "prompt_swapped")
            },
        }
    effect_rows = [
        fold["variants"]["B7_executed_effect_route"]["selected_test_effect"]
        for fold in folds
    ]
    summaries["B7_executed_effect_route"]["effect"] = {
        "overall": mean_std(
            [
                {
                    key: row[key]
                    for key in (
                        "bce",
                        "micro_accuracy",
                        "micro_brier",
                        "candidate_factor_exact_match",
                    )
                }
                for row in effect_rows
            ]
        ),
        "per_factor": {
            factor.value: {
                metric: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
                for metric in ("accuracy", "balanced_accuracy", "brier")
                if (
                    values := [
                        row["per_factor"][factor.value][metric]
                        for row in effect_rows
                        if row["per_factor"][factor.value].get(metric) is not None
                    ]
                )
            }
            if factor not in unsupported
            else {"supported": False}
            for factor in EffectFactor
        },
    }
    deltas = [
        fold["variants"]["B7_executed_effect_route"]["selected_test_route"]["macro_f1"]
        - fold["variants"]["B6_route_only"]["selected_test_route"]["macro_f1"]
        for fold in folds
    ]
    comparison = {
        "metric": "test macro_f1",
        "per_fold_delta_B7_minus_B6": deltas,
        "mean_delta": float(np.mean(deltas)),
        "improved_folds": sum(value > 0 for value in deltas),
        "tied_folds": sum(value == 0 for value in deltas),
        "worse_folds": sum(value < 0 for value in deltas),
    }
    decision = (
        "REJECT_EFFECT_HEAD_AS_ROUTE_CONTRIBUTION"
        if comparison["mean_delta"] <= 0.0
        else "DEVELOPMENT_SIGNAL_ONLY"
    )
    args.model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_family": "CPU public-input baseline; not frozen-VLM main method",
            "fold_state_dicts": selected_models,
            "candidate_ids": [candidate.candidate_id for candidate in candidates],
            "unsupported_effect_factors": sorted(factor.value for factor in unsupported),
        },
        args.model,
    )
    report = {
        "schema_version": "calibrated-interaction.executed-effect-cv.v1",
        "repository_commit_before_cycle": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "status": "DEVELOPMENT_GROUP_CV_ONLY",
        "decision": decision,
        "device": "cpu",
        "gpu_visible_to_process": False,
        "feature_baseline": (
            "fixed public RGB pixels/histograms/edges plus signed text hashing; "
            "not the shared frozen VLM"
        ),
        "grouping": "all six prompt/action forks of each seed stay in one fold",
        "formal_calibration_claim": False,
        "formal_calibration_blocker": (
            "Only ten inspected seed groups; two validation groups per fold are "
            "insufficient for a paper-level conformal guarantee."
        ),
        "samples": len(dataset["ids"]),
        "seed_groups": len(seeds),
        "effect_masked_as_unsupported": sorted(
            factor.value for factor in unsupported
        ),
        "folds": folds,
        "summary": summaries,
        "effect_route_comparison": comparison,
        "sources": {
            "dataset": artifact(args.dataset),
            "manifest": artifact(args.manifest),
            "candidates": artifact(args.candidates),
            "model": artifact(args.model),
            "cpu_features": artifact(ROOT / "src/calibrated_interaction/cpu_baseline.py"),
            "decoder": artifact(ROOT / "src/calibrated_interaction/model.py"),
            "trainer": artifact(Path(__file__).resolve()),
        },
        "online_oracle_inputs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": artifact(args.output)["path"],
                "decision": decision,
                "summary": summaries,
                "comparison": comparison,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
