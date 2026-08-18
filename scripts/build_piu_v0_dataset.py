#!/usr/bin/env python3
"""Build a unified PIU V0 training index without copying oracle data to inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def v9_outcome(row: dict) -> str:
    evaluator = row["evaluator_only"]
    target_maximum = max(
        max(point["target_pixels"].values())
        for point in evaluator["visibility_history"]
    )
    if (
        not row["full_executor"]
        or not evaluator["drawer_opened"]
        or row["return_status"]["phase"] != "COMPLETE"
    ):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--belief-data",
        type=Path,
        default=ROOT / "data/calibration/t01_prompt_state_v1.jsonl",
    )
    parser.add_argument(
        "--effect-data",
        type=Path,
        default=ROOT / "data/calibration/t01_open_and_observe_effect_v3.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/piu_v0/training_index.jsonl",
    )
    args = parser.parse_args()
    for name in ("belief_data", "effect_data", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    manifest = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest.exists():
        raise FileExistsError("PIU training index and manifest are immutable")
    belief_rows = [
        json.loads(line) for line in args.belief_data.read_text().splitlines() if line
    ]
    effect_rows = [
        json.loads(line) for line in args.effect_data.read_text().splitlines() if line
    ]
    location = {
        "closed_hidden_butter": "middle_drawer",
        "closed_visible_cream_cheese": "visible_workspace",
        "open_visible_butter": "visible_workspace",
    }
    indexed = []
    for row_index, row in enumerate(belief_rows):
        indexed.append(
            {
                "schema_version": "interaction-uncertainty.piu-training-sample.v0",
                "sample_id": f"belief:{row['condition']}:{int(row['seed']):04d}",
                "sample_type": "INITIAL_TASK_BELIEF",
                "seed": int(row["seed"]),
                "split": row["split"],
                "scenario_group": row["condition"],
                "prompt": row["prompt"],
                "policy_inputs": {
                    "image_paths": row["image_paths"],
                    "image_sha256": row["image_sha256"],
                    "frozen_feature_store": "outputs/t01_prompt_state_v1/pi05_prefix_embeddings.npz",
                    "frozen_feature_row": row_index,
                },
                "candidate_actions": [
                    "DIRECT_ACT",
                    "OPEN_TO_INSPECT(drawer_middle)",
                    "ABSTAIN",
                ],
                "offline_labels": {
                    "target_location": location[row["condition"]],
                    "preferred_semantic_handoff": (
                        "OPEN_TO_INSPECT"
                        if row["condition"] == "closed_hidden_butter"
                        else "DIRECT_ACT"
                    ),
                    "simulator_teacher_only": True,
                },
                "online_oracle_inputs": [],
            }
        )
    future_location = {
        "FAILED": "middle_drawer",
        "REVEALED": "visible_workspace",
        "EMPTY": "other_unsearched_region",
    }
    for row_index, row in enumerate(effect_rows):
        outcome = v9_outcome(row)
        indexed.append(
            {
                "schema_version": "interaction-uncertainty.piu-training-sample.v0",
                "sample_id": f"effect:{row['regime']}:{int(row['seed']):04d}",
                "sample_type": "ACTION_EFFECT_AND_OUTCOME",
                "seed": int(row["seed"]),
                "split": row["split"],
                "scenario_group": row["regime"],
                "prompt": row["final_prompt"],
                "query_prompt": "Find the butter",
                "action": {
                    "primitive": "OPEN_TO_INSPECT",
                    "target_id": "drawer_middle",
                    "execution_budget_fraction": (
                        1.0 if row["full_executor"] else row["open_steps"] / 300.0
                    ),
                },
                "policy_inputs": {
                    "public_history": row["public_history"],
                    "frozen_feature_store": "outputs/t01_open_and_observe_effect_v3/pi05_temporal_embeddings_v5.npz",
                    "frozen_feature_row": row_index,
                },
                "offline_labels": {
                    "effect_outcome_v9": outcome,
                    "future_target_location": future_location[outcome],
                    "minimum_resolvable_pixels": 256,
                    "simulator_teacher_only": True,
                },
                "online_oracle_inputs": [],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in indexed)
    )
    report = {
        "schema_version": "interaction-uncertainty.piu-training-manifest.v0",
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dataset": str(args.output.relative_to(ROOT)),
        "dataset_sha256": digest(args.output),
        "samples": len(indexed),
        "sample_types": Counter(row["sample_type"] for row in indexed),
        "splits": Counter(row["split"] for row in indexed),
        "sources": {
            str(args.belief_data.relative_to(ROOT)): digest(args.belief_data),
            str(args.effect_data.relative_to(ROOT)): digest(args.effect_data),
        },
        "policy_input_contract": [
            "stock agentview RGB",
            "stock wrist RGB",
            "public robot state",
            "prompt",
            "public action/history",
        ],
        "online_oracle_inputs": [],
        "label_only_privileged_inputs": [
            "segmentation visibility after controller termination",
            "drawer joint after controller termination",
            "registered simulator scenario condition",
        ],
        "claim_status": "prototype/development index; no clean or sealed evidence",
    }
    manifest.write_text(json.dumps(report, indent=2, default=dict) + "\n")
    print(json.dumps(report, indent=2, default=dict))


if __name__ == "__main__":
    main()
