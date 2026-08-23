#!/usr/bin/env python3
"""Replay retained candidate actions with a task-specific evaluator target.

The policy trajectory is not rerun or changed. This creates a separate,
evaluator-only relabel artifact so one physical candidate fork can supervise
multiple prompt counterfactuals without exposing privileged state online.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/infra"), str(ROOT / "src")]
importlib.import_module("bootstrap")


def load_evaluator_replay() -> Any:
    path = ROOT / "scripts/pipeline/execute.py"
    spec = importlib.util.spec_from_file_location("piu_pipeline_execute", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluator_replay


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def initial_state(bddl: Path, seed: int) -> np.ndarray:
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
    )
    try:
        env.seed(seed)
        env.reset()
        return np.asarray(env.get_sim_state(), dtype=float).copy()
    finally:
        env.close()


def relabel(
    config_path: Path,
    *,
    condition_name: str,
    target_object: str,
    target_destination_region: str,
) -> dict[str, Any]:
    evaluator_replay = load_evaluator_replay()
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != "calibrated-interaction.paper-cycle.v2":
        raise ValueError("unsupported experiment schema")
    bddl = ROOT / config["scenario"]
    condition = config["conditions"][condition_name]
    rows = []
    for seed in config["seeds"]:
        report_path = (
            ROOT
            / config["run_root"]
            / f"seed{seed}"
            / condition["directory"]
            / "report.json"
        )
        source = json.loads(report_path.read_text())
        action_path = ROOT / source["controller"]["action_history"]
        actions = json.loads(action_path.read_text())
        evaluator = evaluator_replay(
            bddl=bddl,
            seed=int(seed),
            initial_state=initial_state(bddl, int(seed)),
            actions=actions,
            subtask_steps=int(source["controller"]["subtask_steps"]),
            target_object=target_object,
            target_destination_region=target_destination_region,
            tracked_objects=("butter_1", "cream_cheese_1"),
            tracked_joints=(
                (condition["tracked_joint"],) if condition.get("tracked_joint") else ()
            ),
            metric_contract_version="v1",
        )
        rows.append(
            {
                "seed": int(seed),
                "source_report": {
                    "path": portable(report_path),
                    "sha256": sha256(report_path),
                },
                "source_action_history": {
                    "path": portable(action_path),
                    "sha256": sha256(action_path),
                },
                "evaluator": evaluator,
            }
        )
    return {
        "schema_version": "calibrated-interaction.task-specific-relabel.v1",
        "experiment_id": config["id"],
        "source_condition": condition_name,
        "candidate": (
            "open_middle_drawer"
            if condition["role"] == "OPEN_CONTAINER"
            else "direct_requested_to_basket"
        ),
        "target_object": target_object,
        "target_destination_region": target_destination_region,
        "policy_rerun": False,
        "policy_inputs_changed": False,
        "evaluator_only": True,
        "config": {"path": portable(config_path), "sha256": sha256(config_path)},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiments/original_drawer_paper_cycle_v2.yaml",
    )
    parser.add_argument("--target-object", required=True)
    parser.add_argument("--target-destination-region", required=True)
    parser.add_argument("--condition", default="open_closed_drawer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and not args.force:
        raise FileExistsError(output)
    result = relabel(
        config,
        condition_name=args.condition,
        target_object=args.target_object,
        target_destination_region=args.target_destination_region,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"output": portable(output), "rows": len(result["rows"])}, indent=2)
    )


if __name__ == "__main__":
    main()
