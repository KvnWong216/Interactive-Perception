#!/usr/bin/env python3
"""Create an immutable, compact summary of fixed-scenario executor reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def summarize(report: dict[str, Any], *, drawer_threshold: float) -> dict[str, Any]:
    evaluator = report["evaluator"]
    controller = report["controller"]
    target = evaluator.get("target_object")
    objects = evaluator.get("objects", {})
    target_metrics = objects.get(target, {})
    wrong_objects = {
        name: {
            "minimum_eef_distance_m": values["minimum_eef_distance_m"],
            "grasp_contact_steps": values["grasp_contact_steps"],
            "maximum_lift_m": values["maximum_lift_m"],
        }
        for name, values in objects.items()
        if name != target
    }
    joints = evaluator.get("joints", {})
    drawer = joints.get("wooden_cabinet_1_middle_level")
    return {
        "seed": report["seed"],
        "prompt": report["prompt"],
        "subtask_steps": controller["subtask_steps"],
        "return_phase": controller["return_phase"],
        "online_oracle_inputs": controller["online_oracle_inputs"],
        "task_success": evaluator["task_success"],
        "target_object": target,
        "target_pick_success": evaluator["target_pick_success"],
        "target_reached_destination": evaluator.get("target_reached_destination"),
        "target_minimum_eef_distance_m": target_metrics.get("minimum_eef_distance_m"),
        "target_grasp_contact_steps": target_metrics.get("grasp_contact_steps"),
        "target_maximum_lift_m": target_metrics.get("maximum_lift_m"),
        "wrong_objects": wrong_objects,
        "drawer": drawer,
        "drawer_open_state_observed": bool(
            drawer and drawer["minimum"] <= drawer_threshold
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = resolve(args.config)
    output_path = resolve(args.output)
    if output_path.exists():
        raise FileExistsError("summary outputs are immutable")
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != (
        "calibrated-interaction.executor-qualification.v1"
    ):
        raise ValueError("unsupported executor qualification schema")
    rows = []
    for specification in config["runs"]:
        report_path = resolve(Path(specification["report"]))
        report = json.loads(report_path.read_text())
        if report.get("schema_version") != "piu.semantic-option.v1":
            raise ValueError(f"unsupported report schema: {report_path}")
        values = summarize(
            report, drawer_threshold=float(config["drawer_open_threshold"])
        )
        rows.append(
            {
                "id": specification["id"],
                "condition": specification["condition"],
                "candidate": specification["candidate"],
                "source": {
                    "path": str(report_path.relative_to(ROOT)),
                    "sha256": digest(report_path),
                },
                **values,
            }
        )
    result = {
        "schema_version": "calibrated-interaction.executor-summary.v1",
        "qualification_id": config["id"],
        "scenario": config["scenario"],
        "claim_scope": config["claim_scope"],
        "drawer_open_threshold": config["drawer_open_threshold"],
        "runs": rows,
        "aggregate": {
            "run_count": len(rows),
            "task_success_count": sum(row["task_success"] for row in rows),
            "target_pick_success_count": sum(
                row["target_pick_success"] for row in rows
            ),
            "target_destination_success_count": sum(
                row["target_reached_destination"] is True for row in rows
            ),
            "open_candidate_success_count": sum(
                row["candidate"] == "OPEN" and row["drawer_open_state_observed"]
                for row in rows
            ),
            "online_oracle_input_run_count": sum(
                bool(row["online_oracle_inputs"]) for row in rows
            ),
        },
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": digest(config_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
