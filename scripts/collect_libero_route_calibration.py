#!/usr/bin/env python3
"""Collect calibration-only LIBERO chunks for embodied inspection routes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import _bootstrap  # noqa: F401,E402

from collect_libero_intent_calibration import (  # noqa: E402
    SPLIT_BY_SEED,
    collect_condition,
)
from interactive_perception.policy_client import OpenPiWebsocketPolicy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "benchmarks/calibration/route_intents_v2.yaml",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--samples-per-observation", type=int, default=8)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/calibration/libero_routes_v2.jsonl"
    )
    args = parser.parse_args()

    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    if spec.get("calibration_only") is not True:
        raise SystemExit("route collection spec must be marked calibration_only: true")
    conditions = spec["conditions"]
    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port)
    bddl_root = Path(get_libero_path("bddl_files"))
    rows = []
    total = len(conditions) * len(SPLIT_BY_SEED)
    for condition in conditions:
        label = str(condition["label"])
        prompt = str(condition["prompt"])
        if condition.get("project_bddl"):
            bddl = ROOT / str(condition["project_bddl"])
        else:
            bddl = bddl_root / str(condition["suite"]) / str(condition["bddl"])
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
        )
        try:
            for seed in SPLIT_BY_SEED:
                row = collect_condition(
                    env=env,
                    policy=policy,
                    prompt=prompt,
                    label=label,
                    seed=seed,
                    samples=args.samples_per_observation,
                    wait_steps=args.wait_steps,
                    primary_camera=str(spec["policy_primary_camera"]),
                    wrist_camera=str(spec["policy_wrist_camera"]),
                    wrist_camera_initialization=spec["wrist_camera_initialization"],
                )
                row.update(
                    {
                        "bddl": bddl.name,
                        "suite": str(condition.get("suite", "project_calibration")),
                        "calibration_only": True,
                    }
                )
                rows.append(row)
                print(f"[{len(rows):03d}/{total}] {label} seed={seed}", flush=True)
        finally:
            env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "schema_version": "interactive-perception.intent-calibration-manifest.v1",
        "dataset": str(args.output),
        "dataset_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "policy": "pi05_libero",
        "samples": len(rows),
        "chunks_per_sample": args.samples_per_observation,
        "labels": [str(item["label"]) for item in conditions],
        "split_rule": "seed 0:20 train, 20:40 calibration, 40:50 validation, per class",
        "calibration_only": True,
        "oracle_inputs": [],
        "policy_inputs": ["stock agentview RGB", "forward-facing wrist RGB", "robot state", "prompt"],
        "semantic_definition": {
            "ACT": "execute the final visible-object task",
            "REMOVE_OCCLUDER": "manipulate a physical occluder or container",
            "ROTATE": "grasp, wrist-rotate for a new object view, then place down",
            "MOVE_CLOSER": "grasp and move the end effector toward the camera for inspection",
        },
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
