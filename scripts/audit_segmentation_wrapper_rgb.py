#!/usr/bin/env python3
"""Verify that evaluator segmentation does not alter policy-visible RGB."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from interactive_perception.policy_client import build_observation  # noqa: E402

DUMMY_ACTION = [0.0] * 6 + [-1.0]
PROMPT = "Place the butter in the basket"


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def packet_for(env_class, bddl: Path, seed: int):
    env = env_class(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    try:
        env.seed(seed)
        obs = env.reset()
        for _ in range(10):
            obs, _, _, _ = env.step(DUMMY_ACTION)
        packet = build_observation(obs, PROMPT)
        return packet.image.copy(), packet.wrist_image.copy(), packet.state.copy()
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(400, 410)))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/calibration/segmentation_wrapper_rgb_audit_v1.json",
    )
    args = parser.parse_args()
    if args.seeds != list(range(400, 410)):
        raise ValueError("frozen wrapper-audit seeds are 400-409")
    if not args.output.is_absolute():
        args.output = ROOT / args.output

    from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv

    scenes = (
        ROOT / "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl",
        ROOT / "scenarios/t01_stock_ladder_v1/T01F_middle_drawer_empty_calibration.bddl",
    )
    rows = []
    for bddl in scenes:
        for seed in args.seeds:
            offscreen = packet_for(OffScreenRenderEnv, bddl, seed)
            segmented = packet_for(SegmentationRenderEnv, bddl, seed)
            equal = [np.array_equal(left, right) for left, right in zip(offscreen, segmented, strict=True)]
            row = {
                "scene": str(bddl.relative_to(ROOT)),
                "seed": seed,
                "agentview_equal": equal[0],
                "wrist_equal": equal[1],
                "robot_state_equal": equal[2],
                "offscreen_sha256": {
                    "agentview": array_hash(offscreen[0]),
                    "wrist": array_hash(offscreen[1]),
                    "state": array_hash(offscreen[2]),
                },
                "segmentation_sha256": {
                    "agentview": array_hash(segmented[0]),
                    "wrist": array_hash(segmented[1]),
                    "state": array_hash(segmented[2]),
                },
            }
            rows.append(row)
            print(f"{bddl.name} seed={seed} equal={all(equal)}", flush=True)
    passed = all(
        row["agentview_equal"] and row["wrist_equal"] and row["robot_state_equal"]
        for row in rows
    )
    report = {
        "schema_version": "interactive-perception.segmentation-wrapper-rgb-audit.v1",
        "claim": "SegmentationRenderEnv preserves the stock policy packet",
        "seeds": args.seeds,
        "pairs": len(rows),
        "passed": passed,
        "policy_inputs_compared": ["agentview RGB", "wrist RGB", "robot state"],
        "segmentation_use": "evaluator labels only",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"wrapper audit is immutable: {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pairs": len(rows), "passed": passed}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
