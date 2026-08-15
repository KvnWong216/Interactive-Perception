#!/usr/bin/env python3
"""Validate dual-camera occlusion and oracle reveal in stock-aligned scenes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from validate_information_enrichment_v0 import (  # noqa: E402
    instance_stats,
    object_qpos,
    regenerate_obs,
    set_object_qpos,
)


def visibility(env, obs) -> dict[str, int]:
    return {
        camera: int(instance_stats(env, obs, camera, ["ketchup_1"])["ketchup_1"]["visible_pixels"])
        for camera in ("agentview", "robot0_eye_in_hand")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-initial-pixels", type=int, default=400)
    parser.add_argument("--min-revealed-pixels", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from libero.libero.envs import SegmentationRenderEnv

    bddl = ROOT / "scenarios/stock_aligned_v1/T06_dual_camera_occluded_ketchup.bddl"
    rows = []
    for seed in args.seeds:
        env = SegmentationRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256, initialization_noise=None)
        try:
            env.seed(seed)
            obs = env.reset()
            initial = visibility(env, obs)
            reveal_positions = {
                "milk_1": (-0.20, -0.08),
                "salad_dressing_1": (-0.15, 0.06),
            }
            changes = {}
            for name, xy in reveal_positions.items():
                before = object_qpos(env, name)
                after = before.copy()
                after[:2] = np.asarray(xy)
                set_object_qpos(env, name, after)
                changes[name] = {"before": before.tolist(), "after": after.tolist()}
            revealed = visibility(env, regenerate_obs(env))
            passed = bool(
                all(value <= args.max_initial_pixels for value in initial.values())
                and all(value >= args.min_revealed_pixels for value in revealed.values())
            )
            rows.append({"seed": seed, "initial_target_pixels": initial, "revealed_target_pixels": revealed, "oracle_changes": changes, "passed": passed})
            print(f"seed={seed} initial={initial} revealed={revealed} passed={passed}", flush=True)
        finally:
            env.close()
    report = {
        "scene": str(bddl.relative_to(ROOT)), "policy_cameras": ["agentview", "robot0_eye_in_hand"],
        "max_initial_pixels": args.max_initial_pixels, "min_revealed_pixels": args.min_revealed_pixels,
        "rows": rows, "all_passed": all(row["passed"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
