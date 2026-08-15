#!/usr/bin/env python3
"""Find a single-object occlusion placement in the stock LIBERO ketchup task."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from run_repro_gate import load_init_states  # noqa: E402
from validate_information_enrichment_v0 import instance_stats, object_qpos, regenerate_obs, set_object_qpos  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occluder", default="milk_1")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    from libero.libero import benchmark as libero_benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import SegmentationRenderEnv

    suite = libero_benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(4)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = SegmentationRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    try:
        env.reset()
        env.set_init_state(load_init_states(suite, 4)[0])
        target = object_qpos(env, "ketchup_1")
        occluder = object_qpos(env, args.occluder)
        rows = []
        for dx in np.linspace(-0.10, 0.10, 17):
            for dy in np.linspace(-0.10, 0.10, 17):
                if np.hypot(dx, dy) < 0.035:
                    continue
                moved = occluder.copy()
                moved[:2] = target[:2] + [dx, dy]
                set_object_qpos(env, args.occluder, moved)
                obs = regenerate_obs(env)
                row = {"dx": float(dx), "dy": float(dy), "x": float(moved[0]), "y": float(moved[1])}
                for camera in ("agentview", "robot0_eye_in_hand"):
                    stats = instance_stats(env, obs, camera, ["ketchup_1", args.occluder])
                    row[f"{camera}_target_pixels"] = stats["ketchup_1"]["visible_pixels"]
                    row[f"{camera}_occluder_pixels"] = stats[args.occluder]["visible_pixels"]
                rows.append(row)
        valid = [row for row in rows if row["agentview_occluder_pixels"] > 0 and row["robot0_eye_in_hand_occluder_pixels"] > 0]
        best = min(valid, key=lambda row: row["agentview_target_pixels"] + row["robot0_eye_in_hand_target_pixels"])
        output = args.output or ROOT / f"results/validation/stock_occluder_scan_{args.occluder}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"best": best, "rows": rows}, indent=2) + "\n")
        print(json.dumps(best, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
