#!/usr/bin/env python3
"""Test information primitives in an otherwise stock LIBERO object task."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import _bootstrap  # noqa: F401,E402
from interactive_perception.policy_client import OpenPiWebsocketPolicy, build_observation  # noqa: E402
from run_information_action_ability import quat_angle_degrees  # noqa: E402
from run_pure_pi05_scenario_sr import capture_split_frame  # noqa: E402
from run_repro_gate import load_init_states  # noqa: E402
from validate_information_enrichment_v0 import camera_geometry, object_qpos  # noqa: E402

DUMMY_ACTION = [0.0] * 6 + [-1.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--actions", nargs="+", choices=["MOVE_CLOSER", "ROTATE"], default=["MOVE_CLOSER", "ROTATE"])
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from libero.libero import benchmark as libero_benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = libero_benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(4)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    initial_states = load_init_states(suite, 4)
    if max(args.episodes) >= len(initial_states):
        raise ValueError(f"episode index exceeds {len(initial_states)} stored init states")
    prompts = {
        "MOVE_CLOSER": "Pick up the ketchup, bring it closer to the wrist camera to inspect it, then put it down.",
        "ROTATE": "Pick up the ketchup, rotate it with the wrist to inspect it, then put it down.",
    }
    spec = yaml.safe_load((ROOT / "benchmarks/information_enrichment_v0/benchmark.yaml").read_text())
    backend = {
        "demo_camera": spec["backend"]["evaluator_camera"],
        "demo_camera_pose": spec["backend"]["evaluator_camera_pose"],
        "demo_labels": ["FIRST-PERSON / WRIST", "THIRD-PERSON / GLOBAL"],
    }
    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port)
    report: dict[str, Any] = {
        "policy": "pi05_libero", "source_suite": "libero_object",
        "source_task_index": 4, "source_prompt": str(task.language),
        "camera_protocol": "stock_unmodified", "episodes": args.episodes,
        "actions": {},
    }
    for action_name in args.actions:
        rows = []
        for episode in args.episodes:
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            writer = None
            try:
                env.seed(7)
                env.reset()
                obs = env.set_init_state(initial_states[episode])
                initial = object_qpos(env, "ketchup_1")
                initial_camera = camera_geometry(env, "robot0_eye_in_hand")
                initial_distance = float(np.linalg.norm(initial[:3] - initial_camera["position"]))
                maximum_lift = 0.0
                maximum_rotation = 0.0
                minimum_distance = initial_distance
                task_success = False
                plan: collections.deque[np.ndarray] = collections.deque()
                video_path = None
                if args.video_dir is not None and episode in args.video_episodes:
                    video_path = args.video_dir / f"{action_name.lower()}_episode{episode:03d}.mp4"
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = imageio.get_writer(video_path, fps=20)
                for step in range(args.max_steps + 10):
                    if writer is not None and step % 2 == 0:
                        writer.append_data(capture_split_frame(env, obs, backend))
                    if step < 10:
                        obs, _, done, _ = env.step(DUMMY_ACTION)
                    else:
                        if not plan:
                            chunk = policy.sample_chunks(build_observation(obs, prompts[action_name]), 1)[0]
                            plan.extend(chunk[: args.replan_steps])
                        obs, _, done, _ = env.step(plan.popleft().tolist())
                    task_success = task_success or bool(done)
                    current = object_qpos(env, "ketchup_1")
                    camera = camera_geometry(env, "robot0_eye_in_hand")
                    maximum_lift = max(maximum_lift, float(current[2] - initial[2]))
                    maximum_rotation = max(maximum_rotation, quat_angle_degrees(initial[3:], current[3:]))
                    minimum_distance = min(minimum_distance, float(np.linalg.norm(current[:3] - camera["position"])))
                final = object_qpos(env, "ketchup_1")
                grasped = maximum_lift >= 0.03
                final_grasped = bool(env.env._check_grasp(
                    env.env.robots[0].gripper,
                    env.env.objects_dict["ketchup_1"].contact_geoms,
                ))
                put_down = grasped and not final_grasped
                endpoint = (
                    minimum_distance <= initial_distance - 0.08
                    if action_name == "MOVE_CLOSER"
                    else maximum_rotation >= 60.0
                )
                row = {
                    "episode": episode, "success": bool(grasped and endpoint and put_down),
                    "grasped": bool(grasped), "put_down": bool(put_down),
                    "final_grasped": final_grasped,
                    "stock_task_success": task_success,
                    "maximum_lift_m": maximum_lift,
                    "maximum_rotation_degrees": maximum_rotation,
                    "initial_camera_distance_m": initial_distance,
                    "minimum_camera_distance_m": minimum_distance,
                    "final_height_change_m": float(final[2] - initial[2]),
                    "video": str(video_path) if video_path else None,
                }
                rows.append(row)
                print(f"[{action_name}] episode={episode} success={row['success']} grasped={grasped} endpoint={endpoint} put_down={put_down}", flush=True)
            finally:
                if writer is not None:
                    writer.close()
                env.close()
        report["actions"][action_name] = {
            "prompt": prompts[action_name], "trials": len(rows),
            "successes": sum(row["success"] for row in rows), "rows": rows,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
