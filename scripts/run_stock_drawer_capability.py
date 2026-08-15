#!/usr/bin/env python3
"""Run the exact LIBERO middle-drawer task from stored training-suite states."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from interactive_perception.policy_client import OpenPiWebsocketPolicy, build_observation  # noqa: E402
from run_pure_pi05_scenario_sr import capture_split_frame  # noqa: E402
from run_repro_gate import load_init_states  # noqa: E402

DUMMY_ACTION = [0.0] * 6 + [-1.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, nargs="+", default=list(range(30)))
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

    suite = libero_benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(0)
    states = load_init_states(suite, 0)
    if max(args.episodes) >= len(states):
        raise ValueError(f"episode index exceeds {len(states)} stored init states")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    spec = yaml.safe_load((ROOT / "benchmarks/interactive_manipulation_v0/benchmark.yaml").read_text())
    backend = {
        "demo_camera": spec["backend"]["demo_camera"],
        "demo_camera_pose": spec["backend"]["demo_camera_pose"],
        "demo_labels": ["FIRST-PERSON / WRIST", "THIRD-PERSON / BEV"],
    }
    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port)
    rows = []
    joint = "wooden_cabinet_1_middle_level"
    for episode in args.episodes:
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        writer = None
        try:
            env.seed(7)
            env.reset()
            obs = env.set_init_state(states[episode])
            initial = float(env.env.sim.data.get_joint_qpos(joint))
            minimum = initial
            plan: collections.deque[np.ndarray] = collections.deque()
            done = False
            video_path = None
            if args.video_dir is not None and episode in args.video_episodes:
                video_path = args.video_dir / f"stock_drawer_episode{episode:03d}.mp4"
                video_path.parent.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(video_path, fps=20)
            step = 0
            while step < args.max_steps + 10 and not done:
                if writer is not None and step % 2 == 0:
                    writer.append_data(capture_split_frame(env, obs, backend))
                if step < 10:
                    obs, _, done, _ = env.step(DUMMY_ACTION)
                else:
                    if not plan:
                        chunk = policy.sample_chunks(build_observation(obs, str(task.language)), 1)[0]
                        plan.extend(chunk[: args.replan_steps])
                    obs, _, done, _ = env.step(plan.popleft().tolist())
                minimum = min(minimum, float(env.env.sim.data.get_joint_qpos(joint)))
                step += 1
            final = float(env.env.sim.data.get_joint_qpos(joint))
            row = {
                "episode": episode, "success": bool(done), "steps": step,
                "joint_initial": initial, "joint_minimum": minimum,
                "joint_final": final, "video": str(video_path) if video_path else None,
            }
            rows.append(row)
            print(f"episode={episode} success={done} minimum={minimum:.6f}", flush=True)
        finally:
            if writer is not None:
                writer.close()
            env.close()
    report = {
        "policy": "pi05_libero", "suite": "libero_goal", "task_index": 0,
        "prompt": str(task.language), "bddl": str(bddl),
        "camera_protocol": "stock_policy_cameras; BEV evaluator-only",
        "episodes": len(rows), "successes": sum(row["success"] for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])), "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
