#!/usr/bin/env python3
"""Test OPEN_CONTAINER -> observe -> ACT on the stock-aligned T01 scene."""

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

DUMMY_ACTION = [0.0] * 6 + [-1.0]
OPEN_LIMIT = -0.14  # LIBERO WoodenCabinet.is_open, not a tuned project threshold.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--open-steps", type=int, default=300)
    parser.add_argument("--act-steps", type=int, default=400)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--bddl",
        type=Path,
        default=ROOT / "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl",
    )
    parser.add_argument("--open-prompt", default="Open the middle layer of the drawer")
    parser.add_argument("--act-prompt", default="Place the butter in the basket")
    parser.add_argument("--joint", default="wooden_cabinet_1_middle_level")
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from libero.libero.envs import OffScreenRenderEnv

    bddl = args.bddl if args.bddl.is_absolute() else ROOT / args.bddl
    source = yaml.safe_load((ROOT / "benchmarks/interactive_manipulation_v0/benchmark.yaml").read_text())
    demo = {
        "demo_camera": source["backend"]["demo_camera"],
        "demo_camera_pose": source["backend"]["demo_camera_pose"],
        "demo_labels": ["FIRST-PERSON / WRIST", "THIRD-PERSON / BEV"],
    }
    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port)
    rows = []
    for seed in args.seeds:
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        writer = None
        try:
            env.seed(seed)
            obs = env.reset()
            video_path = None
            if args.video_dir is not None and seed in args.video_seeds:
                video_path = args.video_dir / f"t01_chain_seed{seed:03d}.mp4"
                video_path.parent.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(video_path, fps=20)

            for _ in range(10):
                obs, _, _, _ = env.step(DUMMY_ACTION)

            phase = "OPEN_CONTAINER"
            plan: collections.deque[np.ndarray] = collections.deque()
            open_reached = False
            open_step = None
            task_success = False
            total_step = 0
            while total_step < args.open_steps:
                if writer is not None and total_step % 2 == 0:
                    writer.append_data(capture_split_frame(env, obs, demo))
                if not plan:
                    chunk = policy.sample_chunks(build_observation(obs, args.open_prompt), 1)[0]
                    plan.extend(chunk[: args.replan_steps])
                obs, _, done, _ = env.step(plan.popleft().tolist())
                total_step += 1
                task_success = task_success or bool(done)
                if float(env.env.sim.data.get_joint_qpos(args.joint)) < OPEN_LIMIT:
                    open_reached = True
                    open_step = total_step
                    break

            if open_reached and not task_success:
                phase = "ACT"
                plan.clear()
                act_step = 0
                while act_step < args.act_steps and not task_success:
                    if writer is not None and total_step % 2 == 0:
                        writer.append_data(capture_split_frame(env, obs, demo))
                    if not plan:
                        chunk = policy.sample_chunks(build_observation(obs, args.act_prompt), 1)[0]
                        plan.extend(chunk[: args.replan_steps])
                    obs, _, done, _ = env.step(plan.popleft().tolist())
                    act_step += 1
                    total_step += 1
                    task_success = task_success or bool(done)

            row = {
                "seed": seed,
                "open_reached": open_reached,
                "open_step": open_step,
                "task_success": task_success,
                "success": bool(open_reached and task_success),
                "terminal_phase": phase,
                "total_steps": total_step,
                "drawer_joint": args.joint,
                "drawer_joint_final": float(env.env.sim.data.get_joint_qpos(args.joint)),
                "video": str(video_path) if video_path else None,
            }
            rows.append(row)
            print(
                f"seed={seed} open={open_reached} task={task_success} "
                f"phase={phase} steps={total_step}",
                flush=True,
            )
        finally:
            if writer is not None:
                writer.close()
            env.close()

    report = {
        "policy": "pi05_libero",
        "scene": str(bddl.relative_to(ROOT)),
        "camera_protocol": "stock policy cameras; BEV evaluator-only",
        "phases": [
            {"primitive": "OPEN_CONTAINER", "prompt": args.open_prompt},
            {"primitive": "ACT", "prompt": args.act_prompt},
        ],
        "episodes": len(rows),
        "open_successes": sum(row["open_reached"] for row in rows),
        "task_successes": sum(row["task_success"] for row in rows),
        "chain_successes": sum(row["success"] for row in rows),
        "chain_success_rate": float(np.mean([row["success"] for row in rows])),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
