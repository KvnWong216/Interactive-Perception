#!/usr/bin/env python3
"""Measure pi0.5 SR on custom BDDL scenes with the stock LIBERO rollout path.

This evaluator intentionally has no segmentation, uncertainty probe, router, or
custom rollout wrapper.  Relative to openpi's LIBERO example, only the BDDL
file and its language instruction change.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import _bootstrap  # noqa: F401,E402

from interactive_perception.policy_client import (  # noqa: E402
    OpenPiWebsocketPolicy,
    build_observation,
)

# Copied from openpi/examples/libero/main.py. Keeping it local prevents this
# pure-policy evaluator from importing the benchmark's rollout machinery.
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "benchmarks/interactive_manipulation_v0/benchmark.yaml",
    )
    parser.add_argument("--task-ids", nargs="+", default=["T01_multi_drawer_search"])
    parser.add_argument("--variant", choices=["implicit", "hinted", "explicit", "capability"], default="implicit")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/gates/pure_pi05_scenario_sr.json")
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--video-stride", type=int, default=2)
    return parser.parse_args()


def joint_endpoint(env: Any, endpoint: dict[str, Any] | None) -> bool:
    if not endpoint or endpoint.get("type") != "articulated_reveal":
        return False
    for condition in endpoint.get("joints", []):
        value = float(np.asarray(env.env.sim.data.get_joint_qpos(condition["name"])).reshape(-1)[0])
        threshold = float(condition["threshold"])
        if condition["operator"] == "lt" and not value < threshold:
            return False
        if condition["operator"] == "gt" and not value > threshold:
            return False
    return bool(endpoint.get("joints"))


def run_episode(
    env: Any,
    policy: OpenPiWebsocketPolicy,
    prompt: str,
    endpoint: dict[str, Any] | None,
    diagnostic_joints: list[str],
    args: argparse.Namespace,
    video_path: Path | None = None,
) -> dict[str, Any]:
    obs = env.reset()
    joint_history = {name: [] for name in diagnostic_joints}

    def record_joints() -> None:
        for name in diagnostic_joints:
            value = env.env.sim.data.get_joint_qpos(name)
            joint_history[name].append(float(np.asarray(value).reshape(-1)[0]))

    record_joints()
    video = None
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video = imageio.get_writer(video_path, fps=20)
    plan: collections.deque[np.ndarray] = collections.deque()
    done = False
    endpoint_success = False
    error: str | None = None
    step = 0
    try:
        while step < args.max_steps + args.num_steps_wait:
            if video is not None and step % args.video_stride == 0:
                video.append_data(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            if step < args.num_steps_wait:
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            else:
                if not plan:
                    chunk = policy.sample_chunks(build_observation(obs, prompt), 1)[0]
                    if len(chunk) < args.replan_steps:
                        raise ValueError(f"policy returned {len(chunk)} actions")
                    plan.extend(chunk[: args.replan_steps])
                obs, _, done, _ = env.step(plan.popleft().tolist())
                record_joints()
                endpoint_success = endpoint_success or joint_endpoint(env, endpoint)
                if done or (args.variant == "capability" and endpoint_success):
                    break
            step += 1
    except Exception as exc:  # noqa: BLE001 - retained in the episode report
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if video is not None:
            video.close()
    success = endpoint_success if args.variant == "capability" else bool(done)
    return {
        "success": bool(success),
        "task_success": bool(done),
        "information_endpoint_success": bool(endpoint_success),
        "steps": step,
        "error": error,
        "joint_diagnostics": {
            name: {
                "initial": values[0],
                "minimum": min(values),
                "maximum": max(values),
                "final": values[-1],
            }
            for name, values in joint_history.items()
            if values
        },
    }


def main() -> None:
    args = parse_args()
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    tasks = {task["id"]: task for task in spec["tasks"]}
    unknown = set(args.task_ids) - set(tasks)
    if unknown:
        raise SystemExit(f"unknown task ids: {sorted(unknown)}")

    from libero.libero.envs import OffScreenRenderEnv

    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port, api_key=args.api_key)
    report: dict[str, Any] = {
        "policy": "pi0.5-libero checkpoint",
        "environment": "libero.envs.OffScreenRenderEnv",
        "policy_path": "stock openpi observation/action contract",
        "variant": args.variant,
        "seeds": args.seeds,
        "server_metadata": policy.server_metadata,
        "tasks": {},
    }
    for task_id in args.task_ids:
        task = tasks[task_id]
        prompt = task["prompt_variants"][args.variant]
        endpoint = task.get("information_endpoint")
        diagnostic_joints = [
            str(item["joint"]) for item in task.get("search_locations", []) if item.get("joint")
        ]
        bddl = ROOT / task["bddl"]
        rows = []
        print(f"[pure-pi05] {task_id}: {prompt}", flush=True)
        for seed in args.seeds:
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
            )
            try:
                env.seed(seed)
                video_path = None
                if args.video_dir is not None and seed in args.video_seeds:
                    video_path = args.video_dir / f"{task_id}_{args.variant}_seed{seed:03d}.mp4"
                row = {
                    "seed": seed,
                    **run_episode(
                        env, policy, prompt, endpoint, diagnostic_joints, args, video_path
                    ),
                }
            finally:
                env.close()
            rows.append(row)
            print(f"  seed={seed} success={row['success']} steps={row['steps']}", flush=True)
        report["tasks"][task_id] = {
            "bddl": str(task["bddl"]),
            "prompt": prompt,
            "episodes": len(rows),
            "successes": sum(row["success"] for row in rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "rows": rows,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
