#!/usr/bin/env python3
"""Measure whether pi0.5 executes wrist-rotate and bring-closer inspection skills."""

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
from interactive_perception.camera_views import initialize_attached_camera_look_at  # noqa: E402
from run_pure_pi05_scenario_sr import capture_split_frame  # noqa: E402
from validate_information_enrichment_v0 import (  # noqa: E402
    apply_natural_reset_pose,
    canonicalize_scene,
    camera_geometry,
    movable_instances,
    object_qpos,
    step_actions,
)

DUMMY_ACTION = [0.0] * 6 + [-1.0]


def quat_angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    return float(np.degrees(2.0 * np.arccos(np.clip(abs(left @ right), 0.0, 1.0))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(30)))
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from libero.libero.envs import OffScreenRenderEnv

    spec = yaml.safe_load(
        (ROOT / "benchmarks/information_enrichment_v0/benchmark.yaml").read_text()
    )
    tasks = {item["id"]: item for item in spec["tasks"]}
    selected = ["IE02_resolution_only", "IE03_orientation_only"]
    prompts = {
        "IE02_resolution_only": "Pick up the macaroni and cheese package, bring it closer to the wrist camera to inspect its label, then put it down.",
        "IE03_orientation_only": "Pick up the ketchup bottle, rotate it with the wrist to inspect its label, then put it down.",
    }
    backend = {
        "demo_camera": spec["backend"]["evaluator_camera"],
        "demo_camera_pose": spec["backend"]["evaluator_camera_pose"],
        "demo_labels": ["FIRST-PERSON / WRIST", "THIRD-PERSON / GLOBAL"],
    }
    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port)
    report: dict[str, Any] = {"policy": "pi05_libero", "seeds": args.seeds, "tasks": {}}
    for task_id in selected:
        task = tasks[task_id]
        rows = []
        for seed in args.seeds:
            env = OffScreenRenderEnv(
                bddl_file_name=str(ROOT / task["bddl"]),
                camera_heights=256,
                camera_widths=256,
                initialization_noise=None,
            )
            writer = None
            try:
                env.seed(seed)
                obs = env.reset()
                obs = step_actions(env, obs, np.zeros(7), 5)
                base_qpos = {name: object_qpos(env, name) for name in movable_instances(env)}
                initialization = spec["backend"]["wrist_camera_initialization"]
                initialize_attached_camera_look_at(
                    env,
                    camera=str(initialization["camera"]),
                    target=initialization["look_at"],
                )
                obs = env.regenerate_obs_from_state(env.get_sim_state())
                _, layout = canonicalize_scene(
                    env,
                    task_id,
                    base_qpos=base_qpos,
                    stability_tolerance_m=float(
                        spec["validation"]["final_layout_stability_tolerance_m"]
                    ),
                )
                obs = layout["obs"]
                target = str(task["target"])
                initial = object_qpos(env, target)
                initial_camera = camera_geometry(env, "robot0_eye_in_hand")
                initial_distance = float(
                    np.linalg.norm(initial[:3] - np.asarray(initial_camera["position"]))
                )
                maximum_rotation = 0.0
                maximum_lift = 0.0
                minimum_distance = initial_distance
                plan: collections.deque[np.ndarray] = collections.deque()
                video_path = None
                if args.video_dir is not None and seed in args.video_seeds:
                    video_path = args.video_dir / f"{task_id}_seed{seed:03d}.mp4"
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = imageio.get_writer(video_path, fps=20)
                for step in range(args.max_steps):
                    if writer is not None and step % 2 == 0:
                        writer.append_data(capture_split_frame(env, obs, backend))
                    if step < 10:
                        obs, _, _, _ = env.step(DUMMY_ACTION)
                    else:
                        if not plan:
                            chunk = policy.sample_chunks(
                                build_observation(
                                    obs,
                                    prompts[task_id],
                                    primary_camera="agentview",
                                    wrist_camera="robot0_eye_in_hand",
                                ),
                                1,
                            )[0]
                            plan.extend(chunk[: args.replan_steps])
                        obs, _, _, _ = env.step(plan.popleft())
                    current = object_qpos(env, target)
                    camera = camera_geometry(env, "robot0_eye_in_hand")
                    maximum_rotation = max(
                        maximum_rotation, quat_angle_degrees(initial[3:], current[3:])
                    )
                    maximum_lift = max(maximum_lift, float(current[2] - initial[2]))
                    minimum_distance = min(
                        minimum_distance,
                        float(np.linalg.norm(current[:3] - np.asarray(camera["position"]))),
                    )
                final = object_qpos(env, target)
                if task_id == "IE02_resolution_only":
                    success = maximum_lift >= 0.03 and minimum_distance <= float(
                        task["inspection_distance_m"]
                    )
                else:
                    success = maximum_lift >= 0.03 and maximum_rotation >= float(
                        task["min_rotation_degrees"]
                    )
                rows.append(
                    {
                        "seed": seed,
                        "success": bool(success),
                        "maximum_rotation_degrees": maximum_rotation,
                        "maximum_lift_m": maximum_lift,
                        "initial_camera_distance_m": initial_distance,
                        "minimum_camera_distance_m": minimum_distance,
                        "final_translation_m": float(np.linalg.norm(final[:3] - initial[:3])),
                        "video": str(video_path) if video_path else None,
                    }
                )
                print(f"[{task_id}] seed={seed} success={success}", flush=True)
            finally:
                if writer is not None:
                    writer.close()
                env.close()
        report["tasks"][task_id] = {
            "prompt": prompts[task_id],
            "episodes": len(rows),
            "successes": sum(row["success"] for row in rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "rows": rows,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
