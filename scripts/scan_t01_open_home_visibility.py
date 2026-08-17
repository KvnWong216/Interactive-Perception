#!/usr/bin/env python3
"""Evaluator-only scan of T01 visibility with the arm at its stock home pose.

This is a design diagnostic, not executor evidence.  It uses the simulator
drawer joint to create a common open-container state, then asks whether the
stock policy cameras can see the target while the robot remains at its reset
pose.  A positive result motivates a separately calibrated physical
``OPEN_AND_OBSERVE`` option that retracts to this pose; it cannot authorize that
option by itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from interactive_perception.capability_gate import (  # noqa: E402
    exact_binomial_lower_bound,
)

DUMMY_ACTION = [0.0] * 6 + [-1.0]
MIDDLE_JOINT = "wooden_cabinet_1_middle_level"
OPEN_POSITION = -0.161
MINIMUM_PIXELS = 5


def segmentation_key(obs: dict, camera: str) -> str:
    keys = [key for key in obs if key.startswith(camera) and "segmentation" in key]
    if len(keys) != 1:
        raise KeyError(f"expected one {camera} segmentation key, got {keys}")
    return keys[0]


def target_pixels(env, obs: dict, camera: str) -> int:
    instance_id = env.instance_to_id["butter_1"]
    segmentation = np.asarray(obs[segmentation_key(obs, camera)]).squeeze()
    return int(np.count_nonzero(segmentation == instance_id))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(400, 460)))
    parser.add_argument("--opening-steps", type=int, default=300)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/diagnostics/t01_open_home_visibility_v1.json",
    )
    args = parser.parse_args()
    if args.opening_steps < 2:
        raise ValueError("opening_steps must be at least two")
    if not args.output.is_absolute():
        args.output = ROOT / args.output

    from libero.libero.envs import SegmentationRenderEnv

    bddl = ROOT / "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl"
    env = SegmentationRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
    )
    rows = []
    try:
        for seed in args.seeds:
            env.seed(seed)
            obs = env.reset()
            for _ in range(10):
                obs, _, _, _ = env.step(DUMMY_ACTION)
            before = {
                camera: target_pixels(env, obs, camera)
                for camera in ("agentview", "robot0_eye_in_hand")
            }
            if any(before.values()):
                raise RuntimeError(
                    f"seed {seed} violates hidden-target precondition: {before}"
                )
            home_position = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            home_quaternion = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
            initial_joint = float(env.env.sim.data.get_joint_qpos(MIDDLE_JOINT))

            # Evaluator intervention: move the passive drawer slowly enough for
            # contained objects to travel with it, without rendering every
            # intermediate frame.  This is never exposed to a controller.
            for position in np.linspace(
                initial_joint, OPEN_POSITION, args.opening_steps
            ):
                env.env.sim.data.set_joint_qpos(MIDDLE_JOINT, float(position))
                env.env.sim.forward()
                env.env.sim.step()
            obs = env.env._get_observations(force_update=True)
            after = {
                camera: target_pixels(env, obs, camera)
                for camera in ("agentview", "robot0_eye_in_hand")
            }
            visible = max(after.values()) >= MINIMUM_PIXELS
            row = {
                "seed": seed,
                "hidden_before": before,
                "open_home_pixels": after,
                "visible": visible,
                "home_eef_position": home_position.tolist(),
                "home_eef_quaternion_xyzw": home_quaternion.tolist(),
                "evaluator_target_position": env.env.sim.data.get_body_xpos(
                    "butter_1_main"
                ).tolist(),
            }
            rows.append(row)
            print(
                f"seed={seed} visible={visible} pixels={after}",
                flush=True,
            )
    finally:
        env.close()

    successes = sum(row["visible"] for row in rows)
    report = {
        "schema_version": "interactive-perception.t01-open-home-visibility.v1",
        "diagnostic_only": True,
        "authorizes_executor": False,
        "online_oracle_inputs": [],
        "evaluator_intervention": (
            "middle drawer joint is opened only to test candidate observation "
            "coverage with the stock robot reset pose"
        ),
        "scene": str(bddl.relative_to(ROOT)),
        "seeds": args.seeds,
        "minimum_target_pixels": MINIMUM_PIXELS,
        "opening_steps": args.opening_steps,
        "successes": successes,
        "trials": len(rows),
        "empirical_rate": successes / len(rows),
        "one_sided_95_lower_bound": exact_binomial_lower_bound(
            successes, len(rows), 0.95
        ),
        "interpretation": (
            "camera-coverage feasibility only; a real OPEN_AND_OBSERVE option "
            "must reach the pose without simulator state and pass a new "
            "paired physical action-effect gate"
        ),
        "failures": [row["seed"] for row in rows if not row["visible"]],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "successes",
        "trials",
        "empirical_rate",
        "one_sided_95_lower_bound",
        "failures",
    )}, indent=2))


if __name__ == "__main__":
    main()
