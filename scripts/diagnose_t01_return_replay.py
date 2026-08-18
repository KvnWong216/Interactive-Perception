#!/usr/bin/env python3
"""Replay frozen public OPEN actions and compare proprioceptive return paths.

This is a development diagnostic, not claim-bearing evidence.  Every tested
controller reads only the stock end-effector pose and gripper state.  Drawer
joints, segmentation, target poses, and task predicates are never read.
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
from collect_t01_action_effect_transitions import DUMMY_ACTION  # noqa: E402
from interactive_perception.observation_option import (  # noqa: E402
    ObservationReturnConfig,
    ObservationReturnController,
    relative_axis_angle_xyzw,
)


def _errors(observation: dict, config: ObservationReturnConfig) -> tuple[np.ndarray, np.ndarray]:
    position_error = np.asarray(config.pose.position) - np.asarray(
        observation["robot0_eef_pos"]
    )
    orientation_error = relative_axis_angle_xyzw(
        observation["robot0_eef_quat"], config.pose.quaternion_xyzw
    )
    return position_error, orientation_error


def _scaled_action(
    config: ObservationReturnConfig,
    *,
    position_error: np.ndarray | None = None,
    orientation_error: np.ndarray | None = None,
) -> np.ndarray:
    action = np.zeros(7, dtype=np.float64)
    if position_error is not None:
        action[:3] = np.clip(
            position_error / config.position_output_scale,
            -config.maximum_normalized_translation,
            config.maximum_normalized_translation,
        )
    if orientation_error is not None:
        action[3:6] = np.clip(
            orientation_error / config.orientation_output_scale,
            -config.maximum_normalized_rotation,
            config.maximum_normalized_rotation,
        )
    action[6] = config.gripper_release_command
    return action


def _run_return(env, observation: dict, strategy: str) -> dict:
    config = ObservationReturnConfig(maximum_return_steps=240)
    if strategy == "registered":
        controller = ObservationReturnController(config)
        trace = []
        while not controller.status().terminal:
            action = controller.act(observation)
            observation, _, done, _ = env.step(action.tolist())
            status = controller.status()
            trace.append(
                {
                    "step": len(trace),
                    "position": np.asarray(
                        observation["robot0_eef_pos"], dtype=np.float64
                    ).tolist(),
                    "quaternion_xyzw": np.asarray(
                        observation["robot0_eef_quat"], dtype=np.float64
                    ).tolist(),
                    "position_error": status.position_error_metres,
                    "orientation_error": status.orientation_error_radians,
                    "action": action.tolist(),
                }
            )
            if done:
                break
        status = controller.status()
        return {
            "strategy": strategy,
            "phase": status.phase.value,
            "steps": len(trace),
            "position_error_metres": status.position_error_metres,
            "orientation_error_radians": status.orientation_error_radians,
            "completion_pose_index": status.completion_pose_index,
            "final_position": np.asarray(
                observation["robot0_eef_pos"], dtype=np.float64
            ).tolist(),
            "final_quaternion_xyzw": np.asarray(
                observation["robot0_eef_quat"], dtype=np.float64
            ).tolist(),
            "trace": trace,
        }
    trace = []
    for _ in range(config.release_steps):
        observation, _, done, _ = env.step(_scaled_action(config).tolist())
        if done:
            break

    settled = 0
    stage_settled = 0
    stage_complete = False
    waypoint_stage = 0
    start_position = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
    waypoint_positions = {
        "waypoint_high": np.asarray(
            [start_position[0], start_position[1], 1.30], dtype=np.float64
        ),
        "waypoint_left_high": np.asarray([-0.05, -0.15, 1.30], dtype=np.float64),
        "waypoint_right_high": np.asarray([-0.05, 0.15, 1.30], dtype=np.float64),
    }
    phase = "RETURN"
    for step in range(config.maximum_return_steps):
        position_error, orientation_error = _errors(observation, config)
        position_norm = float(np.linalg.norm(position_error))
        orientation_norm = float(np.linalg.norm(orientation_error))
        reached = (
            position_norm <= config.position_tolerance
            and orientation_norm <= config.orientation_tolerance_radians
        )
        settled = settled + 1 if reached else 0
        if settled >= config.settled_steps:
            phase = "COMPLETE"
            break

        if strategy == "direct":
            action = _scaled_action(
                config,
                position_error=position_error,
                orientation_error=orientation_error,
            )
        elif strategy == "orient_then_translate":
            action = (
                _scaled_action(config, orientation_error=orientation_error)
                if orientation_norm > config.orientation_tolerance_radians
                else _scaled_action(config, position_error=position_error)
            )
        elif strategy == "translate_then_orient":
            action = (
                _scaled_action(config, position_error=position_error)
                if position_norm > config.position_tolerance
                else _scaled_action(config, orientation_error=orientation_error)
            )
        elif strategy == "orient_lock_then_direct":
            if not stage_complete:
                stage_settled = (
                    stage_settled + 1
                    if orientation_norm <= config.orientation_tolerance_radians
                    else 0
                )
                stage_complete = stage_settled >= config.settled_steps
            action = (
                _scaled_action(config, orientation_error=orientation_error)
                if not stage_complete
                else _scaled_action(
                    config,
                    position_error=position_error,
                    orientation_error=orientation_error,
                )
            )
        elif strategy == "translate_lock_then_direct":
            if not stage_complete:
                stage_settled = (
                    stage_settled + 1
                    if position_norm <= config.position_tolerance
                    else 0
                )
                stage_complete = stage_settled >= config.settled_steps
            action = (
                _scaled_action(config, position_error=position_error)
                if not stage_complete
                else _scaled_action(
                    config,
                    position_error=position_error,
                    orientation_error=orientation_error,
                )
            )
        elif strategy in waypoint_positions:
            waypoint_error = waypoint_positions[strategy] - np.asarray(
                observation["robot0_eef_pos"], dtype=np.float64
            )
            waypoint_norm = float(np.linalg.norm(waypoint_error))
            if waypoint_stage == 0:
                stage_settled = stage_settled + 1 if waypoint_norm <= 0.02 else 0
                if stage_settled >= config.settled_steps:
                    waypoint_stage = 1
                    stage_settled = 0
                action = _scaled_action(config, position_error=waypoint_error)
            elif waypoint_stage == 1:
                aligned = (
                    waypoint_norm <= 0.02
                    and orientation_norm <= config.orientation_tolerance_radians
                )
                stage_settled = stage_settled + 1 if aligned else 0
                if stage_settled >= config.settled_steps:
                    waypoint_stage = 2
                    stage_settled = 0
                action = _scaled_action(
                    config,
                    position_error=waypoint_error,
                    orientation_error=orientation_error,
                )
            else:
                action = _scaled_action(
                    config,
                    position_error=position_error,
                    orientation_error=orientation_error,
                )
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        observation, _, done, _ = env.step(action.tolist())
        trace.append(
            {
                "step": step,
                "position": np.asarray(observation["robot0_eef_pos"]).tolist(),
                "position_error": position_norm,
                "orientation_error": orientation_norm,
                "action": action.tolist(),
            }
        )
        if done:
            phase = "ENVIRONMENT_DONE"
            break
    else:
        phase = "TIMED_OUT"

    position_error, orientation_error = _errors(observation, config)
    return {
        "strategy": strategy,
        "phase": phase,
        "steps": len(trace),
        "position_error_metres": float(np.linalg.norm(position_error)),
        "orientation_error_radians": float(np.linalg.norm(orientation_error)),
        "final_position": np.asarray(observation["robot0_eef_pos"]).tolist(),
        "final_quaternion_xyzw": np.asarray(
            observation["robot0_eef_quat"], dtype=np.float64
        ).tolist(),
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/calibration/t01_open_and_observe_effect_v4_extension.jsonl",
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=("direct", "orient_then_translate", "translate_then_orient"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/diagnostics/t01_return_replay_strategies_v1.json",
    )
    args = parser.parse_args()
    if not args.dataset.is_absolute():
        args.dataset = ROOT / args.dataset
    if not args.output.is_absolute():
        args.output = ROOT / args.output

    wanted = set(args.seeds)
    rows = [
        json.loads(line)
        for line in args.dataset.read_text().splitlines()
        if line.strip()
    ]
    rows = [
        row
        for row in rows
        if row["regime"] == "empty_full" and int(row["seed"]) in wanted
    ]
    if {int(row["seed"]) for row in rows} != wanted:
        raise ValueError("one or more requested EMPTY rows are missing")

    from libero.libero.envs import OffScreenRenderEnv

    results = []
    for row in rows:
        open_actions = [
            point["action"]
            for point in row["public_action_history"]
            if point["phase"] == "OPEN"
        ]
        for strategy in args.strategies:
            env = OffScreenRenderEnv(
                bddl_file_name=str(ROOT / row["bddl"]),
                camera_heights=256,
                camera_widths=256,
            )
            try:
                env.seed(int(row["seed"]))
                observation = env.reset()
                for _ in range(10):
                    observation, _, _, _ = env.step(DUMMY_ACTION)
                for action in open_actions:
                    observation, _, done, _ = env.step(action)
                    if done:
                        raise RuntimeError("OPEN replay terminated unexpectedly")
                result = _run_return(env, observation, strategy)
                result["seed"] = int(row["seed"])
                results.append(result)
                print(
                    f"seed={row['seed']} strategy={strategy} phase={result['phase']} "
                    f"position_error={result['position_error_metres']:.4f} "
                    f"orientation_error={result['orientation_error_radians']:.4f}",
                    flush=True,
                )
            finally:
                env.close()

    report = {
        "schema_version": "interactive-perception.return-replay-diagnostic.v1",
        "claim_eligible": False,
        "dataset": str(args.dataset.relative_to(ROOT)),
        "controller_inputs": [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        "online_oracle_inputs": [],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
