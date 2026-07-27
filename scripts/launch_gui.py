#!/usr/bin/env python3
"""Open a real-time LIBERO/MuJoCo window, with optional keyboard control."""

from __future__ import annotations

import argparse
import time
from typing import Any

import glfw
import numpy as np

from _bootstrap import resolve_project_path

import robosuite as suite

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING


DEFAULT_BDDL = (
    "scenarios/interactive_manipulation_v0/"
    "T04_visible_direct.bddl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch an interactive LIBERO scene in a MuJoCo window."
    )
    parser.add_argument("--bddl", default=DEFAULT_BDDL)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--pos-sensitivity", type=float, default=1.0)
    parser.add_argument("--rot-sensitivity", type=float, default=1.0)
    parser.add_argument(
        "--observe-only",
        action="store_true",
        help="Open the live viewer without keyboard event monitoring.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="0 means run until the MuJoCo window is closed.",
    )
    return parser.parse_args()


def reset_env(env, device: Any | None):
    obs = env.reset()
    if device is not None:
        device.start_control()
    env.render()
    if device is None:
        print("Scene reset. Observe-only mode is running with neutral actions.")
    else:
        print("Scene reset. Focus the MuJoCo window and use the control keys.")
    return obs


def viewer_should_close(env) -> bool:
    viewer = getattr(env, "viewer", None)
    inner_viewer = getattr(viewer, "viewer", None)
    window = getattr(inner_viewer, "window", None)
    return window is not None and glfw.window_should_close(window)


def main() -> None:
    args = parse_args()
    bddl_path = resolve_project_path(args.bddl)
    if not bddl_path.exists():
        raise FileNotFoundError(f"BDDL file not found: {bddl_path}")

    problem_info = BDDLUtils.get_problem_info(str(bddl_path))
    problem_name = problem_info["problem_name"]
    instruction = problem_info["language_instruction"]
    if problem_name not in TASK_MAPPING:
        raise KeyError(f"Unknown LIBERO problem type: {problem_name}")

    controller_config = suite.load_controller_config(default_controller="OSC_POSE")
    env = TASK_MAPPING[problem_name](
        bddl_file_name=str(bddl_path),
        robots=["Panda"],
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=args.control_freq,
    )
    env.seed(args.seed)

    device = None
    keyboard_to_action = None
    if not args.observe_only:
        # Import lazily so observe-only mode does not request Accessibility access.
        from robosuite.devices import Keyboard
        from robosuite.utils.input_utils import input2action

        device = Keyboard(
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
        keyboard_to_action = input2action

    print("\nLIBERO real-time GUI")
    print(f"  BDDL:        {bddl_path}")
    print(f"  Instruction: {instruction}")
    print(f"  Camera:      {args.camera}")
    print(f"  Action dim:  {env.action_dim}")
    print(f"  Mode:        {'observe only' if args.observe_only else 'keyboard control'}")
    print("  Exit:        press Esc in the viewer, or Ctrl-C in this terminal\n")

    step_count = 0
    success_announced = False
    try:
        obs = reset_env(env, device)
        while not viewer_should_close(env):
            loop_start = time.perf_counter()

            if device is None:
                action = np.zeros(env.action_dim, dtype=np.float32)
            else:
                action, _ = keyboard_to_action(
                    device=device,
                    robot=env.robots[0],
                    active_arm="right",
                    env_configuration=None,
                )
                if action is None:
                    obs = reset_env(env, device)
                    step_count = 0
                    success_announced = False
                    continue

            obs, reward, done, info = env.step(action)
            env.render()
            step_count += 1

            if env._check_success() and not success_announced:
                print(f"SUCCESS at step {step_count}: {instruction}")
                success_announced = True

            if args.max_steps > 0 and step_count >= args.max_steps:
                print(f"Reached --max-steps={args.max_steps}; closing viewer.")
                break

            target_period = 1.0 / args.control_freq
            remaining = target_period - (time.perf_counter() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nStopped by Ctrl-C.")
    finally:
        listener = getattr(device, "listener", None)
        if listener is not None:
            listener.stop()
        env.close()
        print("Viewer closed cleanly.")


if __name__ == "__main__":
    main()
