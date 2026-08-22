#!/usr/bin/env python3
"""Execute any semantic option with frozen pi0.5 and return to public home.

All scenario details are CLI data.  The controller consumes only stock RGB,
public proprioception, the semantic subtask, and public action history.
Simulator objects, joints, contacts, and task predicates are read only by a
separate evaluator replay after the controller terminates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/infra"), str(ROOT / "src")]
import bootstrap  # noqa: F401

from interactive_perception.action_options import execute_subtask_and_return_home
from interactive_perception.observation_option import home_return_config
from interactive_perception.policy_client import (
    OpenPiWebsocketPolicy,
    build_observation,
)


class RecordingEnvironment:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.actions: list[list[float]] = []

    def step(self, action: list[float]) -> tuple[Mapping[str, Any], float, bool, dict]:
        normalized = np.asarray(action, dtype=float).tolist()
        self.actions.append(normalized)
        return self.env.step(normalized)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_path(path: Path) -> str:
    """Use repository-relative paths for retained assets and absolute temp paths."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def wait_for_port(port: int, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return
        except OSError:
            time.sleep(1.0)
    raise TimeoutError(f"pi0.5 server did not become ready on port {port}")


def save_views(
    observation: Mapping[str, Any], *, prompt: str, name: str, directory: Path
) -> dict[str, Any]:
    packet = build_observation(observation, prompt)
    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for view, image in (("agentview", packet.image), ("wrist", packet.wrist_image)):
        path = directory / f"{name}_{view}.png"
        imageio.imwrite(path, image)
        paths[view] = portable_path(path)
        hashes[view] = digest(path)
    return {
        "name": name,
        "image_paths": paths,
        "image_sha256": hashes,
        "public_robot_state": [float(value) for value in packet.state],
    }


def split_frame(observation: Mapping[str, Any], prompt: str) -> np.ndarray:
    packet = build_observation(observation, prompt)
    separator = np.full((packet.image.shape[0], 4, 3), 255, dtype=np.uint8)
    return np.concatenate((packet.wrist_image, separator, packet.image), axis=1)


def object_qpos(env: Any, name: str) -> np.ndarray:
    obj = env.env.objects_dict[name]
    return np.asarray(
        env.env.sim.data.get_joint_qpos(obj.joints[0]), dtype=float
    ).copy()


def evaluator_replay(
    *,
    bddl: Path,
    seed: int,
    initial_state: np.ndarray,
    actions: list[list[float]],
    subtask_steps: int,
    target_object: str | None,
    target_destination_region: str | None,
    tracked_objects: tuple[str, ...],
    tracked_joints: tuple[str, ...],
) -> dict[str, Any]:
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
    )
    try:
        env.seed(seed)
        env.reset()
        observation = env.set_init_state(initial_state)
        names = tuple(name for name in tracked_objects if name in env.env.objects_dict)
        initial = {name: object_qpos(env, name) for name in names}
        objects = {
            name: {
                "minimum_eef_distance_m": float("inf"),
                "minimum_eef_step": None,
                "eef_position_at_minimum": None,
                "object_position_at_minimum": None,
                "maximum_lift_m": 0.0,
                "grasp_contact_steps": 0,
                "grasp_contact_after_subtask_steps": 0,
            }
            for name in names
        }
        joint_values = {
            name: [float(env.env.sim.data.get_joint_qpos(name))]
            for name in tracked_joints
        }
        first_task_success_step = None
        first_target_destination_step = None
        for step, action in enumerate(actions):
            observation, _, done, _ = env.step(action)
            eef = np.asarray(observation["robot0_eef_pos"], dtype=float)
            for name in names:
                current = object_qpos(env, name)
                row = objects[name]
                distance = float(np.linalg.norm(current[:3] - eef))
                if distance < row["minimum_eef_distance_m"]:
                    row["minimum_eef_distance_m"] = distance
                    row["minimum_eef_step"] = step
                    row["eef_position_at_minimum"] = eef.tolist()
                    row["object_position_at_minimum"] = current[:3].tolist()
                row["maximum_lift_m"] = max(
                    row["maximum_lift_m"], float(current[2] - initial[name][2])
                )
                grasped = bool(
                    env.env._check_grasp(
                        env.env.robots[0].gripper,
                        env.env.objects_dict[name].contact_geoms,
                    )
                )
                if grasped:
                    row["grasp_contact_steps"] += 1
                    if step >= subtask_steps:
                        row["grasp_contact_after_subtask_steps"] += 1
            for name in tracked_joints:
                joint_values[name].append(float(env.env.sim.data.get_joint_qpos(name)))
            if bool(done) and first_task_success_step is None:
                first_task_success_step = step
            if (
                target_object
                and target_destination_region
                and bool(
                    env.env._eval_predicate(
                        ["in", target_object, target_destination_region]
                    )
                )
                and first_target_destination_step is None
            ):
                first_target_destination_step = step
        target = objects.get(target_object) if target_object else None
        return {
            "timing": "separate replay after controller terminal",
            "privileged_inputs": [
                "declared evaluator object poses and contacts",
                "declared evaluator joints",
                "task success predicate",
            ],
            "task_success": first_task_success_step is not None,
            "first_task_success_step": first_task_success_step,
            "target_object": target_object,
            "target_destination_region": target_destination_region,
            "target_reached_destination": first_target_destination_step is not None,
            "first_target_destination_step": first_target_destination_step,
            "target_pick_success": bool(
                target
                and target["grasp_contact_steps"] > 0
                and target["maximum_lift_m"] >= 0.03
            ),
            "objects": objects,
            "joints": {
                name: {
                    "initial": values[0],
                    "final": values[-1],
                    "minimum": min(values),
                    "maximum": max(values),
                }
                for name, values in joint_values.items()
            },
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-config", type=Path)
    parser.add_argument("--bddl", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--role", required=True)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--state-key", default="initial")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--preserve-grasp", action="store_true")
    parser.add_argument("--target-object")
    parser.add_argument("--target-destination-region")
    parser.add_argument("--track-object", action="append", default=[])
    parser.add_argument("--track-joint", action="append", default=[])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--final-state",
        type=Path,
        help=(
            "Opaque simulator transport used only to continue the same physical "
            "episode in a later process; never decoded by the controller"
        ),
    )
    args = parser.parse_args()
    if args.scenario_config is not None:
        if not args.scenario_config.is_absolute():
            args.scenario_config = ROOT / args.scenario_config
        scenario = yaml.safe_load(args.scenario_config.read_text())
        scene = scenario.get("scene", {})
        task = scenario.get("task", {})
        option = scenario.get("actions", {}).get(args.role, {})
        evaluator = scenario.get("evaluator", {})
        if args.bddl is None and scene.get("bddl"):
            args.bddl = Path(str(scene["bddl"]))
        args.seed = args.seed if args.seed is not None else scene.get("seed")
        args.prompt = args.prompt or option.get("prompt") or task.get("prompt")
        args.target_object = args.target_object or evaluator.get("target_object")
        args.target_destination_region = (
            args.target_destination_region or evaluator.get("target_destination_region")
        )
        if not args.track_object:
            args.track_object = list(evaluator.get("track_objects", ()))
        if not args.track_joint:
            args.track_joint = list(evaluator.get("track_joints", ()))
    if args.bddl is None or not str(args.bddl):
        raise ValueError("bddl is required directly or through scenario-config")
    if args.seed is None:
        raise ValueError("seed is required directly or through scenario-config")
    if not args.prompt:
        raise ValueError("prompt is required directly or through scenario-config")
    for name in (
        "scenario_config",
        "bddl",
        "initial_state",
        "assets",
        "work",
        "output",
        "final_state",
    ):
        value = getattr(args, name)
        if value is not None and not value.is_absolute():
            setattr(args, name, ROOT / value)
    if (
        args.assets.exists()
        or args.work.exists()
        or args.output.exists()
        or (args.final_state is not None and args.final_state.exists())
    ):
        raise FileExistsError("run outputs are immutable")

    subprocess.run(
        ["bash", str(ROOT / "scripts/infra/check_gpu.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "EXPERIMENT_GPU_INDEX": str(args.gpu),
            "EXPERIMENT_ALLOW_LOCAL_RUSTDESK": "1",
        },
        check=True,
    )
    from libero.libero.envs import OffScreenRenderEnv

    args.assets.mkdir(parents=True)
    args.work.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    env = OffScreenRenderEnv(
        bddl_file_name=str(args.bddl), camera_heights=256, camera_widths=256
    )
    recording = RecordingEnvironment(env)
    server = None
    server_log = None
    writer = None
    keyframes: list[dict[str, Any]] = []
    last_subtask: Mapping[str, Any] | None = None
    final_state: np.ndarray | None = None
    try:
        env.seed(args.seed)
        observation = env.reset()
        if args.initial_state is not None:
            with np.load(args.initial_state) as store:
                initial_state = np.asarray(store[args.state_key], dtype=float)
            observation = env.set_init_state(initial_state)
        else:
            initial_state = env.get_sim_state().copy()
        home_config = home_return_config(
            observation, preserve_grasp=args.preserve_grasp
        )
        keyframes.append(
            save_views(
                observation, prompt=args.prompt, name="00_before", directory=args.assets
            )
        )
        server_log = (args.work / "pi05_server.log").open("w")
        server = subprocess.Popen(
            ["bash", str(ROOT / "scripts/infra/serve_pi05.sh")],
            cwd=ROOT,
            env={
                **os.environ,
                "EXPERIMENT_GPU_INDEX": str(args.gpu),
                "CUDA_VISIBLE_DEVICES": str(args.gpu),
                "PORT": str(args.port),
            },
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        wait_for_port(args.port)
        policy = OpenPiWebsocketPolicy(host="127.0.0.1", port=args.port)
        metadata = policy.server_metadata
        writer = imageio.get_writer(args.assets / "public_wrist_agentview.mp4", fps=20)
        capture_steps = {
            max(0, round(args.steps * fraction) - 1): index
            for index, fraction in enumerate((0.25, 0.50, 0.75), start=1)
        }

        def observe_step(phase: str, step: int, current: Mapping[str, Any]) -> None:
            nonlocal last_subtask
            if step % 2 == 0:
                writer.append_data(split_frame(current, args.prompt))
            if phase == "SUBTASK":
                last_subtask = current
                if step in capture_steps:
                    keyframes.append(
                        save_views(
                            current,
                            prompt=args.prompt,
                            name=f"0{capture_steps[step]}_subtask",
                            directory=args.assets,
                        )
                    )

        observation, execution = execute_subtask_and_return_home(
            env=recording,
            initial_observation=observation,
            policy=policy,
            subtask_prompt=args.prompt,
            maximum_subtask_steps=args.steps,
            replan_steps=args.replan_steps,
            home_config=home_config,
            step_observer=observe_step,
        )
        if last_subtask is not None:
            keyframes.append(
                save_views(
                    last_subtask,
                    prompt=args.prompt,
                    name="04_subtask_end",
                    directory=args.assets,
                )
            )
        keyframes.append(
            save_views(
                observation,
                prompt=args.prompt,
                name="05_returned_home",
                directory=args.assets,
            )
        )
        # The real robot would simply remain in this physical state.  A process
        # boundary in simulation needs an opaque state transport.  Its values
        # are never exposed to policy, outcome critic, belief update, or action
        # selection.
        final_state = env.get_sim_state().copy()
    finally:
        if writer is not None:
            writer.close()
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
        if server_log is not None:
            server_log.close()
        env.close()

    action_path = args.assets / "public_action_history.json"
    action_path.write_text(json.dumps(recording.actions, separators=(",", ":")) + "\n")
    if args.final_state is not None:
        if final_state is None:
            raise RuntimeError("controller did not produce a final state")
        args.final_state.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.final_state, state=final_state)
    tracked = tuple(
        dict.fromkeys(
            [*args.track_object, *([args.target_object] if args.target_object else [])]
        )
    )
    evaluator = evaluator_replay(
        bddl=args.bddl,
        seed=args.seed,
        initial_state=initial_state,
        actions=recording.actions,
        subtask_steps=execution.subtask_steps,
        target_object=args.target_object,
        target_destination_region=args.target_destination_region,
        tracked_objects=tracked,
        tracked_joints=tuple(args.track_joint),
    )
    report = {
        "schema_version": "piu.semantic-option.v1",
        "scenario": str(args.bddl.relative_to(ROOT)),
        "seed": args.seed,
        "role": args.role,
        "prompt": args.prompt,
        "controller": {
            "policy": "frozen pi05_libero",
            "server_metadata": metadata,
            "online_inputs": [
                "stock agentview RGB",
                "wrist RGB",
                "public proprioception",
                "semantic subtask",
                "public action history",
            ],
            "online_oracle_inputs": [],
            "subtask_steps": execution.subtask_steps,
            "completion_source": execution.completion_source,
            "return_steps": execution.return_steps,
            "return_phase": execution.return_status.phase.value,
            "keyframes": keyframes,
            "video": portable_path(args.assets / "public_wrist_agentview.mp4"),
            "action_history": portable_path(action_path),
            "opaque_state_transport": (
                portable_path(args.final_state)
                if args.final_state is not None
                else None
            ),
        },
        "evaluator": evaluator,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"controller": report["controller"], "evaluator": evaluator}, indent=2
        )
    )


if __name__ == "__main__":
    main()
