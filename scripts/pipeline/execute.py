#!/usr/bin/env python3
"""Execute any semantic option with frozen pi0.5 and return to public home.

The default controller consumes only stock RGB, public proprioception, the
semantic subtask, and public action history.  An explicitly named evaluator-only
oracle mode renders target instance segmentation into RGB solely to measure the
frozen executor's target-binding ceiling.  Reports distinguish these claim
scopes and enumerate all online oracle inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import random
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
from interactive_perception.oracle_visual_prompt import build_oracle_target_packet
from interactive_perception.policy_client import (
    OpenPiWebsocketPolicy,
    build_observation,
)
from piu.policy_identity import load_checkpoint_identity, validate_server_metadata


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


def wait_for_port(host: str, port: int, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            connection = http.client.HTTPConnection(host, port, timeout=1.0)
            connection.request("GET", "/healthz")
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status == 200:
                return
        except (OSError, http.client.HTTPException):
            time.sleep(1.0)
    raise TimeoutError(f"pi0.5 server did not become ready at {host}:{port}")


def save_views(
    observation: Mapping[str, Any], *, prompt: str, name: str, directory: Path
) -> dict[str, Any]:
    packet = build_observation(observation, prompt)
    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    pixel_hashes: dict[str, str] = {}
    for view, image in (("agentview", packet.image), ("wrist", packet.wrist_image)):
        path = directory / f"{name}_{view}.png"
        imageio.imwrite(path, image)
        paths[view] = portable_path(path)
        hashes[view] = digest(path)
        pixel_hashes[view] = hashlib.sha256(
            np.ascontiguousarray(image, dtype=np.uint8).tobytes()
        ).hexdigest()
    return {
        "name": name,
        "image_paths": paths,
        "image_sha256": hashes,
        "image_pixel_sha256": pixel_hashes,
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


def visible_instance_pixels(
    env: Any, observation: Mapping[str, Any], *, camera: str, instance: str
) -> int:
    keys = [
        key for key in observation if key.startswith(camera) and "segmentation" in key
    ]
    if len(keys) != 1:
        raise KeyError(f"expected one {camera} segmentation image, got {keys}")
    raw = np.asarray(observation[keys[0]]).squeeze()
    return int(np.count_nonzero(raw == env.instance_to_id[instance]))


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
    metric_contract_version: str = "v2",
) -> dict[str, Any]:
    from libero.libero.envs import SegmentationRenderEnv

    env = SegmentationRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
    )
    try:
        env.seed(seed)
        env.reset()
        observation = env.set_init_state(initial_state)
        names = tuple(name for name in tracked_objects if name in env.env.objects_dict)
        initial = {name: object_qpos(env, name) for name in names}
        target_visibility_initial = (
            {
                camera: visible_instance_pixels(
                    env, observation, camera=camera, instance=target_object
                )
                for camera in ("agentview", "robot0_eye_in_hand")
            }
            if target_object in names
            else None
        )
        target_visibility_after_subtask = None
        objects = {
            name: {
                "initial_position": initial[name][:3].tolist(),
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
            if step == subtask_steps - 1 and target_object in names:
                target_visibility_after_subtask = {
                    camera: visible_instance_pixels(
                        env, observation, camera=camera, instance=target_object
                    )
                    for camera in ("agentview", "robot0_eye_in_hand")
                }
        for name in names:
            objects[name]["final_position"] = object_qpos(env, name)[:3].tolist()
        target_visibility_final = (
            {
                camera: visible_instance_pixels(
                    env, observation, camera=camera, instance=target_object
                )
                for camera in ("agentview", "robot0_eye_in_hand")
            }
            if target_object in names
            else None
        )
        target = objects.get(target_object) if target_object else None
        target_in_destination_final = bool(
            target_object
            and target_destination_region
            and env.env._eval_predicate(
                ["in", target_object, target_destination_region]
            )
        )
        target_grasp_contact_final = bool(
            target_object
            and target_object in env.env.objects_dict
            and env.env._check_grasp(
                env.env.robots[0].gripper,
                env.env.objects_dict[target_object].contact_geoms,
            )
        )
        if metric_contract_version not in {"v1", "v2"}:
            raise ValueError("metric_contract_version must be v1 or v2")
        result = {
            "timing": "separate replay after controller terminal",
            "privileged_inputs": [
                "declared evaluator object poses and contacts",
                "declared evaluator joints",
                "evaluator-only instance segmentation",
                "task success predicate",
            ],
            "task_success": first_task_success_step is not None,
            "task_success_final": bool(env.env._check_success()),
            "first_task_success_step": first_task_success_step,
            "target_object": target_object,
            "target_destination_region": target_destination_region,
            "target_reached_destination": first_target_destination_step is not None,
            "target_in_destination_final": target_in_destination_final,
            "target_grasp_contact_final": target_grasp_contact_final,
            "first_target_destination_step": first_target_destination_step,
            "target_visibility_pixels": {
                "initial": target_visibility_initial,
                "after_subtask": target_visibility_after_subtask,
                "final": target_visibility_final,
            },
        }
        if metric_contract_version == "v1":
            result["target_pick_success"] = bool(
                target
                and target["grasp_contact_steps"] > 0
                and target["maximum_lift_m"] >= 0.03
            )
        else:
            result["target_grasp_contact_success"] = bool(
                target and target["grasp_contact_steps"] > 0
            )
            result["target_maximum_lift_m"] = (
                float(target["maximum_lift_m"]) if target is not None else None
            )
            result["metric_contract"] = {
                "target_grasp_contact_success": (
                    "LIBERO gripper/object contact predicate observed at least once"
                ),
                "target_maximum_lift_m": (
                    "continuous maximum object-z displacement from replay initial state"
                ),
                "target_pick_threshold": None,
                "note": (
                    "No hand-selected lift threshold determines a v2 binary outcome"
                ),
            }
        result["objects"] = objects
        result["joints"] = {
            name: {
                "initial": values[0],
                "final": values[-1],
                "minimum": min(values),
                "maximum": max(values),
            }
            for name, values in joint_values.items()
        }
        return result
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--server-timeout", type=float, default=180.0)
    parser.add_argument(
        "--report-schema",
        choices=("v1", "v2"),
        default="v2",
        help="v1 is retained only for exact historical metric reproduction",
    )
    parser.add_argument(
        "--external-server",
        action="store_true",
        help="connect to an already validated pi0.5 server and do not own its process",
    )
    parser.add_argument(
        "--expected-policy-identity",
        type=Path,
        help="required checkpoint-tree identity for every external policy call",
    )
    parser.add_argument(
        "--oracle-target-visual-prompt",
        choices=("box", "point", "spotlight"),
        help=(
            "evaluator-only upper bound: render the declared target's simulator "
            "instance segmentation into the policy RGB; never a public-input run"
        ),
    )
    parser.add_argument(
        "--oracle-minimum-visible-pixels",
        type=int,
        default=1,
        help="fail before policy execution unless the oracle target reaches this count",
    )
    parser.add_argument(
        "--oracle-target-allow-absent-until-visible",
        action="store_true",
        help=(
            "full-loop oracle upper bound: keep RGB unchanged while the target mask "
            "is empty, then render the declared visual prompt when it becomes visible"
        ),
    )
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
    if args.server_timeout <= 0:
        raise ValueError("server-timeout must be positive")
    if args.oracle_target_visual_prompt and not args.target_object:
        raise ValueError("oracle visual prompt requires target-object")
    if args.oracle_target_visual_prompt and not args.external_server:
        raise ValueError(
            "oracle visual-prompt qualification requires --external-server; "
            "the experiment runner must not start a local GPU model"
        )
    if args.external_server and args.expected_policy_identity is None:
        raise ValueError("external policy execution requires an expected identity")
    if not args.external_server and args.expected_policy_identity is not None:
        raise ValueError("owned policy execution cannot inject an external identity")
    if (
        args.oracle_target_allow_absent_until_visible
        and not args.oracle_target_visual_prompt
    ):
        raise ValueError(
            "allow-absent oracle mode requires --oracle-target-visual-prompt"
        )
    if not args.external_server and args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("an owned policy server must use a loopback host")
    if args.oracle_minimum_visible_pixels < 1:
        raise ValueError("oracle-minimum-visible-pixels must be positive")
    for name in (
        "scenario_config",
        "bddl",
        "initial_state",
        "assets",
        "work",
        "output",
        "final_state",
        "expected_policy_identity",
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

    # Fail an unavailable remote endpoint before creating an immutable run
    # directory or starting the simulator. This makes a network retry safe.
    if args.external_server:
        wait_for_port(args.host, args.port, timeout=args.server_timeout)

    if not args.external_server:
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
    if args.oracle_target_visual_prompt:
        from libero.libero.envs import SegmentationRenderEnv as RenderEnv
    else:
        from libero.libero.envs import OffScreenRenderEnv as RenderEnv

    args.assets.mkdir(parents=True)
    args.work.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    env = RenderEnv(
        bddl_file_name=str(args.bddl), camera_heights=256, camera_widths=256
    )
    recording = RecordingEnvironment(env)
    server = None
    server_log = None
    writer = None
    keyframes: list[dict[str, Any]] = []
    oracle_prompt_audit: list[dict[str, Any]] = []
    oracle_prompt_keyframes: list[dict[str, Any]] = []
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
        target_instance_id = (
            int(env.instance_to_id[args.target_object])
            if args.oracle_target_visual_prompt
            else None
        )
        if args.oracle_target_visual_prompt:
            initial_prompt_preview = build_oracle_target_packet(
                observation,
                args.prompt,
                target_instance_id=target_instance_id,
                style=args.oracle_target_visual_prompt,
            )
            if (
                max(initial_prompt_preview.diagnostics.visible_pixels.values())
                < args.oracle_minimum_visible_pixels
                and not args.oracle_target_allow_absent_until_visible
            ):
                raise ValueError(
                    f"oracle target {args.target_object!r} is not sufficiently "
                    "visible in the initial state: "
                    f"{dict(initial_prompt_preview.diagnostics.visible_pixels)}"
                )
        home_config = home_return_config(
            observation, preserve_grasp=args.preserve_grasp
        )
        keyframes.append(
            save_views(
                observation, prompt=args.prompt, name="00_before", directory=args.assets
            )
        )
        if not args.external_server:
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
        if not args.external_server:
            wait_for_port(args.host, args.port, timeout=args.server_timeout)
        policy = OpenPiWebsocketPolicy(host=args.host, port=args.port)
        metadata = policy.server_metadata
        if args.expected_policy_identity is not None:
            validate_server_metadata(
                metadata, load_checkpoint_identity(args.expected_policy_identity)
            )
        writer = imageio.get_writer(args.assets / "public_wrist_agentview.mp4", fps=20)
        capture_steps = {
            max(0, round(args.steps * fraction) - 1): index
            for index, fraction in enumerate((0.25, 0.50, 0.75), start=1)
        }

        def build_policy_observation(current: Mapping[str, Any], prompt: str):
            if args.oracle_target_visual_prompt is None:
                return build_observation(current, prompt)
            if target_instance_id is None:
                raise RuntimeError("oracle target instance id was not initialized")
            prompted = build_oracle_target_packet(
                current,
                prompt,
                target_instance_id=target_instance_id,
                style=args.oracle_target_visual_prompt,
            )
            row = {
                "policy_call_index": len(oracle_prompt_audit),
                **prompted.diagnostics.to_json(),
            }
            oracle_prompt_audit.append(row)
            if len(oracle_prompt_audit) == 1:
                paths: dict[str, str] = {}
                hashes: dict[str, str] = {}
                for view, image in (
                    ("agentview", prompted.packet.image),
                    ("wrist", prompted.packet.wrist_image),
                ):
                    path = args.assets / f"00_oracle_prompt_{view}.png"
                    imageio.imwrite(path, image)
                    paths[view] = portable_path(path)
                    hashes[view] = digest(path)
                oracle_prompt_keyframes.append(
                    {
                        "name": "00_first_policy_call",
                        "image_paths": paths,
                        "image_sha256": hashes,
                        "diagnostics": row,
                    }
                )
            return prompted.packet

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
            policy_observation_builder=build_policy_observation,
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
        metric_contract_version=args.report_schema,
    )
    report = {
        "schema_version": f"piu.semantic-option.{args.report_schema}",
        "claim_scope": (
            "EVALUATOR_ONLY_ORACLE_UPPER_BOUND"
            if args.oracle_target_visual_prompt
            else "PUBLIC_INPUT_EXECUTION"
        ),
        "scenario": str(args.bddl.relative_to(ROOT)),
        "seed": args.seed,
        "role": args.role,
        "prompt": args.prompt,
        "controller": {
            "policy": "frozen pi05_libero",
            "server_mode": "external" if args.external_server else "owned",
            "server_metadata": metadata,
            "expected_policy_identity": (
                {
                    "path": portable_path(args.expected_policy_identity),
                    "sha256": digest(args.expected_policy_identity),
                }
                if args.expected_policy_identity is not None
                else None
            ),
            "online_inputs": [
                "stock agentview RGB",
                "wrist RGB",
                "public proprioception",
                "semantic subtask",
                "public action history",
            ],
            "online_oracle_inputs": (
                [
                    "declared target simulator instance identity",
                    "online evaluator instance segmentation rendered into policy RGB",
                ]
                if args.oracle_target_visual_prompt
                else []
            ),
            "oracle_visual_prompt": (
                {
                    "style": args.oracle_target_visual_prompt,
                    "target_instance_id": target_instance_id,
                    "minimum_initial_visible_pixels": (
                        args.oracle_minimum_visible_pixels
                    ),
                    "allow_absent_until_visible": (
                        args.oracle_target_allow_absent_until_visible
                    ),
                    "source_initial_state": (
                        {
                            "path": portable_path(args.initial_state),
                            "sha256": digest(args.initial_state),
                            "state_key": args.state_key,
                        }
                        if args.initial_state is not None
                        else None
                    ),
                    "policy_call_audit": oracle_prompt_audit,
                    "keyframes": oracle_prompt_keyframes,
                }
                if args.oracle_target_visual_prompt
                else None
            ),
            "subtask_steps": execution.subtask_steps,
            "policy_calls": execution.policy_calls,
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
            "opaque_state_transport_sha256": (
                digest(args.final_state) if args.final_state is not None else None
            ),
            "source_initial_state_transport": (
                {
                    "path": portable_path(args.initial_state),
                    "sha256": digest(args.initial_state),
                    "state_key": args.state_key,
                }
                if args.initial_state is not None
                else None
            ),
        },
        "evaluator": evaluator,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    terminal_metrics = {
        "output": portable_path(args.output),
        "seed": args.seed,
        "role": args.role,
        "prompt": args.prompt,
        "subtask_steps": execution.subtask_steps,
        "return_phase": execution.return_status.phase.value,
        "task_success": evaluator["task_success"],
        "target_reached_destination": evaluator["target_reached_destination"],
        "target_visibility_pixels": evaluator["target_visibility_pixels"],
    }
    if args.report_schema == "v1":
        terminal_metrics["target_pick_success"] = evaluator["target_pick_success"]
    else:
        terminal_metrics["target_grasp_contact_success"] = evaluator[
            "target_grasp_contact_success"
        ]
        terminal_metrics["target_maximum_lift_m"] = evaluator["target_maximum_lift_m"]
    print(json.dumps(terminal_metrics, indent=2))


if __name__ == "__main__":
    main()
