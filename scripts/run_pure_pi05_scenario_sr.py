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
from interactive_perception.camera_views import initialize_attached_camera_look_at  # noqa: E402

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
    parser.add_argument(
        "--camera-mode", choices=["stock", "benchmark"], default="stock"
    )
    parser.add_argument("--contact-diagnostics", action="store_true")
    return parser.parse_args()


def information_endpoint(
    env: Any,
    endpoint: dict[str, Any] | None,
    initial_objects: dict[str, np.ndarray],
) -> bool:
    if not endpoint:
        return False
    if endpoint.get("type") == "object_displacement":
        return all(
            float(
                np.linalg.norm(
                    np.asarray(env.env.sim.data.get_joint_qpos(
                        env.env.objects_dict[item["name"]].joints[0]
                    ))[:3]
                    - initial_objects[item["name"]][:3]
                )
            )
            >= float(item["minimum_m"])
            for item in endpoint.get("objects", [])
        ) and bool(endpoint.get("objects"))
    if endpoint.get("type") != "articulated_reveal":
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
    backend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    obs = env.reset()
    backend = backend or {}
    if args.camera_mode == "benchmark":
        initialization = backend["wrist_camera_initialization"]
        initialize_attached_camera_look_at(
            env,
            camera=str(initialization["camera"]),
            target=initialization["look_at"],
        )
        obs = env.regenerate_obs_from_state(env.get_sim_state())
    joint_history = {name: [] for name in diagnostic_joints}
    initial_objects = {
        str(item["name"]): np.asarray(
            env.env.sim.data.get_joint_qpos(
                env.env.objects_dict[str(item["name"])].joints[0]
            )
        ).copy()
        for item in (endpoint or {}).get("objects", [])
    }
    contact_diagnostics = DrawerContactDiagnostics(env) if args.contact_diagnostics else None

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
                if args.camera_mode == "benchmark":
                    video.append_data(capture_split_frame(env, obs, backend))
                else:
                    video.append_data(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            if step < args.num_steps_wait:
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            else:
                if not plan:
                    build_kwargs = {}
                    if args.camera_mode == "benchmark":
                        build_kwargs = {
                            "primary_camera": backend["policy_primary_camera"],
                            "wrist_camera": backend["policy_wrist_camera"],
                        }
                    chunk = policy.sample_chunks(
                        build_observation(obs, prompt, **build_kwargs), 1
                    )[0]
                    if len(chunk) < args.replan_steps:
                        raise ValueError(f"policy returned {len(chunk)} actions")
                    plan.extend(chunk[: args.replan_steps])
                obs, _, done, _ = env.step(plan.popleft().tolist())
                record_joints()
                if contact_diagnostics is not None:
                    contact_diagnostics.record(obs)
                endpoint_success = endpoint_success or information_endpoint(
                    env, endpoint, initial_objects
                )
                if done or (args.variant == "capability" and endpoint_success):
                    break
            step += 1
    except Exception as exc:  # noqa: BLE001 - retained in the episode report
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if video is not None:
            video.close()
    success = endpoint_success if args.variant == "capability" else bool(done)
    result = {
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
    if contact_diagnostics is not None:
        result["contact_diagnostics"] = contact_diagnostics.summary()
    return result


class DrawerContactDiagnostics:
    """Measure whether the gripper reaches, contacts, and pulls the middle handle."""

    HANDLE_LOCAL_POS = np.asarray([-0.00278, -0.10161, 0.11041])

    def __init__(self, env: Any) -> None:
        self.sim = env.env.sim
        model = self.sim.model
        self.middle_body_id = model.body_name2id("wooden_cabinet_1_cabinet_middle")
        self.cabinet_root_body_id = model.body_name2id("wooden_cabinet_1_main")
        self.grip_site_id = model.site_name2id("gripper0_grip_site")
        self.joint_id = model.joint_name2id("wooden_cabinet_1_middle_level")
        self.dof_id = int(model.jnt_dofadr[self.joint_id])
        self.middle_geoms = {
            geom_id
            for geom_id in range(model.ngeom)
            if self._descends_from(int(model.geom_bodyid[geom_id]), self.middle_body_id)
        }
        self.gripper_geoms = {
            geom_id
            for geom_id in range(model.ngeom)
            if "gripper" in (model.body_id2name(int(model.geom_bodyid[geom_id])) or "")
        }
        self.min_handle_distance = float("inf")
        self.contact_steps = 0
        self.two_finger_contact_steps = 0
        self.max_contact_force = 0.0
        self.max_opening_force = 0.0
        self.max_closing_force = 0.0
        self.max_joint_constraint_force = 0.0
        self.pull_alignment: list[float] = []
        self.previous_eef: np.ndarray | None = None

    def _descends_from(self, body_id: int, ancestor: int) -> bool:
        while body_id > 0:
            if body_id == ancestor:
                return True
            body_id = int(self.sim.model.body_parentid[body_id])
        return False

    def _opening_axis(self) -> np.ndarray:
        root_rotation = np.asarray(self.sim.data.body_xmat[self.cabinet_root_body_id]).reshape(3, 3)
        axis = -(root_rotation @ np.asarray([0.0, 1.0, 0.0]))
        return axis / np.linalg.norm(axis)

    def _contact_force(self, contact_id: int) -> float:
        force = np.zeros(6, dtype=np.float64)
        try:
            import mujoco

            mujoco.mj_contactForce(
                self.sim.model._model, self.sim.data._data, contact_id, force
            )
        except Exception:  # pragma: no cover - binding varies across MuJoCo builds
            return 0.0
        return float(np.linalg.norm(force[:3]))

    def record(self, obs: dict[str, Any]) -> None:
        data = self.sim.data
        middle_rotation = np.asarray(data.body_xmat[self.middle_body_id]).reshape(3, 3)
        handle = np.asarray(data.body_xpos[self.middle_body_id]) + middle_rotation @ self.HANDLE_LOCAL_POS
        eef = np.asarray(data.site_xpos[self.grip_site_id]).copy()
        self.min_handle_distance = min(self.min_handle_distance, float(np.linalg.norm(eef - handle)))
        if self.previous_eef is not None:
            displacement = eef - self.previous_eef
            norm = float(np.linalg.norm(displacement))
            if norm > 1e-6 and np.linalg.norm(eef - handle) < 0.08:
                self.pull_alignment.append(float(displacement @ self._opening_axis() / norm))
        self.previous_eef = eef

        touching_fingers: set[int] = set()
        contact_force = 0.0
        for contact_id in range(int(data.ncon)):
            contact = data.contact[contact_id]
            pair = {int(contact.geom1), int(contact.geom2)}
            gripper = pair & self.gripper_geoms
            if gripper and pair & self.middle_geoms:
                touching_fingers.update(gripper)
                contact_force += self._contact_force(contact_id)
        if touching_fingers:
            self.contact_steps += 1
            self.max_contact_force = max(self.max_contact_force, contact_force)
        if len(touching_fingers) >= 2:
            self.two_finger_contact_steps += 1
        generalized_force = float(data.qfrc_constraint[self.dof_id])
        self.max_joint_constraint_force = max(
            self.max_joint_constraint_force, abs(generalized_force)
        )
        opening_component = max(0.0, -generalized_force)
        closing_component = max(0.0, generalized_force)
        self.max_opening_force = max(self.max_opening_force, opening_component)
        self.max_closing_force = max(self.max_closing_force, closing_component)

    def summary(self) -> dict[str, Any]:
        alignments = np.asarray(self.pull_alignment, dtype=float)
        return {
            "min_eef_to_handle_m": self.min_handle_distance,
            "gripper_middle_contact_steps": self.contact_steps,
            "two_finger_middle_contact_steps": self.two_finger_contact_steps,
            "max_contact_force_n": self.max_contact_force,
            "max_middle_joint_constraint_force_n": self.max_joint_constraint_force,
            "max_opening_force_n": self.max_opening_force,
            "max_closing_force_n": self.max_closing_force,
            "near_handle_pull_alignment_mean": (
                float(np.mean(alignments)) if alignments.size else None
            ),
            "near_handle_wrong_direction_fraction": (
                float(np.mean(alignments < 0.0)) if alignments.size else None
            ),
        }


def capture_split_frame(env: Any, obs: dict[str, Any], backend: dict[str, Any]) -> np.ndarray:
    """Compose synchronized wrist/global panels without changing policy pixels."""

    from PIL import Image, ImageDraw

    left = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    camera = str(backend["demo_camera"])
    pose = backend["demo_camera_pose"]
    sim = env.env.sim
    camera_id = sim.model.camera_name2id(camera)
    original_pos = np.asarray(sim.model.cam_pos[camera_id], dtype=float).copy()
    original_quat = np.asarray(sim.model.cam_quat[camera_id], dtype=float).copy()
    try:
        sim.model.cam_pos[camera_id] = np.asarray(pose["position"], dtype=float)
        sim.model.cam_quat[camera_id] = np.asarray(pose["quaternion_wxyz"], dtype=float)
        sim.forward()
        global_obs = env.regenerate_obs_from_state(env.get_sim_state())
        right = np.ascontiguousarray(global_obs[f"{camera}_image"][::-1, ::-1])
    finally:
        sim.model.cam_pos[camera_id] = original_pos
        sim.model.cam_quat[camera_id] = original_quat
        sim.forward()
    panels = []
    for frame, label in zip((left, right), backend["demo_labels"], strict=True):
        panel = Image.fromarray(frame)
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, panel.width, max(20, panel.height // 12)), fill=(0, 0, 0))
        draw.text((8, 4), str(label), fill=(255, 255, 255))
        panels.append(np.asarray(panel))
    return np.ascontiguousarray(np.concatenate(panels, axis=1))


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
        "camera_mode": args.camera_mode,
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
                        env,
                        policy,
                        prompt,
                        endpoint,
                        diagnostic_joints,
                        args,
                        video_path,
                        spec["backend"],
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
