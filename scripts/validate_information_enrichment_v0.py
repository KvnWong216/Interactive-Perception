#!/usr/bin/env python3
"""Render and oracle-validate the wrist Information-Enrichment v0 scenes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import yaml


CAMERAS = ("robot0_eye_in_hand", "agentview")
Z_180 = np.asarray([0.0, 0.0, 0.0, 1.0])


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    starter = project_root
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(project_root))
    parser.add_argument(
        "--spec",
        default=str(
            starter
            / "benchmarks"
            / "information_enrichment_v0"
            / "benchmark.yaml"
        ),
    )
    parser.add_argument("--bddl-root", default=str(starter))
    parser.add_argument(
        "--output",
        default=str(starter / "outputs" / "information_enrichment_v0"),
    )
    parser.add_argument("--task-ids", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def configure_imports(project_root: Path) -> None:
    source = project_root / "third_party" / "LIBERO"
    if not source.exists():
        raise FileNotFoundError(f"LIBERO source not found: {source}")
    sys.path.insert(0, str(source))
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(project_root / ".libero"))
    os.environ.setdefault(
        "MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl"
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return value


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )


def quat_angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    dot = float(np.clip(abs(np.dot(left, right)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def object_qpos(env: Any, instance: str) -> np.ndarray:
    obj = env.env.objects_dict[instance]
    if len(obj.joints) != 1:
        raise ValueError(f"{instance} must have one free joint: {obj.joints}")
    qpos = np.asarray(
        env.env.sim.data.get_joint_qpos(obj.joints[0]), dtype=float
    ).copy()
    if qpos.shape != (7,):
        raise ValueError(f"{instance} is not a free object: {qpos.shape}")
    return qpos


def set_object_qpos(env: Any, instance: str, qpos: np.ndarray) -> None:
    obj = env.env.objects_dict[instance]
    env.env.sim.data.set_joint_qpos(obj.joints[0], np.asarray(qpos, dtype=float))


def set_object_xy(
    env: Any,
    instance: str,
    xy: tuple[float, float],
    *,
    quaternion: np.ndarray | None = None,
) -> None:
    qpos = object_qpos(env, instance)
    qpos[:2] = np.asarray(xy, dtype=float)
    if quaternion is not None:
        qpos[3:] = np.asarray(quaternion, dtype=float)
    set_object_qpos(env, instance, qpos)


def regenerate_obs(env: Any) -> dict[str, Any]:
    env.env.sim.forward()
    return env.regenerate_obs_from_state(env.get_sim_state())


def step_actions(
    env: Any, obs: dict[str, Any], action: np.ndarray, count: int
) -> dict[str, Any]:
    for _ in range(count):
        obs, _, _, _ = env.step(action)
    return obs


def movable_instances(env: Any) -> list[str]:
    instances: list[str] = []
    for instance, obj in env.env.objects_dict.items():
        if len(obj.joints) != 1:
            continue
        qpos = np.asarray(
            env.env.sim.data.get_joint_qpos(obj.joints[0]), dtype=float
        )
        if qpos.shape == (7,):
            instances.append(instance)
    return sorted(instances)


def positions(env: Any, instances: list[str]) -> dict[str, list[float]]:
    return {
        instance: object_qpos(env, instance)[:3].tolist()
        for instance in instances
    }


def apply_natural_reset_pose(
    env: Any,
    obs: dict[str, Any],
    pose_spec: dict[str, Any],
    disturbance_tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    instances = movable_instances(env)
    before = positions(env, instances)
    action = np.asarray(pose_spec["action"], dtype=float)
    obs = step_actions(env, obs, action, int(pose_spec["action_steps"]))
    neutral = np.zeros_like(env.env.action_spec[0])
    obs = step_actions(env, obs, neutral, int(pose_spec["settle_steps"]))
    after = positions(env, instances)
    displacement = {
        instance: float(
            np.linalg.norm(
                np.asarray(after[instance]) - np.asarray(before[instance])
            )
        )
        for instance in instances
    }
    maximum = max(displacement.values(), default=0.0)
    return obs, {
        "action": action.tolist(),
        "action_steps": int(pose_spec["action_steps"]),
        "settle_steps": int(pose_spec["settle_steps"]),
        "object_displacement_m": displacement,
        "maximum_object_displacement_m": maximum,
        "objects_undisturbed": maximum <= disturbance_tolerance,
    }


def camera_geometry(env: Any, camera: str) -> dict[str, Any]:
    sim = env.env.sim
    camera_id = sim.model.camera_name2id(camera)
    position = np.asarray(sim.data.cam_xpos[camera_id], dtype=float)
    rotation = np.asarray(sim.data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
    forward = -rotation[:, 2]
    return {
        "position": position.tolist(),
        "forward": forward.tolist(),
        "downward_degrees_from_horizontal": float(
            np.degrees(np.arcsin(np.clip(-forward[2], -1.0, 1.0)))
        ),
    }


def camera_equal(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    position_tolerance_m: float,
    forward_tolerance: float,
) -> bool:
    return bool(
        np.allclose(
            left["position"],
            right["position"],
            atol=position_tolerance_m,
            rtol=0,
        )
        and np.allclose(
            left["forward"],
            right["forward"],
            atol=forward_tolerance,
            rtol=0,
        )
    )


def segmentation_key(obs: dict[str, Any], camera: str) -> str:
    keys = [
        key
        for key in obs
        if key.startswith(camera) and "segmentation" in key
    ]
    if len(keys) != 1:
        raise KeyError(f"Expected one segmentation key for {camera}: {keys}")
    return keys[0]


def instance_stats(
    env: Any,
    obs: dict[str, Any],
    camera: str,
    instances: list[str],
) -> dict[str, dict[str, Any]]:
    segmentation = np.flipud(
        np.asarray(obs[segmentation_key(obs, camera)]).squeeze()
    )
    stats: dict[str, dict[str, Any]] = {}
    for instance in instances:
        instance_id = env.instance_to_id[instance]
        ys, xs = np.nonzero(segmentation == instance_id)
        stats[instance] = {
            "visible_pixels": int(len(xs)),
            "bbox_xyxy": (
                [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()),
                    int(ys.max()),
                ]
                if len(xs)
                else None
            ),
        }
    return stats


def save_snapshot(
    env: Any,
    obs: dict[str, Any],
    output_dir: Path,
    name: str,
    tracked_instances: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cameras: dict[str, Any] = {}
    for camera in CAMERAS:
        image_path = output_dir / f"{name}_{camera}.png"
        imageio.imwrite(
            image_path,
            np.flipud(np.asarray(obs[f"{camera}_image"])).astype(np.uint8),
        )
        cameras[camera] = {
            "image": str(image_path),
            "instances": instance_stats(
                env, obs, camera, tracked_instances
            ),
        }
    return {"name": name, "cameras": cameras}


def policy_pixels(snapshot: dict[str, Any], instance: str) -> int:
    return int(
        snapshot["cameras"]["robot0_eye_in_hand"]["instances"][instance][
            "visible_pixels"
        ]
    )


def canonicalize_scene(
    env: Any,
    task_id: str,
    *,
    base_qpos: dict[str, np.ndarray],
    stability_tolerance_m: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    base_quaternions = {
        instance: qpos[3:].copy() for instance, qpos in base_qpos.items()
    }
    layouts: dict[str, dict[str, tuple[float, float]]] = {
        "T06_oblique_dense_clutter": {
            "ketchup_1": (0.16, -0.02),
            "alphabet_soup_1": (0.10, -0.02),
            "tomato_sauce_1": (0.15, 0.08),
            "milk_1": (0.20, 0.12),
            "orange_juice_1": (0.20, -0.13),
            "bbq_sauce_1": (0.06, -0.15),
            "basket_1": (0.10, 0.22),
        },
        "IE01_hierarchical_fridge_search": {
            # Keep all four pre-fridge candidates inside the natural wrist FOV.
            "macaroni_and_cheese_1": (0.03, -0.20),
            "cream_cheese_1": (0.07, -0.05),
            "butter_1": (0.16, 0.13),
            "milk_1": (0.06, 0.23),
            "basket_1": (0.37, 0.21),
        },
        "IE02_resolution_only": {
            "macaroni_and_cheese_1": (0.28, -0.09),
            "chocolate_pudding_1": (0.34, 0.01),
            "cream_cheese_1": (0.28, 0.11),
            "basket_1": (0.10, 0.22),
        },
        "IE03_orientation_only": {
            "ketchup_1": (0.06, -0.05),
            "bbq_sauce_1": (0.22, 0.07),
            "basket_1": (0.10, 0.22),
        },
    }
    if task_id not in layouts:
        raise KeyError(f"No canonical layout for {task_id}")
    for instance, xy in layouts[task_id].items():
        qpos = base_qpos[instance].copy()
        qpos[:2] = np.asarray(xy, dtype=float)
        set_object_qpos(env, instance, qpos)
    if task_id == "IE01_hierarchical_fridge_search":
        sim = env.env.sim
        shelf_assignments = {
            "alphabet_soup_1": ("upper", 0.04),
            "chocolate_pudding_1": ("middle", -0.04),
            "tomato_sauce_1": ("middle", 0.04),
            "orange_juice_1": ("lower", 0.0),
        }
        for instance, (shelf, lateral_offset) in shelf_assignments.items():
            site_id = sim.model.site_name2id(
                f"short_fridge_1_{shelf}_region"
            )
            shelf_center = np.asarray(
                sim.data.site_xpos[site_id], dtype=float
            ).copy()
            shelf_rotation = np.asarray(
                sim.data.site_xmat[site_id], dtype=float
            ).reshape(3, 3)
            # The region's local z axis spans shelf width for this fixture.
            position = shelf_center + shelf_rotation[:, 2] * lateral_offset
            bottom_offset = np.asarray(
                env.env.objects_dict[instance].bottom_offset, dtype=float
            )
            position[2] = shelf_center[2] - bottom_offset[2]
            qpos = base_qpos[instance].copy()
            qpos[:3] = position
            set_object_qpos(env, instance, qpos)
    obs = regenerate_obs(env)
    instances = movable_instances(env)
    before_settle = positions(env, instances)
    neutral = np.zeros_like(env.env.action_spec[0])
    needs_long_settle = task_id in {
        "T06_oblique_dense_clutter",
        "IE01_hierarchical_fridge_search",
    }
    placement_settle_steps = 100 if needs_long_settle else 5
    stability_window_steps = 20 if needs_long_settle else 5
    obs = step_actions(env, obs, neutral, placement_settle_steps)
    after_placement_settle = positions(env, instances)
    obs = step_actions(env, obs, neutral, stability_window_steps)
    after_stability_window = positions(env, instances)
    placement_displacement = {
        instance: float(
            np.linalg.norm(
                np.asarray(after_placement_settle[instance])
                - np.asarray(before_settle[instance])
            )
        )
        for instance in instances
    }
    stability_displacement = {
        instance: float(
            np.linalg.norm(
                np.asarray(after_stability_window[instance])
                - np.asarray(after_placement_settle[instance])
            )
        )
        for instance in instances
    }
    maximum = max(stability_displacement.values(), default=0.0)
    return base_quaternions, {
        "obs": obs,
        "layout_xy": layouts[task_id],
        "placement_settle_displacement_m": placement_displacement,
        "placement_settle_steps": placement_settle_steps,
        "stability_window_displacement_m": stability_displacement,
        "stability_window_steps": stability_window_steps,
        "maximum_stability_window_displacement_m": maximum,
        "layout_stable": maximum <= stability_tolerance_m,
    }


def inspection_endpoint(
    env: Any,
    instance: str,
    base_quaternion: np.ndarray,
    distance_m: float = 0.22,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = object_qpos(env, instance)
    camera = camera_geometry(env, "robot0_eye_in_hand")
    position = np.asarray(camera["position"]) + np.asarray(
        camera["forward"]
    ) * distance_m
    after = before.copy()
    after[:3] = position
    after[3:] = base_quaternion
    set_object_qpos(env, instance, after)
    obs = regenerate_obs(env)
    return obs, {
        "before_qpos": before.tolist(),
        "after_qpos": after.tolist(),
        "inspection_distance_m": distance_m,
    }


def validate_task(
    env_class: Any,
    task: dict[str, Any],
    bddl_path: Path,
    output_dir: Path,
    seed: int,
    width: int,
    height: int,
    spec: dict[str, Any],
) -> dict[str, Any]:
    env = env_class(
        bddl_file_name=str(bddl_path),
        camera_heights=height,
        camera_widths=width,
        initialization_noise=None,
    )
    try:
        env.seed(seed)
        obs = env.reset()
        neutral = np.zeros_like(env.env.action_spec[0])
        obs = step_actions(env, obs, neutral, 5)
        base_qpos = {
            instance: object_qpos(env, instance)
            for instance in movable_instances(env)
        }
        from interactive_perception.camera_views import initialize_attached_camera_look_at

        initialization = spec["backend"]["wrist_camera_initialization"]
        reset_pose = {
            **initialize_attached_camera_look_at(
                env,
                camera=str(initialization["camera"]),
                target=initialization["look_at"],
            ),
            "robot_motion_steps": 0,
        }
        base_quaternions, layout_report = canonicalize_scene(
            env,
            task["id"],
            base_qpos=base_qpos,
            stability_tolerance_m=float(
                spec["validation"]["final_layout_stability_tolerance_m"]
            ),
        )
        obs = layout_report.pop("obs")
        tracked = list(
            dict.fromkeys(
                [
                    instance
                    for key in (
                        "target",
                        "candidate_objects",
                        "outside_candidates",
                        "inside_distractors",
                    )
                    for instance in (
                        [task[key]]
                        if key in task and isinstance(task[key], str)
                        else task.get(key, [])
                    )
                ]
            )
        )
        initial_camera = camera_geometry(env, "robot0_eye_in_hand")
        initial = save_snapshot(env, obs, output_dir, "initial", tracked)
        initial_goal_success = bool(env.env._check_success())
        endpoints: dict[str, Any] = {}

        if task["id"] == "T06_oblique_dense_clutter":
            blocker = task["occluder"]
            qpos = object_qpos(env, blocker)
            before = qpos.copy()
            qpos[:2] = [0.06, 0.22]
            set_object_qpos(env, blocker, qpos)
            revealed_obs = regenerate_obs(env)
            endpoints["cleared"] = {
                "snapshot": save_snapshot(
                    env, revealed_obs, output_dir, "cleared", tracked
                ),
                "object_change": {
                    "instance": blocker,
                    "before_qpos": before.tolist(),
                    "after_qpos": qpos.tolist(),
                },
            }
            initial_pixels = policy_pixels(initial, task["target"])
            final_pixels = policy_pixels(
                endpoints["cleared"]["snapshot"], task["target"]
            )
            visible_candidates = sum(
                policy_pixels(initial, instance) > 0
                for instance in task["candidate_objects"]
            )
            metrics = {
                "initial_target_pixels": initial_pixels,
                "cleared_target_pixels": final_pixels,
                "visibility_gain": (
                    final_pixels / initial_pixels if initial_pixels else None
                ),
                "visible_candidates": visible_candidates,
            }
            task_valid = bool(
                initial_pixels >= int(task["min_initial_target_pixels"])
                and final_pixels > initial_pixels
                and final_pixels / initial_pixels
                >= float(task["min_visibility_gain"])
                and visible_candidates >= int(task["min_visible_candidates"])
            )

        elif task["id"] == "IE02_resolution_only":
            target = task["target"]
            before = object_qpos(env, target)
            close_obs, change = inspection_endpoint(
                env,
                target,
                base_quaternions[target],
                float(task["inspection_distance_m"]),
            )
            endpoints["closeup"] = {
                "snapshot": save_snapshot(
                    env, close_obs, output_dir, "closeup", tracked
                ),
                "object_change": change,
            }
            initial_pixels = policy_pixels(initial, target)
            close_pixels = policy_pixels(
                endpoints["closeup"]["snapshot"], target
            )
            orientation_change = quat_angle_degrees(
                before[3:],
                np.asarray(change["after_qpos"])[3:],
            )
            metrics = {
                "initial_target_pixels": initial_pixels,
                "closeup_target_pixels": close_pixels,
                "scale_gain": (
                    close_pixels / initial_pixels if initial_pixels else None
                ),
                "orientation_change_degrees": orientation_change,
            }
            task_valid = bool(
                initial_pixels > 0
                and close_pixels / initial_pixels >= float(task["min_scale_gain"])
                and orientation_change < 1e-5
                and all(
                    policy_pixels(initial, instance) > 0
                    for instance in task["candidate_objects"]
                )
            )

        elif task["id"] == "IE03_orientation_only":
            target = task["target"]
            before = object_qpos(env, target)
            after = before.copy()
            after[3:] = quat_multiply(Z_180, before[3:])
            set_object_qpos(env, target, after)
            rotated_obs = regenerate_obs(env)
            endpoints["label_facing"] = {
                "snapshot": save_snapshot(
                    env, rotated_obs, output_dir, "label_facing", tracked
                ),
                "object_change": {
                    "before_qpos": before.tolist(),
                    "after_qpos": after.tolist(),
                    "curated_from_asset_audit": True,
                },
            }
            translation = float(np.linalg.norm(after[:3] - before[:3]))
            rotation = quat_angle_degrees(before[3:], after[3:])
            metrics = {
                "translation_m": translation,
                "rotation_degrees": rotation,
                "initial_target_pixels": policy_pixels(initial, target),
                "label_facing_target_pixels": policy_pixels(
                    endpoints["label_facing"]["snapshot"], target
                ),
            }
            task_valid = bool(
                translation <= float(task["max_translation_m"])
                and rotation >= float(task["min_rotation_degrees"])
                and metrics["initial_target_pixels"] > 0
                and metrics["label_facing_target_pixels"] > 0
            )

        elif task["id"] == "IE01_hierarchical_fridge_search":
            closeup_gains: dict[str, float | None] = {}
            for index, instance in enumerate(task["outside_candidates"], start=1):
                saved = object_qpos(env, instance)
                inspect_obs, change = inspection_endpoint(
                    env, instance, base_quaternions[instance]
                )
                snapshot = save_snapshot(
                    env,
                    inspect_obs,
                    output_dir,
                    f"outside_{index:02d}_{instance}_closeup",
                    tracked,
                )
                endpoints[f"outside_{instance}"] = {
                    "snapshot": snapshot,
                    "object_change": change,
                    "semantic_result": "not_target",
                }
                baseline_pixels = policy_pixels(initial, instance)
                close_pixels = policy_pixels(snapshot, instance)
                closeup_gains[instance] = (
                    close_pixels / baseline_pixels
                    if baseline_pixels
                    else None
                )
                set_object_qpos(env, instance, saved)
                obs = regenerate_obs(env)

            joint = "short_fridge_1_fridge_door_joint_0"
            joint_before = float(env.env.sim.data.get_joint_qpos(joint))
            env.env.sim.data.set_joint_qpos(joint, 2.2)
            opened_obs = regenerate_obs(env)
            opened_snapshot = save_snapshot(
                env, opened_obs, output_dir, "fridge_open", tracked
            )
            endpoints["fridge_open"] = {
                "snapshot": opened_snapshot,
                "joint": joint,
                "before": joint_before,
                "after": 2.2,
            }
            target = task["target"]
            target_inspect_obs, target_change = inspection_endpoint(
                env, target, base_quaternions[target]
            )
            target_snapshot = save_snapshot(
                env,
                target_inspect_obs,
                output_dir,
                "inside_target_closeup",
                tracked,
            )
            endpoints["inside_target_closeup"] = {
                "snapshot": target_snapshot,
                "object_change": target_change,
                "semantic_result": "target",
            }
            initial_target_pixels = policy_pixels(initial, target)
            open_target_pixels = policy_pixels(opened_snapshot, target)
            close_target_pixels = policy_pixels(target_snapshot, target)
            visible_inside_after_open = sum(
                policy_pixels(opened_snapshot, instance) > 0
                for instance in task["inside_distractors"]
            )
            metrics = {
                "outside_closeup_scale_gains": closeup_gains,
                "initial_target_pixels": initial_target_pixels,
                "open_fridge_target_pixels": open_target_pixels,
                "target_closeup_pixels": close_target_pixels,
                "visible_inside_distractors_after_open": (
                    visible_inside_after_open
                ),
            }
            task_valid = bool(
                initial_target_pixels == 0
                and close_target_pixels > 0
                and visible_inside_after_open
                >= int(task["min_visible_inside_after_open"])
                and all(
                    gain is not None
                    and gain >= float(task["min_closeup_scale_gain"])
                    for gain in closeup_gains.values()
                )
            )
        else:
            raise KeyError(f"Unsupported task: {task['id']}")

        final_camera = camera_geometry(env, "robot0_eye_in_hand")
        actual_prompt = str(env.language_instruction)
        result = {
            "task_id": task["id"],
            "family": task["family"],
            "seed": seed,
            "bddl": str(bddl_path),
            "expected_prompt": task["prompt"],
            "actual_prompt": actual_prompt,
            "prompt_matches": (
                str(task["prompt"]).casefold() == actual_prompt.casefold()
            ),
            "reset_pose": reset_pose,
            "canonical_layout": layout_report,
            "initial_camera": initial_camera,
            "final_camera": final_camera,
            "camera_fixed_after_reset": camera_equal(
                initial_camera,
                final_camera,
                position_tolerance_m=float(
                    spec["validation"]["camera_position_tolerance_m"]
                ),
                forward_tolerance=float(
                    spec["validation"]["camera_forward_tolerance"]
                ),
            ),
            "active_viewpoint_actions_executed": 0,
            "initial_goal_success": initial_goal_success,
            "initial": initial,
            "oracle_endpoints": endpoints,
            "metrics": metrics,
            "task_structure_valid": task_valid,
        }
        result["passed"] = bool(
            result["prompt_matches"]
            and result["canonical_layout"]["layout_stable"]
            and result["reset_pose"]["robot_motion_steps"] == 0
            and result["camera_fixed_after_reset"]
            and not result["initial_goal_success"]
            and result["task_structure_valid"]
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "case_report.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        return result
    finally:
        env.close()


def prompt_contract_errors(spec: dict[str, Any]) -> list[str]:
    forbidden = [
        phrase.casefold()
        for phrase in spec["prompt_contract"][
            "forbidden_exploration_phrases"
        ]
    ]
    errors: list[str] = []
    for task in spec["tasks"]:
        leaks = [
            phrase
            for phrase in forbidden
            if phrase in str(task["prompt"]).casefold()
        ]
        if leaks:
            errors.append(f"{task['id']}: exploration leakage {leaks}")
    return errors


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    spec_path = Path(args.spec).expanduser().resolve()
    bddl_root = Path(args.bddl_root).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    configure_imports(project_root)
    from libero.libero.envs import SegmentationRenderEnv

    spec = load_yaml(spec_path)
    task_ids = set(args.task_ids or [task["id"] for task in spec["tasks"]])
    known = {task["id"] for task in spec["tasks"]}
    if task_ids - known:
        raise KeyError(f"Unknown task IDs: {sorted(task_ids - known)}")
    rows: list[dict[str, Any]] = []
    for task in spec["tasks"]:
        if task["id"] not in task_ids:
            continue
        for seed in args.seeds:
            print(f"[validate] {task['id']} seed={seed}", flush=True)
            case_dir = output_root / task["id"] / f"seed_{seed:03d}"
            try:
                row = validate_task(
                    SegmentationRenderEnv,
                    task,
                    bddl_root / task["bddl"],
                    case_dir,
                    seed,
                    args.width,
                    args.height,
                    spec,
                )
            except Exception as error:
                row = {
                    "task_id": task["id"],
                    "seed": seed,
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                }
                print(f"  FAIL: {row['error']}", flush=True)
            else:
                print(f"  {'PASS' if row['passed'] else 'FAIL'}", flush=True)
            rows.append(row)

    prompt_errors = prompt_contract_errors(spec)
    summary = {
        "benchmark": spec["name"],
        "version": spec["version"],
        "policy_camera": spec["backend"]["policy_camera"],
        "active_viewpoint_actions": spec["backend"][
            "active_viewpoint_actions"
        ],
        "prompt_contract_errors": prompt_errors,
        "task_ids": sorted(task_ids),
        "seeds": args.seeds,
        "case_runs": len(rows),
        "case_passes": sum(bool(row.get("passed")) for row in rows),
        "all_checks_passed": bool(
            not prompt_errors and all(row.get("passed") for row in rows)
        ),
        "rows": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "validation_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key != "rows"
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.strict and not summary["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
