#!/usr/bin/env python3
"""Reset, render, and oracle-validate fixed-view interactive-manipulation scenes."""

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


def parse_args() -> argparse.Namespace:
    default_project_root = Path(__file__).resolve().parents[1]
    default_starter = default_project_root
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        default=str(default_project_root),
    )
    parser.add_argument(
        "--spec",
        default=str(
            default_starter
            / "benchmarks"
            / "interactive_manipulation_v0"
            / "benchmark.yaml"
        ),
    )
    parser.add_argument(
        "--bddl-root",
        default=str(default_starter),
    )
    parser.add_argument(
        "--output",
        default=str(
            default_starter / "outputs" / "interactive_manipulation_v0"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--task-ids", nargs="+", default=None)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--settle-steps", type=int, default=5)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def configure_imports(project_root: Path) -> None:
    libero_source = project_root / "third_party" / "LIBERO"
    if not libero_source.exists():
        raise FileNotFoundError(f"LIBERO source not found: {libero_source}")
    sys.path.insert(0, str(libero_source))
    os.environ.setdefault(
        "LIBERO_CONFIG_PATH", str(project_root / ".libero")
    )
    os.environ.setdefault(
        "MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl"
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return value


def find_segmentation_key(obs: dict[str, Any], camera: str) -> str:
    keys = [
        key
        for key in obs
        if key.startswith(camera) and "segmentation" in key
    ]
    if len(keys) != 1:
        raise KeyError(
            f"Expected one {camera} segmentation key, got {keys}; "
            f"available={sorted(obs)}"
        )
    return keys[0]


def camera_extrinsics(env: Any, camera: str) -> dict[str, list[float]]:
    camera_id = env.env.sim.model.camera_name2id(camera)
    return {
        "pos": np.asarray(
            env.env.sim.model.cam_pos[camera_id], dtype=float
        ).tolist(),
        "quat": np.asarray(
            env.env.sim.model.cam_quat[camera_id], dtype=float
        ).tolist(),
    }


def camera_unchanged(
    before: dict[str, list[float]],
    after: dict[str, list[float]],
    atol: float = 1e-12,
) -> bool:
    return bool(
        np.allclose(before["pos"], after["pos"], atol=atol, rtol=0.0)
        and np.allclose(before["quat"], after["quat"], atol=atol, rtol=0.0)
    )


def neutral_steps(env: Any, obs: dict[str, Any], count: int) -> dict[str, Any]:
    action_low, _ = env.env.action_spec
    action = np.zeros_like(action_low)
    for _ in range(count):
        obs, _, _, _ = env.step(action)
    return obs


def regenerate_obs(env: Any) -> dict[str, Any]:
    env.env.sim.forward()
    state = env.get_sim_state()
    return env.regenerate_obs_from_state(state)


def set_free_joint_pose(
    env: Any,
    instance: str,
    *,
    position: list[float] | None = None,
    quaternion_wxyz: list[float] | None = None,
) -> None:
    obj = env.env.objects_dict[instance]
    if len(obj.joints) != 1:
        raise ValueError(f"{instance} is not a one-free-joint object: {obj.joints}")
    joint = obj.joints[0]
    qpos = np.asarray(env.env.sim.data.get_joint_qpos(joint), dtype=float).copy()
    if qpos.shape != (7,):
        raise ValueError(f"{instance}.{joint} has qpos shape {qpos.shape}")
    if position is not None:
        qpos[:3] = np.asarray(position, dtype=float)
    if quaternion_wxyz is not None:
        qpos[3:] = np.asarray(quaternion_wxyz, dtype=float)
    env.env.sim.data.set_joint_qpos(joint, qpos)


def apply_inverted_bowl_reset(env: Any) -> dict[str, Any]:
    """Place an upside-down ramekin over a flat butter package."""
    target = "butter_1"
    cover = "glazed_rim_porcelain_ramekin_1"
    target_obj = env.env.objects_dict[target]
    cover_obj = env.env.objects_dict[cover]
    target_qpos = np.asarray(
        env.env.sim.data.get_joint_qpos(target_obj.joints[0]), dtype=float
    ).copy()
    cover_qpos = np.asarray(
        env.env.sim.data.get_joint_qpos(cover_obj.joints[0]), dtype=float
    ).copy()

    target_position = [-0.08, 0.03, 0.93]
    # The asset's upright reset quaternion is a +90-degree yaw. Premultiplying
    # it by a 180-degree x rotation yields this upside-down orientation.
    inverted_quaternion = [0.0, np.sqrt(0.5), -np.sqrt(0.5), 0.0]
    cover_position = [target_position[0], target_position[1], 0.97]

    set_free_joint_pose(
        env,
        target,
        position=target_position,
        quaternion_wxyz=target_qpos[3:].tolist(),
    )
    set_free_joint_pose(
        env,
        cover,
        position=cover_position,
        quaternion_wxyz=inverted_quaternion,
    )
    obs = regenerate_obs(env)
    return {
        "obs": obs,
        "target_qpos_before": target_qpos.tolist(),
        "cover_qpos_before": cover_qpos.tolist(),
        "target_position": target_position,
        "cover_position": cover_position,
        "cover_quaternion_wxyz": inverted_quaternion,
    }


def apply_dense_clutter_reset(env: Any) -> dict[str, Any]:
    """Create a repeatable six-object cluster with a partly visible target."""
    positions = {
        "ketchup_1": [-0.14, -0.06, 0.93],
        "alphabet_soup_1": [-0.075, -0.06, 0.93],
        "tomato_sauce_1": [-0.04, 0.015, 0.93],
        "milk_1": [-0.14, 0.045, 0.93],
        "orange_juice_1": [-0.035, -0.16, 0.93],
        "bbq_sauce_1": [-0.17, -0.16, 0.93],
    }
    before: dict[str, list[float]] = {}
    for instance, position in positions.items():
        obj = env.env.objects_dict[instance]
        qpos = np.asarray(
            env.env.sim.data.get_joint_qpos(obj.joints[0]), dtype=float
        ).copy()
        before[instance] = qpos.tolist()
        set_free_joint_pose(
            env,
            instance,
            position=position,
            quaternion_wxyz=qpos[3:].tolist(),
        )
    obs = regenerate_obs(env)
    obs = neutral_steps(env, obs, 10)
    return {
        "obs": obs,
        "object_qpos_before": before,
        "object_positions": positions,
        "post_layout_settle_steps": 10,
    }


def apply_oracle_reveal(env: Any, task_id: str) -> dict[str, Any]:
    """Change only the environment state; never move the policy camera."""
    changed_joints: dict[str, float] = {}
    changed_objects: dict[str, list[float]] = {}

    def open_drawer_with_contents(
        level: str, contents: list[str]
    ) -> None:
        site_name = f"wooden_cabinet_1_{level}_region"
        site_id = env.env.sim.model.site_name2id(site_name)
        before = np.asarray(
            env.env.sim.data.site_xpos[site_id], dtype=float
        ).copy()
        joint = f"wooden_cabinet_1_{level}_level"
        value = -0.15
        env.env.sim.data.set_joint_qpos(joint, value)
        env.env.sim.forward()
        after = np.asarray(
            env.env.sim.data.site_xpos[site_id], dtype=float
        ).copy()
        displacement = after - before
        for instance in contents:
            obj = env.env.objects_dict[instance]
            qpos = np.asarray(
                env.env.sim.data.get_joint_qpos(obj.joints[0]), dtype=float
            ).copy()
            qpos[:3] += displacement
            env.env.sim.data.set_joint_qpos(obj.joints[0], qpos)
            changed_objects[instance] = qpos[:3].tolist()
        changed_joints[joint] = value

    if task_id == "T01_drawer_retrieval":
        open_drawer_with_contents("top", ["butter_1"])
    elif task_id == "T02_fridge_retrieval":
        joint = "short_fridge_1_fridge_door_joint_0"
        value = 2.2
        env.env.sim.data.set_joint_qpos(joint, value)
        changed_joints[joint] = value
    elif task_id == "T03_inverted_bowl_retrieval":
        cover = "glazed_rim_porcelain_ramekin_1"
        cover_obj = env.env.objects_dict[cover]
        qpos = np.asarray(
            env.env.sim.data.get_joint_qpos(cover_obj.joints[0]), dtype=float
        ).copy()
        qpos[0] += 0.20
        env.env.sim.data.set_joint_qpos(cover_obj.joints[0], qpos)
        changed_joints[cover_obj.joints[0]] = float(qpos[0])
    elif task_id == "T05_exhaustive_not_found":
        open_drawer_with_contents("top", ["butter_1"])
        open_drawer_with_contents("middle", [])
        open_drawer_with_contents("bottom", [])
    elif task_id == "T06_dense_clutter_partial_occlusion":
        cleared_positions = {
            "alphabet_soup_1": [0.10, -0.18, 0.93],
        }
        for instance, position in cleared_positions.items():
            obj = env.env.objects_dict[instance]
            qpos = np.asarray(
                env.env.sim.data.get_joint_qpos(obj.joints[0]), dtype=float
            ).copy()
            qpos[:3] = np.asarray(position, dtype=float)
            env.env.sim.data.set_joint_qpos(obj.joints[0], qpos)
            changed_objects[instance] = position
    return {
        "obs": regenerate_obs(env),
        "changed_joints": changed_joints,
        "changed_objects": changed_objects,
    }


def target_pixels(
    env: Any,
    obs: dict[str, Any],
    *,
    camera: str,
    target: str | None,
) -> int | None:
    if target is None:
        return None
    target_id = env.instance_to_id[target]
    segmentation = np.asarray(
        obs[find_segmentation_key(obs, camera)]
    ).squeeze()
    return int(np.count_nonzero(segmentation == target_id))


def instance_mask_stats(
    env: Any,
    obs: dict[str, Any],
    *,
    camera: str,
    instances: list[str],
) -> dict[str, dict[str, Any]]:
    """Return visible pixels and image-space boxes for selected instances."""
    segmentation = np.flipud(
        np.asarray(obs[find_segmentation_key(obs, camera)]).squeeze()
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
    *,
    camera: str,
    target: str | None,
    output_dir: Path,
    name: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_key = f"{camera}_image"
    rgb = np.flipud(np.asarray(obs[image_key])).astype(np.uint8)
    imageio.imwrite(output_dir / f"{name}.png", rgb)
    visible_pixels = target_pixels(
        env, obs, camera=camera, target=target
    )
    if target is not None:
        target_id = env.instance_to_id[target]
        segmentation = np.asarray(
            obs[find_segmentation_key(obs, camera)]
        ).squeeze()
        mask = segmentation == target_id
        imageio.imwrite(
            output_dir / f"{name}_target_mask.png",
            np.flipud(mask.astype(np.uint8) * 255),
        )
    return {
        "image": str(output_dir / f"{name}.png"),
        "target_visible_pixels": visible_pixels,
    }


def validate_prompt_contract(spec: dict[str, Any]) -> list[str]:
    forbidden = [
        value.casefold()
        for value in spec["prompt_contract"]["forbidden_exploration_phrases"]
    ]
    errors: list[str] = []
    for task in spec["tasks"]:
        prompt = task["prompt"].casefold()
        leaks = [word for word in forbidden if word in prompt]
        if leaks:
            errors.append(f'{task["id"]}: exploration leakage {leaks}')
    return errors


def validate_case(
    *,
    env_class: Any,
    task: dict[str, Any],
    bddl_path: Path,
    seed: int,
    output_dir: Path,
    camera: str,
    width: int,
    height: int,
    settle_steps: int,
) -> dict[str, Any]:
    env = env_class(
        bddl_file_name=str(bddl_path),
        camera_heights=height,
        camera_widths=width,
    )
    try:
        env.seed(seed)
        obs = env.reset()
        obs = neutral_steps(env, obs, settle_steps)
        reset_wrapper_report = None
        if task.get("reset_wrapper") == "inverted_bowl_cover":
            reset_wrapper_report = apply_inverted_bowl_reset(env)
            obs = reset_wrapper_report.pop("obs")
        elif task.get("reset_wrapper") == "dense_clutter_partial_occlusion":
            reset_wrapper_report = apply_dense_clutter_reset(env)
            obs = reset_wrapper_report.pop("obs")

        expected_prompt = str(task["prompt"])
        actual_prompt = str(env.language_instruction)
        target = task.get("target")
        if target is not None and target not in env.instance_to_id:
            raise KeyError(
                f"{target} missing; instances={sorted(env.instance_to_id)}"
            )

        camera_before = camera_extrinsics(env, camera)
        initial = save_snapshot(
            env,
            obs,
            camera=camera,
            target=target,
            output_dir=output_dir,
            name="initial",
        )
        initial_first_person = None
        first_person_camera = "robot0_eye_in_hand"
        first_person_key = f"{first_person_camera}_image"
        if (
            task["id"] == "T06_dense_clutter_partial_occlusion"
            and first_person_key in obs
        ):
            first_person_path = output_dir / "initial_first_person.png"
            imageio.imwrite(
                first_person_path,
                np.flipud(np.asarray(obs[first_person_key])).astype(np.uint8),
            )
            initial_first_person = {
                "camera": first_person_camera,
                "image": str(first_person_path),
                "target_visible_pixels": target_pixels(
                    env,
                    obs,
                    camera=first_person_camera,
                    target=target,
                ),
            }
        clutter_instances = [
            instance
            for instance in [
                target,
                *task.get("clutter_objects", []),
            ]
            if instance is not None
        ]
        initial_clutter_stats = (
            instance_mask_stats(
                env,
                obs,
                camera=camera,
                instances=clutter_instances,
            )
            if task["family"] == "partial_occlusion_clutter"
            else None
        )
        initial_bddl_success = bool(env.env._check_success())

        reveal = apply_oracle_reveal(env, task["id"])
        revealed_obs = reveal.pop("obs")
        revealed = save_snapshot(
            env,
            revealed_obs,
            camera=camera,
            target=target,
            output_dir=output_dir,
            name="oracle_revealed",
        )
        revealed_clutter_stats = (
            instance_mask_stats(
                env,
                revealed_obs,
                camera=camera,
                instances=clutter_instances,
            )
            if task["family"] == "partial_occlusion_clutter"
            else None
        )
        camera_after = camera_extrinsics(env, camera)
        revealed_bddl_success = bool(env.env._check_success())

        absent_instance = task.get("absent_instance")
        absent_not_instantiated = (
            absent_instance not in env.instance_to_id
            if absent_instance is not None
            else None
        )
        family = task["family"]
        if family in {
            "articulated_drawer",
            "hinged_fridge",
            "removable_cover",
        }:
            visibility_valid = (
                initial["target_visible_pixels"] == 0
                and isinstance(revealed["target_visible_pixels"], int)
                and revealed["target_visible_pixels"] > 0
            )
        elif family == "visible_control":
            visibility_valid = (
                isinstance(initial["target_visible_pixels"], int)
                and initial["target_visible_pixels"] > 0
            )
        elif family == "target_absent_search":
            visibility_valid = bool(
                absent_not_instantiated and revealed_bddl_success
            )
        elif family == "partial_occlusion_clutter":
            initial_pixels = initial["target_visible_pixels"]
            revealed_pixels = revealed["target_visible_pixels"]
            max_fraction = float(
                task.get("max_initial_visible_fraction", 0.65)
            )
            min_pixels = int(task.get("min_initial_target_pixels", 1))
            min_gain = float(task.get("min_reveal_gain", 1.0))
            min_visible_clutter = int(
                task.get("min_visible_clutter_objects", 1)
            )
            visible_clutter_count = sum(
                stats["visible_pixels"] > 0
                for instance, stats in initial_clutter_stats.items()
                if instance != target
            )
            visibility_valid = bool(
                isinstance(initial_pixels, int)
                and isinstance(revealed_pixels, int)
                and initial_pixels >= min_pixels
                and revealed_pixels > initial_pixels
                and initial_pixels / revealed_pixels <= max_fraction
                and revealed_pixels / initial_pixels >= min_gain
                and visible_clutter_count >= min_visible_clutter
            )
        else:
            raise ValueError(f"Unknown family: {family}")

        result = {
            "task_id": task["id"],
            "family": family,
            "seed": seed,
            "bddl": str(bddl_path),
            "expected_prompt": expected_prompt,
            "actual_prompt": actual_prompt,
            "prompt_matches": (
                expected_prompt.casefold() == actual_prompt.casefold()
            ),
            "policy_camera": camera,
            "camera_before": camera_before,
            "camera_after": camera_after,
            "camera_unchanged": camera_unchanged(
                camera_before, camera_after
            ),
            "active_viewpoint_actions_executed": 0,
            "target": target,
            "absent_instance": absent_instance,
            "absent_not_instantiated": absent_not_instantiated,
            "initial": initial,
            "initial_first_person": initial_first_person,
            "oracle_revealed": revealed,
            "initial_bddl_success": initial_bddl_success,
            "revealed_bddl_success": revealed_bddl_success,
            "oracle_state_change": reveal,
            "reset_wrapper": reset_wrapper_report,
            "initial_clutter_stats": initial_clutter_stats,
            "revealed_clutter_stats": revealed_clutter_stats,
            "partial_occlusion_metrics": (
                {
                    "initial_visible_fraction": (
                        initial["target_visible_pixels"]
                        / revealed["target_visible_pixels"]
                        if revealed["target_visible_pixels"]
                        else None
                    ),
                    "reveal_gain": (
                        revealed["target_visible_pixels"]
                        / initial["target_visible_pixels"]
                        if initial["target_visible_pixels"]
                        else None
                    ),
                    "visible_clutter_objects": sum(
                        stats["visible_pixels"] > 0
                        for instance, stats in initial_clutter_stats.items()
                        if instance != target
                    ),
                }
                if family == "partial_occlusion_clutter"
                else None
            ),
            "visibility_or_absence_valid": visibility_valid,
        }
        result["passed"] = bool(
            result["prompt_matches"]
            and result["camera_unchanged"]
            and not result["initial_bddl_success"]
            and result["visibility_or_absence_valid"]
        )
        with (output_dir / "case_report.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(result, file, indent=2, ensure_ascii=False)
        return result
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    spec_path = Path(args.spec).expanduser().resolve()
    bddl_root = Path(args.bddl_root).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    configure_imports(project_root)

    from libero.libero.envs import SegmentationRenderEnv

    spec = load_yaml(spec_path)
    prompt_errors = validate_prompt_contract(spec)
    selected_task_ids = set(
        args.task_ids or [task["id"] for task in spec["tasks"]]
    )
    known_task_ids = {task["id"] for task in spec["tasks"]}
    unknown_task_ids = selected_task_ids - known_task_ids
    if unknown_task_ids:
        raise KeyError(f"Unknown task IDs: {sorted(unknown_task_ids)}")
    rows: list[dict[str, Any]] = []
    for task in spec["tasks"]:
        if task["id"] not in selected_task_ids:
            continue
        bddl_path = bddl_root / task["bddl"]
        for seed in args.seeds:
            print(f'[validate] {task["id"]} seed={seed}', flush=True)
            case_output = output_root / task["id"] / f"seed_{seed:03d}"
            try:
                row = validate_case(
                    env_class=SegmentationRenderEnv,
                    task=task,
                    bddl_path=bddl_path,
                    seed=seed,
                    output_dir=case_output,
                    camera=spec["backend"]["policy_camera"],
                    width=args.width,
                    height=args.height,
                    settle_steps=args.settle_steps,
                )
            except Exception as error:
                row = {
                    "task_id": task["id"],
                    "seed": seed,
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                }
                print(f'  FAILED: {row["error"]}', flush=True)
            rows.append(row)

    summary = {
        "benchmark": spec["name"],
        "version": spec["version"],
        "spec": str(spec_path),
        "policy_camera": spec["backend"]["policy_camera"],
        "active_viewpoint_actions": spec["backend"][
            "active_viewpoint_actions"
        ],
        "prompt_contract_errors": prompt_errors,
        "seeds": args.seeds,
        "case_runs": len(rows),
        "case_passes": sum(bool(row.get("passed")) for row in rows),
        "all_checks_passed": (
            not prompt_errors and all(row.get("passed") for row in rows)
        ),
        "rows": rows,
    }
    report_path = output_root / "validation_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key != "rows"
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Report: {report_path}")
    if args.strict and not summary["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
