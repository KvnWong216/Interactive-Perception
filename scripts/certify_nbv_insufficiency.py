#!/usr/bin/env python3
"""Certify that no camera pose can substitute for manipulating the scene.

Forbidding viewpoint change by fiat proves nothing: a reader is entitled to ask
whether the robot could simply have looked from somewhere else.  This script
answers that empirically.  It sweeps a dense hemisphere of camera poses around
the workspace, renders instance segmentation from each, and records the best
target visibility any viewpoint achieves.  It then applies the scenario's oracle
manipulation and sweeps again.

A scenario earns an ``NBV_INSUFFICIENT`` certificate when the pre-manipulation
ceiling is at or below ``--visible-threshold`` while the post-manipulation
ceiling clears it.  That is the precise statement the benchmark needs: the
information is absent from *every* viewpoint until the world is changed.

Scenarios that fail the certificate are not defects.  A scene whose target can
be seen from some other angle is a legitimate control condition -- it separates
"needs a better view" from "needs a different world" -- and this script labels
it ``NBV_SUFFICIENT`` rather than rejecting it.

The camera is restored and verified bit-exact before the script exits, so a
certified scene is never left perturbed for downstream runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import _bootstrap  # noqa: F401,E402  (sets LIBERO paths and MUJOCO_GL)

from interactive_perception.anchors import visible_pixels  # noqa: E402


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument(
        "--spec",
        default=str(root / "benchmarks" / "interactive_manipulation_v0" / "benchmark.yaml"),
    )
    parser.add_argument("--output", default=str(root / "outputs" / "nbv_certificates"))
    parser.add_argument("--task-ids", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--settle-steps", type=int, default=5)
    parser.add_argument(
        "--azimuths", type=int, default=24, help="camera positions per elevation ring"
    )
    parser.add_argument("--elevations", type=int, default=5)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.6, 0.9, 1.2])
    parser.add_argument(
        "--visible-threshold",
        type=int,
        default=4,
        help=(
            "target pixels at or below which the target counts as unobservable. "
            "This is the measured instrument noise floor, not a convenience. "
            "Rasterizing a mesh container resting on a mesh surface produces "
            "isolated single-ray seam hits that no geometry change removes: "
            "under a 2184-pose sweep the sealed refrigerator leaks 1 px from 1 "
            "ray, and so does the closed drawer. A strict zero is therefore "
            "unattainable for any scene and merely reports how sparsely the "
            "hemisphere was sampled. 4 px sits above that floor and roughly "
            "180x below the smallest post-manipulation visibility in the suite "
            "(737 px), so it cannot launder a scene that actually leaks -- the "
            "ramekin this suite used to use scored 59 px from 255 of 360 poses "
            "and would still fail by an order of magnitude."
        ),
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def mat_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a ``wxyz`` quaternion (MuJoCo convention)."""

    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        quat = np.array(
            [
                0.25 / s,
                (m[2, 1] - m[1, 2]) * s,
                (m[0, 2] - m[2, 0]) * s,
                (m[1, 0] - m[0, 1]) * s,
            ]
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + m[0, 0] - m[1, 1] - m[2, 2]))
        quat = np.array(
            [
                (m[2, 1] - m[1, 2]) / s,
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s,
            ]
        )
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + m[1, 1] - m[0, 0] - m[2, 2]))
        quat = np.array(
            [
                (m[0, 2] - m[2, 0]) / s,
                (m[0, 1] + m[1, 0]) / s,
                0.25 * s,
                (m[1, 2] + m[2, 1]) / s,
            ]
        )
    else:
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + m[2, 2] - m[0, 0] - m[1, 1]))
        quat = np.array(
            [
                (m[1, 0] - m[0, 1]) / s,
                (m[0, 2] + m[2, 0]) / s,
                (m[1, 2] + m[2, 1]) / s,
                0.25 * s,
            ]
        )
    return quat / float(np.linalg.norm(quat))


def look_at_quaternion(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Orientation placing a MuJoCo camera at ``eye`` looking at ``target``.

    MuJoCo cameras look down their local ``-z`` with ``+y`` up, so the frame's
    ``+z`` axis points from the target back toward the eye.
    """

    z_axis = eye - target
    norm = float(np.linalg.norm(z_axis))
    if norm <= 1e-9:
        raise ValueError("camera eye coincides with its look-at target")
    z_axis = z_axis / norm

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(z_axis @ world_up)) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(world_up, z_axis)
    x_axis = x_axis / float(np.linalg.norm(x_axis))
    y_axis = np.cross(z_axis, x_axis)
    return mat_to_quat_wxyz(np.stack([x_axis, y_axis, z_axis], axis=1))


def hemisphere_poses(
    center: np.ndarray, *, azimuths: int, elevations: int, radii: list[float]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Camera poses on an upper hemisphere, all aimed at ``center``.

    Elevations start above the horizon and stop short of straight down, since a
    camera exactly on the vertical axis has a degenerate up-vector and adds no
    coverage a nearby ring does not already provide.
    """

    poses: list[tuple[np.ndarray, np.ndarray]] = []
    for radius in radii:
        for elevation in np.linspace(np.deg2rad(10.0), np.deg2rad(80.0), elevations):
            for azimuth in np.linspace(0.0, 2.0 * np.pi, azimuths, endpoint=False):
                offset = np.array(
                    [
                        radius * np.cos(elevation) * np.cos(azimuth),
                        radius * np.cos(elevation) * np.sin(azimuth),
                        radius * np.sin(elevation),
                    ]
                )
                eye = center + offset
                poses.append((eye, look_at_quaternion(eye, center)))
    return poses


def sweep(
    env: Any,
    *,
    camera: str,
    target: str,
    center: np.ndarray,
    poses: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Return the best target visibility achievable over all camera poses."""

    camera_id = env.env.sim.model.camera_name2id(camera)
    best_pixels = 0
    best_index = -1
    visible_poses = 0

    for index, (eye, quat) in enumerate(poses):
        env.env.sim.model.cam_pos[camera_id] = eye
        env.env.sim.model.cam_quat[camera_id] = quat
        env.env.sim.forward()
        obs = env.regenerate_obs_from_state(env.get_sim_state())
        pixels = visible_pixels(env, obs, camera=camera, instance=target)
        if pixels > 0:
            visible_poses += 1
        if pixels > best_pixels:
            best_pixels = pixels
            best_index = index

    return {
        "poses_evaluated": len(poses),
        "poses_with_visible_target": visible_poses,
        "max_visible_pixels": int(best_pixels),
        "best_pose_index": int(best_index),
        "best_pose": (
            {
                "eye": poses[best_index][0].tolist(),
                "quat_wxyz": poses[best_index][1].tolist(),
            }
            if best_index >= 0
            else None
        ),
        "look_at_center": center.tolist(),
    }


def _body_position(env: Any, name: str) -> np.ndarray | None:
    try:
        body_id = env.env.sim.model.body_name2id(name)
    except Exception:  # noqa: BLE001 - absence is an expected outcome here
        return None
    return np.asarray(env.env.sim.data.body_xpos[body_id], dtype=np.float64).copy()


def _free_joint_position(env: Any, instance: str) -> np.ndarray | None:
    """Position of a single-free-joint object, or None if it is not one.

    Fixtures also carry joints, but slide and hinge joints have scalar qpos
    rather than the seven-vector a free joint uses. Checking the size keeps a
    cabinet's drawer-slide joint from being read as a position.
    """

    table = getattr(env.env, "objects_dict", None) or {}
    if instance not in table or not table[instance].joints:
        return None
    qpos = np.asarray(
        env.env.sim.data.get_joint_qpos(table[instance].joints[0]), dtype=np.float64
    ).ravel()
    return qpos[:3].copy() if qpos.size >= 7 else None


def look_at_center(env: Any, task: dict[str, Any]) -> np.ndarray:
    """Resolve the point the camera sweep aims at.

    Preference order: an explicit ``nbv_look_at_site``; then the fixture that
    hides or contains the target, tried as ``<fixture>_main``, as a body of the
    same name, and as a free-joint object; then, only when the scene hides
    nothing, the target itself.

    For a scene with a hidden target the target's own position is never used.
    Deriving the look-at point from it would leak its location into the sweep
    geometry and bias the visibility ceiling upward. A scene with no reveal
    fixture has nothing to leak -- its target is already in plain view -- so
    aiming at the target there is both safe and the only sensible choice.

    Raises rather than guessing. A sweep aimed at the wrong place undercounts
    visibility and would manufacture the very verdict this script exists to
    earn honestly.
    """

    site = task.get("nbv_look_at_site")
    if site:
        site_id = env.env.sim.model.site_name2id(site)
        return np.asarray(env.env.sim.data.site_xpos[site_id], dtype=np.float64).copy()

    fixture = task.get("reveal_fixture") or task.get("searched_fixture")
    if fixture:
        for candidate in (
            _body_position(env, f"{fixture}_main"),
            _body_position(env, fixture),
            _free_joint_position(env, fixture),
        ):
            if candidate is not None:
                return candidate
        raise KeyError(
            f"{task['id']}: cannot locate fixture {fixture!r} as a body or object. "
            f"Run scripts/list_scene_handles.py to find the correct name, then set "
            f"`nbv_look_at_site` on the task."
        )

    target = task.get("target")
    if target:
        for candidate in (
            _free_joint_position(env, target),
            _body_position(env, f"{target}_main"),
            _body_position(env, target),
        ):
            if candidate is not None:
                return candidate

    raise KeyError(
        f"{task['id']}: nothing to aim the sweep at. Set `nbv_look_at_site`, "
        f"`reveal_fixture`, or `target`."
    )


def certify_task(
    *,
    env_class: Any,
    task: dict[str, Any],
    bddl_path: Path,
    args: argparse.Namespace,
    camera: str,
    reveal: Any,
    reset_wrappers: dict[str, Any],
) -> dict[str, Any]:
    target = task.get("target")
    if target is None:
        return {
            "task_id": task["id"],
            "verdict": "NOT_APPLICABLE",
            "passed": True,
            "reason": "target-absent scenario has no target to make visible",
        }

    env = env_class(
        bddl_file_name=str(bddl_path),
        camera_heights=args.height,
        camera_widths=args.width,
    )
    try:
        env.seed(args.seed)
        obs = env.reset()
        action = np.zeros_like(env.env.action_spec[0])
        for _ in range(args.settle_steps):
            obs, _, _, _ = env.step(action)

        # Scenes whose configuration is applied at reset rather than encoded in
        # the BDDL must be configured before the sweep. Certifying T03 without
        # its cover, or T06 without its clutter layout, would sweep a scene that
        # is not the one under test and report a verdict for the wrong world.
        wrapper_name = task.get("reset_wrapper")
        if wrapper_name:
            if wrapper_name not in reset_wrappers:
                raise KeyError(
                    f"{task['id']}: no reset wrapper registered for {wrapper_name!r}"
                )
            reset_wrappers[wrapper_name](env)

        if target not in env.instance_to_id:
            raise KeyError(f"{target} not instantiated in {task['id']}")

        camera_id = env.env.sim.model.camera_name2id(camera)
        original_pos = np.array(env.env.sim.model.cam_pos[camera_id], dtype=np.float64)
        original_quat = np.array(env.env.sim.model.cam_quat[camera_id], dtype=np.float64)

        # Aim the sweep at the fixture that hides the target when the target
        # itself is unobservable; a look-at point derived from the hidden object
        # would leak its position into the sweep geometry.
        #
        # This must never fall back to a guessed centre. A sweep aimed at the
        # wrong place sees less than it should, which would fabricate an
        # NBV_INSUFFICIENT verdict -- exactly the conclusion the certificate is
        # supposed to earn. Failing loudly is the only safe behaviour.
        center = look_at_center(env, task)

        poses = hemisphere_poses(
            center,
            azimuths=args.azimuths,
            elevations=args.elevations,
            radii=list(args.radii),
        )
        before = sweep(env, camera=camera, target=target, center=center, poses=poses)

        env.env.sim.model.cam_pos[camera_id] = original_pos
        env.env.sim.model.cam_quat[camera_id] = original_quat
        env.env.sim.forward()

        reveal(env, task["id"])
        after = sweep(env, camera=camera, target=target, center=center, poses=poses)

        env.env.sim.model.cam_pos[camera_id] = original_pos
        env.env.sim.model.cam_quat[camera_id] = original_quat
        env.env.sim.forward()
        restored = bool(
            np.array_equal(env.env.sim.model.cam_pos[camera_id], original_pos)
            and np.array_equal(env.env.sim.model.cam_quat[camera_id], original_quat)
        )

        nbv_insufficient = before["max_visible_pixels"] <= args.visible_threshold
        manipulation_reveals = after["max_visible_pixels"] > args.visible_threshold
        if nbv_insufficient and manipulation_reveals:
            verdict = "NBV_INSUFFICIENT"
        elif not nbv_insufficient:
            verdict = "NBV_SUFFICIENT"
        else:
            verdict = "UNREVEALABLE"

        return {
            "task_id": task["id"],
            "family": task.get("family"),
            "target": target,
            "seed": args.seed,
            "camera": camera,
            "visible_threshold": args.visible_threshold,
            "pre_manipulation_sweep": before,
            "post_manipulation_sweep": after,
            "camera_restored": restored,
            "verdict": verdict,
            "passed": bool(restored and verdict in {"NBV_INSUFFICIENT", "NBV_SUFFICIENT"}),
        }
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    spec_path = Path(args.spec).expanduser().resolve()
    root = Path(args.project_root).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    from libero.libero.envs import SegmentationRenderEnv

    sys.path.insert(0, str(root / "scripts"))
    from validate_interactive_manipulation_v0 import (
        apply_dense_clutter_reset,
        apply_inverted_bowl_reset,
        apply_oracle_reveal,
    )

    # Reuse the validators' wrappers verbatim so a certificate is issued for
    # exactly the scene configuration the benchmark validates.
    reset_wrappers = {
        "inverted_bowl_cover": apply_inverted_bowl_reset,
        "dense_clutter_partial_occlusion": apply_dense_clutter_reset,
    }

    with spec_path.open("r", encoding="utf-8") as file:
        spec = yaml.safe_load(file)

    camera = spec["backend"]["policy_camera"]
    selected = set(args.task_ids or [task["id"] for task in spec["tasks"]])
    rows: list[dict[str, Any]] = []
    for task in spec["tasks"]:
        if task["id"] not in selected:
            continue
        print(f'[certify] {task["id"]}', flush=True)
        try:
            row = certify_task(
                env_class=SegmentationRenderEnv,
                task=task,
                bddl_path=root / task["bddl"],
                args=args,
                camera=camera,
                reveal=apply_oracle_reveal,
                reset_wrappers=reset_wrappers,
            )
        except Exception as error:  # noqa: BLE001 - reported, not hidden
            row = {
                "task_id": task["id"],
                "passed": False,
                "verdict": "ERROR",
                "error": f"{type(error).__name__}: {error}",
            }
        print(f'  {row.get("verdict")} '
              f'pre={row.get("pre_manipulation_sweep", {}).get("max_visible_pixels")} '
              f'post={row.get("post_manipulation_sweep", {}).get("max_visible_pixels")}',
              flush=True)
        rows.append(row)

    summary = {
        "benchmark": spec["name"],
        "version": spec["version"],
        "seed": args.seed,
        "sweep": {
            "azimuths": args.azimuths,
            "elevations": args.elevations,
            "radii": list(args.radii),
            "poses_per_sweep": args.azimuths * args.elevations * len(args.radii),
        },
        "visible_threshold": args.visible_threshold,
        "nbv_insufficient": [
            row["task_id"] for row in rows if row.get("verdict") == "NBV_INSUFFICIENT"
        ],
        "nbv_sufficient": [
            row["task_id"] for row in rows if row.get("verdict") == "NBV_SUFFICIENT"
        ],
        "all_checks_passed": all(row.get("passed") for row in rows),
        "rows": rows,
    }
    report_path = output_root / "nbv_certificates.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(f"Report: {report_path}")
    if args.strict and not summary["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
