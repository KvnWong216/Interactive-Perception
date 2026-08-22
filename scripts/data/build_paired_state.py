#!/usr/bin/env python3
"""Build a closed/open paired state from fully parameterized scene metadata.

This is an evaluator-side data constructor, not an online controller.  The
target follows the region site for translating drawers when
``--target-follows-region`` is set; hinged doors leave it fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/infra"), str(ROOT / "src")]
import bootstrap  # noqa: F401,E402
from interactive_perception.policy_client import build_observation  # noqa: E402


DUMMY_ACTION = [0.0] * 6 + [-1.0]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def object_qpos(env: Any, name: str) -> np.ndarray:
    obj = env.env.objects_dict[name]
    return np.asarray(env.env.sim.data.get_joint_qpos(obj.joints[0]), dtype=float).copy()


def set_object_qpos(env: Any, name: str, qpos: np.ndarray) -> None:
    obj = env.env.objects_dict[name]
    env.env.sim.data.set_joint_qpos(obj.joints[0], np.asarray(qpos, dtype=float))
    env.env.sim.forward()


def regenerate(env: Any) -> Mapping[str, Any]:
    env.env.sim.forward()
    return env.env._get_observations()


def visible_pixels(env: Any, observation: Mapping[str, Any], camera: str, target: str) -> int:
    keys = [key for key in observation if key.startswith(camera) and "segmentation" in key]
    if len(keys) != 1:
        raise KeyError(f"expected one {camera} segmentation image, got {keys}")
    values = np.asarray(observation[keys[0]]).squeeze()
    return int(np.count_nonzero(values == env.instance_to_id[target]))


def visible_bbox(
    env: Any, observation: Mapping[str, Any], camera: str, target: str
) -> list[int] | None:
    """Return the evaluator-only target mask bounds as ``[x0, y0, x1, y1]``."""
    keys = [key for key in observation if key.startswith(camera) and "segmentation" in key]
    if len(keys) != 1:
        raise KeyError(f"expected one {camera} segmentation image, got {keys}")
    values = np.asarray(observation[keys[0]]).squeeze()
    ys, xs = np.nonzero(values == env.instance_to_id[target])
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def snapshot(
    env: Any,
    observation: Mapping[str, Any],
    *,
    target: str,
    joint: str,
    region: str,
) -> dict[str, Any]:
    target_position = object_qpos(env, target)[:3]
    region_position = np.asarray(env.env.sim.data.get_site_xpos(region), dtype=float)
    region_rotation = np.asarray(env.env.sim.data.get_site_xmat(region), dtype=float)
    site_id = env.env.sim.model.site_name2id(region)
    return {
        "target_qpos": object_qpos(env, target).tolist(),
        "joint": float(env.env.sim.data.get_joint_qpos(joint)),
        "inside_region": bool(env.env._eval_predicate(["in", target, region])),
        "region": {
            "position": region_position.tolist(),
            "rendered_half_size": np.asarray(
                env.env.sim.model.site_size[site_id], dtype=float
            ).tolist(),
            "declared_half_size": np.asarray(
                env.env.object_sites_dict[region].size, dtype=float
            ).tolist(),
            "target_center_local": (region_rotation.T @ (target_position - region_position)).tolist(),
        },
        "visible_pixels": {
            camera: visible_pixels(env, observation, camera, target)
            for camera in ("agentview", "robot0_eye_in_hand")
        },
        "visible_bbox": {
            camera: visible_bbox(env, observation, camera, target)
            for camera in ("agentview", "robot0_eye_in_hand")
        },
    }


def save_public_views(
    observation: Mapping[str, Any], *, prompt: str, stem: str, directory: Path
) -> dict[str, str]:
    packet = build_observation(observation, prompt)
    paths = {}
    for view, image in (("agentview", packet.image), ("wrist", packet.wrist_image)):
        path = directory / f"{stem}_{view}.png"
        imageio.imwrite(path, image)
        paths[view] = portable_path(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--joint", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--closed-joint", type=float, required=True)
    parser.add_argument("--open-joint", type=float, required=True)
    parser.add_argument("--target-shift", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--target-follows-region", action="store_true")
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--resolvable-pixels", type=int, default=256)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for name in ("bddl", "states", "assets", "report"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.states.exists() or args.assets.exists() or args.report.exists():
        raise FileExistsError("paired-state outputs are immutable")

    from libero.libero.envs import SegmentationRenderEnv

    # LIBERO's placement samplers also consume process-global RNG state before
    # and during reset.  Seed both sources so a declared benchmark seed really
    # identifies one reproducible scene across fresh processes.
    random.seed(args.seed)
    np.random.seed(args.seed)
    env = SegmentationRenderEnv(
        bddl_file_name=str(args.bddl), camera_heights=256, camera_widths=256
    )
    try:
        env.seed(args.seed)
        env.reset()
        initial_site = np.asarray(
            env.env.sim.data.get_site_xpos(args.region), dtype=float
        ).copy()
        initial_target = object_qpos(env, args.target)
        env.env.sim.data.set_joint_qpos(args.joint, args.closed_joint)
        env.env.sim.forward()
        closed_site = np.asarray(
            env.env.sim.data.get_site_xpos(args.region), dtype=float
        ).copy()
        target = initial_target.copy()
        if args.target_follows_region:
            # Keep the target's local container coordinates fixed when the
            # source BDDL starts with an open articulated region. This lets a
            # paired benchmark close the native clear setup without asking the
            # placement sampler to fit the object through a closed aperture.
            target[:3] += closed_site - initial_site
        target[:3] += np.asarray(args.target_shift, dtype=float)
        set_object_qpos(env, args.target, target)
        observation = regenerate(env)
        for _ in range(args.settle_steps):
            observation, _, _, _ = env.step(DUMMY_ACTION)
        # MuJoCo exposes a live view; copy before changing the articulated state.
        closed_state = env.get_sim_state().copy()
        closed = snapshot(
            env, observation, target=args.target, joint=args.joint, region=args.region
        )

        env.env.sim.data.set_joint_qpos(args.joint, args.open_joint)
        env.env.sim.forward()
        if args.target_follows_region:
            open_site = np.asarray(
                env.env.sim.data.get_site_xpos(args.region), dtype=float
            ).copy()
            target = object_qpos(env, args.target)
            target[:3] += open_site - closed_site
            set_object_qpos(env, args.target, target)
        observation = regenerate(env)
        for _ in range(args.settle_steps):
            observation, _, _, _ = env.step(DUMMY_ACTION)
        open_state = env.get_sim_state().copy()
        opened = snapshot(
            env, observation, target=args.target, joint=args.joint, region=args.region
        )
        qualification = {
            "closed_not_resolvable": max(closed["visible_pixels"].values())
            < args.resolvable_pixels,
            "open_resolvable": max(opened["visible_pixels"].values())
            >= args.resolvable_pixels,
            "inside_closed": closed["inside_region"],
            "inside_open": opened["inside_region"],
        }
        qualification["passed"] = all(qualification.values())

        args.assets.mkdir(parents=True)
        env.set_init_state(closed_state)
        closed_paths = save_public_views(
            regenerate(env), prompt=args.prompt, stem="closed", directory=args.assets
        )
        env.set_init_state(open_state)
        open_paths = save_public_views(
            regenerate(env), prompt=args.prompt, stem="open", directory=args.assets
        )
        report = {
            "schema_version": "piu.paired-state.v1",
            "status": "PASS" if qualification["passed"] else "FAIL",
            "scenario": portable_path(args.bddl),
            "prompt": args.prompt,
            "seed": args.seed,
            "target": args.target,
            "joint": args.joint,
            "region": args.region,
            "target_shift": list(args.target_shift),
            "target_follows_region": args.target_follows_region,
            "closed": closed,
            "open": opened,
            "qualification": qualification,
            "public_images": {"closed": closed_paths, "open": open_paths},
            "online_controller_inputs": [],
            "note": "evaluator-only benchmark construction",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        if not qualification["passed"]:
            raise RuntimeError(json.dumps(qualification))
        args.states.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.states, closed=closed_state, open=open_state)
        report["states_sha256"] = digest(args.states)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
