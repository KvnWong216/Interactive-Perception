#!/usr/bin/env python3
"""Build paired T01 closed/open states with an easy front-positioned target.

This is evaluator-side benchmark construction.  The target pose and drawer
joint are used only while freezing the paired initial states; online
controllers receive neither value.  The two states share every simulator
coordinate except the declared drawer translation and the corresponding rigid
translation of the butter with the drawer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from interactive_perception.policy_client import build_observation  # noqa: E402
from validate_information_enrichment_v0 import (  # noqa: E402
    instance_stats,
    object_qpos,
    regenerate_obs,
    set_object_qpos,
)


DUMMY_ACTION = [0.0] * 6 + [-1.0]
DEFAULT_BDDL = ROOT / "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl"
DRAWER_JOINT = "wooden_cabinet_1_middle_level"
DRAWER_SITE = "wooden_cabinet_1_middle_region"
TARGET = "butter_1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_images(
    observation: dict[str, Any], *, name: str, asset_dir: Path
) -> dict[str, Any]:
    packet = build_observation(observation, "Place the butter in the basket")
    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for view, image in (("agentview", packet.image), ("wrist", packet.wrist_image)):
        path = asset_dir / f"{name}_{view}.png"
        imageio.imwrite(path, image)
        paths[view] = str(path.relative_to(ROOT))
        hashes[view] = digest(path)
    return {"paths": paths, "sha256": hashes}


def evaluator_snapshot(env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    sim = env.env.sim
    return {
        "target_qpos": object_qpos(env, TARGET).tolist(),
        "drawer_joint": float(sim.data.get_joint_qpos(DRAWER_JOINT)),
        "drawer_site_world_xyz": sim.data.get_site_xpos(DRAWER_SITE).tolist(),
        "target_inside_middle_region": bool(
            env.env._eval_predicate(["in", TARGET, DRAWER_SITE])
        ),
        "target_visible_pixels": {
            camera: int(
                instance_stats(env, observation, camera, [TARGET])[TARGET][
                    "visible_pixels"
                ]
            )
            for camera in ("agentview", "robot0_eye_in_hand")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl", type=Path, default=DEFAULT_BDDL)
    parser.add_argument("--seed", type=int, default=1399)
    parser.add_argument("--front-shift-m", type=float, default=0.04)
    parser.add_argument("--lateral-shift-m", type=float, default=0.0)
    parser.add_argument("--open-joint", type=float, default=-0.15)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--output-states", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("bddl", "output_states", "output_report", "asset_dir"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output_states.exists() or args.output_report.exists() or args.asset_dir.exists():
        raise FileExistsError("paired-state artifacts are immutable")
    if not 0.0 <= args.front_shift_m <= 0.05:
        raise ValueError("front shift must lie in the preregistered easy range [0, 0.05]")
    if not -0.06 <= args.lateral_shift_m <= 0.06:
        raise ValueError("lateral shift must lie in the development range [-0.06, 0.06]")
    if not -0.16 <= args.open_joint <= -0.14:
        raise ValueError("open joint must lie in the stock wooden-cabinet range")

    from libero.libero.envs import SegmentationRenderEnv

    env = SegmentationRenderEnv(
        bddl_file_name=str(args.bddl), camera_heights=256, camera_widths=256
    )
    try:
        env.seed(args.seed)
        observation = env.reset()
        for _ in range(args.settle_steps):
            observation, _, _, _ = env.step(DUMMY_ACTION)

        closed_site = np.asarray(
            env.env.sim.data.get_site_xpos(DRAWER_SITE), dtype=np.float64
        ).copy()
        closed_target = object_qpos(env, TARGET)
        # The cabinet is rotated pi around world z; its front points toward +y.
        closed_target[0] += args.lateral_shift_m
        closed_target[1] += args.front_shift_m
        set_object_qpos(env, TARGET, closed_target)
        observation = regenerate_obs(env)
        for _ in range(args.settle_steps):
            observation, _, _, _ = env.step(DUMMY_ACTION)
        closed_state = env.get_sim_state().copy()
        closed_snapshot = evaluator_snapshot(env, observation)

        env.env.sim.data.set_joint_qpos(DRAWER_JOINT, args.open_joint)
        env.env.sim.forward()
        open_site = np.asarray(
            env.env.sim.data.get_site_xpos(DRAWER_SITE), dtype=np.float64
        ).copy()
        drawer_translation = open_site - closed_site
        open_target = object_qpos(env, TARGET)
        open_target[:3] += drawer_translation
        set_object_qpos(env, TARGET, open_target)
        observation = regenerate_obs(env)
        for _ in range(args.settle_steps):
            observation, _, _, _ = env.step(DUMMY_ACTION)
        open_state = env.get_sim_state().copy()
        open_snapshot = evaluator_snapshot(env, observation)

        args.asset_dir.mkdir(parents=True)
        env.set_init_state(closed_state)
        closed_observation = regenerate_obs(env)
        closed_images = public_images(
            closed_observation, name="closed_easy", asset_dir=args.asset_dir
        )
        env.set_init_state(open_state)
        open_observation = regenerate_obs(env)
        open_images = public_images(
            open_observation, name="open_easy", asset_dir=args.asset_dir
        )
    finally:
        env.close()

    if not closed_snapshot["target_inside_middle_region"]:
        raise RuntimeError("closed easy target is not inside the middle drawer")
    if not open_snapshot["target_inside_middle_region"]:
        raise RuntimeError("open easy target is not inside the middle drawer")
    args.output_states.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_states,
        closed_easy=closed_state,
        open_easy=open_state,
    )
    report = {
        "schema_version": "interaction-uncertainty.t01-easy-handoff-states.v1",
        "status": "DISPOSABLE_BENCHMARK_CONSTRUCTION",
        "source_bddl": str(args.bddl.relative_to(ROOT)),
        "seed": args.seed,
        "construction": {
            "front_shift_m": args.front_shift_m,
            "lateral_shift_m": args.lateral_shift_m,
            "open_joint": args.open_joint,
            "settle_steps": args.settle_steps,
            "drawer_translation_world_m": drawer_translation.tolist(),
            "changed_coordinates": [
                "butter world y in both paired states",
                "butter world x in both paired states",
                "middle drawer joint in open_easy",
                "butter rigid translation with the opened drawer in open_easy",
            ],
        },
        "controller_visible_contract": [
            "agentview RGB",
            "wrist RGB",
            "public robot state",
            "task prompt",
        ],
        "controller_online_oracle_inputs": [],
        "evaluator_only_construction_inputs": [
            "target free-joint pose",
            "drawer joint",
            "drawer region site transform",
            "segmentation visibility",
            "containment predicate",
        ],
        "states": {
            "path": str(args.output_states.relative_to(ROOT)),
            "sha256": digest(args.output_states),
            "keys": ["closed_easy", "open_easy"],
        },
        "closed_easy": {"evaluator_only": closed_snapshot, "public_images": closed_images},
        "open_easy": {"evaluator_only": open_snapshot, "public_images": open_images},
        "qualification_rule": "open_easy must pass frozen pi05 original-prompt capability before closed_easy is used for the PIU loop",
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "states": str(args.output_states),
                "closed_pixels": closed_snapshot["target_visible_pixels"],
                "open_pixels": open_snapshot["target_visible_pixels"],
                "closed_inside": closed_snapshot["target_inside_middle_region"],
                "open_inside": open_snapshot["target_inside_middle_region"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
