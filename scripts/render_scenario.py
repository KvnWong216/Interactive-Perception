#!/usr/bin/env python3
"""Render a custom BDDL scenario and report oracle instance visibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from _bootstrap import resolve_project_path

from libero.libero.envs import SegmentationRenderEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl", required=True)
    parser.add_argument("--target", default="alphabet_soup_1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=5,
        help="Run neutral actions before measuring the observation.",
    )
    parser.add_argument(
        "--output",
        default="outputs/custom_scenario",
    )
    return parser.parse_args()


def colorize(labels: np.ndarray) -> np.ndarray:
    """Create a deterministic RGB visualization without opening a GUI window."""
    labels = np.asarray(labels).squeeze().astype(np.int64)
    rng = np.random.RandomState(7)
    palette = rng.randint(0, 256, size=(256, 3), dtype=np.uint8)
    palette[0] = 0
    return palette[np.mod(labels, 256)]


def find_segmentation_key(obs: dict, camera: str = "agentview") -> str:
    candidates = [
        key
        for key in obs
        if key.startswith(camera) and "segmentation" in key
    ]
    if not candidates:
        raise KeyError(
            f"No {camera} segmentation observation. Available: {sorted(obs)}"
        )
    return candidates[0]


def main() -> None:
    args = parse_args()
    bddl_path = resolve_project_path(args.bddl)
    output_dir = resolve_project_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = SegmentationRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=args.height,
        camera_widths=args.width,
    )
    try:
        env.seed(args.seed)
        obs = env.reset()
        action_low, _ = env.env.action_spec
        neutral_action = np.zeros_like(action_low)
        for _ in range(args.settle_steps):
            obs, _, _, _ = env.step(neutral_action)
        rgb = np.flipud(obs["agentview_image"]).astype(np.uint8)
        seg_key = find_segmentation_key(obs)
        segmentation = np.asarray(obs[seg_key]).squeeze()

        if args.target not in env.instance_to_id:
            raise KeyError(
                f"Unknown target {args.target!r}; instances: "
                f"{sorted(env.instance_to_id)}"
            )
        target_id = env.instance_to_id[args.target]
        target_mask = segmentation == target_id
        visible_pixels = int(target_mask.sum())
        image_pixels = int(target_mask.size)
        visibility_ratio = visible_pixels / image_pixels

        imageio.imwrite(output_dir / "agentview.png", rgb)
        imageio.imwrite(
            output_dir / "instance_segmentation.png",
            np.flipud(colorize(segmentation)),
        )
        imageio.imwrite(
            output_dir / "target_mask.png",
            np.flipud(target_mask.astype(np.uint8) * 255),
        )

        report = {
            "bddl": str(bddl_path),
            "instruction": env.language_instruction,
            "target": args.target,
            "target_instance_id": int(target_id),
            "action_dimension": int(neutral_action.size),
            "settle_steps": args.settle_steps,
            "visible_pixels": visible_pixels,
            "image_pixels": image_pixels,
            "image_occupancy_ratio": visibility_ratio,
            "note": "Segmentation is benchmark-only oracle metadata, not a policy input.",
        }
        with (output_dir / "visibility.json").open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)

        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Saved RGB, mask, segmentation, and report under {output_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
