#!/usr/bin/env python3
"""Map object visibility in the unmodified LIBERO wrist-camera workspace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import _bootstrap  # noqa: F401,E402
from validate_information_enrichment_v0 import (  # noqa: E402
    instance_stats,
    object_qpos,
    regenerate_obs,
    set_object_qpos,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bddl",
        type=Path,
        default=ROOT / "scenarios/information_enrichment_v0/IE02_resolution_only.bddl",
    )
    parser.add_argument("--target", default="macaroni_and_cheese_1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "results/validation/native_wrist_scan")
    args = parser.parse_args()

    from libero.libero.envs import SegmentationRenderEnv

    env = SegmentationRenderEnv(
        bddl_file_name=str(args.bddl), camera_heights=256, camera_widths=256,
        initialization_noise=None,
    )
    rows: list[dict[str, float | int]] = []
    try:
        env.seed(args.seed)
        env.reset()
        original = object_qpos(env, args.target)
        best: tuple[int, np.ndarray, dict[str, float | int]] | None = None
        for x in np.linspace(-0.35, 0.45, 17):
            for y in np.linspace(-0.30, 0.30, 13):
                qpos = original.copy()
                qpos[:2] = [x, y]
                set_object_qpos(env, args.target, qpos)
                obs = regenerate_obs(env)
                stats = instance_stats(
                    env, obs, "robot0_eye_in_hand", [args.target]
                )[args.target]
                pixels = int(stats["visible_pixels"])
                bbox = stats["bbox_xyxy"]
                center_error = None
                if bbox is not None:
                    center_error = float(np.linalg.norm(
                        np.asarray([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                        - np.asarray([127.5, 127.5])
                    ))
                row = {
                    "x": float(x), "y": float(y), "visible_pixels": pixels,
                    "bbox_xyxy": bbox, "center_error_pixels": center_error,
                }
                rows.append(row)
                if best is None or pixels > best[0]:
                    best = (pixels, obs["robot0_eye_in_hand_image"].copy(), row)
        assert best is not None
        centered = min(
            (row for row in rows if int(row["visible_pixels"]) >= 500),
            key=lambda row: float(row["center_error_pixels"]),
        )
        args.output.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(args.output / "best_wrist.png", np.flipud(best[1]).astype(np.uint8))
        report = {
            "camera": "robot0_eye_in_hand",
            "camera_extrinsics": "stock_unmodified",
            "target": args.target,
            "seed": args.seed,
            "best": best[2],
            "most_centered_visible": centered,
            "visible_cells": sum(int(row["visible_pixels"]) > 0 for row in rows),
            "rows": rows,
        }
        (args.output / "scan.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
