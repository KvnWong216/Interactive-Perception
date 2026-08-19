#!/usr/bin/env python3
"""Disposable evaluator-only visibility smoke for candidate PIU scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from interactive_perception.anchors import visible_pixels  # noqa: E402

DUMMY_ACTION = np.zeros(7, dtype=np.float64)
DUMMY_ACTION[-1] = -1.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl", type=Path, required=True)
    parser.add_argument("--target-instance", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1498, 1499])
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("bddl", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output.exists():
        raise FileExistsError(f"immutable smoke output exists: {args.output}")

    from libero.libero.envs import SegmentationRenderEnv
    env = SegmentationRenderEnv(
        bddl_file_name=str(args.bddl), camera_heights=256, camera_widths=256
    )
    rows = []
    try:
        for seed in args.seeds:
            env.seed(seed)
            observation = env.reset()
            for _ in range(args.wait_steps):
                observation, _, _, _ = env.step(DUMMY_ACTION)
            pixels = {
                camera: visible_pixels(
                    env,
                    observation,
                    camera=camera,
                    instance=args.target_instance,
                )
                for camera in ("agentview", "robot0_eye_in_hand")
            }
            rows.append(
                {
                    "seed": seed,
                    "target_pixels": pixels,
                    "prompt_resolvable": max(pixels.values()) >= 256,
                }
            )
    finally:
        env.close()
    report = {
        "schema_version": "interaction-uncertainty.visibility-contract-smoke.v1",
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "claim_status": "disposable evaluator-only scenario preflight",
        "bddl": str(args.bddl.relative_to(ROOT)),
        "bddl_sha256": digest(args.bddl),
        "target_instance": args.target_instance,
        "minimum_resolvable_pixels": 256,
        "rows": rows,
        "all_resolvable": all(row["prompt_resolvable"] for row in rows),
        "none_resolvable": not any(row["prompt_resolvable"] for row in rows),
        "policy_inputs": [],
        "evaluator_only": True,
        "paper_eligible": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
