#!/usr/bin/env python3
"""Collect one public-only initial observation for the PIU mechanism runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from interaction_uncertainty.task_parser import FrozenRetrievalTaskParser  # noqa: E402
from interactive_perception.policy_client import build_observation  # noqa: E402


DUMMY_ACTION = [0.0] * 6 + [-1.0]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bddl",
        type=Path,
        default=ROOT / "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl",
    )
    parser.add_argument("--prompt", default="Place the butter in the basket")
    parser.add_argument("--seed", type=int, default=1399)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--sample-id", default="t01_mechanism_hidden_butter_seed1399")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/piu_mechanism/t01_seed1399_snapshot.jsonl",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=ROOT / "outputs/piu_mechanism/t01_seed1399/images",
    )
    args = parser.parse_args()
    for name in ("bddl", "output", "image_dir"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    manifest = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest.exists():
        raise FileExistsError("mechanism snapshot output is immutable")
    if args.wait_steps < 0:
        raise ValueError("wait-steps must be non-negative")

    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(args.bddl), camera_heights=256, camera_widths=256
    )
    try:
        env.seed(args.seed)
        observation = env.reset()
        for _ in range(args.wait_steps):
            observation, _, _, _ = env.step(DUMMY_ACTION)
        packet = build_observation(observation, args.prompt)
    finally:
        env.close()

    task = FrozenRetrievalTaskParser().parse(args.prompt)
    args.image_dir.mkdir(parents=True, exist_ok=True)
    image_paths = {}
    image_hashes = {}
    for view, image in (("agentview", packet.image), ("wrist", packet.wrist_image)):
        path = args.image_dir / f"{view}.png"
        imageio.imwrite(path, image)
        image_paths[view] = str(path.relative_to(ROOT))
        image_hashes[view] = digest(path)
    row = {
        "schema_version": "interaction-uncertainty.piu-mechanism-snapshot.v1",
        "sample_id": args.sample_id,
        "sample_type": "INITIAL_TASK_BELIEF",
        "split": "disposable_mechanism_wiring",
        "seed": args.seed,
        "scenario_id": "t01_hidden_butter",
        "scenario_path": str(args.bddl.relative_to(ROOT)),
        "prompt": args.prompt,
        "target": task.target,
        "destination": task.destination,
        "visual_queries": [task.target, task.destination],
        "candidate_actions": ["DIRECT_ACT", "OPEN_TO_INSPECT", "ABSTAIN"],
        "policy_inputs": {
            "image_paths": image_paths,
            "image_sha256": image_hashes,
            "public_robot_state": [float(value) for value in packet.state],
        },
        "online_oracle_inputs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, separators=(",", ":")) + "\n")
    report = {
        "schema_version": "interaction-uncertainty.piu-mechanism-snapshot-manifest.v1",
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "snapshot": {"path": str(args.output.relative_to(ROOT)), "sha256": digest(args.output)},
        "seed": args.seed,
        "scenario": str(args.bddl.relative_to(ROOT)),
        "prompt": args.prompt,
        "policy_inputs": ["agentview RGB", "wrist RGB", "public robot state", "full prompt"],
        "online_oracle_inputs": [],
        "claim_status": "disposable mechanism wiring; not paper evaluation",
    }
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
