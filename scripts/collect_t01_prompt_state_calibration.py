#!/usr/bin/env python3
"""Collect T01 counterfactual data for prompt-conditioned target-state belief.

The three conditions separate two invariances that an action-intent classifier
does not test: the same RGB scene is queried for a hidden and a visible target,
and the same final-goal prompt is queried before and after the drawer opens.
The state label is evaluator-only; the frozen policy receives stock RGB, robot
state, and prompt exactly as in the LIBERO client.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from collect_libero_intent_calibration import chunk_features  # noqa: E402
from interactive_perception.policy_client import (  # noqa: E402
    OpenPiWebsocketPolicy,
    build_observation,
)

DUMMY_ACTION = [0.0] * 6 + [-1.0]


def split_for_offset(offset: int) -> str:
    if offset < 20:
        return "prototype_train"
    if offset < 40:
        return "conformal_calibration"
    if offset < 50:
        return "heldout_validation"
    return "heldout_audit"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=220)
    parser.add_argument("--seed-count", type=int, default=50)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/calibration/t01_prompt_state_v1.jsonl",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=ROOT / "outputs/t01_prompt_state_v1/images",
    )
    args = parser.parse_args()
    if args.seed_count < 1 or args.seed_count > 60:
        raise ValueError("seed-count must lie in [1, 60]")

    from libero.libero.envs import OffScreenRenderEnv

    conditions = (
        {
            "id": "closed_hidden_butter",
            "bddl": ROOT
            / "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl",
            "prompt": "Place the butter in the basket",
            "target_state": "MANIPULATION_ONLY",
            "resolving_action": "REMOVE_OCCLUDER",
        },
        {
            "id": "closed_visible_cream_cheese",
            "bddl": ROOT
            / "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl",
            "prompt": "Place the cream cheese in the basket",
            "target_state": "OBSERVED",
            "resolving_action": "ACT",
        },
        {
            "id": "open_visible_butter",
            "bddl": ROOT
            / "scenarios/t01_stock_ladder_v1/T01E_open_drawer_retrieval.bddl",
            "prompt": "Place the butter in the basket",
            "target_state": "OBSERVED",
            "resolving_action": "ACT",
        },
    )
    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port)
    args.image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition in conditions:
        env = OffScreenRenderEnv(
            bddl_file_name=str(condition["bddl"]),
            camera_heights=256,
            camera_widths=256,
        )
        try:
            for offset in range(args.seed_count):
                seed = args.seed_start + offset
                env.seed(seed)
                obs = env.reset()
                for _ in range(args.wait_steps):
                    obs, _, _, _ = env.step(DUMMY_ACTION)
                packet = build_observation(obs, str(condition["prompt"]))
                chunks = policy.sample_chunks(packet, args.samples)
                image_paths = {}
                image_hashes = {}
                for view, image in (
                    ("agentview", packet.image),
                    ("wrist", packet.wrist_image),
                ):
                    relative = Path(str(condition["id"])) / f"seed{seed:03d}_{view}.png"
                    path = args.image_dir / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    imageio.imwrite(path, image)
                    encoded = path.read_bytes()
                    image_paths[view] = str(path.relative_to(ROOT))
                    image_hashes[view] = sha256_bytes(encoded)
                row = {
                    "schema_version": "interactive-perception.prompt-state-sample.v1",
                    "condition": condition["id"],
                    "seed": seed,
                    "split": split_for_offset(offset),
                    "prompt": condition["prompt"],
                    "target_state": condition["target_state"],
                    "resolving_action": condition["resolving_action"],
                    "bddl": str(condition["bddl"].relative_to(ROOT)),
                    "image_paths": image_paths,
                    "image_sha256": image_hashes,
                    "chunk_features": [chunk_features(chunk) for chunk in chunks],
                    "policy_inputs": [
                        "agentview RGB",
                        "wrist RGB",
                        "robot state",
                        "prompt",
                    ],
                    "controller_oracle_inputs": [],
                }
                rows.append(row)
                print(
                    f"[{len(rows):03d}/{len(conditions) * args.seed_count}] "
                    f"{condition['id']} seed={seed}",
                    flush=True,
                )
        finally:
            env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "schema_version": "interactive-perception.prompt-state-manifest.v1",
        "dataset": str(args.output.relative_to(ROOT)),
        "dataset_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "policy": "pi05_libero",
        "conditions": [item["id"] for item in conditions],
        "samples": len(rows),
        "chunks_per_observation": args.samples,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "split_rule": "offset 0:20 train, 20:40 calibration, 40:50 validation",
        "target_states": ["OBSERVED", "MANIPULATION_ONLY"],
        "counterfactuals": [
            "same closed scene, different prompt target",
            "same prompt target, closed versus open drawer",
        ],
        "policy_inputs": ["agentview RGB", "wrist RGB", "robot state", "prompt"],
        "controller_oracle_inputs": [],
        "label_source": "frozen benchmark construction; evaluator-only",
        "calibration_only": True,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
