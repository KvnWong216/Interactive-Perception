#!/usr/bin/env python3
"""Collect the preregistered, inference-only T01 prompt-state audit observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from interactive_perception.policy_client import build_observation  # noqa: E402

DUMMY_ACTION = [0.0] * 6 + [-1.0]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--extension",
        action="store_true",
        help="Collect the frozen 70-seed precision extension (310-379).",
    )
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "results/calibration/prompt_state_belief_t01_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/calibration/t01_prompt_state_v1_audit.jsonl",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=ROOT / "outputs/t01_prompt_state_v1_audit/images",
    )
    args = parser.parse_args()
    for name in ("artifact", "output", "image_dir"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    expected_seeds = list(range(310, 380)) if args.extension else list(range(280, 310))
    if args.seeds is None:
        args.seeds = expected_seeds
    if args.seeds != expected_seeds:
        phase = "extension" if args.extension else "initial audit"
        raise ValueError(f"frozen v1 {phase} seeds are {expected_seeds[0]}-{expected_seeds[-1]}")
    artifact = json.loads(args.artifact.read_text())
    expected = artifact["split_contract"]["future_audit"]
    if not args.extension and expected != "seeds 280-309; not collected when this artifact was fit":
        raise ValueError("artifact does not preregister the requested v1 audit")

    from libero.libero.envs import OffScreenRenderEnv

    conditions = (
        (
            "closed_hidden_butter",
            "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl",
            "Place the butter in the basket",
            "MANIPULATION_ONLY",
            "REMOVE_OCCLUDER",
        ),
        (
            "closed_visible_cream_cheese",
            "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl",
            "Place the cream cheese in the basket",
            "OBSERVED",
            "ACT",
        ),
        (
            "open_visible_butter",
            "scenarios/t01_stock_ladder_v1/T01E_open_drawer_retrieval.bddl",
            "Place the butter in the basket",
            "OBSERVED",
            "ACT",
        ),
    )
    args.image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition, bddl, prompt, target_state, action in conditions:
        env = OffScreenRenderEnv(
            bddl_file_name=str(ROOT / bddl),
            camera_heights=256,
            camera_widths=256,
        )
        try:
            for seed in args.seeds:
                env.seed(seed)
                obs = env.reset()
                for _ in range(args.wait_steps):
                    obs, _, _, _ = env.step(DUMMY_ACTION)
                packet = build_observation(obs, prompt)
                image_paths = {}
                image_hashes = {}
                for view, image in (
                    ("agentview", packet.image),
                    ("wrist", packet.wrist_image),
                ):
                    path = args.image_dir / condition / f"seed{seed:03d}_{view}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    imageio.imwrite(path, np.asarray(image, dtype=np.uint8))
                    image_paths[view] = str(path.relative_to(ROOT))
                    image_hashes[view] = digest(path)
                rows.append(
                    {
                        "schema_version": "interactive-perception.prompt-state-audit-sample.v1",
                        "condition": condition,
                        "seed": seed,
                        "split": "heldout_audit",
                        "prompt": prompt,
                        "target_state": target_state,
                        "resolving_action": action,
                        "bddl": bddl,
                        "image_paths": image_paths,
                        "image_sha256": image_hashes,
                        "encoder_inputs": ["agentview RGB", "wrist RGB", "prompt"],
                        "controller_oracle_inputs": [],
                    }
                )
                total = len(conditions) * len(args.seeds)
                print(f"[{len(rows):03d}/{total}] {condition} seed={seed}", flush=True)
        finally:
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    by_key = {(row["condition"], row["seed"]): row for row in rows}
    paired_rgb_equal = all(
        by_key[("closed_hidden_butter", seed)]["image_sha256"]
        == by_key[("closed_visible_cream_cheese", seed)]["image_sha256"]
        for seed in args.seeds
    )
    manifest = {
        "schema_version": "interactive-perception.prompt-state-audit-manifest.v1",
        "dataset": str(args.output.relative_to(ROOT)),
        "dataset_sha256": digest(args.output),
        "frozen_artifact": str(args.artifact.relative_to(ROOT)),
        "frozen_artifact_sha256_before_audit": digest(args.artifact),
        "seeds": args.seeds,
        "conditions": [item[0] for item in conditions],
        "samples": len(rows),
        "audit_phase": "precision_extension" if args.extension else "initial_30",
        "paired_closed_scene_rgb_hashes_equal": paired_rgb_equal,
        "encoder_inputs": ["agentview RGB", "wrist RGB", "prompt"],
        "controller_oracle_inputs": [],
        "label_source": "frozen benchmark construction; evaluator-only",
        "audit_only": True,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
