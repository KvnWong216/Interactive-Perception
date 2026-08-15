#!/usr/bin/env python3
"""Collect a frozen, oracle-free LIBERO action-intent calibration dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import _bootstrap  # noqa: F401,E402

from interactive_perception.policy_client import OpenPiWebsocketPolicy, build_observation  # noqa: E402

DUMMY_ACTION = [0.0] * 6 + [-1.0]
SPLIT_BY_SEED = {
    **{seed: "prototype_train" for seed in range(20)},
    **{seed: "conformal_calibration" for seed in range(20, 40)},
    **{seed: "heldout_validation" for seed in range(40, 50)},
    **{seed: "heldout_audit" for seed in range(50, 60)},
}


def chunk_features(chunk: np.ndarray) -> list[float]:
    chunk = np.asarray(chunk, dtype=np.float64)[:, :7]
    delta = chunk[:, :6]
    return np.concatenate(
        [
            delta.mean(axis=0),
            delta.std(axis=0),
            delta.sum(axis=0),
            np.asarray(
                [
                    np.linalg.norm(delta[:, :3], axis=1).mean(),
                    np.linalg.norm(delta[:, 3:6], axis=1).mean(),
                    chunk[:, 6].mean(),
                    chunk[:, 6].std(),
                ]
            ),
        ]
    ).tolist()


def collect_condition(
    *,
    env,
    policy,
    prompt,
    label,
    seed,
    samples,
    wait_steps,
    primary_camera="agentview",
    wrist_camera="robot0_eye_in_hand",
    reset_sensing_pose=None,
    wrist_camera_initialization=None,
):
    env.seed(seed)
    obs = env.reset()
    if reset_sensing_pose:
        action = np.asarray(reset_sensing_pose["action"], dtype=float)
        for _ in range(int(reset_sensing_pose["action_steps"])):
            obs, _, _, _ = env.step(action)
        neutral = np.zeros(7, dtype=float)
        for _ in range(int(reset_sensing_pose["settle_steps"])):
            obs, _, _, _ = env.step(neutral)
    if wrist_camera_initialization:
        from interactive_perception.camera_views import initialize_attached_camera_look_at

        initialize_attached_camera_look_at(
            env,
            camera=str(wrist_camera_initialization["camera"]),
            target=wrist_camera_initialization["look_at"],
        )
        obs = env.regenerate_obs_from_state(env.get_sim_state())
    for _ in range(wait_steps):
        obs, _, _, _ = env.step(DUMMY_ACTION)
    chunks = policy.sample_chunks(
        build_observation(
            obs,
            prompt,
            primary_camera=primary_camera,
            wrist_camera=wrist_camera,
        ),
        samples,
    )
    return {
        "schema_version": "interactive-perception.intent-calibration-sample.v1",
        "condition": label.lower(),
        "true_intent": label,
        "prompt": prompt,
        "seed": seed,
        "split": SPLIT_BY_SEED[seed],
        "chunk_features": [chunk_features(chunk) for chunk in chunks],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--samples-per-observation", type=int, default=8)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/calibration/libero_intents_v1.jsonl"
    )
    args = parser.parse_args()

    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port)
    bddl_root = Path(get_libero_path("bddl_files")) / "libero_goal"
    conditions = [
        (
            "ACT",
            "Pick up the cream cheese and put it in the bowl",
            bddl_root / "put_the_cream_cheese_in_the_bowl.bddl",
        ),
        (
            "REMOVE_OCCLUDER",
            "Open the middle layer of the drawer",
            bddl_root / "open_the_middle_drawer_of_the_cabinet.bddl",
        ),
    ]
    rows = []
    for label, prompt, bddl in conditions:
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
        )
        try:
            for seed in range(50):
                row = collect_condition(
                    env=env,
                    policy=policy,
                    prompt=prompt,
                    label=label,
                    seed=seed,
                    samples=args.samples_per_observation,
                    wait_steps=args.wait_steps,
                )
                row["bddl"] = bddl.name
                rows.append(row)
                print(f"[{len(rows):03d}/100] {label} seed={seed}", flush=True)
        finally:
            env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "schema_version": "interactive-perception.intent-calibration-manifest.v1",
        "dataset": str(args.output),
        "dataset_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "policy": "pi05_libero",
        "openpi_commit": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
        "checkpoint_metadata_sha256": "303a4e354814928e1d29b75e310f2c1ac7e7e29b62f48395b631045ca1cffc73",
        "server_metadata": policy.server_metadata,
        "samples": len(rows),
        "chunks_per_sample": args.samples_per_observation,
        "labels": ["ACT", "REMOVE_OCCLUDER"],
        "split_rule": "seed 0:20 train, 20:40 calibration, 40:50 validation, per class",
        "oracle_inputs": [],
        "policy_inputs": ["agentview RGB", "wrist RGB", "robot state", "prompt"],
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
