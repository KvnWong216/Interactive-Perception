#!/usr/bin/env python3
"""Collect real before/after RGB transitions for the T01 action-effect critic.

The policy receives only its stock RGB/state/prompt contract.  Instance pixels
and the drawer joint are read after execution by the evaluator to create
offline labels; they are never included in the policy packet or saved as critic
inputs.  The deliberately truncated regime balances FAILED examples for the
post-action visual recognizer and is never used to estimate full-executor
reliability.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from interactive_perception.policy_client import (  # noqa: E402
    OpenPiWebsocketPolicy,
    build_observation,
)
from interactive_perception.action_outcome import label_effect_outcome  # noqa: E402

DUMMY_ACTION = [0.0] * 6 + [-1.0]
FINAL_PROMPT = "Place the butter in the basket"
OPEN_PROMPT = "Open the middle layer of the drawer"
MIDDLE_JOINT = "wooden_cabinet_1_middle_level"
OPEN_LIMIT = -0.14
MIN_TARGET_PIXELS = 5  # Existing benchmark-wide policy-view visibility endpoint.


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def development_split(offset: int) -> str:
    if offset < 20:
        return "prototype_train"
    if offset < 30:
        return "probability_calibration"
    if offset < 40:
        return "conformal_calibration"
    return "heldout_validation"


def segmentation_key(obs: dict, camera: str) -> str:
    keys = [key for key in obs if key.startswith(camera) and "segmentation" in key]
    if len(keys) != 1:
        raise KeyError(f"expected one {camera} segmentation key, got {keys}")
    return keys[0]


def target_pixels(env, obs: dict, camera: str) -> int:
    instance_id = env.instance_to_id["butter_1"]
    segmentation = np.asarray(obs[segmentation_key(obs, camera)]).squeeze()
    return int(np.count_nonzero(segmentation == instance_id))


def policy_visibility(env, obs: dict) -> dict[str, int]:
    return {
        "agentview": target_pixels(env, obs, "agentview"),
        "wrist": target_pixels(env, obs, "robot0_eye_in_hand"),
    }


def save_packet_images(packet, directory: Path, phase: str) -> tuple[dict, dict]:
    paths = {}
    hashes = {}
    for view, image in (("agentview", packet.image), ("wrist", packet.wrist_image)):
        path = directory / f"{phase}_{view}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(path, np.asarray(image, dtype=np.uint8))
        paths[view] = str(path.relative_to(ROOT))
        hashes[view] = digest(path)
    return paths, hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run seed 400 only; output is marked ineligible for calibration.",
    )
    parser.add_argument(
        "--fresh-policy-server",
        action="store_true",
        help=(
            "Assert that this is the first infer workload after a fresh OpenPI "
            "server start (required outside smoke mode)."
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append only missing regime/seed rows after an interrupted run.",
    )
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "results/calibration/t01_action_outcome_critic_v1.json",
        help="Must already exist before the frozen audit is collected.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.audit and args.smoke:
        raise ValueError("audit and smoke modes are mutually exclusive")
    if not args.smoke and not args.fresh_policy_server:
        raise ValueError(
            "development/audit collection requires --fresh-policy-server so the "
            "OpenPI jax.random.key(0) sample sequence is reproducible"
        )
    expected = (
        [400]
        if args.smoke
        else list(range(500, 600))
        if args.audit
        else list(range(400, 460))
    )
    if args.seeds is None:
        args.seeds = expected
    if args.seeds != expected:
        raise ValueError(
            f"frozen {'audit' if args.audit else 'smoke' if args.smoke else 'development'} seeds are "
            f"{expected[0]}-{expected[-1]}"
        )
    if args.output is None:
        suffix = "_audit" if args.audit else "_smoke" if args.smoke else ""
        args.output = ROOT / f"data/calibration/t01_action_effect_v1{suffix}.jsonl"
    if args.image_dir is None:
        suffix = "_audit" if args.audit else "_smoke" if args.smoke else ""
        args.image_dir = ROOT / f"outputs/t01_action_effect_v1{suffix}/images"
    for name in ("artifact", "output", "image_dir"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    frozen_artifact_sha = None
    if args.audit:
        if not args.artifact.exists():
            raise FileNotFoundError("fit and freeze the critic artifact before audit collection")
        frozen_artifact_sha = digest(args.artifact)

    manifest_path = args.output.with_suffix(".manifest.json")
    if manifest_path.exists():
        raise FileExistsError(
            f"completed manifest already exists: {manifest_path}; artifacts are immutable"
        )
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(
                f"dataset already exists: {args.output}; use --resume only after interruption"
            )
        rows = [
            json.loads(line) for line in args.output.read_text().splitlines() if line
        ]
    else:
        if args.resume:
            raise FileNotFoundError("--resume requires an existing partial JSONL dataset")
        if args.image_dir.exists() and any(args.image_dir.rglob("*.png")):
            raise FileExistsError(
                f"image directory contains unindexed frames: {args.image_dir}"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("")
        rows = []
    completed = {(str(row["regime"]), int(row["seed"])) for row in rows}

    regimes = (
        {
            "id": "revealed_full",
            "bddl": ROOT / "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl",
            "steps": 300,
            "target_in_middle": True,
            "intended": "REVEALED",
            "full_executor": True,
        },
        {
            "id": "empty_full",
            "bddl": ROOT / "scenarios/t01_stock_ladder_v1/T01F_middle_drawer_empty_calibration.bddl",
            "steps": 300,
            "target_in_middle": False,
            "intended": "EMPTY",
            "full_executor": True,
        },
        {
            "id": "failed_truncated_control",
            "bddl": ROOT / "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl",
            "steps": 25,
            "target_in_middle": True,
            "intended": "FAILED",
            "full_executor": False,
        },
    )
    ordered_keys = [
        (str(regime["id"]), seed) for regime in regimes for seed in args.seeds
    ]
    row_keys = [(str(row["regime"]), int(row["seed"])) for row in rows]
    if row_keys != ordered_keys[: len(row_keys)]:
        raise ValueError(
            "partial dataset is not an exact prefix of the frozen call order"
        )

    from libero.libero.envs import SegmentationRenderEnv

    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port)
    args.image_dir.mkdir(parents=True, exist_ok=True)
    for regime in regimes:
        env = SegmentationRenderEnv(
            bddl_file_name=str(regime["bddl"]),
            camera_heights=256,
            camera_widths=256,
        )
        try:
            for offset, seed in enumerate(args.seeds):
                key = (str(regime["id"]), seed)
                if key in completed:
                    # A fresh OpenPI server resets its PRNG to key(0). Replay
                    # the exact number of skipped infer calls so the first new
                    # row receives the same key it would have in an uninterrupted
                    # run. Inputs do not affect the server-side key split count.
                    env.seed(seed)
                    replay_obs = env.reset()
                    for _ in range(args.wait_steps):
                        replay_obs, _, _, _ = env.step(DUMMY_ACTION)
                    replay_packet = build_observation(replay_obs, OPEN_PROMPT)
                    replay_calls = math.ceil(
                        int(regime["steps"]) / args.replan_steps
                    )
                    for _ in range(replay_calls):
                        policy.sample_chunks(replay_packet, 1)
                    print(
                        f"[resume] {regime['id']} seed={seed} "
                        f"replayed_infer_calls={replay_calls}",
                        flush=True,
                    )
                    continue
                env.seed(seed)
                obs = env.reset()
                for _ in range(args.wait_steps):
                    obs, _, _, _ = env.step(DUMMY_ACTION)
                before_visibility = policy_visibility(env, obs)
                if any(before_visibility.values()):
                    raise RuntimeError(
                        f"seed {seed} violates hidden-target precondition: {before_visibility}"
                    )
                before_packet = build_observation(obs, FINAL_PROMPT)
                directory = args.image_dir / str(regime["id"]) / f"seed{seed:03d}"
                before_paths, before_hashes = save_packet_images(
                    before_packet, directory, "before"
                )

                plan: collections.deque[np.ndarray] = collections.deque()
                minimum_joint = float(env.env.sim.data.get_joint_qpos(MIDDLE_JOINT))
                for _ in range(int(regime["steps"])):
                    if not plan:
                        chunk = policy.sample_chunks(
                            build_observation(obs, OPEN_PROMPT), 1
                        )[0]
                        plan.extend(chunk[: args.replan_steps])
                    obs, _, _, _ = env.step(plan.popleft().tolist())
                    minimum_joint = min(
                        minimum_joint,
                        float(env.env.sim.data.get_joint_qpos(MIDDLE_JOINT)),
                    )

                after_packet = build_observation(obs, FINAL_PROMPT)
                after_paths, after_hashes = save_packet_images(
                    after_packet, directory, "after"
                )
                after_visibility = policy_visibility(env, obs)
                opened = minimum_joint < OPEN_LIMIT
                outcome = label_effect_outcome(
                    opened=opened,
                    target_in_resolved_location=bool(regime["target_in_middle"]),
                    target_pixels=tuple(after_visibility.values()),
                    minimum_target_pixels=MIN_TARGET_PIXELS,
                ).value
                empty_counterfactual_reveal_certified = None
                if regime["intended"] == "EMPTY":
                    counterpart = next(
                        (
                            row
                            for row in rows
                            if row["regime"] == "revealed_full"
                            and int(row["seed"]) == seed
                        ),
                        None,
                    )
                    if counterpart is None:
                        raise RuntimeError(
                            "EMPTY requires its earlier seed-matched target-present counterpart"
                        )
                    empty_counterfactual_reveal_certified = (
                        counterpart["outcome"] == "REVEALED"
                    )
                    if outcome == "EMPTY" and not empty_counterfactual_reveal_certified:
                        outcome = "FAILED"
                row = {
                    "schema_version": "interactive-perception.action-effect-transition.v1",
                    "regime": regime["id"],
                    "seed": seed,
                    "split": (
                        "heldout_audit"
                        if args.audit
                        else "smoke_only"
                        if args.smoke
                        else development_split(offset)
                    ),
                    "context": "t01_stock_middle_drawer_search",
                    "final_prompt": FINAL_PROMPT,
                    "action": "REMOVE_OCCLUDER",
                    "executor_prompt": OPEN_PROMPT,
                    "executor_steps": regime["steps"],
                    "full_executor": regime["full_executor"],
                    "intended_outcome": regime["intended"],
                    "outcome": outcome,
                    "bddl": str(regime["bddl"].relative_to(ROOT)),
                    "image_paths": {"before": before_paths, "after": after_paths},
                    "image_sha256": {"before": before_hashes, "after": after_hashes},
                    "critic_inputs": [
                        "before agentview RGB",
                        "before wrist RGB",
                        "after agentview RGB",
                        "after wrist RGB",
                        "final prompt",
                        "action label",
                    ],
                    "online_oracle_inputs": [],
                    "evaluator_only": {
                        "middle_joint_minimum": minimum_joint,
                        "drawer_opened": opened,
                        "before_target_pixels": before_visibility,
                        "after_target_pixels": after_visibility,
                        "minimum_revealed_target_pixels": MIN_TARGET_PIXELS,
                        "frozen_target_in_middle": regime["target_in_middle"],
                        "empty_counterfactual_reveal_certified": (
                            empty_counterfactual_reveal_certified
                        ),
                    },
                }
                rows.append(row)
                with args.output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                total = len(regimes) * len(args.seeds)
                print(
                    f"[{len(rows):03d}/{total}] {regime['id']} seed={seed} "
                    f"outcome={outcome} open={opened} pixels={after_visibility}",
                    flush=True,
                )
        finally:
            env.close()

    expected_rows = len(regimes) * len(args.seeds)
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows, found {len(rows)}")
    manifest = {
        "schema_version": "interactive-perception.action-effect-manifest.v1",
        "dataset": str(args.output.relative_to(ROOT)),
        "dataset_sha256": digest(args.output),
        "phase": "heldout_audit" if args.audit else "smoke" if args.smoke else "development",
        "seeds": args.seeds,
        "samples": len(rows),
        "regimes": [regime["id"] for regime in regimes],
        "policy": "pi05_libero",
        "policy_rng_contract": {
            "initialization": "openpi Policy default jax.random.key(0)",
            "infer_calls_before_collection": 0,
            "fresh_policy_server_required": True,
            "sample_order": "regime order, then ascending seed, then online replan order",
            "resume_contract": "fresh server plus exact skipped-infer-call replay",
        },
        "policy_inputs": ["stock agentview RGB", "stock wrist RGB", "robot state", "prompt"],
        "critic_inputs": ["paired stock policy RGB", "final prompt", "action label"],
        "online_oracle_inputs": [],
        "label_source": "evaluator-only segmentation, drawer joint, and frozen scene construction",
        "empty_label_contract": (
            "EMPTY additionally requires the seed-matched target-present "
            "counterpart to reach REVEALED; otherwise OPENED_UNOBSERVED is FAILED"
        ),
        "minimum_revealed_target_pixels": MIN_TARGET_PIXELS,
        "truncated_control_authorizes_executor": False,
        "calibration_only": True,
        "eligible_for_calibration": not args.smoke,
        "frozen_artifact": (
            str(args.artifact.relative_to(ROOT)) if args.audit else None
        ),
        "frozen_artifact_sha256_before_audit": frozen_artifact_sha,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
