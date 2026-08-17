#!/usr/bin/env python3
"""Collect paired effects for the physical ``OPEN_AND_OBSERVE`` option.

The frozen pi0.5 checkpoint opens the middle drawer.  A policy-visible OSC
controller then releases the handle and returns the end effector / wrist camera
to the stock observation pose.  Simulator joints and segmentation are used
only after actions for evaluator labels; they cannot affect execution.

Version-1 action-effect seeds 500--599 remain sealed.  This new executor uses
600--659 for development and reserves 700--799 for a future frozen audit.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from collect_t01_action_effect_transitions import (  # noqa: E402
    DUMMY_ACTION,
    FINAL_PROMPT,
    MIN_TARGET_PIXELS,
    MIDDLE_JOINT,
    OPEN_LIMIT,
    OPEN_PROMPT,
    development_split,
    digest,
    policy_visibility,
    save_packet_images,
)
from interactive_perception.action_options import execute_open_and_observe  # noqa: E402
from interactive_perception.action_outcome import label_effect_outcome  # noqa: E402
from interactive_perception.observation_option import (  # noqa: E402
    ObservationReturnConfig,
)
from interactive_perception.policy_client import (  # noqa: E402
    OpenPiWebsocketPolicy,
    build_observation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fresh-policy-server", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.audit and args.smoke:
        raise ValueError("audit and smoke modes are mutually exclusive")
    if not args.smoke and not args.fresh_policy_server:
        raise ValueError("non-smoke collection requires --fresh-policy-server")
    expected = (
        [600]
        if args.smoke
        else list(range(700, 800))
        if args.audit
        else list(range(600, 660))
    )
    if args.seeds is None:
        args.seeds = expected
    if args.seeds != expected:
        raise ValueError(
            f"frozen {'audit' if args.audit else 'smoke' if args.smoke else 'development'} "
            f"seeds are {expected[0]}-{expected[-1]}"
        )
    if args.output is None:
        suffix = "_audit" if args.audit else "_smoke" if args.smoke else ""
        args.output = ROOT / f"data/calibration/t01_open_and_observe_effect_v1{suffix}.jsonl"
    if args.image_dir is None:
        suffix = "_audit" if args.audit else "_smoke" if args.smoke else ""
        args.image_dir = ROOT / f"outputs/t01_open_and_observe_effect_v1{suffix}/images"
    for name in ("output", "image_dir", "artifact"):
        value = getattr(args, name)
        if value is not None and not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.audit and (args.artifact is None or not args.artifact.exists()):
        raise FileNotFoundError("audit requires a frozen development artifact")

    frozen_artifact_sha = digest(args.artifact) if args.audit else None
    manifest_path = args.output.with_suffix(".manifest.json")
    if manifest_path.exists():
        raise FileExistsError(f"completed manifest is immutable: {manifest_path}")
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"dataset already exists: {args.output}")
        rows = [json.loads(line) for line in args.output.read_text().splitlines() if line]
    else:
        if args.resume:
            raise FileNotFoundError("--resume requires a partial dataset")
        if args.image_dir.exists() and any(args.image_dir.rglob("*.png")):
            raise FileExistsError(f"unindexed images exist in {args.image_dir}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("")
        rows = []

    regimes = (
        {
            "id": "revealed_full",
            "bddl": ROOT / "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl",
            "open_steps": 300,
            "target_in_middle": True,
            "intended": "REVEALED",
            "full_executor": True,
        },
        {
            "id": "empty_full",
            "bddl": ROOT / "scenarios/t01_stock_ladder_v1/T01F_middle_drawer_empty_calibration.bddl",
            "open_steps": 300,
            "target_in_middle": False,
            "intended": "EMPTY",
            "full_executor": True,
        },
        {
            "id": "failed_truncated_control",
            "bddl": ROOT / "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl",
            "open_steps": 25,
            "target_in_middle": True,
            "intended": "FAILED",
            "full_executor": False,
        },
    )
    ordered_keys = [
        (regime["id"], seed) for regime in regimes for seed in args.seeds
    ]
    row_keys = [(row["regime"], int(row["seed"])) for row in rows]
    if row_keys != ordered_keys[: len(row_keys)]:
        raise ValueError("partial dataset is not an exact prefix of frozen call order")
    completed = set(row_keys)

    from libero.libero.envs import SegmentationRenderEnv

    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port)
    return_config = ObservationReturnConfig()
    args.image_dir.mkdir(parents=True, exist_ok=True)
    for regime in regimes:
        env = SegmentationRenderEnv(
            bddl_file_name=str(regime["bddl"]),
            camera_heights=256,
            camera_widths=256,
        )
        try:
            for offset, seed in enumerate(args.seeds):
                key = (regime["id"], seed)
                env.seed(seed)
                obs = env.reset()
                for _ in range(args.wait_steps):
                    obs, _, _, _ = env.step(DUMMY_ACTION)
                if key in completed:
                    packet = build_observation(obs, OPEN_PROMPT)
                    replay_calls = math.ceil(regime["open_steps"] / args.replan_steps)
                    for _ in range(replay_calls):
                        policy.sample_chunks(packet, 1)
                    print(f"[resume] {key} replayed_calls={replay_calls}", flush=True)
                    continue

                before_visibility = policy_visibility(env, obs)
                if any(before_visibility.values()):
                    raise RuntimeError(
                        f"seed {seed} violates hidden-target precondition: {before_visibility}"
                    )
                directory = args.image_dir / regime["id"] / f"seed{seed:03d}"
                before_paths, before_hashes = save_packet_images(
                    build_observation(obs, FINAL_PROMPT), directory, "before"
                )
                minimum_joint = float(env.env.sim.data.get_joint_qpos(MIDDLE_JOINT))

                def observe_step(phase, step, current_obs):
                    nonlocal minimum_joint
                    minimum_joint = min(
                        minimum_joint,
                        float(env.env.sim.data.get_joint_qpos(MIDDLE_JOINT)),
                    )

                obs, execution = execute_open_and_observe(
                    env=env,
                    initial_observation=obs,
                    policy=policy,
                    open_prompt=OPEN_PROMPT,
                    open_steps=regime["open_steps"],
                    replan_steps=args.replan_steps,
                    return_config=return_config,
                    step_observer=observe_step,
                )
                after_visibility = policy_visibility(env, obs)
                after_paths, after_hashes = save_packet_images(
                    build_observation(obs, FINAL_PROMPT), directory, "after"
                )
                opened = minimum_joint < OPEN_LIMIT
                outcome = label_effect_outcome(
                    opened=opened,
                    target_in_resolved_location=regime["target_in_middle"],
                    target_pixels=tuple(after_visibility.values()),
                    minimum_target_pixels=MIN_TARGET_PIXELS,
                ).value
                if not execution.executor_completed:
                    outcome = "FAILED"

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
                        raise RuntimeError("EMPTY requires earlier paired target-present row")
                    empty_counterfactual_reveal_certified = (
                        counterpart["outcome"] == "REVEALED"
                    )
                    if outcome == "EMPTY" and not empty_counterfactual_reveal_certified:
                        outcome = "FAILED"

                row = {
                    "schema_version": "interactive-perception.open-and-observe-effect.v1",
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
                    "action": "OPEN_AND_OBSERVE",
                    "executor_prompt": OPEN_PROMPT,
                    "executor_components": [
                        "frozen pi05_libero OPEN_CONTAINER",
                        "proprioceptive OSC RETURN_TO_OBSERVE",
                    ],
                    "open_steps": regime["open_steps"],
                    "return_steps": execution.return_steps,
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
                    "online_executor_inputs": [
                        "stock RGB/state/prompt for pi0.5 opening",
                        "robot0_eef_pos",
                        "robot0_eef_quat",
                        "robot0_gripper_qpos",
                    ],
                    "online_oracle_inputs": [],
                    "return_config": dataclasses.asdict(return_config),
                    "return_status": {
                        **dataclasses.asdict(execution.return_status),
                        "phase": execution.return_status.phase.value,
                    },
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
                    f"outcome={outcome} return={execution.return_status.phase.value} "
                    f"pixels={after_visibility}",
                    flush=True,
                )
        finally:
            env.close()

    if len(rows) != len(regimes) * len(args.seeds):
        raise RuntimeError("incomplete dataset")
    manifest = {
        "schema_version": "interactive-perception.open-and-observe-manifest.v1",
        "dataset": str(args.output.relative_to(ROOT)),
        "dataset_sha256": digest(args.output),
        "phase": "heldout_audit" if args.audit else "smoke" if args.smoke else "development",
        "seeds": args.seeds,
        "samples": len(rows),
        "regimes": [regime["id"] for regime in regimes],
        "action": "OPEN_AND_OBSERVE",
        "frozen_executor": "pi05_libero + versioned proprioceptive return controller",
        "audit_artifact": str(args.artifact.relative_to(ROOT)) if args.artifact else None,
        "audit_artifact_sha256": frozen_artifact_sha,
        "online_oracle_inputs": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
