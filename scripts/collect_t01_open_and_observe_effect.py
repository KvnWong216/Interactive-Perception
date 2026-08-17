#!/usr/bin/env python3
"""Collect paired effects for the physical ``OPEN_AND_OBSERVE`` option.

The frozen pi0.5 checkpoint opens the middle drawer.  A policy-visible OSC
controller then releases the handle and returns the end effector / wrist camera
to the stock observation pose.  Simulator joints and segmentation are used
only after actions for evaluator labels; they cannot affect execution.

Earlier files remain immutable. Version 4 uses untouched 660--699 as a clean
development extension, seed 1399 for wiring smoke, and reserves 900--999 for a
new frozen audit. The
previous 700--799 audit was opened before a final-frame-only label bug was
found, so it is debug-only and is never reused as an audit.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from collect_t01_action_effect_transitions import (  # noqa: E402
    DUMMY_ACTION,
    FINAL_PROMPT,
    MIDDLE_JOINT,
    OPEN_LIMIT,
    OPEN_PROMPT,
    digest,
    policy_visibility,
    save_packet_images,
)
from interactive_perception.action_options import execute_open_and_observe  # noqa: E402
from interactive_perception.action_outcome import (  # noqa: E402
    PI05_PATCH_EQUIVALENT_TARGET_PIXELS,
    label_temporal_information_outcome,
)
from interactive_perception.observation_option import (  # noqa: E402
    ObservationReturnConfig,
)
from interactive_perception.policy_client import (  # noqa: E402
    OpenPiWebsocketPolicy,
    build_observation,
)


def save_public_history_point(
    observation: dict,
    *,
    directory: Path,
    name: str,
    option_phase: str,
    step: int,
) -> dict:
    """Save one policy-visible temporal sample without evaluator state."""

    packet = build_observation(observation, FINAL_PROMPT)
    paths, hashes = save_packet_images(packet, directory, name)
    return {
        "name": name,
        "option_phase": option_phase,
        "step": int(step),
        "image_paths": paths,
        "image_sha256": hashes,
        "robot_state": [float(value) for value in packet.state],
        "action_role": "INFORMATION",
    }


def open_and_observe_development_split(offset: int) -> str:
    """Seed-grouped split with at least 30 conformal examples per class."""

    if offset < 20:
        return "prototype_train"
    if offset < 53:
        return "conformal_calibration"
    return "heldout_development"


def target_qpos(env) -> list[float]:
    obj = env.env.objects_dict["butter_1"]
    return np.asarray(
        env.env.sim.data.get_joint_qpos(obj.joints[0]), dtype=float
    ).tolist()


def counterfactual_target_visibility(env, qpos) -> dict[str, int]:
    """Render a target at a declared pose, then restore exact simulator state."""

    original_state = np.asarray(env.get_sim_state(), dtype=float).copy()
    obj = env.env.objects_dict["butter_1"]
    try:
        env.env.sim.data.set_joint_qpos(obj.joints[0], np.asarray(qpos, dtype=float))
        env.env.sim.forward()
        observation = env.regenerate_obs_from_state(
            np.asarray(env.get_sim_state(), dtype=float).copy()
        )
        return policy_visibility(env, observation)
    finally:
        env.regenerate_obs_from_state(original_state)
        restored_state = np.asarray(env.get_sim_state(), dtype=float)
        if not np.array_equal(restored_state, original_state):
            raise RuntimeError("counterfactual evaluator did not restore simulator state")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--extension", action="store_true")
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

    if sum((args.audit, args.smoke, args.extension)) > 1:
        raise ValueError("audit, smoke, and extension modes are mutually exclusive")
    if not args.smoke and not args.fresh_policy_server:
        raise ValueError("non-smoke collection requires --fresh-policy-server")
    expected = (
        [1399]
        if args.smoke
        else list(range(900, 1000))
        if args.audit
        else list(range(660, 700))
        if args.extension
        else list(range(600, 660))
    )
    if args.seeds is None:
        args.seeds = expected
    if args.seeds != expected:
        raise ValueError(
            f"frozen {'audit' if args.audit else 'smoke' if args.smoke else 'extension' if args.extension else 'development'} "
            f"seeds are {expected[0]}-{expected[-1]}"
        )
    if args.output is None:
        suffix = (
            "_audit"
            if args.audit
            else "_smoke"
            if args.smoke
            else "_extension"
            if args.extension
            else ""
        )
        args.output = ROOT / f"data/calibration/t01_open_and_observe_effect_v4{suffix}.jsonl"
    if args.image_dir is None:
        suffix = (
            "_audit"
            if args.audit
            else "_smoke"
            if args.smoke
            else "_extension"
            if args.extension
            else ""
        )
        args.image_dir = ROOT / f"outputs/t01_open_and_observe_effect_v4{suffix}/images"
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
                counterpart = None
                counterpart_target_qpos = {}
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
                            "EMPTY requires earlier paired target-present row"
                        )
                    counterpart_target_qpos = {
                        point["name"]: point["qpos"]
                        for point in counterpart["evaluator_only"][
                            "target_qpos_history"
                        ]
                    }
                directory = args.image_dir / regime["id"] / f"seed{seed:03d}"
                before_paths, before_hashes = save_packet_images(
                    build_observation(obs, FINAL_PROMPT), directory, "before"
                )
                public_history = [
                    {
                        "name": "before",
                        "option_phase": "BEFORE",
                        "step": -1,
                        "image_paths": before_paths,
                        "image_sha256": before_hashes,
                        "robot_state": [
                            float(value)
                            for value in build_observation(obs, FINAL_PROMPT).state
                        ],
                        "action_role": "INFORMATION",
                    }
                ]
                visibility_history = [
                    {"name": "before", "target_pixels": before_visibility}
                ]
                target_qpos_history = [
                    {"name": "before", "qpos": target_qpos(env)}
                ]
                counterfactual_visibility_history = []
                if counterpart is not None:
                    counterfactual_visibility_history.append(
                        {
                            "name": "before",
                            "target_pixels": counterfactual_target_visibility(
                                env, counterpart_target_qpos["before"]
                            ),
                        }
                    )
                minimum_joint = float(env.env.sim.data.get_joint_qpos(MIDDLE_JOINT))

                open_history_steps = {
                    max(0, round(regime["open_steps"] * fraction) - 1): index
                    for index, fraction in enumerate((0.25, 0.50, 0.75, 1.00), start=1)
                }

                def observe_step(phase, step, current_obs):
                    nonlocal minimum_joint
                    minimum_joint = min(
                        minimum_joint,
                        float(env.env.sim.data.get_joint_qpos(MIDDLE_JOINT)),
                    )
                    if phase == "OPEN" and step in open_history_steps:
                        index = open_history_steps[step]
                        public_history.append(
                            save_public_history_point(
                                current_obs,
                                directory=directory,
                                name=f"history_{index:02d}",
                                option_phase="OPEN",
                                step=step,
                            )
                        )
                        visibility_history.append(
                            {
                                "name": f"history_{index:02d}",
                                "target_pixels": policy_visibility(env, current_obs),
                            }
                        )
                        name = f"history_{index:02d}"
                        target_qpos_history.append(
                            {"name": name, "qpos": target_qpos(env)}
                        )
                        if counterpart is not None:
                            counterfactual_visibility_history.append(
                                {
                                    "name": name,
                                    "target_pixels": counterfactual_target_visibility(
                                        env, counterpart_target_qpos[name]
                                    ),
                                }
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
                public_history.append(
                    {
                        "name": "after",
                        "option_phase": "AFTER_RETURN",
                        "step": execution.return_steps,
                        "image_paths": after_paths,
                        "image_sha256": after_hashes,
                        "robot_state": [
                            float(value)
                            for value in build_observation(obs, FINAL_PROMPT).state
                        ],
                        "action_role": "INFORMATION",
                    }
                )
                visibility_history.append(
                    {"name": "after", "target_pixels": after_visibility}
                )
                target_qpos_history.append(
                    {"name": "after", "qpos": target_qpos(env)}
                )
                if counterpart is not None:
                    counterfactual_visibility_history.append(
                        {
                            "name": "after",
                            "target_pixels": counterfactual_target_visibility(
                                env, counterpart_target_qpos["after"]
                            ),
                        }
                    )
                if (
                    len(public_history) != 6
                    or len(visibility_history) != 6
                    or len(target_qpos_history) != 6
                    or (
                        counterpart is not None
                        and len(counterfactual_visibility_history) != 6
                    )
                ):
                    raise RuntimeError(
                        "expected six aligned public/evaluator history points"
                    )
                final_joint = float(env.env.sim.data.get_joint_qpos(MIDDLE_JOINT))
                opened = final_joint < OPEN_LIMIT
                empty_counterfactual_reveal_certified = None
                if regime["intended"] == "EMPTY":
                    empty_counterfactual_reveal_certified = any(
                        max(point["target_pixels"].values()) >= MIN_TARGET_PIXELS
                        for point in counterfactual_visibility_history
                    )
                outcome = label_temporal_information_outcome(
                    full_executor=bool(regime["full_executor"]),
                    opened=opened,
                    return_complete=(
                        execution.return_status.phase.value == "COMPLETE"
                    ),
                    target_pixel_history=tuple(
                        tuple(point["target_pixels"].values())
                        for point in visibility_history
                    ),
                    minimum_target_pixels=PI05_PATCH_EQUIVALENT_TARGET_PIXELS,
                    empty_coverage_certified=bool(
                        empty_counterfactual_reveal_certified
                    ),
                ).value

                row = {
                    "schema_version": "interactive-perception.open-and-observe-effect.v4",
                    "regime": regime["id"],
                    "seed": seed,
                    "split": (
                        "heldout_audit"
                        if args.audit
                        else "smoke_only"
                        if args.smoke
                        else "heldout_development_extension"
                        if args.extension
                        else open_and_observe_development_split(offset)
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
                    "public_history": public_history,
                    "critic_inputs": [
                        "six stock agentview RGB frames",
                        "six stock wrist RGB frames",
                        "six public robot-state vectors",
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
                        "middle_joint_final": final_joint,
                        "drawer_opened": opened,
                        "before_target_pixels": before_visibility,
                        "after_target_pixels": after_visibility,
                        "visibility_history": visibility_history,
                        "target_qpos_history": target_qpos_history,
                        "counterfactual_visibility_history": (
                            counterfactual_visibility_history
                        ),
                        "minimum_revealed_target_pixels": (
                            PI05_PATCH_EQUIVALENT_TARGET_PIXELS
                        ),
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
        "schema_version": "interactive-perception.open-and-observe-manifest.v4",
        "dataset": str(args.output.relative_to(ROOT)),
        "dataset_sha256": digest(args.output),
        "phase": (
            "heldout_audit"
            if args.audit
            else "smoke"
            if args.smoke
            else "heldout_development_extension"
            if args.extension
            else "development"
        ),
        "seeds": args.seeds,
        "samples": len(rows),
        "regimes": [regime["id"] for regime in regimes],
        "action": "OPEN_AND_OBSERVE",
        "frozen_executor": "pi05_libero + versioned proprioceptive return controller",
        "public_history_points_per_trial": 6,
        "outcome_label": "temporal prompt-resolvability v4",
        "minimum_resolvable_target_pixels": PI05_PATCH_EQUIVALENT_TARGET_PIXELS,
        "threshold_derivation": "(256 policy pixels / 16 visual tokens per side)^2",
        "empty_coverage_certificate": (
            "same-camera-pose counterfactual rendering of seed-matched target trajectory"
        ),
        "public_history_layout": [
            "before",
            "open 25%",
            "open 50%",
            "open 75%",
            "open 100%",
            "after return-to-observe",
        ],
        "audit_artifact": str(args.artifact.relative_to(ROOT)) if args.artifact else None,
        "audit_artifact_sha256": frozen_artifact_sha,
        "online_oracle_inputs": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
