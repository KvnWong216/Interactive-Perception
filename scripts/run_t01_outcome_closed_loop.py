#!/usr/bin/env python3
"""Run the oracle-free T01 observability loop with the frozen RGB critic.

This is a target-observability experiment, not final butter retrieval. Routing
and stopping use only the two stock policy RGB streams, prompt, frozen
artifacts, and public action history. Segmentation and the drawer joint are read
once after control terminates for evaluator scoring.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from interactive_perception.action_effect import EffectRegistry  # noqa: E402
from interactive_perception.action_outcome import ActionOutcomePredictor  # noqa: E402
from interactive_perception.active_risk import (  # noqa: E402
    DecisionLosses,
    EffectOutcome,
    ExpectedRiskPlanner,
    TargetBelief,
    TargetHypothesis,
    TargetState,
)
from interactive_perception.policy_client import (  # noqa: E402
    OpenPiWebsocketPolicy,
    build_observation,
)
from interactive_perception.prompt_state import PromptStatePredictor  # noqa: E402

DUMMY_ACTION = [0.0] * 6 + [-1.0]
FINAL_PROMPT = "Place the butter in the basket"
OPEN_PROMPT = "Open the middle layer of the drawer"
REMOVE = "REMOVE_OCCLUDER"
MIDDLE_JOINT = "wooden_cabinet_1_middle_level"
OPEN_LIMIT = -0.14
MIN_TARGET_PIXELS = 5


def capture_biview_demo(
    env,
    obs: dict,
    *,
    demo: dict,
    status: str,
) -> np.ndarray:
    """Render wrist-left/global-right presentation frames without policy leakage."""

    from PIL import Image, ImageDraw

    left = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    camera = str(demo["demo_camera"])
    pose = demo["demo_camera_pose"]
    sim = env.env.sim
    camera_id = sim.model.camera_name2id(camera)
    original_pos = np.asarray(sim.model.cam_pos[camera_id], dtype=float).copy()
    original_quat = np.asarray(sim.model.cam_quat[camera_id], dtype=float).copy()
    try:
        sim.model.cam_pos[camera_id] = np.asarray(pose["position"], dtype=float)
        sim.model.cam_quat[camera_id] = np.asarray(
            pose["quaternion_wxyz"], dtype=float
        )
        sim.forward()
        global_obs = env.regenerate_obs_from_state(env.get_sim_state())
        right = np.ascontiguousarray(global_obs[f"{camera}_image"][::-1, ::-1])
    finally:
        sim.model.cam_pos[camera_id] = original_pos
        sim.model.cam_quat[camera_id] = original_quat
        sim.forward()
    panels = []
    labels = ("FIRST-PERSON / WRIST", "THIRD-PERSON / GLOBAL")
    for frame, label in zip((left, right), labels, strict=True):
        panel = Image.fromarray(frame)
        draw = ImageDraw.Draw(panel)
        box_height = max(38, panel.height // 7)
        draw.rectangle((0, 0, panel.width, box_height), fill=(0, 0, 0))
        draw.text((8, 4), label, fill=(255, 255, 255))
        draw.text((8, 20), status, fill=(255, 230, 80))
        panels.append(np.asarray(panel))
    return np.ascontiguousarray(np.concatenate(panels, axis=1))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def segmentation_key(obs: dict, camera: str) -> str:
    keys = [key for key in obs if key.startswith(camera) and "segmentation" in key]
    if len(keys) != 1:
        raise KeyError(f"expected one {camera} segmentation key, got {keys}")
    return keys[0]


def evaluator_pixels(env, obs: dict) -> dict[str, int]:
    instance_id = env.instance_to_id["butter_1"]
    values = {}
    for label, camera in (
        ("agentview", "agentview"),
        ("wrist", "robot0_eye_in_hand"),
    ):
        segmentation = np.asarray(obs[segmentation_key(obs, camera)]).squeeze()
        values[label] = int(np.count_nonzero(segmentation == instance_id))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(600, 700)))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--executor-steps", type=int, default=300)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--video-seeds", type=int, nargs="+", default=[600])
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=ROOT / "results/demos/t01_outcome_closed_loop_v1",
    )
    parser.add_argument(
        "--risk-contract",
        type=Path,
        default=ROOT / "results/calibration/t01_observability_loss_contract_v1.json",
    )
    parser.add_argument(
        "--belief-artifact",
        type=Path,
        default=ROOT / "results/calibration/prompt_state_belief_t01_v1.json",
    )
    parser.add_argument(
        "--effect-artifact",
        type=Path,
        default=ROOT / "results/calibration/t01_action_effect_v2.json",
    )
    parser.add_argument(
        "--outcome-artifact",
        type=Path,
        default=ROOT / "results/calibration/t01_action_outcome_critic_v1.json",
    )
    parser.add_argument(
        "--outcome-audit",
        type=Path,
        default=ROOT / "results/calibration/t01_action_outcome_critic_audit_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/capability/t01_outcome_closed_loop_100seed_v1.json",
    )
    args = parser.parse_args()
    if args.seeds != list(range(600, 700)):
        raise ValueError("frozen v1 closed-loop seeds are exactly 600-699")
    for name in (
        "risk_contract",
        "belief_artifact",
        "effect_artifact",
        "outcome_artifact",
        "outcome_audit",
        "output",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if not args.video_dir.is_absolute():
        args.video_dir = ROOT / args.video_dir
    unknown_video_seeds = set(args.video_seeds) - set(args.seeds)
    if unknown_video_seeds:
        raise ValueError(f"video seeds are not rollout seeds: {sorted(unknown_video_seeds)}")

    outcome_audit = json.loads(args.outcome_audit.read_text())
    if not outcome_audit["fp3_passed"]:
        raise ValueError("the frozen outcome audit must pass before closed-loop execution")
    risk_contract = json.loads(args.risk_contract.read_text())
    information_cost = float(risk_contract["information_cost"])
    losses = risk_contract["losses"]
    belief_predictor = PromptStatePredictor.from_artifact(
        json.loads(args.belief_artifact.read_text())
    )
    outcome_predictor = ActionOutcomePredictor.from_artifact(
        json.loads(args.outcome_artifact.read_text())
    )
    effect_entry = EffectRegistry.from_dict(
        json.loads(args.effect_artifact.read_text())
    ).get("t01_stock_middle_drawer_hidden_butter", REMOVE)
    effect = effect_entry.planner_effect(
        resolves=("MANIPULATION_ONLY",),
        cost=information_cost,
        desired_outcome=EffectOutcome.REVEALED,
    )
    planner = ExpectedRiskPlanner(
        losses=DecisionLosses(
            false_commit=float(losses["false_commit"]),
            false_absent=float(losses["false_absent"]),
            act_execution_failure=float(losses["act_execution_failure"]),
        ),
        act_reliability=1.0,
        horizon=1,
    )
    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port)
    metadata = policy.server_metadata
    if metadata.get("extension_schema") != "interactive-perception.pi05-prefix-action-server.v1":
        raise ValueError("run the versioned pi0.5 action + prefix server")
    if metadata.get("prefix_requests_advance_action_rng") is not False:
        raise ValueError("prefix requests must not advance the action PRNG")

    from libero.libero.envs import SegmentationRenderEnv

    benchmark = yaml.safe_load(
        (ROOT / "benchmarks/interactive_manipulation_v0/benchmark.yaml").read_text()
    )
    demo = {
        "demo_camera": benchmark["backend"]["demo_camera"],
        "demo_camera_pose": benchmark["backend"]["demo_camera_pose"],
    }
    bddl = ROOT / "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl"
    rows = []
    for seed in args.seeds:
        env = SegmentationRenderEnv(
            bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
        )
        controller_joint_reads = 0
        writer = None
        video_path = None
        try:
            env.seed(seed)
            obs = env.reset()
            for _ in range(10):
                obs, _, _, _ = env.step(DUMMY_ACTION)
            initial_packet = build_observation(obs, FINAL_PROMPT)
            before_prefix = policy.encode_prefix(initial_packet)
            initial = belief_predictor.predict(before_prefix)
            belief = TargetBelief(
                hypotheses=(
                    TargetHypothesis("OBSERVED", TargetState.OBSERVED),
                    TargetHypothesis(
                        "MANIPULATION_ONLY", TargetState.MANIPULATION_ONLY, REMOVE
                    ),
                ),
                probabilities=(
                    float(initial.probabilities["OBSERVED"]),
                    float(initial.probabilities["MANIPULATION_ONLY"]),
                ),
            )
            decision = planner.robust_plan(
                belief,
                (effect,),
                conformal_labels=initial.prediction_set,
            )
            if seed in args.video_seeds:
                args.video_dir.mkdir(parents=True, exist_ok=True)
                video_path = args.video_dir / f"t01_outcome_closed_loop_seed{seed:03d}.mp4"
                writer = imageio.get_writer(video_path, fps=20)
                uncertainty = decision.resolvable_uncertainty.get(REMOVE, float("nan"))
                for _ in range(20):
                    writer.append_data(
                        capture_biview_demo(
                            env,
                            obs,
                            demo=demo,
                            status=f"ROUTE {decision.selected_action} | U_res {uncertainty:+.3f}",
                        )
                    )
            history = []
            terminal = decision.selected_action
            outcome_prediction = None
            executed_steps = 0
            if decision.selected_action == REMOVE:
                plan: collections.deque[np.ndarray] = collections.deque()
                for step in range(args.executor_steps):
                    if writer is not None and step % 2 == 0:
                        writer.append_data(
                            capture_biview_demo(
                                env,
                                obs,
                                demo=demo,
                                status="ACT OPEN MIDDLE LAYER",
                            )
                        )
                    if not plan:
                        chunk = policy.sample_chunks(
                            build_observation(obs, OPEN_PROMPT), 1
                        )[0]
                        plan.extend(chunk[: args.replan_steps])
                    obs, _, _, _ = env.step(plan.popleft().tolist())
                    executed_steps = step + 1
                    if executed_steps % args.replan_steps != 0:
                        continue
                    after_prefix = policy.encode_prefix(
                        build_observation(obs, FINAL_PROMPT)
                    )
                    outcome_prediction = outcome_predictor.predict(
                        before_prefix, after_prefix
                    )
                    history.append(
                        {
                            "step": executed_steps,
                            "action": REMOVE,
                            "outcome_set": list(outcome_prediction.prediction_set),
                            "outcome_evidence": outcome_prediction.evidence,
                        }
                    )
                    if outcome_prediction.prediction_set == (
                        EffectOutcome.REVEALED.value,
                    ):
                        terminal = "STOP_REVEALED"
                        break
                    if outcome_prediction.prediction_set == (
                        EffectOutcome.EMPTY.value,
                    ):
                        terminal = "NOT_FOUND"
                        break
                else:
                    if outcome_prediction is None:
                        terminal = "SAFE_STOP_NO_EFFECT_OBSERVATION"
                    elif outcome_prediction.prediction_set == (
                        EffectOutcome.FAILED.value,
                    ):
                        terminal = "SAFE_STOP_FAILED_EFFECT"
                    else:
                        terminal = "SAFE_STOP_AMBIGUOUS_EFFECT"

            if writer is not None:
                for _ in range(30):
                    writer.append_data(
                        capture_biview_demo(
                            env, obs, demo=demo, status=f"OUTCOME {terminal}"
                        )
                    )

            # Evaluation begins only after the controller has stopped.
            final_pixels = evaluator_pixels(env, obs)
            final_joint = float(env.env.sim.data.get_joint_qpos(MIDDLE_JOINT))
            evaluator_revealed = (
                final_joint < OPEN_LIMIT and max(final_pixels.values()) >= MIN_TARGET_PIXELS
            )
            correct_terminal = terminal == "STOP_REVEALED" and evaluator_revealed
            row = {
                "seed": seed,
                "initial_prediction_set": list(initial.prediction_set),
                "initial_probabilities": initial.probabilities,
                "risk_decision": decision.to_dict(),
                "history": history,
                "executor_steps": executed_steps,
                "terminal": terminal,
                "evaluator_revealed": evaluator_revealed,
                "correct_terminal": correct_terminal,
                "evaluator_only": {
                    "middle_joint_final": final_joint,
                    "target_pixels": final_pixels,
                },
                "controller_joint_reads": controller_joint_reads,
                "controller_segmentation_reads": 0,
                "video": str(video_path.relative_to(ROOT)) if video_path else None,
            }
            rows.append(row)
            print(
                f"seed={seed} route={decision.selected_action} terminal={terminal} "
                f"revealed={evaluator_revealed} correct={correct_terminal}",
                flush=True,
            )
        finally:
            if writer is not None:
                writer.close()
            env.close()

    report = {
        "schema_version": "interactive-perception.t01-outcome-closed-loop.v1",
        "objective": "target observability, not final task completion",
        "policy": "pi05_libero frozen action + prefix server",
        "scene": str(bddl.relative_to(ROOT)),
        "seeds": args.seeds,
        "calibration_only": True,
        "information_cost": information_cost,
        "losses": losses,
        "artifacts": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (
                args.belief_artifact,
                args.effect_artifact,
                args.outcome_artifact,
                args.outcome_audit,
                args.risk_contract,
            )
        },
        "episodes": len(rows),
        "correct_terminals": sum(row["correct_terminal"] for row in rows),
        "physical_reveals": sum(row["evaluator_revealed"] for row in rows),
        "controller_joint_reads": sum(row["controller_joint_reads"] for row in rows),
        "controller_segmentation_reads": sum(
            row["controller_segmentation_reads"] for row in rows
        ),
        "demo_contract": {
            "layout": "left wrist first-person; right evaluator-only global view",
            "policy_uses_demo_global_view": False,
            "video_seeds": args.video_seeds,
        },
        "outcome_check_every_steps": args.replan_steps,
        "outcome_prefix_requests_advance_action_rng": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"closed-loop result is immutable: {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("episodes", "correct_terminals", "physical_reveals")}, indent=2))


if __name__ == "__main__":
    main()
