#!/usr/bin/env python3
"""Run the complete oracle-free T01 target-observability pipeline."""

from __future__ import annotations

import argparse
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
from interactive_perception.action_options import execute_open_and_observe  # noqa: E402
from interactive_perception.action_outcome import (  # noqa: E402
    HierarchicalActionOutcomePredictor,
    PI05_PATCH_EQUIVALENT_TARGET_PIXELS,
)
from interactive_perception.active_risk import (  # noqa: E402
    ACT,
    DecisionLosses,
    ExpectedRiskPlanner,
    TargetBelief,
    TargetHypothesis,
    TargetState,
)
from interactive_perception.minimal_pipeline import (  # noqa: E402
    INFORMATION_ACQUIRED,
    MinimalPromptAlignedPipeline,
)
from interactive_perception.policy_client import (  # noqa: E402
    OpenPiWebsocketPolicy,
    build_observation,
)
from interactive_perception.prompt_state import PromptStatePredictor  # noqa: E402
from interactive_perception.seed_registry import (  # noqa: E402
    load_seed_registry,
    seeds_for_blocks,
)
from run_t01_outcome_closed_loop import (  # noqa: E402
    DUMMY_ACTION,
    FINAL_PROMPT,
    MIDDLE_JOINT,
    OPEN_PROMPT,
    capture_biview_demo,
    evaluator_pixels,
)
from collect_t01_open_and_observe_effect import (  # noqa: E402
    RecordingEnvironment,
    replay_evaluator_trace,
)

ACTION = "OPEN_AND_OBSERVE"
OUTCOME_QUERY = "Find the butter"
SEED_REGISTRY = ROOT / "benchmarks/rss_v1/seed_registry.yaml"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open-steps", type=int, default=300)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--video-seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--belief-artifact",
        type=Path,
        default=ROOT / "results/calibration/prompt_state_belief_t01_v1.json",
    )
    parser.add_argument(
        "--effect-artifact",
        type=Path,
        default=ROOT / "results/calibration/t01_open_and_observe_effect_v4.json",
    )
    parser.add_argument(
        "--outcome-artifact",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_critic_v9.json",
    )
    parser.add_argument(
        "--outcome-audit",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_audit_v9.json",
    )
    parser.add_argument(
        "--risk-contract",
        type=Path,
        default=ROOT / "benchmarks/rss_v1/risk_contract_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    seed_registry = load_seed_registry(SEED_REGISTRY)
    seed_block = (
        "t01_closed_loop_smoke" if args.smoke else "t01_closed_loop_validation"
    )
    expected_seeds = seeds_for_blocks(seed_registry, [seed_block])
    if args.seeds is None:
        args.seeds = expected_seeds
    if args.seeds != expected_seeds:
        raise ValueError(
            "closed-loop smoke seed is exactly 1300"
            if args.smoke
            else "the frozen v5 closed-loop seeds are exactly 800-899"
        )
    if args.video_seeds is None:
        args.video_seeds = [1300] if args.smoke else [800]
    if args.video_dir is None:
        args.video_dir = ROOT / (
            "results/demos/t01_closed_loop_v5_smoke"
            if args.smoke
            else "results/demos/t01_closed_loop_v5"
        )
    if args.output is None:
        args.output = ROOT / (
            "results/t01_closed_loop_v5_smoke_seed1300.json"
            if args.smoke
            else "results/t01_closed_loop_v5_100seed.json"
        )
    for name in (
        "video_dir",
        "belief_artifact",
        "effect_artifact",
        "outcome_artifact",
        "outcome_audit",
        "risk_contract",
        "output",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if set(args.video_seeds) - set(args.seeds):
        raise ValueError("video seeds must be part of the frozen rollout")
    if args.output.exists():
        raise FileExistsError(f"immutable closed-loop result exists: {args.output}")

    audit = json.loads(args.outcome_audit.read_text())
    if not audit.get("fp3_passed", False):
        raise ValueError("the sealed v9 action-effect/outcome audit must pass first")
    if audit.get("artifact_sha256") != digest(args.outcome_artifact):
        raise ValueError("outcome artifact differs from the sealed audit")
    effect_artifact = json.loads(args.effect_artifact.read_text())
    if not effect_artifact.get("physical_effect_gate_passed", False):
        raise ValueError("OPEN_AND_OBSERVE is not capability authorized")
    contract = json.loads(args.risk_contract.read_text())
    losses = contract["losses"]
    belief_predictor = PromptStatePredictor.from_artifact(
        json.loads(args.belief_artifact.read_text())
    )
    outcome_predictor = HierarchicalActionOutcomePredictor.from_artifact(
        json.loads(args.outcome_artifact.read_text())
    )
    effect = EffectRegistry.from_dict(effect_artifact).get(
        "t01_stock_middle_drawer_search", ACTION
    ).planner_effect(
        resolves=("MANIPULATION_ONLY",),
        cost=float(contract["information_cost"]),
        desired_outcome=None,
    )
    planner = ExpectedRiskPlanner(
        losses=DecisionLosses(
            false_commit=float(losses["false_commit"]),
            false_absent=float(losses["false_absent"]),
            act_execution_failure=float(losses["act_execution_failure"]),
            safe_stop=float(losses["safe_stop"]),
        ),
        act_reliability=1.0,
        horizon=1,
        temporal_violation_loss=0.0,
    )
    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port)
    metadata = policy.server_metadata
    if metadata.get("prefix_requests_advance_action_rng") is not False:
        raise ValueError("prefix requests must not advance action sampling")
    if metadata.get("prefix_feature_schemas", {}).get("cognitive_spatial_v5") != 21504:
        raise ValueError("server does not expose the frozen v5 spatial feature schema")

    from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv

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
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
        )
        writer = None
        video_path = None
        try:
            env.seed(seed)
            obs = env.reset()
            for _ in range(10):
                obs, _, _, _ = env.step(DUMMY_ACTION)
            evaluator_visibility_history = []
            initial_prefix = policy.encode_prefix(build_observation(obs, FINAL_PROMPT))
            initial = belief_predictor.predict(initial_prefix)
            observed_probability = float(initial.probabilities["OBSERVED"])
            hidden_probability = float(initial.probabilities["MANIPULATION_ONLY"])
            belief = TargetBelief(
                (
                    TargetHypothesis("OBSERVED", TargetState.OBSERVED),
                    TargetHypothesis(
                        "MANIPULATION_ONLY", TargetState.MANIPULATION_ONLY, ACTION
                    ),
                ),
                (observed_probability, hidden_probability),
            )
            controller = MinimalPromptAlignedPipeline(
                planner=planner,
                belief=belief,
                conformal_labels=initial.prediction_set,
                effects=(effect,),
                maximum_attempts_per_action=1,
            )
            decision = controller.plan()
            controller.begin(decision)

            if seed in args.video_seeds:
                args.video_dir.mkdir(parents=True, exist_ok=True)
                video_path = args.video_dir / f"t01_closed_loop_v5_seed{seed:03d}.mp4"
                writer = imageio.get_writer(video_path, fps=20)
                uncertainty = decision.resolvable_uncertainty.get(ACTION, float("nan"))
                for _ in range(20):
                    writer.append_data(
                        capture_biview_demo(
                            env,
                            obs,
                            demo=demo,
                            status=f"ROUTE {decision.selected_action} | U_res {uncertainty:+.3f}",
                        )
                    )

            outcome_set = None
            execution = None
            if decision.selected_action == ACTION:
                history_features = [
                    policy.encode_prefix(
                        build_observation(obs, OUTCOME_QUERY),
                        feature_schema="cognitive_spatial_v5",
                    )
                ]
                robot_history = [build_observation(obs, OUTCOME_QUERY).state]
                capture_steps = {
                    max(0, round(args.open_steps * fraction) - 1): index
                    for index, fraction in enumerate(
                        (0.25, 0.50, 0.75, 1.00), start=1
                    )
                }

                def observe_step(phase, step, current_obs):
                    if writer is not None and step % 2 == 0:
                        writer.append_data(
                            capture_biview_demo(
                                env, current_obs, demo=demo, status=phase
                            )
                        )
                    if phase == "OPEN" and step in capture_steps:
                        index = capture_steps[step]
                        packet = build_observation(current_obs, OUTCOME_QUERY)
                        history_features.append(
                            policy.encode_prefix(
                                packet, feature_schema="cognitive_spatial_v5"
                            )
                        )
                        robot_history.append(packet.state)

                recording_env = RecordingEnvironment(env)
                obs, execution = execute_open_and_observe(
                    env=recording_env,
                    initial_observation=obs,
                    policy=policy,
                    open_prompt=OPEN_PROMPT,
                    open_steps=args.open_steps,
                    replan_steps=args.replan_steps,
                    step_observer=observe_step,
                )
                final_packet = build_observation(obs, OUTCOME_QUERY)
                history_features.append(
                    policy.encode_prefix(
                        final_packet, feature_schema="cognitive_spatial_v5"
                    )
                )
                robot_history.append(final_packet.state)
                if execution.executor_completed and len(history_features) == 6:
                    outcome = outcome_predictor.predict_history(
                        np.asarray(history_features), np.asarray(robot_history)
                    )
                    outcome_set = outcome.prediction_set
                elif not execution.executor_completed:
                    outcome_set = ("FAILED",)
                else:
                    outcome_set = ("FAILED", "REVEALED", "EMPTY")
                controller.observe_information_outcome(outcome_set)
                if controller.terminal is None:
                    next_decision = controller.plan()
                    controller.begin(next_decision)

            if writer is not None:
                for _ in range(30):
                    writer.append_data(
                        capture_biview_demo(
                            env,
                            obs,
                            demo=demo,
                            status=(
                                f"OUTCOME {outcome_set} -> {controller.terminal}"
                                if outcome_set is not None
                                else f"TERMINAL {controller.terminal}"
                            ),
                        )
                    )

            # Private evaluation begins only after controller termination and
            # runs in a separate environment from the online controller.
            if execution is not None:
                evaluator = replay_evaluator_trace(
                    bddl=bddl,
                    seed=seed,
                    wait_steps=10,
                    actions=recording_env.actions,
                    open_history_steps=capture_steps,
                    counterpart_target_qpos={},
                )
                evaluator_visibility_history = evaluator["visibility_history"]
                pixels = evaluator_visibility_history[-1]["target_pixels"]
                joint = float(evaluator["middle_joint_final"])
            else:
                evaluator_env = SegmentationRenderEnv(
                    bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
                )
                try:
                    evaluator_env.seed(seed)
                    evaluator_obs = evaluator_env.reset()
                    for _ in range(10):
                        evaluator_obs, _, _, _ = evaluator_env.step(DUMMY_ACTION)
                    pixels = evaluator_pixels(evaluator_env, evaluator_obs)
                    joint = float(
                        evaluator_env.env.sim.data.get_joint_qpos(MIDDLE_JOINT)
                    )
                    evaluator_visibility_history = [
                        {"name": "before", "target_pixels": pixels},
                        {"name": "after", "target_pixels": pixels},
                    ]
                finally:
                    evaluator_env.close()
            revealed = any(
                max(point["target_pixels"].values())
                >= PI05_PATCH_EQUIVALENT_TARGET_PIXELS
                for point in evaluator_visibility_history
            )
            correct = controller.terminal == INFORMATION_ACQUIRED and revealed
            rows.append(
                {
                    "seed": seed,
                    "initial_prediction_set": list(initial.prediction_set),
                    "initial_probabilities": initial.probabilities,
                    "outcome_set": list(outcome_set) if outcome_set else None,
                    "terminal": controller.terminal,
                    "correct_terminal": correct,
                    "evaluator_revealed": revealed,
                    "pipeline_trace": controller.trace(),
                    "executor": (
                        {
                            "open_steps": execution.open_steps,
                            "return_steps": execution.return_steps,
                            "return_phase": execution.return_status.phase.value,
                        }
                        if execution is not None
                        else None
                    ),
                    "evaluator_only": {
                        "middle_joint": joint,
                        "target_pixels": pixels,
                        "visibility_history": evaluator_visibility_history,
                    },
                    "online_oracle_inputs": [],
                    "evaluator_timing": "separate action replay after controller termination",
                    "video": str(video_path.relative_to(ROOT)) if video_path else None,
                }
            )
            print(
                f"seed={seed} route={decision.selected_action} outcome={outcome_set} "
                f"terminal={controller.terminal} correct={correct}",
                flush=True,
            )
        finally:
            if writer is not None:
                writer.close()
            env.close()

    report = {
        "schema_version": "interactive-perception.t01-closed-loop.v5",
        "objective": (
            "target evidence acquired at any public history point; "
            "not final task completion"
        ),
        "paper_eligible": False,
        "split_role": (
            "non-paper wiring smoke"
            if args.smoke
            else "held-out custom-scenario pipeline validation"
        ),
        "policy": "frozen pi05_libero",
        "seeds": args.seeds,
        "seed_registry": str(SEED_REGISTRY.relative_to(ROOT)),
        "seed_blocks": [seed_block],
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
        "safe_stops": sum(row["terminal"] == "SAFE_STOP" for row in rows),
        "online_oracle_inputs": [],
        "probabilistic_temporal_progress_used": False,
        "control_memory": ["phase", "searched_set", "attempt_counts"],
        "demo_contract": {
            "layout": "left wrist first-person; right evaluator-only global",
            "policy_uses_demo_global_view": False,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "episodes",
                    "correct_terminals",
                    "physical_reveals",
                    "safe_stops",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
