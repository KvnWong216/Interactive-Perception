#!/usr/bin/env python3
"""Run the non-claim PIU V0 end-to-end inference smoke on public observations."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from collect_t01_action_effect_transitions import (  # noqa: E402
    DUMMY_ACTION,
    FINAL_PROMPT,
    OPEN_PROMPT,
    policy_visibility,
    save_packet_images,
)
from collect_t01_open_and_observe_effect import (  # noqa: E402
    RecordingEnvironment,
    replay_evaluator_trace,
)
from interaction_uncertainty.candidates import DrawerV0CandidateGenerator  # noqa: E402
from interaction_uncertainty.contracts import (  # noqa: E402
    ActionEffectForecast,
    CandidateAction,
    FactDistribution,
    ObjectNode,
    Primitive,
    ScenePacket,
    TaskBelief,
    UnknownRegion,
)
from interaction_uncertainty.optimizer import InformationUtilityOptimizer  # noqa: E402
from interaction_uncertainty.scene_memory import SceneMemory  # noqa: E402
from interaction_uncertainty.sidecar import PIUSidecarPredictor  # noqa: E402
from interaction_uncertainty.task_parser import FrozenRetrievalTaskParser  # noqa: E402
from interactive_perception.action_options import execute_open_and_observe  # noqa: E402
from interactive_perception.policy_client import (  # noqa: E402
    OpenPiWebsocketPolicy,
    build_observation,
)
from interactive_perception.rgb_outcome_critic import (  # noqa: E402
    V12bPublicRGBOutcomeCritic,
)

OUTCOME_QUERY = "Find the butter"
TARGET_PIXELS = 256
OPEN_EXECUTION_RELIABILITY_LOWER_BOUND = 0.924


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_scene(*, frame_id: str, prompt: str, state: np.ndarray) -> ScenePacket:
    """V0 scene schema from registered public T01 affordances plus VLM belief.

    ``prompt_target`` is a semantic handoff anchor, not a detector box.  It is
    hard-valid only when the conformal location set says visible.  Grounding
    DINO/SAM replace this proxy in V1.
    """

    return ScenePacket(
        frame_id=frame_id,
        prompt=prompt,
        objects=(
            ObjectNode(
                "drawer_middle",
                {"middle drawer": 1.0},
                affordances=frozenset({"openable"}),
            ),
            ObjectNode(
                "prompt_target",
                {"prompt target": 1.0},
                affordances=frozenset({"task_target"}),
            ),
        ),
        unknown_regions=(
            UnknownRegion(
                "drawer_middle_interior",
                "drawer_middle",
                "UNOBSERVED_CONTAINER_INTERIOR",
                0.0,
                Primitive.OPEN_TO_INSPECT,
            ),
        ),
        public_robot_state=tuple(float(value) for value in state),
        backend_stamp="registered-affordances+pi05-prefix-v0",
    )


def semantic_handoff_forecast(
    candidate: CandidateAction,
    belief: TaskBelief,
    progress: float,
) -> ActionEffectForecast:
    """V0 direct-route forecast; this is not physical ACT reliability."""

    visible = float(belief.fact("target_location").probabilities["visible_workspace"])
    ready = TaskBelief(
        prompt=belief.prompt,
        facts=(
            FactDistribution(
                "target_location",
                {
                    "visible_workspace": 0.997,
                    "middle_drawer": 0.001,
                    "other_unsearched_region": 0.001,
                    "absent": 0.001,
                },
                1.0,
            ),
        ),
        node_uncertainty={},
        model_stamp=belief.model_stamp,
    )
    return ActionEffectForecast(
        candidate_id=candidate.candidate_id,
        outcome_probabilities={"HANDOFF_READY": visible, "HANDOFF_UNSUPPORTED": 1.0 - visible},
        future_beliefs={"HANDOFF_READY": ready, "HANDOFF_UNSUPPORTED": belief},
        execution_success_probability=visible,
        expected_task_progress=float(progress),
        model_stamp=f"{belief.model_stamp}:semantic-handoff-only",
    )


def choose(
    *,
    predictor: PIUSidecarPredictor,
    task,
    belief: TaskBelief,
    scene: ScenePacket,
    memory: SceneMemory,
    global_features: np.ndarray,
    spatial_features: np.ndarray,
):
    candidates = DrawerV0CandidateGenerator().generate(
        task=task, scene=scene, belief=belief, memory=memory
    )
    progress = predictor.learned_progress(global_features=global_features)
    forecasts = {}
    for candidate in candidates:
        if candidate.primitive is Primitive.OPEN_TO_INSPECT:
            learned = predictor.forecast(
                candidate_id=candidate.candidate_id,
                primitive=candidate.primitive.value,
                prompt=belief.prompt,
                spatial_features=spatial_features,
            )
            forecasts[candidate.candidate_id] = dataclasses.replace(
                learned,
                execution_success_probability=min(
                    learned.execution_success_probability,
                    OPEN_EXECUTION_RELIABILITY_LOWER_BOUND,
                ),
            )
        elif candidate.primitive is Primitive.DIRECT_ACT:
            forecasts[candidate.candidate_id] = semantic_handoff_forecast(
                candidate, belief, progress[candidate.candidate_id]
            )
    decision = InformationUtilityOptimizer(
        information_weight=1.0,
        task_progress_weight=1.5,
        cost_weight=1.0,
        risk_weight=0.25,
    ).select(
        belief=belief,
        candidates=candidates,
        forecasts=forecasts,
        learned_progress={
            candidate.candidate_id: progress[candidate.candidate_id]
            for candidate in candidates
            if candidate.primitive
            in {Primitive.DIRECT_ACT, Primitive.STOP_NOT_FOUND, Primitive.COMPLETE}
        },
    )
    return decision, forecasts


def static_evaluator(*, bddl: Path, seed: int, wait_steps: int) -> dict:
    from libero.libero.envs import SegmentationRenderEnv

    env = SegmentationRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
    )
    try:
        env.seed(seed)
        observation = env.reset()
        for _ in range(wait_steps):
            observation, _, _, _ = env.step(DUMMY_ACTION)
        pixels = policy_visibility(env, observation)
        return {
            "visibility_history": [{"name": "before", "target_pixels": pixels}],
            "target_qpos_history": [],
        }
    finally:
        env.close()


def replay_final_task_evaluator(
    *, bddl: Path, seed: int, wait_steps: int, actions: list[list[float]]
) -> dict:
    """Score task success only after the online controller has terminated."""

    from libero.libero.envs import SegmentationRenderEnv

    env = SegmentationRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
    )
    try:
        env.seed(seed)
        observation = env.reset()
        for _ in range(wait_steps):
            observation, _, _, _ = env.step(DUMMY_ACTION)
        task_success = False
        success_step = None
        for index, action in enumerate(actions, start=1):
            observation, _, done, _ = env.step(action)
            if bool(done) and success_step is None:
                task_success = True
                success_step = index
        return {
            "task_success": task_success,
            "first_success_step": success_step,
            "scored_after_controller_terminal": True,
            "actions_replayed": len(actions),
            "final_target_pixels": policy_visibility(env, observation),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=1399)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--execute-final-act", action="store_true")
    parser.add_argument("--act-steps", type=int, default=400)
    parser.add_argument(
        "--case",
        action="append",
        choices=(
            "hidden_butter",
            "same_rgb_visible_cream_cheese",
            "open_visible_butter",
            "middle_drawer_empty",
            "drawer_action_failed_control",
        ),
        default=None,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "results/models/piu_v0_3_sidecar.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/smoke/piu_v0_v12b_full_pipeline_v1_seed1399.json",
    )
    parser.add_argument(
        "--outcome-composite",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_v12b_composite_candidate.json",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=None,
        help=(
            "Optional immutable directory for policy-visible keyframes and "
            "public action histories. Visual demos are rendered from these "
            "assets only after the controller terminates."
        ),
    )
    args = parser.parse_args()
    for name in ("model", "output", "asset_dir", "outcome_composite"):
        value = getattr(args, name)
        if value is None:
            continue
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output.exists():
        raise FileExistsError(f"immutable smoke exists: {args.output}")
    if args.asset_dir is not None:
        if args.asset_dir.exists() and any(args.asset_dir.iterdir()):
            raise FileExistsError(f"immutable asset directory is not empty: {args.asset_dir}")
        args.asset_dir.mkdir(parents=True, exist_ok=True)
    predictor = PIUSidecarPredictor(args.model)
    outcome_critic = V12bPublicRGBOutcomeCritic(
        args.outcome_composite, root=ROOT
    )
    parser_model = FrozenRetrievalTaskParser()
    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port)
    metadata = policy.server_metadata
    if metadata.get("prefix_requests_advance_action_rng") is not False:
        raise RuntimeError("prefix requests must not advance action RNG")

    cases = (
        {
            "id": "hidden_butter",
            "bddl": ROOT / "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl",
            "prompt": FINAL_PROMPT,
            "open_steps": 300,
            "expected_initial": Primitive.OPEN_TO_INSPECT,
            "expected_outcome": "REVEALED",
        },
        {
            "id": "same_rgb_visible_cream_cheese",
            "bddl": ROOT / "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl",
            "prompt": "Place the cream cheese in the basket",
            "open_steps": None,
            "expected_initial": Primitive.DIRECT_ACT,
            "expected_outcome": None,
        },
        {
            "id": "open_visible_butter",
            "bddl": ROOT / "scenarios/t01_stock_ladder_v1/T01E_open_drawer_retrieval.bddl",
            "prompt": FINAL_PROMPT,
            "open_steps": None,
            "expected_initial": Primitive.DIRECT_ACT,
            "expected_outcome": None,
        },
        {
            "id": "middle_drawer_empty",
            "bddl": ROOT / "scenarios/t01_stock_ladder_v1/T01F_middle_drawer_empty_calibration.bddl",
            "prompt": FINAL_PROMPT,
            "open_steps": 300,
            "expected_initial": Primitive.OPEN_TO_INSPECT,
            "expected_outcome": "EMPTY",
        },
        {
            "id": "drawer_action_failed_control",
            "bddl": ROOT / "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl",
            "prompt": FINAL_PROMPT,
            "open_steps": 25,
            "expected_initial": Primitive.OPEN_TO_INSPECT,
            "expected_outcome": "FAILED",
        },
    )
    if args.case is not None:
        requested = set(args.case)
        cases = tuple(case for case in cases if case["id"] in requested)
    if args.execute_final_act and any(
        case["id"] != "hidden_butter" for case in cases
    ):
        raise ValueError(
            "physical final ACT is currently registered only for hidden_butter"
        )
    from libero.libero.envs import OffScreenRenderEnv

    rows = []
    paired_target_qpos = {}
    for case in cases:
        case_asset_dir = (
            args.asset_dir / case["id"] if args.asset_dir is not None else None
        )
        public_history_assets = []

        def save_public_asset(packet, *, name: str, phase: str, step: int) -> None:
            if case_asset_dir is None:
                return
            paths, hashes = save_packet_images(
                packet, case_asset_dir / "public_keyframes", name
            )
            public_history_assets.append(
                {
                    "name": name,
                    "phase": phase,
                    "step": int(step),
                    "image_paths": paths,
                    "image_sha256": hashes,
                    "robot_state": [float(value) for value in packet.state],
                }
            )

        env = OffScreenRenderEnv(
            bddl_file_name=str(case["bddl"]), camera_heights=256, camera_widths=256
        )
        try:
            env.seed(args.seed)
            observation = env.reset()
            for _ in range(args.wait_steps):
                observation, _, _, _ = env.step(DUMMY_ACTION)
            task = parser_model.parse(case["prompt"])
            final_packet = build_observation(observation, case["prompt"])
            query_packet = build_observation(observation, OUTCOME_QUERY)
            save_public_asset(
                final_packet, name="00_before", phase="BEFORE", step=-1
            )
            global_features = policy.encode_prefix(final_packet, feature_schema="global_v1")
            spatial_features = policy.encode_prefix(
                query_packet, feature_schema="cognitive_spatial_v5"
            )
            belief = predictor.initial_belief(
                prompt=case["prompt"], global_features=global_features
            )
            memory = SceneMemory()
            scene = public_scene(
                frame_id=f"{case['id']}:before",
                prompt=case["prompt"],
                state=final_packet.state,
            )
            memory.append_observation(scene, belief)
            decision, forecasts = choose(
                predictor=predictor,
                task=task,
                belief=belief,
                scene=scene,
                memory=memory,
                global_features=global_features,
                spatial_features=spatial_features,
            )
            initial_correct = decision.selected.primitive is case["expected_initial"]
            trace = [
                {
                    "event": "INITIAL_DECISION",
                    "belief": belief.to_dict(),
                    "decision": decision.to_dict(),
                    "effect_forecasts": {
                        key: {
                            "outcomes": dict(value.outcome_probabilities),
                            "expected_future_uncertainty": value.expected_future_uncertainty,
                            "execution_success_probability": value.execution_success_probability,
                            "expected_task_progress": value.expected_task_progress,
                        }
                        for key, value in forecasts.items()
                    },
                }
            ]
            outcome_set = None
            outcome_prediction = None
            execution = None
            evaluator = None
            final_task_evaluator = None
            information_actions: list[list[float]] = []
            act_execution = None
            terminal = "DIRECT_ACT_HANDOFF" if decision.selected.primitive is Primitive.DIRECT_ACT else "ABSTAIN"
            if decision.selected.primitive is Primitive.OPEN_TO_INSPECT:
                memory.begin(decision.selected.candidate_id)
                history_features = [spatial_features]
                robot_history = [query_packet.state]
                rgb_history = [query_packet]
                capture_steps = {
                    max(0, round(case["open_steps"] * fraction) - 1): index
                    for index, fraction in enumerate((0.25, 0.50, 0.75, 1.00), start=1)
                }

                def observe_step(phase, step, current_obs):
                    if phase == "OPEN" and step in capture_steps:
                        packet = build_observation(current_obs, OUTCOME_QUERY)
                        history_features.append(
                            policy.encode_prefix(
                                packet, feature_schema="cognitive_spatial_v5"
                            )
                        )
                        robot_history.append(packet.state)
                        rgb_history.append(packet)
                        index = capture_steps[step]
                        save_public_asset(
                            packet,
                            name=f"{index:02d}_open",
                            phase="OPEN",
                            step=step,
                        )

                recording = RecordingEnvironment(env)
                observation, execution = execute_open_and_observe(
                    env=recording,
                    initial_observation=observation,
                    policy=policy,
                    open_prompt=OPEN_PROMPT,
                    open_steps=case["open_steps"],
                    replan_steps=args.replan_steps,
                    step_observer=observe_step,
                )
                information_actions = list(recording.actions)
                after_query = build_observation(observation, OUTCOME_QUERY)
                history_features.append(
                    policy.encode_prefix(
                        after_query, feature_schema="cognitive_spatial_v5"
                    )
                )
                robot_history.append(after_query.state)
                rgb_history.append(after_query)
                save_public_asset(
                    after_query,
                    name="05_returned",
                    phase="AFTER_RETURN",
                    step=execution.return_steps,
                )
                if len(rgb_history) != 6:
                    outcome_set = ("FAILED", "REVEALED", "EMPTY")
                else:
                    outcome_prediction = outcome_critic.predict(rgb_history)
                    outcome_set = outcome_prediction.prediction_set
                memory.accept_outcome(
                    decision.selected.candidate_id,
                    outcome_set,
                    searched_region="drawer_middle_interior",
                )
                if len(outcome_set) != 1:
                    terminal = "OUTCOME_AMBIGUOUS_SAFE_STOP"
                elif outcome_set == ("FAILED",):
                    terminal = "ACTION_FAILED_SAFE_STOP"
                elif outcome_set == ("EMPTY",):
                    belief = forecasts[decision.selected.candidate_id].future_beliefs["EMPTY"]
                    belief = dataclasses.replace(
                        belief,
                        conformal_sets={"target_location": ("other_unsearched_region",)},
                    )
                    terminal = "LOCAL_EMPTY_SAFE_STOP"
                else:
                    after_final = build_observation(observation, case["prompt"])
                    after_global = policy.encode_prefix(after_final, feature_schema="global_v1")
                    belief = predictor.initial_belief(
                        prompt=case["prompt"], global_features=after_global
                    )
                    after_scene = public_scene(
                        frame_id=f"{case['id']}:after",
                        prompt=case["prompt"],
                        state=after_final.state,
                    )
                    memory.append_observation(after_scene, belief)
                    next_decision, next_forecasts = choose(
                        predictor=predictor,
                        task=task,
                        belief=belief,
                        scene=after_scene,
                        memory=memory,
                        global_features=after_global,
                        spatial_features=history_features[-1],
                    )
                    terminal = (
                        "INFORMATION_ACQUIRED_AND_DIRECT_HANDOFF"
                        if next_decision.selected.primitive is Primitive.DIRECT_ACT
                        else "REVEALED_BUT_REPLAN_SAFE_STOP"
                    )
                    trace.append(
                        {
                            "event": "POST_OUTCOME_REPLAN",
                            "belief": belief.to_dict(),
                            "decision": next_decision.to_dict(),
                            "effect_forecasts": list(next_forecasts),
                        }
                    )
                    if (
                        args.execute_final_act
                        and case["id"] == "hidden_butter"
                        and next_decision.selected.primitive is Primitive.DIRECT_ACT
                    ):
                        act_start_index = len(recording.actions)
                        act_plan: list[np.ndarray] = []
                        capture_act_steps = {
                            max(0, round(args.act_steps * fraction) - 1): index
                            for index, fraction in enumerate(
                                (0.25, 0.50, 0.75, 1.00), start=1
                            )
                        }
                        for act_step in range(args.act_steps):
                            if not act_plan:
                                chunks = policy.sample_chunks(
                                    build_observation(observation, case["prompt"]), 1
                                )
                                act_plan.extend(chunks[0][: args.replan_steps])
                            observation, _, _, _ = recording.step(
                                act_plan.pop(0).tolist()
                            )
                            if act_step in capture_act_steps:
                                save_public_asset(
                                    build_observation(observation, case["prompt"]),
                                    name=(
                                        f"{5 + capture_act_steps[act_step]:02d}_act"
                                    ),
                                    phase="DIRECT_ACT",
                                    step=act_step,
                                )
                        act_execution = {
                            "prompt": case["prompt"],
                            "fixed_budget_steps": args.act_steps,
                            "action_start_index": act_start_index,
                            "task_predicate_used_for_control": False,
                        }
                        terminal = "FINAL_TASK_EVALUATION_PENDING"
                evaluator = replay_evaluator_trace(
                    bddl=case["bddl"],
                    seed=args.seed,
                    wait_steps=args.wait_steps,
                    actions=information_actions,
                    open_history_steps=capture_steps,
                    counterpart_target_qpos=(
                        paired_target_qpos if case["id"] == "middle_drawer_empty" else {}
                    ),
                )
                if case["id"] == "hidden_butter":
                    paired_target_qpos = {
                        point["name"]: point["qpos"]
                        for point in evaluator["target_qpos_history"]
                    }
                if act_execution is not None:
                    final_task_evaluator = replay_final_task_evaluator(
                        bddl=case["bddl"],
                        seed=args.seed,
                        wait_steps=args.wait_steps,
                        actions=recording.actions,
                    )
                    terminal = (
                        "FINAL_TASK_SUCCESS"
                        if final_task_evaluator["task_success"]
                        else "FINAL_TASK_FAILED_AFTER_INFORMATION_ACQUIRED"
                    )
            else:
                evaluator = static_evaluator(
                    bddl=case["bddl"], seed=args.seed, wait_steps=args.wait_steps
                )
            action_asset = None
            if case_asset_dir is not None:
                case_asset_dir.mkdir(parents=True, exist_ok=True)
                action_path = case_asset_dir / "public_action_history.json"
                action_values = recording.actions if decision.selected.primitive is Primitive.OPEN_TO_INSPECT else []
                action_path.write_text(json.dumps(action_values, separators=(",", ":")) + "\n")
                action_asset = {
                    "path": str(action_path.relative_to(ROOT)),
                    "sha256": digest(action_path),
                    "steps": len(action_values),
                }
            evaluator_revealed = any(
                max(point["target_pixels"].values()) >= TARGET_PIXELS
                for point in evaluator["visibility_history"]
            )
            outcome_correct = (
                case["expected_outcome"] is None
                or outcome_set == (case["expected_outcome"],)
            )
            trace.append(
                {
                    "event": "OUTCOME",
                    "prediction_set": list(outcome_set) if outcome_set else None,
                    "public_rgb_critic": (
                        outcome_prediction.to_dict()
                        if outcome_prediction is not None
                        else None
                    ),
                    "memory": memory.to_dict(),
                    "terminal": terminal,
                }
            )
            rows.append(
                {
                    "case": case["id"],
                    "seed": args.seed,
                    "scenario": str(case["bddl"].relative_to(ROOT)),
                    "prompt": case["prompt"],
                    "initial_expected": case["expected_initial"].value,
                    "initial_selected": decision.selected.primitive.value,
                    "initial_correct": initial_correct,
                    "outcome_expected": case["expected_outcome"],
                    "outcome_prediction_set": list(outcome_set) if outcome_set else None,
                    "outcome_correct": outcome_correct,
                    "terminal": terminal,
                    "trace": trace,
                    "executor": (
                        {
                            "open_steps": execution.open_steps,
                            "return_steps": execution.return_steps,
                            "return_phase": execution.return_status.phase.value,
                        }
                        if execution is not None
                        else None
                    ),
                    "act_executor": act_execution,
                    "assets": (
                        {
                            "public_keyframes": public_history_assets,
                            "public_action_history": action_asset,
                            "global_view_available_to_controller": False,
                        }
                        if case_asset_dir is not None
                        else None
                    ),
                    "evaluator_only": {
                        "revealed_any_history": evaluator_revealed,
                        "visibility_history": evaluator["visibility_history"],
                        "final_task": final_task_evaluator,
                    },
                    "online_oracle_inputs": [],
                }
            )
            print(
                f"{case['id']}: {decision.selected.primitive.value} -> {outcome_set} -> {terminal}",
                flush=True,
            )
        finally:
            env.close()
    report = {
        "schema_version": "interaction-uncertainty.piu-v0-v12b-behavior.v1",
        "claim_status": (
            "disposable-seed behavior demonstration using the clean-and-sealed-GO "
            "v12b outcome critic; not an independent validation split"
        ),
        "endpoint": "semantic DIRECT_ACT handoff after prompt-relevant information acquisition; not final manipulation success",
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model": str(args.model.relative_to(ROOT)),
        "model_sha256": digest(args.model),
        "policy": "frozen pi05_libero action + prefix server",
        "outcome_critic": {
            "path": str(args.outcome_composite.relative_to(ROOT)),
            "sha256": outcome_critic.composite_sha256,
            "online_inputs": [
                "six stock agentview RGB frames",
                "six stock wrist RGB frames",
            ],
            "online_oracle_inputs": [],
        },
        "seed": args.seed,
        "cases": len(rows),
        "initial_correct": sum(row["initial_correct"] for row in rows),
        "outcome_correct": sum(row["outcome_correct"] for row in rows),
        "information_acquisition_successes": sum(
            row["terminal"]
            in {
                "INFORMATION_ACQUIRED_AND_DIRECT_HANDOFF",
                "FINAL_TASK_SUCCESS",
                "FINAL_TASK_FAILED_AFTER_INFORMATION_ACQUIRED",
            }
            for row in rows
        ),
        "physical_final_act_executed": args.execute_final_act,
        "final_task_successes": sum(
            bool((row["evaluator_only"].get("final_task") or {}).get("task_success"))
            for row in rows
        ),
        "online_oracle_inputs": [],
        "asset_root": (
            str(args.asset_dir.relative_to(ROOT))
            if args.asset_dir is not None
            else None
        ),
        "asset_contract": {
            "raw_keyframes": "stock policy RGB only",
            "actions": "public executed 7D action history",
            "global_demo_view": "must be rendered by post-terminal evaluator replay",
            "global_view_available_to_controller": False,
        },
        "direct_act_limitation": "DIRECT_ACT is a semantic handoff only; context-specific physical reliability is not yet trained or evaluated",
        "perception_limitation": "V0 uses registered public drawer affordances and a pi05-prefix visible-target proxy; Grounding DINO/SAM object grounding is V1",
        "utility_contract": "physical information actions use effect-head EIG/progress; rank-head readiness is used only by the DIRECT semantic handoff",
        "open_execution_reliability": {
            "trials": 100,
            "successes": 97,
            "point_estimate": 0.97,
            "one_sided_95_lower_bound": OPEN_EXECUTION_RELIABILITY_LOWER_BOUND,
            "optimizer_uses": "minimum of learned forecast and lower bound",
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("cases", "initial_correct", "outcome_correct")}, indent=2))


if __name__ == "__main__":
    main()
