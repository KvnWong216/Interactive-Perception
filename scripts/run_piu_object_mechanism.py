#!/usr/bin/env python3
"""Run one real object-level PIU → π0.5 → RGB outcome → replan loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from collect_t01_action_effect_transitions import (  # noqa: E402
    DUMMY_ACTION,
    OPEN_PROMPT,
    policy_visibility,
    save_packet_images,
)
from collect_t01_open_and_observe_effect import (  # noqa: E402
    RecordingEnvironment,
    replay_evaluator_trace,
)
from interaction_uncertainty.object_runtime import ObjectPIURuntime  # noqa: E402
from interaction_uncertainty.task_parser import FrozenRetrievalTaskParser  # noqa: E402
from interactive_perception.action_options import execute_open_and_observe  # noqa: E402
from interactive_perception.policy_client import (  # noqa: E402
    OpenPiWebsocketPolicy,
    build_observation,
)
from interactive_perception.rgb_outcome_critic import (  # noqa: E402
    V12bPublicRGBOutcomeCritic,
)
from run_piu_v0_smoke import replay_final_task_evaluator  # noqa: E402


OUTCOME_QUERY = "Find the butter"
TARGET_PIXELS = 256


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wait_for_port(port: int, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return
        except OSError:
            time.sleep(1.0)
    raise TimeoutError(f"pi0.5 server did not become ready on port {port}")


def compatible_belief(
    prompt: str,
    probabilities: dict[str, float],
    prediction_set: list[str],
    node_uncertainty: list[dict],
    model_stamp: str,
) -> dict:
    visible = float(probabilities.get("visible_workspace", 0.0))
    closed = float(probabilities.get("closed_container", 0.0))
    return {
        "prompt": prompt,
        "facts": [
            {
                "name": "target_location",
                "probabilities": {
                    "visible_workspace": visible,
                    "middle_drawer": closed,
                    "other_unsearched_region": 0.0,
                    "absent": 0.0,
                },
                "importance": 1.0,
            }
        ],
        "task_uncertainty": float(
            -(visible * np.log(max(visible, 1e-9)) + closed * np.log(max(closed, 1e-9)))
            / np.log(2.0)
        ),
        "node_uncertainty": {
            row["object_id"]: row["relevance_probability"] for row in node_uncertainty
        },
        "model_stamp": model_stamp,
        "conformal_sets": {
            "target_location": [
                "middle_drawer" if value == "closed_container" else value
                for value in prediction_set
            ]
        },
    }


def public_snapshot(
    *,
    packet,
    prompt: str,
    target: str,
    destination: str | None,
    seed: int,
    scenario: Path,
    sample_id: str,
    image_dir: Path,
) -> tuple[dict, dict[str, np.ndarray]]:
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    hashes = {}
    arrays = {"agentview": packet.image, "wrist": packet.wrist_image}
    for view, image in arrays.items():
        path = image_dir / f"{view}.png"
        imageio.imwrite(path, image)
        paths[view] = str(path.relative_to(ROOT))
        hashes[view] = digest(path)
    return (
        {
            "schema_version": "interaction-uncertainty.piu-mechanism-snapshot.v1",
            "sample_id": sample_id,
            "sample_type": "INITIAL_TASK_BELIEF",
            "split": "disposable_mechanism_wiring",
            "seed": seed,
            "scenario_id": "t01_hidden_butter",
            "scenario_path": str(scenario.relative_to(ROOT)),
            "prompt": prompt,
            "target": target,
            "destination": destination,
            "visual_queries": [value for value in (target, destination) if value],
            "candidate_actions": ["DIRECT_ACT", "OPEN_TO_INSPECT", "ABSTAIN"],
            "policy_inputs": {
                "image_paths": paths,
                "image_sha256": hashes,
                "public_robot_state": [float(value) for value in packet.state],
            },
            "online_oracle_inputs": [],
        },
        arrays,
    )


def run_frontend(*, snapshot: Path, work_dir: Path, gpu: int) -> tuple[Path, Path, Path]:
    scene = work_dir / "object_scene.jsonl"
    features = work_dir / "object_features.npz"
    assets = work_dir / "object_assets"
    openpi = ROOT.parent / "openpi"
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "UV_CACHE_DIR": "/tmp/ipu-uv-cache",
        }
    )
    subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "python",
            str(ROOT / "scripts/build_piu_object_scene_packets.py"),
            "--input-index",
            str(snapshot),
            "--output-index",
            str(scene),
            "--feature-store",
            str(features),
            "--asset-dir",
            str(assets),
            "--device",
            "cuda",
        ],
        cwd=openpi,
        env=environment,
        check=True,
    )
    return scene, features, assets


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
    parser.add_argument("--open-steps", type=int, default=300)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--act-steps", type=int, default=400)
    parser.add_argument("--execute-final-act", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "results/models/piu_object_sidecar_v2.pt",
    )
    parser.add_argument(
        "--outcome-composite",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_v12b_composite_candidate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/mechanism/piu_object_t01_seed1399_v1.json",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=ROOT / "results/assets/piu_object_t01_seed1399_v1/raw",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "outputs/piu_object_t01_seed1399_v1",
    )
    args = parser.parse_args()
    for name in ("bddl", "model", "outcome_composite", "output", "asset_dir", "work_dir"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output.exists() or args.work_dir.exists() or args.asset_dir.exists():
        raise FileExistsError("mechanism output/work/assets are immutable")
    args.work_dir.mkdir(parents=True)
    args.asset_dir.mkdir(parents=True)
    log_path = args.work_dir / "pi05_server.log"

    subprocess.run(
        ["bash", str(ROOT / "scripts/check_gpu_preflight.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "EXPERIMENT_GPU_INDEX": str(args.gpu),
            "EXPERIMENT_ALLOW_LOCAL_RUSTDESK": "1",
        },
        check=True,
    )

    from libero.libero.envs import OffScreenRenderEnv

    task = FrozenRetrievalTaskParser().parse(args.prompt)
    env = OffScreenRenderEnv(
        bddl_file_name=str(args.bddl), camera_heights=256, camera_widths=256
    )
    server = None
    try:
        env.seed(args.seed)
        observation = env.reset()
        for _ in range(args.wait_steps):
            observation, _, _, _ = env.step(DUMMY_ACTION)
        before = build_observation(observation, args.prompt)
        snapshot_row, snapshot_arrays = public_snapshot(
            packet=before,
            prompt=args.prompt,
            target=task.target,
            destination=task.destination,
            seed=args.seed,
            scenario=args.bddl,
            sample_id=f"t01_object_mechanism_seed{args.seed}",
            image_dir=args.work_dir / "images",
        )
        snapshot_path = args.work_dir / "snapshot.jsonl"
        snapshot_path.write_text(json.dumps(snapshot_row, separators=(",", ":")) + "\n")
        scene_path, feature_path, object_assets = run_frontend(
            snapshot=snapshot_path, work_dir=args.work_dir, gpu=args.gpu
        )
        scene = json.loads(scene_path.read_text())
        feature_store = np.load(feature_path)
        object_features = np.asarray(feature_store["features"], dtype=np.float32)
        for view, live in snapshot_arrays.items():
            recorded = imageio.imread(ROOT / scene["source_image_paths"][view])
            if not np.array_equal(live, recorded):
                raise RuntimeError(f"frontend source no longer matches live {view} RGB")

        server_environment = os.environ.copy()
        server_environment.update(
            {
                "EXPERIMENT_GPU_INDEX": str(args.gpu),
                "CUDA_VISIBLE_DEVICES": str(args.gpu),
                "PORT": str(args.port),
            }
        )
        log_handle = log_path.open("w")
        server = subprocess.Popen(
            ["bash", str(ROOT / "scripts/serve_pi05_with_prefix.sh")],
            cwd=ROOT,
            env=server_environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        wait_for_port(args.port)
        policy = OpenPiWebsocketPolicy(host="127.0.0.1", port=args.port)
        if policy.server_metadata.get("prefix_requests_advance_action_rng") is not False:
            raise RuntimeError("prefix requests must not advance action RNG")
        prefix = policy.encode_prefix(before, feature_schema="global_v1")
        runtime = ObjectPIURuntime(args.model)
        initial = runtime.infer(
            prefix_features=prefix,
            scene=scene,
            object_features=object_features,
            target=task.target,
        )
        selected = initial["emitted_action"]
        selected_node = initial["node_uncertainty"][0]["object_id"]
        candidate_ids = {
            "DIRECT_ACT": "direct_act:prompt_target",
            "OPEN_TO_INSPECT": f"open_to_inspect:{selected_node}",
        }
        decision = {
            "selected": {
                "candidate_id": candidate_ids.get(selected, "abstain"),
                "primitive": selected if selected != "SAFE_STOP" else "ABSTAIN",
                "target_id": selected_node if selected != "SAFE_STOP" else None,
                "subtask": OPEN_PROMPT if selected == "OPEN_TO_INSPECT" else args.prompt,
                "stop_condition": "six public observations are available",
                "cost": 0.0 if selected == "SAFE_STOP" else initial["action_effects"][selected]["cost"],
                "physical_risk": 0.0,
            },
            "utilities": {
                candidate_ids[action]: {
                    "expected_information_gain": values["expected_information_gain"],
                    "expected_task_progress": values["expected_task_progress"],
                    "execution_success_probability": values[
                        "execution_reliability_lower_bound"
                    ],
                    "cost": values["cost"],
                    "physical_risk": 0.0,
                    "utility": values["utility"],
                }
                for action, values in initial["action_effects"].items()
            },
            "task_uncertainty": initial["task_uncertainty"],
            "valid_candidate_ids": [*candidate_ids.values(), "abstain"],
            "reason": initial["emission_reason"],
        }
        forecasts = {
            candidate_ids[action]: {
                "outcomes": values["outcome_probabilities"],
                "expected_future_uncertainty": values["expected_future_uncertainty"],
                "execution_success_probability": values[
                    "execution_reliability_lower_bound"
                ],
                "expected_task_progress": values["expected_task_progress"],
            }
            for action, values in initial["action_effects"].items()
        }
        trace = [
            {
                "event": "INITIAL_DECISION",
                "belief": compatible_belief(
                    args.prompt,
                    initial["location_probabilities"],
                    initial["location_prediction_set"],
                    initial["node_uncertainty"],
                    f"piu-object-v2:{runtime.model_sha256[:16]}",
                ),
                "object_runtime": initial,
                "decision": decision,
                "effect_forecasts": forecasts,
            }
        ]
        public_assets = []

        def save_asset(packet, *, name: str, phase: str, step: int) -> None:
            paths, hashes = save_packet_images(
                packet, args.asset_dir / "public_keyframes", name
            )
            public_assets.append(
                {
                    "name": name,
                    "phase": phase,
                    "step": int(step),
                    "image_paths": paths,
                    "image_sha256": hashes,
                    "robot_state": [float(value) for value in packet.state],
                }
            )

        save_asset(before, name="00_before", phase="BEFORE", step=-1)
        outcome_set = None
        outcome_prediction = None
        execution = None
        information_actions: list[list[float]] = []
        recording = RecordingEnvironment(env)
        final_task_evaluator = None
        act_execution = None
        terminal = "INITIAL_SAFE_STOP"
        capture_steps = {
            max(0, round(args.open_steps * fraction) - 1): index
            for index, fraction in enumerate((0.25, 0.50, 0.75, 1.00), start=1)
        }
        if selected == "OPEN_TO_INSPECT":
            rgb_history = [build_observation(observation, OUTCOME_QUERY)]

            def observe_step(phase, step, current_obs):
                if phase == "OPEN" and step in capture_steps:
                    packet = build_observation(current_obs, OUTCOME_QUERY)
                    rgb_history.append(packet)
                    index = capture_steps[step]
                    save_asset(packet, name=f"{index:02d}_open", phase="OPEN", step=step)

            observation, execution = execute_open_and_observe(
                env=recording,
                initial_observation=observation,
                policy=policy,
                open_prompt=OPEN_PROMPT,
                open_steps=args.open_steps,
                replan_steps=args.replan_steps,
                step_observer=observe_step,
            )
            information_actions = list(recording.actions)
            returned = build_observation(observation, OUTCOME_QUERY)
            rgb_history.append(returned)
            save_asset(
                returned,
                name="05_returned",
                phase="AFTER_RETURN",
                step=execution.return_steps,
            )
            if len(rgb_history) != 6:
                outcome_set = ("FAILED", "REVEALED", "EMPTY")
            else:
                outcome_prediction = V12bPublicRGBOutcomeCritic(
                    args.outcome_composite, root=ROOT
                ).predict(rgb_history)
                outcome_set = outcome_prediction.prediction_set
            if outcome_set == ("REVEALED",):
                updated_probabilities = {
                    "visible_workspace": 0.999,
                    "closed_container": 0.001,
                }
                terminal = "INFORMATION_ACQUIRED_AND_DIRECT_HANDOFF"
                updated_set = ["visible_workspace"]
                next_action = "DIRECT_ACT"
            elif outcome_set == ("EMPTY",):
                updated_probabilities = {
                    "visible_workspace": 0.001,
                    "closed_container": 0.001,
                }
                terminal = "LOCAL_EMPTY_SAFE_STOP"
                updated_set = ["other_unsearched_region"]
                next_action = "SAFE_STOP"
            elif outcome_set == ("FAILED",):
                updated_probabilities = initial["location_probabilities"]
                terminal = "ACTION_FAILED_SAFE_STOP"
                updated_set = initial["location_prediction_set"]
                next_action = "SAFE_STOP"
            else:
                updated_probabilities = initial["location_probabilities"]
                terminal = "OUTCOME_AMBIGUOUS_SAFE_STOP"
                updated_set = initial["location_prediction_set"]
                next_action = "SAFE_STOP"
            after_belief = compatible_belief(
                args.prompt,
                updated_probabilities,
                updated_set,
                initial["node_uncertainty"],
                f"v12b-outcome-update:{digest(args.outcome_composite)[:16]}",
            )
            if outcome_set == ("EMPTY",):
                after_belief["facts"][0]["probabilities"] = {
                    "visible_workspace": 0.001,
                    "middle_drawer": 0.0,
                    "other_unsearched_region": 0.999,
                    "absent": 0.0,
                }
            trace.append(
                {
                    "event": "POST_OUTCOME_REPLAN",
                    "belief": after_belief,
                    "decision": {"selected": {"primitive": next_action}},
                }
            )
            if args.execute_final_act and next_action == "DIRECT_ACT":
                action_start = len(recording.actions)
                plan: list[np.ndarray] = []
                capture_act_steps = {
                    max(0, round(args.act_steps * fraction) - 1): index
                    for index, fraction in enumerate((0.25, 0.50, 0.75, 1.00), start=1)
                }
                for act_step in range(args.act_steps):
                    if not plan:
                        chunks = policy.sample_chunks(
                            build_observation(observation, args.prompt), 1
                        )
                        plan.extend(chunks[0][: args.replan_steps])
                    observation, _, _, _ = recording.step(plan.pop(0).tolist())
                    if act_step in capture_act_steps:
                        save_asset(
                            build_observation(observation, args.prompt),
                            name=f"{5 + capture_act_steps[act_step]:02d}_act",
                            phase="DIRECT_ACT",
                            step=act_step,
                        )
                act_execution = {
                    "prompt": args.prompt,
                    "fixed_budget_steps": args.act_steps,
                    "action_start_index": action_start,
                    "task_predicate_used_for_control": False,
                }
                terminal = "FINAL_TASK_EVALUATION_PENDING"
        elif selected == "DIRECT_ACT":
            terminal = "DIRECT_ACT_HANDOFF"

        action_path = args.asset_dir / "public_action_history.json"
        action_path.write_text(json.dumps(recording.actions, separators=(",", ":")) + "\n")
        evaluator = replay_evaluator_trace(
            bddl=args.bddl,
            seed=args.seed,
            wait_steps=args.wait_steps,
            actions=information_actions,
            open_history_steps=capture_steps,
            counterpart_target_qpos={},
        )
        if act_execution is not None:
            final_task_evaluator = replay_final_task_evaluator(
                bddl=args.bddl,
                seed=args.seed,
                wait_steps=args.wait_steps,
                actions=recording.actions,
            )
            terminal = (
                "FINAL_TASK_SUCCESS"
                if final_task_evaluator["task_success"]
                else "FINAL_TASK_FAILED_AFTER_INFORMATION_ACQUIRED"
            )
        trace.append(
            {
                "event": "OUTCOME",
                "prediction_set": list(outcome_set) if outcome_set else None,
                "public_rgb_critic": (
                    outcome_prediction.to_dict() if outcome_prediction is not None else None
                ),
                "terminal": terminal,
            }
        )
        row = {
            "case": "hidden_butter",
            "seed": args.seed,
            "scenario": str(args.bddl.relative_to(ROOT)),
            "prompt": args.prompt,
            "initial_expected": "OPEN_TO_INSPECT",
            "initial_selected": selected,
            "initial_correct": selected == "OPEN_TO_INSPECT",
            "outcome_expected": "REVEALED",
            "outcome_prediction_set": list(outcome_set) if outcome_set else None,
            "outcome_correct": outcome_set == ("REVEALED",),
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
            "assets": {
                "public_keyframes": public_assets,
                "public_action_history": {
                    "path": str(action_path.relative_to(ROOT)),
                    "sha256": digest(action_path),
                    "steps": len(recording.actions),
                },
                "object_overlays": scene["overlay_paths"],
                "object_asset_root": str(object_assets.relative_to(ROOT)),
                "global_view_available_to_controller": False,
            },
            "evaluator_only": {
                "revealed_any_history": any(
                    max(point["target_pixels"].values()) >= TARGET_PIXELS
                    for point in evaluator["visibility_history"]
                ),
                "visibility_history": evaluator["visibility_history"],
                "final_task": final_task_evaluator,
            },
            "online_oracle_inputs": [],
        }
        report = {
            "schema_version": "interaction-uncertainty.piu-object-mechanism.v1",
            "claim_status": "disposable mechanism run; paper evaluation still requires frozen held-out initial-decision data",
            "endpoint": "prompt-relevant information acquisition and semantic DIRECT_ACT replan; final task reported separately",
            "repository_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "model": str(args.model.relative_to(ROOT)),
            "model_sha256": runtime.model_sha256,
            "policy": "frozen pi05_libero action + prefix server",
            "outcome_critic": {
                "path": str(args.outcome_composite.relative_to(ROOT)),
                "sha256": digest(args.outcome_composite),
                "online_oracle_inputs": [],
            },
            "seed": args.seed,
            "cases": 1,
            "initial_correct": int(row["initial_correct"]),
            "outcome_correct": int(row["outcome_correct"]),
            "information_acquisition_successes": int(
                terminal
                in {
                    "INFORMATION_ACQUIRED_AND_DIRECT_HANDOFF",
                    "FINAL_TASK_SUCCESS",
                    "FINAL_TASK_FAILED_AFTER_INFORMATION_ACQUIRED",
                }
            ),
            "physical_final_act_executed": args.execute_final_act,
            "final_task_successes": int(
                bool((final_task_evaluator or {}).get("task_success"))
            ),
            "online_oracle_inputs": [],
            "asset_root": str(args.asset_dir.relative_to(ROOT)),
            "rows": [row],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "initial_selected": selected,
                    "outcome_set": list(outcome_set) if outcome_set else None,
                    "terminal": terminal,
                    "information_acquisition_successes": report[
                        "information_acquisition_successes"
                    ],
                    "final_task_successes": report["final_task_successes"],
                },
                indent=2,
            ),
            flush=True,
        )
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
        env.close()


if __name__ == "__main__":
    main()
