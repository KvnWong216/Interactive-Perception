#!/usr/bin/env python3
"""Execute a Qwen-selected OPEN_AND_OBSERVE with fresh frozen pi0.5 actions.

This disposable runner starts from a paired development state, verifies that
its public RGB exactly matches the source images used by the initial PIU
report, executes only the report's registered OPEN_CONTAINER action, and then
classifies six public observations.  Simulator-private scoring happens in a
separate replay after the controller and outcome critic have terminated.
"""

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
from typing import Any

import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from collect_t01_action_effect_transitions import (  # noqa: E402
    MIDDLE_JOINT,
    policy_visibility,
)
from collect_t01_open_and_observe_effect import RecordingEnvironment  # noqa: E402
from interactive_perception.action_options import execute_open_and_observe  # noqa: E402
from interactive_perception.policy_client import (  # noqa: E402
    OpenPiWebsocketPolicy,
    ObservationPacket,
    build_observation,
)
from interactive_perception.rgb_outcome_critic import (  # noqa: E402
    V13ComplementaryPublicRGBOutcomeCritic,
)


OPEN_PROMPT = "Open the middle layer of the drawer"


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


def save_packet(packet: ObservationPacket, directory: Path, name: str) -> dict[str, Any]:
    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for view, image in (("agentview", packet.image), ("wrist", packet.wrist_image)):
        path = directory / f"{name}_{view}.png"
        imageio.imwrite(path, image)
        paths[view] = str(path.relative_to(ROOT))
        hashes[view] = digest(path)
    return {
        "name": name,
        "image_paths": paths,
        "image_sha256": hashes,
        "public_robot_state": [float(value) for value in packet.state],
    }


def split_frame(packet: ObservationPacket) -> np.ndarray:
    separator = np.full((packet.image.shape[0], 4, 3), 255, dtype=np.uint8)
    return np.concatenate((packet.wrist_image, separator, packet.image), axis=1)


def evaluator_replay(
    *,
    bddl: Path,
    seed: int,
    initial_state: np.ndarray,
    actions: list[list[float]],
    capture_indices: dict[int, str],
) -> dict[str, Any]:
    """Replay privileged diagnostics only after controller termination."""

    from libero.libero.envs import SegmentationRenderEnv

    env = SegmentationRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
    )
    history: list[dict[str, Any]] = []
    try:
        env.seed(seed)
        env.reset()
        observation = env.set_init_state(initial_state)

        def capture(name: str) -> None:
            history.append(
                {"name": name, "target_pixels": policy_visibility(env, observation)}
            )

        capture("00_before")
        minimum_joint = float(env.env.sim.data.get_joint_qpos(MIDDLE_JOINT))
        for index, action in enumerate(actions):
            observation, _, done, _ = env.step(action)
            minimum_joint = min(
                minimum_joint,
                float(env.env.sim.data.get_joint_qpos(MIDDLE_JOINT)),
            )
            if index in capture_indices:
                capture(capture_indices[index])
            if done and index != len(actions) - 1:
                raise RuntimeError("evaluator terminated before public trace ended")
        capture("05_returned")
        if len(history) != 6:
            raise RuntimeError(f"expected six evaluator points, got {len(history)}")
        return {
            "timing": "separate replay after controller and critic terminal",
            "privileged_inputs": ["target segmentation", "drawer joint"],
            "visibility_history": history,
            "maximum_target_pixels": max(
                max(row["target_pixels"].values()) for row in history
            ),
            "middle_joint_minimum": minimum_joint,
            "middle_joint_final": float(
                env.env.sim.data.get_joint_qpos(MIDDLE_JOINT)
            ),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-report", type=Path, required=True)
    parser.add_argument("--bddl", type=Path, required=True)
    parser.add_argument("--initial-state-npz", type=Path, required=True)
    parser.add_argument("--initial-state-key", default="closed_easy")
    parser.add_argument("--outcome-composite", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1399)
    parser.add_argument("--open-steps", type=int, default=300)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "initial_report",
        "bddl",
        "initial_state_npz",
        "outcome_composite",
        "asset_dir",
        "work_dir",
        "output",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.asset_dir.exists() or args.work_dir.exists() or args.output.exists():
        raise FileExistsError("fresh option outputs are immutable")

    initial = json.loads(args.initial_report.read_text())
    if initial.get("online_oracle_inputs"):
        raise ValueError("initial decision report declares online oracle inputs")
    decision = initial["selected_action"]
    if decision["action"] != "OPEN_CONTAINER":
        raise ValueError("runner only executes a Qwen-selected OPEN_CONTAINER")
    prompt = initial["prompt"]
    with np.load(args.initial_state_npz) as store:
        if args.initial_state_key not in store:
            raise KeyError(args.initial_state_key)
        initial_state = np.asarray(store[args.initial_state_key], dtype=np.float64)

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

    args.asset_dir.mkdir(parents=True)
    args.work_dir.mkdir(parents=True)
    server = None
    log_handle = None
    env = OffScreenRenderEnv(
        bddl_file_name=str(args.bddl), camera_heights=256, camera_widths=256
    )
    writer = None
    try:
        env.seed(args.seed)
        env.reset()
        observation = env.set_init_state(initial_state)
        before = build_observation(observation, prompt)
        expected_sources = initial["scene_packet"]
        scene = json.loads((ROOT / expected_sources["path"]).read_text())
        for view, live in (("agentview", before.image), ("wrist", before.wrist_image)):
            expected = np.asarray(imageio.imread(ROOT / scene["source_images"][view]))
            if not np.array_equal(live, expected):
                raise RuntimeError(f"live paired state does not match initial {view} RGB")

        server_environment = {
            **os.environ,
            "EXPERIMENT_GPU_INDEX": str(args.gpu),
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PORT": str(args.port),
        }
        log_handle = (args.work_dir / "pi05_server.log").open("w")
        server = subprocess.Popen(
            ["bash", str(ROOT / "scripts/serve_pi05.sh")],
            cwd=ROOT,
            env=server_environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        wait_for_port(args.port)
        policy = OpenPiWebsocketPolicy(host="127.0.0.1", port=args.port)
        metadata = policy.server_metadata

        video_path = args.asset_dir / "fresh_open_and_observe_wrist_agentview.mp4"
        writer = imageio.get_writer(video_path, fps=20)
        keyframe_dir = args.asset_dir / "public_keyframes"
        keyframe_dir.mkdir()
        keyframes = [save_packet(before, keyframe_dir, "00_before")]
        history = [before]
        capture_steps = {
            max(0, round(args.open_steps * fraction) - 1): (index, name)
            for index, (fraction, name) in enumerate(
                (
                    (0.25, "01_quarter"),
                    (0.50, "02_half"),
                    (0.75, "03_three_quarters"),
                    (1.00, "04_open_end"),
                ),
                start=1,
            )
        }
        recording = RecordingEnvironment(env)

        def observe_step(phase: str, step: int, current: dict[str, Any]) -> None:
            packet = build_observation(current, prompt)
            if step % 2 == 0:
                writer.append_data(split_frame(packet))
            if phase == "OPEN" and step in capture_steps:
                _, name = capture_steps[step]
                history.append(packet)
                keyframes.append(save_packet(packet, keyframe_dir, name))

        observation, execution = execute_open_and_observe(
            env=recording,
            initial_observation=observation,
            policy=policy,
            open_prompt=OPEN_PROMPT,
            open_steps=args.open_steps,
            replan_steps=args.replan_steps,
            step_observer=observe_step,
        )
        returned = build_observation(observation, prompt)
        history.append(returned)
        keyframes.append(save_packet(returned, keyframe_dir, "05_returned"))
        if len(history) != 6:
            raise RuntimeError(f"public outcome history has {len(history)} points")
        outcome = V13ComplementaryPublicRGBOutcomeCritic(
            args.outcome_composite, root=ROOT
        ).predict(history)
        if outcome.prediction_set == ("REVEALED",):
            terminal = "INFORMATION_ACQUIRED"
            public_outcome = "EVIDENCE_ACQUIRED"
        elif len(outcome.prediction_set) == 1:
            terminal = f"OUTCOME_{outcome.prediction_set[0]}"
            public_outcome = outcome.prediction_set[0]
        else:
            terminal = "SAFE_STOP_AMBIGUOUS_OUTCOME"
            public_outcome = "AMBIGUOUS"

        action_path = args.asset_dir / "public_action_history.json"
        action_path.write_text(json.dumps(recording.actions, separators=(",", ":")) + "\n")
        open_capture_indices = {
            step: name for step, (_, name) in capture_steps.items()
        }
        evaluator = evaluator_replay(
            bddl=args.bddl,
            seed=args.seed,
            initial_state=initial_state,
            actions=recording.actions,
            capture_indices=open_capture_indices,
        )
        report = {
            "schema_version": "interaction-uncertainty.qwen-selected-fresh-open.v1",
            "claim_status": "disposable development fresh pi0.5 option run; not clean or sealed",
            "prompt": prompt,
            "initial_decision": {
                "report": {
                    "path": str(args.initial_report.relative_to(ROOT)),
                    "sha256": digest(args.initial_report),
                },
                "action": decision["action"],
                "target_id": decision["target_id"],
            },
            "paired_initial_state": {
                "path": str(args.initial_state_npz.relative_to(ROOT)),
                "sha256": digest(args.initial_state_npz),
                "key": args.initial_state_key,
                "constructed_before_controller_start": True,
                "public_rgb_matched_initial_report": True,
            },
            "controller": {
                "policy": "frozen pi05_libero",
                "fresh_policy_sampling": True,
                "fresh_policy_server": True,
                "server_metadata": metadata,
                "semantic_subtask": OPEN_PROMPT,
                "open_steps": execution.open_steps,
                "return_steps": execution.return_steps,
                "return_phase": execution.return_status.phase.value,
                "keyframes": keyframes,
                "video": str(video_path.relative_to(ROOT)),
                "action_history": {
                    "path": str(action_path.relative_to(ROOT)),
                    "sha256": digest(action_path),
                    "steps": len(recording.actions),
                },
                "online_inputs": [
                    "stock agentview RGB",
                    "wrist RGB",
                    "public proprioception",
                    "semantic subtask",
                ],
                "online_oracle_inputs": [],
            },
            "outcome": {
                "camera_fusion": "v13-complementary",
                "composite": {
                    "path": str(args.outcome_composite.relative_to(ROOT)),
                    "sha256": digest(args.outcome_composite),
                },
                "prediction": outcome.to_dict(),
                "public_outcome_for_belief_update": public_outcome,
                "online_oracle_inputs": [],
            },
            "terminal": terminal,
            "target_observability_success": terminal == "INFORMATION_ACQUIRED",
            "evaluator_only_after_controller_terminal": evaluator,
            "online_oracle_inputs": [],
            "limitations": [
                "development/disposable run",
                "v13 camera composition is not clean-validated",
                "return-to-observe uses the current proprioceptive recovery controller",
                "final task manipulation is not executed by this option runner",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "prediction_set": list(outcome.prediction_set),
                    "terminal": terminal,
                    "fresh_policy_sampling": True,
                    "return_steps": execution.return_steps,
                },
                indent=2,
            ),
            flush=True,
        )
    finally:
        if writer is not None:
            writer.close()
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
        if log_handle is not None:
            log_handle.close()
        env.close()


if __name__ == "__main__":
    main()
