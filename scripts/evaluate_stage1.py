"""Paired development rollouts; not the controlled geometry/history benchmark."""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["HF_MODULES_CACHE"] = str(ROOT / ".cache/huggingface/modules")
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
from gpu_devices import verify_cuda_target
from libero_protocol import validate_feedback
from paired_observations import align_initial_observation

from grounded_interaction.predictive_vla.backend import NativeVLABackend
from grounded_interaction.predictive_vla.config import load_config, set_manual_seed
from grounded_interaction.predictive_vla.training import load_checkpoint
from grounded_interaction.predictive_vla.types import (
    AppliedAction,
    CameraFrame,
    Observation,
    PolicyContext,
)

TASKS = ("open_the_middle_drawer_of_the_cabinet", "put_the_bowl_on_the_plate")


def inside(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("path escapes repository")
    return path


def save_report(path, report):
    temporary = inside(path.with_suffix(".partial"))
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_observation(packet, destination):
    path = inside(ROOT / packet["path"])
    if not path.is_relative_to(destination):
        raise ValueError("observation escaped episode directory")
    with np.load(path, allow_pickle=False) as arrays:
        cameras = tuple(
            CameraFrame(
                *(arrays[f"{view}_{key}"] for key in ("rgb", "depth", "K", "T_world"))
            )
            for view in ("agent", "wrist")
        )
        return Observation(packet["step"], cameras, arrays["states"])


def rollout(
    backend, policy, task, episode, output, seed, max_steps, *, diagnostic_case=None,
    physical_gpu=7, reference_output=None,
):
    destination = inside(output / policy / task / episode)
    destination.mkdir(parents=True, exist_ok=False)
    source = ROOT / f"data/libero_raw/libero_goal/{task}_demo.hdf5"
    command = [str(ROOT / ".venv-sim/bin/python"), "-u"]
    if diagnostic_case is None:
        command += [
            str(ROOT / "scripts/libero_probe_server.py"),
            "--file",
            str(source),
            "--episode",
            episode,
            "--manual-seed",
            str(seed),
        ]
    else:
        command += [
            str(ROOT / "scripts/libero_diagnostic_server.py"),
            "--case",
            str(inside(diagnostic_case)),
        ]
    command += ["--output", str(destination), "--physical-gpu", str(physical_gpu)]
    child_env = dict(
        os.environ,
        MUJOCO_GL="egl",
        MUJOCO_EGL_DEVICE_ID=str(physical_gpu),
        PYOPENGL_PLATFORM="egl",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    backend.reset()
    torch.cuda.reset_peak_memory_stats()
    context, requested_count = None, 0
    latencies, started = [], time.monotonic()
    with (
        (destination / "simulator.stderr.log").open("w") as errors,
        (destination / "rollout.jsonl").open("w") as records,
    ):
        child = subprocess.Popen(
            command,
            env=child_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
        )
        try:
            while True:
                line = child.stdout.readline()
                if not line:
                    raise RuntimeError(f"simulator stopped: {errors.name}")
                if not line.startswith("IP_OBSERVATION:"):
                    continue
                packet = json.loads(line.removeprefix("IP_OBSERVATION:"))
                observation = read_observation(packet, destination)
                if context is None:
                    if (diagnostic_case is None and packet["step"] != 0) or packet[
                        "applied_actions"
                    ]:
                        raise ValueError("invalid initial observation")
                    initial_step = packet["step"]
                    history = packet.get("initial_history")
                    previous_observations = (
                        tuple(
                            read_observation(p, destination)
                            for p in history["observations"]
                        )
                        if history
                        else ()
                    )
                    previous_actions = (
                        tuple(
                            AppliedAction(a["step"], a["value"])
                            for a in history["applied_actions"]
                        )
                        if history
                        else ()
                    )
                    context = PolicyContext(
                        packet["task"],
                        (*previous_observations, observation),
                        previous_actions,
                        max_steps,
                    )
                    # Compare arrays directly; paired resets must be identical.
                    reference = (
                        (output if reference_output is None else inside(reference_output))
                        / "native"
                        / task
                        / episode
                        / f"observation_{initial_step:04d}.npz"
                    )
                    if policy != "native":
                        align_initial_observation(reference, ROOT / packet["path"])
                        observation = read_observation(packet, destination)
                        context = PolicyContext(
                            packet["task"], (*previous_observations, observation),
                            previous_actions, max_steps,
                        )
                        if history:
                            from audit_diagnostic_scenes import arrays_equal

                            for past in history["observations"]:
                                if not arrays_equal(
                                    ROOT / past["path"],
                                    reference.parent / Path(past["path"]).name,
                                ):
                                    raise ValueError(
                                        "paired historical observations differ"
                                    )
                            with (
                                reference.parent / "rollout.jsonl"
                            ).open() as baseline:
                                initial_packet = json.loads(baseline.readline())[
                                    "observation"
                                ]
                            if (
                                history["applied_actions"]
                                != initial_packet["initial_history"]["applied_actions"]
                            ):
                                raise ValueError("paired executed history differs")
                else:
                    actual = validate_feedback(
                        packet,
                        previous_step=context.current.step,
                        requested_count=requested_count,
                    )
                    context = context.advance(
                        observation,
                        tuple(
                            AppliedAction(context.current.step + i, a)
                            for i, a in enumerate(actual)
                        ),
                        max_frames=backend.config.history_frames,
                    )
                records.write(json.dumps({"observation": packet}) + "\n")
                records.flush()
                # Only this evaluator reads success, never the policy or prompt.
                if packet["evaluation_success"] or not context.remaining_steps:
                    return {
                        "policy": policy,
                        "task": task,
                        "episode": episode,
                        "manual_seed": seed,
                        "success": packet["evaluation_success"],
                        "steps": packet["step"],
                        "takeover_step": initial_step,
                        "policy_control_steps": packet["step"] - initial_step,
                        "first_choice": packet.get("evaluation_first_choice"),
                        "policy_calls": len(latencies),
                        "wall_seconds": time.monotonic() - started,
                        "policy_seconds_median": float(np.median(latencies))
                        if latencies
                        else None,
                        "policy_seconds_p95": float(np.quantile(latencies, 0.95))
                        if latencies
                        else None,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                        "paired_initial_observation_equal": True
                        if policy != "native"
                        else None,
                    }
                action_seed = seed + context.current.step
                start = time.monotonic()
                act = backend.native_act if policy == "native" else backend.act
                policy_context = context
                if policy in {"native", "best_current"}:
                    policy_context = PolicyContext(
                        context.task, (context.current,), (), context.remaining_steps
                    )
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    actions = act(
                        policy_context,
                        seed=action_seed,
                        steps=min(
                            backend.config.prediction_steps, context.remaining_steps
                        ),
                    )
                torch.cuda.synchronize()
                latency = time.monotonic() - start
                latencies.append(latency)
                prefix = actions[: backend.config.execute_steps]
                requested_count = len(prefix)
                records.write(
                    json.dumps(
                        {
                            "step": context.current.step,
                            "manual_seed": action_seed,
                            "requested_actions": prefix.tolist(),
                            "policy_seconds": latency,
                            "history_observation_steps": [
                                o.step for o in context.observations
                            ],
                            "history_applied_steps": [
                                a.step for a in context.applied_actions
                            ],
                            "policy_input_observation_steps": [
                                o.step for o in policy_context.observations
                            ],
                        }
                    )
                    + "\n"
                )
                records.flush()
                child.stdin.write(json.dumps({"actions": prefix.tolist()}) + "\n")
                child.stdin.flush()
                if len(latencies) % 10 == 0:
                    print(
                        json.dumps(
                            {
                                "policy": policy,
                                "task": task,
                                "episode": episode,
                                "observed_step": context.current.step,
                            }
                        ),
                        flush=True,
                    )
        finally:
            if child.poll() is None:
                try:
                    child.stdin.write('{"close": true}\n')
                    child.stdin.flush()
                    child.wait(timeout=30)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    child.terminate()
                    child.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manual-seed", type=int, default=117)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--episodes", nargs="+", default=["demo_0", "demo_1"])
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["native", "best"],
        choices=["native", "best", "last"],
    )
    args = parser.parse_args()
    if args.policies[0] != "native" or len(set(args.policies)) != len(args.policies):
        raise ValueError(
            "native baseline must run first, before any trained weights are loaded"
        )
    if args.max_steps < 1 or any(e not in {"demo_0", "demo_1"} for e in args.episodes):
        raise ValueError("invalid development episode budget")
    mapping = verify_cuda_target(7)
    config = load_config(ROOT / "experiments/stage1_policy.yaml")
    set_manual_seed(config.manual_seed)
    torch.set_num_threads(4)
    output = inside(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    report = {
        "kind": "paired development integration check; not a benchmark estimate",
        "manual_seed": args.manual_seed,
        "training_manual_seed": config.manual_seed,
        "device": mapping,
        "policies": args.policies,
        "planned_episodes": args.episodes,
        "max_steps": args.max_steps,
        "prediction_steps": config.prediction_steps,
        "execute_steps": config.execute_steps,
        "flow_steps": config.flow_steps,
        "batch": 1,
        "language_stop": False,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "checkpoint_selection": "best chosen by the fixed 64-window validation loss only",
        "checkpoints": {},
        "episodes": [],
        "complete": False,
    }
    if args.resume:
        previous = json.loads((output / "report.json").read_text())
        for key in (
            "manual_seed",
            "training_manual_seed",
            "device",
            "policies",
            "planned_episodes",
            "max_steps",
            "prediction_steps",
            "execute_steps",
            "flow_steps",
            "batch",
            "language_stop",
        ):
            if previous[key] != report[key]:
                raise ValueError(f"resume changes evaluation setting: {key}")
        if previous["complete"]:
            raise ValueError("evaluation is already complete")
        report = previous
        report.setdefault("resumed_utc", []).append(
            datetime.now(timezone.utc).isoformat()
        )
        report["pid"] = os.getpid()
    save_report(output / "report.json", report)
    backend = NativeVLABackend.from_pretrained(
        config, device="cuda:0", local_path=ROOT / "checkpoints/base/MolmoAct2-LIBERO"
    )
    # Match training precision and avoid rounding FP32 saved trainable weights.
    for parameter in backend.model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    backend.train(False)
    for policy in args.policies:
        completed = {
            (r["task"], r["episode"])
            for r in report["episodes"]
            if r["policy"] == policy
        }
        if completed == {
            (task, episode) for task in TASKS for episode in args.episodes
        }:
            continue
        if policy != "native":
            path = ROOT / "runs/stage1_s17_gpu7" / f"{policy}.pt"
            payload = load_checkpoint(path, backend)
            parameters = backend.model.state_dict()
            if any(
                not torch.equal(parameters[name].detach().cpu(), value)
                for name, value in payload["adapters"].items()
            ):
                raise RuntimeError(
                    "checkpoint restoration changed saved parameter values"
                )
            report["checkpoints"][policy] = {
                "path": str(path.relative_to(ROOT)),
                "updates": payload["updates"],
                "manual_seed": payload["manual_seed"],
                "exact_parameter_restore": True,
            }
            del payload
            save_report(output / "report.json", report)
        for task_index, task in enumerate(TASKS):
            for episode in args.episodes:
                if (task, episode) in completed:
                    continue
                seed = (
                    args.manual_seed
                    + 1000 * task_index
                    + 100 * int(episode.removeprefix("demo_"))
                )
                row = rollout(
                    backend, policy, task, episode, output, seed, args.max_steps
                )
                report["episodes"].append(row)
                save_report(output / "report.json", report)
                print(json.dumps(row), flush=True)
    report["complete"] = True
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save_report(output / "report.json", report)


if __name__ == "__main__":
    main()
