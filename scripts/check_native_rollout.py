"""Two native LIBERO rollouts to verify the real observation/action loop."""

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["HF_MODULES_CACHE"] = str(ROOT / ".cache/huggingface/modules")
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch

from grounded_interaction.predictive_vla.backend import NativeVLABackend
from grounded_interaction.predictive_vla.config import load_config
from grounded_interaction.predictive_vla.types import (
    CameraFrame,
    Observation,
    PolicyContext,
)


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
        raise RuntimeError("native rollout must use physical GPU 7")
    config = load_config(ROOT / "experiments/stage1_policy.yaml")
    torch.set_num_threads(4)
    backend = NativeVLABackend.from_pretrained(
        config, device="cuda:0", local_path=ROOT / "checkpoints/base/MolmoAct2-LIBERO"
    )
    source = (
        ROOT
        / "data/libero_raw/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5"
    )
    output = ROOT / "data/preparation/native_rollouts"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "kind": "two-episode integration check; not a benchmark estimate",
        "manual_seed": config.manual_seed,
        "gpu_physical": 7,
        "episodes": [],
    }
    child_env = dict(
        os.environ,
        MUJOCO_GL="egl",
        MUJOCO_EGL_DEVICE_ID="7",
        PYOPENGL_PLATFORM="egl",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    for episode in ("demo_0", "demo_1"):
        destination = output / episode
        latencies, started = [], time.monotonic()
        with (output / f"{episode}.stderr.log").open("w") as errors:
            child = subprocess.Popen(
                [
                    str(ROOT / ".venv-sim/bin/python"),
                    "-u",
                    str(ROOT / "scripts/libero_probe_server.py"),
                    "--file",
                    str(source),
                    "--episode",
                    episode,
                    "--output",
                    str(destination),
                ],
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
                    # Only the evaluator reads the success field.
                    if packet["evaluation_success"] or packet["step"] >= 300:
                        row = {
                            "episode": episode,
                            "steps": packet["step"],
                            "success": packet["evaluation_success"],
                            "wall_seconds": time.monotonic() - started,
                            "policy_seconds_median": float(np.median(latencies))
                            if latencies
                            else None,
                            "policy_seconds_p95": float(np.quantile(latencies, 0.95))
                            if latencies
                            else None,
                        }
                        report["episodes"].append(row)
                        print(json.dumps(row), flush=True)
                        break
                    path = (ROOT / packet["path"]).resolve()
                    if not path.is_relative_to(destination.resolve()):
                        raise ValueError(
                            "simulator observation path escaped episode directory"
                        )
                    with np.load(path, allow_pickle=False) as arrays:
                        cameras = tuple(
                            CameraFrame(
                                *(
                                    arrays[f"{view}_{key}"]
                                    for key in ("rgb", "depth", "K", "T_world")
                                )
                            )
                            for view in ("agent", "wrist")
                        )
                        observation = Observation(
                            packet["step"], cameras, arrays["states"]
                        )
                    context = PolicyContext(
                        packet["task"], (observation,), (), 300 - packet["step"]
                    )
                    start = time.monotonic()
                    actions = backend.native_act(
                        context,
                        seed=config.manual_seed + packet["step"],
                        steps=min(config.prediction_steps, context.remaining_steps),
                    )
                    latencies.append(time.monotonic() - start)
                    prefix = actions[: config.execute_steps]
                    child.stdin.write(json.dumps({"actions": prefix.tolist()}) + "\n")
                    child.stdin.flush()
                    with (output / f"{episode}.actions.jsonl").open("a") as stream:
                        stream.write(
                            json.dumps(
                                {"step": packet["step"], "actions": prefix.tolist()}
                            )
                            + "\n"
                        )
            finally:
                if child.poll() is None:
                    child.stdin.write('{"close": true}\n')
                    child.stdin.flush()
                    child.wait(timeout=30)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    report["integration_check_complete"] = True
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
