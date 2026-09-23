"""Small process supervisor shared by repository-owned diagnostic experiments."""

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs"
STARTED = time.monotonic()


def write(name, value):
    path = OUT / name
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def launch(name, gpu, script, args, *, simulator=False):
    env = dict(
        os.environ,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=str(gpu),
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        HF_HUB_OFFLINE="1",
        HF_HOME=str(ROOT / ".cache/huggingface"),
        HF_MODULES_CACHE=str(ROOT / ".cache/huggingface/modules"),
    )
    python = ROOT / (".venv-sim/bin/python" if simulator else ".venv/bin/python")
    command = [str(python), "-u", str(ROOT / "scripts" / script), *map(str, args)]
    with (OUT / f"{name}.log").open("x") as stream:
        child = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
    return (
        name,
        child,
        {"name": name, "pid": child.pid, "physical_gpu": gpu, "command": command},
    )


def wait_phase(state, jobs):
    state["workers"] = [record for _, _, record in jobs]
    try:
        while True:
            state["exit_codes"] = {name: child.poll() for name, child, _ in jobs}
            state["elapsed_seconds"] = time.monotonic() - state.pop("_started", STARTED)
            write("status.json", state)
            if any(code not in (None, 0) for code in state["exit_codes"].values()):
                raise RuntimeError(
                    f"{state['phase']} failed; inspect the named worker log"
                )
            if all(code == 0 for code in state["exit_codes"].values()):
                write(f"{state['phase']}_complete.json", state)
                return
            time.sleep(3)
    except BaseException:
        for _, child, _ in jobs:
            if child.poll() is None:
                child.terminate()
        for _, child, _ in jobs:
            child.wait()
        raise


def encode(name, shard, gpu, source, destination):
    return launch(
        name,
        gpu,
        "encode_predictor_forks.py",
        [
            "--shard",
            shard,
            "--physical-gpu",
            gpu,
            "--source",
            source,
            "--features-directory",
            destination,
            "--checkpoint",
            "runs/stage1_s17_gpu7/best.pt",
        ],
    )
