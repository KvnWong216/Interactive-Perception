"""Own two encoding workers, then eight head workers; persist failures and results."""

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/stage2_real_debug_s6227"
PYTHON = ROOT / ".venv/bin/python"


def launch(mode, index, gpu):
    env = dict(
        os.environ,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=str(gpu),
        HF_HOME=str(ROOT / ".cache/huggingface"),
        HF_MODULES_CACHE=str(ROOT / ".cache/huggingface/modules"),
        HF_HUB_OFFLINE="1",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    with (OUT / f"{mode}_{index}.log").open("x") as log:
        p = subprocess.Popen(
            [
                str(PYTHON),
                "-u",
                str(ROOT / "scripts/run_real_predictor_debug.py"),
                mode,
                "--output",
                str(OUT),
                "--id",
                str(index),
                "--physical-gpu",
                str(gpu),
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return p


def main():
    state = {
        "supervisor_pid": os.getpid(),
        "physical_gpus": [2, 3],
        "phase": "encoding",
        "complete": False,
    }
    for phase, workers in [
        ("encoding", [(i, g) for i, g in enumerate([2, 3])]),
        ("head_training", [(i, 2 + i % 2) for i in range(8)]),
    ]:
        state["phase"] = phase
        children = [
            launch("encode" if phase == "encoding" else "worker", i, g)
            for i, g in workers
        ]
        state["pids"] = [p.pid for p in children]
        while True:
            codes = [p.poll() for p in children]
            state["exit_codes"] = codes
            state["completed_jobs"] = sum(
                json.loads(p.read_text()).get("complete", False)
                for p in (OUT / "jobs").glob("job_*.json")
            )
            temp = OUT / "status.partial"
            temp.write_text(json.dumps(state, indent=2) + "\n")
            temp.replace(OUT / "status.json")
            if any(code not in [None, 0] for code in codes):
                # Stop only children owned by this experiment, not unrelated work.
                for p in children:
                    if p.poll() is None:
                        p.terminate()
                for p in children:
                    p.wait()
                raise RuntimeError(f"{phase} worker failed; inspect logs")
            if all(code == 0 for code in codes):
                break
            time.sleep(5)
    rows = [
        json.loads(p.read_text()) for p in sorted((OUT / "jobs").glob("job_*.json"))
    ]
    (OUT / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    state.update(complete=True, phase="complete", completed_jobs=len(rows))
    (OUT / "status.json").write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
