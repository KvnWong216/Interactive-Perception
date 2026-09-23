"""Controlled parallel replay for native, no-MSAA, or dither-off rendering.

Every worker repeats the exact requested cases with their original manual seeds.
Failures remain visible; there is no tolerance adjustment or replacement case.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--case-ids", nargs="+", type=int, required=True)
    p.add_argument("--physical-gpu", type=int, required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument(
        "--profile", choices=["native", "no_msaa", "dither_off"], required=True
    )
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    source, out = (ROOT / a.source).resolve(), (ROOT / a.output).resolve()
    if any(not x.is_relative_to(ROOT) for x in [source, out]) or a.workers < 1:
        raise ValueError("repository paths and positive workers required")
    original = json.loads((source / "plan.json").read_text())
    lookup = {c["case_id"]: c for c in original["cases"]}
    cases = [
        {**lookup[case_id], "case_id": i, "original_case_id": case_id}
        for i, case_id in enumerate(a.case_ids)
    ]
    plan = {
        **original,
        "cases": cases,
        "render_profile": "native" if a.profile == "dither_off" else a.profile,
        "diagnostic_intervention": a.profile,
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if a.worker:
        from prepare_libero import configure

        configure()
        from gpu_devices import configure_render_gpu

        configure_render_gpu(a.physical_gpu)
        if a.profile == "dither_off":
            from OpenGL import GL
            from robosuite.utils.binding_utils import MjSim

            original_render = MjSim.render

            def render(self, *args, **kwargs):
                self._render_context_offscreen.gl_ctx.make_current()
                GL.glDisable(GL.GL_DITHER)
                return original_render(self, *args, **kwargs)

            MjSim.render = render
        from collect_predictor_forks import collect

        collect(out, 0, a.physical_gpu, 1)
        return
    jobs = []
    started = time.monotonic()
    env = dict(
        os.environ,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=str(a.physical_gpu),
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    try:
        for i in range(a.workers):
            command = [
                str(ROOT / ".venv-sim/bin/python"),
                "-u",
                str(Path(__file__).resolve()),
                "--worker",
                "--source",
                str(source),
                "--output",
                str(out / f"worker_{i}"),
                "--case-ids",
                *map(str, a.case_ids),
                "--physical-gpu",
                str(a.physical_gpu),
                "--profile",
                a.profile,
            ]
            with (out / f"worker_{i}.log").open("x") as log:
                jobs.append(
                    subprocess.Popen(
                        command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                )
        while True:
            codes = [child.poll() for child in jobs]
            done = all(code is not None for code in codes)
            (out / "status.json").write_text(
                json.dumps(
                    {
                        "exit_codes": codes,
                        "complete": done,
                        "passed": done and all(code == 0 for code in codes),
                        "seconds": time.monotonic() - started,
                    },
                    indent=2,
                )
                + "\n"
            )
            if done:
                break
            time.sleep(3)
    except BaseException:
        for child in jobs:
            if child.poll() is None:
                child.terminate()
        for child in jobs:
            child.wait()
        raise
    if any(code != 0 for code in codes):
        raise SystemExit(
            "Replay failed; inspect preserved worker logs and mismatch arrays."
        )


if __name__ == "__main__":
    main()
