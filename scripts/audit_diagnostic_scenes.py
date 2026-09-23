"""Audit causal history, paired physical resets and calibration without a policy."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def arrays_equal(a, b, keys=None):
    with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
        if set(x.files) != set(y.files):
            raise ValueError("observation schemas differ")
        return all(
            np.array_equal(x[key], y[key])
            for key in (x.files if keys is None else keys)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--physical-gpu", type=int, default=7)
    parser.add_argument("--resume-scene-cache", action="store_true")
    args = parser.parse_args()
    manifest, output = Path(args.manifest).resolve(), Path(args.output).resolve()
    if not manifest.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        raise ValueError("audit paths must stay inside repository")
    spec = json.loads(manifest.read_text())
    output.mkdir(parents=True, exist_ok=args.resume_scene_cache)
    packets = {}
    env = dict(
        os.environ,
        MUJOCO_GL="egl",
        MUJOCO_EGL_DEVICE_ID=str(args.physical_gpu),
        PYOPENGL_PLATFORM="egl",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    for case in spec["cases"]:
        destination = output / case["case_id"]
        if args.resume_scene_cache and (destination / "takeover.json").exists():
            cached = json.loads((destination / "scene_audit.json").read_text())
            if cached["case"] != case:
                raise ValueError("cached fixture differs from the declared case")
            packet = json.loads((destination / "takeover.json").read_text())
            expected_step = 40 if case["kind"] == "H" else 0
            if packet["step"] != expected_step or packet["applied_actions"]:
                raise ValueError("cached takeover is not an initial observation")
            if case["kind"] == "H":
                history = packet["initial_history"]
                assert [o["step"] for o in history["observations"]] == [30, 35]
                assert [a["step"] for a in history["applied_actions"]] == list(range(30, 40))
            packets[case["case_id"]] = packet
            print(json.dumps({"case": case["case_id"], "fixture_reused": True}), flush=True)
            continue
        with (output / f"{case['case_id']}.stderr.log").open("w") as errors:
            result = subprocess.run(
                [
                    str(ROOT / ".venv-sim/bin/python"),
                    "-u",
                    str(ROOT / "scripts/libero_diagnostic_server.py"),
                    "--case",
                    str(manifest.parent / f"{case['case_id']}.json"),
                    "--output",
                    str(destination),
                    "--physical-gpu",
                    str(args.physical_gpu),
                ],
                input='{"close":true}\n',
                stdout=subprocess.PIPE,
                stderr=errors,
                text=True,
                check=False,
                env=env,
                timeout=180,
            )
            if result.returncode:
                raise RuntimeError(f"scene creation failed: {errors.name}")
        lines = [
            line
            for line in result.stdout.splitlines()
            if line.startswith("IP_OBSERVATION:")
        ]
        if len(lines) != 1:
            raise ValueError("expected exactly one takeover observation")
        packet = json.loads(lines[0].removeprefix("IP_OBSERVATION:"))
        packets[case["case_id"]] = packet
        (destination / "takeover.json").write_text(json.dumps(packet, indent=2) + "\n")
        if case["kind"] == "H":
            history = packet["initial_history"]
            assert packet["step"] == 40
            assert [o["step"] for o in history["observations"]] == [30, 35]
            assert [a["step"] for a in history["applied_actions"]] == list(
                range(30, 40)
            )
        print(json.dumps({"case": case["case_id"], "scene_created": True}), flush=True)
    audits = []
    for layout in sorted({c["layout"] for c in spec["cases"]}):
        cases = [c for c in spec["cases"] if c["layout"] == layout]
        scene_reports = [
            json.loads((output / c["case_id"] / "scene_audit.json").read_text())
            for c in cases
        ]
        states = np.asarray([r["physical_state"] for r in scene_reports])
        excluded = scene_reports[0]["private_evaluation"].get(
            "ignored_state_indices", []
        )
        if excluded:
            states = np.delete(states, excluded, axis=1)
        if not np.allclose(states, states[0], atol=1e-10, rtol=0):
            raise ValueError(f"paired physical states differ: {layout}")
        if cases[0]["kind"] == "G":
            a, b = [ROOT / packets[c["case_id"]]["path"] for c in cases]
            keys = [
                "states",
                "agent_K",
                "wrist_rgb",
                "wrist_depth",
                "wrist_K",
                "wrist_T_world",
            ]
            if not arrays_equal(a, b, keys) or arrays_equal(a, b, ["agent_T_world"]):
                raise ValueError(
                    f"camera perturbation changed unrelated inputs or did nothing: {layout}"
                )
        elif cases[0]["kind"] == "S":
            normal = next(c for c in cases if c["condition"] == "normal")
            baseline = ROOT / packets[normal["case_id"]]["path"]
            for c in cases:
                packet = packets[c["case_id"]]
                path = ROOT / packet["path"]
                if packet["evaluation_success"]:
                    raise ValueError(f"task already solved at reset: {c['case_id']}")
                if packet["task"] != packets[normal["case_id"]]["task"]:
                    raise ValueError("scene intervention changed task language")
                if not arrays_equal(baseline, path, ["states", "agent_K", "wrist_K", "wrist_T_world"]):
                    raise ValueError("scene intervention changed unrelated robot/camera inputs")
                camera_changed = bool(c["yaw_degrees"] or c["pitch_degrees"])
                same_camera = arrays_equal(baseline, path, ["agent_T_world"])
                if same_camera == camera_changed:
                    raise ValueError("unexpected camera transformation")
                if not camera_changed and not arrays_equal(baseline, path, ["agent_depth", "wrist_depth"]):
                    raise ValueError("lighting-only intervention changed scene depth")
                if c["condition"] != "normal" and arrays_equal(baseline, path, ["agent_rgb", "wrist_rgb"]):
                    raise ValueError("scene appearance intervention had no effect")
        else:
            hidden = [
                packets[c["case_id"]] for c in cases if c["condition"] == "hidden"
            ]
            a, b = hidden
            if not arrays_equal(ROOT / a["path"], ROOT / b["path"]):
                raise ValueError(
                    f"hidden current frames leak target identity: {layout}"
                )
            if (
                a["task"] != b["task"]
                or a["initial_history"]["applied_actions"]
                != b["initial_history"]["applied_actions"]
            ):
                raise ValueError(
                    f"task or actual controls leak target identity: {layout}"
                )
            past_a, past_b = [
                ROOT / p["initial_history"]["observations"][-1]["path"] for p in hidden
            ]
            if arrays_equal(past_a, past_b, ["agent_rgb"]):
                raise ValueError(
                    f"past evidence does not distinguish targets: {layout}"
                )
        audits.append({"layout": layout, "passed": True})
    report = {
        "manual_seed": spec["manual_seed"],
        "manifest": str(manifest.relative_to(ROOT)),
        "cases": len(spec["cases"]),
        "audits": audits,
        "passed": True,
        "limits": "No policy skill or target visibility judgement is established by this array audit.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
