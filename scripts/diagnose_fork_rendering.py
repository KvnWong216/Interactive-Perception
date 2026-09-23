"""Repeat complete branch lifecycles and retain exact evidence of rendering drift."""

import argparse
import json
import random
import time

import h5py
import numpy as np
from libero_probe_server import create_environment
from prepare_libero import ROOT, public_frame, restore, write_json


def arrays(obj, names):
    return {k: np.asarray(getattr(obj, k)).copy() for k in names if hasattr(obj, k)}


def delta(a, b):
    return {
        k: {
            "different": int(np.count_nonzero(a[k] != b[k])),
            "max_abs": float(np.abs(a[k].astype(float) - b[k].astype(float)).max()),
        }
        for k in a
        if not np.array_equal(a[k], b[k])
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=int, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--repeats", type=int, default=16)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = (ROOT / a.output).resolve()
    assert out.is_relative_to(ROOT)
    out.mkdir(exist_ok=False)
    root = ROOT / "runs/transition_scaling_s10227"
    case = json.loads((root / "plan.json").read_text())["cases"][a.case]
    with h5py.File(ROOT / case["source"], "r") as f:
        state = f[f"data/{case['episode']}/states"][case["recorded_step"]]
        grip = float(f[f"data/{case['episode']}/actions"][case["recorded_step"], -1])
    baseline = None
    rows = []
    started = time.monotonic()
    for repeat in range(a.repeats):
        random.seed(case["manual_seed"])
        np.random.seed(case["manual_seed"])
        env, raw, _ = create_environment(
            ROOT / case["source"], case["episode"], case["manual_seed"], a.gpu
        )
        try:
            raw = restore(env, state)
            hold = np.zeros(7, dtype=np.float32)
            hold[-1] = grip
            history = []
            physical = []
            for step in range(11):
                if step in [0, 5, 10]:
                    history.append(public_frame(env, raw))
                    physical.append(
                        arrays(
                            env.sim.data,
                            [
                                "qpos",
                                "qvel",
                                "qacc",
                                "qacc_warmstart",
                                "ctrl",
                                "act",
                                "mocap_pos",
                                "mocap_quat",
                                "cam_xpos",
                                "cam_xmat",
                                "geom_xpos",
                                "geom_xmat",
                                "light_xpos",
                                "light_xdir",
                            ],
                        )
                    )
                if step < 10:
                    env.step(hold)
                    env._update_observables(force=True)
                    raw = env._get_observations()
            model = arrays(
                env.sim.model,
                [
                    "cam_pos",
                    "cam_quat",
                    "light_pos",
                    "light_dir",
                    "geom_rgba",
                    "mat_rgba",
                    "mat_specular",
                    "mat_shininess",
                    "tex_rgb",
                ],
            )
            if baseline is None:
                baseline = (history, physical, model)
            differences = [delta(x, y) for x, y in zip(baseline[0], history)]
            physdiff = [delta(x, y) for x, y in zip(baseline[1], physical)]
            row = {
                "repeat": repeat,
                "public": differences,
                "physics": physdiff,
                "model": delta(baseline[2], model),
            }
            if any(differences):
                for i, (x, y) in enumerate(zip(baseline[0], history)):
                    np.savez_compressed(
                        out / f"repeat_{repeat}_history_{i}.npz",
                        **{f"reference_{k}": v for k, v in x.items()},
                        **{f"actual_{k}": v for k, v in y.items()},
                    )
                for i, (x, y) in enumerate(zip(baseline[1], physical)):
                    np.savez_compressed(
                        out / f"repeat_{repeat}_physics_{i}.npz",
                        **{f"reference_{k}": v for k, v in x.items()},
                        **{f"actual_{k}": v for k, v in y.items()},
                    )
            # Exercise future-step rendering and destruction exactly as branch collection does.
            action = hold.copy()
            index = repeat % 6
            if index in [1, 2]:
                action[0] = 0.2 if index == 1 else -0.2
            if index in [3, 4]:
                action[1] = 0.2 if index == 3 else -0.2
            if index == 5:
                action[-1] = -1 if grip >= 0 else 1
            for step in range(30):
                env.step(action)
                if step + 1 in [5, 10, 30]:
                    env._update_observables(force=True)
                    public_frame(env, env._get_observations())
            rows.append(row)
            write_json(
                out / "report.json",
                {
                    "case": case,
                    "rows": rows,
                    "complete": len(rows) == a.repeats,
                    "seconds": time.monotonic() - started,
                },
            )
            print(json.dumps(row), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
