"""Measure replay differences without relaxing the collection acceptance rule."""

import argparse
import json
import random

import h5py
import numpy as np
from collect_predictor_forks import private
from libero_probe_server import create_environment
from prepare_libero import ROOT, public_frame, restore


def difference(a, b):
    delta = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    return {
        "max_abs": float(delta.max()),
        "different_values": int(np.count_nonzero(delta)),
        "values": delta.size,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=int, default=82)
    p.add_argument("--physical-gpu", type=int, default=0)
    args = p.parse_args()
    root = ROOT / "runs/transition_scaling_s10227"
    case = next(
        c
        for c in json.loads((root / "plan.json").read_text())["cases"]
        if c["case_id"] == args.case
    )
    folder = root / f"case_{args.case:03d}"
    source = ROOT / case["source"]
    with h5py.File(source, "r") as f:
        state = f[f"data/{case['episode']}/states"][case["recorded_step"]]
        grip = float(f[f"data/{case['episode']}/actions"][case["recorded_step"], -1])
    old = [dict(np.load(folder / f"history_{i}.npz")) for i in range(3)]
    old_private = dict(np.load(folder / "initial_private.npz"))
    rows = []
    fresh = []
    for repeat in range(3):
        random.seed(case["manual_seed"])
        np.random.seed(case["manual_seed"])
        env, raw, _ = create_environment(
            source, case["episode"], case["manual_seed"], args.physical_gpu
        )
        try:
            raw = restore(env, state)
            hold = np.zeros(7, dtype=np.float32)
            hold[-1] = grip
            history = []
            for step in range(11):
                if step in [0, 5, 10]:
                    history.append(public_frame(env, raw))
                if step < 10:
                    env.step(hold)
                    env._update_observables(force=True)
                    raw = env._get_observations()
            initial = private(env)
            rows.append(
                {
                    "repeat": repeat,
                    "versus_saved": [
                        {k: difference(a[k], b[k]) for k in a}
                        for a, b in zip(old, history)
                    ],
                    "physical": difference(
                        old_private["sim_state"], initial["sim_state"]
                    ),
                    "versus_first_fresh": None
                    if not fresh
                    else [
                        {k: difference(a[k], b[k]) for k in a}
                        for a, b in zip(fresh[0], history)
                    ],
                }
            )
            fresh.append(history)
        finally:
            env.close()
    out = root / f"initial_replay_audit_{args.case}.json"
    out.write_text(json.dumps({"case": case, "rows": rows}, indent=2) + "\n")
    for row in rows:
        print(
            json.dumps(
                {
                    "repeat": row["repeat"],
                    "physical": row["physical"],
                    "differences": [
                        {k: v for k, v in a.items() if v["different_values"]}
                        for a in row["versus_saved"]
                    ],
                    "fresh_differences": None
                    if row["versus_first_fresh"] is None
                    else [
                        {k: v for k, v in a.items() if v["different_values"]}
                        for a in row["versus_first_fresh"]
                    ],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
