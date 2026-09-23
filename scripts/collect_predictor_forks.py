"""Fresh-environment, same-prefix action forks; simulator-only private state audit."""

import argparse
import json
import random
import time

import h5py
import numpy as np
from libero_probe_server import create_environment
from prepare_libero import ROOT, inside, public_frame, restore, write_json


class ReplayMismatch(ValueError):
    def __init__(self, case_id, detail):
        self.case_id = case_id
        super().__init__(f"case {case_id}: {detail}")


def plan(output):
    rng = random.Random(7227)
    entries = json.loads(
        (ROOT / "data/prepared/stage1_full/manifest.json").read_text()
    )["episodes"]
    entries = [x for x in entries if x["split"] == "train"]
    families = sorted({x["reset_family"] for x in entries})
    rng.shuffle(families)
    cases = []
    for task_index, family in enumerate(families[:8]):
        choices = [x for x in entries if x["reset_family"] == family]
        tasks = sorted({x["task"] for x in choices})
        task = rng.choice(tasks)
        choices = [x for x in choices if x["task"] == task]
        rng.shuffle(choices)
        for entry in choices[:3]:
            suite, name, episode = entry["episode_id"].split("/")
            source = ROOT / "data/libero_raw" / suite / (name + ".hdf5")
            with h5py.File(source, "r") as data:
                length = len(data[f"data/{episode}/actions"])
            for fraction in [0.25, 0.65]:
                cases.append(
                    {
                        "case_id": len(cases),
                        "task_index": task_index,
                        "family": family,
                        "task": task,
                        "split": "train"
                        if task_index < 4
                        else "development"
                        if task_index < 6
                        else "confirmation",
                        "source": str(source.relative_to(ROOT)),
                        "episode": episode,
                        "recorded_step": int(length * fraction),
                        "manual_seed": 7227 + len(cases),
                    }
                )
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "plan.json",
        {
            "data_manual_seed": 7227,
            "cases": cases,
            "horizons": [5, 10, 30],
            "branches": [
                "hold",
                "positive_x",
                "negative_x",
                "positive_y",
                "negative_y",
                "gripper",
                "hold_repeat_1",
                "hold_repeat_2",
            ],
            "reset": "Fresh environment and full deterministic 10-step warmup for every branch",
            "claim_limit": "Source tasks were seen in original policy training; splits assess new head learning, not policy generalization",
        },
    )


def private(env):
    return {
        "sim_state": env.sim.get_state().flatten(),
        "objects": {
            name: np.asarray(env.sim.data.body_xpos[index]).copy()
            for name, index in env.obj_body_id.items()
        },
    }


def replay_physics(env):
    return {
        key: np.asarray(getattr(env.sim.data, key)).copy()
        for key in (
            "qpos",
            "qvel",
            "qacc",
            "qacc_warmstart",
            "ctrl",
            "cam_xpos",
            "cam_xmat",
            "geom_xpos",
            "geom_xmat",
        )
    }


def save_mismatch(
    folder,
    branch,
    index,
    reference,
    actual,
    physics,
    initial_private,
    reference_private,
):
    destination = folder / "mismatches" / f"{branch}_history_{index}"
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "observations.npz",
        **{f"reference_{k}": v for k, v in reference.items()},
        **{f"actual_{k}": v for k, v in actual.items()},
    )
    np.savez_compressed(destination / "physics.npz", **physics)
    np.savez_compressed(
        destination / "initial_states.npz",
        reference=reference_private["sim_state"],
        actual=initial_private["sim_state"],
    )
    changes = {
        k: {
            "max_abs": float(
                np.abs(reference[k].astype(float) - actual[k].astype(float)).max()
            ),
            "different_values": int(np.count_nonzero(reference[k] != actual[k])),
        }
        for k in reference
        if not np.array_equal(reference[k], actual[k])
    }
    write_json(destination / "differences.json", changes)


def collect(output, shard, physical_gpu, workers=2):
    specification = json.loads((output / "plan.json").read_text())
    render_profile = specification.get("render_profile", "native")
    started = time.monotonic()
    for case in specification["cases"][shard::workers]:
        folder = output / f"case_{case['case_id']:03d}"
        folder.mkdir(exist_ok=True)
        old = (
            json.loads((folder / "report.json").read_text())
            if (folder / "report.json").exists()
            else None
        )
        if old and old.get("render_profile", "native") != render_profile:
            raise ValueError("cannot mix renderer profiles within a dataset")
        if old and old["complete"]:
            continue
        source = ROOT / case["source"]
        with h5py.File(source, "r") as data:
            state = data[f"data/{case['episode']}/states"][case["recorded_step"]]
            hold_grip = float(
                data[f"data/{case['episode']}/actions"][case["recorded_step"], -1]
            )
        canonical = None
        reference_private = None
        hold_futures = {}
        records = []
        if old:
            records = old["records"]
            canonical = [
                dict(np.load(folder / f"history_{i}.npz", allow_pickle=False))
                for i in range(3)
            ]
            stored = dict(np.load(folder / "initial_private.npz", allow_pickle=False))
            reference_private = {
                "sim_state": stored.pop("sim_state"),
                "objects": stored,
            }
            hold_futures = {
                h: dict(np.load(folder / f"hold_{h:02d}.npz", allow_pickle=False))
                for h in specification["horizons"]
            }
        for branch in specification["branches"]:
            if any(r["branch"] == branch for r in records):
                continue
            random.seed(case["manual_seed"])
            np.random.seed(case["manual_seed"])
            env, raw, task = create_environment(
                source,
                case["episode"],
                case["manual_seed"],
                physical_gpu,
                render_profile=render_profile,
            )
            try:
                raw = restore(env, state)
                hold = np.zeros(7, dtype=np.float32)
                hold[-1] = hold_grip
                history = []
                history_physics = []
                for step in range(11):
                    if step in [0, 5, 10]:
                        history.append(public_frame(env, raw))
                        history_physics.append(replay_physics(env))
                    if step < 10:
                        env.step(hold)
                        env._update_observables(force=True)
                        raw = env._get_observations()
                initial_private = private(env)
                if canonical is None:
                    canonical = history
                    reference_private = initial_private
                    for i, frame in enumerate(history):
                        np.savez_compressed(folder / f"history_{i}.npz", **frame)
                        np.savez_compressed(
                            folder / f"history_physics_{i}.npz", **history_physics[i]
                        )
                    np.savez_compressed(
                        folder / "initial_private.npz",
                        sim_state=initial_private["sim_state"],
                        **initial_private["objects"],
                    )
                else:
                    for index, (a, b) in enumerate(zip(canonical, history)):
                        for key in a:
                            if not np.array_equal(a[key], b[key]):
                                save_mismatch(
                                    folder,
                                    branch,
                                    index,
                                    a,
                                    b,
                                    history_physics[index],
                                    initial_private,
                                    reference_private,
                                )
                                raise ReplayMismatch(
                                    case["case_id"],
                                    f"nonidentical initial public observation: {branch} {key}",
                                )
                    if not np.array_equal(
                        reference_private["sim_state"], initial_private["sim_state"]
                    ):
                        raise ReplayMismatch(
                            case["case_id"], "nonidentical physical initial state"
                        )
                control = hold.copy()
                if branch in ["positive_x", "negative_x"]:
                    control[0] = 0.2 if branch == "positive_x" else -0.2
                if branch in ["positive_y", "negative_y"]:
                    control[1] = 0.2 if branch == "positive_y" else -0.2
                if branch == "gripper":
                    control[-1] = -1 if hold_grip >= 0 else 1
                low, high = env.action_spec
                if (control < low).any() or (control > high).any():
                    raise ValueError("invalid control")
                endpoints = []
                for h in range(1, 31):
                    env.step(control)
                    if h in specification["horizons"]:
                        env._update_observables(force=True)
                        frame = public_frame(env, env._get_observations())
                        filename = f"{branch}_{h:02d}.npz"
                        np.savez_compressed(folder / filename, **frame)
                        info = private(env)
                        displacements = {
                            k: float(
                                np.linalg.norm(v - reference_private["objects"][k])
                            )
                            for k, v in info["objects"].items()
                        }
                        endpoint = {
                            "horizon": h,
                            "file": filename,
                            "object_displacement_m": displacements,
                            "eef_displacement_m": float(
                                np.linalg.norm(
                                    frame["states"][:3] - canonical[-1]["states"][:3]
                                )
                            ),
                            "success": bool(env._check_success()),
                        }
                        if branch == "hold":
                            hold_futures[h] = frame
                        elif branch.startswith("hold_repeat"):
                            endpoint["repeat_max_abs"] = {
                                k: float(
                                    np.abs(
                                        frame[k].astype(float)
                                        - hold_futures[h][k].astype(float)
                                    ).max()
                                )
                                for k in frame
                            }
                            if any(endpoint["repeat_max_abs"].values()):
                                raise ReplayMismatch(
                                    case["case_id"],
                                    "nonidentical repeated hold endpoint",
                                )
                        endpoints.append(endpoint)
                records.append(
                    {
                        "branch": branch,
                        "control": control.tolist(),
                        "endpoints": endpoints,
                    }
                )
                write_json(
                    folder / "report.json",
                    {
                        "complete": False,
                        "render_profile": render_profile,
                        "case": case,
                        "task": task,
                        "history_actions": [hold.tolist()] * 10,
                        "records": records,
                    },
                )
                print(
                    json.dumps(
                        {
                            "case": case["case_id"],
                            "branch": branch,
                            "seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
            finally:
                env.close()
        write_json(
            folder / "report.json",
            {
                "complete": True,
                "render_profile": render_profile,
                "case": case,
                "task": task,
                "history_actions": [hold.tolist()] * 10,
                "records": records,
                "initial_arrays_identical": True,
            },
        )
    write_json(
        output / f"collection_{shard}_complete.json",
        {"complete": True, "seconds": time.monotonic() - started, "gpu": physical_gpu},
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["plan", "collect"])
    p.add_argument("--output", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--physical-gpu", type=int, default=2)
    p.add_argument("--replay-retries", type=int, default=0)
    args = p.parse_args()
    output = inside(ROOT / args.output)
    if args.mode == "plan":
        plan(output)
    else:
        attempts = {}
        while True:
            try:
                collect(output, args.shard, args.physical_gpu, args.workers)
                break
            except ReplayMismatch as error:
                attempt = attempts.get(error.case_id, 0) + 1
                if attempt > args.replay_retries:
                    raise
                attempts[error.case_id] = attempt
                archive = output / "replay_retries" / f"case_{error.case_id:03d}"
                archive.mkdir(parents=True, exist_ok=True)
                index = len(list(archive.glob("attempt_*"))) + 1
                destination = archive / f"attempt_{index}"
                (output / f"case_{error.case_id:03d}").rename(destination)
                write_json(
                    destination / "retry_reason.json",
                    {
                        "reason": str(error),
                        "same_manual_seed": True,
                        "strict_equality_retained": True,
                    },
                )


if __name__ == "__main__":
    main()
