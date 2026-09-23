"""Re-render selected demonstrations; export only calibrated public observations.

Run in .venv-sim. --audit-only checks one episode without exporting training data.
All outputs and simulator configuration stay under this repository.
"""

import argparse
import json
import multiprocessing
import os
import shutil
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
LIBERO = ROOT / "third_party/LIBERO/libero/libero"
VIEWS = {"agent": "agentview", "wrist": "robot0_eye_in_hand"}


def inside(path):
    result = Path(path).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError("output/source path escapes repository")
    return result


def write_json(path, value):
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = inside(path.with_suffix(path.suffix + ".partial"))
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def configure():
    paths = {
        "benchmark_root": LIBERO,
        "bddl_files": LIBERO / "bddl_files",
        "init_states": LIBERO / "init_files",
        "assets": LIBERO / "assets",
        "datasets": ROOT / "data/libero_raw",
    }
    for key, relative in {
        "LIBERO_CONFIG_PATH": ".libero",
        "MPLCONFIGDIR": ".cache/matplotlib",
        "NUMBA_CACHE_DIR": ".cache/numba",
    }.items():
        target = inside(ROOT / relative)
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    temporary = inside(ROOT / f".libero/config.{os.getpid()}.partial")
    temporary.write_text(yaml.safe_dump({k: str(v) for k, v in paths.items()}))
    temporary.replace(ROOT / ".libero/config.yaml")


def model_xml(source):
    """Relocate asset filenames without changing cameras or physical properties."""
    import robosuite

    tree = ET.fromstring(source)
    for node in tree.iter():
        name = node.get("file")
        if name is None:
            continue
        parts = Path(name).parts
        if "robosuite" in parts:
            start = max(i for i, p in enumerate(parts) if p == "robosuite")
            path = Path(robosuite.__file__).parent.joinpath(*parts[start + 1 :])
        elif "assets" in parts:
            start = max(i for i, p in enumerate(parts) if p == "assets")
            path = LIBERO.joinpath("assets", *parts[start + 1 :])
        else:
            raise ValueError(f"unrecognized simulation asset: {name}")
        if not inside(path).is_file():
            raise FileNotFoundError(path)
        node.set("file", str(path))
    return ET.tostring(tree, encoding="unicode")


def scene_family(xml):
    """Readable preliminary grouping; finalization checks the full asset inventory."""
    root = ET.fromstring(xml)
    return "scene:" + ",".join(
        sorted(
            node.get("name", "")
            for node in root.find("worldbody")
            if node.tag == "body"
        )
    )


def public_frame(env, raw):
    from robosuite.utils import camera_utils as camera
    from robosuite.utils.transform_utils import quat2axisangle

    result = {
        "states": np.concatenate(
            (
                raw["robot0_eef_pos"],
                quat2axisangle(raw["robot0_eef_quat"]),
                raw["robot0_gripper_qpos"],
            )
        ).astype(np.float32)
    }
    for view, name in VIEWS.items():
        rgb = raw[f"{name}_image"]
        h, w = rgb.shape[:2]
        # robosuite OpenGL raw images are bottom-up. The native VLA rotates
        # both axes, which is a horizontal reflection of a top-down CV image.
        reflection = np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]])
        result[f"{view}_rgb"] = np.ascontiguousarray(rgb[::-1, ::-1])
        depth = camera.get_real_depth_map(env.sim, raw[f"{name}_depth"])
        result[f"{view}_depth"] = np.ascontiguousarray(
            depth[::-1, ::-1, 0], dtype=np.float32
        )
        k = camera.get_camera_intrinsic_matrix(env.sim, name, h, w)
        # Our pixel coordinates refer to pixel centers (integer array indices).
        k[:2, 2] -= 0.5
        result[f"{view}_K"] = (reflection @ k).astype(np.float32)
        result[f"{view}_T_world"] = camera.get_camera_extrinsic_matrix(
            env.sim, name
        ).astype(np.float32)
    return result


def check_depth_rays(env, frame):
    """Independent mesh-ray audit; simulator truth is never added to policy data."""
    import mujoco

    reports = {}
    for view in VIEWS:
        depth, k, pose = (frame[f"{view}_{key}"] for key in ("depth", "K", "T_world"))
        errors = []
        for y in np.linspace(8, depth.shape[0] - 9, 6).astype(int):
            for x in np.linspace(8, depth.shape[1] - 9, 6).astype(int):
                # Exclude pixel footprints crossing a silhouette; a point ray
                # and the rasterizer need not agree at an object boundary.
                neighborhood = depth[y - 1 : y + 2, x - 1 : x + 2]
                if not np.isfinite(neighborhood).all() or np.ptp(neighborhood) > 0.02:
                    continue
                ray = np.linalg.solve(k, np.array([x, y, 1.0]))
                scale = np.linalg.norm(ray)
                direction = np.ascontiguousarray(
                    pose[:3, :3] @ ray / scale, dtype=np.float64
                )
                distance = mujoco.mj_ray(
                    env.sim.model._model,
                    env.sim.data._data,
                    np.ascontiguousarray(pose[:3, 3], dtype=np.float64),
                    direction,
                    np.asarray(
                        env.sim._render_context_offscreen.vopt.geomgroup, dtype=np.uint8
                    ),
                    1,
                    -1,
                    np.array([-1], dtype=np.int32),
                )
                if distance >= 0:
                    errors.append(abs(distance / scale - float(depth[y, x])))
        reports[view] = {
            "rays": len(errors),
            "p90_depth_error_m": float(np.quantile(errors, 0.9)) if errors else None,
        }
        if len(errors) < 6 or reports[view]["p90_depth_error_m"] > 0.015:
            raise RuntimeError(f"depth/calibration ray audit failed: {reports}")
    return reports


def restore(env, state):
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    for robot in env.robots:
        robot.controller.update(force=True)
    env._update_observables(force=True)
    return env._get_observations()


def pose_error(model, actual, expected):
    """Separate metres from radians; a flattened state norm mixes both and velocity."""
    translation, rotation = [], []
    for kind, address in zip(model.jnt_type, model.jnt_qposadr):
        if kind == 0:  # free joint: xyz followed by a unit quaternion
            translation.append(
                np.linalg.norm(
                    actual[address : address + 3] - expected[address : address + 3]
                )
            )
            address += 3
        if kind in (0, 1):
            a, b = actual[address : address + 4], expected[address : address + 4]
            dot = abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            rotation.append(2 * np.arccos(np.clip(dot, 0, 1)))
        elif kind == 2:
            translation.append(abs(actual[address] - expected[address]))
        elif kind == 3:
            rotation.append(abs(actual[address] - expected[address]))
    return float(max(translation, default=0)), float(max(rotation, default=0))


def replay(file, episode, size, *, limit=None, manual_seed=17):
    from gpu_devices import configure_render_gpu

    configure_render_gpu(7)
    from libero.libero.envs import TASK_MAPPING
    from robosuite import macros

    if macros.IMAGE_CONVENTION != "opengl":
        raise RuntimeError(
            "camera calibration requires the audited OpenGL image convention"
        )
    start = time.monotonic()
    with h5py.File(file, "r") as source:
        data, demo = source["data"], source[f"data/{episode}"]
        cfg = json.loads(data.attrs["env_args"])
        kwargs = cfg["env_kwargs"].copy()
        bddl = (
            LIBERO
            / "bddl_files"
            / file.parent.name
            / (file.stem.removesuffix("_demo") + ".bddl")
        )
        kwargs.update(
            bddl_file_name=str(bddl),
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_depths=True,
            camera_names=list(VIEWS.values()),
            camera_heights=size,
            camera_widths=size,
            camera_segmentations=None,
            ignore_done=True,
            render_gpu_device_id=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "7")),
        )
        if kwargs["controller_configs"].get("control_delta") is not True:
            raise ValueError("this converter requires native relative OSC_POSE actions")
        states, actions = demo["states"][:], demo["actions"][:]
        if len(states) != len(actions) or actions.shape[1] != 7:
            raise ValueError("unexpected source transition alignment")
        dt = np.diff(states[:, 0])
        if not np.allclose(dt, 1 / kwargs["control_freq"], atol=1e-6):
            raise ValueError("recorded state times differ from control frequency")
        # Every exported observation comes from a recorded simulator state.
        # The source lacks state[T]; exclude only the final unpaired action.
        count = len(actions) - 1 if limit is None else min(limit, len(actions) - 1)
        if count < 1:
            raise ValueError("source has no complete recorded transition")
        xml = model_xml(demo.attrs["model_file"])
        report = {
            "source": str(file.relative_to(ROOT)),
            "episode": episode,
            "original_actions": len(actions),
            "exported_actions": count,
            "excluded_tail_actions": len(actions) - count,
            "terminal_observation": "last recorded state; final unpaired action excluded",
            "control_freq": kwargs["control_freq"],
            "image_size": size,
            "reset_family": scene_family(xml),
            "task": json.loads(data.attrs["problem_info"])["language_instruction"],
        }
        env = TASK_MAPPING[cfg["problem_name"]](**kwargs)
        frames, state_errors, post_position_errors, rgb_errors = [], [], [], []
        position_errors, velocity_errors = [], []
        translation_errors, rotation_errors = [], []
        try:
            env.seed(manual_seed)
            env.reset()
            env.reset_from_xml_string(xml)
            env.sim.reset()
            for t in range(count):
                raw = restore(env, states[t])
                frames.append(public_frame(env, raw))
                if t == 0:
                    report["calibration_rays"] = check_depth_rays(env, frames[-1])
                post, _, _, _ = env.step(actions[t])
                if t + 1 < len(states):
                    error = env.sim.get_state().flatten() - states[t + 1]
                    state_errors.append(float(np.linalg.norm(error)))
                    position_errors.append(
                        float(np.max(np.abs(error[1 : 1 + env.sim.model.nq])))
                    )
                    velocity_errors.append(
                        float(np.max(np.abs(error[1 + env.sim.model.nq :])))
                    )
                    pos, rot = pose_error(
                        env.sim.model,
                        env.sim.data.qpos,
                        states[t + 1, 1 : 1 + env.sim.model.nq],
                    )
                    translation_errors.append(pos)
                    rotation_errors.append(rot)
                post_position_errors.append(
                    float(
                        np.linalg.norm(post["robot0_eef_pos"] - demo["obs/ee_pos"][t])
                    )
                )
                if size == demo["obs/agentview_rgb"].shape[1]:
                    rgb_errors.append(
                        float(
                            np.mean(
                                np.abs(
                                    post["agentview_image"].astype(float)
                                    - demo["obs/agentview_rgb"][t]
                                )
                            )
                        )
                    )
            post = restore(env, states[count])
            frames.append(public_frame(env, post))
            report.update(
                state_error_max=max(state_errors, default=0),
                qpos_error_max=max(position_errors, default=0),
                qvel_error_max=max(velocity_errors, default=0),
                translation_error_max_m=max(translation_errors, default=0),
                rotation_error_max_rad=max(rotation_errors, default=0),
                post_position_error_max=max(post_position_errors, default=0),
                post_rgb_mae_mean=float(np.mean(rgb_errors)) if rgb_errors else None,
                success_at_export_end=bool(env._check_success()),
            )
        finally:
            env.close()
        report["wall_seconds"] = time.monotonic() - start
        report["dual_view_frames_per_second"] = len(frames) / report["wall_seconds"]
        report["passed"] = (
            report["translation_error_max_m"] <= 0.01
            and report["rotation_error_max_rad"] <= 0.02
        )
        # Legacy obs sensors may be cached within a control interval. Their
        # offset is diagnostic, not a filter on correctly recorded states.
        report["legacy_observation_error_is_diagnostic"] = True
        arrays = {key: np.stack([f[key] for f in frames]) for key in frames[0]}
        arrays["actions"] = actions[:count].astype(np.float32)
        return arrays, report


def convert_one(job):
    row, ep, image_size, output, audit_only, manual_seed = job
    target, file = inside(output), inside(ROOT / row["path"])
    identity = f"{file.parent.name}/{file.stem}/{ep}"
    destination = inside(target / f"{identity}.npz")
    report_path = destination.with_suffix(".json")
    reused = destination.exists() and report_path.exists() and not audit_only
    if reused:
        report = json.loads(report_path.read_text())
        if report.get("image_size") != image_size:
            raise ValueError("existing converted episode differs from its provenance")
    else:
        arrays, report = replay(file, ep, image_size, manual_seed=manual_seed)
        report.update(
            manual_seed=manual_seed,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        if report["passed"]:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = inside(destination.with_suffix(".npz.partial"))
            with temporary.open("wb") as out:
                np.savez_compressed(out, **arrays)
            temporary.replace(destination)
        write_json(report_path, report)
    entry = None
    if report["passed"]:
        entry = {
            "episode_id": identity,
            "reset_family": report["reset_family"],
            "task": report["task"],
            "path": str(destination.relative_to(target)),
        }
    report["reused_in_this_invocation"] = reused
    return entry, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="Freeze a subset of fully downloaded allowlisted task files",
    )
    parser.add_argument("--file")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--manual-seed", type=int, default=17)
    parser.add_argument("--max-output-gb", type=float, default=400)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output", default="data/prepared/stage1")
    args = parser.parse_args()
    if (
        args.episodes_per_task < 1
        or args.image_size < 32
        or not 1 <= args.workers <= 32
    ):
        raise ValueError("invalid episode/resolution/worker budget")
    if not 0 <= args.manual_seed < 2**32 or not 0 < args.max_output_gb <= 400:
        raise ValueError("invalid manual_seed or export budget")
    configure()
    target = inside(ROOT / args.output)
    target.mkdir(parents=True, exist_ok=True)
    rows = json.loads((ROOT / "experiments/stage1_assets.json").read_text())["files"]
    rows = [r for r in rows if r["path"].endswith(".hdf5")]
    selection = target / "source_selection.json"
    if selection.exists() and not args.audit_only:
        selected = json.loads(selection.read_text())
        if (
            selected["manual_seed"] != args.manual_seed
            or selected["episodes_per_task"] != args.episodes_per_task
            or selected["image_size"] != args.image_size
        ):
            raise ValueError("selection budget changed; use a new output directory")
        paths = selected["task_files"]
        rows = [r for r in rows if r["path"] in paths]
    elif args.available_only or args.audit_only:
        rows = [r for r in rows if (ROOT / r["path"]).is_file()]
    if args.file:
        file = inside(args.file)
        rows = [r for r in rows if ROOT / r["path"] == file]
    if args.audit_only:
        rows = rows[:1]
    if not rows:
        raise ValueError("no complete allowlisted source files selected")
    write_json(
        selection,
        {
            "task_files": [r["path"] for r in rows],
            "episodes_per_task": args.episodes_per_task,
            "image_size": args.image_size,
            "manual_seed": args.manual_seed,
            "partial_allowlist": len(rows) != 40,
        },
    )
    jobs = []
    estimated_bytes = 0
    for row in rows:
        file = inside(ROOT / row["path"])
        if not file.is_file() or file.stat().st_size != row["size"]:
            raise ValueError(f"source not complete/verified: {row['path']}")
        with h5py.File(file, "r") as stream:
            names = sorted(stream["data"], key=lambda s: int(s.split("_")[-1]))
            np.random.default_rng(args.manual_seed).shuffle(names)
            for ep in names[: 1 if args.audit_only else args.episodes_per_task]:
                count = len(stream[f"data/{ep}/states"])
                estimated_bytes += count * (
                    2 * args.image_size**2 * 7 + 8 * 4 + 2 * 25 * 4 + 7 * 4
                )
        jobs.extend(
            (row, ep, args.image_size, str(target), args.audit_only, args.manual_seed)
            for ep in names[: 1 if args.audit_only else args.episodes_per_task]
        )
    entries, reports = [], []
    if (
        estimated_bytes > args.max_output_gb * 1_000_000_000
        or shutil.disk_usage(ROOT).free < estimated_bytes + 10_000_000_000
    ):
        raise RuntimeError("RGB-D export exceeds its configured budget or free space")
    started = time.monotonic()
    reused_count, new_observations = 0, 0
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [pool.submit(convert_one, job) for job in jobs]
        for future in as_completed(futures):
            entry, report = future.result()
            reports.append(report)
            reused_count += int(report["reused_in_this_invocation"])
            if not report["reused_in_this_invocation"]:
                new_observations += report["exported_actions"] + 1
            if entry is not None:
                entries.append(entry)
            write_json(target / "latest_audit.json", report)
            write_json(
                target / "progress.json",
                {
                    "manual_seed": args.manual_seed,
                    "workers": args.workers,
                    "completed": len(reports),
                    "planned": len(jobs),
                    "accepted": len(entries),
                    "excluded": len(reports) - len(entries),
                    "elapsed_seconds": time.monotonic() - started,
                    "reused_episodes": reused_count,
                    "new_episodes": len(reports) - reused_count,
                    "new_observations": new_observations,
                    "new_observations_per_second": new_observations
                    / max(time.monotonic() - started, 1e-6),
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            with inside(target / "audit_history.jsonl").open("a") as history:
                history.write(json.dumps(report, allow_nan=False) + "\n")
            print(
                json.dumps(
                    {
                        "completed": len(reports),
                        "planned": len(jobs),
                        "accepted": len(entries),
                        **report,
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
    if not args.audit_only and entries:
        counts = {
            family: sum(e["reset_family"] == family for e in entries)
            for family in {e["reset_family"] for e in entries}
        }
        families = sorted(counts)
        np.random.default_rng(args.manual_seed).shuffle(families)
        if len(families) < 3:
            raise ValueError("need three scene families for separate data splits")
        # Keep the largest scene family in training. Choose validation/test
        # families closest to 10% each; actual ratios are reported, not assumed.
        largest = max(families, key=counts.get)
        candidates = [f for f in families if f != largest]
        validation = min(candidates, key=lambda f: abs(counts[f] - len(entries) * 0.1))
        candidates.remove(validation)
        test = min(candidates, key=lambda f: abs(counts[f] - len(entries) * 0.1))
        for entry in entries:
            entry["split"] = (
                "validation"
                if entry["reset_family"] == validation
                else "test"
                if entry["reset_family"] == test
                else "train"
            )
        entries.sort(key=lambda e: e["episode_id"])
        write_json(
            target / "manifest.json",
            {"schema": "predictive-vla-trajectories-v1", "episodes": entries},
        )
    write_json(
        target / "preparation_report.json",
        {
            "audit_only": args.audit_only,
            "selected_task_files": len(rows),
            "planned_episodes": len(jobs),
            "accepted_episodes": len(entries),
            "excluded_episodes": len(reports) - len(entries),
            "wall_seconds": time.monotonic() - started,
            "estimated_uncompressed_bytes": estimated_bytes,
            "workers": args.workers,
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "episodes": reports,
        },
    )
    if args.audit_only and not entries:
        raise RuntimeError("audit failed; inspect recorded report before continuing")


if __name__ == "__main__":
    main()
