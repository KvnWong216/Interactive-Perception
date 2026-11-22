#!/usr/bin/env python3
"""Zero-leakage ActiveArena Astribot S1 expert trajectory collector for ASA Stage-3 finetuning.

Runs unmodified ActiveArena expert `play_once()` in Sapien + cuRobo on strictly isolated training seeds
(`seed >= 500_000`, verified disjoint from all ID & OOD `eval_seed_lists`), hooks `_take_picture` in RAM
at `save_freq=15` (`~15 Hz`), and writes ASA-aligned 29-DoF episode banks (`ep_*` + `sub_*` poses + `224x224` RGB).

Strict anti-leakage guarantees:
  1. Zero eval-seed overlap: loads all seeds from `eval_seed_lists/{info_gathering_demo,info_gathering_randomized}`
     and asserts every collected training seed is disjoint.
  2. Zero unseen-template leakage: uses ONLY `"seen"` language templates (never `"unseen"` eval templates).
  3. Zero hidden-property text spoiler: sanitizes `{A}`/`{B}`/`{C}` spoilers in task-level instructions so
     the model MUST rely on its 5-frame visual memory of the backside inspection rather than text shortcuts.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import yaml


def _setup_paths(sim_repo: Path, ipvla_repo: Path) -> None:
    curobo_src = sim_repo / "envs" / "curobo" / "src"
    for p in (ipvla_repo, curobo_src, sim_repo, sim_repo / "description" / "utils"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _load_forbidden_eval_seeds(sim_repo: Path, tasks: list[str]) -> dict[str, set[int]]:
    forbidden: dict[str, set[int]] = {t: set() for t in tasks}
    seed_root = sim_repo / "eval_seed_lists"
    for setting in ("info_gathering_demo", "info_gathering_randomized"):
        for t in tasks:
            f = seed_root / setting / f"{t}.json"
            if f.is_file():
                payload = json.loads(f.read_text(encoding="utf-8"))
                seeds = payload.get("seeds", [])
                forbidden[t].update(int(s) for s in seeds)
                for entry in payload.get("entries", []):
                    if isinstance(entry, dict) and "seed" in entry:
                        forbidden[t].add(int(entry["seed"]))
    return forbidden


def _build_non_spoiler_seen_instruction(task_name: str, desc_json: dict[str, Any], info_map: dict[str, Any], rng: random.Random) -> str:
    """Render ONLY `"seen"` templates while sanitizing hidden-label spoilers in `check_block_color` / `check_cola_color`."""
    safe_map = {str(k): str(v) for k, v in info_map.items()}
    if task_name == "check_block_color":
        # Do NOT reveal hidden backside color in the task-level instruction!
        safe_map["{A}"] = "the gray target block"
        safe_map["{B}"] = "the matching color pad"
        safe_map["{C}"] = "back-side"
        candidates = [
            "Lift {A}, check its back-side pad color, then put it down on the corresponding color pad.",
            "Move the gray block to the pad matching the color on its back.",
            "Use {a} arm to inspect the back of {A}, then move it onto {B}.",
        ]
    elif task_name == "check_cola_color":
        safe_map["{A}"] = "the cola can"
        safe_map["{B}"] = "the matching color area"
        safe_map["{C}"] = "back-side"
        candidates = [t for t in desc_json.get("seen", []) if "{C}" not in t] or desc_json.get("seen", [])
    else:
        candidates = list(desc_json.get("seen", []))

    tmpl = rng.choice(candidates)
    for k, v in safe_map.items():
        tmpl = tmpl.replace(k, v)
    return tmpl


def _build_subtask_segments(
    task_name: str,
    desc_json: dict[str, Any],
    info_map: dict[str, Any],
    frame_subtask_ids: list[int | None],
) -> list[dict[str, Any]]:
    sub_map = desc_json.get("subtask_instruction_template_map", {})
    safe_map = {str(k): str(v) for k, v in info_map.items()}
    if task_name == "check_block_color":
        safe_map["{A}"] = "the gray target block"

    segments: list[dict[str, Any]] = []
    if not frame_subtask_ids:
        return segments

    cur_id = frame_subtask_ids[0]
    s_start = 0
    for idx in range(1, len(frame_subtask_ids) + 1):
        nxt_id = frame_subtask_ids[idx] if idx < len(frame_subtask_ids) else None
        if nxt_id != cur_id:
            if cur_id is not None and str(cur_id) in sub_map:
                seen_list = sub_map[str(cur_id)].get("seen", [])
                if seen_list:
                    txt = str(seen_list[0])
                    for k, v in safe_map.items():
                        txt = txt.replace(k, v)
                    segments.append(
                        {
                            "subtask_id": int(cur_id),
                            "start_frame": s_start,
                            "end_frame": idx - 1,
                            "instruction": txt,
                        }
                    )
            cur_id = nxt_id
            s_start = idx
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect zero-leakage Astribot trajectories for ASA Stage-3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sim-repo",
        type=Path,
        default=Path("/home/ubuntu/ygwang/IP-VLA/baseline-study/external/activearena"),
    )
    parser.add_argument("--ipvla-repo", type=Path, default=Path("/home/ubuntu/ygwang/IP-VLA/ip-vla"))
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["check_block_color", "check_cola_date", "check_cola_color"],
    )
    parser.add_argument("--episodes-per-task", type=int, default=40)
    parser.add_argument("--start-seed", type=int, default=500_000)
    parser.add_argument("--save-freq", type=int, default=15)
    args = parser.parse_args()

    sim_repo = args.sim_repo.resolve()
    ipvla_repo = args.ipvla_repo.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _setup_paths(sim_repo, ipvla_repo)
    os.chdir(sim_repo)

    from envs import CONFIGS_PATH
    from envs.utils.create_actor import UnStableError
    from eval.activearena.asa_astribot_embodiment import (
        ASAAstribotEmbodimentAdapter,
        matrix_to_pose9_asa,
        pose7_to_matrix,
    )

    forbidden_by_task = _load_forbidden_eval_seeds(sim_repo, args.tasks)

    with (sim_repo / "task_config/info_gathering_demo.yml").open("r", encoding="utf-8") as f:
        base_task_args = yaml.safe_load(f)
    with Path(CONFIGS_PATH, "_embodiment_config.yml").open("r", encoding="utf-8") as f:
        emb_types = yaml.safe_load(f)
    with Path(CONFIGS_PATH, "_camera_config.yml").open("r", encoding="utf-8") as f:
        cam_types = yaml.safe_load(f)

    robot_file = emb_types["astribot_texture"]["file_path"]
    if not os.path.isabs(robot_file):
        robot_file = str((sim_repo / robot_file).resolve())
    with Path(robot_file, "config.yml").open("r", encoding="utf-8") as f:
        emb_cfg = yaml.safe_load(f)

    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tasks": args.tasks,
            "episodes_per_task_target": args.episodes_per_task,
            "save_freq": args.save_freq,
            "base_frame_mode": "level_head",
            "forbidden_eval_seed_counts": {k: len(v) for k, v in forbidden_by_task.items()},
            "episodes": [],
        }

    existing_by_task: dict[str, set[int]] = {t: set() for t in args.tasks}
    for ep in manifest["episodes"]:
        if ep["task_name"] in existing_by_task:
            existing_by_task[ep["task_name"]].add(int(ep["seed"]))

    rng = random.Random(20261122)

    for task_idx, task_name in enumerate(args.tasks):
        desc_path = sim_repo / "description/task_instruction" / f"{task_name}.json"
        desc_json = json.loads(desc_path.read_text(encoding="utf-8"))
        forbidden = forbidden_by_task[task_name]

        mod = importlib.import_module(f"envs.{task_name}")
        env_cls = getattr(mod, task_name)
        env = env_cls()

        collected = len(existing_by_task[task_name])
        cur_seed = args.start_seed + task_idx * 100_000
        if existing_by_task[task_name]:
            cur_seed = max(cur_seed, max(existing_by_task[task_name]) + 1)

        print(
            f"[collect] Task={task_name} starting at {collected}/{args.episodes_per_task} "
            f"(forbidden eval seeds={len(forbidden)}, start_seed={cur_seed})",
            flush=True,
        )

        attempt_idx = 0
        while collected < args.episodes_per_task:
            if cur_seed in forbidden or cur_seed in existing_by_task[task_name]:
                cur_seed += 1
                continue

            attempt_idx += 1
            t0 = time.monotonic()
            raw_frames: list[dict[str, Any]] = []

            def _capture_hook(env_ref=env, buf=raw_frames) -> None:
                if not env_ref.save_data:
                    return
                obs = env_ref.get_obs()
                rgb = obs["observation"]["camera_head"]["rgb"]
                robot = env_ref.robot
                cam_p = robot.head_camera.get_entity_pose()
                rp = robot.left_entity_origion_pose
                lg, rg = robot.get_normal_real_gripper_val()
                q_head = np.asarray(robot.get_head_real_jointState(), dtype=np.float64)
                q_torso_l = robot.get_torso_real_jointState()
                q_torso = float(q_torso_l[0]) if q_torso_l else 0.0
                ann = dict(env_ref._latest_frame_annotation) if env_ref._latest_frame_annotation else {}

                im224 = np.asarray(
                    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).resize((224, 224), Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                )
                buf.append(
                    {
                        "rgb224": im224,
                        "T_world_cam": pose7_to_matrix(np.concatenate([cam_p.p, cam_p.q])),
                        "T_world_robot0": pose7_to_matrix(np.concatenate([rp.p, rp.q])),
                        "left_tcp7": np.asarray(robot.get_left_tcp_pose(), dtype=np.float64),
                        "right_tcp7": np.asarray(robot.get_right_tcp_pose(), dtype=np.float64),
                        "left_grip": float(lg),
                        "right_grip": float(rg),
                        "q_head1": float(q_head[0]),
                        "q_torso": q_torso,
                        "subtask_id": ann.get("subtask", ann.get("subtask_id")),
                    }
                )
                env_ref.FRAME_IDX += 1

            env._take_picture = _capture_hook
            run_kwargs = dict(base_task_args)
            run_kwargs.update(
                task_name=task_name,
                task_config="info_gathering_demo",
                left_robot_file=robot_file,
                right_robot_file=robot_file,
                dual_arm_embodied=True,
                left_embodiment_config=emb_cfg,
                right_embodiment_config=emb_cfg,
                head_camera_h=cam_types[run_kwargs["camera"]["head_camera_type"]]["h"],
                head_camera_w=cam_types[run_kwargs["camera"]["head_camera_type"]]["w"],
                eval_mode=False,
                save_data=True,
                save_dir=str(out_dir / "_tmp_sim_cache"),
                save_freq=args.save_freq,
                render_freq=0,
            )

            try:
                env.setup_demo(now_ep_num=collected, seed=cur_seed, is_test=False, **run_kwargs)
                info = env.play_once()
                ok = bool(env.plan_success and env.check_success()) and len(raw_frames) >= 20
                info_map = (info or {}).get("info", {}) or {}
                env.close_env(clear_cache=(attempt_idx % 5 == 0))
            except UnStableError:
                try:
                    env.close_env(clear_cache=(attempt_idx % 5 == 0))
                except Exception:
                    pass
                cur_seed += 1
                continue
            except Exception as exc:
                print(f"[collect] {task_name} seed={cur_seed} failed ({type(exc).__name__}: {exc}); skipping.", flush=True)
                try:
                    env.close_env(clear_cache=True)
                except Exception:
                    pass
                env = env_cls()
                cur_seed += 1
                continue

            if not ok:
                cur_seed += 1
                continue

            assert cur_seed not in forbidden, f"Seed leakage check failed for {cur_seed}"

            # Compute unnormalized `ep_*` (anchored at t=0 `level_head`) and `sub_*` (anchored at each subtask start)
            n_f = len(raw_frames)
            sub_ids = [fr["subtask_id"] for fr in raw_frames]
            segments = _build_subtask_segments(task_name, desc_json, info_map, sub_ids)
            task_instr = _build_non_spoiler_seen_instruction(task_name, desc_json, info_map, rng)

            def _make_level_base(fr0: dict[str, Any]) -> np.ndarray:
                yaw0 = float(fr0["q_torso"] + fr0["q_head1"])
                R_loc = Rotation.from_euler("ZYX", [yaw0, 0.0, 0.0]).as_matrix()
                T_b = np.eye(4, dtype=np.float64)
                T_b[:3, :3] = fr0["T_world_robot0"][:3, :3] @ R_loc
                T_b[:3, 3] = fr0["T_world_cam"][:3, 3]
                return T_b

            inv_ep_base = np.linalg.inv(_make_level_base(raw_frames[0]))
            seg_inv_bases: list[np.ndarray] = [inv_ep_base] * n_f
            for seg in segments:
                inv_sb = np.linalg.inv(_make_level_base(raw_frames[int(seg["start_frame"])]))
                for fi in range(int(seg["start_frame"]), int(seg["end_frame"]) + 1):
                    seg_inv_bases[fi] = inv_sb

            ep_head9 = np.zeros((n_f, 9), dtype=np.float32)
            ep_left9 = np.zeros((n_f, 9), dtype=np.float32)
            ep_right9 = np.zeros((n_f, 9), dtype=np.float32)
            sub_head9 = np.zeros((n_f, 9), dtype=np.float32)
            sub_left9 = np.zeros((n_f, 9), dtype=np.float32)
            sub_right9 = np.zeros((n_f, 9), dtype=np.float32)
            left_grip = np.zeros((n_f,), dtype=np.float32)
            right_grip = np.zeros((n_f,), dtype=np.float32)

            for i, fr in enumerate(raw_frames):
                T_cam = fr["T_world_cam"]
                T_l_asa = ASAAstribotEmbodimentAdapter.tcp_world_to_asa_world(fr["left_tcp7"])
                T_r_asa = ASAAstribotEmbodimentAdapter.tcp_world_to_asa_world(fr["right_tcp7"])

                ep_head9[i] = matrix_to_pose9_asa(inv_ep_base @ T_cam)
                ep_left9[i] = matrix_to_pose9_asa(inv_ep_base @ T_l_asa)
                ep_right9[i] = matrix_to_pose9_asa(inv_ep_base @ T_r_asa)

                inv_sb = seg_inv_bases[i]
                sub_head9[i] = matrix_to_pose9_asa(inv_sb @ T_cam)
                sub_left9[i] = matrix_to_pose9_asa(inv_sb @ T_l_asa)
                sub_right9[i] = matrix_to_pose9_asa(inv_sb @ T_r_asa)

                left_grip[i] = fr["left_grip"]
                right_grip[i] = fr["right_grip"]

            ep_uid = f"{task_name}_seed{cur_seed}"
            frames_rel = f"frames/{ep_uid}"
            npz_rel = f"episodes/{ep_uid}.npz"
            (out_dir / frames_rel).mkdir(parents=True, exist_ok=True)
            (out_dir / "episodes").mkdir(parents=True, exist_ok=True)

            for i, fr in enumerate(raw_frames):
                Image.fromarray(fr["rgb224"]).save(out_dir / frames_rel / f"frame_{i:04d}.jpg", quality=92)

            np.savez_compressed(
                out_dir / npz_rel,
                ep_head9=ep_head9,
                ep_left_wrist9=ep_left9,
                ep_right_wrist9=ep_right9,
                sub_head9=sub_head9,
                sub_left_wrist9=sub_left9,
                sub_right_wrist9=sub_right9,
                left_grip=left_grip,
                right_grip=right_grip,
            )

            manifest["episodes"].append(
                {
                    "episode_uid": ep_uid,
                    "task_name": task_name,
                    "seed": int(cur_seed),
                    "num_frames": n_f,
                    "task_instruction": task_instr,
                    "subtask_segments": segments,
                    "frames_relpath": frames_rel,
                    "npz_relpath": npz_rel,
                }
            )
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            existing_by_task[task_name].add(int(cur_seed))
            collected += 1
            dt = time.monotonic() - t0
            print(
                f"[collect] {task_name} {collected}/{args.episodes_per_task} seed={cur_seed} "
                f"frames={n_f} sub_segs={len(segments)} ({dt:.1f}s)",
                flush=True,
            )
            cur_seed += 1

    print(f"[collect] COMPLETE! Total episodes={len(manifest['episodes'])} saved at {out_dir}", flush=True)


if __name__ == "__main__":
    main()
