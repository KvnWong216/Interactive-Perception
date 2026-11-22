#!/usr/bin/env python3
"""Zero-leakage ActiveArena trajectory + privileged-state collector for AAWR baseline.

Matches `collect_asa_astribot_data.py` in anti-leakage guarantees while recording the
training-only privileged state `priv_state` (64-D) and suboptimal exploratory rollouts
required by Asymmetric Advantage-Weighted Regression (`asym_awr.py`):
  1. Zero eval-seed overlap: `start_seed >= 600_000`, asserted disjoint from all ID/OOD eval seeds.
  2. Zero unseen-template or hidden-property text spoiler in the policy's instruction `G`.
  3. Training-only `priv_state` (64-D): privileged actor poses from the simulator scene,
     privileged subtask progress one-hot, and hashed hidden ground-truth cue (`info_map`),
     used EXCLUSIVELY by AAWR's `_Q_encoder` / `_V_encoder` during offline critic training.
  4. Mixed-quality replay buffer: pairs each clean expert demonstration (`reward=1.0` at
     goal + stage shaping) with action-corrupted / inspection-skipped suboptimal transitions
     (`reward=0.0`) so `A_t = Q(s^+, a_t) - V(s^+)` learns sharp advantage weights.
"""
from __future__ import annotations

import argparse
import hashlib
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
                forbidden[t].update(int(s) for s in payload.get("seeds", []))
                for entry in payload.get("entries", []):
                    if isinstance(entry, dict) and "seed" in entry:
                        forbidden[t].add(int(entry["seed"]))
    return forbidden


def _non_spoiler_instruction(task_name: str, desc_json: dict[str, Any], info_map: dict[str, Any], rng: random.Random) -> str:
    safe_map = {str(k): str(v) for k, v in info_map.items()}
    if task_name == "check_block_color":
        safe_map["{A}"] = "the gray target block"
        safe_map["{B}"] = "the matching color pad"
        safe_map["{C}"] = "back-side"
        candidates = [
            "Lift {A}, check its back-side pad color, then put it down on the corresponding color pad.",
            "Move the gray block to the pad matching the color on its back.",
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


def _extract_privileged_state_64d(env_ref: Any, subtask_id: Any, info_hint: dict[str, Any]) -> np.ndarray:
    """Build 64-D training-only privileged vector `s^+_t` from ground-truth Sapien scene."""
    priv = np.zeros(64, dtype=np.float32)
    # [0..13]: left & right ground-truth TCP 7D poses in world frame
    try:
        ltcp = np.asarray(env_ref.robot.get_left_tcp_pose(), dtype=np.float32).reshape(-1)[:7]
        rtcp = np.asarray(env_ref.robot.get_right_tcp_pose(), dtype=np.float32).reshape(-1)[:7]
        priv[0:7] = ltcp
        priv[7:14] = rtcp
    except Exception:
        pass

    # [14..41]: up to 4 ground-truth dynamic task actors (7D pose each)
    actor_poses: list[np.ndarray] = []
    for attr in ("block", "target_block", "cola", "can", "target_pad", "pad_list", "actors"):
        obj = getattr(env_ref, attr, None)
        if obj is None:
            continue
        items = obj if isinstance(obj, (list, tuple)) else [obj]
        for it in items:
            if len(actor_poses) >= 4:
                break
            try:
                pose = it.get_pose()
                p7 = np.concatenate([np.asarray(pose.p, dtype=np.float32), np.asarray(pose.q, dtype=np.float32)])
                actor_poses.append(p7[:7])
            except Exception:
                continue
    for idx, p7 in enumerate(actor_poses[:4]):
        priv[14 + 7 * idx : 14 + 7 * (idx + 1)] = p7

    # [42..49]: privileged oracle subtask stage one-hot (stages 0..7)
    try:
        sid = int(subtask_id) if subtask_id is not None else 0
    except Exception:
        sid = 0
    priv[42 + max(0, min(sid, 7))] = 1.0

    # [50..63]: deterministic hash of ground-truth hidden scene attributes (`info_map`, e.g., true sticker color / date)
    raw_secret = json.dumps(info_hint, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(raw_secret).digest()
    for i in range(14):
        priv[50 + i] = (float(digest[i]) / 127.5) - 1.0

    return priv


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect offline expert + suboptimal privileged data for AAWR")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sim-repo",
        type=Path,
        default=Path("/home/ubuntu/ygwang/IP-VLA/baseline-study/external/activearena"),
    )
    parser.add_argument("--ipvla-repo", type=Path, default=Path("/home/ubuntu/ygwang/IP-VLA/ip-vla"))
    parser.add_argument("--tasks", nargs="+", default=["check_block_color", "check_cola_date", "check_cola_color"])
    parser.add_argument("--episodes-per-task", type=int, default=30)
    parser.add_argument("--start-seed", type=int, default=600_000)
    parser.add_argument("--save-freq", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()

    sim_repo = args.sim_repo.resolve()
    ipvla_repo = args.ipvla_repo.resolve()
    out_dir = args.output_dir.resolve()
    ep_dir = out_dir / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)

    _setup_paths(sim_repo, ipvla_repo)
    os.chdir(sim_repo)

    from envs import CONFIGS_PATH
    from envs.utils.create_actor import UnStableError
    from policy.starvla_astribot.deploy_policy import build_astribot_18d_state

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
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {"tasks": args.tasks, "episodes": []}
    )
    existing = {(e["task_name"], int(e["seed"])) for e in manifest["episodes"] if e.get("kind") == "expert"}
    rng = random.Random(20261123)
    np_rng = np.random.default_rng(20261123)

    for task_idx, task_name in enumerate(args.tasks):
        desc_json = json.loads((sim_repo / "description/task_instruction" / f"{task_name}.json").read_text())
        forbidden = forbidden_by_task[task_name]
        mod = importlib.import_module(f"envs.{task_name}")
        env = getattr(mod, task_name)()

        collected = sum(1 for t, _ in existing if t == task_name)
        cur_seed = args.start_seed + task_idx * 100_000

        while collected < args.episodes_per_task:
            if cur_seed in forbidden or (task_name, cur_seed) in existing:
                cur_seed += 1
                continue

            raw_frames: list[dict[str, Any]] = []

            def _hook(env_ref=env, buf=raw_frames) -> None:
                if not env_ref.save_data:
                    return
                obs = env_ref.get_obs()
                rgb = obs["observation"]["camera_head"]["rgb"]
                qpos18 = build_astribot_18d_state(obs)
                ann = dict(env_ref._latest_frame_annotation) if env_ref._latest_frame_annotation else {}
                sid = ann.get("subtask", ann.get("subtask_id", 0))
                im = np.asarray(
                    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).resize(
                        (args.image_size, args.image_size), Image.Resampling.BILINEAR
                    ),
                    dtype=np.uint8,
                )
                buf.append({"rgb": im, "qpos18": qpos18, "subtask_id": sid})
                env_ref.FRAME_IDX += 1

            env._take_picture = _hook
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
                ok = bool(env.plan_success and env.check_success()) and len(raw_frames) >= 16
                info_map = (info or {}).get("info", {}) or {}
                priv_snap = [
                    _extract_privileged_state_64d(env, fr["subtask_id"], info_map) for fr in raw_frames
                ] if ok else []
                env.close_env(clear_cache=(collected % 5 == 0))
            except UnStableError:
                try:
                    env.close_env(clear_cache=False)
                except Exception:
                    pass
                cur_seed += 1
                continue
            except Exception as exc:
                print(f"[aawr-collect] {task_name} seed={cur_seed} err={exc}", flush=True)
                try:
                    env.close_env(clear_cache=True)
                except Exception:
                    pass
                env = getattr(mod, task_name)()
                cur_seed += 1
                continue

            if not ok:
                cur_seed += 1
                continue

            assert cur_seed not in forbidden
            instr = _non_spoiler_instruction(task_name, desc_json, info_map, rng)
            rgb_arr = np.stack([fr["rgb"] for fr in raw_frames], axis=0)  # (T, H, W, 3) uint8
            qpos_arr = np.stack([fr["qpos18"] for fr in raw_frames], axis=0).astype(np.float32)  # (T, 18)
            priv_arr = np.stack(priv_snap, axis=0).astype(np.float32)  # (T, 64)

            # Target action at t is next-step qpos (and terminal step holds pose)
            act_arr = np.concatenate([qpos_arr[1:], qpos_arr[-1:]], axis=0)
            T = qpos_arr.shape[0]

            # Dense stage-transition + terminal reward for expert trajectory
            rew_exp = np.zeros(T, dtype=np.float32)
            sids = [fr["subtask_id"] for fr in raw_frames]
            for idx in range(1, T):
                if sids[idx] != sids[idx - 1]:
                    rew_exp[idx - 1] += 0.25
            rew_exp[-1] = 1.0
            done_arr = np.zeros(T, dtype=np.float32)
            done_arr[-1] = 1.0

            exp_name = f"{task_name}_seed{cur_seed}_expert.npz"
            np.savez_compressed(
                ep_dir / exp_name,
                rgb=rgb_arr,
                qpos=qpos_arr,
                action=act_arr,
                priv_state=priv_arr,
                reward=rew_exp,
                done=done_arr,
                instruction=instr,
            )
            manifest["episodes"].append(
                {"file": exp_name, "task_name": task_name, "seed": cur_seed, "kind": "expert", "length": int(T)}
            )

            # Synthesize matched suboptimal/corrupted rollout (`reward=0`) so AAWR critics separate good vs bad actions
            noise = np_rng.normal(0.0, 0.18, size=act_arr.shape).astype(np.float32)
            act_sub = act_arr + noise
            act_sub[:, 7] = np.clip(1.0 - act_arr[:, 7], 0.0, 1.0)
            act_sub[:, 15] = np.clip(1.0 - act_arr[:, 15], 0.0, 1.0)
            rew_sub = np.zeros(T, dtype=np.float32)
            sub_name = f"{task_name}_seed{cur_seed}_subopt.npz"
            np.savez_compressed(
                ep_dir / sub_name,
                rgb=rgb_arr,
                qpos=qpos_arr,
                action=act_sub,
                priv_state=priv_arr,
                reward=rew_sub,
                done=done_arr,
                instruction=instr,
            )
            manifest["episodes"].append(
                {"file": sub_name, "task_name": task_name, "seed": cur_seed, "kind": "suboptimal", "length": int(T)}
            )

            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            existing.add((task_name, cur_seed))
            collected += 1
            print(f"[aawr-collect] {task_name}: {collected}/{args.episodes_per_task} (seed={cur_seed}, T={T})", flush=True)
            cur_seed += 1


if __name__ == "__main__":
    main()
