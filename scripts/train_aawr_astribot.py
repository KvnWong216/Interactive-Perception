#!/usr/bin/env python3
"""Offline Asymmetric Advantage-Weighted Regression (AAWR) Trainer for ActiveArena.

Trains privileged twin-Q & expectile-V critics alongside an unprivileged visual/proprio
chunked policy (`eval/activearena/aawr_network.py`), saving `best/ckpt.pt` and
`best/aawr_norm_stats.json` ready for `eval/activearena/run_eval.py --mode aawr_baseline`.

Runs comfortably on a single 8 GB RTX 4060 (~1.2 GB VRAM) or 24 GB RTX 3090.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.activearena.aawr_network import AAWRConfig, AAWRTrainer, hash_instruction_to_vec


def load_episodes(data_root: Path) -> list[dict[str, Any]]:
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    ep_dir = data_root / "episodes"
    episodes: list[dict[str, Any]] = []
    for meta in manifest["episodes"]:
        f = ep_dir / meta["file"]
        if not f.is_file():
            continue
        with np.load(f, allow_pickle=False) as d:
            instr = str(d["instruction"].item()) if d["instruction"].ndim == 0 else str(d["instruction"])
            episodes.append(
                {
                    "rgb": d["rgb"],  # (T, H, W, 3) uint8
                    "qpos": d["qpos"].astype(np.float32),
                    "action": d["action"].astype(np.float32),
                    "priv_state": d["priv_state"].astype(np.float32),
                    "reward": d["reward"].astype(np.float32),
                    "done": d["done"].astype(np.float32),
                    "text_vec": hash_instruction_to_vec(instr, 64),
                    "kind": meta.get("kind", "expert"),
                }
            )
    if not episodes:
        raise RuntimeError(f"No episodes found under {ep_dir}")
    return episodes


def compute_norm_stats(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    all_q = np.concatenate([ep["qpos"] for ep in episodes], axis=0)
    exp_acts = [ep["action"] for ep in episodes if ep["kind"] == "expert"] or [ep["action"] for ep in episodes]
    all_a = np.concatenate(exp_acts, axis=0)
    q_mean = all_q.mean(axis=0)
    q_std = np.maximum(all_q.std(axis=0), 1e-4)
    a_min = np.percentile(all_a, 1.0, axis=0) - 0.02
    a_max = np.percentile(all_a, 99.0, axis=0) + 0.02
    a_min[7], a_max[7] = 0.0, 1.0
    a_min[15], a_max[15] = 0.0, 1.0
    return {
        "qpos_mean": q_mean.tolist(),
        "qpos_std": q_std.tolist(),
        "act_min": a_min.tolist(),
        "act_max": a_max.tolist(),
    }


def sample_batch(
    episodes: list[dict[str, Any]],
    stats: dict[str, Any],
    cfg: AAWRConfig,
    batch_size: int,
    rng: random.Random,
    history_stride: int = 4,
) -> dict[str, torch.Tensor]:
    q_mean = np.asarray(stats["qpos_mean"], dtype=np.float32)
    q_std = np.asarray(stats["qpos_std"], dtype=np.float32)
    a_min = np.asarray(stats["act_min"], dtype=np.float32)
    a_max = np.asarray(stats["act_max"], dtype=np.float32)
    a_mid = 0.5 * (a_max + a_min)
    a_half = np.maximum(0.5 * (a_max - a_min), 1e-5)

    rgbs, q_list, txt_list, priv_list = [], [], [], []
    nq_list, npriv_list, act_list, rew_list, done_list = [], [], [], [], []

    for _ in range(batch_size):
        ep = rng.choice(episodes)
        T = ep["qpos"].shape[0]
        t = rng.randrange(0, T)

        # Causal 4-frame RGB history
        h_idxs = [max(0, t - history_stride * k) for k in reversed(range(cfg.history_frames))]
        hist = ep["rgb"][h_idxs].astype(np.float32) / 255.0  # (K, H, W, 3)
        rgbs.append(np.transpose(hist, (0, 3, 1, 2)))

        q_list.append((ep["qpos"][t] - q_mean) / q_std)
        txt_list.append(ep["text_vec"])
        priv_list.append(ep["priv_state"][t])

        nt = min(T - 1, t + cfg.chunk_len)
        nq_list.append((ep["qpos"][nt] - q_mean) / q_std)
        npriv_list.append(ep["priv_state"][nt])

        # Action chunk normalized to [-1, 1]
        c_idxs = [min(T - 1, t + k) for k in range(cfg.chunk_len)]
        raw_chunk = ep["action"][c_idxs]
        norm_chunk = np.clip((raw_chunk - a_mid[None, :]) / a_half[None, :], -1.0, 1.0)
        act_list.append(norm_chunk)

        # Chunk discounted return & termination
        disc_r = 0.0
        d_flag = 0.0
        for k in range(cfg.chunk_len):
            idx = t + k
            if idx >= T:
                d_flag = 1.0
                break
            disc_r += (cfg.discount ** k) * float(ep["reward"][idx])
            if ep["done"][idx] > 0.5:
                d_flag = 1.0
                break
        rew_list.append(disc_r)
        done_list.append(d_flag)

    return {
        "rgb_hist": torch.from_numpy(np.stack(rgbs, axis=0)),
        "qpos": torch.from_numpy(np.stack(q_list, axis=0)),
        "text_vec": torch.from_numpy(np.stack(txt_list, axis=0)),
        "priv_state": torch.from_numpy(np.stack(priv_list, axis=0)),
        "next_qpos": torch.from_numpy(np.stack(nq_list, axis=0)),
        "next_priv_state": torch.from_numpy(np.stack(npriv_list, axis=0)),
        "action_chunk": torch.from_numpy(np.stack(act_list, axis=0)),
        "reward": torch.tensor(rew_list, dtype=torch.float32),
        "done": torch.tensor(done_list, dtype=torch.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AAWR Baseline on ActiveArena Offline Data")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--expectile", type=float, default=0.9)
    parser.add_argument("--a-scaling", type=float, default=3.0)
    parser.add_argument("--chunk-len", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    episodes = load_episodes(args.data_root.resolve())
    stats = compute_norm_stats(episodes)
    img_sz = int(episodes[0]["rgb"].shape[1])

    cfg = AAWRConfig(
        chunk_len=args.chunk_len,
        image_size=img_sz,
        expectile=args.expectile,
        a_scaling=args.a_scaling,
        lr=args.lr,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trainer = AAWRTrainer(cfg, device=device)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    best_pi_rmse = float("inf")
    history: list[dict[str, Any]] = []
    t0 = time.monotonic()

    for step in range(1, args.max_steps + 1):
        batch = sample_batch(episodes, stats, cfg, args.batch_size, rng)
        metrics = trainer.update_batch(batch)
        if step % args.log_every == 0 or step == 1:
            rec = {"step": step, "elapsed_s": round(time.monotonic() - t0, 1), **metrics}
            history.append(rec)
            print(json.dumps(rec), flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            trainer.save_bundle(out_dir / "latest", stats)
            if metrics["pi_rmse"] <= best_pi_rmse:
                best_pi_rmse = metrics["pi_rmse"]
                trainer.save_bundle(out_dir / "best", stats)

    (out_dir / "training_metrics.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"[aawr-train] Finished {args.max_steps} steps -> {out_dir / 'best'}", flush=True)


if __name__ == "__main__":
    main()
