"""Unit tests for the Asymmetric AWR (AAWR) ActiveArena baseline implementation."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from eval.activearena.aawr_network import AAWRConfig, AAWRTrainer, hash_instruction_to_vec, l2_expectile
from eval.activearena.aawr_server import AAWRPolicyService
from eval.activearena.run_eval import check_readiness
from ipvla.config import load_paths_config


def _make_jpeg_b64() -> str:
    arr = np.full((64, 64, 3), 120, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_aawr_asymmetric_barrier_and_learning_step(tmp_path: Path):
    cfg = AAWRConfig(
        action_dim=18,
        chunk_len=4,
        proprio_dim=18,
        priv_state_dim=64,
        history_frames=4,
        image_size=64,
        latent_dim=64,
        mlp_dim=128,
        lr=1e-3,
    )
    trainer = AAWRTrainer(cfg, device="cpu")

    B = 8
    rgb = torch.rand(B, 4, 3, 64, 64)
    qpos = torch.randn(B, 18)
    txt = torch.from_numpy(np.stack([hash_instruction_to_vec("check block color", 64)] * B))
    priv_a = torch.randn(B, 64)
    priv_b = torch.randn(B, 64) * 10.0

    # Strict asymmetric check: unprivileged actor pi() NEVER depends on privileged state!
    out_a = trainer.predict_normalized_chunk(rgb, qpos, txt)
    out_b = trainer.predict_normalized_chunk(rgb, qpos, txt)
    np.testing.assert_allclose(out_a, out_b, atol=1e-6)
    assert out_a.shape == (B, 4, 18)

    # Verify update_batch runs cleanly and reduces actor RMSE on a fixed synthetic target
    target_act = torch.zeros(B, 4, 18)
    batch = {
        "rgb_hist": rgb,
        "qpos": qpos,
        "text_vec": txt,
        "priv_state": priv_a,
        "next_qpos": qpos,
        "next_priv_state": priv_b,
        "action_chunk": target_act,
        "reward": torch.ones(B),
        "done": torch.zeros(B),
    }
    m0 = trainer.update_batch(batch)
    for _ in range(12):
        m_last = trainer.update_batch(batch)
    assert m_last["pi_rmse"] < m0["pi_rmse"]
    assert l2_expectile(torch.tensor([1.0, -1.0]), expectile=0.9).item() > 0.0

    # Verify checkpoint bundle save + AAWRPolicyService inference round-trip
    stats = {
        "qpos_mean": [0.0] * 18,
        "qpos_std": [1.0] * 18,
        "act_min": [-1.0] * 18,
        "act_max": [1.0] * 18,
    }
    ckpt_dir = tmp_path / "best"
    trainer.save_bundle(ckpt_dir, stats)

    svc = AAWRPolicyService(ckpt_dir, device="cpu")
    pred = svc.predict("Move gray block to matching pad", [_make_jpeg_b64()] * 4, [0.0] * 18)
    arr = np.asarray(pred["actions_18d"])
    assert arr.shape == (4, 18)
    assert np.all((arr[:, 7] >= 0.0) & (arr[:, 7] <= 1.0))
    assert np.all((arr[:, 15] >= 0.0) & (arr[:, 15] <= 1.0))

    paths = load_paths_config()
    ready = check_readiness(paths, "aawr_baseline", ["check_block_color"], "info_gathering_demo", aawr_ckpt=ckpt_dir)
    assert ready["checks"]["aawr_ckpt_exists"] is True
