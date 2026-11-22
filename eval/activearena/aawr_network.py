"""Asymmetric Advantage-Weighted Regression (AAWR) Network & Trainer for ActiveArena.

Faithfully implements upstream `aawr-main/src/algorithm/asym_awr.py` +
`cfgs/tasks/gymnasium_pi0_activevision.yaml`:
  - Privileged Critics (`_Q_encoder`, `_V_encoder`, twin `_Qs`, `_V`) conditioned ONLY at
    training time on privileged environment state `priv_state` (ground-truth object poses,
    hidden target attributes, inspection progress, and 18-DoF robot proprioception):
      * TD target:        y_q = r_t + gamma * (1 - done_t) * V(s^+_{t+1})
      * Value target:     y_v = min(Q1_tgt(s^+_t, a_t), Q2_tgt(s^+_t, a_t))
      * Expectile V loss: L_V = E[ |tau - 1(y_v - V(s^+_t) < 0)| * (y_v - V(s^+_t))^2 ], tau = 0.9
  - Unprivileged Policy (`_pi_encoder`, `_pi`) conditioned ONLY on unprivileged causal RGB
    frame history `rgb_hist` + 18-DoF robot proprioception `qpos` + hashed task instruction
    embedding (strictly ZERO privileged state in policy forward at train or test time):
      * Advantage:        A_t = stop_grad(y_v - V(s^+_t))
      * AWR weight:       w_t = clamp(exp(A_scaling * A_t), max=100.0), A_scaling = 3.0
      * Policy loss:      L_pi = -E[ w_t * gaussian_logprob(pi(o_t) - a_t) ]
  - EMA target network update (`tau_ema = 0.005`) and orthogonal weight initialization.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AAWRConfig:
    action_dim: int = 18
    chunk_len: int = 8
    proprio_dim: int = 18
    priv_state_dim: int = 64
    history_frames: int = 4
    image_size: int = 128
    text_hash_dim: int = 64
    latent_dim: int = 256
    mlp_dim: int = 512
    num_q: int = 2
    discount: float = 0.99
    expectile: float = 0.9
    a_scaling: float = 3.0
    max_adv_weight: float = 100.0
    awr_filter: str = "exp"  # "exp" | "indicator"
    tau_ema: float = 0.005
    lr: float = 3e-4
    grad_clip_norm: float = 10.0


def hash_instruction_to_vec(instruction: str, dim: int = 64) -> np.ndarray:
    """Deterministic lightweight bag-of-ngrams hash embedding for task instructions."""
    vec = np.zeros(dim, dtype=np.float32)
    words = (instruction or "").strip().lower().split()
    tokens = words + [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
    if not tokens:
        return vec
    for tok in tokens:
        digest = hashlib.md5(tok.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:2], "little") % dim
        sign = 1.0 if (digest[2] & 1) == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 1e-6:
        vec /= norm
    return vec


def orthogonal_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


def l2_expectile(diff: torch.Tensor, expectile: float = 0.9) -> torch.Tensor:
    """Upstream AAWR expectile loss `h.l2_expectile(v_target - v, expectile)`."""
    weight = torch.where(diff >= 0, expectile, 1.0 - expectile)
    return (weight * (diff ** 2)).mean()


def gaussian_logprob(eps: torch.Tensor, log_std: float = 0.0) -> torch.Tensor:
    """Upstream AAWR `h.gaussian_logprob(prediction - action, 0)`."""
    residual = -0.5 * (eps ** 2) * math.exp(-2.0 * log_std) - log_std - 0.5 * math.log(2.0 * math.pi)
    return residual.mean(dim=-1, keepdim=True)


def ema_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p_on, p_tg in zip(online.parameters(), target.parameters()):
            p_tg.data.lerp_(p_on.data, tau)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class UnprivilegedVisualEncoder(nn.Module):
    """Encodes ONLY unprivileged `(rgb_hist, qpos_18, text_hash)` -> `z_pi`."""

    def __init__(self, cfg: AAWRConfig) -> None:
        super().__init__()
        in_ch = 3 * cfg.history_frames
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        vis_dim = 128 * 4 * 4
        self.fuse = nn.Sequential(
            nn.Linear(vis_dim + cfg.proprio_dim + cfg.text_hash_dim, cfg.mlp_dim),
            nn.LayerNorm(cfg.mlp_dim),
            nn.SiLU(),
            nn.Linear(cfg.mlp_dim, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
            nn.Tanh(),
        )

    def forward(self, rgb_hist: torch.Tensor, qpos: torch.Tensor, text_vec: torch.Tensor) -> torch.Tensor:
        # rgb_hist: (B, T, 3, H, W) in [0, 1] -> stack temporal channels (B, 3*T, H, W)
        b, t, c, h, w = rgb_hist.shape
        x = rgb_hist.reshape(b, t * c, h, w)
        v = self.conv(x)
        return self.fuse(torch.cat([v, qpos, text_vec], dim=-1))


class AAWRNetwork(nn.Module):
    """Asymmetric critic + unprivileged chunked actor matching upstream `asym_awr.py`."""

    def __init__(self, cfg: AAWRConfig) -> None:
        super().__init__()
        self.cfg = cfg
        flat_act_dim = cfg.chunk_len * cfg.action_dim

        # Privileged state encoders (training-only!)
        priv_in_dim = cfg.priv_state_dim + cfg.proprio_dim + cfg.text_hash_dim
        self._Q_encoder = _mlp(priv_in_dim, cfg.mlp_dim, cfg.latent_dim)
        self._V_encoder = _mlp(priv_in_dim, cfg.mlp_dim, cfg.latent_dim)

        # Unprivileged policy encoder & chunked actor head
        self._pi_encoder = UnprivilegedVisualEncoder(cfg)
        self._pi = _mlp(cfg.latent_dim, cfg.mlp_dim, flat_act_dim)

        # Twin Q ensemble & Expectile V head
        self._Qs = nn.ModuleList(
            [_mlp(cfg.latent_dim + flat_act_dim, cfg.mlp_dim, 1) for _ in range(cfg.num_q)]
        )
        self._V = _mlp(cfg.latent_dim, cfg.mlp_dim, 1)

        self.apply(orthogonal_init)
        for head in [*self._Qs, self._V]:
            last = head[-1]
            if isinstance(last, nn.Linear):
                last.weight.data.zero_()
                last.bias.data.zero_()

    def pi(self, rgb_hist: torch.Tensor, qpos: torch.Tensor, text_vec: torch.Tensor) -> torch.Tensor:
        z_pi = self._pi_encoder(rgb_hist, qpos, text_vec)
        return torch.tanh(self._pi(z_pi))

    def V(self, priv_full: torch.Tensor) -> torch.Tensor:
        zv = self._V_encoder(priv_full)
        return self._V(zv)

    def Q(self, priv_full: torch.Tensor, flat_action: torch.Tensor, return_type: str = "min") -> torch.Tensor:
        zq = self._Q_encoder(priv_full)
        return self.Q_from_latent(zq, flat_action, return_type=return_type)

    def Q_from_latent(self, zq: torch.Tensor, flat_action: torch.Tensor, return_type: str = "min") -> torch.Tensor:
        x = torch.cat([zq, flat_action], dim=-1)
        all_q = torch.stack([q(x) for q in self._Qs], dim=0)  # (num_q, B, 1)
        if return_type == "all":
            return all_q
        if return_type == "min":
            return all_q.min(dim=0).values
        if return_type == "avg":
            return all_q.mean(dim=0)
        raise ValueError(f"Unsupported return_type: {return_type}")


class AAWRTrainer:
    """Single-step update & checkpointing matching upstream `AAWR` class."""

    def __init__(self, cfg: AAWRConfig, device: str | torch.device = "cpu") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = AAWRNetwork(cfg).to(self.device)
        self.model_target = deepcopy(self.model).to(self.device)
        self.model_target.eval()
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    def update_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Run one asymmetric critic + advantage-weighted actor update."""
        self.model.train()
        rgb = batch["rgb_hist"].to(self.device)
        qpos = batch["qpos"].to(self.device)
        text_vec = batch["text_vec"].to(self.device)
        priv = batch["priv_state"].to(self.device)
        next_qpos = batch["next_qpos"].to(self.device)
        next_priv = batch["next_priv_state"].to(self.device)
        action = batch["action_chunk"].to(self.device).reshape(rgb.shape[0], -1)
        reward = batch["reward"].to(self.device).reshape(-1, 1)
        done = batch["done"].to(self.device).reshape(-1, 1).float()

        priv_full = torch.cat([priv, qpos, text_vec], dim=-1)
        next_priv_full = torch.cat([next_priv, next_qpos, text_vec], dim=-1)

        zq = self.model._Q_encoder(priv_full)
        v = self.model.V(priv_full)

        with torch.no_grad():
            next_v = self.model_target.V(next_priv_full)
            td_targets = reward + self.cfg.discount * (1.0 - done) * next_v
            zq_tgt = self.model_target._Q_encoder(priv_full)
            v_target = self.model_target.Q_from_latent(zq_tgt, action, return_type="min")

        qs = self.model.Q_from_latent(zq, action, return_type="all")
        q_loss = F.mse_loss(qs, td_targets.unsqueeze(0).expand_as(qs))
        v_loss = l2_expectile(v_target - v, expectile=self.cfg.expectile)

        pred_action = self.model.pi(rgb, qpos, text_vec)
        log_probs = gaussian_logprob(pred_action - action, 0.0)
        adv = (v_target - v).detach()
        if self.cfg.awr_filter == "indicator":
            adv_weights = (adv > 0.0).float()
        else:
            adv_weights = torch.exp(self.cfg.a_scaling * adv).clamp_max(self.cfg.max_adv_weight)
        pi_loss = -(adv_weights * log_probs).mean()

        total_loss = q_loss + v_loss + pi_loss
        self.optim.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        )
        self.optim.step()

        ema_update(self.model._Q_encoder, self.model_target._Q_encoder, self.cfg.tau_ema)
        ema_update(self.model._V_encoder, self.model_target._V_encoder, self.cfg.tau_ema)
        ema_update(self.model._Qs, self.model_target._Qs, self.cfg.tau_ema)
        ema_update(self.model._V, self.model_target._V, self.cfg.tau_ema)

        rmse = float(F.mse_loss(pred_action.detach(), action).sqrt().item())
        self.model.eval()
        return {
            "loss": float(total_loss.item()),
            "q_loss": float(q_loss.item()),
            "v_loss": float(v_loss.item()),
            "pi_loss": float(pi_loss.item()),
            "pi_rmse": rmse,
            "q_mean": float(qs.mean().item()),
            "v_mean": float(v.mean().item()),
            "adv_mean": float(adv.mean().item()),
            "adv_pos_ratio": float((adv > 0).float().mean().item()),
            "adv_weight_mean": float(adv_weights.mean().item()),
            "grad_norm": grad_norm,
        }

    @torch.no_grad()
    def predict_normalized_chunk(
        self, rgb_hist: torch.Tensor, qpos: torch.Tensor, text_vec: torch.Tensor
    ) -> np.ndarray:
        self.model.eval()
        out = self.model.pi(
            rgb_hist.to(self.device),
            qpos.to(self.device),
            text_vec.to(self.device),
        )
        return out.reshape(-1, self.cfg.chunk_len, self.cfg.action_dim).cpu().numpy()

    def save_bundle(self, out_dir: str | Path, norm_stats: dict[str, Any]) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"config": asdict(self.cfg), "model": self.model.state_dict()},
            out / "ckpt.pt",
        )
        import json

        (out / "aawr_norm_stats.json").write_text(
            json.dumps(norm_stats, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load_for_inference(
        cls, ckpt_dir: str | Path, device: str = "cpu"
    ) -> tuple["AAWRTrainer", dict[str, Any]]:
        import json

        d = Path(ckpt_dir)
        payload = torch.load(d / "ckpt.pt", map_location=device, weights_only=False)
        cfg = AAWRConfig(**payload["config"])
        trainer = cls(cfg, device=device)
        trainer.model.load_state_dict(payload["model"])
        trainer.model.eval()
        stats = json.loads((d / "aawr_norm_stats.json").read_text(encoding="utf-8"))
        return trainer, stats
