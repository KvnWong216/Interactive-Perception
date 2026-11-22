"""Standalone L2-normalized Ridge + positive-slope temperature confidence head.

Bit-for-bit identical forward pass to `ipvla/stage1/train_vlm.py`:
    x_std  = (h_last - h_mean) / clamp_min(h_std, 1e-5)
    x_unit = x_std / ||x_std||_2
    logit  = temp_a_pos * (w^T x_unit + b) + temp_b
    c_k    = sigmoid(logit)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np


class RidgeTempConfidenceHead:
    """NumPy inference wrapper for `stage1_v2_confidence_head_ridge_temp.pt`."""

    def __init__(
        self,
        weight: np.ndarray,
        bias: float,
        h_mean: Optional[np.ndarray] = None,
        h_std: Optional[np.ndarray] = None,
        temp_a_pos: float = 1.0,
        temp_b: float = 0.0,
    ) -> None:
        self.weight = np.asarray(weight, dtype=np.float64).reshape(-1)
        self.bias = float(bias)
        self.h_mean = (
            np.asarray(h_mean, dtype=np.float64).reshape(-1)
            if h_mean is not None
            else np.zeros_like(self.weight)
        )
        self.h_std = (
            np.maximum(np.asarray(h_std, dtype=np.float64).reshape(-1), 1e-5)
            if h_std is not None
            else np.ones_like(self.weight)
        )
        self.temp_a_pos = float(temp_a_pos)
        self.temp_b = float(temp_b)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "RidgeTempConfidenceHead":
        import torch

        ckpt: dict[str, Any] = torch.load(str(path), map_location="cpu", weights_only=False)

        def _to_np(val):
            if val is None:
                return None
            return val.detach().cpu().numpy() if hasattr(val, "detach") else np.asarray(val)

        w = _to_np(ckpt.get("weight", ckpt.get("ridge_coef")))
        if w is None:
            raise KeyError(f"Missing weight/ridge_coef in confidence checkpoint: {path}")
        b_raw = _to_np(ckpt.get("bias", ckpt.get("ridge_intercept", 0.0)))
        b = float(np.asarray(b_raw).reshape(-1)[0])
        h_mean = _to_np(ckpt.get("h_mean"))
        h_std = _to_np(ckpt.get("h_std"))
        a_pos = float(ckpt.get("temp_a_pos", ckpt.get("temp_scale_a_pos", 1.0)))
        tb = float(ckpt.get("temp_b", ckpt.get("temp_bias_b", 0.0)))
        return cls(
            weight=w,
            bias=b,
            h_mean=h_mean,
            h_std=h_std,
            temp_a_pos=a_pos,
            temp_b=tb,
        )

    def predict_proba(self, h_last: np.ndarray) -> float:
        x = np.asarray(h_last, dtype=np.float64).reshape(-1)
        x_std = (x - self.h_mean) / self.h_std
        norm = float(np.linalg.norm(x_std))
        x_unit = x_std / max(norm, 1e-12)
        raw_logit = float(np.dot(self.weight, x_unit) + self.bias)
        cal_logit = max(min(self.temp_a_pos * raw_logit + self.temp_b, 30.0), -30.0)
        return float(1.0 / (1.0 + math.exp(-cal_logit)))
