"""Stage-2 Unified Continuous Cerebellum Inertia (Opus 5.5 §2.6).

Computes a bounded, zero-learned-parameter physical execution inertia signal:
    phi_t = max(phi_kin, phi_grip, phi_dw) in [0, 1]

where:
- phi_kin:  normalized EMA commanded/observed velocity across the 16 non-gripper joints
            (explicitly excluding the 2 gripper indices to avoid scale distortion);
- phi_grip: dedicated gripper commitment/transition activity across left & right grippers;
- phi_dw:   commit-anchored exponential dwell decay exp(-tau_t / tau_0) with
            tau_t = min(1, (t - t_c) / H_e), preventing immediate post-commit chatter
            at t = t_c even before arm velocity ramps up.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


@dataclass
class CerebellumConfig:
    mode: str = "physical_cerebellum"  # "physical_cerebellum" | "no_inertia_k0" | "fixed_period"
    ema_alpha: float = 0.35
    left_gripper_idx: int = 7
    right_gripper_idx: int = 15
    gripper_vel_scale: float = 0.15
    kin_vel_scale: float = 0.25
    horizon_steps: int = 16
    dwell_tau0: float = 0.55
    joint_std: Optional[list[float]] = None  # Optional 16-d training std D for non-gripper joints


@dataclass(frozen=True)
class InertiaSnapshot:
    step: int
    phi: float
    phi_kin: float
    phi_grip: float
    phi_dw: float
    tau: float


class ContinuousCerebellumInertia:
    """Tracks continuous low-level motor commitment without privileged env state."""

    def __init__(self, config: CerebellumConfig | None = None) -> None:
        self.config = config or CerebellumConfig()
        self.reset()

    def reset(self) -> None:
        self._prev_state: Optional[np.ndarray] = None
        self._ema_kin_vel: Optional[np.ndarray] = None
        self._ema_grip_vel: Optional[np.ndarray] = None
        self._commit_step: int = 0
        self._last_snapshot = InertiaSnapshot(
            step=0, phi=1.0, phi_kin=0.0, phi_grip=0.0, phi_dw=1.0, tau=0.0
        )

    def mark_commit(self, step: int) -> None:
        """Anchor the dwell timer t_c whenever a new subtask u_k is committed."""
        self._commit_step = int(step)

    @property
    def last_snapshot(self) -> InertiaSnapshot:
        return self._last_snapshot

    def _split_joints(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vec = np.asarray(state, dtype=np.float64).reshape(-1)
        g_idx = sorted({self.config.left_gripper_idx, self.config.right_gripper_idx})
        valid_g = [i for i in g_idx if 0 <= i < vec.shape[0]]
        grip = vec[valid_g] if valid_g else np.zeros(0, dtype=np.float64)
        mask = np.ones(vec.shape[0], dtype=bool)
        for i in valid_g:
            mask[i] = False
        kin = vec[mask]
        return kin, grip

    def update(
        self,
        step: int,
        state: np.ndarray,
        commanded_action_chunk: Optional[np.ndarray] = None,
    ) -> InertiaSnapshot:
        """Update inertia from observed joint state and optional action chunk."""
        cfg = self.config
        cur = np.asarray(state, dtype=np.float64).reshape(-1)
        kin_cur, grip_cur = self._split_joints(cur)

        # Estimate instantaneous velocity from commanded chunk if available, else state delta.
        if commanded_action_chunk is not None and np.asarray(commanded_action_chunk).size > 0:
            chunk = np.asarray(commanded_action_chunk, dtype=np.float64)
            if chunk.ndim == 1:
                chunk = chunk[None, :]
            if chunk.shape[0] >= 2:
                diffs = np.diff(chunk, axis=0)
                vel_vec = np.mean(np.abs(diffs), axis=0)
            else:
                vel_vec = np.abs(chunk[0] - cur) if chunk.shape[1] == cur.shape[0] else np.zeros_like(cur)
            kin_vel, grip_vel = self._split_joints(vel_vec)
        elif self._prev_state is not None and self._prev_state.shape == cur.shape:
            delta = np.abs(cur - self._prev_state)
            kin_vel, grip_vel = self._split_joints(delta)
        else:
            kin_vel = np.zeros_like(kin_cur)
            grip_vel = np.zeros_like(grip_cur)

        self._prev_state = cur.copy()

        alpha = float(np.clip(cfg.ema_alpha, 0.05, 1.0))
        if self._ema_kin_vel is None or self._ema_kin_vel.shape != kin_vel.shape:
            self._ema_kin_vel = kin_vel.copy()
        else:
            self._ema_kin_vel = alpha * kin_vel + (1.0 - alpha) * self._ema_kin_vel

        if self._ema_grip_vel is None or self._ema_grip_vel.shape != grip_vel.shape:
            self._ema_grip_vel = grip_vel.copy()
        else:
            self._ema_grip_vel = alpha * grip_vel + (1.0 - alpha) * self._ema_grip_vel

        # Normalize 16 non-gripper joints by training std D if provided.
        if cfg.joint_std is not None and len(cfg.joint_std) == self._ema_kin_vel.shape[0]:
            denom = np.maximum(np.asarray(cfg.joint_std, dtype=np.float64), 1e-4)
            norm_kin = float(np.linalg.norm(self._ema_kin_vel / denom) / math.sqrt(len(denom)))
        else:
            norm_kin = float(np.linalg.norm(self._ema_kin_vel) / max(math.sqrt(max(self._ema_kin_vel.size, 1)), 1.0))

        phi_kin = float(math.tanh(norm_kin / max(cfg.kin_vel_scale, 1e-6)))

        if self._ema_grip_vel.size > 0:
            grip_mag = float(np.max(self._ema_grip_vel))
            phi_grip = float(math.tanh(grip_mag / max(cfg.gripper_vel_scale, 1e-6)))
        else:
            phi_grip = 0.0

        elapsed = max(int(step) - self._commit_step, 0)
        horizon = max(int(cfg.horizon_steps), 1)
        tau = min(1.0, float(elapsed) / float(horizon))
        phi_dw = float(math.exp(-tau / max(cfg.dwell_tau0, 1e-6))) if elapsed < horizon else 0.0

        if cfg.mode == "no_inertia_k0":
            phi = 0.0
        elif cfg.mode == "fixed_period":
            phi = 0.0
        else:
            phi = float(np.clip(max(phi_kin, phi_grip, phi_dw), 0.0, 1.0))

        snap = InertiaSnapshot(
            step=int(step),
            phi=phi,
            phi_kin=phi_kin,
            phi_grip=phi_grip,
            phi_dw=phi_dw,
            tau=tau,
        )
        self._last_snapshot = snap
        return snap
