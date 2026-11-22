"""Rigid-body embodiment alignment between ASA (EgoActiveFlowModelv2) and ActiveArena Astribot S1.

Bridges ASA's 29-DoF egocentric human/robot representation:
  - `view`        (9D): head pose `[x, y, z, r11, r21, r31, r12, r22, r32]` in `T_world_base` (FLU)
  - `left_wrist`  (9D): left TCP pose `[x, y, z, rot6d]` in `T_world_base` (FLU: +X approach, +Y left, +Z up)
  - `right_wrist` (9D): right TCP pose `[x, y, z, rot6d]` in `T_world_base` (FLU: +X approach, +Y left, +Z up)
  - `grippers`    (2D): `[left_grip, right_grip]` in `[0, 1]` (`0.0 = closed`, `1.0 = open`)

and ActiveArena's 19-DoF Astribot S1 whole-body embodiment in Sapien:
  - `camera_head`: FLU link whose orientation satisfies exact ZYX kinematics
                   `R_robot0^T @ R_world_cam(q_torso, q_h1, q_h2) = R_z(q_torso + q_h1) @ R_y(q_h2)`
  - `left/right TCP`: `get_left_tcp_pose` / `get_right_tcp_pose` are FRD (`+X` approach, `+Y` right, `+Z` down),
                      related to ASA's FLU wrist frame by exact 180-deg roll `R_EE_TO_ASA = diag(1, -1, -1)`.
  - `grippers`: identical `[0, 1]` command semantics (`0.0 = closed`, `1.0 = open`).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.spatial.transform import Rotation


# Exact SO(3) 180-degree roll around +X (finger approach axis): converts FRD <-> FLU (self-inverse, det=+1).
R_EE_TO_ASA = np.diag([1.0, -1.0, -1.0]).astype(np.float64)

# ActiveArena `_trans_endpose`: `p_tcp = p_ee + R_ee @ [0.12, 0, 0]` (`is_endpose=True` vs `False`).
TCP_TO_EE_OFFSET_IN_EE = np.array([-0.12, 0.0, 0.0], dtype=np.float64)

# Canonical initial gaze pitch (rad) matching ASA's Monte02 / HoloAssist tabletop mean (~26.6 deg = 0.465 rad).
CANONICAL_BASE_PITCH_RAD = 0.465

# Joint limits from `astribot_whole_body_maniskill_merged.urdf`
HEAD_JOINT1_LIMITS = (-1.57, 1.57)   # head pan / yaw (+ = left)
HEAD_JOINT2_LIMITS = (-1.22, 1.22)   # head tilt / pitch (+ = down)
TORSO_JOINT4_LIMITS = (-1.20, 1.20)  # waist yaw (+ = left)


def matrix_to_rot6d_cols(rot_mat: np.ndarray) -> np.ndarray:
    """ASA convention (`data/transformation.py`): first TWO COLUMNS concatenated `[col0, col1]`."""
    r = np.asarray(rot_mat, dtype=np.float64)
    return np.concatenate([r[..., :, 0], r[..., :, 1]], axis=-1)


def rot6d_cols_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Gram-Schmidt orthonormalization of ASA column-wise rot6d -> SO(3) matrix."""
    arr = np.asarray(rot6d, dtype=np.float64)
    a1 = arr[..., 0:3]
    a2 = arr[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    return Rotation.from_quat(q, scalar_first=True).as_matrix()


def matrix_to_quat_wxyz(rot_mat: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(rot_mat, dtype=np.float64)).as_quat(scalar_first=True)


def pose7_to_matrix(pose7_xyz_wxyz: np.ndarray) -> np.ndarray:
    p7 = np.asarray(pose7_xyz_wxyz, dtype=np.float64).reshape(7)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_matrix(p7[3:7])
    T[:3, 3] = p7[:3]
    return T


def matrix_to_pose7(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    return np.concatenate([T[:3, 3], matrix_to_quat_wxyz(T[:3, :3])], axis=0)


def matrix_to_pose9_asa(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    return np.concatenate([T[:3, 3], matrix_to_rot6d_cols(T[:3, :3])], axis=-1)


def pose9_asa_to_matrix(pose9: np.ndarray) -> np.ndarray:
    p9 = np.asarray(pose9, dtype=np.float64)
    out = np.broadcast_to(np.eye(4, dtype=np.float64), p9.shape[:-1] + (4, 4)).copy()
    out[..., :3, :3] = rot6d_cols_to_matrix(p9[..., 3:9])
    out[..., :3, 3] = p9[..., :3]
    return out


@dataclass(frozen=True)
class FieldNormStats:
    mean: np.ndarray
    std: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray

    def norm_pos(self, pose9: np.ndarray) -> np.ndarray:
        arr = np.asarray(pose9, dtype=np.float32).copy()
        pos = np.clip(arr[..., :3], self.lower_bound, self.upper_bound)
        arr[..., :3] = (pos - self.mean) / (self.std + 1e-6)
        return arr

    def denorm_pos(self, pose9: np.ndarray) -> np.ndarray:
        arr = np.asarray(pose9, dtype=np.float32).copy()
        arr[..., :3] = arr[..., :3] * (self.std + 1e-6) + self.mean
        return arr


@dataclass
class ASANormBundle:
    profile_name: str
    head: FieldNormStats
    left_wrist: FieldNormStats
    right_wrist: FieldNormStats

    @classmethod
    def from_asa_json(cls, stats_json_path: str | Path, source_key: str = "HoloAssist", use_step_base: bool = False) -> "ASANormBundle":
        raw = json.loads(Path(stats_json_path).read_text(encoding="utf-8"))[source_key]
        suffix = "in_step_base" if use_step_base and f"current/head/pose_in_step_base" in raw else "in_base"

        def _make(key: str) -> FieldNormStats:
            d = raw[key]
            return FieldNormStats(
                mean=np.asarray(d["mean"], dtype=np.float32),
                std=np.maximum(np.asarray(d["std"], dtype=np.float32), 1e-3),
                lower_bound=np.asarray(d["lower_bound"], dtype=np.float32),
                upper_bound=np.asarray(d["upper_bound"], dtype=np.float32),
            )

        return cls(
            profile_name=f"{source_key}:{suffix}",
            head=_make(f"current/head/pose_{suffix}"),
            left_wrist=_make(f"current/left_hand/wrist_{suffix}"),
            right_wrist=_make(f"current/right_hand/wrist_{suffix}"),
        )

    @classmethod
    def from_astribot_stats_json(cls, stats_json_path: str | Path) -> "ASANormBundle":
        raw = json.loads(Path(stats_json_path).read_text(encoding="utf-8"))

        def _make(key: str) -> FieldNormStats:
            d = raw[key]
            return FieldNormStats(
                mean=np.asarray(d["mean"], dtype=np.float32),
                std=np.maximum(np.asarray(d["std"], dtype=np.float32), 1e-3),
                lower_bound=np.asarray(d["lower_bound"], dtype=np.float32),
                upper_bound=np.asarray(d["upper_bound"], dtype=np.float32),
            )

        return cls(
            profile_name=f"stage3_ckpt:{Path(stats_json_path).parent.name}",
            head=_make("head"),
            left_wrist=_make("left_wrist"),
            right_wrist=_make("right_wrist"),
        )

    @classmethod
    def astribot_human_aligned(cls) -> "ASANormBundle":
        """Calibrated to Astribot S1 tabletop workspace in `level_head` base frame (`z_cam=1.266m`, `z_table=0.74m`).

        Matches CaptainCook4D (`mean_z = -0.42m`) & Monte02 (`mean_z = -0.46m`) so `z_norm = 0`
        lands at `z_world ≈ 0.84m` (10cm above tabletop) and `z_norm = -0.7` reaches `z_world ≈ 0.75m` (grasp height).
        """
        return cls(
            profile_name="astribot_human_aligned",
            head=FieldNormStats(
                mean=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                std=np.array([0.16, 0.18, 0.12], dtype=np.float32),
                lower_bound=np.array([-5.0, -5.0, -1.0], dtype=np.float32),
                upper_bound=np.array([5.0, 5.0, 1.0], dtype=np.float32),
            ),
            left_wrist=FieldNormStats(
                mean=np.array([0.38, 0.26, -0.42], dtype=np.float32),
                std=np.array([0.16, 0.18, 0.14], dtype=np.float32),
                lower_bound=np.array([-5.0, -5.0, -1.0], dtype=np.float32),
                upper_bound=np.array([5.0, 5.0, 1.0], dtype=np.float32),
            ),
            right_wrist=FieldNormStats(
                mean=np.array([0.38, -0.26, -0.42], dtype=np.float32),
                std=np.array([0.16, 0.18, 0.14], dtype=np.float32),
                lower_bound=np.array([-5.0, -5.0, -1.0], dtype=np.float32),
                upper_bound=np.array([5.0, 5.0, 1.0], dtype=np.float32),
            ),
        )


@dataclass
class ASAAstribotAlignmentConfig:
    base_frame_mode: Literal["canonical_pitch_047", "cam_init", "level_head"] = "level_head"
    yaw_torso_deadband_rad: float = 1.05
    torso_yaw_share: float = 0.50
    max_ee_step_pos_m: float = 0.05
    max_head_step_rad: float = 0.12
    max_torso_step_rad: float = 0.06
    ik_damping: float = 0.04
    ik_iters: int = 8
    ik_posture_gain: float = 0.08
    gripper_binary_threshold: float | None = None


class ASAAstribotEmbodimentAdapter:
    """Stateful per-episode/per-subtask kinematic bridge between ASA and ActiveArena Astribot S1."""

    def __init__(
        self,
        norm_bundle: ASANormBundle | None = None,
        cfg: ASAAstribotAlignmentConfig | None = None,
    ) -> None:
        self.norm = norm_bundle or ASANormBundle.astribot_human_aligned()
        self.cfg = cfg or ASAAstribotAlignmentConfig()
        self.T_world_robot0: np.ndarray | None = None
        self.T_world_base: np.ndarray | None = None

    def reset(self) -> None:
        self.T_world_robot0 = None
        self.T_world_base = None

    @staticmethod
    def build_robot0_frame_from_root_pose(root_pose7_xyz_wxyz: np.ndarray) -> np.ndarray:
        """Astribot URDF `torso_base` has FLU axes `x=+Y_root_urdf`, i.e. at `robot_pose=[0,-0.45,-0.15,0.707,0,0,0.707]`
        in ActiveArena Sapien world: `+X_r0 = [0, 1, 0]`, `+Y_r0 = [-1, 0, 0]`, `+Z_r0 = [0, 0, 1]`."""
        T_root = pose7_to_matrix(root_pose7_xyz_wxyz)
        return T_root

    def anchor_base_frame(
        self,
        T_world_cam: np.ndarray,
        T_world_robot0: np.ndarray,
        q_torso: float,
        q_head1: float,
    ) -> np.ndarray:
        """Anchor `T_world_base` at episode/subtask start."""
        self.T_world_robot0 = np.asarray(T_world_robot0, dtype=np.float64).copy()
        T_cam = np.asarray(T_world_cam, dtype=np.float64)

        if self.cfg.base_frame_mode == "cam_init":
            T_base = T_cam.copy()
        else:
            pitch_ref = CANONICAL_BASE_PITCH_RAD if self.cfg.base_frame_mode == "canonical_pitch_047" else 0.0
            yaw0 = float(q_torso + q_head1)
            R_local = Rotation.from_euler("ZYX", [yaw0, pitch_ref, 0.0]).as_matrix()
            T_base = np.eye(4, dtype=np.float64)
            T_base[:3, :3] = self.T_world_robot0[:3, :3] @ R_local
            T_base[:3, 3] = T_cam[:3, 3]

        self.T_world_base = T_base
        return T_base

    @staticmethod
    def tcp_world_to_asa_world(pose7_tcp_world: np.ndarray) -> np.ndarray:
        T_tcp = pose7_to_matrix(pose7_tcp_world)
        T_asa = T_tcp.copy()
        T_asa[:3, :3] = T_tcp[:3, :3] @ R_EE_TO_ASA
        return T_asa

    @staticmethod
    def asa_world_to_ee_pose7(T_world_asa: np.ndarray) -> np.ndarray:
        """Convert target ASA FLU wrist pose in Sapien world -> ActiveArena `get_left_ee_pose` 7D `(xyz + wxyz)`."""
        R_ee = T_world_asa[:3, :3] @ R_EE_TO_ASA
        p_tcp = T_world_asa[:3, 3]
        p_ee = p_tcp + R_ee @ TCP_TO_EE_OFFSET_IN_EE
        T_ee = np.eye(4, dtype=np.float64)
        T_ee[:3, :3] = R_ee
        T_ee[:3, 3] = p_ee
        return matrix_to_pose7(T_ee)

    def encode_proprio29(
        self,
        *,
        T_world_cam: np.ndarray,
        left_tcp_pose7: np.ndarray,
        right_tcp_pose7: np.ndarray,
        left_gripper: float,
        right_gripper: float,
        T_world_robot0: np.ndarray,
        q_torso: float,
        q_head1: float,
    ) -> np.ndarray:
        """Pack live Astribot kinematics into ASA's normalized 29-D proprio vector."""
        if self.T_world_base is None:
            self.anchor_base_frame(T_world_cam, T_world_robot0, q_torso=q_torso, q_head1=q_head1)
        assert self.T_world_base is not None
        inv_base = np.linalg.inv(self.T_world_base)

        T_base_head = inv_base @ np.asarray(T_world_cam, dtype=np.float64)
        T_base_left = inv_base @ self.tcp_world_to_asa_world(left_tcp_pose7)
        T_base_right = inv_base @ self.tcp_world_to_asa_world(right_tcp_pose7)

        head9 = self.norm.head.norm_pos(matrix_to_pose9_asa(T_base_head))
        left9 = self.norm.left_wrist.norm_pos(matrix_to_pose9_asa(T_base_left))
        right9 = self.norm.right_wrist.norm_pos(matrix_to_pose9_asa(T_base_right))

        lg = float(np.clip(left_gripper, 0.0, 1.0))
        rg = float(np.clip(right_gripper, 0.0, 1.0))
        return np.concatenate([head9, left9, right9, [lg, rg]], axis=0).astype(np.float32)

    def decode_view_target_to_head_torso(
        self,
        view_action9_norm: np.ndarray,
        *,
        q_head_now: np.ndarray,
        q_torso_now: float,
    ) -> tuple[np.ndarray, float]:
        """Map ASA normalized `view` action `(9,)` -> target `(q_head1, q_head2)` and `q_torso`."""
        assert self.T_world_base is not None and self.T_world_robot0 is not None
        view9 = self.norm.head.denorm_pos(view_action9_norm)
        T_base_view = pose9_asa_to_matrix(view9)
        R_world_target = self.T_world_base[:3, :3] @ T_base_view[:3, :3]
        R_r0_target = self.T_world_robot0[:3, :3].T @ R_world_target

        # Exact inverse of `R_z(q_torso + q_head1) @ R_y(q_head2)`
        yaw_total, pitch_target, _ = Rotation.from_matrix(R_r0_target).as_euler("ZYX", degrees=False)

        if abs(yaw_total - q_torso_now) <= self.cfg.yaw_torso_deadband_rad:
            q_torso_tgt = float(q_torso_now)
        else:
            excess = yaw_total - np.sign(yaw_total) * self.cfg.yaw_torso_deadband_rad
            q_torso_tgt = float(np.clip(self.cfg.torso_yaw_share * excess, *TORSO_JOINT4_LIMITS))

        q_head1_tgt = float(np.clip(yaw_total - q_torso_tgt, *HEAD_JOINT1_LIMITS))
        # Absorb any remaining yaw if head_joint_1 saturated
        rem_yaw = yaw_total - (q_torso_tgt + q_head1_tgt)
        if abs(rem_yaw) > 1e-4:
            q_torso_tgt = float(np.clip(q_torso_tgt + rem_yaw, *TORSO_JOINT4_LIMITS))

        q_head2_tgt = float(np.clip(pitch_target, *HEAD_JOINT2_LIMITS))

        # Smooth step rate limiting
        dh1 = float(np.clip(q_head1_tgt - q_head_now[0], -self.cfg.max_head_step_rad, self.cfg.max_head_step_rad))
        dh2 = float(np.clip(q_head2_tgt - q_head_now[1], -self.cfg.max_head_step_rad, self.cfg.max_head_step_rad))
        dtorso = float(np.clip(q_torso_tgt - q_torso_now, -self.cfg.max_torso_step_rad, self.cfg.max_torso_step_rad))
        return np.array([q_head_now[0] + dh1, q_head_now[1] + dh2], dtype=np.float64), float(q_torso_now + dtorso)

    def decode_mani_target_to_ee_poses(
        self,
        mani_action_2x10_norm: np.ndarray,
        *,
        left_ee_now7: np.ndarray | None = None,
        right_ee_now7: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, np.ndarray, float]:
        """Map ASA normalized `mani` action `(2, 10)` -> `(left_ee_pose7, left_grip, right_ee_pose7, right_grip)`."""
        assert self.T_world_base is not None
        arr = np.asarray(mani_action_2x10_norm, dtype=np.float32).reshape(2, 10)
        left9 = self.norm.left_wrist.denorm_pos(arr[0, :9])
        right9 = self.norm.right_wrist.denorm_pos(arr[1, :9])

        T_world_left_asa = self.T_world_base @ pose9_asa_to_matrix(left9)
        T_world_right_asa = self.T_world_base @ pose9_asa_to_matrix(right9)

        left_ee7 = self.asa_world_to_ee_pose7(T_world_left_asa)
        right_ee7 = self.asa_world_to_ee_pose7(T_world_right_asa)

        if left_ee_now7 is not None:
            dp = left_ee7[:3] - left_ee_now7[:3]
            norm_dp = float(np.linalg.norm(dp))
            if norm_dp > self.cfg.max_ee_step_pos_m:
                left_ee7[:3] = left_ee_now7[:3] + dp * (self.cfg.max_ee_step_pos_m / norm_dp)

        if right_ee_now7 is not None:
            dp = right_ee7[:3] - right_ee_now7[:3]
            norm_dp = float(np.linalg.norm(dp))
            if norm_dp > self.cfg.max_ee_step_pos_m:
                right_ee7[:3] = right_ee_now7[:3] + dp * (self.cfg.max_ee_step_pos_m / norm_dp)

        lg = float(np.clip(arr[0, 9], 0.0, 1.0))
        rg = float(np.clip(arr[1, 9], 0.0, 1.0))
        if self.cfg.gripper_binary_threshold is not None:
            lg = 1.0 if lg >= self.cfg.gripper_binary_threshold else 0.0
            rg = 1.0 if rg >= self.cfg.gripper_binary_threshold else 0.0

        return left_ee7, lg, right_ee7, rg


def solve_sapien_arm_dls_ik(
    entity: Any,
    arm_joints: list[Any],
    ee_link: Any,
    target_endlink_pose7: np.ndarray,
    *,
    q_homestate: np.ndarray | None = None,
    damping: float = 0.04,
    iters: int = 8,
    posture_gain: float = 0.05,
) -> np.ndarray:
    """Fast (<0.4 ms) damped-least-squares 7-DoF IK directly on Sapien articulation (restores original qpos)."""
    active_joints = list(entity.get_active_joints())
    arm_indices = [active_joints.index(j) for j in arm_joints]
    limits = np.array([j.get_limits()[0] for j in arm_joints], dtype=np.float64)  # (7, 2)

    q_orig = np.array(entity.get_qpos(), dtype=np.float64, copy=True)
    q_work = q_orig.copy()
    q_arm = q_work[arm_indices].copy()
    q_home = np.asarray(q_homestate, dtype=np.float64) if q_homestate is not None else q_arm.copy()

    T_target = pose7_to_matrix(target_endlink_pose7)
    p_tgt = T_target[:3, 3]
    R_tgt = T_target[:3, :3]

    eps = 1e-5
    eye6 = np.eye(6, dtype=np.float64) * (damping ** 2)
    eye7 = np.eye(7, dtype=np.float64)

    try:
        for _ in range(iters):
            q_work[arm_indices] = q_arm
            entity.set_qpos(q_work)
            pose_now = ee_link.get_entity_pose()
            p_now = np.asarray(pose_now.p, dtype=np.float64)
            R_now = quat_wxyz_to_matrix(pose_now.q)

            err_pos = p_tgt - p_now
            err_rot = Rotation.from_matrix(R_tgt @ R_now.T).as_rotvec()
            err6 = np.concatenate([err_pos, err_rot], axis=0)
            if float(np.linalg.norm(err_pos)) < 5e-4 and float(np.linalg.norm(err_rot)) < 3e-3:
                break

            # Finite-difference geometric Jacobian wrt the 7 arm joints (7 FK evaluations, ~30 us in Sapien)
            J = np.zeros((6, 7), dtype=np.float64)
            for k, a_idx in enumerate(arm_indices):
                q_pert = q_work.copy()
                q_pert[a_idx] += eps
                entity.set_qpos(q_pert)
                pk = ee_link.get_entity_pose()
                dp = (np.asarray(pk.p, dtype=np.float64) - p_now) / eps
                dR = quat_wxyz_to_matrix(pk.q) @ R_now.T
                dw = Rotation.from_matrix(dR).as_rotvec() / eps
                J[:, k] = np.concatenate([dp, dw], axis=0)

            JJT_inv = np.linalg.inv(J @ J.T + eye6)
            J_pinv = J.T @ JJT_inv
            null_proj = eye7 - J_pinv @ J
            dq = J_pinv @ err6 + posture_gain * (null_proj @ (q_home - q_arm))
            dq = np.clip(dq, -0.25, 0.25)
            q_arm = np.clip(q_arm + dq, limits[:, 0], limits[:, 1])
    finally:
        entity.set_qpos(q_orig)

    return q_arm
