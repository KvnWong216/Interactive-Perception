"""ActiveArena closed-loop policy adapter for ZS-IP (`arXiv:2602.18374`).

Loaded directly by unmodified `ActiveArena/script/eval_policy.py` via:
    --policy_name eval.activearena.zsip_policy

Implements Sripada, Guerin, & Ghalamzan (2026), *Zero-Shot Interactive Perception:
Leveraging Frozen VLMs for Active Exploration and Manipulation*, adapted to ActiveArena's
`Astribot S1` whole-body embodiment with **0 training** (100% frozen `Qwen3-VL-2B-Instruct`)
and **0 privileged simulator oracle**:
  1. **Enhanced Observation (EO)**: Non-oracle HSV + homography keypoints (`K1..Km`),
     pushlines (`P1->D1`), and 2D fan-grid anchors (`G1..G6`).
  2. **Memory-Guided Context Lookup**: Tracks primitive execution history, holding state,
     and clean proprioceptively cropped reveal images pinned during `INSPECT_HELD`.
  3. **Multi-Strategy Robotic Primitive Controller**: Executes `SCAN_HEAD`, `PUSH`,
     `GRASP`, `INSPECT_HELD`, `PLACE`, and `DONE` via live damped-least-squares Sapien IK.
"""
from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import requests
from scipy.spatial.transform import Rotation

from eval.activearena.asa_astribot_embodiment import (
    HEAD_JOINT1_LIMITS,
    HEAD_JOINT2_LIMITS,
    TCP_TO_EE_OFFSET_IN_EE,
    TORSO_JOINT4_LIMITS,
    matrix_to_quat_wxyz,
    pose7_to_matrix,
    solve_sapien_arm_dls_ik,
)
from eval.activearena.zsip_eo_and_memory import (
    CameraPinhole,
    EnhancedObservationResult,
    ZSIPMemory,
    build_enhanced_observation,
    extract_proprio_reveal_crop,
)


# Canonical wrist-inspection joint postures on Astribot S1 (`envs/_GLOBAL_CONFIGS.py`)
LEFT_CHECK_POSE = np.array([-1.5, -0.7, 0.4, 1.8, 2.6, -0.76, -0.7], dtype=np.float64)
RIGHT_CHECK_POSE = np.array([1.5, -0.7, -0.4, 1.8, -2.6, -0.76, 0.7], dtype=np.float64)
CLOSED_GRIP_VAL = 0.18  # Firm grasp avoiding 0-gap PhysX rigid-cylinder interpenetration impulse

# Straight top-down EE orientation at azimuth=0 (`X_ee=[0,0,-1]`, `Y_ee=[-1,0,0]`, `Z_ee=[0,1,0]`)
R_TOP_DOWN_ZERO = Rotation.from_quat([-0.5, 0.5, -0.5, -0.5], scalar_first=True).as_matrix()


def _encode_jpeg_b64(rgb: np.ndarray, quality: int = 92) -> str:
    arr = np.asarray(rgb, dtype=np.uint8)
    buf = io.BytesIO()
    with Image.fromarray(arr).convert("RGB") as im:
        im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_head_rgb(observation: dict[str, Any]) -> np.ndarray:
    obs = observation.get("observation", observation)
    for cam_key in ("camera_head", "head_camera"):
        if cam_key in obs and "rgb" in obs[cam_key]:
            return np.asarray(obs[cam_key]["rgb"], dtype=np.uint8)
    raise KeyError(f"Head camera RGB not found in observation keys: {list(obs.keys())}")


def _radial_top_down_ee_quat(world_xy: np.ndarray, T_world_robot0: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute radially aligned top-down EE quaternion `(w, x, y, z)` and azimuth `theta` (rad)."""
    rel = np.asarray(world_xy[:2], dtype=np.float64) - T_world_robot0[:2, 3]
    fwd = float(np.dot(rel, T_world_robot0[:2, 0]))
    left = float(np.dot(rel, T_world_robot0[:2, 1]))
    theta = math.atan2(left, max(fwd, 0.04))
    R_z = Rotation.from_euler("z", theta).as_matrix()
    R_ee = R_z @ R_TOP_DOWN_ZERO
    return matrix_to_quat_wxyz(R_ee), theta


def _tcp_to_ee_pose7(tcp_xyz: np.ndarray, ee_quat_wxyz: np.ndarray) -> np.ndarray:
    R_ee = Rotation.from_quat(ee_quat_wxyz, scalar_first=True).as_matrix()
    p_ee = np.asarray(tcp_xyz, dtype=np.float64) + R_ee @ TCP_TO_EE_OFFSET_IN_EE
    return np.concatenate([p_ee, ee_quat_wxyz], axis=0)


@dataclass
class _PrimitiveStage:
    name: str
    steps_remaining: int
    arm: str = "left"
    tcp_tgt_xyz: np.ndarray | None = None
    ee_quat_wxyz: np.ndarray | None = None
    joint_tgt7: np.ndarray | None = None
    grip_tgt: float = 1.0
    q_head_tgt: np.ndarray | None = None
    q_torso_tgt: float | None = None
    save_carry_qpos_at_end: bool = False
    pin_reveal_at_end: bool = False
    set_holding_at_end: bool | None = None
    set_inspected_at_end: bool = False


class ZSIPAstribotPolicyClient:
    """Zero-Shot Interactive Perception controller for ActiveArena's `Astribot S1`."""

    def __init__(self, usr_args: dict[str, Any]) -> None:
        self.usr_args = dict(usr_args)
        self.server_url = os.environ.get("ZSIP_SERVER_URL", "http://127.0.0.1:19300/decide")
        self.timeout_s = float(os.environ.get("ZSIP_SERVER_TIMEOUT", "45.0"))
        self.max_turns = int(os.environ.get("ZSIP_MAX_TURNS", "12"))

        trace_path = os.environ.get("ZSIP_TRACE_LOG_PATH", "")
        self.trace_path = Path(trace_path) if trace_path else None
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.eo_debug_dir = self.trace_path.parent / "zsip_eo_frames"
            self.eo_debug_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.eo_debug_dir = None

        self._http = requests.Session()
        self._http.trust_env = False

        self.memory = ZSIPMemory(max_reveal_frames=2)
        self.episode_idx = 0
        self.turn_idx = 0
        self._stage_queue: deque[_PrimitiveStage] = deque()
        self._carry_qpos7: np.ndarray | None = None
        self._placed_at_least_once: bool = False
        self._episode_halted: bool = False

    def reset(self) -> None:
        self.memory.reset()
        self._stage_queue.clear()
        self._carry_qpos7 = None
        self._placed_at_least_once = False
        self._episode_halted = False
        self.turn_idx = 0
        self.episode_idx += 1

    def _read_robot_state(self, task_env: Any) -> dict[str, Any]:
        robot = task_env.robot
        cam_pose = robot.head_camera.get_entity_pose()
        T_world_cam = pose7_to_matrix(np.concatenate([cam_pose.p, cam_pose.q]))

        rp = robot.left_entity_origion_pose
        T_world_robot0 = pose7_to_matrix(np.concatenate([rp.p, rp.q]))

        left_tcp7 = np.asarray(robot.get_left_tcp_pose(), dtype=np.float64)
        right_tcp7 = np.asarray(robot.get_right_tcp_pose(), dtype=np.float64)
        left_ee7 = np.asarray(robot.get_left_ee_pose(), dtype=np.float64)
        right_ee7 = np.asarray(robot.get_right_ee_pose(), dtype=np.float64)

        q_left7 = np.asarray(robot.get_left_arm_real_jointState()[:-1], dtype=np.float64)
        q_right7 = np.asarray(robot.get_right_arm_real_jointState()[:-1], dtype=np.float64)
        left_grip, right_grip = robot.get_normal_real_gripper_val()

        q_head = np.asarray(robot.get_head_real_jointState(), dtype=np.float64)
        q_torso_list = robot.get_torso_real_jointState()
        q_torso = float(q_torso_list[0]) if len(q_torso_list) > 0 else 0.0

        return {
            "T_world_cam": T_world_cam,
            "T_world_robot0": T_world_robot0,
            "left_tcp7": left_tcp7,
            "right_tcp7": right_tcp7,
            "left_ee7": left_ee7,
            "right_ee7": right_ee7,
            "q_left7": q_left7,
            "q_right7": q_right7,
            "left_grip": float(left_grip),
            "right_grip": float(right_grip),
            "q_head": q_head,
            "q_torso": q_torso,
        }

    def _proprio_exclude_uvs(self, rgb_shape: tuple[int, ...], kin: dict[str, Any]) -> list[tuple[float, float]]:
        """Project held/raised gripper TCP & EE positions so EO never labels the robot's own hand as a table anchor."""
        if not self.memory.holding:
            return []
        cam = CameraPinhole(width=rgb_shape[1], height=rgb_shape[0])
        pts = (
            [kin["left_tcp7"][:3], kin["left_ee7"][:3]]
            if self.memory.holding_arm == "left"
            else [kin["right_tcp7"][:3], kin["right_ee7"][:3]]
        )
        uvs: list[tuple[float, float]] = []
        for p in pts:
            uv = cam.project_world_to_pixel(p, kin["T_world_cam"])
            if uv is not None:
                uvs.append(uv)
        return uvs

    def _enqueue_primitive(
        self,
        decision: dict[str, Any],
        eo: EnhancedObservationResult,
        kin: dict[str, Any],
        task_env: Any,
    ) -> str:
        prim = str(decision.get("primitive", "SCAN_HEAD")).upper()
        target = str(decision.get("target", "center")).strip()
        robot = task_env.robot
        left_home = np.asarray(robot.left_homestate, dtype=np.float64)
        right_home = np.asarray(robot.right_homestate, dtype=np.float64)

        if prim == "SCAN_HEAD":
            direction = target.lower()
            if direction not in {"left", "center", "right"}:
                direction = "left" if "left" not in self.memory.scanned_directions else "right"
            if direction not in self.memory.scanned_directions:
                self.memory.scanned_directions.append(direction)
            if direction == "left":
                h1, t_yaw = +0.62, +0.35
            elif direction == "right":
                h1, t_yaw = -0.62, -0.35
            else:
                h1, t_yaw = 0.0, 0.0
            arm = self.memory.holding_arm if self.memory.holding else "left"
            grip = CLOSED_GRIP_VAL if self.memory.holding else 1.0
            hold_q = (
                self._carry_qpos7
                if (self.memory.holding and self._carry_qpos7 is not None)
                else (left_home if arm == "left" else right_home)
            )
            self._stage_queue.append(
                _PrimitiveStage(
                    name=f"SCAN_HEAD:{direction}",
                    steps_remaining=12,
                    arm=arm,
                    joint_tgt7=hold_q,
                    grip_tgt=grip,
                    q_head_tgt=np.array([h1, 0.96], dtype=np.float64),
                    q_torso_tgt=t_yaw,
                )
            )
            return f"Scanned camera toward {direction}"

        if prim == "PUSH":
            pid = target.upper()
            if pid not in eo.pushlines and eo.pushlines:
                pid = next(iter(eo.pushlines.keys()))
            if pid not in eo.pushlines:
                self._stage_queue.append(
                    _PrimitiveStage(name="PUSH:noop", steps_remaining=3, grip_tgt=0.0 if self.memory.holding else 1.0)
                )
                return f"No valid pushline {target} available"
            pl = eo.pushlines[pid]
            ee_quat, theta = _radial_top_down_ee_quat(pl.pre_world_xyz, kin["T_world_robot0"])
            arm = "left" if theta >= -0.25 else "right"
            t_yaw = float(np.clip(theta * 0.7, -0.80, 0.80))
            h1 = float(np.clip(theta - t_yaw, -0.55, 0.55))
            pre_hi = np.array([pl.pre_world_xyz[0], pl.pre_world_xyz[1], 0.88], dtype=np.float64)
            pre_lo = np.array([pl.pre_world_xyz[0], pl.pre_world_xyz[1], 0.774], dtype=np.float64)
            post_lo = np.array([pl.post_world_xyz[0], pl.post_world_xyz[1], 0.774], dtype=np.float64)
            post_hi = np.array([pl.post_world_xyz[0], pl.post_world_xyz[1], 0.89], dtype=np.float64)
            home_q = left_home if arm == "left" else right_home
            self._stage_queue.extend(
                [
                    _PrimitiveStage(f"PUSH:{pid}:align", 10, arm=arm, joint_tgt7=home_q, grip_tgt=0.0, q_head_tgt=np.array([h1, 0.96]), q_torso_tgt=t_yaw),
                    _PrimitiveStage(f"PUSH:{pid}:pre_hi", 10, arm=arm, tcp_tgt_xyz=pre_hi, ee_quat_wxyz=ee_quat, grip_tgt=0.0, q_head_tgt=np.array([h1, 0.96]), q_torso_tgt=t_yaw),
                    _PrimitiveStage(f"PUSH:{pid}:pre_lo", 8, arm=arm, tcp_tgt_xyz=pre_lo, ee_quat_wxyz=ee_quat, grip_tgt=0.0, q_head_tgt=np.array([h1, 0.96]), q_torso_tgt=t_yaw),
                    _PrimitiveStage(f"PUSH:{pid}:sweep", 12, arm=arm, tcp_tgt_xyz=post_lo, ee_quat_wxyz=ee_quat, grip_tgt=0.0, q_head_tgt=np.array([h1, 0.96]), q_torso_tgt=t_yaw),
                    _PrimitiveStage(f"PUSH:{pid}:retreat", 8, arm=arm, tcp_tgt_xyz=post_hi, ee_quat_wxyz=ee_quat, grip_tgt=1.0, q_head_tgt=np.array([h1, 0.96]), q_torso_tgt=t_yaw),
                    _PrimitiveStage(f"PUSH:{pid}:home", 10, arm=arm, joint_tgt7=home_q, grip_tgt=1.0, q_head_tgt=np.array([h1, 0.96]), q_torso_tgt=t_yaw),
                ]
            )
            return f"Executed push along {pid}->D{pid[1:]}"

        if prim == "GRASP":
            if self.memory.holding:
                self._stage_queue.append(_PrimitiveStage("GRASP:already_holding", 2, arm=self.memory.holding_arm, grip_tgt=0.0))
                return "Precondition warning: already holding an object; call INSPECT_HELD or PLACE"
            kid = target.upper()
            if kid not in eo.keypoints:
                self._stage_queue.append(_PrimitiveStage("GRASP:missing_keypoint", 2, grip_tgt=1.0))
                return f"Anchor {kid} not in current keypoints {list(eo.keypoints.keys())}; call SCAN_HEAD or choose visible Ki"
            kp = eo.keypoints[kid]
            ee_quat, theta = _radial_top_down_ee_quat(kp.grasp_xyz, kin["T_world_robot0"])
            # Prefer left arm (`LEFT_CHECK_POSE`) across reachable torso-rotated workspace
            arm = "left" if theta >= -1.05 else "right"
            t_yaw = float(np.clip(theta, -0.92, 0.92))
            h1 = float(np.clip(theta - t_yaw, -0.55, 0.55))
            home_q = left_home if arm == "left" else right_home
            pre_xyz = np.array([kp.grasp_xyz[0], kp.grasp_xyz[1], 0.915], dtype=np.float64)
            grasp_xyz = kp.grasp_xyz.copy()
            lift_xyz = np.array([kp.grasp_xyz[0], kp.grasp_xyz[1], 0.915], dtype=np.float64)

            self.memory.holding_arm = arm
            self.memory.origin_xyz = [round(float(v), 3) for v in grasp_xyz]
            self.memory.inspected_current_object = False

            self._stage_queue.extend(
                [
                    _PrimitiveStage(f"GRASP:{kid}:align", 16, arm=arm, joint_tgt7=home_q, grip_tgt=1.0, q_head_tgt=np.array([h1, 0.98]), q_torso_tgt=t_yaw),
                    _PrimitiveStage(f"GRASP:{kid}:pre", 32, arm=arm, tcp_tgt_xyz=pre_xyz, ee_quat_wxyz=ee_quat, grip_tgt=1.0, q_head_tgt=np.array([h1, 0.98]), q_torso_tgt=t_yaw),
                    _PrimitiveStage(f"GRASP:{kid}:descend", 22, arm=arm, tcp_tgt_xyz=grasp_xyz, ee_quat_wxyz=ee_quat, grip_tgt=1.0, q_head_tgt=np.array([h1, 0.98]), q_torso_tgt=t_yaw),
                    _PrimitiveStage(f"GRASP:{kid}:close", 12, arm=arm, tcp_tgt_xyz=grasp_xyz, ee_quat_wxyz=ee_quat, grip_tgt=CLOSED_GRIP_VAL, q_head_tgt=np.array([h1, 0.98]), q_torso_tgt=t_yaw),
                    _PrimitiveStage(
                        f"GRASP:{kid}:lift",
                        20,
                        arm=arm,
                        tcp_tgt_xyz=lift_xyz,
                        ee_quat_wxyz=ee_quat,
                        grip_tgt=CLOSED_GRIP_VAL,
                        q_head_tgt=np.array([0.0, 0.96]),
                        q_torso_tgt=t_yaw,
                        save_carry_qpos_at_end=True,
                        set_holding_at_end=True,
                    ),
                ]
            )
            return f"Grasped {kid} (color={kp.color_tag}) at xy=({grasp_xyz[0]:+.2f}, {grasp_xyz[1]:+.2f}) with {arm} arm"

        if prim == "INSPECT_HELD":
            if not self.memory.holding:
                self._stage_queue.append(_PrimitiveStage("INSPECT_HELD:empty", 2, grip_tgt=1.0))
                return "Precondition failed: grippers are empty; call GRASP(Ki) before INSPECT_HELD"
            arm = self.memory.holding_arm
            check_q = LEFT_CHECK_POSE if arm == "left" else RIGHT_CHECK_POSE
            ret_q = self._carry_qpos7 if self._carry_qpos7 is not None else (left_home if arm == "left" else right_home)
            self._stage_queue.extend(
                [
                    _PrimitiveStage("INSPECT:move_to_cam", 42, arm=arm, joint_tgt7=check_q, grip_tgt=CLOSED_GRIP_VAL, q_head_tgt=np.array([0.0, 0.98])),
                    _PrimitiveStage(
                        "INSPECT:hold_and_pin",
                        8,
                        arm=arm,
                        joint_tgt7=check_q,
                        grip_tgt=CLOSED_GRIP_VAL,
                        q_head_tgt=np.array([0.0, 0.98]),
                        pin_reveal_at_end=True,
                        set_inspected_at_end=True,
                    ),
                    _PrimitiveStage("INSPECT:restore_carry", 38, arm=arm, joint_tgt7=ret_q, grip_tgt=CLOSED_GRIP_VAL, q_head_tgt=np.array([0.0, 0.96]), q_torso_tgt=0.0),
                ]
            )
            return "Rotated held object in front of head camera and pinned clean backside reveal image in memory"

        if prim == "PLACE":
            if not self.memory.holding:
                self._stage_queue.append(_PrimitiveStage("PLACE:empty", 2, grip_tgt=1.0))
                return "Precondition failed: grippers are empty; cannot PLACE without holding an object"
            key = target.upper()
            if key in eo.keypoints:
                place_xyz = eo.keypoints[key].place_xyz.copy()
                desc = f"{key} (color={eo.keypoints[key].color_tag})"
            elif key in eo.grid_anchors:
                place_xyz = eo.grid_anchors[key].copy()
                desc = f"grid sector {key}"
            else:
                self._stage_queue.append(_PrimitiveStage("PLACE:missing_target", 2, arm=self.memory.holding_arm, grip_tgt=0.0))
                return f"Target {key} not visible in current keypoints {list(eo.keypoints.keys())}; call SCAN_HEAD to find matching pad"

            arm = self.memory.holding_arm
            ee_quat, theta = _radial_top_down_ee_quat(place_xyz, kin["T_world_robot0"])
            t_yaw = float(np.clip(theta, -0.90, 0.90))
            h1 = float(np.clip(theta - t_yaw, -0.55, 0.55))
            home_q = left_home if arm == "left" else right_home
            hold_q = self._carry_qpos7 if self._carry_qpos7 is not None else home_q
            pre_xyz = np.array([place_xyz[0], place_xyz[1], 0.915], dtype=np.float64)
            retreat_xyz = np.array([place_xyz[0], place_xyz[1], 0.925], dtype=np.float64)

            self._placed_at_least_once = True
            self._stage_queue.extend(
                [
                    _PrimitiveStage(f"PLACE:{key}:align", 18, arm=arm, joint_tgt7=hold_q, grip_tgt=CLOSED_GRIP_VAL, q_head_tgt=np.array([h1, 0.98]), q_torso_tgt=t_yaw),
                    _PrimitiveStage(f"PLACE:{key}:pre", 28, arm=arm, tcp_tgt_xyz=pre_xyz, ee_quat_wxyz=ee_quat, grip_tgt=CLOSED_GRIP_VAL, q_head_tgt=np.array([h1, 0.98]), q_torso_tgt=t_yaw),
                    _PrimitiveStage(f"PLACE:{key}:lower", 22, arm=arm, tcp_tgt_xyz=place_xyz, ee_quat_wxyz=ee_quat, grip_tgt=CLOSED_GRIP_VAL, q_head_tgt=np.array([h1, 0.98]), q_torso_tgt=t_yaw),
                    _PrimitiveStage(f"PLACE:{key}:open", 12, arm=arm, tcp_tgt_xyz=place_xyz, ee_quat_wxyz=ee_quat, grip_tgt=1.0, q_head_tgt=np.array([h1, 0.98]), q_torso_tgt=t_yaw, set_holding_at_end=False),
                    _PrimitiveStage(f"PLACE:{key}:retreat", 20, arm=arm, tcp_tgt_xyz=retreat_xyz, ee_quat_wxyz=ee_quat, grip_tgt=1.0, q_head_tgt=np.array([h1, 0.98]), q_torso_tgt=t_yaw),
                    _PrimitiveStage(f"PLACE:{key}:home", 26, arm=arm, joint_tgt7=home_q, grip_tgt=1.0, q_head_tgt=np.array([0.0, 0.96]), q_torso_tgt=0.0),
                ]
            )
            return f"Placed held object onto {desc} at xy=({place_xyz[0]:+.2f}, {place_xyz[1]:+.2f})"

        # DONE
        if self.memory.holding or not self._placed_at_least_once:
            self._stage_queue.append(
                _PrimitiveStage("DONE:premature", 2, arm=self.memory.holding_arm, grip_tgt=0.0 if self.memory.holding else 1.0)
            )
            return "Precondition warning: task not finished yet (object has not been placed on destination pad)"
        self._episode_halted = True
        self._stage_queue.append(
            _PrimitiveStage("DONE:hold", 8, arm="left", joint_tgt7=left_home, grip_tgt=1.0, q_head_tgt=np.array([0.0, 0.96]), q_torso_tgt=0.0)
        )
        return "Completed task"

    def _execute_stage_step(self, stg: _PrimitiveStage, task_env: Any, kin: dict[str, Any], rgb: np.ndarray) -> None:
        robot = task_env.robot
        q_left_cmd = kin["q_left7"].copy()
        q_right_cmd = kin["q_right7"].copy()

        # Non-active arm stays cleanly at homestate with open gripper
        if stg.arm == "left":
            q_right_cmd = kin["q_right7"] + np.clip(
                np.asarray(robot.right_homestate, dtype=np.float64) - kin["q_right7"], -0.10, 0.10
            )
            left_grip_cmd = float(stg.grip_tgt)
            right_grip_cmd = 1.0
        else:
            q_left_cmd = kin["q_left7"] + np.clip(
                np.asarray(robot.left_homestate, dtype=np.float64) - kin["q_left7"], -0.10, 0.10
            )
            left_grip_cmd = 1.0
            right_grip_cmd = float(stg.grip_tgt)

        converged_early = False
        # Lazily solve exact Pinocchio DLS IK once at the start of each Cartesian stage (after torso alignment!)
        if stg.joint_tgt7 is None and stg.tcp_tgt_xyz is not None and stg.ee_quat_wxyz is not None:
            ee_pose7 = _tcp_to_ee_pose7(stg.tcp_tgt_xyz, stg.ee_quat_wxyz)
            if stg.arm == "left":
                stg.joint_tgt7 = solve_sapien_arm_dls_ik(
                    robot.left_entity,
                    robot.left_arm_joints,
                    robot.left_ee_link,
                    ee_pose7,
                    current_ee_pose7=kin["left_ee7"],
                    q_homestate=np.asarray(robot.left_homestate, dtype=np.float64),
                    iters=36,
                )
            else:
                stg.joint_tgt7 = solve_sapien_arm_dls_ik(
                    robot.right_entity,
                    robot.right_arm_joints,
                    robot.right_ee_link,
                    ee_pose7,
                    current_ee_pose7=kin["right_ee7"],
                    q_homestate=np.asarray(robot.right_homestate, dtype=np.float64),
                    iters=36,
                )

        if stg.joint_tgt7 is not None:
            cur_q = kin["q_left7"] if stg.arm == "left" else kin["q_right7"]
            err_q = stg.joint_tgt7 - cur_q
            step_q = cur_q + np.clip(err_q, -0.12, 0.12)
            hold_stage = stg.pin_reveal_at_end or (":close" in stg.name) or (":open" in stg.name)
            torso_settled = stg.q_torso_tgt is None or (
                abs(kin["q_torso"] - float(np.clip(stg.q_torso_tgt, *TORSO_JOINT4_LIMITS))) < 0.035
            )
            head_settled = stg.q_head_tgt is None or (
                float(
                    np.max(
                        np.abs(
                            kin["q_head"]
                            - np.array(
                                [
                                    np.clip(stg.q_head_tgt[0], *HEAD_JOINT1_LIMITS),
                                    np.clip(stg.q_head_tgt[1], *HEAD_JOINT2_LIMITS),
                                ]
                            )
                        )
                    )
                )
                < 0.05
            )
            if float(np.max(np.abs(err_q))) < 0.045 and torso_settled and head_settled and not hold_stage:
                converged_early = True
            if stg.arm == "left":
                q_left_cmd = step_q
            else:
                q_right_cmd = step_q

        # Head & Torso smooth deltas relative to current drive targets
        head_drive_now = np.asarray(robot.get_head_jointState(), dtype=np.float64)
        if stg.q_head_tgt is not None:
            h_clamped = np.array(
                [
                    np.clip(stg.q_head_tgt[0], *HEAD_JOINT1_LIMITS),
                    np.clip(stg.q_head_tgt[1], *HEAD_JOINT2_LIMITS),
                ],
                dtype=np.float64,
            )
            head_next = kin["q_head"] + np.clip(h_clamped - kin["q_head"], -0.10, 0.10)
            head_delta = head_next - head_drive_now
        else:
            head_delta = np.zeros(2, dtype=np.float64)

        torso_drive_list = robot.get_torso_jointState()
        torso_drive_now = float(torso_drive_list[0]) if torso_drive_list else 0.0
        if stg.q_torso_tgt is not None:
            t_clamped = float(np.clip(stg.q_torso_tgt, *TORSO_JOINT4_LIMITS))
            torso_next = kin["q_torso"] + float(np.clip(t_clamped - kin["q_torso"], -0.08, 0.08))
            torso_delta = np.array([torso_next - torso_drive_now], dtype=np.float64)
        else:
            torso_delta = np.zeros(1, dtype=np.float64)

        action19 = np.concatenate(
            [q_left_cmd, [left_grip_cmd], q_right_cmd, [right_grip_cmd], head_delta, torso_delta],
            axis=0,
        )
        task_env.take_action(action19, action_type="qpos")

        stg.steps_remaining -= 1
        if converged_early:
            stg.steps_remaining = 0
        if stg.steps_remaining <= 0:
            kin_after = self._read_robot_state(task_env)
            if stg.save_carry_qpos_at_end:
                self._carry_qpos7 = (
                    kin_after["q_left7"].copy() if stg.arm == "left" else kin_after["q_right7"].copy()
                )
            if stg.set_holding_at_end is not None:
                self.memory.holding = bool(stg.set_holding_at_end)
            if stg.set_inspected_at_end:
                self.memory.inspected_current_object = True
            if stg.pin_reveal_at_end:
                tcp_sync = (
                    kin["left_tcp7"][:3] if stg.arm == "left" else kin["right_tcp7"][:3]
                )
                crop = extract_proprio_reveal_crop(
                    rgb=rgb,
                    tcp_world_xyz=tcp_sync,
                    T_world_cam=kin["T_world_cam"],
                    crop_size=256,
                )
                self.memory.pin_reveal_frame(crop)
            self._stage_queue.popleft()

    def step_action(self, task_env: Any, observation: dict[str, Any]) -> None:
        rgb = _extract_head_rgb(observation)
        kin = self._read_robot_state(task_env)
        step = int(getattr(task_env, "take_action_cnt", 0))

        if not self._stage_queue:
            if self._episode_halted or self.turn_idx >= self.max_turns:
                # Hold open homestate quietly until simulator loop concludes
                left_home = np.asarray(task_env.robot.left_homestate, dtype=np.float64)
                self._stage_queue.append(
                    _PrimitiveStage("IDLE:limit", 10, arm="left", joint_tgt7=left_home, grip_tgt=1.0)
                )
            else:
                self.turn_idx += 1
                instruction = str(task_env.get_instruction())
                excl = self._proprio_exclude_uvs(rgb.shape, kin)
                eo = build_enhanced_observation(
                    rgb=rgb,
                    T_world_cam=kin["T_world_cam"],
                    T_world_robot0=kin["T_world_robot0"],
                    exclude_uvs=excl,
                )
                if self.eo_debug_dir is not None and self.episode_idx <= 3:
                    Image.fromarray(eo.annotated_rgb).save(
                        self.eo_debug_dir / f"ep{self.episode_idx:02d}_turn{self.turn_idx:02d}_step{step:03d}.jpg"
                    )
                    if self.memory.reveal_frames:
                        Image.fromarray(self.memory.reveal_frames[-1]).save(
                            self.eo_debug_dir / f"ep{self.episode_idx:02d}_turn{self.turn_idx:02d}_reveal.jpg"
                        )

                resp = self._http.post(
                    self.server_url,
                    json={
                        "instruction": instruction,
                        "eo_image_b64": _encode_jpeg_b64(eo.annotated_rgb),
                        "reveal_images_b64": [_encode_jpeg_b64(im) for im in self.memory.reveal_frames],
                        "eo_catalog": eo.text_catalog,
                        "memory_context": self.memory.format_context_lookup(),
                    },
                    timeout=self.timeout_s,
                )
                resp.raise_for_status()
                decision = resp.json()
                outcome_note = self._enqueue_primitive(decision, eo, kin, task_env)
                self.memory.record_action(
                    turn=self.turn_idx,
                    primitive=str(decision.get("primitive", "")),
                    target=str(decision.get("target", "")),
                    note=outcome_note,
                )

                if self.trace_path is not None:
                    with self.trace_path.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "episode_idx": self.episode_idx,
                                    "turn_idx": self.turn_idx,
                                    "sim_step": step,
                                    "instruction": instruction,
                                    "keypoints": {
                                        k: {
                                            "color": v.color_tag,
                                            "zone": v.kind,
                                            "xy": [round(float(x), 3) for x in v.world_xyz[:2]],
                                        }
                                        for k, v in eo.keypoints.items()
                                    },
                                    "pushlines": list(eo.pushlines.keys()),
                                    "holding_before": self.memory.holding,
                                    "reveal_frames_count": len(self.memory.reveal_frames),
                                    "vlm_thought": decision.get("thought", ""),
                                    "primitive": decision.get("primitive", ""),
                                    "target": decision.get("target", ""),
                                    "vlm_latency_ms": decision.get("latency_ms", 0.0),
                                    "outcome_note": outcome_note,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

        self._execute_stage_step(self._stage_queue[0], task_env, kin, rgb)


def get_model(usr_args: dict[str, Any]) -> ZSIPAstribotPolicyClient:
    return ZSIPAstribotPolicyClient(usr_args)


def reset_model(model: ZSIPAstribotPolicyClient) -> None:
    model.reset()


def eval(TASK_ENV: Any, model: ZSIPAstribotPolicyClient, observation: dict[str, Any]) -> None:
    model.step_action(TASK_ENV, observation)
