"""ActiveArena closed-loop policy adapter for Embodiment-Aligned ASA (`EgoActiveFlowModelv2`).

Loaded directly by unmodified `ActiveArena/script/eval_policy.py` via:
    --policy_name eval.activearena.asa_policy

Aligns ActiveArena's `Astribot S1` whole-body kinematics (head camera, head pan/tilt,
torso yaw, dual 7-DoF arms + mimic grippers) to ASA's 29-DoF egocentric representation
and executes coordinated active-perception (`view`) + bimanual (`mani`) trajectories.
"""
from __future__ import annotations

import base64
from collections import deque
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import requests

from eval.activearena.asa_astribot_embodiment import (
    ASAAstribotAlignmentConfig,
    ASAAstribotEmbodimentAdapter,
    ASANormBundle,
    pose7_to_matrix,
    solve_sapien_arm_dls_ik,
)


def _encode_jpeg_b64(rgb: np.ndarray, quality: int = 90) -> str:
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


class ASAAstribotPolicyClient:
    """Embodiment-aligned ASA policy client for ActiveArena's `Astribot S1`."""

    def __init__(self, usr_args: dict[str, Any]) -> None:
        self.usr_args = dict(usr_args)
        self.server_url = os.environ.get("ASA_SERVER_URL", "http://127.0.0.1:19200/act")
        self.timeout_s = float(os.environ.get("ASA_SERVER_TIMEOUT", "30.0"))
        self.exec_horizon = int(os.environ.get("ASA_EXEC_HORIZON", "6"))
        self.history_stride = int(os.environ.get("ASA_HISTORY_STRIDE", "12"))
        self.history_max = 5
        self.control_mode = os.environ.get("ASA_CONTROL_MODE", "ik_qpos")  # "ik_qpos" | "curobo_ee"
        self.subtask_mode = os.environ.get("ASA_SUBTASK_MODE", "instruction")  # "instruction" | "oracle_subtask"

        norm_profile = os.environ.get("ASA_NORM_PROFILE", "astribot_human_aligned")
        asa_ckpt_dir = os.environ.get("ASA_CKPT_DIR", "")
        ckpt_norm_file = Path(asa_ckpt_dir) / "astribot_norm_stats.json" if asa_ckpt_dir else None
        asa_repo = Path(os.environ.get("ASA_REPO_ROOT", "/home/ubuntu/ygwang/IP-VLA/asa-iteration"))
        if ckpt_norm_file is not None and ckpt_norm_file.is_file():
            norm = ASANormBundle.from_astribot_stats_json(ckpt_norm_file)
        elif norm_profile == "holoassist":
            norm = ASANormBundle.from_asa_json(
                asa_repo / "dataset_preprocessing/norm_stats/human/normalization_stats_human_history.json",
                source_key="HoloAssist",
            )
        elif norm_profile == "monte02_step":
            norm = ASANormBundle.from_asa_json(
                asa_repo / "dataset_preprocessing/norm_stats/mix_tasks/monte02_hand_frame1_chunk3.json",
                source_key="Monte02",
                use_step_base=True,
            )
        else:
            norm = ASANormBundle.astribot_human_aligned()

        base_mode = os.environ.get("ASA_BASE_FRAME_MODE", "level_head")
        self.adapter = ASAAstribotEmbodimentAdapter(
            norm_bundle=norm,
            cfg=ASAAstribotAlignmentConfig(base_frame_mode=base_mode),  # type: ignore[arg-type]
        )

        trace_path = os.environ.get("ASA_TRACE_LOG_PATH", "")
        self.trace_path = Path(trace_path) if trace_path else None
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)

        self._http = requests.Session()
        self._http.trust_env = False

        self.episode_idx = 0
        self._history_buffer: list[tuple[int, str, list[float], int]] = []
        self._pending_chunk: deque[tuple[np.ndarray, np.ndarray]] = deque()
        self._last_instruction: str | None = None
        self._last_p_complete: float = 0.0

    def reset(self) -> None:
        self.adapter.reset()
        self._history_buffer.clear()
        self._pending_chunk.clear()
        self._last_instruction = None
        self._last_p_complete = 0.0
        self.episode_idx += 1

    def _read_astribot_kinematics(self, task_env: Any) -> dict[str, Any]:
        robot = task_env.robot
        cam_pose = robot.head_camera.get_entity_pose()
        T_world_cam = pose7_to_matrix(np.concatenate([cam_pose.p, cam_pose.q]))

        rp = robot.left_entity_origion_pose
        T_world_robot0 = pose7_to_matrix(np.concatenate([rp.p, rp.q]))

        left_tcp7 = np.asarray(robot.get_left_tcp_pose(), dtype=np.float64)
        right_tcp7 = np.asarray(robot.get_right_tcp_pose(), dtype=np.float64)
        left_ee7 = np.asarray(robot.get_left_ee_pose(), dtype=np.float64)
        right_ee7 = np.asarray(robot.get_right_ee_pose(), dtype=np.float64)

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
            "left_grip": float(left_grip),
            "right_grip": float(right_grip),
            "q_head": q_head,
            "q_torso": q_torso,
        }

    def _select_causal_history(self, step: int) -> tuple[list[str], list[list[float]], list[int]]:
        if not self._history_buffer:
            return [], [], []
        # Select up to 5 past snapshots spaced by `history_stride` steps (matching ~1 Hz ASA history)
        selected: list[tuple[int, str, list[float], int]] = []
        target_step = step - self.history_stride
        for s, img_b64, p29, comp in reversed(self._history_buffer):
            if s <= target_step:
                selected.append((s, img_b64, p29, comp))
                target_step = s - self.history_stride
                if len(selected) >= self.history_max:
                    break
        selected.reverse()
        return (
            [x[1] for x in selected],
            [x[2] for x in selected],
            [x[3] for x in selected],
        )

    def _resolve_text_prompt(self, task_env: Any) -> str:
        base_instr = str(task_env.get_instruction())
        if self.subtask_mode == "oracle_subtask":
            for attr in ("get_current_subtask_instruction", "current_subtask_instruction", "subtask_instruction"):
                val = getattr(task_env, attr, None)
                text = val() if callable(val) else val
                if isinstance(text, str) and text.strip():
                    return text.strip()
        return base_instr

    def step_action(self, task_env: Any, observation: dict[str, Any]) -> None:
        prompt = self._resolve_text_prompt(task_env)
        if self._last_instruction is not None and prompt != self._last_instruction:
            # Re-anchor `step_base` when subtask prompt changes (matching ASA `pose_in_step_base`!)
            self.adapter.reset()
            self._pending_chunk.clear()
        self._last_instruction = prompt

        step = int(getattr(task_env, "take_action_cnt", 0))
        rgb = _extract_head_rgb(observation)
        kin = self._read_astribot_kinematics(task_env)

        proprio29 = self.adapter.encode_proprio29(
            T_world_cam=kin["T_world_cam"],
            left_tcp_pose7=kin["left_tcp7"],
            right_tcp_pose7=kin["right_tcp7"],
            left_gripper=kin["left_grip"],
            right_gripper=kin["right_grip"],
            T_world_robot0=kin["T_world_robot0"],
            q_torso=kin["q_torso"],
            q_head1=float(kin["q_head"][0]),
        )
        cur_img_b64 = _encode_jpeg_b64(rgb)

        queried_server = False
        server_latency_ms = 0.0
        if not self._pending_chunk:
            hist_imgs, hist_p29, hist_comp = self._select_causal_history(step)
            resp = self._http.post(
                self.server_url,
                json={
                    "instruction": prompt,
                    "current_image_b64": cur_img_b64,
                    "history_images_b64": hist_imgs,
                    "current_proprio29": proprio29.tolist(),
                    "history_proprio29": hist_p29,
                    "history_complete": hist_comp,
                },
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            pred = resp.json()
            queried_server = True
            server_latency_ms = float(pred.get("latency_ms", 0.0))
            self._last_p_complete = float(pred.get("p_complete", 0.0))

            view_seq = np.asarray(pred["view_actions"], dtype=np.float32)   # (30, 9)
            mani_seq = np.asarray(pred["mani_actions"], dtype=np.float32)   # (30, 2, 10)
            # Sample `exec_horizon` evenly strided waypoints across the first 18 steps (~0.6s) for smooth control
            n_exec = min(self.exec_horizon, len(view_seq))
            stride = max(1, min(3, len(view_seq) // max(n_exec, 1)))
            for k in range(n_exec):
                idx = min(k * stride, len(view_seq) - 1)
                self._pending_chunk.append((view_seq[idx], mani_seq[idx]))

        self._history_buffer.append(
            (step, cur_img_b64, proprio29.tolist(), int(self._last_p_complete >= 0.5))
        )
        if len(self._history_buffer) > 120:
            self._history_buffer = self._history_buffer[-120:]

        view_step, mani_step = self._pending_chunk.popleft()

        q_head_tgt, q_torso_tgt = self.adapter.decode_view_target_to_head_torso(
            view_step,
            q_head_now=kin["q_head"],
            q_torso_now=kin["q_torso"],
        )
        left_ee_tgt7, left_grip_tgt, right_ee_tgt7, right_grip_tgt = self.adapter.decode_mani_target_to_ee_poses(
            mani_step,
            left_ee_now7=kin["left_ee7"],
            right_ee_now7=kin["right_ee7"],
        )

        robot = task_env.robot
        head_delta = q_head_tgt - np.asarray(robot.get_head_jointState(), dtype=np.float64)
        torso_now_drive = robot.get_torso_jointState()
        torso_delta = np.array([q_torso_tgt - (float(torso_now_drive[0]) if torso_now_drive else 0.0)], dtype=np.float64)

        if self.control_mode == "curobo_ee":
            action19 = np.concatenate(
                [left_ee_tgt7, [left_grip_tgt], right_ee_tgt7, [right_grip_tgt], head_delta, torso_delta],
                axis=0,
            )
            task_env.take_action(action19, action_type="ee")
        else:
            left_endlink = robot._trans_from_gripper_to_endlink(left_ee_tgt7, arm_tag="left")
            right_endlink = robot._trans_from_gripper_to_endlink(right_ee_tgt7, arm_tag="right")
            q_left7 = solve_sapien_arm_dls_ik(
                robot.left_entity,
                robot.left_arm_joints,
                robot.left_ee_link,
                np.concatenate([left_endlink.p, left_endlink.q]),
                q_homestate=np.asarray(robot.left_homestate, dtype=np.float64),
                damping=self.adapter.cfg.ik_damping,
                iters=self.adapter.cfg.ik_iters,
                posture_gain=self.adapter.cfg.ik_posture_gain,
            )
            q_right7 = solve_sapien_arm_dls_ik(
                robot.right_entity,
                robot.right_arm_joints,
                robot.right_ee_link,
                np.concatenate([right_endlink.p, right_endlink.q]),
                q_homestate=np.asarray(robot.right_homestate, dtype=np.float64),
                damping=self.adapter.cfg.ik_damping,
                iters=self.adapter.cfg.ik_iters,
                posture_gain=self.adapter.cfg.ik_posture_gain,
            )
            action19 = np.concatenate(
                [q_left7, [left_grip_tgt], q_right7, [right_grip_tgt], head_delta, torso_delta],
                axis=0,
            )
            task_env.take_action(action19, action_type="qpos")

        if self.trace_path:
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "episode_idx": self.episode_idx,
                            "step": step,
                            "prompt": prompt,
                            "queried_server": queried_server,
                            "server_latency_ms": server_latency_ms,
                            "p_complete": round(self._last_p_complete, 4),
                            "q_head_now": np.round(kin["q_head"], 4).tolist(),
                            "q_head_tgt": np.round(q_head_tgt, 4).tolist(),
                            "q_torso_now": round(kin["q_torso"], 4),
                            "q_torso_tgt": round(q_torso_tgt, 4),
                            "left_ee_tgt_xyz": np.round(left_ee_tgt7[:3], 4).tolist(),
                            "right_ee_tgt_xyz": np.round(right_ee_tgt7[:3], 4).tolist(),
                            "left_grip_tgt": round(left_grip_tgt, 3),
                            "right_grip_tgt": round(right_grip_tgt, 3),
                            "control_mode": self.control_mode,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def get_model(usr_args: dict[str, Any]) -> ASAAstribotPolicyClient:
    return ASAAstribotPolicyClient(usr_args)


def reset_model(model: ASAAstribotPolicyClient) -> None:
    model.reset()


def eval(TASK_ENV: Any, model: ASAAstribotPolicyClient, observation: dict[str, Any]) -> None:
    model.step_action(TASK_ENV, observation)
