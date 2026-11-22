"""ActiveArena closed-loop policy adapter for AAWR (`Asymmetric Advantage-Weighted Regression`).

Loaded directly by unmodified `ActiveArena/script/eval_policy.py` via:
    --policy_name eval.activearena.aawr_policy

Strict unprivileged deployment contract:
- Passes ONLY causal head RGB frames + 18-DoF robot joint state + task instruction to `aawr_server.py`.
- Never accesses privileged object poses, backside labels, or simulator subtask indices at evaluation time.
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

try:
    from policy.starvla_astribot.deploy_policy import (
        adapt_model_action_to_robotwin,
        build_astribot_18d_state,
        encode_obs,
    )
except ImportError:  # pragma: no cover - fallback when unit-tested outside simulator repo
    def build_astribot_18d_state(observation: dict[str, Any], torso_index: int = 0, head2_index: int = 1) -> np.ndarray:
        ja = observation.get("joint_action", {})
        la = np.asarray(ja.get("left_arm", np.zeros(7)), dtype=np.float32).reshape(7)
        ra = np.asarray(ja.get("right_arm", np.zeros(7)), dtype=np.float32).reshape(7)
        lg = float(ja.get("left_gripper", 1.0))
        rg = float(ja.get("right_gripper", 1.0))
        torso = float(np.asarray(ja.get("torso", [0.0])).reshape(-1)[torso_index])
        head = float(np.asarray(ja.get("head", [0.0, 1.0])).reshape(-1)[head2_index])
        return np.concatenate([la, [lg], ra, [rg], [torso, head]]).astype(np.float32)

    def encode_obs(observation: dict[str, Any]) -> tuple[np.ndarray, str]:
        obs = observation["observation"]
        for k in ("camera_head", "head_camera"):
            if k in obs and "rgb" in obs[k]:
                return np.asarray(obs[k]["rgb"], dtype=np.uint8), k
        raise KeyError("Missing head camera RGB")

    def adapt_model_action_to_robotwin(action: np.ndarray, task_env: Any) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64).reshape(18)
        out = np.zeros(19, dtype=np.float64)
        out[:16] = a[:16]
        return out


def _encode_jpeg_b64(rgb: np.ndarray, quality: int = 88) -> str:
    arr = np.asarray(rgb, dtype=np.uint8)
    buf = io.BytesIO()
    with Image.fromarray(arr).convert("RGB") as im:
        im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class AAWRAstribotPolicyClient:
    """Unprivileged deployment client for AAWR on ActiveArena's `Astribot S1`."""

    def __init__(self, usr_args: dict[str, Any]) -> None:
        self.usr_args = dict(usr_args)
        self.server_url = os.environ.get("AAWR_SERVER_URL", "http://127.0.0.1:19400/act")
        self.timeout_s = float(os.environ.get("AAWR_SERVER_TIMEOUT", "20.0"))
        self.exec_horizon = int(os.environ.get("AAWR_EXEC_HORIZON", "4"))
        self.history_stride = int(os.environ.get("AAWR_HISTORY_STRIDE", "8"))
        self.history_frames = 4

        trace_path = os.environ.get("AAWR_TRACE_LOG_PATH", "")
        self.trace_path = Path(trace_path) if trace_path else None
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)

        self._http = requests.Session()
        self._http.trust_env = False
        self.episode_idx = 0
        self._frame_history: list[tuple[int, str]] = []
        self._pending_actions: deque[np.ndarray] = deque()
        self._last_instruction: str | None = None

    def reset(self) -> None:
        self._frame_history.clear()
        self._pending_actions.clear()
        self._last_instruction = None
        self.episode_idx += 1

    def _select_causal_b64(self) -> list[str]:
        if not self._frame_history:
            return []
        cur_step, cur_b64 = self._frame_history[-1]
        selected = [cur_b64]
        target = cur_step - self.history_stride
        for s, b64 in reversed(self._frame_history[:-1]):
            if s <= target:
                selected.append(b64)
                target = s - self.history_stride
                if len(selected) >= self.history_frames:
                    break
        selected.reverse()
        while len(selected) < self.history_frames:
            selected.insert(0, selected[0])
        return selected

    def _log_trace(self, entry: dict[str, Any]) -> None:
        if not self.trace_path:
            return
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def step_action(self, task_env: Any, observation: dict[str, Any]) -> np.ndarray:
        instruction = str(task_env.get_instruction())
        if self._last_instruction is not None and instruction != self._last_instruction:
            self.reset()
        self._last_instruction = instruction

        step = int(getattr(task_env, "take_action_cnt", 0))
        rgb, _ = encode_obs(observation)
        qpos_18 = build_astribot_18d_state(observation)
        self._frame_history.append((step, _encode_jpeg_b64(rgb)))
        if len(self._frame_history) > 128:
            self._frame_history = self._frame_history[-128:]

        if not self._pending_actions:
            payload = {
                "instruction": instruction,
                "images_b64": self._select_causal_b64(),
                "qpos_18": qpos_18.tolist(),
            }
            resp = self._http.post(self.server_url, json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()
            chunk = np.asarray(data["actions_18d"], dtype=np.float64)
            for row in chunk[: self.exec_horizon]:
                self._pending_actions.append(row)
            self._log_trace(
                {
                    "episode_idx": self.episode_idx,
                    "step": step,
                    "instruction": instruction,
                    "latency_ms": data.get("latency_ms"),
                    "chunk_len": int(chunk.shape[0]),
                    "exec_horizon": self.exec_horizon,
                }
            )

        act_18 = self._pending_actions.popleft()
        return adapt_model_action_to_robotwin(act_18, task_env)


def get_model(usr_args: dict[str, Any]) -> AAWRAstribotPolicyClient:
    return AAWRAstribotPolicyClient(usr_args)


def reset_model(model: AAWRAstribotPolicyClient) -> None:
    model.reset()


def eval(TASK_ENV: Any, model: AAWRAstribotPolicyClient, observation: dict[str, Any]) -> None:
    env_action = model.step_action(TASK_ENV, observation)
    TASK_ENV.take_action(env_action)
