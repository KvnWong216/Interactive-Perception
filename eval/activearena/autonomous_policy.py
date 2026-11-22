"""Drop-in ActiveArena closed-loop policy adapter for IP-VLA Stage-2.

Loaded directly by ActiveArena's unmodified `script/eval_policy.py` via:
    --policy_name eval.activearena.autonomous_policy

Architecture:
- Subclasses ActiveArena's official `StarVLAOFTClient` (`oft_subtask_action_12_ws`) so
  observation encoding, 18-DoF state ordering, 12-keyframe executor history, and action
  chunk unnormalization remain 100% byte-for-byte identical to the official benchmark.
- Replaces ONLY `subtask_instruction = self._subtask_instruction(task_env, annotation)`
  (the privileged simulator oracle) with our autonomous Stage-2 controller:
    1. Continuous cerebellum inertia: phi_t = max(phi_kin, phi_grip, phi_dw)
    2. Non-oracle inspect-keyframe memory buffer (pins reveal frames in slots 1..2)
    3. Two-stage wake gate (r_t > b + kappa * phi_t and t - t_v >= T_lock)
    4. Autonomous Qwen3-VL-2B planner call over localhost HTTP (`IPVLA_PLANNER_URL`)
    5. Relative confidence commit gate (c(u') - c(u_k) >= delta_0 + kappa_c * phi_t)
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
import requests

from ipvla.config import load_stage2_config
from ipvla.stage2.cerebellum import CerebellumConfig
from ipvla.stage2.memory import MemoryConfig
from ipvla.stage2.switching import SwitchingConfig, TwoStageSwitchingController

try:
    from policy.starvla_astribot.deploy_policy import (
        StarVLAOFTClient,
        eval as _official_eval,
        get_model as _official_get_model,
        reset_model as _official_reset_model,
    )
except ImportError as exc:  # pragma: no cover - only triggered outside ActiveArena sim env
    StarVLAOFTClient = object  # type: ignore[misc,assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _encode_rgb_b64(img: np.ndarray, quality: int = 88) -> str:
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.5 else arr, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    with Image.fromarray(arr).convert("RGB") as pil:
        pil.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class IPVLAStage2ActiveArenaClient(StarVLAOFTClient):  # type: ignore[misc]
    """Autonomous IP-VLA Stage-2 planner coupled to frozen `oft_subtask_action_12_ws`."""

    def __init__(self, *args, **kwargs) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "ActiveArena simulator repo must be on PYTHONPATH to instantiate IPVLAStage2ActiveArenaClient"
            ) from _IMPORT_ERROR
        super().__init__(*args, **kwargs)

        cfg_path = os.environ.get("IPVLA_STAGE2_CONFIG") or None
        cfg = load_stage2_config(cfg_path)
        c_cfg = dict(cfg.get("cerebellum", {}))
        s_cfg = dict(cfg.get("switching", {}))
        m_cfg = dict(cfg.get("memory", {}))

        if os.environ.get("IPVLA_CEREBELLUM_MODE"):
            c_cfg["mode"] = os.environ["IPVLA_CEREBELLUM_MODE"]
        if os.environ.get("IPVLA_MEMORY_MODE"):
            m_cfg["mode"] = os.environ["IPVLA_MEMORY_MODE"]

        self.planner_url = os.environ.get("IPVLA_PLANNER_URL", "http://127.0.0.1:19100/plan")
        self.planner_timeout = float(os.environ.get("IPVLA_PLANNER_TIMEOUT", "20.0"))
        trace_path = os.environ.get("IPVLA_TRACE_LOG_PATH", "")
        self.trace_log_path = Path(trace_path) if trace_path else None
        if self.trace_log_path:
            self.trace_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.controller = TwoStageSwitchingController(
            cerebellum_cfg=CerebellumConfig(**{k: v for k, v in c_cfg.items() if k in CerebellumConfig.__dataclass_fields__}),
            memory_cfg=MemoryConfig(**{k: v for k, v in m_cfg.items() if k in MemoryConfig.__dataclass_fields__}),
            switching_cfg=SwitchingConfig(**{k: v for k, v in s_cfg.items() if k in SwitchingConfig.__dataclass_fields__}),
        )
        self._last_action_chunk: Optional[np.ndarray] = None
        self._http = requests.Session()
        self._http.trust_env = False

    def reset(self) -> None:
        super().reset()
        self.controller.reset()
        self._last_action_chunk = None

    def _query_planner(self, instruction: str, frames: list[np.ndarray]) -> dict[str, Any]:
        payload = {
            "instruction": instruction,
            "images_b64": [_encode_rgb_b64(im) for im in frames],
            "max_new_tokens": 28,
        }
        resp = self._http.post(self.planner_url, json=payload, timeout=self.planner_timeout)
        resp.raise_for_status()
        return resp.json()

    def _append_trace(self, entry: dict[str, Any]) -> None:
        if not self.trace_log_path:
            return
        with self.trace_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_actions(self, task_env, observation: dict[str, Any]) -> np.ndarray:
        """Autonomous closed-loop step: NEVER reads privileged env subtask oracle."""
        instruction = str(task_env.get_instruction())
        if self.language is not None and instruction != self.language:
            self.reset()
        self.language = instruction

        # Observe frame & 18-DoF joint state via official StarVLAOFTClient encoder.
        # Note: `_annotation_from_env` is NOT used for subtask selection!
        current, camera_key, _unused_annotation = self.observe_frame(task_env, observation)
        step = int(getattr(task_env, "take_action_cnt", current.step))

        wake_dec, inertia_snap, buf_frame = self.controller.observe_and_check_wake(
            step=step,
            image=current.image,
            joint_state=current.state,
            commanded_action_chunk=self._last_action_chunk,
        )

        commit_info: Optional[dict[str, Any]] = None
        planner_resp: Optional[dict[str, Any]] = None
        if wake_dec.should_wake:
            causal_frames = self.controller.memory.select_causal_frames()
            planner_resp = self._query_planner(
                instruction=instruction,
                frames=[f.image for f in causal_frames],
            )
            commit_dec = self.controller.evaluate_candidate(
                step=step,
                candidate_subtask=str(planner_resp.get("subtask", "")),
                candidate_confidence=float(planner_resp.get("confidence", 0.5)),
                frame=buf_frame,
                snap=inertia_snap,
                novelty_r=wake_dec.novelty_r,
            )
            commit_info = {
                "committed": commit_dec.committed,
                "reason": commit_dec.reason,
                "candidate_subtask": commit_dec.candidate_subtask,
                "candidate_confidence": commit_dec.candidate_confidence,
                "margin_required": commit_dec.margin_required,
                "pinned_inspect_frame": commit_dec.pinned_inspect_frame,
                "selected_frame_steps": [f.step for f in causal_frames],
            }

        autonomous_subtask = self.controller.active_subtask or instruction
        records = self._select_history(current)
        actions = self._request_actions(records, instruction, camera_key, autonomous_subtask)
        self._last_action_chunk = np.asarray(actions, dtype=np.float64)

        self._append_trace(
            {
                "episode_idx": int(getattr(self, "episode_idx", 0)),
                "step": step,
                "instruction": instruction,
                "active_subtask": autonomous_subtask,
                "active_confidence": round(self.controller.active_confidence, 4),
                "phi": round(inertia_snap.phi, 4),
                "phi_kin": round(inertia_snap.phi_kin, 4),
                "phi_grip": round(inertia_snap.phi_grip, 4),
                "phi_dw": round(inertia_snap.phi_dw, 4),
                "novelty_r": round(wake_dec.novelty_r, 4),
                "wake": wake_dec.should_wake,
                "wake_reason": wake_dec.reason,
                "pinned_steps": self.controller.memory.pinned_steps,
                "total_vlm_calls": self.controller.total_vlm_calls,
                "total_commits": self.controller.total_commits,
                "planner": planner_resp,
                "commit": commit_info,
            }
        )
        return actions


def get_model(usr_args: dict[str, Any]):
    base_client = _official_get_model(usr_args)
    base_client.__class__ = IPVLAStage2ActiveArenaClient
    # Initialize Stage-2 controller attributes onto the migrated instance cleanly.
    cfg_path = os.environ.get("IPVLA_STAGE2_CONFIG") or None
    cfg = load_stage2_config(cfg_path)
    c_cfg = dict(cfg.get("cerebellum", {}))
    s_cfg = dict(cfg.get("switching", {}))
    m_cfg = dict(cfg.get("memory", {}))
    if os.environ.get("IPVLA_CEREBELLUM_MODE"):
        c_cfg["mode"] = os.environ["IPVLA_CEREBELLUM_MODE"]
    if os.environ.get("IPVLA_MEMORY_MODE"):
        m_cfg["mode"] = os.environ["IPVLA_MEMORY_MODE"]
    base_client.planner_url = os.environ.get("IPVLA_PLANNER_URL", "http://127.0.0.1:19100/plan")
    base_client.planner_timeout = float(os.environ.get("IPVLA_PLANNER_TIMEOUT", "20.0"))
    trace_path = os.environ.get("IPVLA_TRACE_LOG_PATH", "")
    base_client.trace_log_path = Path(trace_path) if trace_path else None
    if base_client.trace_log_path:
        base_client.trace_log_path.parent.mkdir(parents=True, exist_ok=True)
    base_client.controller = TwoStageSwitchingController(
        cerebellum_cfg=CerebellumConfig(**{k: v for k, v in c_cfg.items() if k in CerebellumConfig.__dataclass_fields__}),
        memory_cfg=MemoryConfig(**{k: v for k, v in m_cfg.items() if k in MemoryConfig.__dataclass_fields__}),
        switching_cfg=SwitchingConfig(**{k: v for k, v in s_cfg.items() if k in SwitchingConfig.__dataclass_fields__}),
    )
    base_client._last_action_chunk = None
    base_client._http = requests.Session()
    base_client._http.trust_env = False
    return base_client


def reset_model(model):
    return _official_reset_model(model)


def eval(TASK_ENV, model, observation: dict[str, Any]):
    return _official_eval(TASK_ENV, model, observation)
