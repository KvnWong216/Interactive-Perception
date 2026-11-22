"""Portable configuration & path resolution for IP-VLA.

Priority order (highest first):
1. Explicit environment variables (`IPVLA_BASE_VLM`, `IPVLA_STAGE1_LORA`,
   `IPVLA_STAGE1_CONF_HEAD`, `ACTIVEARENA_ROOT`, `ACTIVEARENA_VLA_ROOT`,
   `ACTIVEARENA_VLA_WEIGHTS_ROOT`, `ACTIVEARENA_PYTHON`, `ACTIVEARENA_VLA_PYTHON`).
2. `configs/paths.local.yaml` (git-ignored machine-local overrides).
3. `configs/paths.example.yaml` (portable defaults relative to repository root).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(raw: str | Path) -> Path:
    p = Path(os.path.expandvars(str(raw))).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = _resolve_path(path)
    if not p.is_file():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def load_paths_config(local_override: str | Path | None = None) -> dict[str, Any]:
    base = load_yaml(REPO_ROOT / "configs" / "paths.example.yaml")
    local_path = (
        _resolve_path(local_override)
        if local_override
        else REPO_ROOT / "configs" / "paths.local.yaml"
    )
    local = load_yaml(local_path)

    merged: dict[str, Any] = dict(base)
    for key, val in local.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **val}
        else:
            merged[key] = val

    aa = merged.setdefault("activearena", {})
    env_top = {
        "base_vlm_dir": "IPVLA_BASE_VLM",
        "stage1_lora_path": "IPVLA_STAGE1_LORA",
        "stage1_confidence_head_path": "IPVLA_STAGE1_CONF_HEAD",
    }
    for k, env_name in env_top.items():
        if os.environ.get(env_name):
            merged[k] = os.environ[env_name]

    env_aa = {
        "sim_repo": "ACTIVEARENA_ROOT",
        "vla_repo": "ACTIVEARENA_VLA_ROOT",
        "weights_root": "ACTIVEARENA_VLA_WEIGHTS_ROOT",
        "sim_python": "ACTIVEARENA_PYTHON",
        "vla_python": "ACTIVEARENA_VLA_PYTHON",
        "planner_python": "IPVLA_PLANNER_PYTHON",
        "output_root": "IPVLA_ACTIVEARENA_OUTPUT_ROOT",
    }
    for k, env_name in env_aa.items():
        if os.environ.get(env_name):
            aa[k] = os.environ[env_name]

    for k in ("base_vlm_dir", "stage1_lora_path", "stage1_confidence_head_path"):
        if merged.get(k):
            merged[k] = str(_resolve_path(merged[k]))
    for k in ("sim_repo", "vla_repo", "weights_root", "output_root"):
        if aa.get(k):
            aa[k] = str(_resolve_path(aa[k]))
    for k in ("sim_python", "vla_python", "planner_python"):
        val = aa.get(k, "")
        if val and ("/" in str(val)):
            aa[k] = str(_resolve_path(val))

    return merged


def load_stage2_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = _resolve_path(path) if path else REPO_ROOT / "configs" / "stage2_activearena.yaml"
    return load_yaml(cfg_path)
