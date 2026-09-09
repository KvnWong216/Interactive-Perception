"""Frozen, weight-free configuration validation for PSR V1."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from grounded_interaction.contracts import canonical_json_bytes, canonical_sha256

PSR_CONFIG_SCHEMA = "psr-v1-config-v1"
PSR_METHOD = "psr_v1"

DEFAULT_PSR_CONFIG: dict[str, Any] = {
    "schema_version": PSR_CONFIG_SCHEMA,
    "method": PSR_METHOD,
    "base": {
        "model_id": "allenai/MolmoAct2-LIBERO",
        "revision": "0d24a92bd1faf321ef497c3bbd5681af97c65aa2",
        "upstream_revision": "66b87e64efd99dfd103241418113955cf64dfa9c",
        "norm_tag": "libero",
        "inference_action_mode": "continuous",
        "enable_depth_reasoning": False,
        "enable_cuda_graph": False,
    },
    "state": {
        "num_tokens": 6,
        "previous_observation_bundles": 2,
        "history_sampling": "high_level_boundary",
    },
    "intent": {
        "generated_candidates": 2,
        "include_native_candidate": True,
        "max_new_tokens": 32,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_generation_attempts": 4,
        "readout_dim": 512,
        "readout_heads": 8,
        "encoder_layers": 1,
        "predictor_layers": 2,
    },
    "evidence": {
        "target": "frozen_native_visual_patch_features",
        "projection_dim": 128,
        "projection_seed": 17,
        "mixture_components": 4,
        "min_log_std": -5.0,
        "max_log_std": 2.0,
    },
    "execution": {
        "total_control_steps": 300,
        "intent_window_steps": 50,
        "execute_chunk_steps": 10,
        "flow_steps": 10,
        "continuation": "native_molmoact2",
    },
    "adaptation": {
        "upper_vlm_layers": 8,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "adapt_native_context_kv_projections": True,
    },
    "training": {
        "dtype": "bfloat16",
        "vlm_lr": 0.00002,
        "new_parameters_lr": 0.0001,
        "weight_decay": 0.01,
        "grad_clip_norm": 1.0,
        "microbatch": 1,
        "effective_batch": 16,
        "seed": 17,
    },
}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_plain(child) for child in value]
    return value


def validate_psr_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Require the byte-semantic frozen V1 protocol, with useful diffs."""

    try:
        canonical = json.loads(canonical_json_bytes(value).decode("utf-8"))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            "PSR configuration must be finite JSON-compatible data"
        ) from error
    if not isinstance(canonical, dict):
        raise TypeError("PSR configuration root must be a mapping")
    expected = DEFAULT_PSR_CONFIG
    if canonical != expected:
        observed_keys = set(canonical)
        expected_keys = set(expected)
        if observed_keys != expected_keys:
            raise ValueError(
                "PSR config keys differ from frozen V1; "
                f"missing={sorted(expected_keys - observed_keys)}, "
                f"extra={sorted(observed_keys - expected_keys)}"
            )
        differing = [name for name in expected if canonical[name] != expected[name]]
        raise ValueError(f"PSR config sections differ from frozen V1: {differing}")
    return canonical


@dataclasses.dataclass(frozen=True)
class PSRConfig:
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _freeze(validate_psr_config(self.values)))

    def section(self, name: str) -> Mapping[str, Any]:
        section = self.values.get(name)
        if not isinstance(section, Mapping):
            raise KeyError(name)
        return section

    def to_dict(self) -> dict[str, Any]:
        return _plain(self.values)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())


def load_psr_config(path: str | Path) -> PSRConfig:
    source = Path(path).expanduser().resolve()
    suffix = source.suffix.lower()
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read PSR configuration {source}") from error
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("YAML configs require PyYAML") from error
        parsed = yaml.safe_load(text)
    elif suffix == ".json":
        parsed = json.loads(text)
    else:
        raise ValueError("PSR configuration must use .yaml, .yml, or .json")
    if not isinstance(parsed, Mapping):
        raise TypeError("PSR configuration root must be a mapping")
    return PSRConfig(parsed)


@dataclasses.dataclass(frozen=True)
class NativeArchitecture:
    """Facts that must be discovered from the loaded checkpoint."""

    hidden_size: int
    num_hidden_layers: int
    num_key_value_heads: int
    action_horizon: int
    action_dim: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> NativeArchitecture:
        expected = {
            "hidden_size",
            "num_hidden_layers",
            "num_key_value_heads",
            "action_horizon",
            "action_dim",
        }
        if set(value) != expected:
            raise ValueError("checkpoint architecture fields are incomplete or unknown")
        parsed: dict[str, int] = {}
        for name in expected:
            item = value[name]
            if not isinstance(item, int) or isinstance(item, bool) or item < 1:
                raise ValueError(f"{name} must be a positive discovered integer")
            parsed[name] = item
        return cls(**parsed)


def validate_native_architecture(
    config: PSRConfig,
    discovered: Mapping[str, Any],
) -> NativeArchitecture:
    """Check V1 against real architecture values instead of hard-coding them."""

    architecture = NativeArchitecture.from_mapping(discovered)
    upper_layers = int(config.section("adaptation")["upper_vlm_layers"])
    chunk_steps = int(config.section("execution")["execute_chunk_steps"])
    if upper_layers > architecture.num_hidden_layers:
        raise ValueError("upper_vlm_layers exceeds the discovered layer count")
    if chunk_steps > architecture.action_horizon:
        raise ValueError("execute_chunk_steps exceeds the native action horizon")
    return architecture
