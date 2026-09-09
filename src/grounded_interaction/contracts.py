"""Small, model-agnostic audit helpers shared by the PSR-VLA pipeline.

This module intentionally contains no policy schema, action vocabulary, or
learned decision rule. It only validates public JSON payloads and computes
stable content fingerprints.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

_FORBIDDEN_PUBLIC_KEY_FRAGMENTS = (
    "privileged",
    "oracle",
    "evaluator",
    "semantic_id",
    "instance_id",
    "segmentation",
    "ground_truth",
    "groundtruth",
    "target_pose",
    "object_pose",
    "simulator",
    "sim_state",
    "simulator_state",
    "task_predicate",
    "joint_qpos",
    "route_label",
    "effect_label",
    "correct_action",
    "task_success",
    "reward",
)


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _freeze_json_value(value: Any, *, path: str) -> Any:
    """Validate and recursively freeze a finite JSON-compatible value."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite public value at {path}")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(f"public mapping key at {path} must be a string")
            key = " ".join(raw_key.split())
            if not key:
                raise ValueError(f"empty public mapping key at {path}")
            normalized = _normalized_key(key)
            if any(part in normalized for part in _FORBIDDEN_PUBLIC_KEY_FRAGMENTS):
                raise ValueError(f"privileged policy field at {path}.{key}")
            if key in frozen:
                raise ValueError(f"duplicate public mapping key at {path}.{key}")
            frozen[key] = _freeze_json_value(child, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_json_value(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        )
    raise TypeError(
        f"public value at {path} must be JSON-compatible, got {type(value).__name__}"
    )


def _plain_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_plain_json_value(child) for child in value]
    return value


def assert_public_policy_value(value: Any, *, path: str = "policy_input") -> None:
    """Fail closed when a policy payload is not finite public JSON data."""

    _freeze_json_value(value, path=path)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a finite JSON-compatible value deterministically."""

    return json.dumps(
        _plain_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return a reproducible SHA-256 fingerprint for a contract payload."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


__all__ = [
    "assert_public_policy_value",
    "canonical_json_bytes",
    "canonical_sha256",
]
