"""Pinned MolmoAct2-LIBERO inference boundary.

This is a narrow transport adapter, not a replacement for MolmoAct2.  The GPU
server calls the checkpoint's public ``predict_action`` API.  The client sends
only the two deployment RGB views, the public 8-D robot state, and the selected
subtask text.  Candidate and evaluator identities remain outside model input.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import math
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

from .contracts import canonical_sha256
from .rgb import (
    canonical_rgb_array,
    canonical_rgb_sha256,
    decode_rgb_png,
    encode_rgb_png,
)

MOLMOACT2_REPO_ID = "allenai/MolmoAct2-LIBERO"
MOLMOACT2_MODEL_REVISION = "0d24a92bd1faf321ef497c3bbd5681af97c65aa2"
MOLMOACT2_CODE_REVISION = "66b87e64efd99dfd103241418113955cf64dfa9c"
MOLMOACT2_HTTP_SCHEMA = "molmoact2-libero-http-v1"
MOLMOACT2_LIVE_BACKEND_KIND = "transformers-molmoact2-predict-action-v1"
MOLMOACT2_TORCH_VERSION = "2.11.0"
MOLMOACT2_TRANSFORMERS_VERSION = "4.57.6"
MOLMOACT2_CUDA_VERSION = "12.8"
MOLMOACT2_NORM_STATS_FORMAT = "molmoact2_norm_stats.v1"
MOLMOACT2_NORM_MODE = "q01_q99"
MOLMOACT2_CONTROL_MODE = "delta end-effector pose"
_UNFROZEN_SHA256 = "0" * 64
_MOLMOACT2_INFERENCE_FILES = (
    "chat_template.jinja",
    "config.json",
    "configuration_molmoact2.py",
    "generation_config.json",
    "image_processing_molmoact2.py",
    "inference.py",
    "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors",
    "model-00005-of-00005.safetensors",
    "model.safetensors.index.json",
    "modeling_molmoact2.py",
    "norm_stats.json",
    "processing_molmoact2.py",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_processing_molmoact2.py",
)


def molmoact2_adapter_sha256() -> str:
    """Hash the exact client/server adapter source used for one execution."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class MolmoAct2Error(RuntimeError):
    """Base class for the concrete Stage-2 transport boundary."""


class MolmoAct2TransportError(MolmoAct2Error):
    """Endpoint, model-runtime, or network failure with no behavioral label."""


class MolmoAct2ProtocolError(MolmoAct2Error):
    """Frozen client/server contract mismatch; fail closed with no label."""


class MolmoAct2PolicyOutputError(MolmoAct2Error):
    """The live policy returned a malformed/non-finite action chunk."""


def _clean_text(value: object, *, name: str) -> str:
    result = " ".join(str(value or "").split())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _finite_vector(
    value: Sequence[float], *, length: int, name: str
) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric sequence") from error
    if len(result) != length or any(not math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def _require_sha256(value: object, *, name: str) -> str:
    result = _clean_text(value, name=name)
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


@dataclasses.dataclass(frozen=True)
class MolmoAct2ServerIdentity:
    """Every setting that changes the Stage-2 action distribution."""

    checkpoint_id: str = MOLMOACT2_REPO_ID
    checkpoint_revision: str = MOLMOACT2_MODEL_REVISION
    upstream_code_revision: str = MOLMOACT2_CODE_REVISION
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    norm_tag: str = "libero"
    inference_action_mode: str = "continuous"
    normalize_language: bool = True
    camera_order: tuple[str, str] = ("agentview", "wrist")
    image_size: int = 256
    state_dim: int = 8
    action_dim: int = 7
    action_horizon: int = 10
    num_steps: int = 10
    enable_depth_reasoning: bool = False
    enable_cuda_graph: bool = False
    config_sha256: str = _UNFROZEN_SHA256
    norm_stats_sha256: str = _UNFROZEN_SHA256
    checkpoint_manifest_sha256: str = _UNFROZEN_SHA256
    checkpoint_file_count: int = 0
    checkpoint_total_bytes: int = 0
    norm_stats_format: str = MOLMOACT2_NORM_STATS_FORMAT
    norm_mode: str = MOLMOACT2_NORM_MODE
    control_mode: str = MOLMOACT2_CONTROL_MODE

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_id",
            "checkpoint_revision",
            "upstream_code_revision",
            "dtype",
            "device",
            "norm_tag",
            "inference_action_mode",
            "norm_stats_format",
            "norm_mode",
            "control_mode",
        ):
            object.__setattr__(self, name, _clean_text(getattr(self, name), name=name))
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("dtype must be bfloat16 or float32")
        if tuple(self.camera_order) != ("agentview", "wrist"):
            raise ValueError("MolmoAct2-LIBERO camera order must be agentview, wrist")
        for name, expected in (
            ("state_dim", 8),
            ("action_dim", 7),
            ("action_horizon", 10),
            ("num_steps", 10),
            ("image_size", 256),
        ):
            if getattr(self, name) != expected:
                raise ValueError(
                    f"{name} must be {expected} for the frozen E1 contract"
                )
        if self.norm_tag != "libero":
            raise ValueError("norm_tag must be libero")
        if self.inference_action_mode != "continuous":
            raise ValueError("inference_action_mode must be continuous")
        if self.normalize_language is not True:
            raise ValueError("normalize_language must be enabled")
        if self.enable_depth_reasoning:
            raise ValueError("depth reasoning is unsupported by MolmoAct2-LIBERO")
        for name in (
            "config_sha256",
            "norm_stats_sha256",
            "checkpoint_manifest_sha256",
        ):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), name=name)
            )
        if (
            not isinstance(self.checkpoint_file_count, int)
            or isinstance(self.checkpoint_file_count, bool)
            or self.checkpoint_file_count < 0
            or not isinstance(self.checkpoint_total_bytes, int)
            or isinstance(self.checkpoint_total_bytes, bool)
            or self.checkpoint_total_bytes < 0
        ):
            raise ValueError("checkpoint manifest counts must be non-negative integers")
        if self.norm_stats_format != MOLMOACT2_NORM_STATS_FORMAT:
            raise ValueError("unexpected MolmoAct2 norm-stats format")
        if self.norm_mode != MOLMOACT2_NORM_MODE:
            raise ValueError("unexpected MolmoAct2 normalization mode")
        if self.control_mode != MOLMOACT2_CONTROL_MODE:
            raise ValueError("MolmoAct2-LIBERO requires delta end-effector control")

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_revision": self.checkpoint_revision,
            "upstream_code_revision": self.upstream_code_revision,
            "dtype": self.dtype,
            "device": self.device,
            "norm_tag": self.norm_tag,
            "inference_action_mode": self.inference_action_mode,
            "normalize_language": self.normalize_language,
            "camera_order": list(self.camera_order),
            "image_size": self.image_size,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "num_steps": self.num_steps,
            "enable_depth_reasoning": self.enable_depth_reasoning,
            "enable_cuda_graph": self.enable_cuda_graph,
            "config_sha256": self.config_sha256,
            "norm_stats_sha256": self.norm_stats_sha256,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "checkpoint_file_count": self.checkpoint_file_count,
            "checkpoint_total_bytes": self.checkpoint_total_bytes,
            "norm_stats_format": self.norm_stats_format,
            "norm_mode": self.norm_mode,
            "control_mode": self.control_mode,
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> MolmoAct2ServerIdentity:
        if not isinstance(value, Mapping):
            raise TypeError("server identity must be a mapping")
        return cls(
            checkpoint_id=str(value["checkpoint_id"]),
            checkpoint_revision=str(value["checkpoint_revision"]),
            upstream_code_revision=str(value["upstream_code_revision"]),
            dtype=str(value["dtype"]),
            device=str(value["device"]),
            norm_tag=str(value["norm_tag"]),
            inference_action_mode=str(value["inference_action_mode"]),
            normalize_language=bool(value["normalize_language"]),
            camera_order=tuple(value["camera_order"]),  # type: ignore[arg-type]
            image_size=int(value["image_size"]),
            state_dim=int(value["state_dim"]),
            action_dim=int(value["action_dim"]),
            action_horizon=int(value["action_horizon"]),
            num_steps=int(value["num_steps"]),
            enable_depth_reasoning=bool(value["enable_depth_reasoning"]),
            enable_cuda_graph=bool(value["enable_cuda_graph"]),
            config_sha256=str(value["config_sha256"]),
            norm_stats_sha256=str(value["norm_stats_sha256"]),
            checkpoint_manifest_sha256=str(value["checkpoint_manifest_sha256"]),
            checkpoint_file_count=int(value["checkpoint_file_count"]),
            checkpoint_total_bytes=int(value["checkpoint_total_bytes"]),
            norm_stats_format=str(value["norm_stats_format"]),
            norm_mode=str(value["norm_mode"]),
            control_mode=str(value["control_mode"]),
        )


@dataclasses.dataclass(frozen=True)
class MolmoAct2ActionChunk:
    request_id: str
    server_identity_sha256: str
    actions: tuple[tuple[float, ...], ...]
    latency_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_id", _require_sha256(self.request_id, name="request_id")
        )
        object.__setattr__(
            self,
            "server_identity_sha256",
            _require_sha256(self.server_identity_sha256, name="server_identity_sha256"),
        )
        rows = tuple(tuple(float(item) for item in row) for row in self.actions)
        if len(rows) != 10:
            raise ValueError("frozen MolmoAct2-LIBERO must return exactly 10 actions")
        if any(len(row) != 7 for row in rows):
            raise ValueError("each MolmoAct2-LIBERO action must be 7-D")
        if any(not math.isfinite(item) for row in rows for item in row):
            raise ValueError("action chunk contains non-finite values")
        object.__setattr__(self, "actions", rows)
        if not isinstance(self.latency_ms, int) or self.latency_ms < 0:
            raise ValueError("latency_ms must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "server_identity_sha256": self.server_identity_sha256,
            "actions": [list(row) for row in self.actions],
            "latency_ms": self.latency_ms,
        }


def molmoact2_public_request_payload(
    *,
    agentview_rgb_sha256: str,
    wrist_rgb_sha256: str,
    state: Sequence[float],
    instruction: str,
    session_id: str,
    seed: int,
    expected_identity: MolmoAct2ServerIdentity,
) -> dict[str, object]:
    """Build the exact hash-bearing request without image transport bytes."""

    agent_digest = _require_sha256(agentview_rgb_sha256, name="agentview_rgb_sha256")
    wrist_digest = _require_sha256(wrist_rgb_sha256, name="wrist_rgb_sha256")
    values = _finite_vector(state, length=8, name="MolmoAct2-LIBERO state")
    task = _clean_text(instruction, name="instruction")
    session = _clean_text(session_id, name="session_id")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return {
        "schema_version": MOLMOACT2_HTTP_SCHEMA,
        "session_id": session,
        "seed": seed,
        "instruction": task,
        "agentview_rgb_sha256": agent_digest,
        "wrist_rgb_sha256": wrist_digest,
        "state": list(values),
        "norm_tag": expected_identity.norm_tag,
        "inference_action_mode": expected_identity.inference_action_mode,
        "enable_depth_reasoning": expected_identity.enable_depth_reasoning,
        "num_steps": expected_identity.num_steps,
        "normalize_language": expected_identity.normalize_language,
        "enable_cuda_graph": expected_identity.enable_cuda_graph,
        "expected_server_identity_sha256": expected_identity.digest,
    }


class MolmoAct2HTTPClient:
    """Fail-closed client for the repository's pinned GPU server."""

    def __init__(
        self,
        base_url: str,
        *,
        expected_identity: MolmoAct2ServerIdentity,
        timeout_seconds: float = 180.0,
    ) -> None:
        value = str(base_url).strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if not isinstance(expected_identity, MolmoAct2ServerIdentity):
            raise TypeError("expected_identity must be MolmoAct2ServerIdentity")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = value
        self.expected_identity = expected_identity
        self.timeout_seconds = float(timeout_seconds)

    def _exchange(
        self,
        path: str,
        *,
        payload: Mapping[str, object] | None = None,
        method: str = "POST",
    ) -> dict[str, object]:
        body = None
        headers: dict[str, str] = {}
        if payload is not None:
            body = json.dumps(
                dict(payload),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=body, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as reply:
                raw = reply.read()
        except urllib.error.HTTPError as error:
            detail = error.read()
            try:
                failure = json.loads(detail.decode("utf-8")) if detail else {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                failure = {}
            message = str(failure.get("error", f"HTTP {error.code}"))
            kind = failure.get("error_kind")
            if error.code == 422 and kind == "policy_output":
                raise MolmoAct2PolicyOutputError(message) from error
            if error.code == 400 and kind == "protocol":
                raise MolmoAct2ProtocolError(message) from error
            raise MolmoAct2TransportError(message) from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise MolmoAct2TransportError(
                f"MolmoAct2 endpoint request failed: {error}"
            ) from error
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MolmoAct2ProtocolError(
                "MolmoAct2 endpoint returned invalid JSON"
            ) from error
        if not isinstance(decoded, dict):
            raise MolmoAct2ProtocolError(
                "MolmoAct2 endpoint response must be an object"
            )
        return decoded

    def health(self) -> dict[str, object]:
        response = self._exchange("/health", method="GET")
        if response.get("status") != "ok":
            raise MolmoAct2ProtocolError("MolmoAct2 endpoint is not healthy")
        observed = MolmoAct2ServerIdentity.from_mapping(response["identity"])  # type: ignore[arg-type]
        if observed.digest != self.expected_identity.digest:
            raise MolmoAct2ProtocolError(
                "MolmoAct2 server identity mismatch: "
                f"expected {self.expected_identity.digest}, observed {observed.digest}"
            )
        if response.get("identity_sha256") != observed.digest:
            raise MolmoAct2ProtocolError(
                "MolmoAct2 health identity digest is inconsistent"
            )
        runtime_identity = response.get("runtime_identity")
        if not isinstance(runtime_identity, dict) or not runtime_identity:
            raise MolmoAct2ProtocolError(
                "MolmoAct2 health response lacks runtime identity"
            )
        if runtime_identity.get("snapshot_revision") != observed.checkpoint_revision:
            raise MolmoAct2ProtocolError(
                "MolmoAct2 runtime snapshot revision is inconsistent"
            )
        return response

    def reset(self, session_id: str) -> None:
        self.health()
        response = self._exchange(
            "/reset",
            payload={"schema_version": MOLMOACT2_HTTP_SCHEMA, "session_id": session_id},
        )
        if response.get("status") != "ok" or response.get("session_id") != session_id:
            raise MolmoAct2ProtocolError(
                "MolmoAct2 endpoint reset acknowledgement is invalid"
            )

    def predict_action_chunk(
        self,
        *,
        agentview_rgb: Any,
        wrist_rgb: Any,
        state: Sequence[float],
        instruction: str,
        session_id: str,
        seed: int,
    ) -> MolmoAct2ActionChunk:
        agentview = canonical_rgb_array(agentview_rgb)
        wrist = canonical_rgb_array(wrist_rgb)
        expected_shape = (
            self.expected_identity.image_size,
            self.expected_identity.image_size,
            3,
        )
        if agentview.shape != expected_shape or wrist.shape != expected_shape:
            raise ValueError(
                f"MolmoAct2-LIBERO RGB inputs must have shape {expected_shape}"
            )
        public_request = molmoact2_public_request_payload(
            agentview_rgb_sha256=canonical_rgb_sha256(agentview),
            wrist_rgb_sha256=canonical_rgb_sha256(wrist),
            state=state,
            instruction=instruction,
            session_id=session_id,
            seed=seed,
            expected_identity=self.expected_identity,
        )
        request_id = canonical_sha256(public_request)
        transport = dict(public_request)
        transport.update(
            {
                "request_id": request_id,
                "agentview_png_base64": base64.b64encode(
                    encode_rgb_png(agentview)
                ).decode("ascii"),
                "wrist_png_base64": base64.b64encode(encode_rgb_png(wrist)).decode(
                    "ascii"
                ),
            }
        )
        response = self._exchange("/act", payload=transport)
        if response.get("request_id") != request_id:
            raise MolmoAct2ProtocolError(
                "MolmoAct2 response does not match request identity"
            )
        if response.get("server_identity_sha256") != self.expected_identity.digest:
            raise MolmoAct2ProtocolError(
                "MolmoAct2 response came from an unexpected server identity"
            )
        raw_actions = response.get("actions")
        if not isinstance(raw_actions, list):
            raise MolmoAct2PolicyOutputError("MolmoAct2 response has no action chunk")
        try:
            return MolmoAct2ActionChunk(
                request_id=request_id,
                server_identity_sha256=self.expected_identity.digest,
                actions=tuple(tuple(row) for row in raw_actions),  # type: ignore[arg-type]
                latency_ms=int(response.get("latency_ms", 0)),
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise MolmoAct2PolicyOutputError(
                f"invalid MolmoAct2 action chunk: {error}"
            ) from error


class _ActionBackend(Protocol):
    identity: MolmoAct2ServerIdentity
    runtime_identity: Mapping[str, object]

    def predict(
        self,
        *,
        agentview_rgb: Any,
        wrist_rgb: Any,
        state: Sequence[float],
        instruction: str,
        seed: int,
    ) -> Sequence[Sequence[float]]: ...

    def reset(self, session_id: str) -> None: ...


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {name}: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _stats_dimension(value: object, *, name: str) -> int:
    if not isinstance(value, dict):
        raise TypeError(f"MolmoAct2 {name} must be an object")
    dimensions: set[int] = set()
    for key in (
        "min",
        "max",
        "mean",
        "std",
        "q01",
        "q10",
        "q50",
        "q90",
        "q99",
        "names",
        "mask",
    ):
        child = value.get(key)
        if not isinstance(child, list) or not child:
            raise RuntimeError(f"MolmoAct2 {name}.{key} must be a non-empty list")
        dimensions.add(len(child))
    if len(dimensions) != 1:
        raise RuntimeError(f"MolmoAct2 {name} vector dimensions disagree")
    return dimensions.pop()


def inspect_molmoact2_snapshot(
    local_dir: str | Path, *, norm_tag: str
) -> dict[str, object]:
    """Hash and semantically validate the complete resolved checkpoint snapshot."""

    root = Path(local_dir).expanduser().resolve()
    config_path = root / "config.json"
    norm_stats_path = root / "norm_stats.json"
    if not config_path.is_file() or not norm_stats_path.is_file():
        raise RuntimeError(
            "MolmoAct2 snapshot is missing config.json or norm_stats.json"
        )
    config = _json_object(config_path, name="MolmoAct2 config")
    norm_stats = _json_object(norm_stats_path, name="MolmoAct2 norm stats")
    metadata_by_tag = norm_stats.get("metadata_by_tag")
    if not isinstance(metadata_by_tag, dict):
        raise TypeError("MolmoAct2 norm stats lack metadata_by_tag")
    metadata = metadata_by_tag.get(norm_tag)
    if not isinstance(metadata, dict):
        raise TypeError(f"MolmoAct2 norm stats lack tag {norm_tag!r}")
    state_dim = _stats_dimension(metadata.get("state_stats"), name="state_stats")
    action_dim = _stats_dimension(metadata.get("action_stats"), name="action_stats")
    semantic = {
        "norm_stats_format": str(norm_stats.get("format", "")),
        "norm_mode": str(norm_stats.get("norm_mode", "")),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_horizon": int(metadata.get("action_horizon", 0)),
        "n_action_steps": int(metadata.get("n_action_steps", 0)),
        "control_mode": str(metadata.get("control_mode", "")),
        "camera_count": len(metadata.get("camera_keys", ())),
        "normalize_gripper": metadata.get("normalize_gripper"),
        "config_max_action_horizon": int(config.get("max_action_horizon", 0)),
    }
    expected = {
        "norm_stats_format": MOLMOACT2_NORM_STATS_FORMAT,
        "norm_mode": MOLMOACT2_NORM_MODE,
        "state_dim": 8,
        "action_dim": 7,
        "action_horizon": 10,
        "n_action_steps": 10,
        "control_mode": MOLMOACT2_CONTROL_MODE,
        "camera_count": 2,
        "normalize_gripper": False,
        "config_max_action_horizon": 10,
    }
    if semantic != expected:
        raise RuntimeError(
            f"MolmoAct2-LIBERO snapshot semantics mismatch: expected {expected}, observed {semantic}"
        )
    files: list[dict[str, object]] = []
    for relative in _MOLMOACT2_INFERENCE_FILES:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(
                f"MolmoAct2 snapshot is missing required file {relative}"
            )
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not files:
        raise RuntimeError("MolmoAct2 snapshot contains no files")
    manifest = {
        "schema_version": "molmoact2-inference-snapshot-manifest-v1",
        "files": files,
    }
    return {
        "config_sha256": _file_sha256(config_path),
        "norm_stats_sha256": _file_sha256(norm_stats_path),
        "checkpoint_manifest_sha256": canonical_sha256(manifest),
        "checkpoint_file_count": len(files),
        "checkpoint_total_bytes": sum(int(item["size"]) for item in files),
        **semantic,
    }


def _assert_snapshot_matches_identity(
    observed: Mapping[str, object], identity: MolmoAct2ServerIdentity
) -> None:
    expected = {
        "config_sha256": identity.config_sha256,
        "norm_stats_sha256": identity.norm_stats_sha256,
        "checkpoint_manifest_sha256": identity.checkpoint_manifest_sha256,
        "checkpoint_file_count": identity.checkpoint_file_count,
        "checkpoint_total_bytes": identity.checkpoint_total_bytes,
        "norm_stats_format": identity.norm_stats_format,
        "norm_mode": identity.norm_mode,
        "state_dim": identity.state_dim,
        "action_dim": identity.action_dim,
        "action_horizon": identity.action_horizon,
        "n_action_steps": identity.num_steps,
        "control_mode": identity.control_mode,
    }
    mismatches = [key for key, value in expected.items() if observed.get(key) != value]
    if mismatches:
        raise RuntimeError(
            "resolved MolmoAct2 snapshot does not match frozen identity: "
            + ", ".join(mismatches)
        )


class TransformersMolmoAct2Backend:
    """Direct wrapper around the official Hugging Face ``predict_action`` API."""

    def __init__(self, identity: MolmoAct2ServerIdentity) -> None:
        if not isinstance(identity, MolmoAct2ServerIdentity):
            raise TypeError("identity must be MolmoAct2ServerIdentity")
        try:
            import numpy as np
            import torch
            import transformers
            from huggingface_hub import snapshot_download
            from PIL import Image
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:  # pragma: no cover - requires GPU extra.
            raise RuntimeError(
                "GPU server requires torch, transformers, numpy, and Pillow"
            ) from error
        if identity.dtype == "bfloat16" and not identity.device.startswith("cuda"):
            raise ValueError("the frozen BF16 server contract requires a CUDA device")
        self.identity = identity
        self._np = np
        self._torch = torch
        self._Image = Image
        dtype = torch.bfloat16 if identity.dtype == "bfloat16" else torch.float32
        local_dir = Path(
            snapshot_download(
                repo_id=identity.checkpoint_id,
                revision=identity.checkpoint_revision,
            )
        ).resolve()
        if local_dir.name != identity.checkpoint_revision:
            raise RuntimeError(
                "resolved Hugging Face snapshot does not match the frozen revision"
            )
        snapshot_identity = inspect_molmoact2_snapshot(
            local_dir, norm_tag=identity.norm_tag
        )
        _assert_snapshot_matches_identity(snapshot_identity, identity)
        self.snapshot_dir = local_dir
        self.processor = AutoProcessor.from_pretrained(
            str(local_dir),
            trust_remote_code=True,
            extra_special_tokens={},
        )
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                str(local_dir),
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            .to(identity.device)
            .eval()
        )
        model_parameter = next(self.model.parameters())
        model_dtype = str(model_parameter.dtype).removeprefix("torch.")
        model_device = str(model_parameter.device)
        if model_dtype != identity.dtype:
            raise RuntimeError(
                "loaded MolmoAct2 dtype does not match the frozen server identity"
            )
        if model_device != identity.device:
            raise RuntimeError(
                "loaded MolmoAct2 device does not match the frozen server identity"
            )
        self.runtime_identity = {
            "backend_kind": MOLMOACT2_LIVE_BACKEND_KIND,
            "checkpoint_id": identity.checkpoint_id,
            "checkpoint_revision": identity.checkpoint_revision,
            "upstream_code_revision": identity.upstream_code_revision,
            "model_class": type(self.model).__name__,
            "model_dtype": model_dtype,
            "model_device": model_device,
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "device_name": str(torch.cuda.get_device_name(identity.device)),
            "snapshot_revision": local_dir.name,
            **snapshot_identity,
            "adapter_source_sha256": molmoact2_adapter_sha256(),
        }

        # This compatibility hook is copied from the pinned upstream inference
        # server.  The remote model helper moves tensors but does not cast its
        # float image tensors to the BF16 model dtype.
        target_dtype = next(self.model.parameters()).dtype

        def _move_and_cast(
            inputs: Any,
            device: Any,
            _target: Any = target_dtype,
        ) -> dict[str, Any]:
            converted: dict[str, Any] = {}
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    value = value.to(device)
                    if value.is_floating_point() and value.dtype != _target:
                        value = value.to(_target)
                converted[key] = value
            return converted

        self.model._move_inputs_to_device = _move_and_cast
        self._lock = threading.Lock()

    def reset(self, session_id: str) -> None:
        _clean_text(session_id, name="session_id")

    def predict(
        self,
        *,
        agentview_rgb: Any,
        wrist_rgb: Any,
        state: Sequence[float],
        instruction: str,
        seed: int,
    ) -> Sequence[Sequence[float]]:
        torch = self._torch
        np = self._np
        images = [
            self._Image.fromarray(canonical_rgb_array(agentview_rgb), mode="RGB"),
            self._Image.fromarray(canonical_rgb_array(wrist_rgb), mode="RGB"),
        ]
        robot_state = np.asarray(
            _finite_vector(state, length=8, name="MolmoAct2-LIBERO state"),
            dtype=np.float32,
        )
        with self._lock:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            generator = torch.Generator(device=self.identity.device).manual_seed(seed)
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if self.identity.dtype == "bfloat16"
                else torch.autocast("cuda", enabled=False)
            )
            with torch.inference_mode(), autocast:
                output = self.model.predict_action(
                    processor=self.processor,
                    images=images,
                    task=instruction,
                    state=robot_state,
                    norm_tag=self.identity.norm_tag,
                    inference_action_mode=self.identity.inference_action_mode,
                    enable_depth_reasoning=False,
                    num_steps=self.identity.num_steps,
                    generator=generator,
                    normalize_language=self.identity.normalize_language,
                    enable_cuda_graph=self.identity.enable_cuda_graph,
                )
        actions = output.actions.detach().float().cpu().numpy()
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise RuntimeError("MolmoAct2 server supports batch size one only")
            actions = actions[0]
        return actions.tolist()


class _MolmoAct2RequestHandler(BaseHTTPRequestHandler):
    server_version = "MolmoAct2PinnedHTTP/1"

    @property
    def backend(self) -> _ActionBackend:
        return self.server.backend  # type: ignore[attr-defined, no-any-return]

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send(self, status: int, payload: Mapping[str, object]) -> None:
        encoded = json.dumps(
            dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 2 or length > 20_000_000:
            raise ValueError("invalid request body length")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise TypeError("request body must be a JSON object")
        return value

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send(404, {"error": "not found"})
            return
        identity = self.backend.identity
        self._send(
            200,
            {
                "schema_version": MOLMOACT2_HTTP_SCHEMA,
                "status": "ok",
                "identity": identity.to_dict(),
                "identity_sha256": identity.digest,
                "runtime_identity": dict(self.backend.runtime_identity),
            },
        )

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if payload.get("schema_version") != MOLMOACT2_HTTP_SCHEMA:
                raise ValueError("unsupported schema_version")
            if self.path == "/reset":
                session_id = _clean_text(payload.get("session_id"), name="session_id")
                self.backend.reset(session_id)
                self._send(200, {"status": "ok", "session_id": session_id})
                return
            if self.path != "/act":
                self._send(404, {"error": "not found"})
                return
            identity = self.backend.identity
            if payload.get("expected_server_identity_sha256") != identity.digest:
                raise ValueError("client expected a different server identity")
            public_request = {
                key: payload[key]
                for key in (
                    "schema_version",
                    "session_id",
                    "seed",
                    "instruction",
                    "agentview_rgb_sha256",
                    "wrist_rgb_sha256",
                    "state",
                    "norm_tag",
                    "inference_action_mode",
                    "enable_depth_reasoning",
                    "num_steps",
                    "normalize_language",
                    "enable_cuda_graph",
                    "expected_server_identity_sha256",
                )
            }
            request_id = canonical_sha256(public_request)
            if payload.get("request_id") != request_id:
                raise ValueError("request_id does not match public payload")
            if (
                payload.get("norm_tag") != "libero"
                or payload.get("inference_action_mode")
                != identity.inference_action_mode
                or payload.get("enable_depth_reasoning") is not False
                or payload.get("num_steps") != identity.num_steps
                or payload.get("normalize_language") is not identity.normalize_language
                or payload.get("enable_cuda_graph") is not identity.enable_cuda_graph
            ):
                raise ValueError("request does not match frozen inference settings")
            agentview = decode_rgb_png(
                base64.b64decode(str(payload["agentview_png_base64"]), validate=True)
            )
            wrist = decode_rgb_png(
                base64.b64decode(str(payload["wrist_png_base64"]), validate=True)
            )
            if canonical_rgb_sha256(agentview) != payload["agentview_rgb_sha256"]:
                raise ValueError("agentview digest mismatch")
            if canonical_rgb_sha256(wrist) != payload["wrist_rgb_sha256"]:
                raise ValueError("wrist digest mismatch")
            expected_shape = (identity.image_size, identity.image_size, 3)
            if agentview.shape != expected_shape or wrist.shape != expected_shape:
                raise ValueError(
                    f"MolmoAct2-LIBERO RGB inputs must have shape {expected_shape}"
                )
            state = _finite_vector(
                payload["state"],  # type: ignore[arg-type]
                length=identity.state_dim,
                name="MolmoAct2-LIBERO state",
            )
            instruction = _clean_text(payload["instruction"], name="instruction")
            raw_seed = payload["seed"]
            if (
                not isinstance(raw_seed, int)
                or isinstance(raw_seed, bool)
                or raw_seed < 0
            ):
                raise ValueError("seed must be a non-negative integer")
            seed = raw_seed
            start = time.monotonic()
            try:
                actions = self.backend.predict(
                    agentview_rgb=agentview,
                    wrist_rgb=wrist,
                    state=state,
                    instruction=instruction,
                    seed=seed,
                )
            except Exception as error:  # noqa: BLE001 - process boundary
                self._send(
                    500,
                    {
                        "error_kind": "backend_inference",
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
                return
            try:
                chunk = MolmoAct2ActionChunk(
                    request_id=request_id,
                    server_identity_sha256=identity.digest,
                    actions=tuple(tuple(row) for row in actions),
                    latency_ms=int((time.monotonic() - start) * 1000),
                )
            except Exception as error:  # noqa: BLE001 - process boundary
                self._send(
                    422,
                    {
                        "error_kind": "policy_output",
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
                return
            self._send(200, chunk.to_dict())
        except Exception as error:  # noqa: BLE001 - process boundary
            self._send(
                400,
                {
                    "error_kind": "protocol",
                    "error": f"{type(error).__name__}: {error}",
                },
            )


def make_molmoact2_http_server(
    backend: _ActionBackend,
    *,
    host: str,
    port: int,
) -> ThreadingHTTPServer:
    """Construct a server; exposed separately so tests can use a faithful double."""

    server = ThreadingHTTPServer((host, port), _MolmoAct2RequestHandler)
    server.backend = backend  # type: ignore[attr-defined]
    return server


def serve_main() -> None:
    parser = argparse.ArgumentParser(description="Serve pinned MolmoAct2-LIBERO")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--checkpoint", default=MOLMOACT2_REPO_ID)
    parser.add_argument("--revision", default=MOLMOACT2_MODEL_REVISION)
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="download/hash the snapshot and exit without loading the model",
    )
    parser.add_argument(
        "--identity-json",
        help="frozen E1 plan JSON containing the exact executor identity",
    )
    args = parser.parse_args()
    if args.inspect_only:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError(
                "snapshot inspection requires huggingface_hub"
            ) from error
        local_dir = Path(
            snapshot_download(repo_id=args.checkpoint, revision=args.revision)
        ).resolve()
        report = {
            "checkpoint_id": args.checkpoint,
            "checkpoint_revision": args.revision,
            "snapshot_dir": str(local_dir),
            **inspect_molmoact2_snapshot(local_dir, norm_tag="libero"),
        }
        print(json.dumps(report, sort_keys=True), flush=True)
        return
    if not args.identity_json:
        parser.error("serving requires --identity-json from a frozen E1 plan")
    identity_payload = _json_object(
        Path(args.identity_json).expanduser().resolve(), name="E1 identity plan"
    )
    executor = identity_payload.get("executor")
    if not isinstance(executor, dict):
        parser.error("--identity-json must contain an executor object")
    identity = MolmoAct2ServerIdentity.from_mapping(executor)
    if (
        identity.checkpoint_id != args.checkpoint
        or identity.checkpoint_revision != args.revision
    ):
        parser.error("checkpoint/revision do not match --identity-json")
    backend = TransformersMolmoAct2Backend(identity)
    server = make_molmoact2_http_server(backend, host=args.host, port=args.port)
    print(
        json.dumps(
            {
                "status": "ready",
                "url": f"http://{args.host}:{args.port}",
                "identity_sha256": identity.digest,
                "identity": identity.to_dict(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":  # pragma: no cover
    serve_main()
