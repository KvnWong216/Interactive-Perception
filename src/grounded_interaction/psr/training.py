"""Training-stage contracts and checkpointing for PSR-VLA V1.

The execution snapshot and the outcome predictor deliberately have separate
lifecycles.  Stage A adapts the latent execution path with *real* action,
intent, and future-observation supervision.  Stage B freezes that complete
execution path as ``S0``.  Stage C may then train only parameters disjoint from
S0, using outcomes actually collected under that exact snapshot.  Stage D
fits one positive scalar temperature on a separate calibration split.

This module contains no synthetic data generator and no success-rate
defaults.  Callers must supply tensors produced from real records and the
in-process Molmo backend.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import math
import os
import random
import re
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from grounded_interaction.contracts import canonical_sha256

from .model import (
    MaskedLoss,
    PredictiveOutput,
    evidence_mixture_nll,
    failure_bce_loss,
)
from .types import IntentCandidate

try:  # Keep CPU/config-only repository imports possible without the ML extra.
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - exercised in torch-free installs.
    torch = None
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


CHECKPOINT_SCHEMA = "psr-v1-checkpoint-v1"
SNAPSHOT_SCHEMA = "psr-v1-execution-snapshot-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_torch() -> Any:
    if torch is None or nn is None:
        raise RuntimeError("PSR training requires the PyTorch learned extra")
    return torch


def _require_digest(value: str, name: str) -> str:
    value = str(value)
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _text(value: Any, name: str) -> str:
    result = " ".join(str(value or "").split())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _json_plain(value: Any, *, name: str) -> Any:
    """Validate a JSON-like value by asking the canonical serializer to hash it."""

    try:
        canonical_sha256(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be finite JSON-compatible data") from error
    if isinstance(value, Mapping):
        return {str(key): _json_plain(child, name=name) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_json_plain(child, name=name) for child in value]
    if isinstance(value, list):
        return [_json_plain(child, name=name) for child in value]
    return value


def _model_owner(value: Any) -> Any:
    candidate = getattr(value, "model", value)
    if not callable(getattr(candidate, "named_parameters", None)):
        raise TypeError("object does not expose torch named_parameters()")
    return candidate


def _named_parameters(value: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(_model_owner(value).named_parameters())


def _tensor_digest(tensor: Any) -> str:
    library = _require_torch()
    if not isinstance(tensor, library.Tensor):
        raise TypeError("checkpoint state contains a non-tensor parameter")
    if tensor.is_meta:
        raise ValueError("cannot fingerprint meta tensors")
    value = tensor.detach().contiguous().cpu()
    hasher = hashlib.sha256()
    hasher.update(str(value.dtype).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(str(tuple(value.shape)).encode("utf-8"))
    hasher.update(b"\0")
    # NumPy does not support every torch dtype (notably bfloat16).  Viewing the
    # contiguous storage as bytes preserves the exact serialized values.
    hasher.update(value.view(library.uint8).numpy().tobytes())
    return hasher.hexdigest()


def _update_object_hash(hasher: Any, value: Any) -> None:
    library = _require_torch()
    if isinstance(value, library.Tensor):
        hasher.update(b"tensor:")
        hasher.update(_tensor_digest(value).encode("ascii"))
        return
    if isinstance(value, Mapping):
        hasher.update(b"mapping{")
        for key in sorted(value, key=lambda item: str(item)):
            _update_object_hash(hasher, str(key))
            _update_object_hash(hasher, value[key])
        hasher.update(b"}")
        return
    if isinstance(value, (tuple, list)):
        hasher.update(b"sequence[")
        for child in value:
            _update_object_hash(hasher, child)
        hasher.update(b"]")
        return
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("checkpoint state contains a non-finite scalar")
        hasher.update(repr(value).encode("utf-8"))
        return
    raise TypeError(f"unsupported checkpoint value {type(value).__name__}")


def object_sha256(value: Any) -> str:
    """Hash nested optimizer/RNG state without relying on pickle byte order."""

    hasher = hashlib.sha256()
    _update_object_hash(hasher, value)
    return hasher.hexdigest()


def parameter_sha256(value: Any) -> str:
    """Fingerprint every named parameter of an execution or predictor module."""

    hasher = hashlib.sha256()
    parameters = _named_parameters(value)
    if not parameters:
        raise ValueError("cannot snapshot a module with no parameters")
    for name, parameter in sorted(parameters):
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(_tensor_digest(parameter).encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def module_state_sha256(value: Any) -> str:
    """Fingerprint parameters and persistent buffers in a module state dict."""

    state = _model_owner(value).state_dict()
    if not state:
        raise ValueError("cannot snapshot an empty module state")
    return object_sha256(state)


@dataclasses.dataclass(frozen=True)
class CheckpointIdentity:
    """Version identities that prevent relabeling results across protocols."""

    base_model_id: str
    base_revision: str
    upstream_revision: str
    config_sha256: str
    split_sha256: str
    protocol_sha256: str
    projection_sha256: str
    normalization_sha256: str
    seed: int
    execution_snapshot_id: str | None = None
    extra: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("base_model_id", "base_revision", "upstream_revision"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "config_sha256",
            "split_sha256",
            "protocol_sha256",
            "projection_sha256",
            "normalization_sha256",
        ):
            object.__setattr__(self, name, _require_digest(getattr(self, name), name))
        if self.execution_snapshot_id is not None:
            object.__setattr__(
                self,
                "execution_snapshot_id",
                _require_digest(self.execution_snapshot_id, "execution_snapshot_id"),
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        object.__setattr__(self, "extra", _json_plain(self.extra, name="extra"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model_id": self.base_model_id,
            "base_revision": self.base_revision,
            "upstream_revision": self.upstream_revision,
            "config_sha256": self.config_sha256,
            "split_sha256": self.split_sha256,
            "protocol_sha256": self.protocol_sha256,
            "projection_sha256": self.projection_sha256,
            "normalization_sha256": self.normalization_sha256,
            "seed": self.seed,
            "execution_snapshot_id": self.execution_snapshot_id,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CheckpointIdentity:
        expected = {field.name for field in dataclasses.fields(cls)}
        if set(value) != expected:
            raise ValueError("checkpoint identity fields are incomplete or unknown")
        return cls(**dict(value))


@dataclasses.dataclass(frozen=True)
class FrozenExecutionSnapshot:
    """Stage-B S0 identity for the complete physical execution model."""

    schema_version: str
    snapshot_id: str
    parameter_sha256: str
    state_sha256: str
    parameter_count: int
    parameter_names: tuple[str, ...]
    trainable_before_freeze: tuple[str, ...]
    identity: CheckpointIdentity

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "parameter_sha256": self.parameter_sha256,
            "state_sha256": self.state_sha256,
            "parameter_count": self.parameter_count,
            "parameter_names": list(self.parameter_names),
            "trainable_before_freeze": list(self.trainable_before_freeze),
            "identity": self.identity.to_dict(),
        }


def freeze_execution_snapshot(
    execution_model: Any, *, identity: CheckpointIdentity
) -> FrozenExecutionSnapshot:
    """Freeze all S0 parameters and bind their values to the execution protocol."""

    parameters = _named_parameters(execution_model)
    digest = parameter_sha256(execution_model)
    state_digest = module_state_sha256(execution_model)
    names = tuple(name for name, _ in parameters)
    trainable = tuple(name for name, parameter in parameters if parameter.requires_grad)
    payload = {
        "schema_version": SNAPSHOT_SCHEMA,
        "parameter_sha256": digest,
        "state_sha256": state_digest,
        "parameter_names": names,
        "trainable_before_freeze": trainable,
        "identity": identity.to_dict(),
    }
    snapshot = FrozenExecutionSnapshot(
        schema_version=SNAPSHOT_SCHEMA,
        snapshot_id=canonical_sha256(payload),
        parameter_sha256=digest,
        state_sha256=state_digest,
        parameter_count=sum(parameter.numel() for _, parameter in parameters),
        parameter_names=names,
        trainable_before_freeze=trainable,
        identity=identity,
    )
    for _, parameter in parameters:
        parameter.requires_grad_(False)
        parameter.grad = None
    return snapshot


def assert_execution_snapshot_unchanged(
    execution_model: Any, snapshot: FrozenExecutionSnapshot
) -> None:
    """Fail closed if Stage C changed S0 or swapped its parameter structure."""

    if snapshot.schema_version != SNAPSHOT_SCHEMA:
        raise ValueError("unsupported S0 snapshot schema")
    parameters = _named_parameters(execution_model)
    names = tuple(name for name, _ in parameters)
    if names != snapshot.parameter_names:
        raise RuntimeError("S0 parameter structure changed after snapshot freeze")
    if any(parameter.requires_grad for _, parameter in parameters):
        raise RuntimeError("S0 contains trainable parameters during Stage C")
    observed_parameters = parameter_sha256(execution_model)
    observed_state = module_state_sha256(execution_model)
    if (
        observed_parameters != snapshot.parameter_sha256
        or observed_state != snapshot.state_sha256
    ):
        raise RuntimeError("S0 parameter or buffer values changed during Stage C")


def _optimizer_parameters(optimizer: Any) -> tuple[Any, ...]:
    parameters: list[Any] = []
    for group in getattr(optimizer, "param_groups", ()):
        parameters.extend(group.get("params", ()))
    if not parameters:
        raise ValueError("optimizer has no parameters")
    return tuple(parameters)


def assert_disjoint_stage_c_parameters(execution_model: Any, optimizer: Any) -> None:
    """Assert Stage-C optimizer owns no parameter from frozen S0."""

    s0_ids = {id(parameter) for _, parameter in _named_parameters(execution_model)}
    overlap = [
        parameter
        for parameter in _optimizer_parameters(optimizer)
        if id(parameter) in s0_ids
    ]
    if overlap:
        raise RuntimeError("Stage-C optimizer overlaps frozen S0 parameters")


class FrozenExecutionGuard(AbstractContextManager["FrozenExecutionGuard"]):
    """Verify S0 immediately before and after a Stage-C training interval."""

    def __init__(
        self,
        execution_model: Any,
        snapshot: FrozenExecutionSnapshot,
        optimizer: Any,
    ) -> None:
        self.execution_model = execution_model
        self.snapshot = snapshot
        self.optimizer = optimizer

    def __enter__(self) -> Self:
        assert_execution_snapshot_unchanged(self.execution_model, self.snapshot)
        assert_disjoint_stage_c_parameters(self.execution_model, self.optimizer)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        assert_execution_snapshot_unchanged(self.execution_model, self.snapshot)


@dataclasses.dataclass(frozen=True)
class StageALoss:
    """The three equally weighted, normalized Stage-A objectives."""

    total: Tensor
    native_flow: Tensor
    future_evidence_nll: Tensor
    intent_ce: Tensor
    evidence_items: int


def _require_finite_scalar_tensor(value: Any, name: str) -> Any:
    library = _require_torch()
    if not isinstance(value, library.Tensor) or value.numel() != 1:
        raise TypeError(f"{name} must return one scalar tensor")
    if not bool(library.isfinite(value.detach()).all()):
        raise ValueError(f"{name} is non-finite")
    return value.reshape(())


def compose_stage_a_loss(
    *,
    backend: Any,
    encoded_history: Any,
    executed_intent: IntentCandidate,
    actions: Any,
    intent_target_token_ids: Sequence[int],
    prediction: PredictiveOutput,
    evidence_targets: Tensor,
    evidence_valid_dimension_mask: Tensor,
    evidence_supervised_candidates: Tensor | None = None,
    flow_timesteps: Any | None = None,
    flow_noise: Any | None = None,
) -> StageALoss:
    """Compose ``flow + future-E NLL + intent CE`` from real backend paths.

    ``backend.flow_matching_loss`` must be the differentiable Action Expert
    path, not ``predict_action`` or an HTTP/no-grad inference endpoint.
    """

    if not isinstance(executed_intent, IntentCandidate):
        raise TypeError("executed_intent must be an IntentCandidate")
    if executed_intent.execution_route != "conditioned":
        raise ValueError("Stage-A flow supervision requires a conditioned intent")
    token_ids = tuple(int(item) for item in intent_target_token_ids)
    if not token_ids:
        raise ValueError("intent CE requires at least one public target token")
    flow = _require_finite_scalar_tensor(
        backend.flow_matching_loss(
            encoded_history,
            executed_intent,
            actions,
            timesteps=flow_timesteps,
            noise=flow_noise,
        ),
        "native flow loss",
    )
    intent = _require_finite_scalar_tensor(
        backend.intent_language_loss(encoded_history, token_ids),
        "intent language loss",
    )
    evidence = evidence_mixture_nll(
        prediction,
        evidence_targets,
        evidence_valid_dimension_mask,
        supervised_candidates=evidence_supervised_candidates,
    )
    if not evidence.has_supervision:
        raise ValueError("Stage A record has no valid future-evidence target")
    evidence_value = _require_finite_scalar_tensor(
        evidence.require_value(), "future evidence NLL"
    )
    total = flow + evidence_value + intent
    _require_finite_scalar_tensor(total, "Stage-A total loss")
    return StageALoss(
        total=total,
        native_flow=flow,
        future_evidence_nll=evidence_value,
        intent_ce=intent,
        evidence_items=evidence.valid_item_count,
    )


@dataclasses.dataclass(frozen=True)
class StageCBatch:
    """Predictor inputs and separate labels from S0 snapshot rollouts."""

    state_tokens: Tensor
    intent_token_ids: Tensor
    intent_token_mask: Tensor
    route_ids: Tensor
    candidate_valid_mask: Tensor
    evidence_targets: Tensor
    evidence_valid_dimension_mask: Tensor
    evidence_supervised_candidates: Tensor
    failure_targets: Tensor
    failure_supervised_candidates: Tensor

    def forward(self, predictor: Any) -> PredictiveOutput:
        return predictor(
            self.state_tokens,
            self.intent_token_ids,
            self.intent_token_mask,
            self.route_ids,
            self.candidate_valid_mask,
        )


@dataclasses.dataclass(frozen=True)
class StageCLoss:
    total: Tensor | None
    evidence: MaskedLoss
    failure: MaskedLoss

    @property
    def has_supervision(self) -> bool:
        return self.total is not None


def compose_stage_c_loss(
    prediction: PredictiveOutput, batch: StageCBatch
) -> StageCLoss:
    """Compose actual S0 future-E NLL and actual terminal-failure BCE."""

    evidence = evidence_mixture_nll(
        prediction,
        batch.evidence_targets,
        batch.evidence_valid_dimension_mask,
        supervised_candidates=batch.evidence_supervised_candidates,
    )
    failure = failure_bce_loss(
        prediction.failure_logits,
        batch.failure_targets,
        batch.failure_supervised_candidates,
        candidate_valid_mask=batch.candidate_valid_mask,
    )
    values = [item.value for item in (evidence, failure) if item.value is not None]
    total = sum(values[1:], values[0]) if values else None
    if total is not None:
        _require_finite_scalar_tensor(total, "Stage-C total loss")
    return StageCLoss(total=total, evidence=evidence, failure=failure)


@dataclasses.dataclass(frozen=True)
class StageCStepResult:
    updated: bool
    skip_reason: str | None
    total_loss: float | None
    evidence_loss: float | None
    failure_loss: float | None
    evidence_items: int
    failure_items: int
    gradient_norm: float | None


def train_stage_c_step(
    *,
    predictor: Any,
    optimizer: Any,
    batch: StageCBatch,
    execution_model: Any,
    grad_clip_norm: float = 1.0,
) -> StageCStepResult:
    """Perform one Stage-C update, or explicitly skip an unlabeled batch."""

    library = _require_torch()
    limit = _finite_scalar(grad_clip_norm, "grad_clip_norm")
    if limit <= 0:
        raise ValueError("grad_clip_norm must be positive")
    assert_disjoint_stage_c_parameters(execution_model, optimizer)
    if any(
        parameter.requires_grad for _, parameter in _named_parameters(execution_model)
    ):
        raise RuntimeError("S0 must be frozen before Stage-C training")
    optimizer.zero_grad(set_to_none=True)
    prediction = batch.forward(predictor)
    losses = compose_stage_c_loss(prediction, batch)
    if not losses.has_supervision:
        # Do not call optimizer.step(): even a numeric zero would allow weight
        # decay or optimizer momentum to change an unsupervised model.
        return StageCStepResult(
            updated=False,
            skip_reason="no_valid_evidence_or_failure_supervision",
            total_loss=None,
            evidence_loss=None,
            failure_loss=None,
            evidence_items=0,
            failure_items=0,
            gradient_norm=None,
        )
    total = losses.total
    assert total is not None
    total.backward()
    trainable = [
        parameter
        for parameter in _optimizer_parameters(optimizer)
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("Stage-C optimizer contains no trainable parameters")
    gradient_norm = library.nn.utils.clip_grad_norm_(trainable, max_norm=limit)
    if not bool(library.isfinite(gradient_norm)):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("Stage-C gradient norm is non-finite")
    optimizer.step()
    return StageCStepResult(
        updated=True,
        skip_reason=None,
        total_loss=float(total.detach().cpu().item()),
        evidence_loss=(
            None
            if losses.evidence.value is None
            else float(losses.evidence.value.detach().cpu().item())
        ),
        failure_loss=(
            None
            if losses.failure.value is None
            else float(losses.failure.value.detach().cpu().item())
        ),
        evidence_items=losses.evidence.valid_item_count,
        failure_items=losses.failure.valid_item_count,
        gradient_norm=float(gradient_norm.detach().cpu().item()),
    )


@dataclasses.dataclass(frozen=True)
class TemperatureCalibration:
    """Positive held-out temperature; it cannot change candidate ordering."""

    temperature: float
    calibration_sha256: str
    examples: int
    failures: int
    successes: int
    bce_before: float
    bce_after: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("calibration temperature must be positive and finite")
        _require_digest(self.calibration_sha256, "calibration_sha256")

    def apply(self, logits: Any) -> Any:
        return logits / self.temperature


def fit_positive_temperature(
    logits: Tensor,
    failure_targets: Tensor,
    *,
    calibration_sha256: str,
    valid_mask: Tensor | None = None,
    max_iterations: int = 100,
) -> TemperatureCalibration:
    """Fit scalar temperature by held-out BCE, refusing one-class data."""

    library = _require_torch()
    identity = _require_digest(calibration_sha256, "calibration_sha256")
    if not isinstance(logits, library.Tensor) or not isinstance(
        failure_targets, library.Tensor
    ):
        raise TypeError("calibration logits and targets must be tensors")
    if logits.shape != failure_targets.shape or logits.numel() < 1:
        raise ValueError("calibration logits and targets must be non-empty and aligned")
    if valid_mask is None:
        valid_mask = library.ones_like(logits, dtype=library.bool)
    if (
        not isinstance(valid_mask, library.Tensor)
        or valid_mask.dtype is not library.bool
    ):
        raise TypeError("valid_mask must be a bool tensor")
    if valid_mask.shape != logits.shape:
        raise ValueError("valid_mask must match calibration logits")
    selected_logits = logits.detach()[valid_mask].double().reshape(-1)
    selected_targets = failure_targets.detach()[valid_mask].double().reshape(-1)
    if selected_logits.numel() < 2 or not bool(library.isfinite(selected_logits).all()):
        raise ValueError("calibration requires at least two finite held-out logits")
    if not bool(library.isfinite(selected_targets).all()) or bool(
        ((selected_targets != 0) & (selected_targets != 1)).any()
    ):
        raise ValueError("calibration labels must be finite binary values")
    failures = int(selected_targets.sum().item())
    successes = int(selected_targets.numel()) - failures
    if failures == 0 or successes == 0:
        raise ValueError("temperature cannot be estimated from single-class data")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")

    raw = library.nn.Parameter(library.zeros((), dtype=library.float64))
    optimizer = library.optim.LBFGS(
        [raw], lr=0.25, max_iter=max_iterations, line_search_fn="strong_wolfe"
    )

    def objective() -> Any:
        optimizer.zero_grad(set_to_none=True)
        temperature = raw.exp().clamp(1e-4, 1e4)
        loss = library.nn.functional.binary_cross_entropy_with_logits(
            selected_logits / temperature, selected_targets
        )
        if not bool(library.isfinite(loss)):
            raise RuntimeError("temperature objective became non-finite")
        loss.backward()
        return loss

    before = library.nn.functional.binary_cross_entropy_with_logits(
        selected_logits, selected_targets
    )
    optimizer.step(objective)
    temperature = float(raw.detach().exp().clamp(1e-4, 1e4).item())
    after = library.nn.functional.binary_cross_entropy_with_logits(
        selected_logits / temperature, selected_targets
    )
    if not math.isfinite(temperature) or not bool(library.isfinite(after)):
        raise RuntimeError("temperature calibration did not converge to a finite value")
    return TemperatureCalibration(
        temperature=temperature,
        calibration_sha256=identity,
        examples=int(selected_targets.numel()),
        failures=failures,
        successes=successes,
        bce_before=float(before.item()),
        bce_after=float(after.item()),
    )


class CheckpointStage(str, enum.Enum):
    WARMUP = "warmup"
    EXECUTION_SNAPSHOT = "S0"
    OUTCOME_PREDICTOR = "predictor"
    CALIBRATED_PREDICTOR = "calibrated_predictor"


@dataclasses.dataclass(frozen=True)
class CheckpointManifest:
    schema_version: str
    stage: str
    payload_sha256: str
    file_sha256: str
    model_state_sha256: str
    optimizer_state_sha256: str | None
    identity: CheckpointIdentity
    calibration: TemperatureCalibration | None


def _stage(value: CheckpointStage | str) -> str:
    try:
        return CheckpointStage(value).value
    except (TypeError, ValueError) as error:
        raise ValueError("unsupported PSR checkpoint stage") from error


def _rng_state() -> dict[str, Any]:
    library = _require_torch()
    result: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": library.get_rng_state(),
    }
    if library.cuda.is_available():
        result["torch_cuda"] = library.cuda.get_rng_state_all()
    return result


def _payload_identity(
    *,
    stage: str,
    identity: CheckpointIdentity,
    model_state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any] | None,
    rng_state: Mapping[str, Any],
    calibration: TemperatureCalibration | None,
) -> tuple[dict[str, Any], str]:
    fields = {
        "schema_version": CHECKPOINT_SCHEMA,
        "stage": stage,
        "identity": identity.to_dict(),
        "model_state_sha256": object_sha256(model_state),
        "optimizer_state_sha256": (
            None if optimizer_state is None else object_sha256(optimizer_state)
        ),
        "rng_state_sha256": object_sha256(rng_state),
        "calibration": (
            None if calibration is None else dataclasses.asdict(calibration)
        ),
    }
    return fields, canonical_sha256(fields)


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def save_training_checkpoint(
    path: str | Path,
    *,
    stage: CheckpointStage | str,
    model: Any,
    identity: CheckpointIdentity,
    optimizer: Any | None = None,
    calibration: TemperatureCalibration | None = None,
) -> CheckpointManifest:
    """Atomically save model/optimizer/RNG state and content identities."""

    library = _require_torch()
    stage_name = _stage(stage)
    if stage_name == CheckpointStage.CALIBRATED_PREDICTOR.value and calibration is None:
        raise ValueError("calibrated predictor checkpoint requires calibration data")
    if (
        stage_name != CheckpointStage.CALIBRATED_PREDICTOR.value
        and calibration is not None
    ):
        raise ValueError(
            "temperature belongs only to a calibrated predictor checkpoint"
        )
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    model_state = _model_owner(model).state_dict()
    optimizer_state = None if optimizer is None else optimizer.state_dict()
    rng_state = _rng_state()
    fields, payload_digest = _payload_identity(
        stage=stage_name,
        identity=identity,
        model_state=model_state,
        optimizer_state=optimizer_state,
        rng_state=rng_state,
        calibration=calibration,
    )
    payload = {
        **fields,
        "payload_sha256": payload_digest,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "rng_state": rng_state,
    }
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        library.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return CheckpointManifest(
        schema_version=CHECKPOINT_SCHEMA,
        stage=stage_name,
        payload_sha256=payload_digest,
        file_sha256=_file_sha256(destination),
        model_state_sha256=fields["model_state_sha256"],
        optimizer_state_sha256=fields["optimizer_state_sha256"],
        identity=identity,
        calibration=calibration,
    )


def load_training_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any | None = None,
    expected_stage: CheckpointStage | str | None = None,
    expected_execution_snapshot_id: str | None = None,
    expected_file_sha256: str | None = None,
    restore_rng: bool = False,
    strict: bool = True,
) -> CheckpointManifest:
    """Verify all identities before mutating the supplied model or optimizer."""

    library = _require_torch()
    source = Path(path).expanduser().resolve()
    observed_file_digest = _file_sha256(source)
    if expected_file_sha256 is not None and observed_file_digest != _require_digest(
        expected_file_sha256, "expected_file_sha256"
    ):
        raise ValueError("checkpoint file SHA-256 mismatch")
    payload = library.load(source, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != CHECKPOINT_SCHEMA
    ):
        raise ValueError("unsupported or malformed PSR checkpoint")
    stage_name = _stage(payload.get("stage"))
    if expected_stage is not None and stage_name != _stage(expected_stage):
        raise ValueError("checkpoint training stage mismatch")
    identity = CheckpointIdentity.from_mapping(payload.get("identity", {}))
    if expected_execution_snapshot_id is not None:
        expected = _require_digest(
            expected_execution_snapshot_id, "expected_execution_snapshot_id"
        )
        if identity.execution_snapshot_id != expected:
            raise ValueError("checkpoint S0 identity mismatch")
    raw_calibration = payload.get("calibration")
    calibration = (
        None
        if raw_calibration is None
        else TemperatureCalibration(**dict(raw_calibration))
    )
    fields, observed_payload_digest = _payload_identity(
        stage=stage_name,
        identity=identity,
        model_state=payload.get("model_state", {}),
        optimizer_state=payload.get("optimizer_state"),
        rng_state=payload.get("rng_state", {}),
        calibration=calibration,
    )
    if payload.get("payload_sha256") != observed_payload_digest:
        raise ValueError("checkpoint payload SHA-256 mismatch")
    for name in ("model_state_sha256", "optimizer_state_sha256", "rng_state_sha256"):
        if payload.get(name) != fields[name]:
            raise ValueError(f"checkpoint {name} does not match its stored state")

    # Mutation begins only after every content and protocol check succeeded.
    _model_owner(model).load_state_dict(payload["model_state"], strict=strict)
    if optimizer is not None:
        if payload.get("optimizer_state") is None:
            raise ValueError("checkpoint has no optimizer state to resume")
        optimizer.load_state_dict(payload["optimizer_state"])
    if restore_rng:
        state = payload["rng_state"]
        random.setstate(tuple(state["python"]))
        library.set_rng_state(state["torch_cpu"])
        if library.cuda.is_available() and "torch_cuda" in state:
            library.cuda.set_rng_state_all(state["torch_cuda"])
    return CheckpointManifest(
        schema_version=CHECKPOINT_SCHEMA,
        stage=stage_name,
        payload_sha256=observed_payload_digest,
        file_sha256=observed_file_digest,
        model_state_sha256=fields["model_state_sha256"],
        optimizer_state_sha256=fields["optimizer_state_sha256"],
        identity=identity,
        calibration=calibration,
    )


__all__ = [
    "CHECKPOINT_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "CheckpointIdentity",
    "CheckpointManifest",
    "CheckpointStage",
    "FrozenExecutionGuard",
    "FrozenExecutionSnapshot",
    "StageALoss",
    "StageCBatch",
    "StageCLoss",
    "StageCStepResult",
    "TemperatureCalibration",
    "assert_disjoint_stage_c_parameters",
    "assert_execution_snapshot_unchanged",
    "compose_stage_a_loss",
    "compose_stage_c_loss",
    "fit_positive_temperature",
    "freeze_execution_snapshot",
    "load_training_checkpoint",
    "module_state_sha256",
    "object_sha256",
    "parameter_sha256",
    "save_training_checkpoint",
    "train_stage_c_step",
]
