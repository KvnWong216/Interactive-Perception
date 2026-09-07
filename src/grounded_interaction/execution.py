"""Stage-2 execution contract with end-to-end candidate identity binding."""

from __future__ import annotations

import dataclasses
import hmac
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from .contracts import (
    ExecutionStatus,
    PolicyContext,
    Primitive,
    PublicFrame,
    canonical_sha256,
)
from .serialization import SerializedSubtask

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _clean_text(value: object, *, name: str) -> str:
    result = " ".join(str(value or "").split())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _require_sha256(value: object, *, name: str) -> str:
    result = str(value)
    if not _SHA256_RE.fullmatch(result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _same_utf8(left: str, right: str) -> bool:
    """Compare exact UTF-8 bytes without normalization or coercion."""

    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    return value


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


@dataclasses.dataclass(frozen=True)
class ExecutorRequest:
    """One physical execution request derived from one serialized candidate."""

    serializer_id: str
    serialized_subtask_digest: str
    candidate_id: str
    candidate_fingerprint: str
    primitive: Primitive
    context_fingerprint: str
    subtask_text: str
    spatial_audit_payload: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in ("serializer_id", "candidate_id", "subtask_text"):
            object.__setattr__(self, name, _clean_text(getattr(self, name), name=name))
        for name in (
            "serialized_subtask_digest",
            "candidate_fingerprint",
            "context_fingerprint",
        ):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), name=name)
            )
        if not isinstance(self.primitive, Primitive):
            raise TypeError("primitive must be a Primitive")
        if self.primitive is Primitive.STOP:
            raise ValueError("STOP must never be converted into an executor request")
        if not isinstance(self.spatial_audit_payload, Mapping):
            raise TypeError("spatial_audit_payload must be a mapping")
        plain_audit = _plain(self.spatial_audit_payload)
        if not isinstance(plain_audit, dict):
            raise TypeError("spatial_audit_payload must be a mapping")
        # Validation through canonical serialization rejects opaque and
        # non-finite objects before the payload is recursively frozen.
        canonical_sha256(plain_audit)
        object.__setattr__(
            self,
            "spatial_audit_payload",
            _freeze(plain_audit),
        )

    @classmethod
    def from_serialized(cls, value: SerializedSubtask) -> ExecutorRequest:
        if not isinstance(value, SerializedSubtask):
            raise TypeError("value must be a SerializedSubtask")
        return cls(
            serializer_id=value.serializer_id,
            serialized_subtask_digest=value.digest,
            candidate_id=value.candidate_id,
            candidate_fingerprint=value.candidate_fingerprint,
            primitive=value.primitive,
            context_fingerprint=value.context_fingerprint,
            subtask_text=value.subtask_text,
            spatial_audit_payload=value.spatial_audit_payload,
        )

    def payload(self) -> dict[str, object]:
        return {
            "serializer_id": self.serializer_id,
            "serialized_subtask_digest": self.serialized_subtask_digest,
            "candidate_id": self.candidate_id,
            "candidate_fingerprint": self.candidate_fingerprint,
            "primitive": self.primitive.value,
            "context_fingerprint": self.context_fingerprint,
            "subtask_text": self.subtask_text,
            "spatial_audit_payload": _plain(self.spatial_audit_payload),
        }

    @property
    def request_digest(self) -> str:
        return canonical_sha256(self.payload())

    def to_dict(self) -> dict[str, object]:
        value = self.payload()
        value["request_digest"] = self.request_digest
        return value

    def validate_context(self, context: PolicyContext) -> None:
        if not isinstance(context, PolicyContext):
            raise TypeError("context must be a PolicyContext")
        if not _same_utf8(self.context_fingerprint, context.fingerprint()):
            raise ValueError("executor request does not match the policy context")

    def stage2_policy_payload(self, context: PolicyContext) -> dict[str, object]:
        """Return the stock text-VLA view; exact spatial audit data is excluded."""

        self.validate_context(context)
        return {
            "subtask_text": self.subtask_text,
            "frames": [frame.to_dict() for frame in context.frames],
            "proprioception": (
                list(context.proprioception)
                if context.proprioception is not None
                else None
            ),
        }


@dataclasses.dataclass(frozen=True)
class ExecutorReceipt:
    """Receipt whose identity fields must exactly echo the execution request."""

    receipt_id: str
    executor_id: str
    candidate_id: str
    candidate_fingerprint: str
    request_digest: str
    status: ExecutionStatus
    post_frames: tuple[PublicFrame, ...]

    def __post_init__(self) -> None:
        for name in ("receipt_id", "executor_id", "candidate_id"):
            object.__setattr__(self, name, _clean_text(getattr(self, name), name=name))
        for name in ("candidate_fingerprint", "request_digest"):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), name=name)
            )
        if not isinstance(self.status, ExecutionStatus):
            raise TypeError("status must be an ExecutionStatus")
        frames = tuple(self.post_frames)
        if not frames or any(not isinstance(frame, PublicFrame) for frame in frames):
            raise ValueError("post_frames must contain at least one PublicFrame")
        object.__setattr__(self, "post_frames", frames)

    def validate_request(self, request: ExecutorRequest) -> None:
        if not isinstance(request, ExecutorRequest):
            raise TypeError("request must be an ExecutorRequest")
        comparisons = (
            ("candidate_id", self.candidate_id, request.candidate_id),
            (
                "candidate_fingerprint",
                self.candidate_fingerprint,
                request.candidate_fingerprint,
            ),
            ("request_digest", self.request_digest, request.request_digest),
        )
        for name, receipt_value, request_value in comparisons:
            if not _same_utf8(receipt_value, request_value):
                raise ValueError(f"receipt {name} does not byte-match request")

    def to_dict(self) -> dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            "executor_id": self.executor_id,
            "candidate_id": self.candidate_id,
            "candidate_fingerprint": self.candidate_fingerprint,
            "request_digest": self.request_digest,
            "status": self.status.value,
            "post_frames": [frame.to_dict() for frame in self.post_frames],
        }


@runtime_checkable
class FrozenVLAExecutor(Protocol):
    """Stage-2 boundary; a concrete adapter may call any frozen VLA."""

    @property
    def executor_id(self) -> str: ...

    def execute(
        self,
        request: ExecutorRequest,
        context: PolicyContext,
    ) -> ExecutorReceipt: ...


@dataclasses.dataclass(frozen=True)
class ReplayOutcome:
    """Recorded post-action observation used only for deterministic wiring tests."""

    status: ExecutionStatus
    post_frames: tuple[PublicFrame, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.status, ExecutionStatus):
            raise TypeError("status must be an ExecutionStatus")
        frames = tuple(self.post_frames)
        if not frames or any(not isinstance(frame, PublicFrame) for frame in frames):
            raise ValueError("post_frames must contain at least one PublicFrame")
        object.__setattr__(self, "post_frames", frames)


class ReplayExecutor:
    """Deterministic, outcome-free adapter for contract and loop tests.

    Fixtures are keyed by immutable candidate fingerprint.  Replaying a fixture
    demonstrates data flow only and must not be reported as model performance.
    """

    def __init__(
        self,
        fixtures: Mapping[str, ReplayOutcome],
        *,
        executor_id: str = "replay-executor-v1",
    ) -> None:
        self._executor_id = _clean_text(executor_id, name="executor_id")
        normalized: dict[str, ReplayOutcome] = {}
        for fingerprint, outcome in fixtures.items():
            digest = _require_sha256(fingerprint, name="fixture candidate fingerprint")
            if not isinstance(outcome, ReplayOutcome):
                raise TypeError("fixtures must map fingerprints to ReplayOutcome")
            normalized[digest] = outcome
        self._fixtures = normalized

    @property
    def executor_id(self) -> str:
        return self._executor_id

    def execute(
        self,
        request: ExecutorRequest,
        context: PolicyContext,
    ) -> ExecutorReceipt:
        if not isinstance(request, ExecutorRequest):
            raise TypeError("request must be an ExecutorRequest")
        request.validate_context(context)
        try:
            outcome = self._fixtures[request.candidate_fingerprint]
        except KeyError as error:
            raise KeyError(
                "no replay fixture for candidate fingerprint "
                f"{request.candidate_fingerprint}"
            ) from error
        receipt_id = canonical_sha256(
            {
                "executor_id": self.executor_id,
                "request_digest": request.request_digest,
                "status": outcome.status.value,
                "post_frames": [frame.to_dict() for frame in outcome.post_frames],
            }
        )
        receipt = ExecutorReceipt(
            receipt_id=receipt_id,
            executor_id=self.executor_id,
            candidate_id=request.candidate_id,
            candidate_fingerprint=request.candidate_fingerprint,
            request_digest=request.request_digest,
            status=outcome.status,
            post_frames=outcome.post_frames,
        )
        receipt.validate_request(request)
        return receipt
