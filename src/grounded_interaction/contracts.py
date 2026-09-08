"""Public-input and grounded-intervention contracts for the formal pipeline.

This module deliberately contains no learned decision rule.  It defines the
smallest auditable unit that a later model may score: a physical intervention
whose primitive, public referring expression, high-level parameters, and
current-image grounding are already fixed.

Online-policy state is represented by typed schemas and checked recursively.
Private simulator state and evaluator annotations may exist in dataset
sidecars, but they have no explicit field in :class:`PolicyContext` or
``GroundedIntervention.policy_payload``. Concrete collectors remain responsible
for provenance: a schema cannot detect private state deliberately encoded into
an otherwise public string or tensor.

Only the Python standard library is used so these contracts can be validated
before a simulator, VLM, or VLA runtime is installed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from enum import Enum
from types import MappingProxyType
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
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
    "hidden",
    "route_label",
    "effect_label",
    "correct_action",
    "task_success",
    "reward",
)


def _clean_text(value: Any, *, name: str) -> str:
    result = " ".join(str(value or "").split())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _require_sha256(value: Any, *, name: str) -> str:
    result = str(value)
    if not _SHA256_RE.fullmatch(result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _exact_mapping(value: Any, *, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    observed = set(value)
    if observed != keys:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(keys - observed)}, "
            f"extra={sorted(observed - keys)}"
        )
    return value


def _freeze_json_value(value: Any, *, path: str) -> Any:
    """Validate and recursively freeze a JSON-compatible public value."""

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
            key = _clean_text(raw_key, name=f"{path} key")
            normalized = _normalized_key(key)
            if any(
                fragment in normalized for fragment in _FORBIDDEN_PUBLIC_KEY_FRAGMENTS
            ):
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
    """Return mutable JSON primitives from recursively frozen values."""

    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_plain_json_value(child) for child in value]
    return value


def assert_public_policy_value(value: Any, *, path: str = "policy_input") -> None:
    """Fail closed if ``value`` is not a clean public-policy payload.

    The function validates keys recursively rather than trusting a top-level
    source declaration.  It also rejects opaque Python objects and non-finite
    numbers, which prevents accidental serialization of simulator handles or
    array objects through a nominally public field.
    """

    _freeze_json_value(value, path=path)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value deterministically."""

    plain = _plain_json_value(value)
    return json.dumps(
        plain,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Compute a reproducible SHA-256 fingerprint for a contract payload."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


class Primitive(str, Enum):
    """High-level choices made by Stage 1; trajectories remain Stage-2 work."""

    DIRECT = "DIRECT"
    OPEN = "OPEN"
    REMOVE = "REMOVE"
    ROTATE = "ROTATE"
    BRING_CLOSE = "BRING_CLOSE"
    STOP = "STOP"

    @property
    def is_information_action(self) -> bool:
        return self in {
            Primitive.OPEN,
            Primitive.REMOVE,
            Primitive.ROTATE,
            Primitive.BRING_CLOSE,
        }


class ExecutionStatus(str, Enum):
    """Deployment-visible status returned by the frozen executor."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclasses.dataclass(frozen=True)
class PublicActionEvent:
    """The only action-history record visible to the next policy step."""

    step_index: int
    primitive: Primitive
    subtask_text: str
    execution_status: ExecutionStatus

    def __post_init__(self) -> None:
        if (
            not isinstance(self.step_index, int)
            or isinstance(self.step_index, bool)
            or self.step_index < 0
        ):
            raise ValueError("step_index must be a non-negative integer")
        if not isinstance(self.primitive, Primitive):
            raise TypeError("primitive must be a Primitive")
        if self.primitive is Primitive.STOP:
            raise ValueError("STOP does not create an executor history event")
        object.__setattr__(
            self,
            "subtask_text",
            _clean_text(self.subtask_text, name="subtask_text"),
        )
        if not isinstance(self.execution_status, ExecutionStatus):
            raise TypeError("execution_status must be an ExecutionStatus")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "primitive": self.primitive.value,
            "subtask_text": self.subtask_text,
            "execution_status": self.execution_status.value,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> PublicActionEvent:
        mapping = _exact_mapping(
            value,
            keys={"step_index", "primitive", "subtask_text", "execution_status"},
            name="public action event",
        )
        return cls(
            step_index=int(mapping["step_index"]),
            primitive=Primitive(str(mapping["primitive"])),
            subtask_text=str(mapping["subtask_text"]),
            execution_status=ExecutionStatus(str(mapping["execution_status"])),
        )


@dataclasses.dataclass(frozen=True)
class PublicFrame:
    """Content-addressed public RGB frame metadata.

    ``frame_id`` identifies one observation sample, whereas ``frame_index`` is
    its public temporal position.  The image bytes live in a dataset or runtime
    frame store and are bound here by ``image_sha256``; filesystem paths are not
    policy features.
    """

    frame_id: str
    camera: str
    frame_index: int
    image_sha256: str
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "frame_id", _clean_text(self.frame_id, name="frame_id")
        )
        object.__setattr__(self, "camera", _clean_text(self.camera, name="camera"))
        if (
            not isinstance(self.frame_index, int)
            or isinstance(self.frame_index, bool)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer")
        object.__setattr__(
            self,
            "image_sha256",
            _require_sha256(self.image_sha256, name="image_sha256"),
        )
        for name, value in (("width", self.width), ("height", self.height)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "camera": self.camera,
            "frame_index": self.frame_index,
            "image_sha256": self.image_sha256,
            "width": self.width,
            "height": self.height,
        }

    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Any) -> PublicFrame:
        mapping = _exact_mapping(
            value,
            keys={
                "frame_id",
                "camera",
                "frame_index",
                "image_sha256",
                "width",
                "height",
            },
            name="public frame",
        )
        return cls(
            frame_id=str(mapping["frame_id"]),
            camera=str(mapping["camera"]),
            frame_index=int(mapping["frame_index"]),
            image_sha256=str(mapping["image_sha256"]),
            width=int(mapping["width"]),
            height=int(mapping["height"]),
        )


@dataclasses.dataclass(frozen=True)
class PolicyContext:
    """All and only information available to the online Stage-1 policy."""

    prompt: str
    frames: tuple[PublicFrame, ...]
    public_history: tuple[PublicActionEvent, ...] = ()
    proprioception: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", _clean_text(self.prompt, name="prompt"))
        frames = tuple(self.frames)
        if not frames or any(not isinstance(frame, PublicFrame) for frame in frames):
            raise ValueError("frames must contain at least one PublicFrame")
        frame_ids = [frame.frame_id for frame in frames]
        frame_keys = [(frame.camera, frame.frame_index) for frame in frames]
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("frame_id values must be unique within a policy context")
        if len(set(frame_keys)) != len(frame_keys):
            raise ValueError(
                "each camera/frame_index pair must be unique within a policy context"
            )
        object.__setattr__(self, "frames", frames)

        history = tuple(self.public_history)
        if any(not isinstance(event, PublicActionEvent) for event in history):
            raise TypeError("public_history accepts PublicActionEvent values only")
        if any(event.step_index != index for index, event in enumerate(history)):
            raise ValueError("public_history step_index values must be consecutive")
        object.__setattr__(self, "public_history", history)

        if self.proprioception is not None:
            values = tuple(float(item) for item in self.proprioception)
            if not values or any(not math.isfinite(item) for item in values):
                raise ValueError(
                    "proprioception must be a non-empty finite numeric sequence"
                )
            object.__setattr__(self, "proprioception", values)

        assert_public_policy_value(self.to_dict(), path="policy_context")

    def frame_by_id(self, frame_id: str) -> PublicFrame:
        matches = [frame for frame in self.frames if frame.frame_id == frame_id]
        if len(matches) != 1:
            raise ValueError(f"unknown or ambiguous public frame_id {frame_id!r}")
        return matches[0]

    def latest_frame_index(self, camera: str) -> int:
        indices = [frame.frame_index for frame in self.frames if frame.camera == camera]
        if not indices:
            raise ValueError(f"unknown public camera {camera!r}")
        return max(indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "frames": [frame.to_dict() for frame in self.frames],
            "public_history": [event.to_dict() for event in self.public_history],
            "proprioception": (
                list(self.proprioception) if self.proprioception is not None else None
            ),
        }

    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Any) -> PolicyContext:
        mapping = _exact_mapping(
            value,
            keys={"prompt", "frames", "public_history", "proprioception"},
            name="policy context",
        )
        raw_proprioception = mapping["proprioception"]
        return cls(
            prompt=str(mapping["prompt"]),
            frames=tuple(PublicFrame.from_mapping(item) for item in mapping["frames"]),
            public_history=tuple(
                PublicActionEvent.from_mapping(item)
                for item in mapping["public_history"]
            ),
            proprioception=(
                None
                if raw_proprioception is None
                else tuple(float(item) for item in raw_proprioception)
            ),
        )


@dataclasses.dataclass(frozen=True)
class GroundingReference:
    """A point-and-box grounding tied to one exact current public frame."""

    camera: str
    frame_id: str
    frame_index: int
    image_sha256: str
    box_xyxy: tuple[float, float, float, float]
    point_xy: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera", _clean_text(self.camera, name="camera"))
        object.__setattr__(
            self, "frame_id", _clean_text(self.frame_id, name="frame_id")
        )
        if (
            not isinstance(self.frame_index, int)
            or isinstance(self.frame_index, bool)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer")
        object.__setattr__(
            self,
            "image_sha256",
            _require_sha256(self.image_sha256, name="image_sha256"),
        )
        box = tuple(float(item) for item in self.box_xyxy)
        point = tuple(float(item) for item in self.point_xy)
        if len(box) != 4 or any(not math.isfinite(item) for item in box):
            raise ValueError("box_xyxy must contain four finite values")
        if len(point) != 2 or any(not math.isfinite(item) for item in point):
            raise ValueError("point_xy must contain two finite values")
        x0, y0, x1, y1 = box
        x, y = point
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError("box_xyxy must be a non-empty normalized box")
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            raise ValueError("point_xy must lie inside box_xyxy")
        object.__setattr__(self, "box_xyxy", box)
        object.__setattr__(self, "point_xy", point)

    def validate_against(self, context: PolicyContext) -> None:
        frame = context.frame_by_id(self.frame_id)
        if (
            frame.camera != self.camera
            or frame.frame_index != self.frame_index
            or frame.image_sha256 != self.image_sha256
        ):
            raise ValueError(
                "grounding camera/frame/digest does not match the public frame"
            )
        if self.frame_index != context.latest_frame_index(self.camera):
            raise ValueError("grounding must reference the latest public camera frame")

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "image_sha256": self.image_sha256,
            "box_xyxy": list(self.box_xyxy),
            "point_xy": list(self.point_xy),
        }

    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Any) -> GroundingReference:
        mapping = _exact_mapping(
            value,
            keys={
                "camera",
                "frame_id",
                "frame_index",
                "image_sha256",
                "box_xyxy",
                "point_xy",
            },
            name="grounding reference",
        )
        return cls(
            camera=str(mapping["camera"]),
            frame_id=str(mapping["frame_id"]),
            frame_index=int(mapping["frame_index"]),
            image_sha256=str(mapping["image_sha256"]),
            box_xyxy=tuple(float(item) for item in mapping["box_xyxy"]),
            point_xy=tuple(float(item) for item in mapping["point_xy"]),
        )


@dataclasses.dataclass(frozen=True)
class GroundedIntervention:
    """One complete physical intervention available for outcome scoring.

    ``parameters`` contains only public high-level distinctions that alter the
    intended effect (for example ``("direction", "pull outward")``).  It must
    not contain a continuous robot trajectory, simulator identifier, or label.
    Every physical candidate is grounded to one exact public frame before it is
    eligible for scoring.  ``STOP`` is the sole non-physical exception.
    """

    candidate_id: str
    primitive: Primitive
    referent: str | None
    parameters: tuple[tuple[str, str], ...]
    grounding: GroundingReference | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _clean_text(self.candidate_id, name="candidate_id"),
        )
        if not isinstance(self.primitive, Primitive):
            raise TypeError("primitive must be a Primitive")

        if self.primitive is Primitive.STOP:
            if self.referent is not None or self.grounding is not None:
                raise ValueError("STOP must not claim a referent or grounding")
        else:
            object.__setattr__(
                self, "referent", _clean_text(self.referent, name="referent")
            )
            if not isinstance(self.grounding, GroundingReference):
                raise ValueError("every physical intervention requires grounding")

        normalized_parameters: list[tuple[str, str]] = []
        for index, item in enumerate(self.parameters):
            if not isinstance(item, Sequence) or isinstance(
                item, (str, bytes, bytearray)
            ):
                raise TypeError(f"parameter {index} must be a name/value pair")
            pair = tuple(item)
            if len(pair) != 2:
                raise ValueError(f"parameter {index} must contain exactly two values")
            name = _clean_text(pair[0], name=f"parameter {index} name")
            value = _clean_text(pair[1], name=f"parameter {index} value")
            normalized_parameters.append((name, value))
        names = [name for name, _ in normalized_parameters]
        if len(set(names)) != len(names):
            raise ValueError("intervention parameter names must be unique")
        normalized_parameters.sort(key=lambda item: item[0])
        object.__setattr__(self, "parameters", tuple(normalized_parameters))
        assert_public_policy_value(self.policy_payload(), path="grounded_intervention")

    def validate_against(self, context: PolicyContext) -> None:
        if self.grounding is not None:
            self.grounding.validate_against(context)

    def policy_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "primitive": self.primitive.value,
            "referent": self.referent,
            "parameters": [list(item) for item in self.parameters],
            "grounding": (
                self.grounding.to_dict() if self.grounding is not None else None
            ),
        }

    def fingerprint(self) -> str:
        return canonical_sha256(self.policy_payload())

    def assert_fingerprint(self, expected: str) -> None:
        expected_digest = _require_sha256(expected, name="candidate fingerprint")
        if self.fingerprint() != expected_digest:
            raise ValueError("grounded intervention fingerprint mismatch")

    @classmethod
    def from_mapping(cls, value: Any) -> GroundedIntervention:
        mapping = _exact_mapping(
            value,
            keys={"candidate_id", "primitive", "referent", "parameters", "grounding"},
            name="grounded intervention",
        )
        raw_grounding = mapping["grounding"]
        return cls(
            candidate_id=str(mapping["candidate_id"]),
            primitive=Primitive(str(mapping["primitive"])),
            referent=(
                None if mapping["referent"] is None else str(mapping["referent"])
            ),
            parameters=tuple(
                (str(item[0]), str(item[1])) for item in mapping["parameters"]
            ),
            grounding=(
                None
                if raw_grounding is None
                else GroundingReference.from_mapping(raw_grounding)
            ),
        )


@dataclasses.dataclass(frozen=True)
class OutcomeContract:
    """Frozen semantics of the single primary outcome predicted by Version 1.

    None of these fields has a repository default.  A caller must identify the
    exact continuation policy, evaluation horizon, executor, serializer, and
    treatment of attempted executions that do not complete normally before
    collecting outcome-bearing branches.
    """

    outcome_name: str
    continuation_policy_id: str
    horizon: int
    executor_id: str
    serializer_id: str
    failure_handling: str

    def __post_init__(self) -> None:
        for name in (
            "outcome_name",
            "continuation_policy_id",
            "executor_id",
            "serializer_id",
            "failure_handling",
        ):
            object.__setattr__(self, name, _clean_text(getattr(self, name), name=name))
        if (
            not isinstance(self.horizon, int)
            or isinstance(self.horizon, bool)
            or self.horizon < 1
        ):
            raise ValueError("horizon must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())

    def assert_fingerprint(self, expected: str) -> None:
        expected_digest = _require_sha256(expected, name="outcome contract fingerprint")
        if self.fingerprint() != expected_digest:
            raise ValueError("outcome contract fingerprint mismatch")

    @classmethod
    def from_mapping(cls, value: Any) -> OutcomeContract:
        mapping = _exact_mapping(
            value,
            keys={
                "outcome_name",
                "continuation_policy_id",
                "horizon",
                "executor_id",
                "serializer_id",
                "failure_handling",
            },
            name="outcome contract",
        )
        return cls(
            outcome_name=str(mapping["outcome_name"]),
            continuation_policy_id=str(mapping["continuation_policy_id"]),
            horizon=int(mapping["horizon"]),
            executor_id=str(mapping["executor_id"]),
            serializer_id=str(mapping["serializer_id"]),
            failure_handling=str(mapping["failure_handling"]),
        )
