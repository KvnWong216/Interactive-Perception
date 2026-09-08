"""Serialize one grounded intervention into a frozen-VLA subtask.

The serializer turns the public referring expression and a coarse image
location into instance-distinguishing language.  Exact point/box coordinates
remain in an audit payload.  This contract therefore does *not* claim that an
unmodified text-conditioned VLA consumes native spatial tokens or coordinates.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Any

from .contracts import (
    GroundedIntervention,
    PolicyContext,
    Primitive,
    canonical_json_bytes,
    canonical_sha256,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SpatialConditioningMode(str, Enum):
    """Stage-2 referent interfaces evaluated by the E1 ceiling experiment.

    ``VISUAL_MARKER`` is an explicitly diagnostic, public-image intervention;
    it is not native point/box support in the frozen VLA.
    """

    COARSE_TEXT = "coarse_text"
    PRECISE_TEXT = "precise_text"
    VISUAL_MARKER = "visual_marker"


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


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
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


@dataclasses.dataclass(frozen=True)
class SerializedSubtask:
    """Content-addressed text command and its non-executed spatial audit data."""

    serializer_id: str
    candidate_id: str
    candidate_fingerprint: str
    primitive: Primitive
    context_fingerprint: str
    subtask_text: str
    spatial_audit_payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("serializer_id", "candidate_id", "subtask_text"):
            object.__setattr__(self, name, _clean_text(getattr(self, name), name=name))
        for name in ("candidate_fingerprint", "context_fingerprint"):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), name=name)
            )
        if not isinstance(self.primitive, Primitive):
            raise TypeError("primitive must be a Primitive")
        try:
            detached = json.loads(
                canonical_json_bytes(self.spatial_audit_payload).decode("utf-8")
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise TypeError(
                "spatial_audit_payload must be finite and JSON-compatible"
            ) from error
        if not isinstance(detached, dict):
            raise TypeError("spatial_audit_payload must be a mapping")
        object.__setattr__(self, "spatial_audit_payload", _freeze(detached))

    def payload(self) -> dict[str, Any]:
        return {
            "serializer_id": self.serializer_id,
            "candidate_id": self.candidate_id,
            "candidate_fingerprint": self.candidate_fingerprint,
            "primitive": self.primitive.value,
            "context_fingerprint": self.context_fingerprint,
            "subtask_text": self.subtask_text,
            "spatial_audit_payload": _plain(self.spatial_audit_payload),
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.payload())

    def to_dict(self) -> dict[str, Any]:
        value = self.payload()
        value["serialized_subtask_digest"] = self.digest
        return value


class GroundedTextSerializer:
    """Create deterministic, instance-distinguishing natural-language subtasks."""

    def __init__(
        self,
        serializer_id: str = "grounded-text-v1",
        *,
        spatial_mode: SpatialConditioningMode = SpatialConditioningMode.COARSE_TEXT,
    ) -> None:
        self.serializer_id = _clean_text(serializer_id, name="serializer_id")
        if not isinstance(spatial_mode, SpatialConditioningMode):
            raise TypeError("spatial_mode must be a SpatialConditioningMode")
        self.spatial_mode = spatial_mode

    @staticmethod
    def _qualitative_location(intervention: GroundedIntervention) -> str:
        grounding = intervention.grounding
        if grounding is None:
            return ""
        x, y = grounding.point_xy
        horizontal = "left" if x < 1.0 / 3.0 else "right" if x > 2.0 / 3.0 else "center"
        vertical = "upper" if y < 1.0 / 3.0 else "lower" if y > 2.0 / 3.0 else "middle"
        location = horizontal if vertical == "middle" else f"{vertical}-{horizontal}"
        return f"the {location} of the current {grounding.camera} image"

    @staticmethod
    def _precise_location(intervention: GroundedIntervention) -> str:
        grounding = intervention.grounding
        if grounding is None:
            return ""
        x, y = grounding.point_xy
        x0, y0, x1, y1 = grounding.box_xyxy
        return (
            f"normalized center (x={x:.3f}, y={y:.3f}) in the current "
            f"{grounding.camera} image, inside normalized box "
            f"(x0={x0:.3f}, y0={y0:.3f}, x1={x1:.3f}, y1={y1:.3f}); "
            "x is measured from the left and y from the top"
        )

    def _identified_referent(self, intervention: GroundedIntervention) -> str:
        referent = str(intervention.referent)
        if intervention.grounding is None:
            return referent
        if self.spatial_mode is SpatialConditioningMode.COARSE_TEXT:
            return f"{referent}, the instance near {self._qualitative_location(intervention)}"
        if self.spatial_mode is SpatialConditioningMode.PRECISE_TEXT:
            return f"{referent} at {self._precise_location(intervention)}"
        if self.spatial_mode is SpatialConditioningMode.VISUAL_MARKER:
            return (
                f"{referent}, the instance enclosed by the magenta rectangle "
                f"and centered on the magenta cross in the current "
                f"{intervention.grounding.camera} image"
            )
        raise ValueError(f"unsupported spatial mode {self.spatial_mode!r}")

    @staticmethod
    def _parameter_sentence(intervention: GroundedIntervention) -> str:
        if not intervention.parameters:
            return ""
        values = "; ".join(
            f"{name}: {value}" for name, value in intervention.parameters
        )
        return f" Use these high-level parameters: {values}."

    def serialize(
        self,
        intervention: GroundedIntervention,
        context: PolicyContext,
    ) -> SerializedSubtask:
        if not isinstance(intervention, GroundedIntervention):
            raise TypeError("intervention must be a GroundedIntervention")
        if not isinstance(context, PolicyContext):
            raise TypeError("context must be a PolicyContext")
        intervention.validate_against(context)

        if intervention.primitive is Primitive.STOP:
            text = "Stop without issuing a robot action."
        else:
            identified = self._identified_referent(intervention)
            if intervention.primitive is Primitive.DIRECT:
                text = f"Complete this task: {context.prompt} Use {identified}."
            elif intervention.primitive is Primitive.OPEN:
                text = f"Open {identified}."
            elif intervention.primitive is Primitive.REMOVE:
                text = f"Move {identified} aside to clear the view."
            elif intervention.primitive is Primitive.ROTATE:
                text = f"Rotate {identified} so its visible sides can be inspected."
            elif intervention.primitive is Primitive.BRING_CLOSE:
                text = f"Bring {identified} closer for visual inspection."
            else:  # pragma: no cover - exhaustive guard for future enum changes
                raise ValueError(f"unsupported primitive {intervention.primitive!r}")
            text += self._parameter_sentence(intervention)

        grounding = intervention.grounding
        audit_payload: dict[str, Any] = {
            "candidate_id": intervention.candidate_id,
            "candidate_fingerprint": intervention.fingerprint(),
            "grounding": grounding.to_dict() if grounding is not None else None,
            "spatial_conditioning_mode": self.spatial_mode.value,
            "exact_spatial_binding_sent_as_native_vla_input": False,
            "qualitative_location_rendered_in_text": (
                grounding is not None
                and self.spatial_mode is SpatialConditioningMode.COARSE_TEXT
            ),
            "normalized_coordinates_rendered_in_text": (
                grounding is not None
                and self.spatial_mode is SpatialConditioningMode.PRECISE_TEXT
            ),
            "public_visual_marker_required": (
                grounding is not None
                and self.spatial_mode is SpatialConditioningMode.VISUAL_MARKER
            ),
        }
        return SerializedSubtask(
            serializer_id=self.serializer_id,
            candidate_id=intervention.candidate_id,
            candidate_fingerprint=intervention.fingerprint(),
            primitive=intervention.primitive,
            context_fingerprint=context.fingerprint(),
            subtask_text=text,
            spatial_audit_payload=audit_payload,
        )
