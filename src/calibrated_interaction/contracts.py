"""Policy-visible contracts for candidate-conditioned interaction selection."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any


class Primitive(str, Enum):
    """High-level actions exposed by the real executor capability registry."""

    DIRECT = "DIRECT"
    OPEN = "OPEN"
    REMOVE_OCCLUDER = "REMOVE_OCCLUDER"
    MOVE_CLOSER = "MOVE_CLOSER"
    ROTATE = "ROTATE"
    STOP = "STOP"

    @property
    def is_information_action(self) -> bool:
        return self in {
            Primitive.OPEN,
            Primitive.REMOVE_OCCLUDER,
            Primitive.MOVE_CLOSER,
            Primitive.ROTATE,
        }


class EffectFactor(str, Enum):
    """Observable, non-exclusive post-action facts.

    The initially proposed six-way outcome ontology is not mutually exclusive:
    for example, an opened empty region can both reject a candidate and certify
    the region empty.  Independent Bernoulli factors preserve those semantics
    without inventing a manually weighted aggregate score.
    """

    EXECUTION_SUCCEEDED = "execution_succeeded"
    TASK_RELEVANT_CHANGE = "task_relevant_change"
    AMBIGUITY_REDUCED = "ambiguity_reduced"
    TARGET_CONFIRMED = "target_confirmed"
    CANDIDATE_REJECTED = "candidate_rejected"
    REGION_CONFIRMED_EMPTY = "region_confirmed_empty"


def _clean_text(value: Any, *, name: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    result = " ".join(str(value or "").split())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


@dataclasses.dataclass(frozen=True)
class CandidateAction:
    """One schema-validated high-level action proposed for the current image."""

    candidate_id: str
    primitive: Primitive
    target: str | None
    reference: str | None = None
    reference_region_xyxy: tuple[float, float, float, float] | None = None
    purpose: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_id", _clean_text(self.candidate_id, name="candidate_id")
        )
        if self.primitive is Primitive.STOP:
            if self.target is not None or self.reference_region_xyxy is not None:
                raise ValueError("STOP must not claim a grounded target")
        else:
            object.__setattr__(self, "target", _clean_text(self.target, name="target"))
        if self.reference is not None:
            object.__setattr__(
                self,
                "reference",
                _clean_text(self.reference, name="reference", optional=True),
            )
        if self.purpose is not None:
            object.__setattr__(
                self,
                "purpose",
                _clean_text(self.purpose, name="purpose", optional=True),
            )
        if self.reference_region_xyxy is not None:
            values = tuple(float(value) for value in self.reference_region_xyxy)
            if len(values) != 4 or any(not math.isfinite(value) for value in values):
                raise ValueError(
                    "reference_region_xyxy must contain four finite values"
                )
            x0, y0, x1, y1 = values
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                raise ValueError(
                    "reference_region_xyxy must be a normalized non-empty box"
                )
            object.__setattr__(self, "reference_region_xyxy", values)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CandidateAction:
        region = value.get("reference_region_xyxy")
        return cls(
            candidate_id=str(value.get("candidate_id", "")),
            primitive=Primitive(str(value.get("primitive", ""))),
            target=value.get("target"),
            reference=value.get("reference"),
            reference_region_xyxy=(tuple(region) if region is not None else None),
            purpose=value.get("purpose"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "primitive": self.primitive.value,
            "target": self.target,
            "reference": self.reference,
            "reference_region_xyxy": (
                list(self.reference_region_xyxy)
                if self.reference_region_xyxy is not None
                else None
            ),
            "purpose": self.purpose,
        }


def validate_candidate_set(
    candidates: Sequence[CandidateAction],
) -> tuple[CandidateAction, ...]:
    result = tuple(candidates)
    if len(result) < 2:
        raise ValueError("candidate set must contain at least two actions")
    identifiers = [candidate.candidate_id for candidate in result]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate IDs must be unique")
    if not any(candidate.primitive is Primitive.STOP for candidate in result):
        raise ValueError("candidate set must include STOP")
    return result


@dataclasses.dataclass(frozen=True)
class PublicContext:
    """All and only the state available to the online policy."""

    prompt: str
    observation_frames: tuple[str, ...]
    history: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", _clean_text(self.prompt, name="prompt"))
        if not self.observation_frames or any(
            not str(path).strip() for path in self.observation_frames
        ):
            raise ValueError("at least one public observation frame is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "observation_frames": list(self.observation_frames),
            "history": [dict(item) for item in self.history],
        }
