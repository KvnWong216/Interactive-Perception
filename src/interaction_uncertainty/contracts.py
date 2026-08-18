"""Strict, serializable runtime contracts for PIU."""

from __future__ import annotations

import dataclasses
import math
from enum import Enum
from typing import Any, Mapping, Sequence


def _probabilities(values: Mapping[str, float], *, name: str) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError(f"{name} must have at least two outcomes")
    result = {str(key): float(value) for key, value in values.items()}
    if any(not key or not math.isfinite(value) or value < 0.0 for key, value in result.items()):
        raise ValueError(f"{name} must contain finite non-negative probabilities")
    total = sum(result.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{name} probabilities must sum to one; got {total}")
    return result


class Primitive(str, Enum):
    DIRECT_ACT = "DIRECT_ACT"
    OPEN_TO_INSPECT = "OPEN_TO_INSPECT"
    STOP_NOT_FOUND = "STOP_NOT_FOUND"
    COMPLETE = "COMPLETE"
    ABSTAIN = "ABSTAIN"


class EffectOutcome(str, Enum):
    FAILED = "FAILED"
    REVEALED = "REVEALED"
    EMPTY = "EMPTY"


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    prompt: str
    target: str
    destination: str | None
    goal_relation: str | None
    required_facts: tuple[str, ...]
    completion_description: str

    def __post_init__(self) -> None:
        if not self.prompt.strip() or not self.target.strip():
            raise ValueError("prompt and target must be non-empty")
        if not self.required_facts or len(set(self.required_facts)) != len(self.required_facts):
            raise ValueError("required_facts must be unique and non-empty")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ObjectNode:
    object_id: str
    public_label_candidates: Mapping[str, float]
    bbox_xyxy: tuple[float, float, float, float] | None = None
    visible_area: int = 0
    affordances: frozenset[str] = frozenset()
    track_id: str | None = None

    def __post_init__(self) -> None:
        if not self.object_id.strip() or self.visible_area < 0:
            raise ValueError("invalid public object node")
        if self.public_label_candidates:
            labels = {
                str(key): float(value)
                for key, value in self.public_label_candidates.items()
            }
            if any(not key or not math.isfinite(value) or value < 0.0 for key, value in labels.items()):
                raise ValueError("label candidates must be finite and non-negative")
            total = sum(labels.values())
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("label candidate probabilities must sum to one")
            object.__setattr__(
                self,
                "public_label_candidates",
                labels,
            )


@dataclasses.dataclass(frozen=True)
class UnknownRegion:
    region_id: str
    parent_object_id: str
    region_type: str
    observed_fraction: float
    accessible_via: Primitive

    def __post_init__(self) -> None:
        if not self.region_id.strip() or not self.parent_object_id.strip():
            raise ValueError("unknown region IDs must be non-empty")
        if not 0.0 <= float(self.observed_fraction) <= 1.0:
            raise ValueError("observed_fraction must lie in [0,1]")


@dataclasses.dataclass(frozen=True)
class ScenePacket:
    frame_id: str
    prompt: str
    objects: tuple[ObjectNode, ...]
    unknown_regions: tuple[UnknownRegion, ...]
    public_robot_state: tuple[float, ...]
    source_views: tuple[str, ...] = ("agentview", "wrist")
    backend_stamp: str = "registered-public-affordance-v0"
    online_oracle_inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.frame_id.strip() or not self.prompt.strip():
            raise ValueError("frame_id and prompt must be non-empty")
        if self.online_oracle_inputs:
            raise ValueError("ScenePacket cannot contain online oracle inputs")
        identifiers = [item.object_id for item in self.objects] + [
            item.region_id for item in self.unknown_regions
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("scene node IDs must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "prompt": self.prompt,
            "objects": [dataclasses.asdict(item) for item in self.objects],
            "unknown_regions": [dataclasses.asdict(item) for item in self.unknown_regions],
            "public_robot_state": list(self.public_robot_state),
            "source_views": list(self.source_views),
            "backend_stamp": self.backend_stamp,
            "online_oracle_inputs": [],
        }


@dataclasses.dataclass(frozen=True)
class FactDistribution:
    name: str
    probabilities: Mapping[str, float]
    importance: float

    def __post_init__(self) -> None:
        if not self.name.strip() or not 0.0 <= float(self.importance) <= 1.0:
            raise ValueError("invalid fact name or importance")
        object.__setattr__(
            self,
            "probabilities",
            _probabilities(self.probabilities, name=f"fact {self.name}"),
        )

    @property
    def normalized_entropy(self) -> float:
        entropy = -sum(value * math.log(value) for value in self.probabilities.values() if value > 0.0)
        return entropy / math.log(len(self.probabilities))


@dataclasses.dataclass(frozen=True)
class TaskBelief:
    prompt: str
    facts: tuple[FactDistribution, ...]
    node_uncertainty: Mapping[str, float]
    model_stamp: str
    conformal_sets: Mapping[str, tuple[str, ...]] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt.strip() or not self.facts or not self.model_stamp.strip():
            raise ValueError("belief requires prompt, facts, and model stamp")
        if len({fact.name for fact in self.facts}) != len(self.facts):
            raise ValueError("fact names must be unique")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.node_uncertainty.values()):
            raise ValueError("node uncertainty must lie in [0,1]")

    def fact(self, name: str) -> FactDistribution:
        return next(item for item in self.facts if item.name == name)

    @property
    def task_uncertainty(self) -> float:
        denominator = sum(fact.importance for fact in self.facts)
        if denominator <= 0.0:
            return 0.0
        return sum(fact.importance * fact.normalized_entropy for fact in self.facts) / denominator

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "facts": [
                {
                    "name": fact.name,
                    "probabilities": dict(fact.probabilities),
                    "importance": fact.importance,
                    "normalized_entropy": fact.normalized_entropy,
                }
                for fact in self.facts
            ],
            "task_uncertainty": self.task_uncertainty,
            "node_uncertainty": dict(self.node_uncertainty),
            "model_stamp": self.model_stamp,
            "conformal_sets": {key: list(value) for key, value in self.conformal_sets.items()},
        }


@dataclasses.dataclass(frozen=True)
class CandidateAction:
    candidate_id: str
    primitive: Primitive
    target_id: str | None
    subtask: str
    stop_condition: str
    cost: float
    physical_risk: float

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.subtask.strip() or not self.stop_condition.strip():
            raise ValueError("candidate fields must be non-empty")
        if self.primitive in {Primitive.OPEN_TO_INSPECT, Primitive.DIRECT_ACT} and not self.target_id:
            raise ValueError("physical candidates require a public target ID")
        if self.cost < 0.0 or not 0.0 <= self.physical_risk <= 1.0:
            raise ValueError("invalid cost/risk")


@dataclasses.dataclass(frozen=True)
class ActionEffectForecast:
    candidate_id: str
    outcome_probabilities: Mapping[str, float]
    future_beliefs: Mapping[str, TaskBelief]
    execution_success_probability: float
    expected_task_progress: float
    model_stamp: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outcome_probabilities",
            _probabilities(self.outcome_probabilities, name="effect outcomes"),
        )
        if set(self.outcome_probabilities) != set(self.future_beliefs):
            raise ValueError("every effect outcome requires a future belief")
        if not 0.0 <= self.execution_success_probability <= 1.0:
            raise ValueError("execution success must lie in [0,1]")
        if not 0.0 <= self.expected_task_progress <= 1.0:
            raise ValueError("task progress must lie in [0,1]")

    @property
    def expected_future_uncertainty(self) -> float:
        return sum(
            probability * self.future_beliefs[outcome].task_uncertainty
            for outcome, probability in self.outcome_probabilities.items()
        )


@dataclasses.dataclass(frozen=True)
class PIUDecision:
    selected: CandidateAction
    utilities: Mapping[str, Mapping[str, float]]
    task_uncertainty: float
    valid_candidate_ids: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        selected = dataclasses.asdict(self.selected)
        selected["primitive"] = self.selected.primitive.value
        return {
            "selected": selected,
            "utilities": {key: dict(value) for key, value in self.utilities.items()},
            "task_uncertainty": self.task_uncertainty,
            "valid_candidate_ids": list(self.valid_candidate_ids),
            "reason": self.reason,
        }
