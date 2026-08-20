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
    OBSERVE = "OBSERVE"
    OPEN_TO_INSPECT = "OPEN_TO_INSPECT"
    STOP_NOT_FOUND = "STOP_NOT_FOUND"
    COMPLETE = "COMPLETE"
    ABSTAIN = "ABSTAIN"


class EffectOutcome(str, Enum):
    FAILED = "FAILED"
    REVEALED = "REVEALED"
    EMPTY = "EMPTY"


class ObserveMode(str, Enum):
    """Physical realizations of one prompt-conditioned OBSERVE action."""

    CENTER_TARGET = "CENTER_TARGET"
    MOVE_CLOSER = "MOVE_CLOSER"
    NEXT_BEST_VIEW = "NEXT_BEST_VIEW"
    RETURN_TO_OBSERVE = "RETURN_TO_OBSERVE"
    OPEN_CONTAINER = "OPEN_CONTAINER"
    REMOVE_OCCLUDER = "REMOVE_OCCLUDER"
    ROTATE_TARGET = "ROTATE_TARGET"


class EvidenceNeed(str, Enum):
    IDENTITY = "IDENTITY"
    LOCATION = "LOCATION"
    RESOLUTION = "RESOLUTION"
    COVERAGE = "COVERAGE"
    GRASPABILITY = "GRASPABILITY"
    EXECUTOR_HANDOFF = "EXECUTOR_HANDOFF"


class EvidenceStatus(str, Enum):
    FAILED = "FAILED"
    EVIDENCE_ACQUIRED = "EVIDENCE_ACQUIRED"
    EMPTY = "EMPTY"
    AMBIGUOUS = "AMBIGUOUS"


@dataclasses.dataclass(frozen=True)
class ObserveRequest:
    """A task-level evidence request, independent of its physical mode."""

    query: str
    missing_facts: tuple[EvidenceNeed, ...]
    allowed_modes: tuple[ObserveMode, ...]
    target_id: str | None = None
    region_id: str | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("OBSERVE query must be non-empty")
        if not self.target_id and not self.region_id:
            raise ValueError("OBSERVE requires a public target or region ID")
        if not self.missing_facts or len(set(self.missing_facts)) != len(self.missing_facts):
            raise ValueError("OBSERVE missing facts must be unique and non-empty")
        if not self.allowed_modes or len(set(self.allowed_modes)) != len(self.allowed_modes):
            raise ValueError("OBSERVE modes must be unique and non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "target_id": self.target_id,
            "region_id": self.region_id,
            "missing_facts": [value.value for value in self.missing_facts],
            "allowed_modes": [value.value for value in self.allowed_modes],
        }


@dataclasses.dataclass(frozen=True)
class PublicTargetEvidence:
    """Oracle-free target proposal set used to authorize an OBSERVE motion."""

    frame_id: str
    view: str
    query: str
    prediction_set: tuple[str, ...]
    model_stamp: str
    image_size: tuple[int, int]
    bbox_xyxy: tuple[float, float, float, float] | None = None
    projected_area: float = 0.0
    calibration_artifact: str | None = None
    online_oracle_inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.online_oracle_inputs:
            raise ValueError("public target evidence cannot contain oracle inputs")
        if not self.frame_id.strip() or not self.view.strip() or not self.query.strip():
            raise ValueError("target evidence identifiers must be non-empty")
        if not self.model_stamp.strip() or min(self.image_size) <= 0:
            raise ValueError("target evidence requires a model stamp and image size")
        if self.projected_area < 0.0:
            raise ValueError("projected area must be non-negative")
        if self.bbox_xyxy is not None:
            x0, y0, x1, y1 = self.bbox_xyxy
            if not all(math.isfinite(value) for value in self.bbox_xyxy) or x1 <= x0 or y1 <= y0:
                raise ValueError("target evidence bbox must be finite and non-empty")

    @property
    def singleton(self) -> bool:
        return len(self.prediction_set) == 1

    @property
    def normalized_center_error(self) -> tuple[float, float] | None:
        if self.bbox_xyxy is None:
            return None
        width, height = self.image_size
        x0, y0, x1, y1 = self.bbox_xyxy
        return (
            ((x0 + x1) / 2.0 - width / 2.0) / width,
            ((y0 + y1) / 2.0 - height / 2.0) / height,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "view": self.view,
            "query": self.query,
            "prediction_set": list(self.prediction_set),
            "model_stamp": self.model_stamp,
            "image_size": list(self.image_size),
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy is not None else None,
            "projected_area": self.projected_area,
            "calibration_artifact": self.calibration_artifact,
            "normalized_center_error": self.normalized_center_error,
            "online_oracle_inputs": [],
        }


@dataclasses.dataclass(frozen=True)
class PublicGraspEvidence:
    """Executor-specific clear-setup readiness from public observations.

    ``GRASPABLE`` means more than a recognizable target or a 3-D target point.
    Candidates must be compatible with the declared executor's current RGB and
    robot-state handoff contract, then survive collision and reachability
    filtering.  This keeps scene transformation in OBSERVE and low-level task
    execution in the frozen VLA.
    """

    frame_id: str
    target_object_id: str
    prediction_set: tuple[str, ...]
    model_stamp: str
    candidate_count: int
    collision_free_candidate_count: int
    reachable_candidate_count: int
    executor_id: str = "frozen-pi05-libero"
    calibration_artifact: str | None = None
    online_oracle_inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.online_oracle_inputs:
            raise ValueError("public grasp evidence cannot contain oracle inputs")
        if not self.frame_id.strip() or not self.target_object_id.strip():
            raise ValueError("grasp evidence identifiers must be non-empty")
        if not self.model_stamp.strip() or not self.executor_id.strip():
            raise ValueError("grasp evidence requires model and executor stamps")
        counts = (
            self.candidate_count,
            self.collision_free_candidate_count,
            self.reachable_candidate_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("grasp candidate counts must be non-negative")
        if not (
            self.reachable_candidate_count
            <= self.collision_free_candidate_count
            <= self.candidate_count
        ):
            raise ValueError("grasp candidate counts must be monotonically filtered")

    @property
    def singleton_graspable(self) -> bool:
        return self.prediction_set == ("GRASPABLE",)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "target_object_id": self.target_object_id,
            "prediction_set": list(self.prediction_set),
            "model_stamp": self.model_stamp,
            "candidate_count": self.candidate_count,
            "collision_free_candidate_count": self.collision_free_candidate_count,
            "reachable_candidate_count": self.reachable_candidate_count,
            "executor_id": self.executor_id,
            "calibration_artifact": self.calibration_artifact,
            "online_oracle_inputs": [],
        }


@dataclasses.dataclass(frozen=True)
class ObserveResult:
    """Unified evidence result; EMPTY is valid only with a coverage certificate."""

    request: ObserveRequest
    mode: ObserveMode
    status: EvidenceStatus
    public_frame_ids: tuple[str, ...]
    model_stamp: str
    target_evidence: PublicTargetEvidence | None = None
    coverage_certified: bool = False
    online_oracle_inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.online_oracle_inputs:
            raise ValueError("OBSERVE result cannot contain oracle inputs")
        if self.mode not in self.request.allowed_modes:
            raise ValueError("executed OBSERVE mode was not allowed by the request")
        if not self.public_frame_ids or not self.model_stamp.strip():
            raise ValueError("OBSERVE result requires public history and model stamp")
        if self.status is EvidenceStatus.EVIDENCE_ACQUIRED:
            if self.target_evidence is None or not self.target_evidence.singleton:
                raise ValueError("acquired evidence requires a singleton public target set")
        if self.status is EvidenceStatus.EMPTY and not self.coverage_certified:
            raise ValueError("EMPTY requires a public coverage certificate")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "mode": self.mode.value,
            "status": self.status.value,
            "public_frame_ids": list(self.public_frame_ids),
            "model_stamp": self.model_stamp,
            "target_evidence": (
                self.target_evidence.to_dict() if self.target_evidence is not None else None
            ),
            "coverage_certified": self.coverage_certified,
            "online_oracle_inputs": [],
        }


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
    observe_request: ObserveRequest | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.subtask.strip() or not self.stop_condition.strip():
            raise ValueError("candidate fields must be non-empty")
        if self.primitive in {
            Primitive.OPEN_TO_INSPECT,
            Primitive.DIRECT_ACT,
        } and not self.target_id:
            raise ValueError("physical candidates require a public target ID")
        if self.primitive is Primitive.OBSERVE and self.observe_request is None:
            raise ValueError("OBSERVE candidate requires an evidence request")
        if (
            self.primitive is Primitive.OBSERVE
            and self.observe_request is not None
            and self.target_id
            not in {self.observe_request.target_id, self.observe_request.region_id}
        ):
            raise ValueError("OBSERVE target must match its public evidence request")
        if self.primitive is not Primitive.OBSERVE and self.observe_request is not None:
            raise ValueError("only OBSERVE candidates may carry an evidence request")
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
        selected["observe_request"] = (
            self.selected.observe_request.to_dict()
            if self.selected.observe_request is not None
            else None
        )
        return {
            "selected": selected,
            "utilities": {key: dict(value) for key, value in self.utilities.items()},
            "task_uncertainty": self.task_uncertainty,
            "valid_candidate_ids": list(self.valid_candidate_ids),
            "reason": self.reason,
        }
