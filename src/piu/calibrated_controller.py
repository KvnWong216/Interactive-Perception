"""Set-valued PIU decision semantics without hand-weighted utilities."""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Mapping, Sequence

import numpy as np

from .action_effect import EFFECT_FACTORS

INFORMATION_PRIMITIVES = frozenset(
    {"OPEN", "REMOVE", "ROTATE", "MOVE_CLOSER", "PICK_TO_INSPECT"}
)
TASK_PRIMITIVES = frozenset({"DIRECT", "PICK", "PLACE"})


class DecisionKind(str, enum.Enum):
    EXECUTE = "EXECUTE"
    INTERACT = "INTERACT"
    ABSTAIN = "ABSTAIN"
    REPORT_NOT_FOUND = "REPORT_NOT_FOUND"
    STOP = "STOP"


def _binary_set(values: Sequence[bool]) -> frozenset[bool]:
    if any(not isinstance(value, bool) for value in values):
        raise TypeError("binary prediction set values must be booleans")
    result = frozenset(bool(value) for value in values)
    if not result <= {False, True}:
        raise ValueError("binary prediction set contains another label")
    return result


@dataclasses.dataclass(frozen=True)
class ControllerBeliefSets:
    task_sufficiency: frozenset[bool]
    target_presence: frozenset[bool]
    spatial_reference_available: frozenset[bool]
    holding_requested_target: frozenset[bool]
    search_coverage_sufficient: frozenset[bool]
    task_complete: frozenset[bool]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Sequence[bool]]) -> ControllerBeliefSets:
        required = {
            "task_sufficiency",
            "target_presence",
            "spatial_reference_available",
            "holding_requested_target",
            "search_coverage_sufficient",
            "task_complete",
        }
        if set(value) != required:
            raise ValueError("controller belief-set family mismatch")
        return cls(**{name: _binary_set(value[name]) for name in required})


@dataclasses.dataclass(frozen=True)
class CalibratedCandidate:
    candidate_id: str
    primitive: str
    factor_probabilities: Mapping[str, float | None]
    factor_prediction_sets: Mapping[str, frozenset[bool]]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> CalibratedCandidate:
        candidate_id = " ".join(str(value.get("candidate_id", "")).split())
        primitive = " ".join(str(value.get("primitive", "")).split()).upper()
        probability: dict[str, float | None] = {
            str(name): None if item is None else float(item)
            for name, item in dict(value.get("factor_probabilities", {})).items()
        }
        sets = {
            str(name): _binary_set(item)
            for name, item in dict(value.get("factor_prediction_sets", {})).items()
        }
        if not candidate_id or not primitive:
            raise ValueError("calibrated candidate identity and primitive are required")
        if set(probability) != set(EFFECT_FACTORS) or set(sets) != set(EFFECT_FACTORS):
            raise ValueError("calibrated candidate factor family mismatch")
        if any(
            not math.isfinite(item) or not 0.0 <= item <= 1.0
            for item in probability.values()
            if item is not None
        ):
            raise ValueError("effect probabilities must lie in [0,1]")
        return cls(candidate_id, primitive, probability, sets)


@dataclasses.dataclass(frozen=True)
class ControllerDecision:
    kind: DecisionKind
    selected_candidate_id: str | None
    route_prediction_set: tuple[str, ...]
    reason: str
    diagnostic_values: Mapping[str, float]


def frechet_joint_lower_bound(left: float, right: float) -> float:
    """Conservative joint-event probability without an independence assumption."""

    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (left, right)):
        raise ValueError("joint-event marginals must lie in [0,1]")
    return max(0.0, left + right - 1.0)


def diagnostic_candidate_value(candidate: CalibratedCandidate) -> float:
    """Return an interpretable probability lower bound, never a selection weight."""

    execution = candidate.factor_probabilities["execution_succeeded"]
    if candidate.primitive in INFORMATION_PRIMITIVES:
        outcome = candidate.factor_probabilities["task_relevant_change"]
    elif candidate.primitive in TASK_PRIMITIVES:
        outcome = candidate.factor_probabilities["task_progress_succeeded"]
    else:
        return 0.0
    if execution is None or outcome is None:
        return 0.0
    return frechet_joint_lower_bound(execution, outcome)


def decide_calibrated_sample(
    *,
    candidate_id: np.ndarray,
    candidate_primitive: np.ndarray,
    candidate_valid_mask: np.ndarray,
    calibrated: Mapping[str, np.ndarray],
    sample_index: int,
    belief_sets: ControllerBeliefSets,
    require_spatial_reference: bool = True,
) -> ControllerDecision:
    """Convert calibrated array axes to explicit candidate/decision contracts."""

    identifiers = np.asarray(candidate_id).astype(str)
    primitives = np.asarray(candidate_primitive).astype(str)
    valid = np.asarray(candidate_valid_mask, dtype=bool)
    probability = np.asarray(calibrated["factor_probability"], dtype=np.float64)
    factor_sets = np.asarray(calibrated["factor_prediction_sets"], dtype=bool)
    route_sets = np.asarray(calibrated["route_prediction_set"], dtype=bool)
    if identifiers.shape != primitives.shape or identifiers.shape != valid.shape:
        raise ValueError("candidate identity/primitive/mask array shapes differ")
    if probability.shape != (*valid.shape, len(EFFECT_FACTORS)):
        raise ValueError("calibrated factor probability shape mismatch")
    if factor_sets.shape != (*valid.shape, len(EFFECT_FACTORS), 2):
        raise ValueError("calibrated factor-set shape mismatch")
    if route_sets.shape != valid.shape:
        raise ValueError("calibrated route-set shape mismatch")
    if not 0 <= sample_index < len(valid):
        raise IndexError("calibrated controller sample index is out of range")
    candidates = []
    route = []
    for candidate_index in np.flatnonzero(valid[sample_index]):
        factor_probability = {}
        prediction_sets = {}
        for factor_index, factor_name in enumerate(EFFECT_FACTORS):
            item = probability[sample_index, candidate_index, factor_index]
            factor_probability[factor_name] = (
                float(item) if np.isfinite(item) else None
            )
            label_membership = factor_sets[
                sample_index, candidate_index, factor_index
            ]
            prediction_sets[factor_name] = frozenset(
                label
                for label, included in zip(
                    (False, True), label_membership, strict=True
                )
                if included
            )
        candidate = CalibratedCandidate(
            candidate_id=identifiers[sample_index, candidate_index],
            primitive=primitives[sample_index, candidate_index].upper(),
            factor_probabilities=factor_probability,
            factor_prediction_sets=prediction_sets,
        )
        candidates.append(candidate)
        if route_sets[sample_index, candidate_index]:
            route.append(candidate.candidate_id)
    return decide(
        candidates=candidates,
        route_prediction_set=route,
        belief_sets=belief_sets,
        require_spatial_reference=require_spatial_reference,
    )


def _is_positive(candidate: CalibratedCandidate, factor: str) -> bool:
    return candidate.factor_prediction_sets[factor] == frozenset({True})


def decide(
    *,
    candidates: Sequence[CalibratedCandidate],
    route_prediction_set: Sequence[str],
    belief_sets: ControllerBeliefSets,
    require_spatial_reference: bool = True,
) -> ControllerDecision:
    """Map calibrated sets to five explicit, non-overloaded decision semantics."""

    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("duplicate calibrated candidate ID")
    route = tuple(dict.fromkeys(str(value) for value in route_prediction_set))
    if not set(route) <= set(by_id):
        raise ValueError("route prediction set references an unknown candidate")
    values = {
        candidate.candidate_id: diagnostic_candidate_value(candidate)
        for candidate in candidates
    }
    if len(route) != 1:
        return ControllerDecision(
            DecisionKind.ABSTAIN,
            None,
            route,
            "route conformal set is not singleton",
            values,
        )
    candidate = by_id[route[0]]
    if candidate.primitive == "REPORT_NOT_FOUND":
        if (
            belief_sets.target_presence == frozenset({False})
            and belief_sets.search_coverage_sufficient == frozenset({True})
        ):
            return ControllerDecision(
                DecisionKind.REPORT_NOT_FOUND,
                candidate.candidate_id,
                route,
                "target absence and search coverage are both singleton-certified",
                values,
            )
        return ControllerDecision(
            DecisionKind.ABSTAIN,
            None,
            route,
            "not-found lacks explicit absence or search-coverage evidence",
            values,
        )
    if candidate.primitive == "STOP":
        if belief_sets.task_complete == frozenset({True}):
            return ControllerDecision(
                DecisionKind.STOP,
                candidate.candidate_id,
                route,
                "task completion is singleton-certified",
                values,
            )
        return ControllerDecision(
            DecisionKind.ABSTAIN,
            None,
            route,
            "STOP is not authorized by task completion",
            values,
        )
    if candidate.primitive in INFORMATION_PRIMITIVES:
        if belief_sets.task_sufficiency != frozenset({False}):
            return ControllerDecision(
                DecisionKind.ABSTAIN,
                None,
                route,
                "information action requires singleton insufficient evidence",
                values,
            )
        if not _is_positive(candidate, "execution_succeeded") or not _is_positive(
            candidate, "task_relevant_change"
        ):
            return ControllerDecision(
                DecisionKind.ABSTAIN,
                None,
                route,
                "information action effect set is ambiguous or negative",
                values,
            )
        return ControllerDecision(
            DecisionKind.INTERACT,
            candidate.candidate_id,
            route,
            "route and information-effect sets are singleton-certified",
            values,
        )
    if candidate.primitive in TASK_PRIMITIVES:
        if belief_sets.task_sufficiency != frozenset({True}):
            return ControllerDecision(
                DecisionKind.ABSTAIN,
                None,
                route,
                "task action requires singleton sufficient evidence",
                values,
            )
        if (
            candidate.primitive in {"DIRECT", "PICK"}
            and belief_sets.target_presence != frozenset({True})
        ):
            return ControllerDecision(
                DecisionKind.ABSTAIN,
                None,
                route,
                "task action requires singleton target presence",
                values,
            )
        required_holding = {
            "DIRECT": frozenset({False}),
            "PICK": frozenset({False}),
            "PLACE": frozenset({True}),
        }[candidate.primitive]
        if belief_sets.holding_requested_target != required_holding:
            return ControllerDecision(
                DecisionKind.ABSTAIN,
                None,
                route,
                f"{candidate.primitive} lacks singleton-compatible holding state",
                values,
            )
        if (
            candidate.primitive in {"DIRECT", "PICK"}
            and require_spatial_reference
            and belief_sets.spatial_reference_available != frozenset({True})
        ):
            return ControllerDecision(
                DecisionKind.ABSTAIN,
                None,
                route,
                "task action requires a calibrated current-frame spatial reference",
                values,
            )
        if not _is_positive(candidate, "execution_succeeded") or not _is_positive(
            candidate, "task_progress_succeeded"
        ):
            return ControllerDecision(
                DecisionKind.ABSTAIN,
                None,
                route,
                "task action effect set is ambiguous or negative",
                values,
            )
        return ControllerDecision(
            DecisionKind.EXECUTE,
            candidate.candidate_id,
            route,
            "route and task-progress sets are singleton-certified",
            values,
        )
    return ControllerDecision(
        DecisionKind.ABSTAIN,
        None,
        route,
        f"unregistered primitive {candidate.primitive}",
        values,
    )
