"""Structured Qwen2.5-VL semantic evidence for prompt-aligned uncertainty.

The VLM is deliberately separated from the frozen robot policy.  It consumes
only public images plus an oracle-free object table, and emits a constrained
semantic assessment.  A deterministic selector turns that assessment into a
typed action; free-form VLM prose never authorizes ACT directly.
"""

from __future__ import annotations

import dataclasses
import json
import math
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


IDENTITY_OUTCOMES = (
    "target",
    "visually_similar_non_target",
    "other",
    "insufficient_visual_evidence",
)
SPECIAL_LOCATION_HYPOTHESES = ("OTHER_UNSEARCHED", "ABSENT")
GRASPABILITY_OUTCOMES = ("GRASPABLE", "NOT_GRASPABLE", "INSUFFICIENT_EVIDENCE")
EFFECT_OUTCOMES = (
    "FAILED",
    "SUCCESS",
    "NO_RELEVANT_CHANGE",
    "IDENTITY_RESOLVED",
    "TARGET_REVEALED",
    "REGION_EMPTY",
    "TARGET_GRASPABLE",
    "TASK_COMPLETED",
)


class SemanticAction(str, Enum):
    MOVE_CLOSER = "MOVE_CLOSER"
    NEXT_BEST_VIEW = "NEXT_BEST_VIEW"
    REMOVE_OCCLUDER = "REMOVE_OCCLUDER"
    OPEN_CONTAINER = "OPEN_CONTAINER"
    ACT = "ACT"
    STOP = "STOP"


def _finite_probability(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return result


def _identity_belief(value: Mapping[str, Any]) -> dict[str, float]:
    if set(value) != set(IDENTITY_OUTCOMES):
        raise ValueError(f"identity_belief must contain exactly {IDENTITY_OUTCOMES}")
    result = {
        name: _finite_probability(value[name], name=f"identity_belief.{name}")
        for name in IDENTITY_OUTCOMES
    }
    total = sum(result.values())
    # Decimal JSON generations commonly sum to 0.99 or 1.01.  Normalize only
    # this small rounding error; materially invalid VLM output is rejected.
    if not 0.98 <= total <= 1.02:
        raise ValueError(f"identity_belief must sum to one; got {total}")
    return {name: probability / total for name, probability in result.items()}


def _normalized_entropy(distribution: Mapping[str, float]) -> float:
    entropy = -sum(
        probability * math.log(probability)
        for probability in distribution.values()
        if probability > 0.0
    )
    return entropy / math.log(len(distribution))


@dataclasses.dataclass(frozen=True)
class RegionSemanticEvidence:
    object_id: str
    prompt_relevance: float
    identity_belief: Mapping[str, float]
    graspability_belief: Mapping[str, float]
    resolution_uncertainty: float
    occlusion_uncertainty: float
    state_uncertainty: float
    move_closer_effect_probability: float
    reason: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, allowed_object_ids: set[str]
    ) -> "RegionSemanticEvidence":
        object_id = str(value.get("object_id", ""))
        if object_id not in allowed_object_ids:
            raise ValueError(f"VLM referenced unknown object_id: {object_id}")
        reason = str(value.get("reason", "")).strip()
        if not reason:
            raise ValueError("region evidence requires a concise reason")
        return cls(
            object_id=object_id,
            prompt_relevance=_finite_probability(
                value.get("prompt_relevance"), name="prompt_relevance"
            ),
            identity_belief=_identity_belief(value.get("identity_belief", {})),
            graspability_belief=_named_distribution(
                value.get("graspability_belief", {}),
                names=GRASPABILITY_OUTCOMES,
                label="graspability_belief",
            ),
            resolution_uncertainty=_finite_probability(
                value.get("resolution_uncertainty"), name="resolution_uncertainty"
            ),
            occlusion_uncertainty=_finite_probability(
                value.get("occlusion_uncertainty"), name="occlusion_uncertainty"
            ),
            state_uncertainty=_finite_probability(
                value.get("state_uncertainty"), name="state_uncertainty"
            ),
            move_closer_effect_probability=_finite_probability(
                value.get("move_closer_effect_probability"),
                name="move_closer_effect_probability",
            ),
            reason=reason,
        )

    @property
    def identity_entropy(self) -> float:
        return _normalized_entropy(self.identity_belief)

    @property
    def uncertainty_mass(self) -> float:
        local_uncertainty = (
            0.60 * self.identity_entropy
            + 0.20 * self.resolution_uncertainty
            + 0.10 * self.occlusion_uncertainty
            + 0.10 * self.state_uncertainty
        )
        return self.prompt_relevance * local_uncertainty

    @property
    def move_closer_utility(self) -> float:
        return self.uncertainty_mass * self.move_closer_effect_probability

    def to_dict(self) -> dict[str, Any]:
        return {
            **dataclasses.asdict(self),
            "identity_belief": dict(self.identity_belief),
            "graspability_belief": dict(self.graspability_belief),
            "identity_entropy": self.identity_entropy,
            "uncertainty_mass": self.uncertainty_mass,
            "move_closer_utility": self.move_closer_utility,
        }


@dataclasses.dataclass(frozen=True)
class UnobservedRegionEvidence:
    object_id: str
    target_probability: float
    inspectability: float
    reason: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, allowed_object_ids: set[str]
    ) -> "UnobservedRegionEvidence":
        object_id = str(value.get("object_id", ""))
        if object_id not in allowed_object_ids:
            raise ValueError(f"VLM referenced unknown unobserved-region object: {object_id}")
        reason = str(value.get("reason", "")).strip()
        if not reason:
            raise ValueError("unobserved-region evidence requires a concise reason")
        return cls(
            object_id=object_id,
            target_probability=_finite_probability(
                value.get("target_probability"), name="target_probability"
            ),
            inspectability=_finite_probability(
                value.get("inspectability"), name="inspectability"
            ),
            reason=reason,
        )

    @property
    def inspect_utility(self) -> float:
        return self.target_probability * self.inspectability

    def to_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "inspect_utility": self.inspect_utility}


def _named_distribution(
    value: Mapping[str, Any], *, names: Sequence[str], label: str
) -> dict[str, float]:
    if set(value) != set(names):
        raise ValueError(f"{label} must contain exactly {tuple(names)}")
    result = {
        name: _finite_probability(value[name], name=f"{label}.{name}")
        for name in names
    }
    total = sum(result.values())
    if not 0.98 <= total <= 1.02:
        raise ValueError(f"{label} must sum to one; got {total}")
    return {name: probability / total for name, probability in result.items()}


@dataclasses.dataclass(frozen=True)
class ActionEffectEvidence:
    action: SemanticAction
    target_id: str
    applicable_probability: float
    execution_success_probability: float
    outcome_distribution: Mapping[str, float]
    expected_posterior_uncertainty: float
    expected_task_progress: float
    normalized_cost: float
    normalized_risk: float
    semantic_subtask: str
    reason: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, allowed_object_ids: set[str]
    ) -> "ActionEffectEvidence":
        action = SemanticAction(str(value.get("action", "")))
        if action is SemanticAction.STOP:
            raise ValueError("STOP is a selector terminal, not an effect option")
        target_id = str(value.get("target_id", ""))
        if target_id not in allowed_object_ids:
            raise ValueError(f"action effect referenced unknown target_id: {target_id}")
        raw_outcomes = value.get("outcome_distribution", {})
        if not isinstance(raw_outcomes, Mapping) or len(raw_outcomes) < 2:
            raise ValueError("outcome_distribution requires at least two outcomes")
        unknown = set(raw_outcomes) - set(EFFECT_OUTCOMES)
        if unknown:
            raise ValueError(f"unknown action-effect outcomes: {unknown}")
        outcomes = {
            str(name): _finite_probability(
                probability, name=f"outcome_distribution.{name}"
            )
            for name, probability in raw_outcomes.items()
        }
        total = sum(outcomes.values())
        if not 0.98 <= total <= 1.02:
            raise ValueError(f"outcome_distribution must sum to one; got {total}")
        outcomes = {name: probability / total for name, probability in outcomes.items()}
        reason = str(value.get("reason", "")).strip()
        semantic_subtask = str(value.get("semantic_subtask", "")).strip()
        if not reason or not semantic_subtask:
            raise ValueError("action effect requires a subtask and concise reason")
        return cls(
            action=action,
            target_id=target_id,
            applicable_probability=_finite_probability(
                value.get("applicable_probability"), name="applicable_probability"
            ),
            execution_success_probability=_finite_probability(
                value.get("execution_success_probability"),
                name="execution_success_probability",
            ),
            outcome_distribution=outcomes,
            expected_posterior_uncertainty=_finite_probability(
                value.get("expected_posterior_uncertainty"),
                name="expected_posterior_uncertainty",
            ),
            expected_task_progress=_finite_probability(
                value.get("expected_task_progress"), name="expected_task_progress"
            ),
            normalized_cost=_finite_probability(
                value.get("normalized_cost"), name="normalized_cost"
            ),
            normalized_risk=_finite_probability(
                value.get("normalized_risk"), name="normalized_risk"
            ),
            semantic_subtask=semantic_subtask,
            reason=reason,
        )

    def utility(self, current_uncertainty: float) -> float:
        information_gain = max(0.0, current_uncertainty - self.expected_posterior_uncertainty)
        return (
            self.applicable_probability
            * self.execution_success_probability
            * (information_gain + self.expected_task_progress)
            - self.normalized_cost
            - self.normalized_risk
        )

    def to_dict(self, *, current_uncertainty: float | None = None) -> dict[str, Any]:
        value = {
            **dataclasses.asdict(self),
            "action": self.action.value,
            "outcome_distribution": dict(self.outcome_distribution),
        }
        if current_uncertainty is not None:
            value["information_utility"] = self.utility(current_uncertainty)
        return value


@dataclasses.dataclass(frozen=True)
class ActionEffectAssessment:
    """VLM output restricted to counterfactual effects, never current belief."""

    interaction_options: tuple[ActionEffectEvidence, ...]
    advisory_action: SemanticAction
    advisory_target_id: str | None
    summary: str

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        allowed_object_ids: Sequence[str],
        registered_candidates: Sequence[Mapping[str, Any]],
    ) -> "ActionEffectAssessment":
        allowed = set(allowed_object_ids)
        candidate_by_pair = {
            (SemanticAction(str(row["action"])), str(row["target_id"])): row
            for row in registered_candidates
        }
        registered = set(candidate_by_pair)
        if not registered:
            raise ValueError("effect assessment requires registered candidates")
        parsed_options = []
        for row in value.get("interaction_options", ()):
            pair = (
                SemanticAction(str(row.get("action", ""))),
                str(row.get("target_id", "")),
            )
            if pair not in registered:
                raise ValueError(f"unregistered effect option: {(pair[0].value, pair[1])}")
            candidate = candidate_by_pair[pair]
            # Execution reliability, cost, and physical risk are registered
            # controller priors, not quantities the language model may rewrite.
            merged = {
                **row,
                "execution_success_probability": candidate[
                    "execution_success_prior"
                ],
                "normalized_cost": candidate["normalized_cost_prior"],
                "normalized_risk": candidate["normalized_risk_prior"],
            }
            parsed_options.append(
                ActionEffectEvidence.from_mapping(merged, allowed_object_ids=allowed)
            )
        options = tuple(parsed_options)
        proposed = {(row.action, row.target_id) for row in options}
        if proposed != registered:
            missing = sorted((a.value, t) for a, t in registered - proposed)
            extra = sorted((a.value, t) for a, t in proposed - registered)
            raise ValueError(
                "effect options must exactly match registered candidates; "
                f"missing={missing}, extra={extra}"
            )
        advisory_action = SemanticAction(str(value.get("advisory_action", "")))
        raw_target = value.get("advisory_target_id")
        advisory_target_id = None if raw_target is None else str(raw_target)
        if advisory_action is not SemanticAction.STOP and (
            advisory_action,
            str(advisory_target_id),
        ) not in registered:
            raise ValueError("advisory action/target is not a registered candidate")
        if advisory_action is SemanticAction.STOP and advisory_target_id is not None:
            raise ValueError("STOP advisory target must be null")
        summary = str(value.get("summary", "")).strip()
        if not summary:
            raise ValueError("effect assessment requires a summary")
        return cls(
            interaction_options=options,
            advisory_action=advisory_action,
            advisory_target_id=advisory_target_id,
            summary=summary,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "interaction_options": [row.to_dict() for row in self.interaction_options],
            "advisory_action": self.advisory_action.value,
            "advisory_target_id": self.advisory_target_id,
            "summary": self.summary,
        }

    @classmethod
    def from_ranked_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        allowed_object_ids: Sequence[str],
        registered_candidates: Sequence[Mapping[str, Any]],
        current_uncertainty: float,
        target_location_belief: Mapping[str, float],
    ) -> "ActionEffectAssessment":
        """Convert a robust discrete VLM ranking into numeric effect evidence."""

        allowed = set(allowed_object_ids)
        candidate_by_pair = {
            (SemanticAction(str(row["action"])), str(row["target_id"])): row
            for row in registered_candidates
        }
        rows = value.get("ranked_candidates", ())
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError("ranked_candidates must be a JSON list")
        ranked_pairs = []
        options = []
        decrease = {
            "LARGE_DECREASE": 0.50,
            "MODERATE_DECREASE": 0.25,
            "SMALL_DECREASE": 0.10,
            "NO_CHANGE": 0.0,
            "INCREASE": -0.10,
        }
        progress = {"NONE": 0.0, "INDIRECT": 0.15, "DIRECT": 0.65, "COMPLETE": 1.0}
        total_rows = max(1, len(rows))
        for rank, row in enumerate(rows):
            pair = (
                SemanticAction(str(row.get("action", ""))),
                str(row.get("target_id", "")),
            )
            if pair not in candidate_by_pair:
                # Free-form VLM suggestions never enter the executable set.
                # Ignore them and require every registered pair below.
                continue
            if pair in ranked_pairs:
                raise ValueError(f"duplicate ranked candidate: {(pair[0].value, pair[1])}")
            ranked_pairs.append(pair)
            outcome = str(row.get("likely_outcome", ""))
            if outcome not in EFFECT_OUTCOMES:
                raise ValueError(f"unknown likely_outcome: {outcome}")
            change = str(row.get("uncertainty_change", ""))
            if change not in decrease:
                raise ValueError(f"unknown uncertainty_change: {change}")
            progress_name = str(row.get("task_progress", ""))
            if progress_name not in progress:
                raise ValueError(f"unknown task_progress: {progress_name}")
            reason = str(row.get("reason", "")).strip()
            subtask = str(row.get("semantic_subtask", "")).strip()
            if not reason or not subtask:
                raise ValueError("ranked effect requires reason and semantic_subtask")
            candidate = candidate_by_pair[pair]
            success = float(candidate["execution_success_prior"])
            if outcome == "FAILED":
                outcomes = {"FAILED": 0.80, "NO_RELEVANT_CHANGE": 0.20}
            else:
                outcomes = {
                    outcome: success,
                    "FAILED": 1.0 - success,
                }
            minimum_information_gain = 0.0
            if outcome == "REGION_EMPTY":
                minimum_information_gain = 0.80 * float(
                    target_location_belief.get(pair[1], 0.0)
                )
            elif outcome in {"TARGET_REVEALED", "TASK_COMPLETED"}:
                minimum_information_gain = 0.80 * current_uncertainty
            elif outcome == "IDENTITY_RESOLVED":
                minimum_information_gain = 0.20
            elif outcome == "TARGET_GRASPABLE":
                minimum_information_gain = 0.10
            predicted_information_gain = max(
                decrease[change], minimum_information_gain
            )
            expected_uncertainty = min(
                1.0,
                max(0.0, current_uncertainty - predicted_information_gain),
            )
            # The VLM ranking is advisory. Applicability follows its predicted
            # physical outcome; an illegal high-ranked ACT cannot suppress a
            # legal information action in the explicit optimizer.
            if outcome in {
                "IDENTITY_RESOLVED",
                "TARGET_REVEALED",
                "REGION_EMPTY",
                "TARGET_GRASPABLE",
                "TASK_COMPLETED",
            }:
                applicability = 0.90
            elif outcome in {"FAILED", "NO_RELEVANT_CHANGE"}:
                applicability = 0.25
            else:
                applicability = 0.60
            task_progress = progress[progress_name]
            if pair[0] in {
                SemanticAction.MOVE_CLOSER,
                SemanticAction.NEXT_BEST_VIEW,
                SemanticAction.REMOVE_OCCLUDER,
                SemanticAction.OPEN_CONTAINER,
            }:
                task_progress = min(task_progress, progress["INDIRECT"])
            options.append(
                ActionEffectEvidence(
                    action=pair[0],
                    target_id=pair[1],
                    applicable_probability=applicability,
                    execution_success_probability=success,
                    outcome_distribution=outcomes,
                    expected_posterior_uncertainty=expected_uncertainty,
                    expected_task_progress=task_progress,
                    normalized_cost=float(candidate["normalized_cost_prior"]),
                    normalized_risk=float(candidate["normalized_risk_prior"]),
                    semantic_subtask=subtask,
                    reason=reason,
                )
            )
        if set(ranked_pairs) != set(candidate_by_pair):
            missing = sorted(
                (action.value, target)
                for action, target in set(candidate_by_pair) - set(ranked_pairs)
            )
            raise ValueError(f"ranking omitted registered candidates: {missing}")
        summary = str(value.get("summary", "")).strip()
        if not summary:
            raise ValueError("ranked effect assessment requires a summary")
        top = options[0]
        return cls(
            interaction_options=tuple(options),
            advisory_action=top.action,
            advisory_target_id=top.target_id,
            summary=summary,
        )


def assessment_from_pre_vlm_field(
    field: Mapping[str, Any],
    effects: ActionEffectAssessment,
    *,
    allowed_object_ids: Sequence[str],
) -> "SemanticAssessment":
    """Join measured current evidence with VLM-predicted future effects."""

    allowed = set(allowed_object_ids)
    regions = tuple(
        RegionSemanticEvidence.from_mapping(
            {
                **row,
                "move_closer_effect_probability": max(
                    float(row.get("identity_entropy", 0.0)),
                    float(row.get("resolution_uncertainty", 0.0)),
                ),
                "reason": "frozen visual frontend evidence",
            },
            allowed_object_ids=allowed,
        )
        for row in field.get("regions", ())
    )
    unobserved = tuple(
        UnobservedRegionEvidence.from_mapping(row, allowed_object_ids=allowed)
        for row in field.get("unobserved_regions", ())
    )
    task = field["task_spec"]
    return SemanticAssessment(
        target_query=str(task["target"]),
        destination_query=(
            None if task.get("destination") is None else str(task["destination"])
        ),
        goal_relation=str(task["goal_relation"]),
        required_facts=tuple(str(value) for value in task["required_facts"]),
        target_location_belief=dict(field["target_location_belief"]),
        regions=regions,
        unobserved_regions=unobserved,
        interaction_options=effects.interaction_options,
        advisory_action=effects.advisory_action,
        advisory_target_id=effects.advisory_target_id,
        summary=effects.summary,
        search_domain_exhausted=bool(field.get("search_domain_exhausted", False)),
    )


@dataclasses.dataclass(frozen=True)
class SemanticAssessment:
    target_query: str
    destination_query: str | None
    goal_relation: str
    required_facts: tuple[str, ...]
    target_location_belief: Mapping[str, float]
    regions: tuple[RegionSemanticEvidence, ...]
    unobserved_regions: tuple[UnobservedRegionEvidence, ...]
    interaction_options: tuple[ActionEffectEvidence, ...]
    advisory_action: SemanticAction
    advisory_target_id: str | None
    summary: str
    search_domain_exhausted: bool = False

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, allowed_object_ids: Sequence[str]
    ) -> "SemanticAssessment":
        allowed = set(allowed_object_ids)
        if not allowed:
            raise ValueError("semantic assessment requires public object proposals")
        target_query = str(value.get("target_query", "")).strip()
        raw_destination = value.get("destination_query")
        destination_query = (
            None if raw_destination is None else str(raw_destination).strip() or None
        )
        goal_relation = str(value.get("goal_relation", "")).strip()
        required_facts = tuple(
            str(fact).strip() for fact in value.get("required_facts", ())
        )
        summary = str(value.get("summary", "")).strip()
        if not target_query or not goal_relation or not summary:
            raise ValueError(
                "semantic assessment requires target_query, goal_relation, and summary"
            )
        if not required_facts or any(not fact for fact in required_facts):
            raise ValueError("semantic assessment requires non-empty task facts")
        if len(set(required_facts)) != len(required_facts):
            raise ValueError("required task facts must be unique")
        raw_location = value.get("target_location_belief", {})
        if not isinstance(raw_location, Mapping) or len(raw_location) < 2:
            raise ValueError("target_location_belief requires at least two hypotheses")
        allowed_locations = allowed | set(SPECIAL_LOCATION_HYPOTHESES)
        unknown_locations = set(str(key) for key in raw_location) - allowed_locations
        if unknown_locations:
            raise ValueError(
                f"target_location_belief contains unknown hypotheses: {unknown_locations}"
            )
        target_location_belief = {
            str(name): _finite_probability(probability, name=f"location.{name}")
            for name, probability in raw_location.items()
        }
        location_total = sum(target_location_belief.values())
        if not 0.98 <= location_total <= 1.02:
            raise ValueError(
                f"target_location_belief must sum to one; got {location_total}"
            )
        target_location_belief = {
            name: probability / location_total
            for name, probability in target_location_belief.items()
        }
        regions = tuple(
            RegionSemanticEvidence.from_mapping(row, allowed_object_ids=allowed)
            for row in value.get("regions", ())
        )
        if len({row.object_id for row in regions}) != len(regions):
            raise ValueError("semantic regions must have unique object IDs")
        unobserved = tuple(
            UnobservedRegionEvidence.from_mapping(row, allowed_object_ids=allowed)
            for row in value.get("unobserved_regions", ())
        )
        if len({row.object_id for row in unobserved}) != len(unobserved):
            raise ValueError("unobserved regions must have unique object IDs")
        interaction_options = tuple(
            ActionEffectEvidence.from_mapping(row, allowed_object_ids=allowed)
            for row in value.get("interaction_options", ())
        )
        if len({(row.action, row.target_id) for row in interaction_options}) != len(
            interaction_options
        ):
            raise ValueError("interaction options must have unique action/target pairs")
        advisory_action = SemanticAction(str(value.get("advisory_action", "")))
        advisory_target = value.get("advisory_target_id")
        advisory_target_id = None if advisory_target is None else str(advisory_target)
        if advisory_target_id is not None and advisory_target_id not in allowed | {"GLOBAL"}:
            raise ValueError("advisory_target_id is not a public object proposal")
        return cls(
            target_query=target_query,
            destination_query=destination_query,
            goal_relation=goal_relation,
            required_facts=required_facts,
            target_location_belief=target_location_belief,
            regions=regions,
            unobserved_regions=unobserved,
            interaction_options=interaction_options,
            advisory_action=advisory_action,
            advisory_target_id=advisory_target_id,
            summary=summary,
            search_domain_exhausted=bool(value.get("search_domain_exhausted", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_query": self.target_query,
            "destination_query": self.destination_query,
            "goal_relation": self.goal_relation,
            "required_facts": list(self.required_facts),
            "target_location_belief": dict(self.target_location_belief),
            "regions": [row.to_dict() for row in self.regions],
            "unobserved_regions": [row.to_dict() for row in self.unobserved_regions],
            "interaction_options": [row.to_dict() for row in self.interaction_options],
            "advisory_action": self.advisory_action.value,
            "advisory_target_id": self.advisory_target_id,
            "summary": self.summary,
            "search_domain_exhausted": self.search_domain_exhausted,
        }


@dataclasses.dataclass(frozen=True)
class SemanticDecisionConfig:
    act_false_commit_cost: float = 0.75
    # STOP/ABSTAIN is a controlled failure, not a zero-cost default action. Its
    # utility therefore pays the unresolved, prompt-relevant task loss.  This
    # remains a continuous expected-utility term; it is not a confidence gate.
    stop_failure_cost: float = 1.00


@dataclasses.dataclass(frozen=True)
class SemanticDecision:
    action: SemanticAction
    target_id: str | None
    reasons: tuple[str, ...]
    task_uncertainty: float
    target_prediction_set: tuple[str, ...]
    option_utilities: tuple[Mapping[str, Any], ...]
    advisory_agrees: bool
    stop_utility: float
    stop_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "target_id": self.target_id,
            "reasons": list(self.reasons),
            "task_uncertainty": self.task_uncertainty,
            "target_prediction_set": list(self.target_prediction_set),
            "option_utilities": [dict(value) for value in self.option_utilities],
            "advisory_agrees": self.advisory_agrees,
            "stop_utility": self.stop_utility,
            "stop_reason": self.stop_reason,
        }


def select_semantic_action(
    assessment: SemanticAssessment,
    *,
    scene_objects: Sequence[Mapping[str, Any]],
    config: SemanticDecisionConfig = SemanticDecisionConfig(),
) -> SemanticDecision:
    """Continuous expected-risk selection with no confidence cutoffs."""

    object_rows = {str(row["object_id"]): row for row in scene_objects}
    ranked = sorted(
        assessment.regions,
        key=lambda row: float(
            assessment.target_location_belief.get(row.object_id, 0.0)
        ),
        reverse=True,
    )
    location_uncertainty = _normalized_entropy(assessment.target_location_belief)
    task_uncertainty = max(
        location_uncertainty,
        max((row.uncertainty_mass for row in assessment.regions), default=0.0),
    )
    target_rank = tuple(row.object_id for row in ranked)
    region_by_id = {row.object_id: row for row in assessment.regions}
    unobserved_by_id = {
        row.object_id: row for row in assessment.unobserved_regions
    }
    areas = sorted(
        max(0.0, float(row.get("visible_area", 0.0)))
        for row in scene_objects
    )
    area_reference = areas[len(areas) // 2] if areas else 1.0

    def geometric_mean(values: Sequence[float]) -> float:
        clipped = [min(1.0, max(0.0, float(value))) for value in values]
        if not clipped or any(value == 0.0 for value in clipped):
            return 0.0
        return math.exp(sum(math.log(value) for value in clipped) / len(clipped))

    def readiness(option: ActionEffectEvidence) -> tuple[float, dict[str, float], float]:
        row = region_by_id.get(option.target_id)
        location = float(
            assessment.target_location_belief.get(option.target_id, 0.0)
        )
        consequence = 0.0
        if option.action is SemanticAction.ACT and row is not None:
            area = float(object_rows.get(option.target_id, {}).get("visible_area", 0.0))
            area_quality = area / (area + area_reference) if area + area_reference else 0.0
            components = {
                "target_location": location,
                "target_identity": float(row.identity_belief["target"]),
                "prompt_relevance": row.prompt_relevance,
                "graspability": float(row.graspability_belief["GRASPABLE"]),
                "projected_area_quality": area_quality,
                "unoccluded_quality": 1.0 - row.occlusion_uncertainty,
            }
            value = geometric_mean(tuple(components.values()))
            consequence = config.act_false_commit_cost * (1.0 - value)
            return value, components, consequence
        if option.action is SemanticAction.OPEN_CONTAINER:
            region = unobserved_by_id.get(option.target_id)
            components = {
                "target_location": location,
                "inspectability": 0.0 if region is None else region.inspectability,
            }
            return geometric_mean(tuple(components.values())), components, 0.0
        if row is None:
            return 0.0, {"target_location": location}, 0.0
        if option.action in {SemanticAction.MOVE_CLOSER, SemanticAction.NEXT_BEST_VIEW}:
            information_need = (
                0.35 * row.identity_entropy
                + 0.25 * row.resolution_uncertainty
                + 0.20 * row.occlusion_uncertainty
                + 0.20 * (1.0 - float(row.graspability_belief["GRASPABLE"]))
            )
            components = {
                "target_location": location,
                "prompt_relevance": row.prompt_relevance,
                "information_need": information_need,
            }
            return geometric_mean(tuple(components.values())), components, 0.0
        components = {
            "prompt_relevance": row.prompt_relevance,
            "occlusion_uncertainty": row.occlusion_uncertainty,
            "non_target_probability": 1.0 - float(row.identity_belief["target"]),
        }
        return geometric_mean(tuple(components.values())), components, 0.0

    def registry_valid(option: ActionEffectEvidence) -> tuple[bool, str]:
        """Apply only categorical action-schema and controller-memory constraints."""

        if option.action is SemanticAction.OPEN_CONTAINER:
            if option.target_id not in unobserved_by_id:
                return False, "target is not a registered unobserved container"
            return True, "target is a registered unobserved container"
        if option.target_id not in region_by_id:
            return False, "target is not a public visible region"
        return True, "target is a public registered region"

    ranked_options: list[dict[str, Any]] = []
    for option in assessment.interaction_options:
        legal, legality_reason = registry_valid(option)
        action_readiness, components, consequence = readiness(option)
        information_gain = max(
            0.0, task_uncertainty - option.expected_posterior_uncertainty
        )
        expected_benefit = (
            option.applicable_probability
            * option.execution_success_probability
            * (information_gain + option.expected_task_progress)
        )
        utility = (
            action_readiness * expected_benefit
            - option.normalized_cost
            - option.normalized_risk
            - consequence
        )
        if not legal:
            utility = None
        ranked_options.append(
            {
                "action": option.action.value,
                "target_id": option.target_id,
                "utility": utility,
                "legal": legal,
                "gate_reason": (
                    f"{legality_reason}; confidence enters utility continuously"
                ),
                "continuous_readiness": action_readiness,
                "readiness_components": components,
                "residual_consequence_cost": consequence,
                "expected_benefit_before_readiness": expected_benefit,
                "vlm_reason": option.reason,
            }
        )
    utility, selected = max(
        (
            (float(audit["utility"]), option)
            for option, audit in zip(
                assessment.interaction_options, ranked_options, strict=True
            )
            if audit["legal"] and audit["utility"] is not None
        ),
        default=(float("-inf"), None),
        key=lambda item: item[0],
    )
    stop_utility = -config.stop_failure_cost * task_uncertainty
    if selected is not None and utility > stop_utility:
        action = selected.action
        target_id = selected.target_id
        stop_reason = None
        reasons = (
            "selected maximum action-conditioned utility over controlled failure",
            f"utility={utility:.6f}",
            selected.reason,
        )
    else:
        action = SemanticAction.STOP
        target_id = None
        location_argmax = max(
            assessment.target_location_belief,
            key=assessment.target_location_belief.__getitem__,
        )
        stop_reason = (
            "NOT_FOUND"
            if assessment.search_domain_exhausted and location_argmax == "ABSENT"
            else "ABSTAIN"
        )
        reasons = (
            "STOP utility exceeds every registry-valid physical action",
            f"stop_utility={stop_utility:.6f}",
            f"stop_reason={stop_reason}",
        )
    return SemanticDecision(
        action=action,
        target_id=target_id,
        reasons=reasons,
        task_uncertainty=task_uncertainty,
        target_prediction_set=target_rank[:1],
        option_utilities=tuple(
            sorted(
                ranked_options,
                key=lambda row: (
                    row["utility"] is not None,
                    float(row["utility"]) if row["utility"] is not None else -math.inf,
                ),
                reverse=True,
            )
        ),
        advisory_agrees=(
            assessment.advisory_action is action
            and assessment.advisory_target_id == target_id
        ),
        stop_utility=stop_utility,
        stop_reason=stop_reason,
    )


def extract_json_object(text: str) -> Mapping[str, Any]:
    """Extract one JSON object without accepting trailing natural-language text."""

    stripped = text.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[7:-3].strip()
    elif stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped[3:-3].strip()
    decoder = json.JSONDecoder()
    value, end = decoder.raw_decode(stripped)
    if stripped[end:].strip():
        raise ValueError("VLM response contains text after the JSON object")
    if not isinstance(value, Mapping):
        raise ValueError("VLM response must be a JSON object")
    return value


def build_semantic_prompt(
    *,
    prompt: str,
    target_query: str,
    scene_objects: Sequence[Mapping[str, Any]],
    belief_history: Sequence[Mapping[str, Any]] = (),
) -> str:
    if not scene_objects:
        raise ValueError("semantic prompt requires public object proposals")
    object_table = []
    for row in scene_objects:
        object_table.append(
            {
                "object_id": row["object_id"],
                "display_id": row.get("display_id"),
                "view": row.get("view"),
                "label_candidates": row.get("label_candidates", {}),
                "bbox_xyxy": row.get("bbox_xyxy"),
                "visible_area": row.get("visible_area"),
                "mask_score": row.get("mask_score"),
                "cross_view_best_match": row.get("cross_view_best_match"),
            }
        )
    example_object_id = str(scene_objects[0]["object_id"])
    container_example = next(
        (
            str(row["object_id"])
            for row in scene_objects
            if any(
                word in str(label).lower()
                for label in row.get("label_candidates", {})
                for word in ("drawer", "cabinet", "container", "refrigerator")
            )
        ),
        example_object_id,
    )
    location_template = {
        str(row["object_id"]): 0.0 for row in scene_objects
    }
    location_template.update({"OTHER_UNSEARCHED": 1.0, "ABSENT": 0.0})
    schema = {
        "target_query": "infer the exact target noun phrase from the task prompt",
        "destination_query": "infer destination noun phrase, or null",
        "goal_relation": "infer the requested task relation/action",
        "required_facts": [
            "infer the prompt-relevant facts needed before acting"
        ],
        "target_location_belief": location_template,
        "regions": [
            {
                "object_id": example_object_id,
                "prompt_relevance": 0.0,
                "identity_belief": {
                    "target": 0.0,
                    "visually_similar_non_target": 0.0,
                    "other": 0.0,
                    "insufficient_visual_evidence": 1.0,
                },
                "graspability_belief": {
                    "GRASPABLE": 0.0,
                    "NOT_GRASPABLE": 0.0,
                    "INSUFFICIENT_EVIDENCE": 1.0,
                },
                "resolution_uncertainty": 0.0,
                "occlusion_uncertainty": 0.0,
                "state_uncertainty": 0.0,
                "move_closer_effect_probability": 0.0,
                "reason": "short visual reason",
            }
        ],
        "unobserved_regions": [
            {
                "object_id": container_example,
                "target_probability": 0.0,
                "inspectability": 0.0,
                "reason": "short reason",
            }
        ],
        "interaction_options": [
            {
                "action": "MOVE_CLOSER",
                "target_id": example_object_id,
                "applicable_probability": 0.0,
                "execution_success_probability": 0.0,
                "outcome_distribution": {
                    "FAILED": 0.1,
                    "NO_RELEVANT_CHANGE": 0.9,
                },
                "expected_posterior_uncertainty": 0.0,
                "expected_task_progress": 0.0,
                "normalized_cost": 0.0,
                "normalized_risk": 0.0,
                "semantic_subtask": "typed natural-language subtask for the registered executor",
                "reason": "short counterfactual effect reason",
            }
        ],
        "advisory_action": "STOP",
        "advisory_target_id": None,
        "summary": "one sentence",
    }
    return (
        "You are the semantic evidence module of an interactive-perception robot. "
        "Use ONLY the supplied public RGB images, proposal overlays, crop montage, "
        "object table, task prompt, and ordinary visual commonsense. Do not infer "
        "simulator state, hidden poses, semantic IDs, or task predicates.\n\n"
        f"Task prompt: {prompt}\nUntrusted lexical query hint: {target_query}\n\n"
        f"Public action/evidence history:\n{json.dumps(list(belief_history), indent=2)}\n\n"
        "The same physical proposal may occur in two camera views. Use the display "
        "IDs and cross-view matches to compare them. Infer the target, destination, goal "
        "relation, and required task facts from the complete prompt; the lexical hint is "
        "only detector support and may be incomplete. When any prompt-relevant visible "
        "candidates are semantically or visually confusable, assign substantial mass to "
        "target, visually_similar_non_target, or insufficient_visual_evidence as appropriate, "
        "and consider MOVE_CLOSER or NEXT_BEST_VIEW. "
        "Use REMOVE_OCCLUDER when a movable object blocks prompt-relevant evidence. If "
        "the target is absent from view but a closed or unobserved drawer may contain it, "
        "put that drawer in unobserved_regions and propose OPEN_CONTAINER. ACT is advisory "
        "only for one clear, readable, graspable target. STOP is only an advisory fallback; "
        "the deterministic controller assigns its NOT_FOUND or ABSTAIN reason.\n\n"
        "Treat the following as one optimization problem. For every useful legal action/"
        "target pair, predict its effect-outcome distribution, execution success, expected "
        "posterior task uncertainty, task progress, normalized cost, and risk. Include "
        "competing actions rather than only the preferred one. Allowed action meanings:\n"
        "- MOVE_CLOSER: translate the wrist camera toward a visible ambiguous region.\n"
        "- NEXT_BEST_VIEW: move the camera to a different view without manipulating objects.\n"
        "- REMOVE_OCCLUDER: manipulate a movable blocker to expose a relevant region.\n"
        "- OPEN_CONTAINER: open an unobserved container to reveal or certify its interior.\n"
        "- ACT: hand the original task to the frozen manipulation policy.\n\n"
        f"Allowed effect outcome keys are exactly: {', '.join(EFFECT_OUTCOMES)}.\n\n"
        f"Public object table:\n{json.dumps(object_table, indent=2)}\n\n"
        "The target_location_belief is one normalized distribution over exact public "
        "object IDs plus OTHER_UNSEARCHED and ABSENT. When history has resolved a "
        "visible candidate as a non-target, give it low location probability and low "
        "identity entropy; redistribute rather than delete belief mass. Return exactly "
        "one JSON object, no markdown and no prose outside JSON. "
        "Every region object_id must be copied exactly from the table. Include every "
        "prompt-relevant or visually target-like proposal, but omit clearly unrelated "
        "duplicate/background proposals. Probabilities must lie in [0,1], and each "
        "identity_belief, graspability_belief, target_location_belief, and each effect "
        "outcome_distribution must sum to one. The template contains every allowed "
        "location key; keep those keys and redistribute their probabilities. Duplicate "
        "the example region/action rows as needed using exact IDs and allowed enum values. "
        "Follow this schema:\n"
        f"{json.dumps(schema, indent=2)}"
    )


def build_action_effect_prompt(
    *,
    prompt: str,
    pre_vlm_field: Mapping[str, Any],
    scene_objects: Sequence[Mapping[str, Any]],
    registered_candidates: Sequence[Mapping[str, Any]],
    belief_history: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Prompt Qwen only for action-conditioned future effects."""

    if not scene_objects or len(registered_candidates) != 1:
        raise ValueError("effect prompt requires public proposals and exactly one candidate")
    fixed_candidate = dict(registered_candidates[0])
    object_table = [
        {
            "object_id": row["object_id"],
            "display_id": row.get("display_id"),
            "view": row.get("view"),
            "label_candidates": row.get("label_candidates", {}),
            "bbox_xyxy": row.get("bbox_xyxy"),
            "visible_area": row.get("visible_area"),
            "cross_view_best_match": row.get("cross_view_best_match"),
        }
        for row in scene_objects
    ]
    compact_field = {
        "task_spec": pre_vlm_field["task_spec"],
        "task_uncertainty": pre_vlm_field["task_uncertainty"],
        "target_location_belief": pre_vlm_field["target_location_belief"],
        "regions": [
            {
                key: row.get(key)
                for key in (
                    "object_id",
                    "view",
                    "prompt_relevance",
                    "identity_belief",
                    "identity_entropy",
                    "graspability_belief",
                    "resolution_uncertainty",
                    "occlusion_uncertainty",
                    "state_uncertainty",
                    "closed_container_probability",
                    "uncertainty_mass",
                )
            }
            for row in pre_vlm_field.get("regions", ())
        ],
        "unobserved_regions": pre_vlm_field.get("unobserved_regions", ()),
        "calibration_status": pre_vlm_field.get("calibration_status"),
    }
    schema = {
        "effect": {
            "likely_outcome": "one allowed outcome key",
            "uncertainty_change": "LARGE_DECREASE | MODERATE_DECREASE | SMALL_DECREASE | NO_CHANGE | INCREASE",
            "task_progress": "NONE | INDIRECT | DIRECT | COMPLETE",
            "semantic_subtask": "specific instruction for the fixed registered executor",
            "reason": "short visual and task-grounded reason about the fixed pair",
        },
        "summary": "one sentence about the fixed candidate's predicted effect",
    }
    return (
        "You are the counterfactual action-effect module of an interactive-perception "
        "robot. The CURRENT task belief and uncertainty were already computed by a "
        "frozen vision/prompt frontend before you. Never recompute, revise, renormalize, "
        "or replace that current belief. Your only job is to predict what each registered "
        "candidate action is likely to reveal or change, using public RGB, the supplied "
        "field, action history, and ordinary physical/semantic commonsense. Never infer "
        "simulator state, hidden poses, semantic IDs, joints, or task predicates.\n\n"
        f"Original task prompt: {prompt}\n\n"
        f"Frozen current field:\n{json.dumps(compact_field, indent=2)}\n\n"
        f"Public object table:\n{json.dumps(object_table, indent=2)}\n\n"
        f"Public history:\n{json.dumps(list(belief_history), indent=2)}\n\n"
        f"Fixed registered candidate (INPUT, not a choice):\n"
        f"{json.dumps(fixed_candidate, indent=2)}\n\n"
        "Do not select or propose an action. Assume the exact fixed action/target pair "
        "above is executed, and predict only its most likely outcome, qualitative "
        "uncertainty change, task progress, and a concise "
        "semantic subtask. Both revealing the target and certifying a local "
        "region empty may reduce uncertainty. A failed action must not reduce belief. "
        "Rank actions by prompt-relevant information utility, not only immediate task "
        "completion. Inspecting an opaque closed container requires opening/removing its "
        "barrier; merely moving a camera closer to its exterior cannot reveal the interior. "
        "Opening a registered unobserved container can strongly reduce location uncertainty "
        "whether the result reveals the target or certifies the local interior empty. "
        "Task uncertainty is multi-fact: target identity, visibility, resolution, and "
        "graspability still matter after rough location is known. A target-like proposal "
        "that is small, clipped by an image border, high-entropy, or not yet graspability-"
        "singleton can benefit from MOVE_CLOSER or NEXT_BEST_VIEW even when its location "
        "is already plausible. Judge the supplied candidate itself rather than replacing "
        "it with ACT. "
        "For ACT, semantic_subtask must preserve the original goal while grounding the "
        "target with visible color/packaging attributes and a current spatial relation "
        "such as 'inside the open middle drawer' or 'directly in front of the wrist "
        "camera'. Use ordinary object words, never proposal IDs. Keep it one executable "
        "imperative sentence aligned with the frozen policy's pick/place vocabulary. "
        "MOVE_CLOSER/NEXT_BEST_VIEW are camera observations, REMOVE_OCCLUDER and "
        "OPEN_CONTAINER alter visibility, and ACT delegates the original task to the frozen "
        "policy. STOP is evaluated once by the deterministic utility selector, not by this "
        "per-physical-action effect prediction. "
        f"Allowed outcome keys are exactly: {', '.join(EFFECT_OUTCOMES)}. Return exactly "
        "one JSON object, without action or target fields, markdown, or extra prose, using "
        "this complete schema:\n"
        f"{json.dumps(schema, indent=2)}"
    )


class Qwen25VLSemanticReasoner:
    """Local sequential Qwen2.5-VL backend; no network access at inference time."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cuda",
        max_new_tokens: int = 2400,
        min_pixels: int = 128 * 28 * 28,
        max_pixels: int = 256 * 28 * 28,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        ).to(device)
        self.model.eval()
        # transformers 4.53 computes vocabulary logits for every prompt token
        # even though greedy generation consumes only the final position.  With
        # multi-image prompts this transient tensor alone can exceed 6 GiB.
        # Restricting the LM head to the last position is generation-equivalent.
        original_lm_head_forward = self.model.lm_head.forward

        def last_position_lm_head(hidden_states: Any) -> Any:
            return original_lm_head_forward(hidden_states[..., -1:, :])

        self.model.lm_head.forward = last_position_lm_head
        self.model_stamp = f"qwen2.5-vl:{model_path.name}"
        self.failed_attempts: list[dict[str, str]] = []

    def _generate(self, *, messages: Sequence[Mapping[str, Any]], images: Sequence[Any]) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=list(images), padding=True, return_tensors="pt"
        ).to(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = generated[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def assess(
        self,
        *,
        images: Sequence[Any],
        prompt: str,
        target_query: str,
        scene_objects: Sequence[Mapping[str, Any]],
        belief_history: Sequence[Mapping[str, Any]] = (),
    ) -> tuple[SemanticAssessment, str]:
        semantic_prompt = build_semantic_prompt(
            prompt=prompt,
            target_query=target_query,
            scene_objects=scene_objects,
            belief_history=belief_history,
        )
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": semantic_prompt})
        messages = [
            {
                "role": "system",
                "content": "Output valid JSON only. Never invent object IDs or privileged state.",
            },
            {"role": "user", "content": content},
        ]
        errors: list[str] = []
        self.failed_attempts = []
        raw = ""
        for _ in range(3):
            raw = self._generate(messages=messages, images=images)
            try:
                assessment = SemanticAssessment.from_mapping(
                    extract_json_object(raw),
                    allowed_object_ids=[str(row["object_id"]) for row in scene_objects],
                )
                return assessment, raw
            except (KeyError, TypeError, ValueError) as error:
                errors.append(str(error))
                self.failed_attempts.append({"error": str(error), "raw": raw})
                messages = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Your JSON failed schema validation: "
                            f"{error}. Return a corrected complete JSON object only."
                        ),
                    },
                ]
        raise ValueError(f"Qwen failed structured validation after 3 attempts: {errors}")

    def assess_effects(
        self,
        *,
        images: Sequence[Any],
        prompt: str,
        pre_vlm_field: Mapping[str, Any],
        scene_objects: Sequence[Mapping[str, Any]],
        registered_candidates: Sequence[Mapping[str, Any]],
        belief_history: Sequence[Mapping[str, Any]] = (),
    ) -> tuple[ActionEffectAssessment, str]:
        errors: list[str] = []
        self.failed_attempts = []
        raw_by_candidate = []
        options = []
        allowed_ids = [str(row["object_id"]) for row in scene_objects]
        for candidate in registered_candidates:
            effect_prompt = build_action_effect_prompt(
                prompt=prompt,
                pre_vlm_field=pre_vlm_field,
                scene_objects=scene_objects,
                registered_candidates=[candidate],
                belief_history=belief_history,
            )
            content = [{"type": "image"} for _ in images]
            content.append({"type": "text", "text": effect_prompt})
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Output valid JSON only. Analyze the one registered candidate "
                        "exactly; do not substitute a different action or target."
                    ),
                },
                {"role": "user", "content": content},
            ]
            accepted = None
            raw = ""
            for _ in range(2):
                raw = self._generate(messages=messages, images=images)
                try:
                    parsed = extract_json_object(raw)
                    effect = parsed.get("effect")
                    if not isinstance(effect, Mapping):
                        raise ValueError("effect must be one JSON object")
                    # The action/target pair is an input supplied by the typed
                    # registry.  Qwen predicts its effect and cannot rewrite the
                    # executable pair through generated text.
                    normalized = {
                        "ranked_candidates": [
                            {
                                **effect,
                                "action": str(candidate["action"]),
                                "target_id": str(candidate["target_id"]),
                            }
                        ],
                        "summary": parsed.get("summary"),
                    }
                    accepted = ActionEffectAssessment.from_ranked_mapping(
                        normalized,
                        allowed_object_ids=allowed_ids,
                        registered_candidates=[candidate],
                        current_uncertainty=float(pre_vlm_field["task_uncertainty"]),
                        target_location_belief=pre_vlm_field[
                            "target_location_belief"
                        ],
                    )
                    break
                except (KeyError, TypeError, ValueError) as error:
                    errors.append(str(error))
                    self.failed_attempts.append(
                        {
                            "candidate": json.dumps(candidate, sort_keys=True),
                            "error": str(error),
                            "raw": raw,
                        }
                    )
                    messages = [
                        *messages,
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "Your fixed-candidate effect failed schema validation. "
                                f"Validation error: {error}. Return exactly the requested "
                                "effect and summary fields; do not output action or target."
                            ),
                        },
                    ]
            if accepted is None:
                action = SemanticAction(str(candidate["action"]))
                fallback = ActionEffectEvidence(
                    action=action,
                    target_id=str(candidate["target_id"]),
                    applicable_probability=0.0,
                    execution_success_probability=float(
                        candidate["execution_success_prior"]
                    ),
                    outcome_distribution={"FAILED": 0.80, "NO_RELEVANT_CHANGE": 0.20},
                    expected_posterior_uncertainty=float(
                        pre_vlm_field["task_uncertainty"]
                    ),
                    expected_task_progress=0.0,
                    normalized_cost=float(candidate["normalized_cost_prior"]),
                    normalized_risk=float(candidate["normalized_risk_prior"]),
                    semantic_subtask=str(candidate["semantic_subtask_hint"]),
                    reason=(
                        "VLM failed the fixed-candidate effect schema twice; candidate "
                        "conservatively marked inapplicable"
                    ),
                )
                accepted = ActionEffectAssessment(
                    interaction_options=(fallback,),
                    advisory_action=SemanticAction.STOP,
                    advisory_target_id=None,
                    summary="invalid per-candidate response conservatively rejected",
                )
            options.extend(accepted.interaction_options)
            try:
                recorded_response: Any = extract_json_object(raw)
            except (TypeError, ValueError):
                recorded_response = {"invalid_raw_response": raw}
            raw_by_candidate.append(
                {
                    "candidate": dict(candidate),
                    "response": recorded_response,
                    "accepted": accepted.interaction_options[0].to_dict(),
                }
            )
        current = float(pre_vlm_field["task_uncertainty"])
        advisory = max(options, key=lambda row: row.utility(current))
        assessment = ActionEffectAssessment(
            interaction_options=tuple(options),
            advisory_action=advisory.action,
            advisory_target_id=advisory.target_id,
            summary="Qwen effects were inferred independently per registered candidate.",
        )
        return assessment, json.dumps(
            {"per_candidate_inference": raw_by_candidate}, indent=2
        )
