"""Finite-horizon Bayes-risk planning for active perception.

The planner makes the two mechanisms of active perception explicit: an
information action changes what can be observed, and its outcome branches the
next decision.  No confidence threshold is used.  Every numeric preference is
supplied as a measured action effect, an action cost, or an explicit loss.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from functools import lru_cache
from collections.abc import Mapping
from typing import Any

import numpy as np

from .temporal_belief import (
    TemporalActionRole,
    TemporalBelief,
)

ACT = "ACT"
NOT_FOUND = "NOT_FOUND"
SAFE_STOP = "SAFE_STOP"


class TargetState(str, Enum):
    OBSERVED = "OBSERVED"
    VIEWPOINT_BLOCKED = "VIEWPOINT_BLOCKED"
    MANIPULATION_ONLY = "MANIPULATION_ONLY"
    ABSENT = "ABSENT"


class EffectOutcome(str, Enum):
    FAILED = "FAILED"
    REVEALED = "REVEALED"
    EMPTY = "EMPTY"


@dataclasses.dataclass(frozen=True)
class TargetHypothesis:
    label: str
    state: TargetState
    resolving_action: str | None = None

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("hypothesis label is required")
        if self.state in {
            TargetState.VIEWPOINT_BLOCKED,
            TargetState.MANIPULATION_ONLY,
        } and not self.resolving_action:
            raise ValueError("a hidden hypothesis requires its resolving action")
        if self.state in {TargetState.OBSERVED, TargetState.ABSENT} and self.resolving_action:
            raise ValueError("terminal hypotheses cannot declare a resolving action")


@dataclasses.dataclass(frozen=True)
class TargetBelief:
    hypotheses: tuple[TargetHypothesis, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.hypotheses) < 1 or len(self.hypotheses) != len(self.probabilities):
            raise ValueError("belief requires at least one aligned hypothesis")
        labels = [item.label for item in self.hypotheses]
        if len(labels) != len(set(labels)):
            raise ValueError("hypothesis labels must be unique")
        values = np.asarray(self.probabilities, dtype=np.float64)
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-9)
        ):
            raise ValueError("belief probabilities must form a distribution")

    def mass(self, state: TargetState) -> float:
        return float(
            sum(
                probability
                for hypothesis, probability in zip(
                    self.hypotheses, self.probabilities, strict=True
                )
                if hypothesis.state is state
            )
        )

    def one_hot(self, label: str) -> "TargetBelief":
        if label not in {item.label for item in self.hypotheses}:
            raise ValueError(f"unknown plausible label {label!r}")
        return TargetBelief(
            hypotheses=self.hypotheses,
            probabilities=tuple(float(item.label == label) for item in self.hypotheses),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypotheses": [
                {
                    "label": item.label,
                    "state": item.state.value,
                    "resolving_action": item.resolving_action,
                }
                for item in self.hypotheses
            ],
            "probabilities": list(self.probabilities),
        }


@dataclasses.dataclass(frozen=True)
class ActionEffect:
    action: str
    resolves: tuple[str, ...]
    reliability: float
    cost: float
    source: str

    def __post_init__(self) -> None:
        if not self.action or self.action in {ACT, NOT_FOUND, SAFE_STOP}:
            raise ValueError("information action must be non-terminal")
        if not self.resolves or len(self.resolves) != len(set(self.resolves)):
            raise ValueError("effect must resolve unique hypothesis labels")
        if not 0.0 <= self.reliability <= 1.0 or not np.isfinite(self.reliability):
            raise ValueError("effect reliability must lie in [0, 1]")
        if self.cost < 0.0 or not np.isfinite(self.cost):
            raise ValueError("effect cost must be finite and non-negative")
        if not self.source:
            raise ValueError("effect source is required")


@dataclasses.dataclass(frozen=True)
class DecisionLosses:
    false_commit: float
    false_absent: float
    act_execution_failure: float
    safe_stop: float | None = None

    def __post_init__(self) -> None:
        values = dataclasses.astuple(self)[:3]
        if any(value < 0.0 or not np.isfinite(value) for value in values):
            raise ValueError("decision losses must be finite and non-negative")
        if not any(value > 0.0 for value in values):
            raise ValueError("at least one decision loss must be positive")
        if self.safe_stop is not None and (
            self.safe_stop < 0.0 or not np.isfinite(self.safe_stop)
        ):
            raise ValueError("safe_stop must be finite and non-negative when set")


@dataclasses.dataclass(frozen=True)
class ActionValue:
    action: str
    objective: float
    branches: dict[str, float]
    temporal_violation_probability: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class RiskDecision:
    selected_action: str
    ranking: tuple[ActionValue, ...]
    stop_objective: float
    resolvable_uncertainty: dict[str, float]
    margin: float
    unstable_tie: bool
    horizon: int
    temporal_belief: dict[str, Any] | None = None
    temporal_violation_loss: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_action": self.selected_action,
            "ranking": [item.to_dict() for item in self.ranking],
            "stop_objective": self.stop_objective,
            "resolvable_uncertainty": dict(self.resolvable_uncertainty),
            "margin": self.margin,
            "unstable_tie": self.unstable_tie,
            "horizon": self.horizon,
            "temporal_belief": self.temporal_belief,
            "temporal_violation_loss": self.temporal_violation_loss,
        }


def update_from_effect(
    belief: TargetBelief, effect: ActionEffect, outcome: EffectOutcome
) -> TargetBelief:
    """Exact posterior update for an observed physical action outcome."""

    if outcome is EffectOutcome.FAILED:
        return belief
    resolved = set(effect.resolves)
    keep_resolved = outcome is EffectOutcome.REVEALED
    selected = [
        (hypothesis, probability)
        for hypothesis, probability in zip(
            belief.hypotheses, belief.probabilities, strict=True
        )
        if probability > 0.0 and (hypothesis.label in resolved) is keep_resolved
    ]
    total = sum(probability for _, probability in selected)
    if total <= 0.0:
        raise ValueError(f"outcome {outcome.value} has zero probability under the belief")
    hypotheses = tuple(
        TargetHypothesis(hypothesis.label, TargetState.OBSERVED)
        if outcome is EffectOutcome.REVEALED
        else hypothesis
        for hypothesis, _ in selected
    )
    probabilities = tuple(probability / total for _, probability in selected)
    return TargetBelief(hypotheses, probabilities)


class ExpectedRiskPlanner:
    def __init__(
        self,
        *,
        losses: DecisionLosses,
        act_reliability: float | Mapping[str, float],
        horizon: int,
        tie_tolerance: float = 1e-9,
        temporal_violation_loss: float = 0.0,
    ) -> None:
        if isinstance(act_reliability, Mapping):
            values = {str(label): float(value) for label, value in act_reliability.items()}
            if not values or any(not 0.0 <= value <= 1.0 for value in values.values()):
                raise ValueError("mapped act reliabilities must lie in [0, 1]")
            self.act_reliability: float | dict[str, float] = values
        else:
            if not 0.0 <= act_reliability <= 1.0:
                raise ValueError("act_reliability must lie in [0, 1]")
            self.act_reliability = float(act_reliability)
        if horizon < 0:
            raise ValueError("horizon must be non-negative")
        if tie_tolerance < 0.0 or not np.isfinite(tie_tolerance):
            raise ValueError("tie_tolerance must be finite and non-negative")
        if temporal_violation_loss < 0.0 or not np.isfinite(
            temporal_violation_loss
        ):
            raise ValueError("temporal_violation_loss must be finite and non-negative")
        self.losses = losses
        self.horizon = int(horizon)
        self.tie_tolerance = float(tie_tolerance)
        self.temporal_violation_loss = float(temporal_violation_loss)

    def _act_reliability(self, label: str) -> float:
        if isinstance(self.act_reliability, dict):
            if label not in self.act_reliability:
                raise KeyError(f"missing ACT reliability for hypothesis {label!r}")
            return self.act_reliability[label]
        return self.act_reliability

    def _terminal_values(
        self, belief: TargetBelief, temporal: TemporalBelief | None
    ) -> dict[str, ActionValue]:
        absent = belief.mass(TargetState.ABSENT)
        act_risk = sum(
            probability
            * (
                (1.0 - self._act_reliability(hypothesis.label))
                * self.losses.act_execution_failure
                if hypothesis.state is TargetState.OBSERVED
                else self.losses.false_commit
            )
            for hypothesis, probability in zip(
                belief.hypotheses, belief.probabilities, strict=True
            )
            if probability > 0.0
        )
        absent_risk = (1.0 - absent) * self.losses.false_absent
        act_violation = (
            temporal.violation_probability(TemporalActionRole.COMMIT)
            if temporal is not None
            else 0.0
        )
        absent_violation = (
            temporal.violation_probability(TemporalActionRole.NOT_FOUND)
            if temporal is not None
            else 0.0
        )
        act_objective = act_risk + self.temporal_violation_loss * act_violation
        absent_objective = (
            absent_risk + self.temporal_violation_loss * absent_violation
        )
        values = {
            ACT: ActionValue(
                ACT,
                act_objective,
                {"terminal": act_risk},
                act_violation,
            ),
            NOT_FOUND: ActionValue(
                NOT_FOUND,
                absent_objective,
                {"terminal": absent_risk},
                absent_violation,
            ),
        }
        if self.losses.safe_stop is not None:
            values[SAFE_STOP] = ActionValue(
                SAFE_STOP,
                self.losses.safe_stop,
                {"defer": self.losses.safe_stop},
            )
        return values

    def plan(
        self,
        belief: TargetBelief,
        effects: tuple[ActionEffect, ...],
        *,
        temporal_belief: TemporalBelief | None = None,
    ) -> RiskDecision:
        actions = [effect.action for effect in effects]
        if len(actions) != len(set(actions)):
            raise ValueError("information actions must be unique")
        known = {item.label for item in belief.hypotheses}
        if any(set(effect.resolves) - known for effect in effects):
            raise ValueError("effect resolves labels outside the belief")
        if self.temporal_violation_loss > 0.0 and temporal_belief is None:
            raise ValueError(
                "temporal_belief is required when temporal_violation_loss is positive"
            )

        @lru_cache(maxsize=None)
        def value(
            hypotheses: tuple[TargetHypothesis, ...],
            probabilities: tuple[float, ...],
            available: tuple[ActionEffect, ...],
            depth: int,
            temporal_probabilities: tuple[float, ...],
        ) -> tuple[float, dict[str, ActionValue]]:
            current = TargetBelief(hypotheses, probabilities)
            current_temporal = (
                TemporalBelief(temporal_probabilities)
                if temporal_probabilities
                else None
            )
            candidates = self._terminal_values(current, current_temporal)
            if depth > 0:
                by_label = {
                    item.label: probability
                    for item, probability in zip(
                        current.hypotheses, current.probabilities, strict=True
                    )
                }
                for effect in available:
                    resolved_mass = sum(by_label.get(label, 0.0) for label in effect.resolves)
                    fail_value, _ = value(
                        current.hypotheses,
                        current.probabilities,
                        available,
                        depth - 1,
                        (
                            current_temporal.resolve_information(
                                EffectOutcome.FAILED.value,
                                search_exhausted=False,
                            ).probabilities
                            if current_temporal is not None
                            else ()
                        ),
                    )
                    remaining = tuple(item for item in available if item != effect)
                    reveal_value = 0.0
                    if resolved_mass > 1e-12:
                        reveal_belief = update_from_effect(
                            current, effect, EffectOutcome.REVEALED
                        )
                        reveal_value, _ = value(
                            reveal_belief.hypotheses,
                            reveal_belief.probabilities,
                            remaining,
                            depth - 1,
                            (
                                current_temporal.resolve_information(
                                    EffectOutcome.REVEALED.value,
                                    search_exhausted=False,
                                ).probabilities
                                if current_temporal is not None
                                else ()
                            ),
                        )
                    empty_mass = 1.0 - resolved_mass
                    empty_value = 0.0
                    if empty_mass > 1e-12:
                        empty_belief = update_from_effect(
                            current, effect, EffectOutcome.EMPTY
                        )
                        empty_value, _ = value(
                            empty_belief.hypotheses,
                            empty_belief.probabilities,
                            remaining,
                            depth - 1,
                            (
                                current_temporal.resolve_information(
                                    EffectOutcome.EMPTY.value,
                                    search_exhausted=all(
                                        hypothesis.state is TargetState.ABSENT
                                        for hypothesis in empty_belief.hypotheses
                                    ),
                                ).probabilities
                                if current_temporal is not None
                                else ()
                            ),
                        )
                    information_violation = (
                        current_temporal.violation_probability(
                            TemporalActionRole.INFORMATION
                        )
                        if current_temporal is not None
                        else 0.0
                    )
                    objective = (
                        effect.cost
                        + self.temporal_violation_loss * information_violation
                        + (
                        (1.0 - effect.reliability) * fail_value
                        + effect.reliability
                        * (
                            resolved_mass * reveal_value
                            + empty_mass * empty_value
                        )
                        )
                    )
                    candidates[effect.action] = ActionValue(
                        action=effect.action,
                        objective=objective,
                        branches={
                            EffectOutcome.FAILED.value: 1.0 - effect.reliability,
                            EffectOutcome.REVEALED.value: effect.reliability
                            * resolved_mass,
                            EffectOutcome.EMPTY.value: effect.reliability * empty_mass,
                        },
                        temporal_violation_probability=information_violation,
                    )
            return min(item.objective for item in candidates.values()), candidates

        _, values = value(
            belief.hypotheses,
            belief.probabilities,
            effects,
            self.horizon,
            temporal_belief.probabilities if temporal_belief is not None else (),
        )
        ranking = tuple(sorted(values.values(), key=lambda item: (item.objective, item.action)))
        terminal_actions = {ACT, NOT_FOUND, SAFE_STOP}
        stop_objective = min(
            item.objective
            for action, item in values.items()
            if action in terminal_actions
        )
        resolvable_uncertainty = {
            action: stop_objective - values[action].objective for action in actions
        }
        margin = float("inf") if len(ranking) == 1 else ranking[1].objective - ranking[0].objective
        return RiskDecision(
            selected_action=ranking[0].action,
            ranking=ranking,
            stop_objective=stop_objective,
            resolvable_uncertainty=resolvable_uncertainty,
            margin=margin,
            unstable_tie=len(ranking) > 1 and margin <= self.tie_tolerance,
            horizon=self.horizon,
            temporal_belief=(
                temporal_belief.to_dict() if temporal_belief is not None else None
            ),
            temporal_violation_loss=self.temporal_violation_loss,
        )

    def robust_plan(
        self,
        belief: TargetBelief,
        effects: tuple[ActionEffect, ...],
        *,
        conformal_labels: tuple[str, ...],
        temporal_belief: TemporalBelief | None = None,
    ) -> RiskDecision:
        """Minimize worst-case risk over hypotheses retained by conformal prediction."""

        if not conformal_labels:
            raise ValueError("conformal_labels cannot be empty")
        scenario_decisions = [
            self.plan(
                belief.one_hot(label),
                effects,
                temporal_belief=temporal_belief,
            )
            for label in conformal_labels
        ]
        actions = {value.action for decision in scenario_decisions for value in decision.ranking}
        robust_values = []
        for action in actions:
            scenario_values = [
                next(value for value in decision.ranking if value.action == action)
                for decision in scenario_decisions
            ]
            worst = max(value.objective for value in scenario_values)
            robust_values.append(
                ActionValue(
                    action=action,
                    objective=worst,
                    branches={
                        label: value.objective
                        for label, value in zip(
                            conformal_labels, scenario_values, strict=True
                        )
                    },
                    temporal_violation_probability=max(
                        value.temporal_violation_probability
                        for value in scenario_values
                    ),
                )
            )
        ranking = tuple(
            sorted(robust_values, key=lambda item: (item.objective, item.action))
        )
        by_action = {item.action: item for item in robust_values}
        stop_objective = min(
            item.objective
            for action, item in by_action.items()
            if action in {ACT, NOT_FOUND, SAFE_STOP}
        )
        resolvable_uncertainty = {
            effect.action: stop_objective - by_action[effect.action].objective
            for effect in effects
        }
        margin = float("inf") if len(ranking) == 1 else ranking[1].objective - ranking[0].objective
        return RiskDecision(
            selected_action=ranking[0].action,
            ranking=ranking,
            stop_objective=stop_objective,
            resolvable_uncertainty=resolvable_uncertainty,
            margin=margin,
            unstable_tie=len(ranking) > 1 and margin <= self.tie_tolerance,
            horizon=self.horizon,
            temporal_belief=(
                temporal_belief.to_dict() if temporal_belief is not None else None
            ),
            temporal_violation_loss=self.temporal_violation_loss,
        )
