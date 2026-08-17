"""Stateful prompt-aligned active-perception pipeline.

This module is deliberately independent of LIBERO and of any particular
perception network.  Calibrated heads construct a :class:`TargetBelief` and a
conformal plausible set; capability-gated option executors provide
``ActionEffect`` objects.  The class below owns the part that is the method:
plan in the product of world and temporal beliefs, consume a public action
outcome, update both beliefs, and plan again.

No simulator-private signal is accepted by this interface.  In particular,
drawer joints, segmentation, hidden poses, and task predicates belong only in
offline label construction and evaluation.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from .active_risk import (
    ACT,
    NOT_FOUND,
    SAFE_STOP,
    ActionEffect,
    EffectOutcome,
    ExpectedRiskPlanner,
    RiskDecision,
    TargetBelief,
    TargetState,
    update_from_effect,
)
from .temporal_belief import TemporalBelief, TemporalPhase


@dataclasses.dataclass(frozen=True)
class PipelineEvent:
    """One auditable decision or observation in an episode trace."""

    index: int
    kind: str
    action: str | None
    prediction_set: tuple[str, ...]
    world_belief: dict[str, Any]
    temporal_belief: dict[str, Any]
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class OutcomeUpdate:
    """Result of consuming one conformal action-outcome set."""

    accepted_outcome: EffectOutcome | None
    must_safe_stop: bool
    reason: str
    exhausted_search: bool
    world_belief: TargetBelief
    temporal_belief: TemporalBelief


class PromptAlignedPipeline:
    """Pure closed-loop controller over calibrated public beliefs.

    The class never turns a multi-label outcome set into a point prediction.
    Such a set means the physical result is unresolved, so the only valid
    online transition is an explicit ``SAFE_STOP``.  This is intentionally
    conservative: conformal coverage would be meaningless if the controller
    silently selected the most convenient member of the set.
    """

    def __init__(
        self,
        *,
        planner: ExpectedRiskPlanner,
        belief: TargetBelief,
        conformal_labels: Sequence[str],
        temporal_belief: TemporalBelief,
        effects: Sequence[ActionEffect],
        maximum_attempts_per_action: int = 1,
    ) -> None:
        labels = tuple(str(item) for item in conformal_labels)
        known = {item.label for item in belief.hypotheses}
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("conformal_labels must be unique and non-empty")
        if set(labels) - known:
            raise ValueError("conformal_labels must name current belief hypotheses")
        actions = [effect.action for effect in effects]
        if len(actions) != len(set(actions)):
            raise ValueError("effects must have unique action names")
        if maximum_attempts_per_action < 1:
            raise ValueError("maximum_attempts_per_action must be positive")

        self.planner = planner
        self.belief = belief
        self.conformal_labels = labels
        self.temporal_belief = temporal_belief
        self.effects = {effect.action: effect for effect in effects}
        self.maximum_attempts_per_action = int(maximum_attempts_per_action)
        self.attempts = {action: 0 for action in self.effects}
        self.pending_action: str | None = None
        self.terminal: str | None = None
        self.events: list[PipelineEvent] = []

    def _available_effects(self) -> tuple[ActionEffect, ...]:
        return tuple(
            effect
            for action, effect in self.effects.items()
            if self.attempts[action] < self.maximum_attempts_per_action
            and any(
                hypothesis.label in effect.resolves
                and hypothesis.state
                in {TargetState.VIEWPOINT_BLOCKED, TargetState.MANIPULATION_ONLY}
                for hypothesis in self.belief.hypotheses
            )
        )

    def _record(
        self,
        *,
        kind: str,
        action: str | None,
        prediction_set: Sequence[str] = (),
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            PipelineEvent(
                index=len(self.events),
                kind=kind,
                action=action,
                prediction_set=tuple(str(item) for item in prediction_set),
                world_belief=self.belief.to_dict(),
                temporal_belief=self.temporal_belief.to_dict(),
                detail=detail or {},
            )
        )

    def plan(self) -> RiskDecision:
        """Choose a terminal or information action without a confidence trigger."""

        if self.terminal is not None:
            raise RuntimeError(f"episode is already terminal: {self.terminal}")
        if self.pending_action is not None:
            raise RuntimeError(
                f"action {self.pending_action!r} is awaiting a public outcome"
            )
        decision = self.planner.robust_plan(
            self.belief,
            self._available_effects(),
            conformal_labels=self.conformal_labels,
            temporal_belief=self.temporal_belief,
        )
        self._record(
            kind="DECISION",
            action=decision.selected_action,
            prediction_set=self.conformal_labels,
            detail={"risk": decision.to_dict()},
        )
        return decision

    def begin(self, decision: RiskDecision) -> None:
        """Record that the selected action has actually been dispatched."""

        if self.terminal is not None or self.pending_action is not None:
            raise RuntimeError("cannot begin another action in the current state")
        action = decision.selected_action
        if action in self.effects:
            if action not in {item.action for item in self._available_effects()}:
                raise ValueError(f"information action {action!r} is not available")
            self.attempts[action] += 1
            self.pending_action = action
            self._record(kind="ACTION_STARTED", action=action)
            return
        if action not in {ACT, NOT_FOUND, SAFE_STOP}:
            raise ValueError(f"planner selected an unknown action {action!r}")
        self.terminal = action
        self._record(kind="TERMINAL", action=action)

    def observe_information_outcome(
        self, prediction_set: Sequence[str]
    ) -> OutcomeUpdate:
        """Update the product belief from a conformal public outcome set.

        Only singleton sets advance the belief.  An empty or ambiguous set is
        retained in the trace and forces explicit deferral; it is never mapped
        to ``EMPTY`` or ``REVEALED`` by a confidence threshold.
        """

        if self.pending_action is None:
            raise RuntimeError("no information action is awaiting an outcome")
        labels = tuple(str(item) for item in prediction_set)
        allowed = {item.value for item in EffectOutcome}
        if not labels or len(labels) != len(set(labels)) or set(labels) - allowed:
            raise ValueError(
                "prediction_set must contain unique FAILED/REVEALED/EMPTY labels"
            )
        action = self.pending_action
        effect = self.effects[action]
        self.pending_action = None

        if len(labels) != 1:
            reason = "action outcome is not singleton under the frozen conformal critic"
            self.terminal = SAFE_STOP
            self._record(
                kind="AMBIGUOUS_OUTCOME_SAFE_STOP",
                action=action,
                prediction_set=labels,
                detail={"reason": reason},
            )
            return OutcomeUpdate(
                accepted_outcome=None,
                must_safe_stop=True,
                reason=reason,
                exhausted_search=False,
                world_belief=self.belief,
                temporal_belief=self.temporal_belief,
            )

        outcome = EffectOutcome(labels[0])
        try:
            self.belief = update_from_effect(self.belief, effect, outcome)
        except ValueError as error:
            # The observed branch can contradict every retained world
            # hypothesis (for example, a calibrated drawer prior says the
            # target is inside but the opened drawer is visibly empty).  That
            # is model mismatch, not permission to invent ABSENT mass.
            reason = f"observed outcome contradicts the retained world belief: {error}"
            self.terminal = SAFE_STOP
            self._record(
                kind="BELIEF_CONTRADICTION_SAFE_STOP",
                action=action,
                prediction_set=labels,
                detail={"reason": reason},
            )
            return OutcomeUpdate(
                accepted_outcome=outcome,
                must_safe_stop=True,
                reason=reason,
                exhausted_search=False,
                world_belief=self.belief,
                temporal_belief=self.temporal_belief,
            )
        remaining_hidden = any(
            hypothesis.state
            in {TargetState.VIEWPOINT_BLOCKED, TargetState.MANIPULATION_ONLY}
            for hypothesis in self.belief.hypotheses
        )
        exhausted_search = (
            outcome is EffectOutcome.EMPTY
            and not remaining_hidden
            and self.belief.mass(TargetState.ABSENT) == 1.0
        )
        self.temporal_belief = self.temporal_belief.resolve_information(
            outcome.value,
            search_exhausted=exhausted_search,
        )
        known = {item.label for item in self.belief.hypotheses}
        self.conformal_labels = tuple(
            label for label in self.conformal_labels if label in known
        )
        if not self.conformal_labels:
            # This should be impossible for a covered outcome.  Refuse to
            # invent a new plausible hypothesis if the artifacts disagree.
            self.terminal = SAFE_STOP
            reason = "outcome update removed every conformal world hypothesis"
            self._record(
                kind="INCONSISTENT_OUTCOME_SAFE_STOP",
                action=action,
                prediction_set=labels,
                detail={"reason": reason},
            )
            return OutcomeUpdate(
                accepted_outcome=outcome,
                must_safe_stop=True,
                reason=reason,
                exhausted_search=exhausted_search,
                world_belief=self.belief,
                temporal_belief=self.temporal_belief,
            )

        self._record(
            kind="OUTCOME_ACCEPTED",
            action=action,
            prediction_set=labels,
            detail={"search_exhausted": exhausted_search},
        )
        return OutcomeUpdate(
            accepted_outcome=outcome,
            must_safe_stop=False,
            reason="singleton conformal outcome accepted",
            exhausted_search=exhausted_search,
            world_belief=self.belief,
            temporal_belief=self.temporal_belief,
        )

    def trace(self) -> dict[str, Any]:
        return {
            "schema_version": "interactive-perception.pipeline-trace.v1",
            "terminal": self.terminal,
            "pending_action": self.pending_action,
            "attempts": dict(self.attempts),
            "events": [event.to_dict() for event in self.events],
            "online_oracle_inputs": [],
        }
