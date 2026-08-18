"""Minimal T01 loop: uncertain world belief plus deterministic control memory."""

from __future__ import annotations

import dataclasses
import enum
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

INFORMATION_ACQUIRED = "INFORMATION_ACQUIRED"


class ControlPhase(str, enum.Enum):
    IDLE = "IDLE"
    AWAITING_EFFECT = "AWAITING_EFFECT"
    COMMITTING = "COMMITTING"
    DONE = "DONE"


@dataclasses.dataclass
class DeterministicControlMemory:
    phase: ControlPhase = ControlPhase.IDLE
    searched_set: set[str] = dataclasses.field(default_factory=set)
    attempt_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    information_acquired: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "searched_set": sorted(self.searched_set),
            "attempt_counts": dict(sorted(self.attempt_counts.items())),
            "information_acquired": self.information_acquired,
        }


@dataclasses.dataclass(frozen=True)
class MinimalOutcomeUpdate:
    accepted_outcome: EffectOutcome | None
    must_safe_stop: bool
    reason: str
    world_belief: TargetBelief
    memory: dict[str, Any]


class MinimalPromptAlignedPipeline:
    """No probabilistic progress state; action legality is a hard memory rule."""

    def __init__(
        self,
        *,
        planner: ExpectedRiskPlanner,
        belief: TargetBelief,
        conformal_labels: Sequence[str],
        effects: Sequence[ActionEffect],
        maximum_attempts_per_action: int = 1,
    ) -> None:
        labels = tuple(str(label) for label in conformal_labels)
        known = {hypothesis.label for hypothesis in belief.hypotheses}
        if not labels or set(labels) - known:
            raise ValueError("conformal labels must name retained world hypotheses")
        if len(labels) != len(set(labels)):
            raise ValueError("conformal labels must be unique")
        if maximum_attempts_per_action < 1:
            raise ValueError("maximum attempts must be positive")
        action_names = [effect.action for effect in effects]
        if len(action_names) != len(set(action_names)):
            raise ValueError("information action names must be unique")
        if planner.temporal_violation_loss != 0.0:
            raise ValueError("minimal pipeline must not use probabilistic temporal risk")
        self.planner = planner
        self.belief = belief
        self.conformal_labels = labels
        self.effects = {effect.action: effect for effect in effects}
        self.maximum_attempts_per_action = int(maximum_attempts_per_action)
        self.memory = DeterministicControlMemory(
            attempt_counts={action: 0 for action in self.effects}
        )
        self.pending_action: str | None = None
        self.terminal: str | None = None
        self.events: list[dict[str, Any]] = []

    def _record(self, kind: str, **detail: Any) -> None:
        self.events.append(
            {
                "index": len(self.events),
                "kind": kind,
                "world_belief": self.belief.to_dict(),
                "conformal_set": list(self.conformal_labels),
                "control_memory": self.memory.to_dict(),
                **detail,
            }
        )

    def _valid_effects(self) -> tuple[ActionEffect, ...]:
        if self.memory.phase is not ControlPhase.IDLE:
            return ()
        return tuple(
            effect
            for effect in self.effects.values()
            if self.memory.attempt_counts[effect.action]
            < self.maximum_attempts_per_action
            and not set(effect.resolves) <= self.memory.searched_set
            and any(
                hypothesis.label in effect.resolves
                and hypothesis.label in self.conformal_labels
                and hypothesis.state is TargetState.MANIPULATION_ONLY
                and probability > 0.0
                for hypothesis, probability in zip(
                    self.belief.hypotheses,
                    self.belief.probabilities,
                    strict=True,
                )
            )
        )

    def plan(self) -> RiskDecision:
        if self.terminal is not None:
            raise RuntimeError(f"episode is terminal: {self.terminal}")
        if self.pending_action is not None or self.memory.phase is not ControlPhase.IDLE:
            raise RuntimeError("an information effect is still pending")
        valid_effects = self._valid_effects()
        decision = self.planner.robust_plan(
            self.belief,
            valid_effects,
            conformal_labels=self.conformal_labels,
            temporal_belief=None,
        )
        if (
            decision.selected_action in self.effects
            and decision.resolvable_uncertainty[decision.selected_action] <= 0.0
        ):
            raise RuntimeError("planner selected non-positive action-resolvable uncertainty")
        self._record(
            "DECISION",
            selected_action=decision.selected_action,
            valid_information_actions=[effect.action for effect in valid_effects],
            risk=decision.to_dict(),
        )
        return decision

    def begin(self, decision: RiskDecision) -> None:
        if self.terminal is not None or self.pending_action is not None:
            raise RuntimeError("cannot dispatch an action in the current state")
        action = decision.selected_action
        valid = {effect.action for effect in self._valid_effects()}
        if action in self.effects:
            if action not in valid:
                raise ValueError(f"information action {action!r} is not hard-valid")
            self.memory.attempt_counts[action] += 1
            self.memory.phase = ControlPhase.AWAITING_EFFECT
            self.pending_action = action
            self._record("ACTION_STARTED", action=action)
            return
        if action == NOT_FOUND:
            self.terminal = SAFE_STOP
            self.memory.phase = ControlPhase.DONE
            self._record(
                "FORBIDDEN_GLOBAL_NOT_FOUND_SAFE_STOP",
                planned_action=NOT_FOUND,
                terminal=SAFE_STOP,
                reason="the two-state T01 experiment has no global exhaustion certificate",
            )
            return
        if action == ACT:
            self.memory.phase = ControlPhase.COMMITTING
            self.terminal = (
                INFORMATION_ACQUIRED if self.memory.information_acquired else ACT
            )
            self.memory.phase = ControlPhase.DONE
            self._record("TERMINAL", planned_action=ACT, terminal=self.terminal)
            return
        if action != SAFE_STOP:
            raise ValueError(f"unknown terminal action: {action}")
        self.terminal = SAFE_STOP
        self.memory.phase = ControlPhase.DONE
        self._record("TERMINAL", planned_action=SAFE_STOP, terminal=SAFE_STOP)

    def observe_information_outcome(
        self, prediction_set: Sequence[str]
    ) -> MinimalOutcomeUpdate:
        if self.pending_action is None:
            raise RuntimeError("no information action is awaiting an outcome")
        labels = tuple(str(label) for label in prediction_set)
        allowed = {outcome.value for outcome in EffectOutcome}
        if not labels or len(labels) != len(set(labels)) or set(labels) - allowed:
            raise ValueError("invalid FAILED/REVEALED/EMPTY prediction set")
        action = self.pending_action
        effect = self.effects[action]
        self.pending_action = None
        if len(labels) != 1:
            return self._safe_stop(
                None,
                "ambiguous conformal outcome; no member was selected",
                prediction_set=labels,
            )
        outcome = EffectOutcome(labels[0])
        if outcome is EffectOutcome.FAILED:
            self.memory.phase = ControlPhase.IDLE
            if (
                self.memory.attempt_counts[action]
                >= self.maximum_attempts_per_action
            ):
                return self._safe_stop(
                    outcome,
                    "FAILED preserved the belief and exhausted the retry budget",
                    prediction_set=labels,
                )
            self._record(
                "OUTCOME_ACCEPTED",
                action=action,
                prediction_set=list(labels),
                belief_changed=False,
            )
            return MinimalOutcomeUpdate(
                outcome,
                False,
                "FAILED preserved the world belief",
                self.belief,
                self.memory.to_dict(),
            )
        if outcome is EffectOutcome.EMPTY:
            self.memory.searched_set.update(effect.resolves)
        try:
            self.belief = update_from_effect(self.belief, effect, outcome)
        except ValueError as error:
            return self._safe_stop(
                outcome,
                f"local outcome contradicts the retained two-state belief: {error}",
                prediction_set=labels,
            )
        known = {hypothesis.label for hypothesis in self.belief.hypotheses}
        self.conformal_labels = tuple(
            label for label in self.conformal_labels if label in known
        )
        if not self.conformal_labels:
            return self._safe_stop(
                outcome,
                "outcome removed every retained conformal hypothesis",
                prediction_set=labels,
            )
        self.memory.information_acquired = outcome is EffectOutcome.REVEALED
        self.memory.phase = ControlPhase.IDLE
        self._record(
            "OUTCOME_ACCEPTED",
            action=action,
            prediction_set=list(labels),
            belief_changed=True,
        )
        return MinimalOutcomeUpdate(
            outcome,
            False,
            "singleton conformal outcome accepted",
            self.belief,
            self.memory.to_dict(),
        )

    def _safe_stop(
        self,
        outcome: EffectOutcome | None,
        reason: str,
        *,
        prediction_set: Sequence[str],
    ) -> MinimalOutcomeUpdate:
        self.terminal = SAFE_STOP
        self.memory.phase = ControlPhase.DONE
        self._record(
            "OUTCOME_SAFE_STOP",
            accepted_outcome=outcome.value if outcome else None,
            prediction_set=list(prediction_set),
            reason=reason,
            terminal=SAFE_STOP,
        )
        return MinimalOutcomeUpdate(
            outcome, True, reason, self.belief, self.memory.to_dict()
        )

    def trace(self) -> dict[str, Any]:
        return {
            "schema_version": "interactive-perception.minimal-pipeline-trace.v1",
            "world_state": ["OBSERVED", "MANIPULATION_ONLY"],
            "control_memory": self.memory.to_dict(),
            "pending_action": self.pending_action,
            "terminal": self.terminal,
            "events": self.events,
            "probabilistic_temporal_progress_used": False,
            "online_oracle_inputs": [],
        }
