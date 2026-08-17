"""Probabilistic finite-trace task progress for interactive perception.

The automaton contains only task-generic ordering rules.  It never names a
drawer, target location, or benchmark-specific information action.  Perception
supplies a belief over propositions; the automaton says which action roles are
currently admissible and records how observed action effects advance the task.

Unlike a T-LEAF-style learned embedding, this first version is an exact runtime
monitor.  That makes logic violations auditable while the VLA and proposition
estimators remain independently replaceable.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any

import numpy as np


class TemporalPhase(str, Enum):
    NEEDS_EVIDENCE = "NEEDS_EVIDENCE"
    AWAITING_EFFECT = "AWAITING_EFFECT"
    READY_TO_COMMIT = "READY_TO_COMMIT"
    SEARCH_EXHAUSTED = "SEARCH_EXHAUSTED"
    COMMITTING = "COMMITTING"
    COMPLETE = "COMPLETE"
    JUSTIFIED_NOT_FOUND = "JUSTIFIED_NOT_FOUND"
    VIOLATION = "VIOLATION"


class TemporalActionRole(str, Enum):
    INFORMATION = "INFORMATION"
    COMMIT = "COMMIT"
    NOT_FOUND = "NOT_FOUND"
    OBSERVE = "OBSERVE"


PHASE_ORDER = tuple(TemporalPhase)


_VALID_PHASES = {
    TemporalActionRole.INFORMATION: frozenset({TemporalPhase.NEEDS_EVIDENCE}),
    TemporalActionRole.COMMIT: frozenset({TemporalPhase.READY_TO_COMMIT}),
    TemporalActionRole.NOT_FOUND: frozenset({TemporalPhase.SEARCH_EXHAUSTED}),
    TemporalActionRole.OBSERVE: frozenset(
        {TemporalPhase.AWAITING_EFFECT, TemporalPhase.COMMITTING}
    ),
}


@dataclasses.dataclass(frozen=True)
class TemporalBelief:
    """A distribution over finite-trace automaton states."""

    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.probabilities, dtype=np.float64)
        if values.shape != (len(PHASE_ORDER),):
            raise ValueError(
                f"temporal belief requires {len(PHASE_ORDER)} probabilities"
            )
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-9)
        ):
            raise ValueError("temporal probabilities must form a distribution")

    @classmethod
    def point(cls, phase: TemporalPhase) -> "TemporalBelief":
        return cls(tuple(float(item is phase) for item in PHASE_ORDER))

    @classmethod
    def from_mapping(
        cls, probabilities: dict[TemporalPhase | str, float]
    ) -> "TemporalBelief":
        normalized = {
            TemporalPhase(key): float(value) for key, value in probabilities.items()
        }
        unknown = set(normalized) - set(PHASE_ORDER)
        if unknown:
            raise ValueError(f"unknown temporal phases: {sorted(unknown)}")
        return cls(tuple(normalized.get(phase, 0.0) for phase in PHASE_ORDER))

    def mass(self, phase: TemporalPhase) -> float:
        return float(self.probabilities[PHASE_ORDER.index(phase)])

    def violation_probability(self, role: TemporalActionRole) -> float:
        """Probability that an action role violates the current task prefix.

        Existing ``VIOLATION`` mass is not charged again.  It represents a
        violation already recorded earlier in the same trace.
        """

        valid = _VALID_PHASES[role]
        return float(
            sum(
                probability
                for phase, probability in zip(
                    PHASE_ORDER, self.probabilities, strict=True
                )
                if phase not in valid and phase is not TemporalPhase.VIOLATION
            )
        )

    def begin(self, role: TemporalActionRole) -> "TemporalBelief":
        """Advance from a decision state to the action's pending state."""

        valid = _VALID_PHASES[role]
        output = {phase: 0.0 for phase in PHASE_ORDER}
        for phase, probability in zip(PHASE_ORDER, self.probabilities, strict=True):
            if probability <= 0.0:
                continue
            if phase is TemporalPhase.VIOLATION:
                output[phase] += probability
            elif phase not in valid:
                output[TemporalPhase.VIOLATION] += probability
            elif role is TemporalActionRole.INFORMATION:
                output[TemporalPhase.AWAITING_EFFECT] += probability
            elif role is TemporalActionRole.COMMIT:
                output[TemporalPhase.COMMITTING] += probability
            elif role is TemporalActionRole.NOT_FOUND:
                output[TemporalPhase.JUSTIFIED_NOT_FOUND] += probability
            else:
                output[phase] += probability
        return TemporalBelief(tuple(output[phase] for phase in PHASE_ORDER))

    def resolve_information(
        self, outcome: str, *, search_exhausted: bool
    ) -> "TemporalBelief":
        """Consume an observed FAILED/REVEALED/EMPTY information effect."""

        if outcome not in {"FAILED", "REVEALED", "EMPTY"}:
            raise ValueError(f"unknown information outcome {outcome!r}")
        pending = self.begin(TemporalActionRole.INFORMATION)
        output = {phase: 0.0 for phase in PHASE_ORDER}
        for phase, probability in zip(
            PHASE_ORDER, pending.probabilities, strict=True
        ):
            if probability <= 0.0:
                continue
            if phase is TemporalPhase.VIOLATION:
                output[phase] += probability
            elif phase is not TemporalPhase.AWAITING_EFFECT:
                output[TemporalPhase.VIOLATION] += probability
            elif outcome == "REVEALED":
                output[TemporalPhase.READY_TO_COMMIT] += probability
            elif outcome == "EMPTY" and search_exhausted:
                output[TemporalPhase.SEARCH_EXHAUSTED] += probability
            else:
                output[TemporalPhase.NEEDS_EVIDENCE] += probability
        return TemporalBelief(tuple(output[phase] for phase in PHASE_ORDER))

    def resolve_commit(self, *, task_complete: bool) -> "TemporalBelief":
        pending = self.begin(TemporalActionRole.COMMIT)
        output = {phase: 0.0 for phase in PHASE_ORDER}
        for phase, probability in zip(
            PHASE_ORDER, pending.probabilities, strict=True
        ):
            if phase is TemporalPhase.COMMITTING:
                next_phase = (
                    TemporalPhase.COMPLETE
                    if task_complete
                    else TemporalPhase.READY_TO_COMMIT
                )
                output[next_phase] += probability
            else:
                output[phase] += probability
        return TemporalBelief(tuple(output[phase] for phase in PHASE_ORDER))

    def to_dict(self) -> dict[str, Any]:
        return {
            "probabilities": {
                phase.value: probability
                for phase, probability in zip(
                    PHASE_ORDER, self.probabilities, strict=True
                )
                if probability > 0.0
            }
        }
