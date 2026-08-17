"""Context-scoped calibration for physical information-action effects.

An executor success rate is not a universal property of an action name.  This
module therefore refuses to fall back across contexts: an ``OPEN_MIDDLE``
estimate measured in one drawer layout cannot silently authorize another
layout.  Evaluator-only labels may be used to fit the registry, but online
planning consumes only the frozen distribution and its conservative lower
bound.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from .active_risk import ActionEffect, EffectOutcome
from .capability_gate import exact_binomial_lower_bound


@dataclasses.dataclass(frozen=True)
class EffectTrial:
    context: str
    action: str
    outcome: EffectOutcome
    source_id: str

    def __post_init__(self) -> None:
        if not self.context or not self.action or not self.source_id:
            raise ValueError("context, action, and source_id are required")


@dataclasses.dataclass(frozen=True)
class CalibratedEffectDistribution:
    context: str
    action: str
    outcome_counts: Mapping[str, int]
    confidence: float
    required_reliability: float
    source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.context or not self.action:
            raise ValueError("context and action are required")
        expected = {outcome.value for outcome in EffectOutcome}
        if set(self.outcome_counts) != expected:
            raise ValueError(f"outcome_counts must contain exactly {sorted(expected)}")
        if any(not isinstance(value, int) or value < 0 for value in self.outcome_counts.values()):
            raise ValueError("outcome counts must be non-negative integers")
        if self.trials < 1 or len(self.source_ids) != self.trials:
            raise ValueError("one unique source_id is required per trial")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("source_ids must be unique")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie in (0, 1)")
        if not 0.0 < self.required_reliability <= 1.0:
            raise ValueError("required_reliability must lie in (0, 1]")

    @property
    def trials(self) -> int:
        return sum(self.outcome_counts.values())

    def empirical_probability(self, outcome: EffectOutcome) -> float:
        return self.outcome_counts[outcome.value] / self.trials

    def lower_bound(self, outcome: EffectOutcome) -> float:
        return exact_binomial_lower_bound(
            self.outcome_counts[outcome.value], self.trials, self.confidence
        )

    def information_completion_lower_bound(self) -> float:
        """Bound the chance that the option returns usable information.

        ``REVEALED`` and ``EMPTY`` are both successful information effects;
        their difference is the hidden target state, not executor quality.
        """

        successes = (
            self.outcome_counts[EffectOutcome.REVEALED.value]
            + self.outcome_counts[EffectOutcome.EMPTY.value]
        )
        return exact_binomial_lower_bound(successes, self.trials, self.confidence)

    def passes(self, outcome: EffectOutcome) -> bool:
        return self.lower_bound(outcome) >= self.required_reliability

    def planner_effect(
        self,
        *,
        resolves: tuple[str, ...],
        cost: float,
        desired_outcome: EffectOutcome | None = EffectOutcome.REVEALED,
    ) -> ActionEffect:
        """Use a conservative, measured outcome bound in Bayes-risk planning."""

        if desired_outcome is None:
            reliability = self.information_completion_lower_bound()
            count = (
                self.outcome_counts[EffectOutcome.REVEALED.value]
                + self.outcome_counts[EffectOutcome.EMPTY.value]
            )
            label = "informative completion"
        else:
            reliability = self.lower_bound(desired_outcome)
            count = self.outcome_counts[desired_outcome.value]
            label = desired_outcome.value

        return ActionEffect(
            action=self.action,
            resolves=resolves,
            reliability=reliability,
            cost=cost,
            source=(
                f"{self.context}: {count}/{self.trials} {label}, one-sided "
                f"{self.confidence:.3f} lower bound"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        outcomes = {}
        for outcome in EffectOutcome:
            outcomes[outcome.value] = {
                "count": self.outcome_counts[outcome.value],
                "empirical_probability": self.empirical_probability(outcome),
                "one_sided_lower_bound": self.lower_bound(outcome),
                "passes_required_reliability": self.passes(outcome),
            }
        return {
            "context": self.context,
            "action": self.action,
            "trials": self.trials,
            "confidence": self.confidence,
            "required_reliability": self.required_reliability,
            "outcomes": outcomes,
            "information_completion": {
                "count": (
                    self.outcome_counts[EffectOutcome.REVEALED.value]
                    + self.outcome_counts[EffectOutcome.EMPTY.value]
                ),
                "one_sided_lower_bound": self.information_completion_lower_bound(),
                "passes_required_reliability": (
                    self.information_completion_lower_bound()
                    >= self.required_reliability
                ),
            },
            "source_ids": list(self.source_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibratedEffectDistribution":
        outcomes = value["outcomes"]
        return cls(
            context=str(value["context"]),
            action=str(value["action"]),
            outcome_counts={
                outcome.value: int(outcomes[outcome.value]["count"])
                for outcome in EffectOutcome
            },
            confidence=float(value["confidence"]),
            required_reliability=float(value["required_reliability"]),
            source_ids=tuple(str(item) for item in value["source_ids"]),
        )


@dataclasses.dataclass(frozen=True)
class EffectRegistry:
    entries: tuple[CalibratedEffectDistribution, ...]

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("effect registry cannot be empty")
        keys = [(entry.context, entry.action) for entry in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("effect registry keys must be unique")

    @classmethod
    def fit(
        cls,
        trials: Iterable[EffectTrial],
        *,
        confidence: float,
        required_reliability: float,
    ) -> "EffectRegistry":
        grouped: dict[tuple[str, str], list[EffectTrial]] = {}
        for trial in trials:
            grouped.setdefault((trial.context, trial.action), []).append(trial)
        if not grouped:
            raise ValueError("at least one effect trial is required")
        entries = []
        for (context, action), rows in sorted(grouped.items()):
            counts = Counter(row.outcome.value for row in rows)
            entries.append(
                CalibratedEffectDistribution(
                    context=context,
                    action=action,
                    outcome_counts={
                        outcome.value: int(counts[outcome.value])
                        for outcome in EffectOutcome
                    },
                    confidence=confidence,
                    required_reliability=required_reliability,
                    source_ids=tuple(row.source_id for row in rows),
                )
            )
        return cls(tuple(entries))

    def get(self, context: str, action: str) -> CalibratedEffectDistribution:
        matches = [
            entry
            for entry in self.entries
            if entry.context == context and entry.action == action
        ]
        if len(matches) != 1:
            raise KeyError(
                f"no exact calibrated effect for context={context!r}, action={action!r}"
            )
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "interactive-perception.effect-registry.v1",
            "fallback_across_contexts": False,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EffectRegistry":
        if value.get("fallback_across_contexts") is not False:
            raise ValueError("effect artifact must explicitly disable context fallback")
        return cls(
            tuple(
                CalibratedEffectDistribution.from_dict(entry)
                for entry in value["entries"]
            )
        )
