"""Statistical gate for whether an executor can realize a chosen primitive."""

from __future__ import annotations

import dataclasses
from typing import Any

from scipy.stats import beta


def exact_binomial_lower_bound(successes: int, trials: int, confidence: float) -> float:
    """One-sided Clopper-Pearson lower confidence bound."""

    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials and trials >= 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if successes == 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence, successes, trials - successes + 1))


@dataclasses.dataclass(frozen=True)
class CapabilityGate:
    successes: int
    trials: int
    confidence: float
    required_reliability: float

    def __post_init__(self) -> None:
        exact_binomial_lower_bound(self.successes, self.trials, self.confidence)
        if not 0.0 < self.required_reliability <= 1.0:
            raise ValueError("required_reliability must lie in (0, 1]")

    @property
    def empirical_rate(self) -> float:
        return self.successes / self.trials

    @property
    def lower_bound(self) -> float:
        return exact_binomial_lower_bound(self.successes, self.trials, self.confidence)

    @property
    def passed(self) -> bool:
        return self.lower_bound >= self.required_reliability

    def to_dict(self) -> dict[str, Any]:
        return {
            **dataclasses.asdict(self),
            "empirical_rate": self.empirical_rate,
            "lower_bound": self.lower_bound,
            "passed": self.passed,
            "interpretation": (
                "executor competence in the evaluated context; not semantic-intent coverage"
            ),
        }
