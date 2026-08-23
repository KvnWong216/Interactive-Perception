"""Auditable primitive reliability estimates and prospective binomial design."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def wilson_interval(
    successes: int, trials: int, *, z: float = 1.959963984540054
) -> list[float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    rate = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (rate + z**2 / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def reliability_record(values: Sequence[bool]) -> dict[str, Any]:
    if not values:
        raise ValueError("reliability evidence must be non-empty")
    successes = sum(bool(value) for value in values)
    trials = len(values)
    return {
        "successes": successes,
        "trials": trials,
        "rate": successes / trials,
        "wilson_95": wilson_interval(successes, trials),
    }


def binomial_upper_tail(successes: int, trials: int, probability: float) -> float:
    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial tail counts")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("binomial probability must lie in [0,1]")
    return float(
        sum(
            math.comb(trials, count)
            * probability**count
            * (1.0 - probability) ** (trials - count)
            for count in range(successes, trials + 1)
        )
    )


def exact_binomial_rejection_count(
    trials: int, *, null_success_probability: float, alpha: float
) -> int | None:
    """Smallest success count rejecting an absolute reliability null."""

    if trials < 1 or not 0.0 < alpha < 1.0:
        raise ValueError("trials and alpha must be positive")
    for successes in range(trials + 1):
        if (
            binomial_upper_tail(
                successes, trials, null_success_probability
            )
            <= alpha
        ):
            return successes
    return None


def exact_binomial_power(
    trials: int,
    *,
    null_success_probability: float,
    alternative_success_probability: float,
    alpha: float,
) -> dict[str, float | int | None]:
    if not 0.0 <= null_success_probability < alternative_success_probability <= 1.0:
        raise ValueError("alternative reliability must exceed the null contract")
    rejection_count = exact_binomial_rejection_count(
        trials,
        null_success_probability=null_success_probability,
        alpha=alpha,
    )
    power = (
        0.0
        if rejection_count is None
        else binomial_upper_tail(
            rejection_count, trials, alternative_success_probability
        )
    )
    return {
        "trials": trials,
        "rejection_success_count": rejection_count,
        "power": power,
    }


def smallest_binomial_design(
    *,
    null_success_probability: float,
    alternative_success_probability: float,
    alpha: float,
    target_power: float,
    search_limit: int = 1000,
) -> dict[str, float | int | None] | None:
    """Freeze the first exact one-sided binomial design reaching target power."""

    if not 0.0 < target_power < 1.0 or search_limit < 1:
        raise ValueError("invalid power target or numerical search limit")
    for trials in range(1, search_limit + 1):
        design = exact_binomial_power(
            trials,
            null_success_probability=null_success_probability,
            alternative_success_probability=alternative_success_probability,
            alpha=alpha,
        )
        if float(design["power"]) >= target_power:
            return design
    return None


def evaluate_frozen_binomial_design(
    values: Sequence[bool],
    *,
    null_success_probability: float,
    alpha: float,
    expected_trials: int,
    rejection_success_count: int,
) -> dict[str, Any]:
    """Evaluate a prospectively frozen exact-binomial rejection region."""

    if len(values) != expected_trials:
        raise ValueError("formal primitive outcomes do not match the frozen trial count")
    if not 0 <= rejection_success_count <= expected_trials:
        raise ValueError("invalid frozen primitive rejection count")
    if any(not isinstance(value, bool) for value in values):
        raise TypeError("formal primitive outcomes must be booleans")
    successes = sum(values)
    p_value = binomial_upper_tail(
        successes, expected_trials, null_success_probability
    )
    qualified = successes >= rejection_success_count and p_value <= alpha
    return {
        "successes": successes,
        "trials": expected_trials,
        "rate": successes / expected_trials,
        "exact_one_sided_p_value": p_value,
        "rejection_success_count": rejection_success_count,
        "qualified": qualified,
    }
