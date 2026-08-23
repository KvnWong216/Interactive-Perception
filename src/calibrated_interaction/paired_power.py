"""Exact prospective power for paired binary (McNemar/binomial) tests."""

from __future__ import annotations

import math


def _advance_distribution(
    distribution: dict[tuple[int, int], float],
    *,
    left: float,
    right: float,
    concordant: float,
) -> dict[tuple[int, int], float]:
    updated: dict[tuple[int, int], float] = {}
    for (left_count, right_count), probability in distribution.items():
        for key, mass in (
            ((left_count + 1, right_count), left),
            ((left_count, right_count + 1), right),
            ((left_count, right_count), concordant),
        ):
            updated[key] = updated.get(key, 0.0) + probability * mass
    return updated


def _rejection_power(distribution: dict[tuple[int, int], float], alpha: float) -> float:
    return sum(
        probability
        for (left_count, right_count), probability in distribution.items()
        if exact_paired_binomial_p(left_count, right_count) <= alpha
    )


def exact_paired_binomial_p(left_only: int, right_only: int) -> float:
    """Return the conventional exact two-sided paired-binomial p-value."""

    if min(left_only, right_only) < 0:
        raise ValueError("discordant counts must be nonnegative")
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def prospective_power(
    groups: int,
    *,
    intervention_only_probability: float,
    baseline_only_probability: float,
    alpha: float,
) -> float:
    """Exact unconditional rejection probability under paired multinomial rates."""

    if groups < 1:
        raise ValueError("groups must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    left = float(intervention_only_probability)
    right = float(baseline_only_probability)
    if min(left, right) < 0.0 or left + right > 1.0:
        raise ValueError("discordant probabilities must be nonnegative and sum <=1")
    concordant = 1.0 - left - right
    distribution: dict[tuple[int, int], float] = {(0, 0): 1.0}
    for _ in range(groups):
        distribution = _advance_distribution(
            distribution, left=left, right=right, concordant=concordant
        )
    return _rejection_power(distribution, alpha)


def smallest_prospective_group_count(
    *,
    intervention_only_probability: float,
    baseline_only_probability: float,
    alpha: float,
    target_power: float,
    search_limit: int = 200,
) -> tuple[int, float] | None:
    """Find the first prospectively frozen group count reaching target power."""

    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must lie in (0,1)")
    if search_limit < 1:
        raise ValueError("search_limit must be positive")
    left = float(intervention_only_probability)
    right = float(baseline_only_probability)
    if min(left, right) < 0.0 or left + right > 1.0:
        raise ValueError("discordant probabilities must be nonnegative and sum <=1")
    concordant = 1.0 - left - right
    distribution: dict[tuple[int, int], float] = {(0, 0): 1.0}
    for groups in range(1, search_limit + 1):
        distribution = _advance_distribution(
            distribution,
            left=left,
            right=right,
            concordant=concordant,
        )
        power = _rejection_power(distribution, alpha)
        if power >= target_power:
            return groups, power
    return None
