from __future__ import annotations

import pytest

from calibrated_interaction.paired_power import (
    exact_paired_binomial_p,
    prospective_power,
    smallest_prospective_group_count,
)


def test_exact_paired_binomial_known_values() -> None:
    assert exact_paired_binomial_p(0, 0) == 1.0
    assert exact_paired_binomial_p(5, 0) == 0.0625
    assert exact_paired_binomial_p(6, 0) == 0.03125


def test_all_pairs_improve_requires_six_groups_at_alpha_005() -> None:
    result = smallest_prospective_group_count(
        intervention_only_probability=1.0,
        baseline_only_probability=0.0,
        alpha=0.05,
        target_power=0.80,
    )
    assert result == (6, 1.0)


def test_unconditional_power_accounts_for_concordant_pairs() -> None:
    power = prospective_power(
        8,
        intervention_only_probability=0.8,
        baseline_only_probability=0.0,
        alpha=0.05,
    )
    assert power == pytest.approx(0.79691776)
