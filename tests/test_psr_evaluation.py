from __future__ import annotations

import math

import pytest

from grounded_interaction.psr.evaluation import (
    CandidateOutcome,
    TrialOutcome,
    binary_probability_metrics,
    candidate_ranking_metrics,
    execution_metrics,
    paired_cluster_bootstrap,
)


def test_probability_metrics_report_bce_brier_and_weighted_ece() -> None:
    metrics = binary_probability_metrics(
        [0.1, 0.8, 0.4, 0.99],
        [0, 1, 1, 0],
        valid_mask=[True, True, False, True],
        bins=2,
    )
    assert metrics.examples == 3
    assert metrics.failures == 1 and metrics.successes == 2
    assert math.isfinite(metrics.bce)
    assert metrics.brier == pytest.approx((0.01 + 0.04 + 0.9801) / 3)
    expected_ece = (abs(0.1 - 0.0) + 2 * abs((0.8 + 0.99) / 2 - 0.5)) / 3
    assert metrics.ece == pytest.approx(expected_ece)
    assert sum(item.count for item in metrics.reliability) == 3
    with pytest.raises(ValueError, match="at least one"):
        binary_probability_metrics([0.2], [0], valid_mask=[False])


def test_candidate_ranking_uses_real_group_branches_and_native_tie_break() -> None:
    rows = [
        CandidateOutcome("g1", "family-a", "n", "native", 0.3, 1),
        CandidateOutcome("g1", "family-a", "c", "conditioned", 0.1, 0),
        CandidateOutcome("g2", "family-b", "n", "native", 0.2, 1),
        CandidateOutcome("g2", "family-b", "c", "conditioned", 0.2, 0),
    ]
    metrics = candidate_ranking_metrics(rows)
    assert metrics.groups == 2 and metrics.candidates == 4
    assert metrics.comparable_pairs == 2
    assert metrics.pairwise_accuracy == pytest.approx(0.75)
    # g1 selects the successful conditioned branch; g2's exact tie selects
    # native by the frozen runtime rule and incurs one unit of oracle gap.
    assert metrics.selected_failure_rate == 0.5
    assert metrics.oracle_failure_rate == 0.0
    assert metrics.mean_oracle_gap == 0.5
    assert metrics.native_selected_rate == 0.5


def test_candidate_ranking_rejects_duplicate_or_mixed_reset_group() -> None:
    duplicate = CandidateOutcome("g", "f", "c", "conditioned", 0.1, 0)
    with pytest.raises(ValueError, match="more than once"):
        candidate_ranking_metrics([duplicate, duplicate])
    with pytest.raises(ValueError, match="mixes reset"):
        candidate_ranking_metrics(
            [
                duplicate,
                CandidateOutcome("g", "other", "n", "native", 0.2, 1),
            ]
        )


def test_paired_bootstrap_resamples_episode_families_not_chunks() -> None:
    first = {"a": [1.0, 1.0, 1.0], "b": [0.0], "c": [0.5, 0.5]}
    second = {"a": [0.0], "b": [0.0, 0.0], "c": [0.0]}
    result = paired_cluster_bootstrap(
        first, second, bootstrap_samples=1000, confidence=0.9, seed=17
    )
    # Each family has equal weight despite different numbers of chunks.
    assert result.estimate == pytest.approx((1.0 + 0.0 + 0.5) / 3)
    assert result.clusters == 3
    assert result.lower <= result.estimate <= result.upper
    repeat = paired_cluster_bootstrap(
        first, second, bootstrap_samples=1000, confidence=0.9, seed=17
    )
    assert repeat == result
    with pytest.raises(ValueError, match="identical cluster"):
        paired_cluster_bootstrap({"a": 1.0}, {"b": 0.0})


def test_execution_metrics_do_not_drop_infrastructure_failures() -> None:
    report = execution_metrics(
        [
            TrialOutcome("a", True, True, 100, 1.5),
            TrialOutcome("b", True, False, 300, 2.5),
            TrialOutcome("c", False, None, 0, 0.2),
        ]
    )
    assert report.planned == 3
    assert report.completed == 2
    assert report.infrastructure_failures == 1
    assert report.completion_rate == pytest.approx(2 / 3)
    assert report.task_success_rate_completed == 0.5
    assert report.task_success_rate_planned == pytest.approx(1 / 3)
    assert report.mean_steps_completed == 200
