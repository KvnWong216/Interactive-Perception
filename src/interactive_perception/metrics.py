"""Episode outcomes and the aggregate scores reported for each condition.

Success rate alone is close to useless on these scenarios.  A policy that never
opens the drawer and a policy that opens it and then fumbles the grasp both
score zero, yet they are different findings and only one of them supports the
claim this benchmark exists to test.  Every episode therefore also records
whether the information the task needed ever became available, and what the
policy committed to before it did.

Confidence intervals bootstrap over episodes, never over adjacent frames within
an episode: consecutive steps of one rollout are not independent samples.
"""

from __future__ import annotations

import dataclasses
import statistics
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "AggregateReport",
    "EpisodeOutcome",
    "aggregate",
    "bootstrap_interval",
    "paired_difference",
]


@dataclasses.dataclass
class EpisodeOutcome:
    """One rollout, scored.

    ``information_endpoint_reached`` is the graded signal that keeps a
    zero-success condition interpretable: it is true when the target became
    visible to the policy camera at any point, whatever the policy did next.
    """

    task_id: str
    prompt_variant: str
    seed: int
    steps: int
    task_success: bool
    information_endpoint_reached: bool
    max_target_visible_pixels: int
    steps_to_endpoint: int | None
    first_committed_anchor: str | None
    first_committed_step: int | None
    committed_before_endpoint: bool
    terminal_decision: str
    expected_terminal: str
    mean_vacuity: float
    min_vacuity: float
    mean_dissonance: float
    mean_predictive_entropy: float
    not_found_evidence: float
    # Fraction of sampled chunks whose decisiveness hit its ceiling. At 1.0 the
    # evidence total degenerates to the sample count and vacuity stops
    # depending on the observation, so the uncertainty columns of this episode
    # are not interpretable. See `uncertainty_is_informative`.
    saturated_fraction: float = 0.0
    error: str | None = None

    @property
    def correct_terminal_decision(self) -> bool:
        return self.terminal_decision == self.expected_terminal

    @property
    def premature_commit(self) -> bool:
        """Committed to a physical target before the scene could justify it.

        On a scene whose target starts hidden, any commitment recorded before
        the information endpoint is a commitment made without the evidence that
        would identify the target.
        """

        return self.committed_before_endpoint and not self.information_endpoint_reached

    @property
    def false_not_found(self) -> bool:
        return (
            self.terminal_decision == "NOT_FOUND"
            and self.expected_terminal != "NOT_FOUND"
        )

    @property
    def uncertainty_is_informative(self) -> bool:
        """Whether this episode's uncertainty readings mean anything.

        When every sampled chunk saturates its decisiveness ceiling, the
        Dirichlet strength is exactly ``K + N`` regardless of what the policy
        saw, so vacuity is a constant of the sampling budget rather than a
        measurement. Curves still plot in that regime, which is precisely why
        the condition has to be reported rather than inferred by eye.
        """

        return self.saturated_fraction < 0.99

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload.update(
            {
                "correct_terminal_decision": self.correct_terminal_decision,
                "premature_commit": self.premature_commit,
                "false_not_found": self.false_not_found,
                "uncertainty_is_informative": self.uncertainty_is_informative,
            }
        )
        return payload


def bootstrap_interval(
    values: Sequence[float],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval over episodes."""

    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return (float("nan"), float("nan"))
    if array.size == 1:
        return (float(array[0]), float(array[0]))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[draws].mean(axis=1)
    lower = (1.0 - confidence) / 2.0 * 100.0
    return (
        float(np.percentile(means, lower)),
        float(np.percentile(means, 100.0 - lower)),
    )


def paired_difference(
    treatment: Sequence[EpisodeOutcome],
    control: Sequence[EpisodeOutcome],
    *,
    attribute: str = "task_success",
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare two conditions on episodes sharing ``(task_id, seed)``.

    Pairing matters here because scenario difficulty varies far more between
    tasks than between conditions; an unpaired comparison would be dominated by
    which tasks happened to land in each group.
    """

    keyed = {(item.task_id, item.seed): item for item in control}
    differences: list[float] = []
    for item in treatment:
        counterpart = keyed.get((item.task_id, item.seed))
        if counterpart is None:
            continue
        differences.append(
            float(getattr(item, attribute)) - float(getattr(counterpart, attribute))
        )
    if not differences:
        return {"pairs": 0, "mean_difference": float("nan"), "ci95": (float("nan"),) * 2}
    return {
        "pairs": len(differences),
        "attribute": attribute,
        "mean_difference": float(statistics.fmean(differences)),
        "ci95": bootstrap_interval(differences, resamples=resamples, seed=seed),
    }


@dataclasses.dataclass
class AggregateReport:
    condition: str
    episodes: int
    success_rate: float
    success_ci95: tuple[float, float]
    endpoint_rate: float
    endpoint_ci95: tuple[float, float]
    correct_terminal_rate: float
    premature_commit_rate: float
    false_not_found_rate: float
    mean_vacuity: float
    mean_dissonance: float
    mean_not_found_evidence: float
    mean_saturated_fraction: float
    uninformative_episodes: int
    errors: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def aggregate(
    outcomes: Sequence[EpisodeOutcome], *, condition: str, seed: int = 0
) -> AggregateReport:
    """Summarize one condition.

    ``endpoint_rate`` is reported beside ``success_rate`` on purpose: when the
    former is zero the latter carries no diagnostic weight, because the policy
    never got into a state where success was reachable.
    """

    if not outcomes:
        raise ValueError("cannot aggregate an empty set of outcomes")
    scored = [item for item in outcomes if item.error is None]
    if not scored:
        raise ValueError(f"every episode in condition {condition!r} errored")

    successes = [float(item.task_success) for item in scored]
    endpoints = [float(item.information_endpoint_reached) for item in scored]
    return AggregateReport(
        condition=condition,
        episodes=len(scored),
        success_rate=float(statistics.fmean(successes)),
        success_ci95=bootstrap_interval(successes, seed=seed),
        endpoint_rate=float(statistics.fmean(endpoints)),
        endpoint_ci95=bootstrap_interval(endpoints, seed=seed),
        correct_terminal_rate=float(
            statistics.fmean([float(item.correct_terminal_decision) for item in scored])
        ),
        premature_commit_rate=float(
            statistics.fmean([float(item.premature_commit) for item in scored])
        ),
        false_not_found_rate=float(
            statistics.fmean([float(item.false_not_found) for item in scored])
        ),
        mean_vacuity=float(statistics.fmean([item.mean_vacuity for item in scored])),
        mean_dissonance=float(
            statistics.fmean([item.mean_dissonance for item in scored])
        ),
        mean_not_found_evidence=float(
            statistics.fmean([item.not_found_evidence for item in scored])
        ),
        mean_saturated_fraction=float(
            statistics.fmean([item.saturated_fraction for item in scored])
        ),
        uninformative_episodes=sum(
            not item.uncertainty_is_informative for item in scored
        ),
        errors=len(outcomes) - len(scored),
    )
