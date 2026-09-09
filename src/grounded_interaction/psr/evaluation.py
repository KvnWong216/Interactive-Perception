"""Auditable probability, ranking, and episode-level metrics for PSR V1.

Probability calibration and candidate selection answer different questions:
BCE/Brier/ECE measure probability quality, while within-reset ranking measures
whether the lowest predicted failure candidate is actually preferable.  All
intervals resample episode/reset families, never correlated chunks as if they
were independent trials.
"""

from __future__ import annotations

import dataclasses
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def _finite_probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0,1]")
    return result


def _binary(value: Any, name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {0, 1}:
        return value
    if isinstance(value, float) and value in {0.0, 1.0}:
        return int(value)
    raise ValueError(f"{name} must be a binary failure indicator")


def _text(value: Any, name: str) -> str:
    result = " ".join(str(value or "").split())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


@dataclasses.dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_predicted_failure: float | None
    empirical_failure_rate: float | None
    absolute_gap: float | None


@dataclasses.dataclass(frozen=True)
class BinaryProbabilityMetrics:
    """Failure-probability metrics on exactly the supplied valid labels."""

    examples: int
    failures: int
    successes: int
    bce: float
    brier: float
    ece: float
    reliability: tuple[ReliabilityBin, ...]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def binary_probability_metrics(
    probabilities: Sequence[float],
    failure_targets: Sequence[int | bool | float],
    *,
    valid_mask: Sequence[bool] | None = None,
    bins: int = 10,
    log_epsilon: float = 1e-12,
) -> BinaryProbabilityMetrics:
    """Compute Bernoulli NLL, Brier score, and fixed-width ECE."""

    if len(probabilities) != len(failure_targets):
        raise ValueError("probabilities and failure targets must be aligned")
    if valid_mask is None:
        valid_mask = [True] * len(probabilities)
    if len(valid_mask) != len(probabilities) or any(
        not isinstance(value, bool) for value in valid_mask
    ):
        raise ValueError("valid_mask must be an aligned sequence of bool values")
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise ValueError("bins must be a positive integer")
    if (
        isinstance(log_epsilon, bool)
        or not isinstance(log_epsilon, (int, float))
        or not math.isfinite(float(log_epsilon))
        or not 0.0 < float(log_epsilon) < 0.5
    ):
        raise ValueError("log_epsilon must be finite and in (0,0.5)")
    selected: list[tuple[float, int]] = []
    for probability, target, valid in zip(probabilities, failure_targets, valid_mask):
        if valid:
            selected.append(
                (
                    _finite_probability(probability, "failure probability"),
                    _binary(target, "failure target"),
                )
            )
    if not selected:
        raise ValueError("probability metrics require at least one valid label")

    epsilon = float(log_epsilon)
    bce_values: list[float] = []
    brier_values: list[float] = []
    grouped: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for probability, target in selected:
        clipped = min(max(probability, epsilon), 1.0 - epsilon)
        bce_values.append(
            -(target * math.log(clipped) + (1 - target) * math.log(1.0 - clipped))
        )
        brier_values.append((probability - target) ** 2)
        grouped[min(int(probability * bins), bins - 1)].append((probability, target))

    reliability: list[ReliabilityBin] = []
    weighted_gap = 0.0
    for index, members in enumerate(grouped):
        lower = index / bins
        upper = (index + 1) / bins
        if not members:
            reliability.append(ReliabilityBin(lower, upper, 0, None, None, None))
            continue
        predicted = sum(item[0] for item in members) / len(members)
        observed = sum(item[1] for item in members) / len(members)
        gap = abs(predicted - observed)
        weighted_gap += len(members) * gap
        reliability.append(
            ReliabilityBin(lower, upper, len(members), predicted, observed, gap)
        )
    failures = sum(target for _, target in selected)
    count = len(selected)
    return BinaryProbabilityMetrics(
        examples=count,
        failures=failures,
        successes=count - failures,
        bce=sum(bce_values) / count,
        brier=sum(brier_values) / count,
        ece=weighted_gap / count,
        reliability=tuple(reliability),
    )


@dataclasses.dataclass(frozen=True)
class CandidateOutcome:
    """One genuinely executed candidate branch in a common-reset group."""

    group_id: str
    reset_family: str
    candidate_id: str
    execution_route: str
    predicted_failure: float
    observed_failure: int | bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", _text(self.group_id, "group_id"))
        object.__setattr__(
            self, "reset_family", _text(self.reset_family, "reset_family")
        )
        object.__setattr__(
            self, "candidate_id", _text(self.candidate_id, "candidate_id")
        )
        if self.execution_route not in {"native", "conditioned"}:
            raise ValueError("execution_route must be native or conditioned")
        object.__setattr__(
            self,
            "predicted_failure",
            _finite_probability(self.predicted_failure, "predicted_failure"),
        )
        object.__setattr__(
            self, "observed_failure", _binary(self.observed_failure, "observed_failure")
        )


@dataclasses.dataclass(frozen=True)
class CandidateRankingMetrics:
    groups: int
    candidates: int
    comparable_pairs: int
    pairwise_accuracy: float | None
    selected_failure_rate: float
    oracle_failure_rate: float
    mean_oracle_gap: float
    native_selected_rate: float
    selected_absolute_error: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def candidate_ranking_metrics(
    outcomes: Sequence[CandidateOutcome],
) -> CandidateRankingMetrics:
    """Evaluate finite-candidate ranking using real same-reset branches only."""

    rows = tuple(outcomes)
    if not rows or any(not isinstance(row, CandidateOutcome) for row in rows):
        raise ValueError("candidate ranking requires CandidateOutcome records")
    duplicate_keys = [(row.group_id, row.candidate_id) for row in rows]
    if len(set(duplicate_keys)) != len(duplicate_keys):
        raise ValueError("candidate branch appears more than once in a group")
    groups: dict[str, list[CandidateOutcome]] = defaultdict(list)
    for row in rows:
        groups[row.group_id].append(row)
    selected_costs: list[int] = []
    oracle_costs: list[int] = []
    gaps: list[int] = []
    selected_errors: list[float] = []
    native_selected = 0
    correct_pairs = 0.0
    comparable_pairs = 0
    for group_id, members in groups.items():
        families = {member.reset_family for member in members}
        if len(families) != 1:
            raise ValueError(f"candidate group {group_id} mixes reset families")
        selected = min(
            members,
            key=lambda row: (
                row.predicted_failure,
                0 if row.execution_route == "native" else 1,
                row.candidate_id,
            ),
        )
        oracle_cost = min(member.observed_failure for member in members)
        selected_costs.append(selected.observed_failure)
        oracle_costs.append(oracle_cost)
        gaps.append(selected.observed_failure - oracle_cost)
        selected_errors.append(
            abs(selected.predicted_failure - selected.observed_failure)
        )
        native_selected += int(selected.execution_route == "native")
        for left_index, left in enumerate(members):
            for right in members[left_index + 1 :]:
                observed_delta = left.observed_failure - right.observed_failure
                if observed_delta == 0:
                    continue
                comparable_pairs += 1
                predicted_delta = left.predicted_failure - right.predicted_failure
                if predicted_delta == 0:
                    correct_pairs += 0.5
                elif (predicted_delta > 0) == (observed_delta > 0):
                    correct_pairs += 1.0
    group_count = len(groups)
    return CandidateRankingMetrics(
        groups=group_count,
        candidates=len(rows),
        comparable_pairs=comparable_pairs,
        pairwise_accuracy=(
            None if comparable_pairs == 0 else correct_pairs / comparable_pairs
        ),
        selected_failure_rate=sum(selected_costs) / group_count,
        oracle_failure_rate=sum(oracle_costs) / group_count,
        mean_oracle_gap=sum(gaps) / group_count,
        native_selected_rate=native_selected / group_count,
        selected_absolute_error=sum(selected_errors) / group_count,
    )


@dataclasses.dataclass(frozen=True)
class PairedBootstrapResult:
    """Cluster-level paired difference ``first - second`` and percentile CI."""

    estimate: float
    confidence: float
    lower: float
    upper: float
    clusters: int
    bootstrap_samples: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _cluster_means(values: Mapping[str, Sequence[float] | float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for raw_key, raw_value in values.items():
        key = _text(raw_key, "cluster identity")
        if isinstance(raw_value, Sequence) and not isinstance(
            raw_value, (str, bytes, bytearray)
        ):
            members = list(raw_value)
        else:
            members = [raw_value]
        if not members:
            raise ValueError(f"cluster {key} has no observations")
        parsed: list[float] = []
        for member in members:
            if isinstance(member, bool):
                member = float(member)
            if not isinstance(member, (int, float)) or not math.isfinite(float(member)):
                raise ValueError(f"cluster {key} contains a non-finite observation")
            parsed.append(float(member))
        if key in result:
            raise ValueError("duplicate normalized cluster identity")
        result[key] = sum(parsed) / len(parsed)
    if not result:
        raise ValueError("paired bootstrap requires at least one cluster")
    return result


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile input is empty")
    location = probability * (len(sorted_values) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = location - lower
    return float(
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    )


def paired_cluster_bootstrap(
    first: Mapping[str, Sequence[float] | float],
    second: Mapping[str, Sequence[float] | float],
    *,
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 17,
) -> PairedBootstrapResult:
    """Pair by reset family, average within family, then resample families."""

    first_means = _cluster_means(first)
    second_means = _cluster_means(second)
    if set(first_means) != set(second_means):
        missing_first = sorted(set(second_means) - set(first_means))
        missing_second = sorted(set(first_means) - set(second_means))
        raise ValueError(
            "paired methods require identical cluster identities; "
            f"missing_first={missing_first[:3]}, missing_second={missing_second[:3]}"
        )
    if (
        not isinstance(bootstrap_samples, int)
        or isinstance(bootstrap_samples, bool)
        or bootstrap_samples < 1
    ):
        raise ValueError("bootstrap_samples must be a positive integer")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValueError("confidence must be in (0,1)")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    keys = sorted(first_means)
    differences = [first_means[key] - second_means[key] for key in keys]
    estimate = sum(differences) / len(differences)
    rng = random.Random(seed)
    replicates = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in keys) / len(keys)
        for _ in range(bootstrap_samples)
    )
    alpha = (1.0 - float(confidence)) / 2.0
    return PairedBootstrapResult(
        estimate=estimate,
        confidence=float(confidence),
        lower=_quantile(replicates, alpha),
        upper=_quantile(replicates, 1.0 - alpha),
        clusters=len(keys),
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


@dataclasses.dataclass(frozen=True)
class TrialOutcome:
    """One planned closed-loop episode, including infrastructure failures."""

    reset_family: str
    completed: bool
    success: bool | None
    actual_steps: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reset_family", _text(self.reset_family, "reset_family")
        )
        if not isinstance(self.completed, bool):
            raise TypeError("completed must be bool")
        if self.completed and not isinstance(self.success, bool):
            raise TypeError("completed trial requires a binary task outcome")
        if not self.completed and self.success is not None:
            raise ValueError("infrastructure failure cannot carry a task outcome")
        if (
            not isinstance(self.actual_steps, int)
            or isinstance(self.actual_steps, bool)
            or self.actual_steps < 0
        ):
            raise ValueError("actual_steps must be a non-negative integer")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or not math.isfinite(float(self.elapsed_seconds))
            or float(self.elapsed_seconds) < 0
        ):
            raise ValueError("elapsed_seconds must be finite and non-negative")


@dataclasses.dataclass(frozen=True)
class ExecutionMetrics:
    planned: int
    completed: int
    infrastructure_failures: int
    completion_rate: float
    task_successes: int
    task_success_rate_completed: float | None
    task_success_rate_planned: float
    mean_steps_completed: float | None
    mean_elapsed_seconds_completed: float | None


def execution_metrics(trials: Sequence[TrialOutcome]) -> ExecutionMetrics:
    """Keep infrastructure completion and task success as separate quantities."""

    rows = tuple(trials)
    if not rows or any(not isinstance(row, TrialOutcome) for row in rows):
        raise ValueError("execution metrics require TrialOutcome records")
    completed = [row for row in rows if row.completed]
    successes = sum(bool(row.success) for row in completed)
    return ExecutionMetrics(
        planned=len(rows),
        completed=len(completed),
        infrastructure_failures=len(rows) - len(completed),
        completion_rate=len(completed) / len(rows),
        task_successes=successes,
        task_success_rate_completed=(
            None if not completed else successes / len(completed)
        ),
        task_success_rate_planned=successes / len(rows),
        mean_steps_completed=(
            None
            if not completed
            else sum(row.actual_steps for row in completed) / len(completed)
        ),
        mean_elapsed_seconds_completed=(
            None
            if not completed
            else sum(float(row.elapsed_seconds) for row in completed) / len(completed)
        ),
    )


__all__ = [
    "BinaryProbabilityMetrics",
    "CandidateOutcome",
    "CandidateRankingMetrics",
    "ExecutionMetrics",
    "PairedBootstrapResult",
    "ReliabilityBin",
    "TrialOutcome",
    "binary_probability_metrics",
    "candidate_ranking_metrics",
    "execution_metrics",
    "paired_cluster_bootstrap",
]
