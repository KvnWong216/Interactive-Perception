"""Conformal calibration for semantic action-intent evidence.

Continuous action chunks are first decoded into the fixed coarse primitive
space. Calibration happens on that semantic distribution, not on Euclidean
trajectory spread. The finite-sample split-conformal quantile provides intent
coverage under exchangeability; it does not guarantee task success.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def intent_probabilities(evidence: Mapping[str, float]) -> dict[str, float]:
    """Normalize non-negative semantic evidence without a tuned temperature."""

    if not evidence:
        raise ValueError("intent evidence cannot be empty")
    values = {str(label): float(value) for label, value in evidence.items()}
    if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("intent evidence must be finite and non-negative")
    total = sum(values.values())
    if total <= 0.0:
        # No action sample committed. Preserve ignorance instead of inventing a
        # winning class: every semantic intent remains possible.
        return {label: 1.0 / len(values) for label in values}
    return {label: value / total for label, value in values.items()}


@dataclasses.dataclass(frozen=True)
class SemanticConformalCalibrator:
    alpha: float
    threshold: float
    labels: tuple[str, ...]
    calibration_size: int
    policy_id: str
    split_id: str

    @classmethod
    def fit(
        cls,
        examples: Sequence[tuple[Mapping[str, float], str]],
        *,
        alpha: float,
        policy_id: str,
        split_id: str,
    ) -> "SemanticConformalCalibrator":
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not policy_id or not split_id:
            raise ValueError("policy_id and split_id are required")
        if not examples:
            raise ValueError("at least one calibration example is required")
        labels = tuple(sorted(intent_probabilities(examples[0][0])))
        scores = []
        for evidence, truth in examples:
            probabilities = intent_probabilities(evidence)
            if tuple(sorted(probabilities)) != labels:
                raise ValueError("all examples must use the same intent labels")
            if truth not in probabilities:
                raise ValueError(f"unknown true intent {truth!r}")
            scores.append(1.0 - probabilities[truth])
        n = len(scores)
        rank = min(n, math.ceil((n + 1) * (1.0 - alpha)))
        threshold = float(np.partition(np.asarray(scores), rank - 1)[rank - 1])
        return cls(alpha, threshold, labels, n, policy_id, split_id)

    def predict(self, evidence: Mapping[str, float]) -> tuple[str, ...]:
        probabilities = intent_probabilities(evidence)
        if tuple(sorted(probabilities)) != self.labels:
            raise ValueError("intent labels differ from the calibration artifact")
        result = tuple(
            label for label in self.labels if 1.0 - probabilities[label] <= self.threshold
        )
        if result:
            return result
        # LAC sets can be empty under a sharp calibration distribution. Return
        # every tied maximizer as a conservative superset. Adding labels cannot
        # reduce conformal coverage, and preserves semantic ambiguity instead
        # of resolving a tie with label order.
        maximum = max(probabilities.values())
        return tuple(
            label for label in self.labels if np.isclose(probabilities[label], maximum)
        )

    @property
    def finite_sample_resolution(self) -> float:
        return 1.0 / (self.calibration_size + 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "interactive-perception.semantic-conformal.v1",
            **dataclasses.asdict(self),
            "labels": list(self.labels),
            "finite_sample_resolution": self.finite_sample_resolution,
            "guarantee": "marginal semantic-intent coverage under exchangeability",
            "non_guarantee": "robot task success",
        }


@dataclasses.dataclass(frozen=True)
class MondrianSemanticConformalCalibrator:
    """Class-conditional conformal sets with one finite-sample quantile per intent."""

    alpha: float
    thresholds: Mapping[str, float]
    labels: tuple[str, ...]
    calibration_size_per_class: Mapping[str, int]
    policy_id: str
    split_id: str

    @classmethod
    def fit(
        cls,
        examples: Sequence[tuple[Mapping[str, float], str]],
        *,
        alpha: float,
        policy_id: str,
        split_id: str,
    ) -> "MondrianSemanticConformalCalibrator":
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not policy_id or not split_id or not examples:
            raise ValueError("examples, policy_id, and split_id are required")
        labels = tuple(sorted(intent_probabilities(examples[0][0])))
        scores: dict[str, list[float]] = {label: [] for label in labels}
        for evidence, truth in examples:
            probabilities = intent_probabilities(evidence)
            if tuple(sorted(probabilities)) != labels or truth not in scores:
                raise ValueError("inconsistent intent labels")
            scores[truth].append(1.0 - probabilities[truth])
        if any(not values for values in scores.values()):
            raise ValueError("every intent requires calibration examples")
        thresholds = {}
        for label, values in scores.items():
            n = len(values)
            rank = min(n, math.ceil((n + 1) * (1.0 - alpha)))
            thresholds[label] = float(np.partition(np.asarray(values), rank - 1)[rank - 1])
        return cls(
            alpha=alpha,
            thresholds=thresholds,
            labels=labels,
            calibration_size_per_class={label: len(values) for label, values in scores.items()},
            policy_id=policy_id,
            split_id=split_id,
        )

    def predict(self, evidence: Mapping[str, float]) -> tuple[str, ...]:
        probabilities = intent_probabilities(evidence)
        if tuple(sorted(probabilities)) != self.labels:
            raise ValueError("intent labels differ from the calibration artifact")
        result = tuple(
            label
            for label in self.labels
            if 1.0 - probabilities[label] <= self.thresholds[label]
        )
        if result:
            return result
        maximum = max(probabilities.values())
        return tuple(
            label for label in self.labels if np.isclose(probabilities[label], maximum)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "interactive-perception.semantic-mondrian-conformal.v1",
            "alpha": self.alpha,
            "thresholds": dict(self.thresholds),
            "labels": list(self.labels),
            "calibration_size_per_class": dict(self.calibration_size_per_class),
            "finite_sample_resolution_per_class": {
                label: 1.0 / (count + 1)
                for label, count in self.calibration_size_per_class.items()
            },
            "policy_id": self.policy_id,
            "split_id": self.split_id,
            "guarantee": "class-conditional semantic-intent coverage under within-class exchangeability",
            "non_guarantee": "robot task success",
        }
