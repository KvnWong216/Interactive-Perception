"""Finite-sample split conformal wrapper for primitive routing."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class PrimitiveDecision:
    prediction_set: tuple[str, ...]
    selected_primitive: str | None
    abstain: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class SplitConformalPrimitiveSet:
    """LAC primitive set with marginal coverage under exchangeability."""

    alpha: float
    threshold: float
    labels: tuple[str, ...]
    calibration_size: int
    split_id: str

    @classmethod
    def fit(
        cls,
        probabilities: np.ndarray,
        true_indices: Sequence[int],
        *,
        labels: Sequence[str],
        alpha: float,
        split_id: str,
    ) -> "SplitConformalPrimitiveSet":
        values = np.asarray(probabilities, dtype=np.float64)
        truth = np.asarray(true_indices, dtype=np.int64)
        names = tuple(str(label) for label in labels)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0,1)")
        if not split_id:
            raise ValueError("split_id is required")
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("probabilities must be a non-empty [N,K] array")
        if values.shape[1] != len(names) or truth.shape != (values.shape[0],):
            raise ValueError("probabilities, labels, and truth dimensions disagree")
        if not names or len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("labels must be non-empty and unique")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("probabilities must be finite and non-negative")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("probability rows must sum to one")
        if np.any((truth < 0) | (truth >= len(names))):
            raise ValueError("true index out of range")

        scores = 1.0 - values[np.arange(len(truth)), truth]
        rank = math.ceil((len(scores) + 1) * (1.0 - alpha))
        threshold = 1.0 if rank > len(scores) else float(np.sort(scores)[rank - 1])
        return cls(alpha, threshold, names, len(scores), split_id)

    def prediction_set(
        self, probabilities: Mapping[str, float] | Sequence[float]
    ) -> tuple[str, ...]:
        if isinstance(probabilities, Mapping):
            if set(probabilities) != set(self.labels):
                raise ValueError("probability mapping must match calibrated labels")
            values = np.asarray(
                [probabilities[label] for label in self.labels], dtype=np.float64
            )
        else:
            values = np.asarray(probabilities, dtype=np.float64)
        if values.shape != (len(self.labels),):
            raise ValueError("probabilities must have one entry per label")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("probabilities must be finite and non-negative")
        if not np.isclose(values.sum(), 1.0, atol=1e-6):
            raise ValueError("probabilities must sum to one")
        return tuple(
            label
            for label, probability in zip(self.labels, values)
            if 1.0 - float(probability) <= self.threshold
        )

    def decide(
        self, probabilities: Mapping[str, float] | Sequence[float]
    ) -> PrimitiveDecision:
        candidates = self.prediction_set(probabilities)
        if len(candidates) == 1:
            return PrimitiveDecision(candidates, candidates[0], False, "singleton")
        reason = "empty_set" if not candidates else "ambiguous_set"
        return PrimitiveDecision(candidates, None, True, reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "latent-interaction.primitive-conformal.v1",
            **dataclasses.asdict(self),
            "labels": list(self.labels),
            "guarantee": "marginal primitive-label coverage under exchangeability",
            "execution_rule": "execute singleton; abstain otherwise",
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SplitConformalPrimitiveSet":
        if value.get("schema_version") != "latent-interaction.primitive-conformal.v1":
            raise ValueError("unsupported primitive conformal schema")
        result = cls(
            alpha=float(value["alpha"]),
            threshold=float(value["threshold"]),
            labels=tuple(str(label) for label in value["labels"]),
            calibration_size=int(value["calibration_size"]),
            split_id=str(value["split_id"]),
        )
        if not 0.0 < result.alpha < 1.0 or not 0.0 <= result.threshold <= 1.0:
            raise ValueError("invalid serialized conformal parameters")
        if result.calibration_size < 1 or not result.split_id:
            raise ValueError("invalid serialized calibration provenance")
        if not result.labels or len(result.labels) != len(set(result.labels)):
            raise ValueError("invalid serialized labels")
        return result
