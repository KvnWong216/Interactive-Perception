"""Prompt-conditioned target-state evidence from frozen multimodal features."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np

from .semantic_conformal import MondrianSemanticConformalCalibrator


def normalize_rows(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("features must have shape [N, D]")
    if not np.all(np.isfinite(values)):
        raise ValueError("features must be finite")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float64).eps)


@dataclasses.dataclass(frozen=True)
class BalancedRidgeBinary:
    """A deterministic linear probe with equal total weight per class."""

    negative_label: str
    positive_label: str
    regularization: float
    weights: tuple[float, ...]
    intercept: float

    def __post_init__(self) -> None:
        if (
            not self.negative_label
            or not self.positive_label
            or self.negative_label == self.positive_label
        ):
            raise ValueError("two distinct non-empty labels are required")
        if self.regularization <= 0.0 or not np.isfinite(self.regularization):
            raise ValueError("regularization must be finite and positive")
        if not self.weights or not np.all(np.isfinite(self.weights)):
            raise ValueError("weights must be non-empty and finite")
        if not np.isfinite(self.intercept):
            raise ValueError("intercept must be finite")

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: Sequence[str],
        *,
        negative_label: str,
        positive_label: str,
        regularization: float,
    ) -> "BalancedRidgeBinary":
        if not negative_label or not positive_label or negative_label == positive_label:
            raise ValueError("two distinct non-empty labels are required")
        if regularization <= 0.0 or not np.isfinite(regularization):
            raise ValueError("regularization must be finite and positive")
        x = normalize_rows(features)
        labels = np.asarray(list(labels), dtype=str)
        if labels.shape != (x.shape[0],):
            raise ValueError("labels must align with features")
        unknown = set(labels) - {negative_label, positive_label}
        if unknown:
            raise ValueError(f"unknown labels: {sorted(unknown)}")
        y = np.where(labels == positive_label, 1.0, -1.0)
        counts = {value: int(np.sum(labels == value)) for value in (negative_label, positive_label)}
        if any(count == 0 for count in counts.values()):
            raise ValueError("both labels require examples")
        sample_weight = np.asarray(
            [0.5 / counts[label] for label in labels], dtype=np.float64
        )
        x_mean = np.sum(sample_weight[:, None] * x, axis=0)
        y_mean = float(np.sum(sample_weight * y))
        centered_x = x - x_mean
        centered_y = y - y_mean
        root_weight = np.sqrt(sample_weight)
        weighted_x = centered_x * root_weight[:, None]
        dual = weighted_x @ weighted_x.T + regularization * np.eye(x.shape[0])
        coefficient = np.linalg.solve(dual, root_weight * centered_y)
        weights = weighted_x.T @ coefficient
        intercept = y_mean - float(x_mean @ weights)
        return cls(
            negative_label=negative_label,
            positive_label=positive_label,
            regularization=float(regularization),
            weights=tuple(float(value) for value in weights),
            intercept=float(intercept),
        )

    def score(self, features: np.ndarray) -> np.ndarray:
        x = normalize_rows(features)
        weights = np.asarray(self.weights, dtype=np.float64)
        if x.shape[1] != weights.size:
            raise ValueError("feature dimension differs from fitted probe")
        return x @ weights + self.intercept

    def predict(self, features: np.ndarray) -> np.ndarray:
        scores = self.score(features)
        return np.where(scores >= 0.0, self.positive_label, self.negative_label)

    def to_dict(self) -> dict[str, Any]:
        return {
            "negative_label": self.negative_label,
            "positive_label": self.positive_label,
            "regularization": self.regularization,
            "weights": list(self.weights),
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BalancedRidgeBinary":
        return cls(
            negative_label=str(value["negative_label"]),
            positive_label=str(value["positive_label"]),
            regularization=float(value["regularization"]),
            weights=tuple(float(item) for item in value["weights"]),
            intercept=float(value["intercept"]),
        )


@dataclasses.dataclass(frozen=True)
class GaussianScoreCalibrator:
    """Equal-prior class likelihoods on a frozen one-dimensional probe score."""

    labels: tuple[str, str]
    means: tuple[float, float]
    variances: tuple[float, float]

    def __post_init__(self) -> None:
        if len(self.labels) != 2 or len(set(self.labels)) != 2 or any(
            not label for label in self.labels
        ):
            raise ValueError("labels must contain two distinct non-empty values")
        if len(self.means) != 2 or not np.all(np.isfinite(self.means)):
            raise ValueError("means must contain two finite values")
        if len(self.variances) != 2 or not np.all(np.isfinite(self.variances)) or any(
            value <= 0.0 for value in self.variances
        ):
            raise ValueError("variances must contain two finite positive values")

    @classmethod
    def fit(
        cls, scores: Sequence[float], labels: Sequence[str], *, ordered_labels: Sequence[str]
    ) -> "GaussianScoreCalibrator":
        ordered = tuple(str(value) for value in ordered_labels)
        if len(ordered) != 2 or len(set(ordered)) != 2:
            raise ValueError("ordered_labels must contain two distinct labels")
        values = np.asarray(list(scores), dtype=np.float64)
        labels = np.asarray(list(labels), dtype=str)
        if values.ndim != 1 or labels.shape != values.shape or not np.all(np.isfinite(values)):
            raise ValueError("finite one-dimensional scores and aligned labels are required")
        means = []
        variances = []
        for label in ordered:
            subset = values[labels == label]
            if subset.size < 2:
                raise ValueError("each class needs at least two probability-calibration scores")
            means.append(float(np.mean(subset)))
            variances.append(float(max(np.var(subset, ddof=1), 1e-12)))
        return cls(ordered, tuple(means), tuple(variances))

    def probabilities(self, scores: Sequence[float]) -> np.ndarray:
        values = np.asarray(list(scores), dtype=np.float64)[:, None]
        means = np.asarray(self.means, dtype=np.float64)[None, :]
        variances = np.asarray(self.variances, dtype=np.float64)[None, :]
        log_likelihood = -0.5 * (
            np.log(2.0 * np.pi * variances) + (values - means) ** 2 / variances
        )
        shifted = log_likelihood - np.max(log_likelihood, axis=1, keepdims=True)
        likelihood = np.exp(shifted)
        return likelihood / np.sum(likelihood, axis=1, keepdims=True)

    def evidence(self, scores: Sequence[float]) -> list[dict[str, float]]:
        probabilities = self.probabilities(scores)
        return [
            {label: float(row[index]) for index, label in enumerate(self.labels)}
            for row in probabilities
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "means": list(self.means),
            "variances": list(self.variances),
            "prior": "equal class prior",
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GaussianScoreCalibrator":
        return cls(
            labels=tuple(str(item) for item in value["labels"]),
            means=tuple(float(item) for item in value["means"]),
            variances=tuple(float(item) for item in value["variances"]),
        )


@dataclasses.dataclass(frozen=True)
class PromptStatePrediction:
    probabilities: dict[str, float]
    prediction_set: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class PromptStatePredictor:
    feature_start: int
    feature_stop: int
    probe: BalancedRidgeBinary
    probability_model: GaussianScoreCalibrator
    conformal: MondrianSemanticConformalCalibrator

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "PromptStatePredictor":
        feature = artifact["feature_block"]
        return cls(
            feature_start=int(feature["start"]),
            feature_stop=int(feature["stop"]),
            probe=BalancedRidgeBinary.from_dict(artifact["probe"]),
            probability_model=GaussianScoreCalibrator.from_dict(
                artifact["probability_model"]
            ),
            conformal=MondrianSemanticConformalCalibrator.from_dict(
                artifact["conformal"]
            ),
        )

    def predict(self, prefix_features: np.ndarray) -> PromptStatePrediction:
        features = np.asarray(prefix_features, dtype=np.float64)
        if features.shape != (8192,) or not np.all(np.isfinite(features)):
            raise ValueError("prefix_features must be one finite 8192-vector")
        selected = features[None, self.feature_start : self.feature_stop]
        evidence = self.probability_model.evidence(self.probe.score(selected))[0]
        return PromptStatePrediction(
            probabilities=evidence,
            prediction_set=self.conformal.predict(evidence),
        )
