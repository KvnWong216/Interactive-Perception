"""Lightweight outcome recognition on frozen before/after VLA features."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np

from .prompt_state import BalancedRidgeBinary
from .active_risk import EffectOutcome
from .semantic_conformal import MondrianSemanticConformalCalibrator


@dataclasses.dataclass(frozen=True)
class FeatureStandardizer:
    """Train-split-only centering and scaling for heterogeneous frozen features."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.mean or len(self.mean) != len(self.scale):
            raise ValueError("mean and scale must be aligned and non-empty")
        if not np.all(np.isfinite(self.mean)):
            raise ValueError("standardizer mean must be finite")
        if not np.all(np.isfinite(self.scale)) or any(
            value <= 0.0 for value in self.scale
        ):
            raise ValueError("standardizer scale must be finite and positive")

    @classmethod
    def fit(cls, features: np.ndarray, *, minimum_scale: float = 1e-6) -> "FeatureStandardizer":
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
            raise ValueError("features must have shape [N >= 2, D >= 1]")
        if not np.all(np.isfinite(values)):
            raise ValueError("features must be finite")
        if minimum_scale <= 0.0 or not np.isfinite(minimum_scale):
            raise ValueError("minimum_scale must be finite and positive")
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        scale = np.where(scale < minimum_scale, 1.0, scale)
        return cls(tuple(mean.tolist()), tuple(scale.tolist()))

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.mean):
            raise ValueError("features differ from the fitted standardizer dimension")
        if not np.all(np.isfinite(values)):
            raise ValueError("features must be finite")
        return (values - np.asarray(self.mean)) / np.asarray(self.scale)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": list(self.mean),
            "scale": list(self.scale),
            "fit_scope": "prototype training split only",
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FeatureStandardizer":
        return cls(
            mean=tuple(float(item) for item in value["mean"]),
            scale=tuple(float(item) for item in value["scale"]),
        )


def label_effect_outcome(
    *,
    opened: bool,
    target_in_resolved_location: bool,
    target_pixels: Sequence[int],
    minimum_target_pixels: int,
) -> EffectOutcome:
    """Evaluator-only label for one physical information-action transition."""

    pixels = [int(value) for value in target_pixels]
    if not pixels or any(value < 0 for value in pixels):
        raise ValueError("target_pixels must be non-negative and non-empty")
    if minimum_target_pixels < 1:
        raise ValueError("minimum_target_pixels must be positive")
    if opened and target_in_resolved_location and max(pixels) >= minimum_target_pixels:
        return EffectOutcome.REVEALED
    if opened and not target_in_resolved_location:
        return EffectOutcome.EMPTY
    return EffectOutcome.FAILED


@dataclasses.dataclass(frozen=True)
class BalancedRidgeMulticlass:
    """One-vs-rest ridge heads with equal positive/negative class mass.

    Scores are converted to normalized evidence only for conformal set
    construction.  They are not advertised as calibrated transition
    probabilities; ex-ante probabilities come from the physical effect
    registry instead.
    """

    labels: tuple[str, ...]
    heads: tuple[BalancedRidgeBinary, ...]

    def __post_init__(self) -> None:
        if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels):
            raise ValueError("at least two unique labels are required")
        if len(self.heads) != len(self.labels):
            raise ValueError("one binary head is required per label")
        for label, head in zip(self.labels, self.heads, strict=True):
            if head.positive_label != label or head.negative_label != f"NOT_{label}":
                raise ValueError("binary head labels do not match multiclass labels")

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: Sequence[str],
        *,
        regularization: float,
    ) -> "BalancedRidgeMulticlass":
        truth = np.asarray(list(labels), dtype=str)
        ordered = tuple(sorted(set(truth.tolist())))
        if len(ordered) < 2:
            raise ValueError("at least two observed classes are required")
        heads = []
        for label in ordered:
            binary = np.where(truth == label, label, f"NOT_{label}")
            heads.append(
                BalancedRidgeBinary.fit(
                    features,
                    binary,
                    negative_label=f"NOT_{label}",
                    positive_label=label,
                    regularization=regularization,
                )
            )
        return cls(ordered, tuple(heads))

    def scores(self, features: np.ndarray) -> np.ndarray:
        return np.stack([head.score(features) for head in self.heads], axis=1)

    def evidence(self, features: np.ndarray) -> list[dict[str, float]]:
        scores = self.scores(features)
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        weights = np.exp(shifted)
        weights /= np.sum(weights, axis=1, keepdims=True)
        return [
            {label: float(row[index]) for index, label in enumerate(self.labels)}
            for row in weights
        ]

    def predict(self, features: np.ndarray) -> np.ndarray:
        indices = np.argmax(self.scores(features), axis=1)
        return np.asarray([self.labels[index] for index in indices])

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "heads": [head.to_dict() for head in self.heads],
            "evidence_note": "softmax-normalized discriminative scores; not calibrated probabilities",
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BalancedRidgeMulticlass":
        return cls(
            labels=tuple(str(label) for label in value["labels"]),
            heads=tuple(
                BalancedRidgeBinary.from_dict(head) for head in value["heads"]
            ),
        )


def transition_feature_block(
    before: np.ndarray, after: np.ndarray, block: str
) -> np.ndarray:
    """Build an explicitly history-dependent feature without simulator state."""

    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if before.ndim != 2 or before.shape != after.shape or before.shape[1] < 1:
        raise ValueError("before and after features must have the same non-empty [N, D] shape")
    if block == "after":
        return after
    if block == "delta":
        return after - before
    if block == "history":
        return np.concatenate([before, after, after - before], axis=1)
    if block.startswith("spatial_"):
        if before.shape[1] <= 8192:
            raise ValueError("spatial blocks require global-plus-spatial v2 features")
        spatial_before = before[:, 8192:]
        spatial_after = after[:, 8192:]
        if block == "spatial_after":
            return spatial_after
        if block == "spatial_delta":
            return spatial_after - spatial_before
        if block == "spatial_history":
            return np.concatenate(
                [spatial_before, spatial_after, spatial_after - spatial_before],
                axis=1,
            )
        raise ValueError(f"unknown transition feature block {block!r}")
    if before.shape[1] != 8192:
        raise ValueError("legacy prompt blocks require 8192-D global features")
    prompt = slice(2048, 4096)
    if block == "after_prompt":
        return after[:, prompt]
    if block == "prompt_delta":
        return after[:, prompt] - before[:, prompt]
    if block == "prompt_history":
        return np.concatenate(
            [before[:, prompt], after[:, prompt], after[:, prompt] - before[:, prompt]],
            axis=1,
        )
    if block == "all_delta":
        return after - before
    if block == "all_history":
        return np.concatenate([before, after, after - before], axis=1)
    raise ValueError(f"unknown transition feature block {block!r}")


@dataclasses.dataclass(frozen=True)
class ActionOutcomePrediction:
    evidence: dict[str, float]
    prediction_set: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ActionOutcomePredictor:
    input_dimension: int
    block: str
    critic: BalancedRidgeMulticlass
    conformal: MondrianSemanticConformalCalibrator
    standardizer: FeatureStandardizer | None = None

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "ActionOutcomePredictor":
        return cls(
            input_dimension=int(artifact.get("input_feature_dimension", 8192)),
            block=str(artifact["model_selection"]["selected"]["block"]),
            critic=BalancedRidgeMulticlass.from_dict(artifact["critic"]),
            conformal=MondrianSemanticConformalCalibrator.from_dict(
                artifact["conformal"]
            ),
            standardizer=(
                FeatureStandardizer.from_dict(artifact["feature_standardizer"])
                if artifact.get("feature_standardizer") is not None
                else None
            ),
        )

    def predict(
        self, before_prefix: np.ndarray, after_prefix: np.ndarray
    ) -> ActionOutcomePrediction:
        before = np.asarray(before_prefix, dtype=np.float64)
        after = np.asarray(after_prefix, dtype=np.float64)
        expected = (self.input_dimension,)
        if before.shape != expected or after.shape != expected:
            raise ValueError(
                f"before/after prefix features must both have shape {expected}"
            )
        values = transition_feature_block(before[None, :], after[None, :], self.block)
        if self.standardizer is not None:
            values = self.standardizer.transform(values)
        evidence = self.critic.evidence(values)[0]
        return ActionOutcomePrediction(
            evidence=evidence,
            prediction_set=self.conformal.predict(evidence),
        )
