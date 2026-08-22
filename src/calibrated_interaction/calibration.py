"""Temperature scaling and finite-sample split conformal prediction sets."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .contracts import EffectFactor


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


@dataclasses.dataclass(frozen=True)
class TemperatureScaler:
    """A single held-out scalar fitted by deterministic NLL minimization."""

    temperature: float

    @classmethod
    def fit(cls, logits: np.ndarray, labels: Sequence[int]) -> TemperatureScaler:
        values = np.asarray(logits, dtype=np.float64)
        truth = np.asarray(labels, dtype=np.int64)
        if values.ndim != 2 or truth.shape != (values.shape[0],):
            raise ValueError("logits must be [N,K] and labels [N]")
        if not np.all(np.isfinite(values)) or np.any(
            (truth < 0) | (truth >= values.shape[1])
        ):
            raise ValueError("invalid calibration logits or labels")

        def objective(log_temperature: float) -> float:
            probabilities = softmax(values / math.exp(log_temperature))
            return float(
                -np.mean(
                    np.log(
                        np.clip(probabilities[np.arange(len(truth)), truth], 1e-12, 1.0)
                    )
                )
            )

        low, high = math.log(0.05), math.log(20.0)
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        left = high - ratio * (high - low)
        right = low + ratio * (high - low)
        left_value, right_value = objective(left), objective(right)
        for _ in range(96):
            if left_value <= right_value:
                high, right, right_value = right, left, left_value
                left = high - ratio * (high - low)
                left_value = objective(left)
            else:
                low, left, left_value = left, right, right_value
                right = low + ratio * (high - low)
                right_value = objective(right)
        return cls(temperature=math.exp((low + high) / 2.0))

    def probabilities(self, logits: np.ndarray) -> np.ndarray:
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be positive and finite")
        return softmax(np.asarray(logits, dtype=np.float64) / self.temperature)


@dataclasses.dataclass(frozen=True)
class LACCalibrator:
    """Least-ambiguous-class conformal set over a fixed decision ontology."""

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
    ) -> LACCalibrator:
        values = np.asarray(probabilities, dtype=np.float64)
        truth = np.asarray(true_indices, dtype=np.int64)
        names = tuple(str(label) for label in labels)
        if not 0.0 < alpha < 1.0 or not split_id:
            raise ValueError("alpha must be in (0,1) and split_id is required")
        if (
            values.ndim != 2
            or values.shape[1] != len(names)
            or truth.shape != (values.shape[0],)
        ):
            raise ValueError(
                "probabilities must be [N,K] with matching truth and labels"
            )
        if len(values) == 0 or len(set(names)) != len(names):
            raise ValueError(
                "non-empty calibration data and unique labels are required"
            )
        if np.any((truth < 0) | (truth >= len(names))) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError("invalid calibration labels or probabilities")
        if np.any(values < 0.0) or not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("rows must be probability distributions")
        scores = 1.0 - values[np.arange(len(truth)), truth]
        rank = math.ceil((len(scores) + 1) * (1.0 - alpha))
        threshold = 1.0 if rank > len(scores) else float(np.sort(scores)[rank - 1])
        return cls(alpha, threshold, names, len(scores), split_id)

    def predict(
        self, probabilities: Mapping[str, float] | Sequence[float]
    ) -> tuple[str, ...]:
        if isinstance(probabilities, Mapping):
            if set(probabilities) != set(self.labels):
                raise ValueError("prediction labels differ from calibration labels")
            values = np.asarray(
                [probabilities[label] for label in self.labels], dtype=np.float64
            )
        else:
            values = np.asarray(probabilities, dtype=np.float64)
        if values.shape != (len(self.labels),) or np.any(values < 0.0):
            raise ValueError("invalid probability vector")
        if not np.isclose(values.sum(), 1.0, atol=1e-6):
            raise ValueError("probabilities must sum to one")
        return tuple(
            label
            for label, probability in zip(self.labels, values, strict=True)
            if 1.0 - float(probability) <= self.threshold
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "calibrated-interaction.lac.v1",
            **dataclasses.asdict(self),
            "labels": list(self.labels),
            "guarantee": "marginal label coverage under exchangeability",
            "non_guarantee": "single-example correctness or task success",
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LACCalibrator:
        return cls(
            alpha=float(value["alpha"]),
            threshold=float(value["threshold"]),
            labels=tuple(str(label) for label in value["labels"]),
            calibration_size=int(value["calibration_size"]),
            split_id=str(value["split_id"]),
        )


@dataclasses.dataclass(frozen=True)
class BinaryEffectCalibration:
    """Bonferroni-calibrated prediction sets for non-exclusive effect factors."""

    calibrators: Mapping[EffectFactor, LACCalibrator]
    joint_alpha: float

    @classmethod
    def fit(
        cls,
        positive_probabilities: np.ndarray,
        labels: np.ndarray,
        *,
        joint_alpha: float,
        split_id: str,
    ) -> BinaryEffectCalibration:
        probabilities = np.asarray(positive_probabilities, dtype=np.float64)
        truth = np.asarray(labels, dtype=np.int64)
        factors = tuple(EffectFactor)
        if probabilities.ndim != 2 or probabilities.shape != truth.shape:
            raise ValueError("effect probabilities and labels must both be [N,F]")
        if probabilities.shape[1] != len(factors) or np.any((truth < 0) | (truth > 1)):
            raise ValueError("effect tensor does not match the factor ontology")
        per_factor_alpha = joint_alpha / len(factors)
        calibrators = {}
        for index, factor in enumerate(factors):
            binary = np.stack(
                (1.0 - probabilities[:, index], probabilities[:, index]), axis=1
            )
            calibrators[factor] = LACCalibrator.fit(
                binary,
                truth[:, index],
                labels=("0", "1"),
                alpha=per_factor_alpha,
                split_id=f"{split_id}:{factor.value}",
            )
        return cls(calibrators=calibrators, joint_alpha=joint_alpha)

    def predict(
        self, positive_probabilities: Mapping[EffectFactor | str, float]
    ) -> dict[EffectFactor, tuple[str, ...]]:
        normalized = {
            (key if isinstance(key, EffectFactor) else EffectFactor(str(key))): float(
                value
            )
            for key, value in positive_probabilities.items()
        }
        if set(normalized) != set(EffectFactor):
            raise ValueError("effect probabilities must contain every factor")
        return {
            factor: calibrator.predict((1.0 - normalized[factor], normalized[factor]))
            for factor, calibrator in self.calibrators.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "calibrated-interaction.binary-effects.v1",
            "joint_alpha": self.joint_alpha,
            "calibrators": {
                factor.value: calibrator.to_dict()
                for factor, calibrator in self.calibrators.items()
            },
            "guarantee": "simultaneous factor coverage lower bound via Bonferroni under exchangeability",
            "non_guarantee": "action usefulness or task success",
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BinaryEffectCalibration:
        return cls(
            calibrators={
                EffectFactor(name): LACCalibrator.from_dict(row)
                for name, row in value["calibrators"].items()
            },
            joint_alpha=float(value["joint_alpha"]),
        )
