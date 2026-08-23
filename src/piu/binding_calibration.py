"""Held-out temperature scaling and conformal sets for target binding."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(values)
    nonnegative = values >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponent = np.exp(values[~nonnegative])
    result[~nonnegative] = exponent / (1.0 + exponent)
    return result


def _bounded_scalar_minimize(
    objective: Callable[[float], float], *, lower: float = -6.0, upper: float = 6.0
) -> float:
    """Deterministic golden-section search over log temperature."""

    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = float(lower)
    right = float(upper)
    middle_left = right - ratio * (right - left)
    middle_right = left + ratio * (right - left)
    value_left = objective(middle_left)
    value_right = objective(middle_right)
    for _ in range(96):
        if value_left <= value_right:
            right = middle_right
            middle_right = middle_left
            value_right = value_left
            middle_left = right - ratio * (right - left)
            value_left = objective(middle_left)
        else:
            left = middle_left
            middle_left = middle_right
            value_left = value_right
            middle_right = left + ratio * (right - left)
            value_right = objective(middle_right)
    return float(math.exp((left + right) / 2.0))


def binary_probability_metrics(
    logits: np.ndarray, truth: np.ndarray, *, temperature: float
) -> dict[str, float | int]:
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(truth, dtype=np.float64)
    if values.shape != labels.shape or values.ndim != 1:
        raise ValueError("binary logits and truth must be matching vectors")
    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    if not np.isfinite(values).all() or not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("binary calibration inputs are malformed")
    probability = sigmoid(values / temperature)
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return {
        "samples": len(labels),
        "nll": float(
            -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))
        ),
        "brier": float(np.mean((probability - labels) ** 2)),
    }


def fit_binary_temperature(logits: np.ndarray, truth: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(truth, dtype=np.float64)
    binary_probability_metrics(values, labels, temperature=1.0)
    if set(labels.tolist()) != {0.0, 1.0}:
        raise ValueError("temperature fitting requires both binary classes")

    def objective(log_temperature: float) -> float:
        return float(
            binary_probability_metrics(
                values, labels, temperature=math.exp(log_temperature)
            )["nll"]
        )

    return _bounded_scalar_minimize(objective)


def spatial_probabilities(
    logits: np.ndarray, valid_mask: np.ndarray, *, temperature: float
) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if values.ndim != 2 or values.shape != valid.shape:
        raise ValueError("spatial logits and mask must be matching matrices")
    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    if not valid.any(axis=1).all() or not np.isfinite(values[valid]).all():
        raise ValueError("each spatial row needs finite valid logits")
    scaled = np.where(valid, values / temperature, -np.inf)
    maximum = scaled.max(axis=1, keepdims=True)
    numerator = np.where(valid, np.exp(scaled - maximum), 0.0)
    return numerator / numerator.sum(axis=1, keepdims=True)


def spatial_probability_metrics(
    logits: np.ndarray,
    valid_mask: np.ndarray,
    patch_target: np.ndarray,
    *,
    temperature: float,
) -> dict[str, float | int]:
    target = np.asarray(patch_target, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if target.shape != valid.shape or np.any(target < 0.0):
        raise ValueError("spatial target/mask mismatch")
    if np.any((target > 0.0) & ~valid):
        raise ValueError("spatial target occupies an invalid token")
    localized = target.sum(axis=1) > 0.0
    if not localized.any():
        raise ValueError("spatial temperature fitting needs localizable examples")
    probability = spatial_probabilities(logits, valid, temperature=temperature)
    normalized = target[localized] / target[localized].sum(axis=1, keepdims=True)
    nll = -(normalized * np.log(np.clip(probability[localized], 1e-12, 1.0))).sum(
        axis=1
    )
    predicted_patch = probability[localized].argmax(axis=1)
    return {
        "samples": int(localized.sum()),
        "nll": float(nll.mean()),
        "point_hit": float(
            np.mean(
                target[localized][np.arange(int(localized.sum())), predicted_patch]
                > 0.0
            )
        ),
        "target_probability_mass": float(
            np.mean((probability[localized] * (target[localized] > 0.0)).sum(axis=1))
        ),
    }


def fit_spatial_temperature(
    logits: np.ndarray, valid_mask: np.ndarray, patch_target: np.ndarray
) -> float:
    spatial_probability_metrics(logits, valid_mask, patch_target, temperature=1.0)

    def objective(log_temperature: float) -> float:
        return float(
            spatial_probability_metrics(
                logits,
                valid_mask,
                patch_target,
                temperature=math.exp(log_temperature),
            )["nll"]
        )

    return _bounded_scalar_minimize(objective)


def multiclass_probability_metrics(
    logits: np.ndarray,
    valid_mask: np.ndarray,
    target_distribution: np.ndarray,
    *,
    temperature: float,
) -> dict[str, float | int]:
    values = np.asarray(logits, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    target = np.asarray(target_distribution, dtype=np.float64)
    if values.shape != valid.shape or values.shape != target.shape or values.ndim != 2:
        raise ValueError("multiclass logits, mask, and target must match")
    if np.any(target < 0.0) or np.any((target > 0.0) & ~valid):
        raise ValueError("multiclass target is invalid")
    if not np.allclose(target.sum(axis=1), 1.0):
        raise ValueError("multiclass target rows must sum to one")
    probability = spatial_probabilities(values, valid, temperature=temperature)
    nll = -(target * np.log(np.clip(probability, 1e-12, 1.0))).sum(axis=1)
    truth = target.argmax(axis=1)
    return {
        "samples": len(values),
        "nll": float(nll.mean()),
        "top1_accuracy": float((probability.argmax(axis=1) == truth).mean()),
    }


def fit_multiclass_temperature(
    logits: np.ndarray, valid_mask: np.ndarray, target_distribution: np.ndarray
) -> float:
    multiclass_probability_metrics(
        logits, valid_mask, target_distribution, temperature=1.0
    )

    def objective(log_temperature: float) -> float:
        return float(
            multiclass_probability_metrics(
                logits,
                valid_mask,
                target_distribution,
                temperature=math.exp(log_temperature),
            )["nll"]
        )

    return _bounded_scalar_minimize(objective)


def finite_sample_quantile(
    scores: np.ndarray, *, alpha: float, maximum_score: float
) -> dict[str, float | int | bool]:
    """Return the split-conformal order statistic without interpolation."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("conformal scores must be a non-empty finite vector")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between zero and one")
    if np.any(values > maximum_score):
        raise ValueError("conformal score exceeds its declared maximum")
    rank = math.ceil((values.size + 1) * (1.0 - alpha))
    saturated = rank > values.size
    quantile = (
        float(maximum_score)
        if saturated
        else float(np.partition(values, rank - 1)[rank - 1])
    )
    return {
        "alpha": float(alpha),
        "samples": int(values.size),
        "rank": rank,
        "quantile": quantile,
        "finite_sample_saturated": saturated,
        "minimum_resolvable_alpha": float(1.0 / (values.size + 1)),
    }


@dataclasses.dataclass(frozen=True)
class MondrianBinaryLAC:
    """Class-conditional least-ambiguous conformal prediction sets."""

    alpha: float
    class_quantiles: Mapping[int, float]
    class_counts: Mapping[int, int]

    @classmethod
    def fit(
        cls, probabilities: np.ndarray, truth: np.ndarray, *, alpha: float
    ) -> MondrianBinaryLAC:
        positive = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(truth, dtype=np.int64)
        if positive.ndim != 1 or positive.shape != labels.shape:
            raise ValueError("binary probabilities and truth must be matching vectors")
        if not np.isfinite(positive).all() or np.any((positive < 0) | (positive > 1)):
            raise ValueError("binary probabilities must lie in [0,1]")
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("Mondrian calibration requires both binary classes")
        matrix = np.stack((1.0 - positive, positive), axis=1)
        quantiles = {}
        counts = {}
        for label in (0, 1):
            scores = 1.0 - matrix[labels == label, label]
            fitted = finite_sample_quantile(scores, alpha=alpha, maximum_score=1.0)
            quantiles[label] = float(fitted["quantile"])
            counts[label] = int(fitted["samples"])
        return cls(alpha=float(alpha), class_quantiles=quantiles, class_counts=counts)

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        positive = np.asarray(probabilities, dtype=np.float64)
        if positive.ndim != 1:
            raise ValueError("binary probabilities must be a vector")
        matrix = np.stack((1.0 - positive, positive), axis=1)
        return np.stack(
            [1.0 - matrix[:, label] <= self.class_quantiles[label] for label in (0, 1)],
            axis=1,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MondrianBinaryLAC:
        if value.get("method") != "class_conditional_LAC":
            raise ValueError("unsupported binary conformal calibrator")
        quantiles = {
            int(key): float(item)
            for key, item in dict(value.get("class_quantiles", {})).items()
        }
        counts = {
            int(key): int(item)
            for key, item in dict(value.get("class_counts", {})).items()
        }
        if set(quantiles) != {0, 1} or set(counts) != {0, 1}:
            raise ValueError("binary calibrator must contain both classes")
        if any(not 0.0 <= item <= 1.0 for item in quantiles.values()):
            raise ValueError("binary conformal quantiles must lie in [0,1]")
        return cls(
            alpha=float(value["alpha"]),
            class_quantiles=quantiles,
            class_counts=counts,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "class_conditional_LAC",
            "alpha": self.alpha,
            "class_quantiles": {
                str(key): float(value) for key, value in self.class_quantiles.items()
            },
            "class_counts": {
                str(key): int(value) for key, value in self.class_counts.items()
            },
        }


@dataclasses.dataclass(frozen=True)
class MulticlassLAC:
    """Marginal split-conformal set over dynamically instantiated candidates."""

    alpha: float
    quantile: float
    samples: int
    finite_sample_saturated: bool

    @classmethod
    def fit(
        cls,
        probabilities: np.ndarray,
        valid_mask: np.ndarray,
        target_distribution: np.ndarray,
        *,
        alpha: float,
    ) -> MulticlassLAC:
        probability = np.asarray(probabilities, dtype=np.float64)
        valid = np.asarray(valid_mask, dtype=bool)
        target = np.asarray(target_distribution, dtype=np.float64)
        if probability.shape != valid.shape or probability.shape != target.shape:
            raise ValueError("route probability, mask, and target differ")
        if np.any((target > 0.0) & ~valid) or not np.allclose(target.sum(axis=1), 1.0):
            raise ValueError("route target must select one valid candidate")
        truth = target.argmax(axis=1)
        scores = 1.0 - probability[np.arange(len(probability)), truth]
        fitted = finite_sample_quantile(scores, alpha=alpha, maximum_score=1.0)
        return cls(
            alpha=float(alpha),
            quantile=float(fitted["quantile"]),
            samples=int(fitted["samples"]),
            finite_sample_saturated=bool(fitted["finite_sample_saturated"]),
        )

    def predict(self, probabilities: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        probability = np.asarray(probabilities, dtype=np.float64)
        valid = np.asarray(valid_mask, dtype=bool)
        if probability.shape != valid.shape:
            raise ValueError("route probability and candidate mask differ")
        return (1.0 - probability <= self.quantile) & valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "multiclass_LAC",
            "alpha": self.alpha,
            "quantile": self.quantile,
            "samples": self.samples,
            "finite_sample_saturated": self.finite_sample_saturated,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MulticlassLAC:
        if value.get("method") != "multiclass_LAC":
            raise ValueError("unsupported route conformal calibrator")
        return cls(
            alpha=float(value["alpha"]),
            quantile=float(value["quantile"]),
            samples=int(value["samples"]),
            finite_sample_saturated=bool(value["finite_sample_saturated"]),
        )


def multiclass_prediction_set_metrics(
    prediction_sets: np.ndarray, target_distribution: np.ndarray
) -> dict[str, float | int]:
    sets = np.asarray(prediction_sets, dtype=bool)
    target = np.asarray(target_distribution, dtype=np.float64)
    if sets.shape != target.shape or sets.ndim != 2:
        raise ValueError("route prediction-set/target shape mismatch")
    truth = target.argmax(axis=1)
    sizes = sets.sum(axis=1)
    return {
        "samples": len(sets),
        "coverage": float(sets[np.arange(len(sets)), truth].mean()),
        "mean_set_size": float(sizes.mean()),
        "empty_rate": float((sizes == 0).mean()),
        "singleton_rate": float((sizes == 1).mean()),
    }


def spatial_conformal_fit(
    probabilities: np.ndarray,
    patch_target: np.ndarray,
    *,
    alpha: float,
) -> dict[str, float | int | bool | str]:
    values = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(patch_target, dtype=np.float64)
    if values.ndim != 2 or values.shape != target.shape:
        raise ValueError("spatial probabilities and target must be matching matrices")
    localized = target.sum(axis=1) > 0.0
    if not localized.any():
        raise ValueError("spatial conformal calibration needs localizable examples")
    maximum_target_probability = np.where(
        target[localized] > 0.0, values[localized], -np.inf
    ).max(axis=1)
    fitted = finite_sample_quantile(
        1.0 - maximum_target_probability,
        alpha=alpha,
        maximum_score=1.0,
    )
    return {
        "method": "target_intersection_split_conformal",
        **fitted,
    }


def spatial_prediction_sets(
    probabilities: np.ndarray,
    valid_mask: np.ndarray,
    calibration: Mapping[str, Any],
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if values.shape != valid.shape:
        raise ValueError("spatial probabilities and valid mask differ")
    threshold = 1.0 - float(calibration["quantile"])
    return (values >= threshold) & valid


def prediction_set_metrics(
    prediction_sets: np.ndarray, truth: np.ndarray
) -> dict[str, float | int]:
    sets = np.asarray(prediction_sets, dtype=bool)
    labels = np.asarray(truth, dtype=np.int64)
    if sets.ndim != 2 or sets.shape[1] != 2 or sets.shape[0] != labels.size:
        raise ValueError("binary prediction-set shape mismatch")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("binary truth must contain only zero and one")
    return {
        "samples": len(labels),
        "coverage": float(sets[np.arange(len(labels)), labels].mean()),
        "mean_set_size": float(sets.sum(axis=1).mean()),
        "empty_rate": float((sets.sum(axis=1) == 0).mean()),
        "singleton_rate": float((sets.sum(axis=1) == 1).mean()),
    }


def spatial_prediction_set_metrics(
    prediction_sets: np.ndarray, patch_target: np.ndarray
) -> dict[str, float | int]:
    sets = np.asarray(prediction_sets, dtype=bool)
    target = np.asarray(patch_target, dtype=np.float64)
    if sets.shape != target.shape or sets.ndim != 2:
        raise ValueError("spatial prediction-set/target shape mismatch")
    localized = target.sum(axis=1) > 0.0
    if not localized.any():
        raise ValueError("spatial set metrics need localizable examples")
    intersects = ((target[localized] > 0.0) & sets[localized]).any(axis=1)
    return {
        "localized_samples": int(localized.sum()),
        "target_intersection_coverage": float(intersects.mean()),
        "mean_set_size": float(sets[localized].sum(axis=1).mean()),
        "empty_rate": float((sets[localized].sum(axis=1) == 0).mean()),
    }


def apply_binding_calibration(
    values: Mapping[str, Any], calibration: Mapping[str, Any], *, alpha: float
) -> dict[str, np.ndarray]:
    """Apply frozen binder calibration without accepting evaluator truth."""

    leaked = {
        "patch_target",
        "target_present",
        "task_sufficient",
        "task_sufficient_mask",
        "holding_requested_target",
        "holding_requested_target_mask",
        "region_confirmed_empty",
        "region_confirmed_empty_mask",
        "task_complete",
        "task_complete_mask",
    } & set(values)
    if leaked:
        raise ValueError(
            f"online binder calibration contains evaluator targets {sorted(leaked)}"
        )
    required = {
        "spatial_logits",
        "image_valid_mask",
        "target_present_logit",
        "task_sufficiency_logit",
        "holding_requested_target_logit",
        "region_confirmed_empty_logit",
        "task_complete_logit",
    }
    if not required <= set(values):
        raise ValueError(f"binder scores lack {sorted(required - set(values))}")
    spatial_logits = np.asarray(values["spatial_logits"], dtype=np.float64)
    valid = np.asarray(values["image_valid_mask"], dtype=bool)
    presence_logits = np.asarray(values["target_present_logit"], dtype=np.float64)
    sufficiency_logits = np.asarray(values["task_sufficiency_logit"], dtype=np.float64)
    holding_logits = np.asarray(
        values["holding_requested_target_logit"], dtype=np.float64
    )
    region_empty_logits = np.asarray(
        values["region_confirmed_empty_logit"], dtype=np.float64
    )
    task_complete_logits = np.asarray(values["task_complete_logit"], dtype=np.float64)
    if spatial_logits.ndim != 2 or spatial_logits.shape != valid.shape:
        raise ValueError("binder spatial score shapes differ")
    if any(
        logits.shape != (len(valid),)
        for logits in (
            presence_logits,
            sufficiency_logits,
            holding_logits,
            region_empty_logits,
            task_complete_logits,
        )
    ):
        raise ValueError("binder binary score shapes differ")
    key = str(float(alpha))
    if key not in calibration["spatial"]["conformal"]:
        raise ValueError("binder alpha was not fitted")
    spatial_probability = spatial_probabilities(
        spatial_logits,
        valid,
        temperature=float(calibration["spatial"]["temperature"]),
    )
    spatial_set = spatial_prediction_sets(
        spatial_probability,
        valid,
        calibration["spatial"]["conformal"][key]["calibrator"],
    )

    def binary(
        section: Mapping[str, Any], logits: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if section.get("status") != "SUPPORTED":
            return (
                np.full(logits.shape, np.nan, dtype=np.float64),
                np.zeros((*logits.shape, 2), dtype=bool),
            )
        probability = sigmoid(logits / float(section["temperature"]))
        calibrator = MondrianBinaryLAC.from_dict(
            section["conformal"][key]["calibrator"]
        )
        return probability, calibrator.predict(probability)

    presence_probability, presence_set = binary(
        calibration["target_presence"], presence_logits
    )
    sufficiency_probability, sufficiency_set = binary(
        calibration["task_sufficiency"], sufficiency_logits
    )
    holding_probability, holding_set = binary(
        calibration["holding_requested_target"], holding_logits
    )
    region_empty_probability, region_empty_set = binary(
        calibration["region_confirmed_empty"], region_empty_logits
    )
    task_complete_probability, task_complete_set = binary(
        calibration["task_complete"], task_complete_logits
    )
    return {
        "spatial_probability": spatial_probability,
        "spatial_prediction_set": spatial_set,
        "target_presence_probability": presence_probability,
        "target_presence_prediction_set": presence_set,
        "task_sufficiency_probability": sufficiency_probability,
        "task_sufficiency_prediction_set": sufficiency_set,
        "holding_requested_target_probability": holding_probability,
        "holding_requested_target_prediction_set": holding_set,
        "region_confirmed_empty_probability": region_empty_probability,
        "region_confirmed_empty_prediction_set": region_empty_set,
        "task_complete_probability": task_complete_probability,
        "task_complete_prediction_set": task_complete_set,
    }
