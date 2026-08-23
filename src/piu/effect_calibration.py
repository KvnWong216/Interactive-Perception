"""Isolated route/effect calibration for the candidate-conditioned predictor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .action_effect import EFFECT_FACTORS
from .binding_calibration import (
    MondrianBinaryLAC,
    MulticlassLAC,
    binary_probability_metrics,
    fit_binary_temperature,
    fit_multiclass_temperature,
    multiclass_prediction_set_metrics,
    multiclass_probability_metrics,
    prediction_set_metrics,
    sigmoid,
    spatial_probabilities,
)


def _validate_scores(values: Mapping[str, Any]) -> None:
    required = {
        "route_logits",
        "candidate_valid_mask",
        "factor_logits",
    }
    if not required <= set(values):
        raise ValueError(f"effect scores lack {sorted(required - set(values))}")
    route = np.asarray(values["route_logits"])
    valid = np.asarray(values["candidate_valid_mask"])
    factors = np.asarray(values["factor_logits"])
    if route.ndim != 2 or route.shape != valid.shape:
        raise ValueError("effect route score shapes differ")
    if factors.shape != (*route.shape, len(EFFECT_FACTORS)):
        raise ValueError("effect factor logit shape mismatch")


def _validate_predictions(values: Mapping[str, Any]) -> None:
    _validate_scores(values)
    required = {
        "route_target",
        "factor_target",
        "factor_support_mask",
    }
    if not required <= set(values):
        raise ValueError(f"effect predictions lack {sorted(required - set(values))}")
    route = np.asarray(values["route_logits"])
    target = np.asarray(values["route_target"])
    factors = np.asarray(values["factor_logits"])
    factor_target = np.asarray(values["factor_target"])
    support = np.asarray(values["factor_support_mask"])
    if route.shape != target.shape:
        raise ValueError("effect route prediction shapes differ")
    if factor_target.shape != factors.shape or support.shape != factors.shape:
        raise ValueError("effect factor target/support shapes differ")


def fit_effect_calibration(
    *,
    temperature_values: Mapping[str, Any],
    conformal_values: Mapping[str, Any],
    alphas: Sequence[float],
) -> dict[str, Any]:
    """Fit temperatures and sets on two already group-disjoint roles."""

    _validate_predictions(temperature_values)
    _validate_predictions(conformal_values)
    risks = [float(alpha) for alpha in alphas]
    if not risks or len(set(risks)) != len(risks) or any(
        not 0.0 < alpha < 1.0 for alpha in risks
    ):
        raise ValueError("effect calibration alphas must be unique and valid")
    temperature_route = np.asarray(temperature_values["route_logits"])
    temperature_valid = np.asarray(
        temperature_values["candidate_valid_mask"], dtype=bool
    )
    temperature_target = np.asarray(temperature_values["route_target"])
    route_temperature = fit_multiclass_temperature(
        temperature_route, temperature_valid, temperature_target
    )
    conformal_route = np.asarray(conformal_values["route_logits"])
    conformal_valid = np.asarray(conformal_values["candidate_valid_mask"], dtype=bool)
    conformal_target = np.asarray(conformal_values["route_target"])
    route_probability = spatial_probabilities(
        conformal_route, conformal_valid, temperature=route_temperature
    )
    route_sets = {}
    for alpha in risks:
        calibrator = MulticlassLAC.fit(
            route_probability, conformal_valid, conformal_target, alpha=alpha
        )
        route_sets[str(alpha)] = {
            "calibrator": calibrator.to_dict(),
            "calibration_diagnostic": multiclass_prediction_set_metrics(
                calibrator.predict(route_probability, conformal_valid),
                conformal_target,
            ),
        }
    factors = {}
    for factor_index, factor_name in enumerate(EFFECT_FACTORS):
        temperature_support = np.asarray(
            temperature_values["factor_support_mask"], dtype=bool
        )[:, :, factor_index] & temperature_valid
        conformal_support = np.asarray(
            conformal_values["factor_support_mask"], dtype=bool
        )[:, :, factor_index] & conformal_valid
        temperature_truth = np.asarray(temperature_values["factor_target"])[
            :, :, factor_index
        ][temperature_support]
        conformal_truth = np.asarray(conformal_values["factor_target"])[
            :, :, factor_index
        ][conformal_support]
        if set(temperature_truth.tolist()) != {0.0, 1.0} or set(
            conformal_truth.tolist()
        ) != {0.0, 1.0}:
            factors[factor_name] = {
                "status": "UNSUPPORTED",
                "reason": "both calibration roles require positive and negative labels",
                "temperature_labeled": int(temperature_support.sum()),
                "conformal_labeled": int(conformal_support.sum()),
            }
            continue
        temperature_logits = np.asarray(temperature_values["factor_logits"])[
            :, :, factor_index
        ][temperature_support]
        factor_temperature = fit_binary_temperature(
            temperature_logits, temperature_truth
        )
        conformal_logits = np.asarray(conformal_values["factor_logits"])[
            :, :, factor_index
        ][conformal_support]
        conformal_probability = sigmoid(conformal_logits / factor_temperature)
        factor_sets = {}
        for alpha in risks:
            calibrator = MondrianBinaryLAC.fit(
                conformal_probability, conformal_truth, alpha=alpha
            )
            factor_sets[str(alpha)] = {
                "calibrator": calibrator.to_dict(),
                "calibration_diagnostic": prediction_set_metrics(
                    calibrator.predict(conformal_probability), conformal_truth
                ),
            }
        factors[factor_name] = {
            "status": "SUPPORTED",
            "temperature": factor_temperature,
            "temperature_fit_metrics": {
                "before": binary_probability_metrics(
                    temperature_logits, temperature_truth, temperature=1.0
                ),
                "after": binary_probability_metrics(
                    temperature_logits,
                    temperature_truth,
                    temperature=factor_temperature,
                ),
            },
            "conformal": factor_sets,
        }
    return {
        "schema_version": "piu.action-effect-calibration.v1",
        "route": {
            "temperature": route_temperature,
            "temperature_fit_metrics": {
                "before": multiclass_probability_metrics(
                    temperature_route,
                    temperature_valid,
                    temperature_target,
                    temperature=1.0,
                ),
                "after": multiclass_probability_metrics(
                    temperature_route,
                    temperature_valid,
                    temperature_target,
                    temperature=route_temperature,
                ),
            },
            "conformal": route_sets,
        },
        "factors": factors,
        "reported_alpha": risks,
        "manual_confidence_thresholds": None,
    }


def apply_effect_calibration(
    values: Mapping[str, Any], calibration: Mapping[str, Any], *, alpha: float
) -> dict[str, np.ndarray]:
    leaked = {"route_target", "factor_target", "factor_support_mask"} & set(values)
    if leaked:
        raise ValueError(
            f"online effect calibration contains evaluator targets {sorted(leaked)}"
        )
    _validate_scores(values)
    key = str(float(alpha))
    valid = np.asarray(values["candidate_valid_mask"], dtype=bool)
    route_probability = spatial_probabilities(
        np.asarray(values["route_logits"]),
        valid,
        temperature=float(calibration["route"]["temperature"]),
    )
    route_calibrator = MulticlassLAC.from_dict(
        calibration["route"]["conformal"][key]["calibrator"]
    )
    factor_probability = np.full(
        (*valid.shape, len(EFFECT_FACTORS)), np.nan, dtype=np.float64
    )
    factor_sets = np.zeros(
        (*valid.shape, len(EFFECT_FACTORS), 2), dtype=bool
    )
    for factor_index, factor_name in enumerate(EFFECT_FACTORS):
        section = calibration["factors"][factor_name]
        if section.get("status") != "SUPPORTED":
            continue
        probability = sigmoid(
            np.asarray(values["factor_logits"])[:, :, factor_index]
            / float(section["temperature"])
        )
        factor_probability[:, :, factor_index] = np.where(valid, probability, np.nan)
        calibrator = MondrianBinaryLAC.from_dict(
            section["conformal"][key]["calibrator"]
        )
        predicted = calibrator.predict(probability.reshape(-1)).reshape(
            *valid.shape, 2
        )
        factor_sets[:, :, factor_index] = predicted & valid[:, :, None]
    return {
        "route_probability": route_probability,
        "route_prediction_set": route_calibrator.predict(route_probability, valid),
        "factor_probability": factor_probability,
        "factor_prediction_sets": factor_sets,
    }
