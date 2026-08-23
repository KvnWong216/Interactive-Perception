from __future__ import annotations

import numpy as np
import pytest

from piu.action_effect import EFFECT_FACTORS
from piu.effect_calibration import apply_effect_calibration, fit_effect_calibration


def _predictions(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples = 8
    candidates = 2
    route_target = np.zeros((samples, candidates), dtype=np.float32)
    route_target[np.arange(samples), np.arange(samples) % 2] = 1.0
    factor_target = np.zeros(
        (samples, candidates, len(EFFECT_FACTORS)), dtype=np.float32
    )
    for factor_index in range(len(EFFECT_FACTORS)):
        factor_target[:, :, factor_index] = (
            np.arange(samples * candidates).reshape(samples, candidates)
            + factor_index
        ) % 2
    return {
        "route_logits": rng.normal(size=(samples, candidates)),
        "candidate_valid_mask": np.ones((samples, candidates), dtype=bool),
        "route_target": route_target,
        "factor_logits": rng.normal(
            size=(samples, candidates, len(EFFECT_FACTORS))
        ),
        "factor_target": factor_target,
        "factor_support_mask": np.ones_like(factor_target, dtype=bool),
    }


def _scores(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: values[name]
        for name in ("route_logits", "candidate_valid_mask", "factor_logits")
    }


def test_effect_calibration_fits_route_and_every_supported_factor() -> None:
    temperature = _predictions(3)
    conformal = _predictions(5)
    calibration = fit_effect_calibration(
        temperature_values=temperature,
        conformal_values=conformal,
        alphas=(0.1, 0.2),
    )
    assert calibration["manual_confidence_thresholds"] is None
    assert all(
        calibration["factors"][name]["status"] == "SUPPORTED"
        for name in EFFECT_FACTORS
    )
    applied = apply_effect_calibration(_scores(conformal), calibration, alpha=0.1)
    assert applied["route_prediction_set"].shape == (8, 2)
    assert applied["factor_prediction_sets"].shape == (
        8,
        2,
        len(EFFECT_FACTORS),
        2,
    )
    with pytest.raises(ValueError, match="evaluator targets"):
        apply_effect_calibration(conformal, calibration, alpha=0.1)


def test_unsupported_effect_factor_stays_masked() -> None:
    temperature = _predictions(7)
    conformal = _predictions(11)
    temperature["factor_target"][:, :, 0] = 1.0
    calibration = fit_effect_calibration(
        temperature_values=temperature,
        conformal_values=conformal,
        alphas=(0.2,),
    )
    assert calibration["factors"][EFFECT_FACTORS[0]]["status"] == "UNSUPPORTED"
    applied = apply_effect_calibration(_scores(conformal), calibration, alpha=0.2)
    assert np.isnan(applied["factor_probability"][:, :, 0]).all()
    assert not applied["factor_prediction_sets"][:, :, 0].any()
