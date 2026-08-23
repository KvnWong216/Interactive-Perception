from __future__ import annotations

import numpy as np

from piu.binding_calibration import (
    MondrianBinaryLAC,
    binary_probability_metrics,
    finite_sample_quantile,
    fit_binary_temperature,
    prediction_set_metrics,
    spatial_conformal_fit,
    spatial_prediction_set_metrics,
    spatial_prediction_sets,
)


def test_temperature_scaling_uses_a_proper_score() -> None:
    logits = np.asarray([-6.0, -2.0, 2.0, 6.0])
    truth = np.asarray([0.0, 1.0, 0.0, 1.0])
    temperature = fit_binary_temperature(logits, truth)
    before = binary_probability_metrics(logits, truth, temperature=1.0)
    after = binary_probability_metrics(logits, truth, temperature=temperature)
    assert temperature > 0.0
    assert after["nll"] <= before["nll"]


def test_finite_sample_quantile_saturates_instead_of_undercovering() -> None:
    fitted = finite_sample_quantile(
        np.asarray([0.2, 0.4]), alpha=0.1, maximum_score=1.0
    )
    assert fitted["rank"] == 3
    assert fitted["finite_sample_saturated"] is True
    assert fitted["quantile"] == 1.0


def test_mondrian_binary_sets_report_coverage_and_ambiguity() -> None:
    probability = np.asarray([0.1, 0.3, 0.7, 0.9])
    truth = np.asarray([0, 0, 1, 1])
    calibrator = MondrianBinaryLAC.fit(probability, truth, alpha=0.5)
    metrics = prediction_set_metrics(calibrator.predict(probability), truth)
    assert metrics["coverage"] == 1.0
    assert 0.0 <= metrics["singleton_rate"] <= 1.0


def test_spatial_conformal_event_is_target_patch_intersection() -> None:
    probability = np.asarray(
        [[0.6, 0.2, 0.1, 0.1], [0.1, 0.2, 0.6, 0.1]], dtype=np.float64
    )
    target = np.asarray([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
    fitted = spatial_conformal_fit(probability, target, alpha=0.5)
    sets = spatial_prediction_sets(
        probability, np.ones_like(probability, dtype=bool), fitted
    )
    metrics = spatial_prediction_set_metrics(sets, target)
    assert metrics["target_intersection_coverage"] == 1.0
    assert metrics["mean_set_size"] == 1.0
