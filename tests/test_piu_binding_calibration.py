from __future__ import annotations

import numpy as np
import pytest

from piu.binding_calibration import (
    MondrianBinaryLAC,
    MulticlassLAC,
    apply_binding_calibration,
    binary_probability_metrics,
    finite_sample_quantile,
    fit_binary_temperature,
    fit_multiclass_temperature,
    multiclass_prediction_set_metrics,
    multiclass_probability_metrics,
    prediction_set_metrics,
    spatial_conformal_fit,
    spatial_prediction_set_metrics,
    spatial_prediction_sets,
    spatial_probabilities,
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


def test_dynamic_candidate_route_temperature_and_conformal_set() -> None:
    logits = np.asarray([[3.0, 0.0, -np.inf], [0.0, 3.0, -np.inf], [2.0, 0.0, 1.0]])
    valid = np.asarray([[True, True, False], [True, True, False], [True, True, True]])
    target = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float64)
    temperature = fit_multiclass_temperature(logits, valid, target)
    before = multiclass_probability_metrics(logits, valid, target, temperature=1.0)
    after = multiclass_probability_metrics(
        logits, valid, target, temperature=temperature
    )
    assert after["nll"] <= before["nll"]
    probabilities = spatial_probabilities(logits, valid, temperature=temperature)
    calibrator = MulticlassLAC.fit(probabilities, valid, target, alpha=0.5)
    metrics = multiclass_prediction_set_metrics(
        calibrator.predict(probabilities, valid), target
    )
    assert metrics["coverage"] == 1.0


def test_online_binding_calibration_needs_no_truth_arrays() -> None:
    binary = MondrianBinaryLAC.fit(
        np.asarray([0.1, 0.9]), np.asarray([0, 1]), alpha=0.5
    )
    calibration = {
        "spatial": {
            "temperature": 1.0,
            "conformal": {
                "0.5": {
                    "calibrator": {
                        "method": "target_intersection_split_conformal",
                        "quantile": 0.5,
                    }
                }
            },
        },
        "target_presence": {
            "status": "SUPPORTED",
            "temperature": 1.0,
            "conformal": {"0.5": {"calibrator": binary.to_dict()}},
        },
        "task_sufficiency": {
            "status": "UNSUPPORTED",
        },
        "holding_requested_target": {
            "status": "UNSUPPORTED",
        },
        "region_confirmed_empty": {"status": "UNSUPPORTED"},
        "task_complete": {"status": "UNSUPPORTED"},
    }
    result = apply_binding_calibration(
        {
            "spatial_logits": np.asarray([[2.0, 0.0]]),
            "image_valid_mask": np.asarray([[True, True]]),
            "target_present_logit": np.asarray([2.0]),
            "task_sufficiency_logit": np.asarray([0.0]),
            "holding_requested_target_logit": np.asarray([0.0]),
            "region_confirmed_empty_logit": np.asarray([0.0]),
            "task_complete_logit": np.asarray([0.0]),
        },
        calibration,
        alpha=0.5,
    )
    assert result["spatial_prediction_set"].shape == (1, 2)
    assert result["target_presence_prediction_set"].shape == (1, 2)
    assert not result["task_sufficiency_prediction_set"].any()
    assert not result["holding_requested_target_prediction_set"].any()
    with pytest.raises(ValueError, match="evaluator targets"):
        apply_binding_calibration(
            {
                "spatial_logits": np.asarray([[2.0, 0.0]]),
                "image_valid_mask": np.asarray([[True, True]]),
                "target_present_logit": np.asarray([2.0]),
                "task_sufficiency_logit": np.asarray([0.0]),
                "holding_requested_target_logit": np.asarray([0.0]),
                "region_confirmed_empty_logit": np.asarray([0.0]),
                "task_complete_logit": np.asarray([0.0]),
                "patch_target": np.asarray([[1.0, 0.0]]),
            },
            calibration,
            alpha=0.5,
        )
