import numpy as np
import pytest

from interactive_perception.prompt_state import (
    BalancedRidgeBinary,
    GaussianScoreCalibrator,
    PromptStatePredictor,
    normalize_rows,
)
from interactive_perception.semantic_conformal import MondrianSemanticConformalCalibrator


def test_balanced_ridge_separates_and_round_trips():
    features = np.asarray([[-2.0, 0.0], [-1.0, 0.1], [1.0, -0.1], [2.0, 0.0]])
    labels = ["HIDDEN", "HIDDEN", "OBSERVED", "OBSERVED"]
    model = BalancedRidgeBinary.fit(
        features,
        labels,
        negative_label="HIDDEN",
        positive_label="OBSERVED",
        regularization=0.01,
    )
    restored = BalancedRidgeBinary.from_dict(model.to_dict())
    assert restored.predict(features).tolist() == labels
    assert np.allclose(restored.score(features), model.score(features))


def test_gaussian_score_calibration_is_normalized_and_ordered():
    model = GaussianScoreCalibrator.fit(
        [-2.0, -1.0, 1.0, 2.0],
        ["HIDDEN", "HIDDEN", "OBSERVED", "OBSERVED"],
        ordered_labels=("HIDDEN", "OBSERVED"),
    )
    probabilities = model.probabilities([-1.5, 1.5])
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[0, 0] > probabilities[0, 1]
    assert probabilities[1, 1] > probabilities[1, 0]
    assert GaussianScoreCalibrator.from_dict(model.to_dict()) == model


def test_prompt_state_inputs_are_strictly_validated():
    with pytest.raises(ValueError, match="shape"):
        normalize_rows(np.asarray([1.0, 2.0]))
    with pytest.raises(ValueError, match="both labels"):
        BalancedRidgeBinary.fit(
            np.ones((2, 2)),
            ["OBSERVED", "OBSERVED"],
            negative_label="HIDDEN",
            positive_label="OBSERVED",
            regularization=0.1,
        )


def test_prompt_state_predictor_consumes_one_frozen_prefix():
    features = np.asarray([[-2.0, 0.0], [-1.0, 0.1], [1.0, -0.1], [2.0, 0.0]])
    labels = np.asarray(["HIDDEN", "HIDDEN", "VISIBLE", "VISIBLE"])
    probe = BalancedRidgeBinary.fit(
        features,
        labels,
        negative_label="HIDDEN",
        positive_label="VISIBLE",
        regularization=0.01,
    )
    probability_model = GaussianScoreCalibrator.fit(
        probe.score(features), labels, ordered_labels=("HIDDEN", "VISIBLE")
    )
    evidence = probability_model.evidence(probe.score(features))
    conformal = MondrianSemanticConformalCalibrator.fit(
        list(zip(evidence, labels, strict=True)),
        alpha=0.1,
        policy_id="test",
        split_id="test",
    )
    predictor = PromptStatePredictor(0, 2, probe, probability_model, conformal)
    prefix = np.zeros(8192)
    prefix[0] = 2.0
    assert predictor.predict(prefix).prediction_set == ("VISIBLE",)
