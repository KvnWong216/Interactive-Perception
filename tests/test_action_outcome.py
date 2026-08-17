import numpy as np
import pytest

from interactive_perception.action_outcome import (
    ActionOutcomePredictor,
    BalancedRidgeMulticlass,
    FeatureStandardizer,
    label_effect_outcome,
    transition_feature_block,
)
from interactive_perception.active_risk import EffectOutcome
from interactive_perception.semantic_conformal import MondrianSemanticConformalCalibrator


def test_multiclass_ridge_separates_three_outcomes_and_roundtrips() -> None:
    features = np.asarray(
        [
            [3.0, 0.0, 0.0],
            [2.0, 0.1, 0.0],
            [0.0, 3.0, 0.0],
            [0.1, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.1, 2.0],
        ]
    )
    labels = ["FAILED", "FAILED", "EMPTY", "EMPTY", "REVEALED", "REVEALED"]
    model = BalancedRidgeMulticlass.fit(features, labels, regularization=1e-3)
    assert model.predict(features).tolist() == labels
    restored = BalancedRidgeMulticlass.from_dict(model.to_dict())
    assert restored.predict(features).tolist() == labels
    assert all(sum(row.values()) == pytest.approx(1.0) for row in model.evidence(features))


def test_transition_feature_uses_before_after_and_delta() -> None:
    before = np.zeros((2, 8192))
    after = np.ones((2, 8192))
    value = transition_feature_block(before, after, "prompt_history")
    assert value.shape == (2, 6144)
    assert np.all(value[:, :2048] == 0.0)
    assert np.all(value[:, 2048:] == 1.0)


def test_generic_and_spatial_transition_features() -> None:
    before = np.zeros((2, 15248))
    after = np.ones((2, 15248))
    assert transition_feature_block(before, after, "after").shape == (2, 15248)
    assert transition_feature_block(before, after, "delta").shape == (2, 15248)
    assert transition_feature_block(before, after, "history").shape == (2, 45744)
    assert transition_feature_block(before, after, "spatial_after").shape == (2, 7056)
    assert transition_feature_block(before, after, "spatial_delta").shape == (2, 7056)
    assert transition_feature_block(before, after, "spatial_history").shape == (2, 21168)


def test_transition_feature_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="8192"):
        transition_feature_block(np.zeros((1, 2)), np.zeros((1, 2)), "after_prompt")


def test_feature_standardizer_fits_training_values_and_roundtrips() -> None:
    values = np.asarray([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    standardizer = FeatureStandardizer.fit(values)
    transformed = standardizer.transform(values)
    assert np.mean(transformed[:, 0]) == pytest.approx(0.0)
    assert np.std(transformed[:, 0]) == pytest.approx(1.0)
    assert np.all(transformed[:, 1] == 0.0)
    restored = FeatureStandardizer.from_dict(standardizer.to_dict())
    assert restored.transform(values) == pytest.approx(transformed)


def test_single_aliasing_pixel_is_not_revealed_information() -> None:
    assert label_effect_outcome(
        opened=True,
        target_in_resolved_location=True,
        target_pixels=(0, 1),
        minimum_target_pixels=5,
    ) is EffectOutcome.FAILED
    assert label_effect_outcome(
        opened=True,
        target_in_resolved_location=True,
        target_pixels=(5, 0),
        minimum_target_pixels=5,
    ) is EffectOutcome.REVEALED


def test_opened_wrong_location_is_empty() -> None:
    assert label_effect_outcome(
        opened=True,
        target_in_resolved_location=False,
        target_pixels=(0, 0),
        minimum_target_pixels=5,
    ) is EffectOutcome.EMPTY


def test_action_outcome_predictor_consumes_paired_prefixes() -> None:
    values = np.zeros((4, 2048))
    values[0, 0] = 3.0
    values[1, :2] = (2.0, 0.1)
    values[2, 1] = 3.0
    values[3, :2] = (0.1, 2.0)
    labels = np.asarray(["FAILED", "FAILED", "REVEALED", "REVEALED"])
    critic = BalancedRidgeMulticlass.fit(values, labels, regularization=1e-3)
    evidence = critic.evidence(values)
    conformal = MondrianSemanticConformalCalibrator.fit(
        list(zip(evidence, labels, strict=True)),
        alpha=0.1,
        policy_id="test",
        split_id="test",
    )
    predictor = ActionOutcomePredictor(8192, "after_prompt", critic, conformal)
    before = np.zeros(8192)
    after = np.zeros(8192)
    after[2048] = 3.0
    assert predictor.predict(before, after).prediction_set == ("FAILED",)


def test_action_outcome_predictor_applies_frozen_standardizer() -> None:
    raw = np.zeros((4, 2048))
    raw[:, :2] = np.asarray([[1.0, 10.0], [2.0, 10.0], [8.0, 20.0], [9.0, 20.0]])
    labels = np.asarray(["EMPTY", "EMPTY", "REVEALED", "REVEALED"])
    standardizer = FeatureStandardizer.fit(raw)
    critic = BalancedRidgeMulticlass.fit(
        standardizer.transform(raw), labels, regularization=1e-3
    )
    evidence = critic.evidence(standardizer.transform(raw))
    conformal = MondrianSemanticConformalCalibrator.fit(
        list(zip(evidence, labels, strict=True)),
        alpha=0.1,
        policy_id="test",
        split_id="test",
    )
    predictor = ActionOutcomePredictor(
        8192, "after_prompt", critic, conformal, standardizer
    )
    before = np.zeros(8192)
    after = np.zeros(8192)
    after[2048:4096] = raw[-1]
    artifact = {
        "input_feature_dimension": 8192,
        "model_selection": {"selected": {"block": "after_prompt"}},
        "critic": critic.to_dict(),
        "conformal": conformal.to_dict(),
        "feature_standardizer": standardizer.to_dict(),
    }
    restored = ActionOutcomePredictor.from_artifact(artifact)
    assert predictor.predict(before, after) == restored.predict(before, after)
