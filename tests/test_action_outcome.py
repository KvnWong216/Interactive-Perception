import numpy as np
import pytest

from interactive_perception.action_outcome import (
    ActionOutcomePredictor,
    BalancedRidgeMulticlass,
    FeatureStandardizer,
    HierarchicalActionOutcomePredictor,
    PI05_PATCH_EQUIVALENT_TARGET_PIXELS,
    PI05_POLICY_IMAGE_SIDE,
    PI05_VISUAL_TOKEN_GRID_SIDE,
    label_effect_outcome,
    label_observable_information_outcome,
    label_temporal_information_outcome,
    temporal_history_feature_block,
    transition_feature_block,
)
from interactive_perception.active_risk import EffectOutcome
from interactive_perception.semantic_conformal import MondrianSemanticConformalCalibrator


def test_pi05_resolvable_target_area_is_one_visual_token_footprint() -> None:
    assert PI05_POLICY_IMAGE_SIDE == 256
    assert PI05_VISUAL_TOKEN_GRID_SIDE == 16
    assert PI05_PATCH_EQUIVALENT_TARGET_PIXELS == 256


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


def test_temporal_history_separates_full_visual_robot_and_ablations() -> None:
    visual = np.arange(2 * 6 * 10, dtype=float).reshape(2, 6, 10)
    robot = np.arange(2 * 6 * 8, dtype=float).reshape(2, 6, 8)
    no_history = temporal_history_feature_block(visual, robot, "no_history")
    visual_only = temporal_history_feature_block(
        visual, robot, "visual_only_history"
    )
    full = temporal_history_feature_block(visual, robot, "temporal_history")
    assert no_history.shape == (2, 10)
    assert visual_only.shape == (2, 60)
    assert full.shape == (2, 148)
    assert np.array_equal(no_history, visual[:, -1])


def test_temporal_history_rejects_wrong_public_contract() -> None:
    with pytest.raises(ValueError, match="6"):
        temporal_history_feature_block(
            np.zeros((1, 5, 10)), np.zeros((1, 6, 8)), "temporal_history"
        )


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


def test_observable_empty_does_not_read_hidden_target_location() -> None:
    assert label_observable_information_outcome(
        full_executor=True,
        opened=True,
        return_complete=True,
        target_pixels=(0, 0),
        minimum_target_pixels=5,
    ) is EffectOutcome.EMPTY
    assert label_observable_information_outcome(
        full_executor=True,
        opened=True,
        return_complete=False,
        target_pixels=(0, 0),
        minimum_target_pixels=5,
    ) is EffectOutcome.FAILED


def test_temporal_label_retains_evidence_seen_before_final_frame() -> None:
    assert label_temporal_information_outcome(
        full_executor=True,
        opened=True,
        return_complete=False,
        target_pixel_history=((0, 0), (12, 0), (0, 0)),
        minimum_target_pixels=5,
        empty_coverage_certified=False,
    ) is EffectOutcome.REVEALED


def test_temporal_empty_requires_an_independent_coverage_certificate() -> None:
    common = {
        "full_executor": True,
        "opened": True,
        "return_complete": True,
        "target_pixel_history": ((0, 0), (0, 0), (0, 0)),
        "minimum_target_pixels": 5,
    }
    assert label_temporal_information_outcome(
        **common, empty_coverage_certified=False
    ) is EffectOutcome.FAILED
    assert label_temporal_information_outcome(
        **common, empty_coverage_certified=True
    ) is EffectOutcome.EMPTY
    assert label_observable_information_outcome(
        full_executor=False,
        opened=True,
        return_complete=True,
        target_pixels=(100, 0),
        minimum_target_pixels=5,
    ) is EffectOutcome.FAILED


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


def test_hierarchical_predictor_separates_completion_from_content() -> None:
    levels = np.asarray([-5.0, -4.0, 2.0, 3.0, 7.0, 8.0])
    history = np.repeat(levels[:, None, None], 6, axis=1)
    robot = np.zeros((len(levels), 6, 8))
    raw = temporal_history_feature_block(history, robot, "temporal_history")
    scaler = FeatureStandardizer.fit(raw)
    values = scaler.transform(raw)
    effect_labels = np.asarray(
        ["FAILED", "FAILED", "COMPLETED", "COMPLETED", "COMPLETED", "COMPLETED"]
    )
    content_labels = np.asarray(
        ["REVEALED", "REVEALED", "REVEALED", "REVEALED", "EMPTY", "EMPTY"]
    )
    effect = BalancedRidgeMulticlass.fit(values, effect_labels, regularization=1e-3)
    content = BalancedRidgeMulticlass.fit(values, content_labels, regularization=1e-3)
    effect_conformal = MondrianSemanticConformalCalibrator.fit(
        list(zip(effect.evidence(values), effect_labels, strict=True)),
        alpha=0.1,
        policy_id="test-effect",
        split_id="test",
    )
    content_conformal = MondrianSemanticConformalCalibrator.fit(
        list(zip(content.evidence(values), content_labels, strict=True)),
        alpha=0.1,
        policy_id="test-content",
        split_id="test",
    )
    predictor = HierarchicalActionOutcomePredictor(
        input_dimension=1,
        block="temporal_history",
        standardizer=scaler,
        effect_critic=effect,
        effect_conformal=effect_conformal,
        content_critic=content,
        content_conformal=content_conformal,
    )
    assert predictor.predict_history(history[0], robot[0]).prediction_set == (
        "FAILED",
    )
    revealed_set = predictor.predict_history(history[2], robot[2]).prediction_set
    empty_set = predictor.predict_history(history[-1], robot[-1]).prediction_set
    assert "FAILED" not in revealed_set and "REVEALED" in revealed_set
    assert "FAILED" not in empty_set and "EMPTY" in empty_set
