import pytest

from interactive_perception.semantic_conformal import (
    MondrianSemanticConformalCalibrator,
    SemanticConformalCalibrator,
    intent_probabilities,
)


LABELS = {"ACT": 0.0, "REMOVE_OCCLUDER": 0.0, "NOT_FOUND": 0.0}


def evidence(**values):
    return {**LABELS, **values}


def test_zero_evidence_preserves_complete_ignorance() -> None:
    assert intent_probabilities(LABELS) == pytest.approx(
        {"ACT": 1 / 3, "REMOVE_OCCLUDER": 1 / 3, "NOT_FOUND": 1 / 3}
    )


def test_conformal_set_groups_semantically_equivalent_action_samples() -> None:
    examples = [
        (evidence(REMOVE_OCCLUDER=9, ACT=1), "REMOVE_OCCLUDER"),
        (evidence(REMOVE_OCCLUDER=8, ACT=2), "REMOVE_OCCLUDER"),
        (evidence(ACT=9, REMOVE_OCCLUDER=1), "ACT"),
        (evidence(ACT=8, REMOVE_OCCLUDER=2), "ACT"),
    ]
    calibrator = SemanticConformalCalibrator.fit(
        examples, alpha=0.25, policy_id="pi05-libero@sha", split_id="heldout-v1"
    )
    assert calibrator.predict(evidence(REMOVE_OCCLUDER=9, ACT=1)) == (
        "REMOVE_OCCLUDER",
    )
    assert set(calibrator.predict(evidence(REMOVE_OCCLUDER=5, ACT=5))) == {
        "ACT",
        "REMOVE_OCCLUDER",
    }


def test_artifact_records_scope_and_non_guarantee() -> None:
    calibrator = SemanticConformalCalibrator.fit(
        [(evidence(ACT=1), "ACT")],
        alpha=0.1,
        policy_id="pi05",
        split_id="heldout",
    )
    artifact = calibrator.to_dict()
    assert artifact["policy_id"] == "pi05"
    assert artifact["split_id"] == "heldout"
    assert artifact["non_guarantee"] == "robot task success"


def test_mondrian_uses_class_specific_thresholds() -> None:
    examples = [
        (evidence(ACT=9, REMOVE_OCCLUDER=1), "ACT"),
        (evidence(ACT=8, REMOVE_OCCLUDER=2), "ACT"),
        (evidence(ACT=4, REMOVE_OCCLUDER=6), "REMOVE_OCCLUDER"),
        (evidence(ACT=3, REMOVE_OCCLUDER=7), "REMOVE_OCCLUDER"),
        (evidence(NOT_FOUND=9, ACT=1), "NOT_FOUND"),
        (evidence(NOT_FOUND=8, ACT=2), "NOT_FOUND"),
    ]
    calibrator = MondrianSemanticConformalCalibrator.fit(
        examples, alpha=0.25, policy_id="pi05", split_id="mondrian-v1"
    )
    artifact = calibrator.to_dict()
    assert set(artifact["thresholds"]) == set(LABELS)
    assert artifact["calibration_size_per_class"] == {
        "ACT": 2,
        "NOT_FOUND": 2,
        "REMOVE_OCCLUDER": 2,
    }
    restored = MondrianSemanticConformalCalibrator.from_dict(artifact)
    assert restored.predict(evidence(ACT=9, REMOVE_OCCLUDER=1)) == ("ACT",)
