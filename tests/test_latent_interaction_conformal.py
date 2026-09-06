import numpy as np
import pytest

from latent_interaction.conformal import SplitConformalPrimitiveSet


def test_conformal_round_trip_and_singleton_rule() -> None:
    probabilities = np.array(
        [
            [0.92, 0.04, 0.04],
            [0.04, 0.91, 0.05],
            [0.06, 0.05, 0.89],
            [0.88, 0.07, 0.05],
            [0.05, 0.90, 0.05],
        ]
    )
    calibrator = SplitConformalPrimitiveSet.fit(
        probabilities,
        [0, 1, 2, 0, 1],
        labels=("ACT", "OPEN", "STOP"),
        alpha=0.2,
        split_id="scene-disjoint-calibration-v1",
    )
    restored = SplitConformalPrimitiveSet.from_dict(calibrator.to_dict())
    assert restored == calibrator
    decision = restored.decide([0.96, 0.02, 0.02])
    assert decision.prediction_set == ("ACT",)
    assert decision.selected_primitive == "ACT"
    assert decision.abstain is False


def test_non_singleton_conformal_set_abstains() -> None:
    calibrator = SplitConformalPrimitiveSet(
        alpha=0.1,
        threshold=0.75,
        labels=("ACT", "OPEN", "STOP"),
        calibration_size=20,
        split_id="cal",
    )
    decision = calibrator.decide([0.40, 0.35, 0.25])
    assert len(decision.prediction_set) == 3
    assert decision.selected_primitive is None
    assert decision.abstain is True
    assert decision.reason == "ambiguous_set"


def test_serialization_rejects_wrong_schema() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        SplitConformalPrimitiveSet.from_dict(
            {
                "schema_version": "wrong",
                "alpha": 0.1,
                "threshold": 0.5,
                "labels": ["ACT"],
                "calibration_size": 1,
                "split_id": "cal",
            }
        )

