"""Control history must reflect simulator feedback, not the requested trajectory."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "libero_protocol",
    Path(__file__).resolve().parents[1] / "scripts/libero_protocol.py",
)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def test_feedback_records_clipped_controls_and_gripper_without_mutating_prediction():
    prediction = np.array([[2, -3, 0, 0, 0, 0, -0.2], [0, 0, 0, 0, 0, 0, 0.3]])
    applied = protocol.applied_prefix(prediction, -np.ones(7), np.ones(7))
    np.testing.assert_array_equal(applied[0], [1, -1, 0, 0, 0, 0, -1])
    assert applied[1, -1] == 1
    assert prediction[0, 0] == 2 and prediction[0, -1] == -0.2
    packet = {
        "step": 7,
        "applied_actions": [
            {"step": 5 + i, "value": a.tolist()} for i, a in enumerate(applied)
        ],
    }
    np.testing.assert_array_equal(
        protocol.validate_feedback(packet, previous_step=5, requested_count=2), applied
    )
    packet["applied_actions"][0]["step"] = 4
    with pytest.raises(ValueError, match="misaligned"):
        protocol.validate_feedback(packet, previous_step=5, requested_count=2)
    with pytest.raises(ValueError, match="requested execution"):
        protocol.validate_feedback(packet, previous_step=5, requested_count=10)


@pytest.mark.parametrize("actions", [[], [[float("nan")] * 7], [[0] * 6]])
def test_invalid_control_prefix_rejected(actions):
    with pytest.raises(ValueError, match="invalid action"):
        protocol.applied_prefix(actions, -np.ones(7), np.ones(7))
