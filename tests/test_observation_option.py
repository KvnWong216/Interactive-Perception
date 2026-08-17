from __future__ import annotations

import math

import numpy as np
import pytest

from interactive_perception.observation_option import (
    ObservationPose,
    ObservationReturnConfig,
    ObservationReturnController,
    ObservationReturnPhase,
    relative_axis_angle_xyzw,
)


def _quaternion_xyzw(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([axis * math.sin(angle / 2.0), [math.cos(angle / 2.0)]])


def _observation(position=(0.0, 0.0, 0.0), quaternion=(0.0, 0.0, 0.0, 1.0)):
    return {
        "robot0_eef_pos": np.asarray(position, dtype=np.float64),
        "robot0_eef_quat": np.asarray(quaternion, dtype=np.float64),
        "robot0_gripper_qpos": np.asarray([0.02, -0.02]),
    }


def test_relative_axis_angle_is_sign_invariant_and_shortest() -> None:
    desired = _quaternion_xyzw([0.0, 0.0, 1.0], math.pi / 3.0)
    expected = np.asarray([0.0, 0.0, math.pi / 3.0])
    assert np.allclose(relative_axis_angle_xyzw([0, 0, 0, 1], desired), expected)
    assert np.allclose(relative_axis_angle_xyzw([0, 0, 0, -1], -desired), expected)


def test_first_phase_only_releases_gripper() -> None:
    config = ObservationReturnConfig(
        pose=ObservationPose((0.2, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        release_steps=2,
    )
    controller = ObservationReturnController(config)
    first = controller.act(_observation())
    second = controller.act(_observation())
    assert np.array_equal(first, [0, 0, 0, 0, 0, 0, -1])
    assert np.array_equal(second, first)
    assert controller.status().phase is ObservationReturnPhase.RETURN


def test_return_action_is_scaled_clipped_and_keeps_gripper_open() -> None:
    desired_quaternion = _quaternion_xyzw([0.0, 1.0, 0.0], 0.5)
    config = ObservationReturnConfig(
        pose=ObservationPose((0.2, -0.01, 0.0), tuple(desired_quaternion)),
        release_steps=0,
        maximum_normalized_translation=0.6,
        maximum_normalized_rotation=0.5,
    )
    action = ObservationReturnController(config).act(_observation())
    assert np.allclose(action[:3], [0.6, -0.2, 0.0])
    assert np.allclose(action[3:6], [0.0, 0.5, 0.0])
    assert action[6] == -1.0


def test_completion_requires_consecutive_settled_observations() -> None:
    config = ObservationReturnConfig(
        pose=ObservationPose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        release_steps=0,
        settled_steps=3,
    )
    controller = ObservationReturnController(config)
    for expected in (1, 2, 3):
        action = controller.act(_observation())
        assert np.array_equal(action, [0, 0, 0, 0, 0, 0, -1])
        assert controller.status().consecutive_settled_steps == expected
    assert controller.status().succeeded
    with pytest.raises(RuntimeError, match="already terminal"):
        controller.act(_observation())


def test_timeout_is_explicit_failure_not_false_completion() -> None:
    config = ObservationReturnConfig(
        pose=ObservationPose((1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        release_steps=0,
        maximum_return_steps=2,
    )
    controller = ObservationReturnController(config)
    controller.act(_observation())
    controller.act(_observation())
    assert controller.status().phase is ObservationReturnPhase.TIMED_OUT
    assert not controller.status().succeeded


class _NoOracleObservation(dict):
    def __getitem__(self, key):
        if key not in ObservationReturnController._ALLOWED_OBSERVATION_KEYS:
            raise AssertionError(f"attempted evaluator-private read: {key}")
        return super().__getitem__(key)


def test_controller_reads_only_declared_policy_visible_fields() -> None:
    observation = _NoOracleObservation(_observation())
    observation["target_position"] = np.ones(3)
    observation["segmentation"] = np.ones((4, 4))
    controller = ObservationReturnController(
        ObservationReturnConfig(
            pose=ObservationPose((0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            release_steps=0,
        )
    )
    controller.act(observation)
    assert controller.allowed_observation_keys == {
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    }


def test_configuration_rejects_routing_like_or_invalid_values() -> None:
    with pytest.raises(ValueError, match="maximum_normalized_translation"):
        ObservationReturnConfig(maximum_normalized_translation=1.1)
    with pytest.raises(ValueError, match="normalized"):
        ObservationPose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 2.0))
