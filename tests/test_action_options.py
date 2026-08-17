from __future__ import annotations

import numpy as np

from interactive_perception.action_options import execute_open_and_observe
from interactive_perception.observation_option import (
    ObservationPose,
    ObservationReturnConfig,
    ObservationReturnPhase,
)
from interactive_perception.policy_client import ScriptedStubPolicy


class _FakeEnvironment:
    def __init__(self, position=(0.15, 0.0, 0.0)) -> None:
        self.position = np.asarray(position, dtype=np.float64)
        self.steps = 0

    def observation(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        return {
            "agentview_image": image,
            "robot0_eye_in_hand_image": image,
            "robot0_eef_pos": self.position.copy(),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.asarray([0.02, -0.02]),
        }

    def step(self, action):
        self.steps += 1
        self.position += np.asarray(action[:3], dtype=np.float64) * 0.05
        return self.observation(), 0.0, False, {}


def _config(**overrides):
    values = {
        "pose": ObservationPose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        "release_steps": 1,
        "settled_steps": 2,
        "maximum_return_steps": 30,
    }
    values.update(overrides)
    return ObservationReturnConfig(**values)


def test_composite_executor_opens_then_returns_without_oracle_feedback() -> None:
    env = _FakeEnvironment()
    phases = []
    final_observation, result = execute_open_and_observe(
        env=env,
        initial_observation=env.observation(),
        policy=ScriptedStubPolicy(
            goal_position=(0.15, 0.0, 0.0), horizon=4, noise_scale=0.0
        ),
        open_prompt="Open the middle layer of the drawer",
        open_steps=3,
        replan_steps=2,
        return_config=_config(),
        step_observer=lambda phase, step, obs: phases.append(phase),
    )
    assert result.open_steps == 3
    assert result.open_policy_calls == 2
    assert result.return_status.phase is ObservationReturnPhase.COMPLETE
    assert result.executor_completed
    assert phases[:3] == ["OPEN"] * 3
    assert set(phases[3:]) == {"RETURN_TO_OBSERVE"}
    assert np.linalg.norm(final_observation["robot0_eef_pos"]) <= 0.012


def test_return_timeout_is_propagated_as_executor_failure() -> None:
    env = _FakeEnvironment(position=(1.0, 0.0, 0.0))
    _, result = execute_open_and_observe(
        env=env,
        initial_observation=env.observation(),
        policy=ScriptedStubPolicy(horizon=1, noise_scale=0.0),
        open_prompt="Open",
        open_steps=0,
        return_config=_config(maximum_return_steps=1),
    )
    assert result.return_status.phase is ObservationReturnPhase.TIMED_OUT
    assert not result.executor_completed
