"""Composable information-action executors for the uncertainty router."""

from __future__ import annotations

import collections
import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np

from .observation_option import (
    ObservationReturnConfig,
    ObservationReturnController,
    ObservationReturnStatus,
)
from .policy_client import PolicyBackend, build_observation

__all__ = [
    "OpenAndObserveResult",
    "StepObserver",
    "execute_open_and_observe",
]


class StepEnvironment(Protocol):
    def step(self, action: list[float]) -> tuple[Mapping[str, Any], float, bool, dict]: ...


StepObserver = Callable[[str, int, Mapping[str, Any]], None]


@dataclasses.dataclass(frozen=True)
class OpenAndObserveResult:
    open_steps: int
    open_policy_calls: int
    return_steps: int
    environment_done: bool
    return_status: ObservationReturnStatus

    @property
    def executor_completed(self) -> bool:
        return self.return_status.succeeded and not self.environment_done


def execute_open_and_observe(
    *,
    env: StepEnvironment,
    initial_observation: Mapping[str, Any],
    policy: PolicyBackend,
    open_prompt: str,
    open_steps: int = 300,
    replan_steps: int = 5,
    return_config: ObservationReturnConfig | None = None,
    step_observer: StepObserver | None = None,
) -> tuple[Mapping[str, Any], OpenAndObserveResult]:
    """Execute frozen-VLA opening followed by proprioceptive camera recovery.

    ``step_observer`` exists for evaluator-side logging only.  Its return value
    is ignored, so simulator-private measurements cannot alter execution.
    """

    if open_steps < 0:
        raise ValueError("open_steps must be nonnegative")
    if replan_steps < 1:
        raise ValueError("replan_steps must be positive")

    observation = initial_observation
    plan: collections.deque[np.ndarray] = collections.deque()
    policy_calls = 0
    environment_done = False
    executed_open_steps = 0
    for step in range(open_steps):
        if not plan:
            chunk = policy.sample_chunks(
                build_observation(dict(observation), open_prompt), 1
            )[0]
            plan.extend(chunk[:replan_steps])
            policy_calls += 1
        observation, _, environment_done, _ = env.step(plan.popleft().tolist())
        executed_open_steps += 1
        if step_observer is not None:
            step_observer("OPEN", step, observation)
        if environment_done:
            break

    controller = ObservationReturnController(return_config)
    executed_return_steps = 0
    while not environment_done and not controller.status().terminal:
        action = controller.act(observation)
        observation, _, environment_done, _ = env.step(action.tolist())
        if step_observer is not None:
            step_observer("RETURN_TO_OBSERVE", executed_return_steps, observation)
        executed_return_steps += 1

    return observation, OpenAndObserveResult(
        open_steps=executed_open_steps,
        open_policy_calls=policy_calls,
        return_steps=executed_return_steps,
        environment_done=environment_done,
        return_status=controller.status(),
    )
