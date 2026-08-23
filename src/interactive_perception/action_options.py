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
from .policy_client import ObservationPacket, PolicyBackend, build_observation

__all__ = [
    "OpenAndObserveResult",
    "SemanticSubtaskHomeResult",
    "StepObserver",
    "execute_open_and_observe",
    "execute_subtask_and_return_home",
]


class StepEnvironment(Protocol):
    def step(
        self, action: list[float]
    ) -> tuple[Mapping[str, Any], float, bool, dict]: ...


StepObserver = Callable[[str, int, Mapping[str, Any]], None]
PolicyObservationBuilder = Callable[[Mapping[str, Any], str], ObservationPacket]


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


@dataclasses.dataclass(frozen=True)
class SemanticSubtaskHomeResult:
    """Execution record for the canonical policy handoff contract.

    The semantic subtask runs first.  Its public observations can be retained
    by ``step_observer``.  Only then does a proprioceptive controller return to
    the exact registered home pose.  Consequently the next planner call can
    use the full visual/action history while every frozen-policy subtask starts
    from the same pose distribution.
    """

    subtask_steps: int
    policy_calls: int
    return_steps: int
    environment_done: bool
    return_status: ObservationReturnStatus
    completion_source: str

    @property
    def executor_completed(self) -> bool:
        return self.return_status.succeeded and not self.environment_done


SubtaskCompletionMonitor = Callable[[int, Mapping[str, Any]], bool]


def execute_subtask_and_return_home(
    *,
    env: StepEnvironment,
    initial_observation: Mapping[str, Any],
    policy: PolicyBackend,
    subtask_prompt: str,
    maximum_subtask_steps: int,
    replan_steps: int = 5,
    home_config: ObservationReturnConfig,
    completion_monitor: SubtaskCompletionMonitor | None = None,
    step_observer: StepObserver | None = None,
    policy_observation_builder: PolicyObservationBuilder = build_observation,
) -> tuple[Mapping[str, Any], SemanticSubtaskHomeResult]:
    """Execute one semantic subtask, retain evidence, then return home.

    ``completion_monitor`` may inspect public observations only.  It is the
    preferred way to end a subtask; ``maximum_subtask_steps`` is a safety
    budget, not evidence of completion.  The callback and observer return
    values cannot affect the home controller's goal pose.
    """

    if maximum_subtask_steps < 0:
        raise ValueError("maximum_subtask_steps must be nonnegative")
    if replan_steps < 1:
        raise ValueError("replan_steps must be positive")
    if not subtask_prompt.strip():
        raise ValueError("subtask_prompt must be nonempty")
    if home_config.alternative_completion_poses:
        raise ValueError("home handoff cannot accept alternative completion poses")

    observation = initial_observation
    plan: collections.deque[np.ndarray] = collections.deque()
    policy_calls = 0
    environment_done = False
    executed_steps = 0
    completion_source = "SAFETY_BUDGET"
    for step in range(maximum_subtask_steps):
        if not plan:
            chunk = policy.sample_chunks(
                policy_observation_builder(observation, subtask_prompt), 1
            )[0]
            plan.extend(chunk[:replan_steps])
            policy_calls += 1
        observation, _, environment_done, _ = env.step(plan.popleft().tolist())
        executed_steps += 1
        if step_observer is not None:
            step_observer("SUBTASK", step, observation)
        if environment_done:
            completion_source = "ENVIRONMENT_TERMINATED"
            break
        if completion_monitor is not None and completion_monitor(step, observation):
            completion_source = "PUBLIC_COMPLETION_MONITOR"
            break

    controller = ObservationReturnController(home_config)
    executed_return_steps = 0
    while not environment_done and not controller.status().terminal:
        action = controller.act(observation)
        observation, _, environment_done, _ = env.step(action.tolist())
        if step_observer is not None:
            step_observer("RETURN_HOME", executed_return_steps, observation)
        executed_return_steps += 1

    return observation, SemanticSubtaskHomeResult(
        subtask_steps=executed_steps,
        policy_calls=policy_calls,
        return_steps=executed_return_steps,
        environment_done=environment_done,
        return_status=controller.status(),
        completion_source=completion_source,
    )


def execute_open_and_observe(
    *,
    env: StepEnvironment,
    initial_observation: Mapping[str, Any],
    policy: PolicyBackend,
    open_prompt: str,
    return_config: ObservationReturnConfig,
    open_steps: int = 300,
    replan_steps: int = 5,
    step_observer: StepObserver | None = None,
    policy_observation_builder: PolicyObservationBuilder = build_observation,
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
                policy_observation_builder(observation, open_prompt), 1
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
