"""Direct continuous policy, real prefix feedback, and native language completion."""

from dataclasses import dataclass

import numpy as np

from .types import AppliedAction, PolicyContext


@dataclass(frozen=True)
class Response:
    done: bool
    answer: str = ""


def parse_response(text):
    """Only a complete, explicit native-language response can stop the robot."""
    text = text.strip()
    if text == "CONTINUE":
        return Response(False)
    if text.startswith("DONE:") and text[5:].strip():
        return Response(True, text[5:].strip())
    raise ValueError("language output must be CONTINUE or DONE: <answer>")


@dataclass(frozen=True)
class Rollout:
    context: PolicyContext
    answer: str
    reason: str
    actions: tuple[AppliedAction, ...]
    responses: tuple[str, ...]


def run_episode(*, policy, environment, context, config, seed=None, allow_answer=True):
    """environment.step returns (new public observation, actual applied action).

    Private task success is deliberately absent. An answer is a policy claim,
    not evaluator success. Environment failures propagate to the caller.
    """
    if context.current.step + context.remaining_steps != config.total_steps:
        raise ValueError("context must preserve the original episode clock")
    if seed is None:
        seed = config.manual_seed
    policy.reset()
    executed, responses = [], []
    while True:
        if allow_answer:
            text = policy.respond(context)
            responses.append(text)
            # Invalid language is retained for auditing and cannot stop execution.
            try:
                response = parse_response(text)
            except ValueError:
                response = Response(False)
            if response.done:
                return Rollout(
                    context,
                    response.answer,
                    "policy_done",
                    tuple(executed),
                    tuple(responses),
                )
        if context.remaining_steps == 0:
            return Rollout(
                context, "", "budget_exhausted", tuple(executed), tuple(responses)
            )
        steps = min(config.execute_steps, context.remaining_steps)
        action_seed = seed + context.current.step
        chunk = np.asarray(
            policy.act(context, seed=action_seed, steps=steps), dtype=np.float32
        )
        if chunk.shape != (steps, 7) or not np.isfinite(chunk).all():
            raise RuntimeError("policy must return the requested finite 7-D prefix")
        actual = []
        for action in chunk:
            observation, applied = environment.step(action)
            step = context.current.step + len(actual)
            if observation.step != step + 1:
                raise RuntimeError("environment feedback is not contiguous")
            actual.append(AppliedAction(step, applied))
        context = context.advance(observation, actual, max_frames=config.history_frames)
        executed.extend(actual)
