from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from grounded_interaction.psr.molmo_backend import ActionChunk
from grounded_interaction.psr.runtime import (
    CandidatePrediction,
    run_closed_loop,
    stable_argmin,
)
from grounded_interaction.psr.types import (
    IntentCandidate,
    PublicHistory,
    PublicObservation,
    RGBReference,
)


def _observation(step: int) -> PublicObservation:
    digest = f"{step + 1:064x}"[-64:]
    return PublicObservation(
        agentview_rgb=RGBReference(f"a-{step}", digest, 8, 8),
        wrist_rgb=RGBReference(f"w-{step}", digest, 8, 8),
        robot_state=(0.0,) * 8,
        control_step=step,
    )


class _Environment:
    def __init__(self, observation: PublicObservation) -> None:
        self.observation = observation

    def step_public(self, action: tuple[float, ...]) -> PublicObservation:
        assert len(action) == 7
        self.observation = _observation(self.observation.control_step + 1)
        return self.observation

    def public_terminal(self) -> bool:
        return False


class _Policy:
    def __init__(self) -> None:
        self.encoded_steps: list[int] = []
        self.chunk_history_steps: list[int] = []
        self.chunk_intents: list[str] = []
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def encode(self, history: PublicHistory) -> types.SimpleNamespace:
        self.encoded_steps.append(history.current.control_step)
        return types.SimpleNamespace(state_tokens=torch.zeros(1, 6, 4))

    def propose(self, encoded, *, episode_seed: int, global_step: int):
        del encoded, episode_seed, global_step
        return [
            IntentCandidate("inspect the object", (1, 2), "conditioned"),
            IntentCandidate("finish the task", (3, 4), "native"),
        ]

    def predict(self, state_tokens, candidates):
        del state_tokens, candidates
        return types.SimpleNamespace()

    def choose(self, candidates, output):
        del output
        probabilities = (0.1, 0.9)
        return candidates[0], tuple(
            CandidatePrediction(item.candidate_id, value)
            for item, value in zip(candidates, probabilities)
        )

    def act_chunk(
        self,
        history: PublicHistory,
        selected_intent: IntentCandidate,
        *,
        rng_seed: int,
        requested_steps: int,
    ) -> ActionChunk:
        del rng_seed
        self.chunk_history_steps.append(history.current.control_step)
        self.chunk_intents.append(selected_intent.candidate_id)
        return ActionChunk(
            actions=tuple((0.0,) * 7 for _ in range(requested_steps)),
            execution_route=selected_intent.execution_route,
            requested_steps=requested_steps,
        )


def test_stable_argmin_prefers_native_then_content_id() -> None:
    conditioned = IntentCandidate("same", (1,), "conditioned")
    native = IntentCandidate("same", (1,), "native")
    assert stable_argmin((conditioned, native), (0.5, 0.5)) == 1
    assert stable_argmin((native, conditioned), (0.5, 0.5)) == 0
    with pytest.raises(ValueError, match="finite"):
        stable_argmin((native,), (float("nan"),))


def test_closed_loop_holds_intent_50_steps_then_replans_remainder() -> None:
    initial = PublicHistory(
        task="retrieve the labelled food",
        current=_observation(0),
        previous=(),
        executed=(),
        remaining_control_steps=53,
    )
    policy = _Policy()
    result = run_closed_loop(
        policy=policy,
        environment=_Environment(initial.current),
        initial_history=initial,
        episode_seed=17,
    )
    assert result.total_steps == 53
    assert len(result.decisions) == 2
    assert [len(chunk) for chunk in result.decisions[0].action_chunks] == [10] * 5
    assert [len(chunk) for chunk in result.decisions[1].action_chunks] == [3]
    assert result.final_history.remaining_control_steps == 0
    assert [
        (item.start_step, item.end_step) for item in result.final_history.executed
    ] == [
        (0, 50),
        (50, 53),
    ]
    assert policy.encoded_steps == [0, 50]
    # Each chunk receives the current real observation; PSRPolicy.act_chunk
    # re-encodes exactly this history before calling the Action Expert.
    assert policy.chunk_history_steps == [0, 10, 20, 30, 40, 50]
    assert len(set(policy.chunk_intents[:5])) == 1
    assert len(result.final_history.previous) == 2


def test_runtime_rejects_protocol_drift() -> None:
    initial = PublicHistory("task", _observation(0), (), (), 1)
    with pytest.raises(ValueError, match="freezes"):
        run_closed_loop(
            policy=_Policy(),
            environment=_Environment(initial.current),
            initial_history=initial,
            episode_seed=1,
            intent_window_steps=49,
        )
