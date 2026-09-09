"""Closed-loop PSR-VLA v1 decision and execution semantics.

The runtime holds one selected open-vocabulary intent for at most 50 control
steps, replans a native 10-step action chunk after each real reobservation, and
never resets the original 300-step task budget.  It contains no simulator
success predicate and never feeds predicted future evidence back into H.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from .model import PredictiveOutput, PredictiveStateModel
from .molmo_backend import ActionChunk, EncodedHistory, MolmoPSRBackend
from .types import ExecutedIntent, IntentCandidate, PublicHistory, PublicObservation


class PublicStepEnvironment(Protocol):
    """Only deployment-visible state crosses this boundary."""

    def step_public(self, action: Sequence[float]) -> PublicObservation: ...

    def public_terminal(self) -> bool: ...


@dataclasses.dataclass(frozen=True)
class CandidatePrediction:
    candidate_id: str
    failure_probability: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.failure_probability) or not (
            0.0 <= self.failure_probability <= 1.0
        ):
            raise ValueError("failure probability must be finite and in [0,1]")


@dataclasses.dataclass(frozen=True)
class DecisionRecord:
    decision_step: int
    remaining_before: int
    candidates: tuple[IntentCandidate, ...]
    predictions: tuple[CandidatePrediction, ...]
    selected_candidate_id: str
    action_chunks: tuple[tuple[tuple[float, ...], ...], ...]
    observations: tuple[PublicObservation, ...]
    remaining_after: int
    terminal_after: bool


@dataclasses.dataclass(frozen=True)
class ClosedLoopResult:
    final_history: PublicHistory
    decisions: tuple[DecisionRecord, ...]
    total_steps: int
    terminal: bool


def _action_reference(actions: Sequence[Sequence[float]]) -> str:
    payload = json.dumps(
        [[float(value) for value in row] for row in actions],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def stable_argmin(
    candidates: Sequence[IntentCandidate], probabilities: Sequence[float]
) -> int:
    """Rank by E, then native route, then stable content identity."""

    if len(candidates) != len(probabilities) or not candidates:
        raise ValueError(
            "candidate and probability arrays must be non-empty and aligned"
        )
    rows = []
    for index, (candidate, probability) in enumerate(zip(candidates, probabilities)):
        probability = float(probability)
        if not math.isfinite(probability):
            raise ValueError("PSR mode refuses non-finite candidate predictions")
        rows.append(
            (
                probability,
                0 if candidate.execution_route == "native" else 1,
                candidate.candidate_id,
                index,
            )
        )
    return min(rows)[-1]


class PSRPolicy:
    """Concrete composition of the in-process Molmo bridge and E/C readout."""

    def __init__(
        self,
        *,
        backend: MolmoPSRBackend,
        predictor: PredictiveStateModel,
        calibration_temperature: float = 1.0,
    ) -> None:
        if not math.isfinite(calibration_temperature) or calibration_temperature <= 0:
            raise ValueError("calibration_temperature must be positive")
        self.backend = backend
        self.predictor = predictor
        self.calibration_temperature = float(calibration_temperature)
        self._last_prediction: PredictiveOutput | None = None

    def encode(self, history: PublicHistory) -> EncodedHistory:
        if not isinstance(history, PublicHistory):
            raise TypeError("history must be PublicHistory")
        return self.backend.encode(history)

    def propose(
        self,
        encoded: EncodedHistory,
        *,
        episode_seed: int,
        global_step: int,
    ) -> list[IntentCandidate]:
        return self.backend.propose(
            encoded, episode_seed=episode_seed, global_step=global_step
        )

    def predict(
        self,
        state_tokens: Any,
        candidates: Sequence[IntentCandidate],
    ) -> PredictiveOutput:
        ids, mask, routes, valid = self.backend.tokenize_candidates(candidates)
        output = self.predictor(state_tokens, ids, mask, routes, valid)
        self._last_prediction = output
        return output

    def choose(
        self,
        candidates: Sequence[IntentCandidate],
        output: PredictiveOutput,
    ) -> tuple[IntentCandidate, tuple[CandidatePrediction, ...]]:
        probabilities = (
            output.failure_logits.div(self.calibration_temperature)
            .sigmoid()[0, : len(candidates)]
            .detach()
            .float()
            .cpu()
            .tolist()
        )
        selected = stable_argmin(candidates, probabilities)
        predictions = tuple(
            CandidatePrediction(candidate.candidate_id, float(probability))
            for candidate, probability in zip(candidates, probabilities)
        )
        return candidates[selected], predictions

    def act_chunk(
        self,
        history: PublicHistory,
        selected_intent: IntentCandidate,
        *,
        rng_seed: int,
        requested_steps: int,
    ) -> ActionChunk:
        # Actual H is re-encoded after every real observation; no predicted E
        # and no stale candidate cache enters this call.
        encoded = self.encode(history)
        return self.backend.act_chunk(
            encoded,
            selected_intent,
            seed=rng_seed,
            requested_steps=requested_steps,
        )

    def reset(self) -> None:
        self._last_prediction = None
        reset = getattr(self.backend, "reset", None)
        if callable(reset):
            reset()


def run_closed_loop(
    *,
    policy: PSRPolicy,
    environment: PublicStepEnvironment,
    initial_history: PublicHistory,
    episode_seed: int,
    total_control_steps: int = 300,
    intent_window_steps: int = 50,
    execute_chunk_steps: int = 10,
    public_status: Callable[[PublicObservation], str] | None = None,
) -> ClosedLoopResult:
    """Execute a real-observation closed loop under the frozen v1 timeline."""

    if (
        total_control_steps != 300
        or intent_window_steps != 50
        or execute_chunk_steps != 10
    ):
        raise ValueError("PSR v1 freezes the 300/50/10 execution protocol")
    if initial_history.remaining_control_steps > total_control_steps:
        raise ValueError("initial remaining budget exceeds the original task budget")
    if (
        initial_history.current.control_step + initial_history.remaining_control_steps
        > total_control_steps
    ):
        raise ValueError("history control step and remaining budget exceed 300")
    status_fn = public_status or (lambda _observation: "executed")
    policy.reset()
    history = initial_history
    decisions: list[DecisionRecord] = []
    while history.remaining_control_steps > 0 and not environment.public_terminal():
        decision_start = history.current
        encoded = policy.encode(history)
        candidates = policy.propose(
            encoded,
            episode_seed=episode_seed,
            global_step=history.current.control_step,
        )
        if not candidates:
            raise RuntimeError("PSR candidate generation returned no native option")
        prediction = policy.predict(encoded.state_tokens, candidates)
        selected, reported = policy.choose(candidates, prediction)
        window = min(intent_window_steps, history.remaining_control_steps)
        window_actions: list[tuple[float, ...]] = []
        chunk_rows: list[tuple[tuple[float, ...], ...]] = []
        observations: list[PublicObservation] = []
        used = 0
        while used < window and not environment.public_terminal():
            requested = min(execute_chunk_steps, window - used)
            chunk_seed = int.from_bytes(
                hashlib.sha256(
                    (
                        f"psr-action-seed-v1\n{episode_seed}\n"
                        f"{history.current.control_step}\n{selected.candidate_id}\n"
                    ).encode()
                ).digest()[:8],
                "big",
            )
            action_chunk = policy.act_chunk(
                history,
                selected,
                rng_seed=chunk_seed,
                requested_steps=requested,
            )
            applied: list[tuple[float, ...]] = []
            for action in action_chunk.actions[:requested]:
                observation = environment.step_public(action)
                expected_step = history.current.control_step + 1
                if observation.control_step != expected_step:
                    raise RuntimeError(
                        "environment returned a non-contiguous public step"
                    )
                applied.append(tuple(float(value) for value in action))
                observations.append(observation)
                used += 1
                history = PublicHistory(
                    task=history.task,
                    current=observation,
                    previous=history.previous,
                    executed=history.executed,
                    remaining_control_steps=history.remaining_control_steps - 1,
                )
                if environment.public_terminal() or used == window:
                    break
            if not applied:
                raise RuntimeError("action backend returned an empty executable chunk")
            window_actions.extend(applied)
            chunk_rows.append(tuple(applied))
        event = ExecutedIntent(
            text=selected.text,
            execution_route=selected.execution_route,
            start_step=decision_start.control_step,
            end_step=history.current.control_step,
            actions_ref=_action_reference(window_actions),
            public_status=status_fn(history.current),
        )
        previous = (history.previous + (decision_start,))[-2:]
        history = PublicHistory(
            task=history.task,
            current=history.current,
            previous=previous,
            executed=history.executed + (event,),
            remaining_control_steps=history.remaining_control_steps,
        )
        decisions.append(
            DecisionRecord(
                decision_step=decision_start.control_step,
                remaining_before=history.remaining_control_steps + used,
                candidates=tuple(candidates),
                predictions=reported,
                selected_candidate_id=selected.candidate_id,
                action_chunks=tuple(chunk_rows),
                observations=tuple(observations),
                remaining_after=history.remaining_control_steps,
                terminal_after=environment.public_terminal(),
            )
        )
    return ClosedLoopResult(
        final_history=history,
        decisions=tuple(decisions),
        total_steps=initial_history.remaining_control_steps
        - history.remaining_control_steps,
        terminal=environment.public_terminal(),
    )
