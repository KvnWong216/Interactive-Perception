"""Minimal closed loop for grounded, outcome-conditioned interaction.

The runtime order is explicit and auditable:

``observe -> propose -> score -> select -> serialize -> execute
          -> actual post frames -> public history -> reobserve``

The module contains protocols rather than model-specific adapters.  Its replay
utilities validate wiring and branch handling; they are not empirical results.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from enum import Enum
from typing import Protocol, runtime_checkable

from .contracts import (
    GroundedIntervention,
    PolicyContext,
    Primitive,
    PublicActionEvent,
    PublicFrame,
    canonical_sha256,
)
from .execution import ExecutorReceipt, ExecutorRequest, FrozenVLAExecutor
from .selection import (
    ExpectedSuccessSelector,
    SelectionDecision,
    SelectionStatus,
    ValuePrediction,
)
from .serialization import GroundedTextSerializer, SerializedSubtask


@runtime_checkable
class ObservationProvider(Protocol):
    """Provide the initial public observation and rebuild it after execution."""

    def observe(self) -> PolicyContext: ...

    def reobserve(
        self,
        previous: PolicyContext,
        post_frames: tuple[PublicFrame, ...],
        public_event: PublicActionEvent,
    ) -> PolicyContext: ...


@runtime_checkable
class CandidateProposer(Protocol):
    """Ground complete intervention instances before value prediction."""

    def propose(self, context: PolicyContext) -> Sequence[GroundedIntervention]: ...


@runtime_checkable
class CandidateScorer(Protocol):
    """Adapt a learned model to the stable runtime prediction contract."""

    def score(
        self,
        context: PolicyContext,
        candidates: Sequence[GroundedIntervention],
    ) -> Sequence[ValuePrediction]: ...


class LoopTermination(str, Enum):
    STOP = "STOP"
    ABSTAIN = "ABSTAIN"
    MAX_STEPS = "MAX_STEPS"


@dataclasses.dataclass(frozen=True)
class LoopStep:
    """One immutable decision/execution trace item."""

    step_index: int
    context_fingerprint: str
    candidate_ids: tuple[str, ...]
    candidate_fingerprints: tuple[str, ...]
    predictions: tuple[ValuePrediction, ...]
    decision: SelectionDecision
    serialized_subtask: SerializedSubtask | None
    request: ExecutorRequest | None
    receipt: ExecutorReceipt | None
    next_context_fingerprint: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "context_fingerprint": self.context_fingerprint,
            "candidate_ids": list(self.candidate_ids),
            "candidate_fingerprints": list(self.candidate_fingerprints),
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "decision": self.decision.to_dict(),
            "serialized_subtask": (
                self.serialized_subtask.to_dict()
                if self.serialized_subtask is not None
                else None
            ),
            "request": self.request.to_dict() if self.request is not None else None,
            "receipt": self.receipt.to_dict() if self.receipt is not None else None,
            "next_context_fingerprint": self.next_context_fingerprint,
        }


@dataclasses.dataclass(frozen=True)
class LoopTrace:
    """Complete closed-loop wiring trace."""

    initial_context_fingerprint: str
    final_context_fingerprint: str
    termination: LoopTermination
    steps: tuple[LoopStep, ...]

    def to_dict(self) -> dict[str, object]:
        payload = {
            "initial_context_fingerprint": self.initial_context_fingerprint,
            "final_context_fingerprint": self.final_context_fingerprint,
            "termination": self.termination.value,
            "steps": [step.to_dict() for step in self.steps],
        }
        return {**payload, "trace_digest": canonical_sha256(payload)}


class ClosedLoop:
    """Coordinate Stage 1, the frozen Stage-2 executor, and reobservation."""

    def __init__(
        self,
        *,
        observer: ObservationProvider,
        proposer: CandidateProposer,
        scorer: CandidateScorer,
        executor: FrozenVLAExecutor,
        selector: ExpectedSuccessSelector | None = None,
        serializer: GroundedTextSerializer | None = None,
    ) -> None:
        self.observer = observer
        self.proposer = proposer
        self.scorer = scorer
        self.executor = executor
        self.selector = selector or ExpectedSuccessSelector()
        self.serializer = serializer or GroundedTextSerializer()

    @staticmethod
    def _validate_reobservation(
        previous: PolicyContext,
        actual_post_frames: tuple[PublicFrame, ...],
        public_event: PublicActionEvent,
        observed: PolicyContext,
    ) -> None:
        if not isinstance(observed, PolicyContext):
            raise TypeError("reobserve must return a PolicyContext")
        if observed.prompt != previous.prompt:
            raise ValueError("reobservation must preserve the user prompt")
        latest_by_camera = {
            camera: previous.latest_frame_index(camera)
            for camera in {frame.camera for frame in previous.frames}
        }
        for frame in actual_post_frames:
            previous_index = latest_by_camera.get(frame.camera, -1)
            if frame.frame_index <= previous_index:
                raise ValueError(
                    "post frames must advance monotonically within each camera"
                )
            latest_by_camera[frame.camera] = frame.frame_index
        expected_frames = previous.frames + actual_post_frames
        if observed.frames != expected_frames:
            raise ValueError(
                "reobservation must preserve prior frames and append exact "
                "executor post frames"
            )
        if len(observed.public_history) != len(previous.public_history) + 1:
            raise ValueError("reobservation must append exactly one history event")
        if observed.public_history[:-1] != previous.public_history:
            raise ValueError("reobservation must preserve prior public history")
        if observed.public_history[-1] != public_event:
            raise ValueError("reobservation history does not match execution event")

    def run(self, *, max_steps: int) -> LoopTrace:
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer")
        context = self.observer.observe()
        if not isinstance(context, PolicyContext):
            raise TypeError("observe must return a PolicyContext")
        initial_fingerprint = context.fingerprint()
        steps: list[LoopStep] = []

        for step_index in range(max_steps):
            candidates = tuple(self.proposer.propose(context))
            for candidate in candidates:
                if not isinstance(candidate, GroundedIntervention):
                    raise TypeError("proposer must return GroundedIntervention values")
                candidate.validate_against(context)
            predictions = tuple(self.scorer.score(context, candidates))
            decision = self.selector.select(candidates, predictions)

            if decision.status is SelectionStatus.ABSTAIN:
                steps.append(
                    LoopStep(
                        step_index=step_index,
                        context_fingerprint=context.fingerprint(),
                        candidate_ids=tuple(
                            candidate.candidate_id for candidate in candidates
                        ),
                        candidate_fingerprints=tuple(
                            candidate.fingerprint() for candidate in candidates
                        ),
                        predictions=predictions,
                        decision=decision,
                        serialized_subtask=None,
                        request=None,
                        receipt=None,
                        next_context_fingerprint=None,
                    )
                )
                return LoopTrace(
                    initial_context_fingerprint=initial_fingerprint,
                    final_context_fingerprint=context.fingerprint(),
                    termination=LoopTermination.ABSTAIN,
                    steps=tuple(steps),
                )

            candidate = decision.candidate
            prediction = decision.prediction
            if candidate is None or prediction is None:  # defensive type narrowing
                raise RuntimeError("selected decision lost its candidate identity")
            serialized = self.serializer.serialize(candidate, context)
            if (
                serialized.candidate_id.encode("utf-8")
                != candidate.candidate_id.encode("utf-8")
                or serialized.candidate_fingerprint.encode("utf-8")
                != candidate.fingerprint().encode("utf-8")
                or serialized.context_fingerprint.encode("utf-8")
                != context.fingerprint().encode("utf-8")
            ):
                raise ValueError(
                    "serializer changed candidate or public-context identity"
                )

            if candidate.primitive is Primitive.STOP:
                steps.append(
                    LoopStep(
                        step_index=step_index,
                        context_fingerprint=context.fingerprint(),
                        candidate_ids=tuple(item.candidate_id for item in candidates),
                        candidate_fingerprints=tuple(
                            item.fingerprint() for item in candidates
                        ),
                        predictions=predictions,
                        decision=decision,
                        serialized_subtask=serialized,
                        request=None,
                        receipt=None,
                        next_context_fingerprint=None,
                    )
                )
                return LoopTrace(
                    initial_context_fingerprint=initial_fingerprint,
                    final_context_fingerprint=context.fingerprint(),
                    termination=LoopTermination.STOP,
                    steps=tuple(steps),
                )

            request = ExecutorRequest.from_serialized(serialized)
            receipt = self.executor.execute(request, context)
            receipt.validate_request(request)
            if receipt.executor_id.encode("utf-8") != str(
                self.executor.executor_id
            ).encode("utf-8"):
                raise ValueError("receipt executor_id does not match called executor")
            actual_post_frames = receipt.post_frames
            public_event = PublicActionEvent(
                step_index=step_index,
                primitive=candidate.primitive,
                subtask_text=serialized.subtask_text,
                execution_status=receipt.status,
            )
            next_context = self.observer.reobserve(
                context,
                actual_post_frames,
                public_event,
            )
            self._validate_reobservation(
                context,
                actual_post_frames,
                public_event,
                next_context,
            )
            steps.append(
                LoopStep(
                    step_index=step_index,
                    context_fingerprint=context.fingerprint(),
                    candidate_ids=tuple(item.candidate_id for item in candidates),
                    candidate_fingerprints=tuple(
                        item.fingerprint() for item in candidates
                    ),
                    predictions=predictions,
                    decision=decision,
                    serialized_subtask=serialized,
                    request=request,
                    receipt=receipt,
                    next_context_fingerprint=next_context.fingerprint(),
                )
            )
            context = next_context

        return LoopTrace(
            initial_context_fingerprint=initial_fingerprint,
            final_context_fingerprint=context.fingerprint(),
            termination=LoopTermination.MAX_STEPS,
            steps=tuple(steps),
        )


class PublicReplayObserver:
    """Build subsequent policy contexts from the executor's actual frames."""

    def __init__(self, initial_context: PolicyContext) -> None:
        if not isinstance(initial_context, PolicyContext):
            raise TypeError("initial_context must be a PolicyContext")
        self.initial_context = initial_context

    def observe(self) -> PolicyContext:
        return self.initial_context

    def reobserve(
        self,
        previous: PolicyContext,
        post_frames: tuple[PublicFrame, ...],
        public_event: PublicActionEvent,
    ) -> PolicyContext:
        return PolicyContext(
            prompt=previous.prompt,
            frames=previous.frames + tuple(post_frames),
            public_history=previous.public_history + (public_event,),
            proprioception=previous.proprioception,
        )
