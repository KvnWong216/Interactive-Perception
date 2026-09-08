from __future__ import annotations

import pytest

from grounded_interaction.continuation import (
    BudgetedExecution,
    FixedContinuationIdentity,
    FixedHorizonContinuation,
    FixedHorizonStatus,
)
from grounded_interaction.contracts import (
    ExecutionStatus,
    GroundedIntervention,
    GroundingReference,
    PolicyContext,
    Primitive,
    PublicActionEvent,
    PublicFrame,
    canonical_sha256,
)
from grounded_interaction.execution import ExecutorReceipt, ExecutorRequest
from grounded_interaction.serialization import GroundedTextSerializer


def _digest(character: str) -> str:
    return character * 64


def _frame(frame_id: str, camera: str, index: int, character: str) -> PublicFrame:
    return PublicFrame(
        frame_id=frame_id,
        camera=camera,
        frame_index=index,
        image_sha256=_digest(character),
        width=256,
        height=256,
    )


def _context() -> PolicyContext:
    return PolicyContext(
        prompt="Put the butter in the basket.",
        frames=(
            _frame("agent-0", "agentview", 0, "a"),
            _frame("wrist-0", "wrist", 0, "b"),
        ),
        proprioception=(0.0,) * 8,
    )


def _candidate(
    candidate_id: str,
    primitive: Primitive,
    context: PolicyContext,
    *,
    x: float = 0.5,
) -> GroundedIntervention:
    frame = max(
        (item for item in context.frames if item.camera == "agentview"),
        key=lambda item: item.frame_index,
    )
    return GroundedIntervention(
        candidate_id=candidate_id,
        primitive=primitive,
        referent="middle drawer" if primitive is Primitive.OPEN else "butter",
        parameters=(),
        grounding=GroundingReference(
            camera=frame.camera,
            frame_id=frame.frame_id,
            frame_index=frame.frame_index,
            image_sha256=frame.image_sha256,
            box_xyxy=(x - 0.1, 0.3, x + 0.1, 0.7),
            point_xy=(x, 0.5),
        ),
    )


def _identity() -> FixedContinuationIdentity:
    return FixedContinuationIdentity(
        proposer_id="qwen-continuation-revision-x",
        proposer_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        proposer_revision="revision-x",
        proposer_prompt_sha256=_digest("c"),
        proposer_seed=17,
        executor_id="molmoact2-libero-pinned",
        serializer_id="grounded-precise-text-v1",
    )


class FakeDirectProposer:
    proposer_id = "qwen-continuation-revision-x"

    def __init__(self, *, count: int = 2, primitive: Primitive = Primitive.DIRECT):
        self.count = count
        self.primitive = primitive
        self.calls: list[tuple[PolicyContext, int, int]] = []

    def propose_direct(
        self,
        context: PolicyContext,
        *,
        max_candidates: int,
        deterministic_seed: int,
    ) -> tuple[GroundedIntervention, ...]:
        self.calls.append((context, max_candidates, deterministic_seed))
        return tuple(
            _candidate(
                f"continuation-{index}",
                self.primitive,
                context,
                x=0.2 + index * 0.15,
            )
            for index in range(self.count)
        )


class FakeBudgetedExecutor:
    executor_id = "molmoact2-libero-pinned"
    serializer_id = "grounded-precise-text-v1"
    replan_interval = 10

    def __init__(self) -> None:
        self.serializer = GroundedTextSerializer(serializer_id=self.serializer_id)
        self.calls: list[tuple[str, int, int, str]] = []

    def execute_candidate(
        self,
        candidate: GroundedIntervention,
        context: PolicyContext,
        *,
        control_step_budget: int,
        model_seed: int,
    ) -> BudgetedExecution:
        self.calls.append(
            (
                candidate.candidate_id,
                control_step_budget,
                model_seed,
                context.fingerprint(),
            )
        )
        serialized = self.serializer.serialize(candidate, context)
        request = ExecutorRequest.from_serialized(serialized)
        next_index = max(item.frame_index for item in context.frames) + 1
        suffix = len(self.calls)
        post_frames = (
            _frame(f"agent-{suffix}", "agentview", next_index, str(suffix)),
            _frame(f"wrist-{suffix}", "wrist", next_index, str(suffix + 3)),
        )
        receipt = ExecutorReceipt(
            receipt_id=f"receipt-{suffix}",
            executor_id=self.executor_id,
            candidate_id=candidate.candidate_id,
            candidate_fingerprint=candidate.fingerprint(),
            request_digest=request.request_digest,
            status=ExecutionStatus.COMPLETED,
            post_frames=post_frames,
        )
        event = PublicActionEvent(
            step_index=len(context.public_history),
            primitive=candidate.primitive,
            subtask_text=request.subtask_text,
            execution_status=receipt.status,
        )
        next_context = PolicyContext(
            prompt=context.prompt,
            frames=context.frames + post_frames,
            public_history=context.public_history + (event,),
            proprioception=context.proprioception,
        )
        trace = {
            "candidate": candidate.fingerprint(),
            "budget": control_step_budget,
            "model_seed": model_seed,
            "context": context.fingerprint(),
            "next_context": next_context.fingerprint(),
        }
        return BudgetedExecution(
            request=request,
            receipt=receipt,
            previous_context=context,
            next_context=next_context,
            requested_control_steps=control_step_budget,
            control_steps_used=control_step_budget,
            model_seed=model_seed,
            public_trace_sha256=canonical_sha256(trace),
        )


def test_direct_uses_full_budget_and_never_calls_continuation_proposer() -> None:
    context = _context()
    proposer = FakeDirectProposer()
    executor = FakeBudgetedExecutor()
    controller = FixedHorizonContinuation(
        identity=_identity(), proposer=proposer, executor=executor
    )
    selected = _candidate("direct-initial", Primitive.DIRECT, context)
    result = controller.run(
        context=context, selected_candidate=selected, model_seed=101
    )

    assert result.status is FixedHorizonStatus.DIRECT_EXECUTED
    assert result.control_steps_used == 300
    assert [item[1] for item in executor.calls] == [300]
    assert proposer.calls == []
    assert result.continuation_candidate_ids == ()


def test_open_reobserves_then_uses_frozen_top1_direct_without_rescoring() -> None:
    context = _context()
    proposer = FakeDirectProposer(count=2)
    executor = FakeBudgetedExecutor()
    identity = _identity()
    controller = FixedHorizonContinuation(
        identity=identity, proposer=proposer, executor=executor
    )
    selected = _candidate("open-initial", Primitive.OPEN, context)
    result = controller.run(
        context=context, selected_candidate=selected, model_seed=202
    )

    assert result.status is FixedHorizonStatus.OPEN_THEN_DIRECT_EXECUTED
    assert [item[1] for item in executor.calls] == [100, 200]
    assert result.control_steps_used == 300
    assert len(proposer.calls) == 1
    proposed_context, max_candidates, deterministic_seed = proposer.calls[0]
    assert proposed_context == result.stages[0].next_context
    assert len(proposed_context.public_history) == 1
    assert max_candidates == 3
    assert deterministic_seed == identity.proposer_seed
    assert executor.calls[1][0] == "continuation-0"
    assert result.continuation_candidate_ids == (
        "continuation-0",
        "continuation-1",
    )


def test_open_with_no_direct_candidate_stops_after_public_reobservation() -> None:
    context = _context()
    proposer = FakeDirectProposer(count=0)
    executor = FakeBudgetedExecutor()
    result = FixedHorizonContinuation(
        identity=_identity(), proposer=proposer, executor=executor
    ).run(
        context=context,
        selected_candidate=_candidate("open-initial", Primitive.OPEN, context),
        model_seed=303,
    )
    assert result.status is FixedHorizonStatus.OPEN_WITH_NO_DIRECT_CANDIDATE
    assert result.control_steps_used == 100
    assert len(executor.calls) == 1
    assert result.final_context == result.stages[0].next_context


def test_continuation_rejects_non_direct_or_more_than_three_candidates() -> None:
    context = _context()
    selected = _candidate("open-initial", Primitive.OPEN, context)
    executor = FakeBudgetedExecutor()
    wrong_primitive = FixedHorizonContinuation(
        identity=_identity(),
        proposer=FakeDirectProposer(count=1, primitive=Primitive.OPEN),
        executor=executor,
    )
    with pytest.raises(ValueError, match="DIRECT only"):
        wrong_primitive.run(context=context, selected_candidate=selected, model_seed=1)

    too_many = FixedHorizonContinuation(
        identity=_identity(),
        proposer=FakeDirectProposer(count=4),
        executor=FakeBudgetedExecutor(),
    )
    with pytest.raises(ValueError, match="exceeded max_direct_candidates"):
        too_many.run(context=context, selected_candidate=selected, model_seed=1)


def test_runtime_and_outcome_contract_must_match_frozen_identity() -> None:
    identity = _identity()
    contract = identity.outcome_contract()
    identity.validate_outcome_contract(contract)
    assert contract.horizon == 300
    assert contract.continuation_policy_id == identity.policy_id

    class WrongExecutor(FakeBudgetedExecutor):
        replan_interval = 5

    with pytest.raises(ValueError, match="runtime identity"):
        FixedHorizonContinuation(
            identity=identity,
            proposer=FakeDirectProposer(),
            executor=WrongExecutor(),
        )
