from __future__ import annotations

import dataclasses

import pytest

from grounded_interaction.contracts import (
    ExecutionStatus,
    GroundedIntervention,
    GroundingReference,
    PolicyContext,
    Primitive,
    PublicActionEvent,
    PublicFrame,
)
from grounded_interaction.execution import ExecutorRequest
from grounded_interaction.selection import (
    ExpectedSuccessSelector,
    SelectionStatus,
    ValuePrediction,
)
from grounded_interaction.serialization import GroundedTextSerializer


def _digest(character: str) -> str:
    return character * 64


def _context() -> PolicyContext:
    return PolicyContext(
        prompt="Put the butter in the basket.",
        frames=(
            PublicFrame(
                frame_id="wrist-000",
                camera="wrist",
                frame_index=0,
                image_sha256=_digest("a"),
                width=256,
                height=256,
            ),
        ),
    )


def _candidate(candidate_id: str, primitive: Primitive) -> GroundedIntervention:
    if primitive is Primitive.STOP:
        return GroundedIntervention(candidate_id, primitive, None, (), None)
    return GroundedIntervention(
        candidate_id=candidate_id,
        primitive=primitive,
        referent="middle drawer below the countertop",
        parameters=(("direction", "pull outward"),),
        grounding=GroundingReference(
            camera="wrist",
            frame_id="wrist-000",
            frame_index=0,
            image_sha256=_digest("a"),
            box_xyxy=(0.25, 0.25, 0.75, 0.75),
            point_xy=(0.5, 0.5),
        ),
    )


def _prediction(
    candidate: GroundedIntervention,
    probability: float,
    *,
    feasible: bool = True,
) -> ValuePrediction:
    return ValuePrediction(
        candidate_id=candidate.candidate_id,
        candidate_fingerprint=candidate.fingerprint(),
        success_probability=probability,
        feasible=feasible,
    )


def test_multiple_equal_best_actions_use_stable_order_instead_of_abstaining() -> None:
    first = _candidate("open-middle", Primitive.OPEN)
    second = _candidate("direct", Primitive.DIRECT)
    stop = _candidate("stop", Primitive.STOP)
    decision = ExpectedSuccessSelector().select(
        (first, second, stop),
        (
            _prediction(first, 0.8),
            _prediction(second, 0.8),
            _prediction(stop, 0.1),
        ),
    )
    assert decision.status is SelectionStatus.SELECTED
    assert decision.candidate == first


def test_selector_abstains_only_when_no_candidate_is_feasible() -> None:
    candidate = _candidate("open-middle", Primitive.OPEN)
    decision = ExpectedSuccessSelector().select(
        (candidate,),
        (_prediction(candidate, 0.99, feasible=False),),
    )
    assert decision.status is SelectionStatus.ABSTAIN
    assert decision.candidate is None


def test_selector_abstains_when_the_proposer_returns_nothing() -> None:
    decision = ExpectedSuccessSelector().select((), ())
    assert decision.status is SelectionStatus.ABSTAIN
    assert decision.candidate is None


def test_selector_rejects_identity_drift_even_for_an_infeasible_prediction() -> None:
    candidate = _candidate("open-middle", Primitive.OPEN)
    corrupt = dataclasses.replace(
        _prediction(candidate, 0.2, feasible=False),
        candidate_fingerprint=_digest("f"),
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ExpectedSuccessSelector().select((candidate,), (corrupt,))


def test_direct_serialization_carries_prompt_but_coordinates_are_audit_only() -> None:
    context = _context()
    candidate = _candidate("direct", Primitive.DIRECT)
    serialized = GroundedTextSerializer().serialize(candidate, context)
    assert context.prompt in serialized.subtask_text
    assert (
        serialized.spatial_audit_payload[
            "exact_spatial_binding_sent_as_native_vla_input"
        ]
        is False
    )
    request = ExecutorRequest.from_serialized(serialized)
    request.validate_context(context)
    policy_payload = request.stage2_policy_payload(context)
    assert "spatial_audit_payload" not in policy_payload
    assert request.candidate_id == candidate.candidate_id
    assert request.candidate_fingerprint == candidate.fingerprint()


def test_stop_never_becomes_an_executor_request() -> None:
    serialized = GroundedTextSerializer().serialize(
        _candidate("stop", Primitive.STOP),
        _context(),
    )
    with pytest.raises(ValueError, match="STOP"):
        ExecutorRequest.from_serialized(serialized)


def test_full_synthetic_smoke_preserves_identity_and_reobserves() -> None:
    pytest.importorskip("torch")
    from grounded_interaction.smoke import build_smoke_report

    report = build_smoke_report()
    assert report["software_verification_only"] is True
    assert report["empirical_evidence"] is False
    assert report["replay_path"]["sequence_verified"] is True
    assert report["replay_path"]["identity_chain_verified"] is True
    assert report["tensor_path"]["probabilities_finite"] is True
    assert report["tensor_path"]["executed_only_loss_finite"] is True


def test_open_direct_stop_calls_executor_twice_and_appends_actual_frames() -> None:
    from grounded_interaction.execution import (
        ReplayExecutor,
        ReplayOutcome,
    )
    from grounded_interaction.loop import ClosedLoop, PublicReplayObserver
    from grounded_interaction.smoke import (
        _frame,
        _ReplayProposer,
        _ReplayScorer,
    )

    initial = PolicyContext(
        prompt="Put the butter in the basket.",
        frames=(_frame(0, "a"),),
        proprioception=(0.0, 0.0, 0.0),
    )
    proposer = _ReplayProposer()
    open_candidate = proposer.propose(initial)[1]
    after_open_fixture = PolicyContext(
        prompt=initial.prompt,
        frames=initial.frames + (_frame(1, "b"),),
        public_history=(
            PublicActionEvent(
                step_index=0,
                primitive=Primitive.OPEN,
                subtask_text="Open the middle drawer below the countertop.",
                execution_status=ExecutionStatus.COMPLETED,
            ),
        ),
        proprioception=initial.proprioception,
    )
    direct_candidate = proposer.propose(after_open_fixture)[0]

    class RecordingObserver(PublicReplayObserver):
        def __init__(self, initial_context: PolicyContext) -> None:
            super().__init__(initial_context)
            self.reobserved: list[PolicyContext] = []

        def reobserve(self, previous, post_frames, public_event):
            context = super().reobserve(previous, post_frames, public_event)
            self.reobserved.append(context)
            return context

    class CountingReplayExecutor(ReplayExecutor):
        def __init__(self) -> None:
            super().__init__(
                {
                    open_candidate.fingerprint(): ReplayOutcome(
                        status=ExecutionStatus.COMPLETED,
                        post_frames=(_frame(1, "b"),),
                    ),
                    direct_candidate.fingerprint(): ReplayOutcome(
                        status=ExecutionStatus.COMPLETED,
                        post_frames=(_frame(2, "c"),),
                    ),
                }
            )
            self.calls = 0

        def execute(self, request, context):
            self.calls += 1
            return super().execute(request, context)

    observer = RecordingObserver(initial)
    executor = CountingReplayExecutor()
    trace = ClosedLoop(
        observer=observer,
        proposer=proposer,
        scorer=_ReplayScorer(),
        executor=executor,
    ).run(max_steps=3)

    assert [step.decision.candidate.candidate_id for step in trace.steps] == [
        "open-middle",
        "direct-butter",
        "stop",
    ]
    assert executor.calls == 2
    assert trace.steps[-1].request is None
    assert trace.steps[-1].receipt is None
    assert observer.reobserved[0].frames == initial.frames + (_frame(1, "b"),)
    assert observer.reobserved[1].frames == (
        initial.frames + (_frame(1, "b"), _frame(2, "c"))
    )
    assert all(
        set(context.public_history[-1].to_dict())
        == {"step_index", "primitive", "subtask_text", "execution_status"}
        for context in observer.reobserved
    )
    for step in trace.steps[:2]:
        assert step.request is not None
        assert step.receipt is not None
        assert step.request.candidate_id.encode(
            "utf-8"
        ) == step.receipt.candidate_id.encode("utf-8")
        assert step.request.candidate_fingerprint.encode(
            "utf-8"
        ) == step.receipt.candidate_fingerprint.encode("utf-8")
        assert step.request.request_digest.encode(
            "utf-8"
        ) == step.receipt.request_digest.encode("utf-8")
