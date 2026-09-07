from __future__ import annotations

import dataclasses

import pytest

from grounded_interaction.contracts import (
    ExecutionStatus,
    GroundedIntervention,
    GroundingReference,
    OutcomeContract,
    PolicyContext,
    Primitive,
    PublicActionEvent,
    PublicFrame,
    assert_public_policy_value,
)
from grounded_interaction.data import (
    ObservedBranch,
    ResetControlledBranchDataset,
    validate_group_splits,
)


def _digest(character: str) -> str:
    return character * 64


def _frame(
    frame_id: str,
    index: int,
    digest: str,
    *,
    camera: str = "wrist",
) -> PublicFrame:
    return PublicFrame(
        frame_id=frame_id,
        camera=camera,
        frame_index=index,
        image_sha256=digest,
        width=256,
        height=256,
    )


def _context(*, history: tuple[PublicActionEvent, ...] = ()) -> PolicyContext:
    return PolicyContext(
        prompt="Put the butter in the basket.",
        frames=(
            _frame("wrist-000", 0, _digest("a")),
            _frame("wrist-001", 1, _digest("b")),
        ),
        public_history=history,
        proprioception=(0.0, 1.0),
    )


def _grounding() -> GroundingReference:
    return GroundingReference(
        camera="wrist",
        frame_id="wrist-001",
        frame_index=1,
        image_sha256=_digest("b"),
        box_xyxy=(0.1, 0.2, 0.7, 0.8),
        point_xy=(0.4, 0.5),
    )


def _candidate(candidate_id: str, primitive: Primitive) -> GroundedIntervention:
    return GroundedIntervention(
        candidate_id=candidate_id,
        primitive=primitive,
        referent="middle drawer below the countertop",
        parameters=(("direction", "pull outward"),),
        grounding=_grounding(),
    )


def _contract() -> OutcomeContract:
    return OutcomeContract(
        outcome_name="synthetic_task_success_within_horizon",
        continuation_policy_id="synthetic-continuation-v1",
        horizon=2,
        executor_id="replay-executor-v1",
        serializer_id="referential-text-v1",
        failure_handling="count_completed_executor_failures_as_negative",
    )


def test_candidate_is_content_addressed_and_parameter_order_is_canonical() -> None:
    first = GroundedIntervention(
        candidate_id="open-middle",
        primitive=Primitive.OPEN,
        referent="middle drawer",
        parameters=(("direction", "outward"), ("extent", "fully open")),
        grounding=_grounding(),
    )
    second = dataclasses.replace(
        first,
        parameters=(("extent", "fully open"), ("direction", "outward")),
    )
    assert first.fingerprint() == second.fingerprint()
    first.validate_against(_context())


def test_grounding_must_reference_the_latest_exact_public_frame() -> None:
    stale = dataclasses.replace(
        _grounding(),
        frame_id="wrist-000",
        frame_index=0,
        image_sha256=_digest("a"),
    )
    with pytest.raises(ValueError, match="latest"):
        dataclasses.replace(
            _candidate("open-middle", Primitive.OPEN), grounding=stale
        ).validate_against(_context())


def test_stop_is_the_only_candidate_without_a_physical_grounding() -> None:
    stop = GroundedIntervention(
        candidate_id="stop",
        primitive=Primitive.STOP,
        referent=None,
        parameters=(),
        grounding=None,
    )
    assert stop.policy_payload()["grounding"] is None
    with pytest.raises(ValueError, match="requires grounding"):
        GroundedIntervention(
            candidate_id="open",
            primitive=Primitive.OPEN,
            referent="drawer",
            parameters=(),
            grounding=None,
        )


def test_policy_context_accepts_only_typed_public_action_history() -> None:
    event = PublicActionEvent(
        step_index=0,
        primitive=Primitive.OPEN,
        subtask_text="Open the middle drawer.",
        execution_status=ExecutionStatus.COMPLETED,
    )
    assert _context(history=(event,)).public_history == (event,)
    with pytest.raises(TypeError, match="PublicActionEvent"):
        _context(history=({"primitive": "OPEN"},))  # type: ignore[arg-type]


def test_recursive_firewall_rejects_nested_privileged_fields() -> None:
    with pytest.raises(ValueError, match="privileged policy field"):
        assert_public_policy_value({"observation": {"semantic_id": 7}})


def test_observed_branch_exposes_only_pre_action_public_input() -> None:
    branch = ObservedBranch(
        branch_id="branch-open-0",
        initial_state_group="state-001",
        decision_group_id="decision-001",
        split="train",
        reset_state_sha256=_digest("c"),
        repeat_index=0,
        context=_context(),
        executed_intervention=_candidate("open-middle", Primitive.OPEN),
        post_action_frames=(_frame("wrist-002", 2, _digest("d")),),
        outcome_contract=_contract(),
        observed_outcome=True,
        execution_status=ExecutionStatus.COMPLETED,
        execution_receipt_id="receipt-open-0",
        diagnostics={"steps": 12},
        private_evaluator_metadata={"target_instance_id": 42},
    )
    model_input = branch.model_input()
    assert set(model_input) == {"context", "candidate"}
    assert "observed_outcome" not in str(model_input)
    assert "reset_state_sha256" not in str(model_input)
    assert "target_instance_id" not in str(model_input)
    assert branch.supervision()["executed_candidate_id"] == "open-middle"


def test_reset_controlled_dataset_requires_real_receipts_for_every_branch() -> None:
    context = _context()
    post = _frame("wrist-002", 2, _digest("d"))
    branches = tuple(
        ObservedBranch(
            branch_id=f"branch-{candidate_id}-0",
            initial_state_group="state-001",
            decision_group_id="decision-001",
            split="train",
            reset_state_sha256=_digest("c"),
            repeat_index=0,
            context=context,
            executed_intervention=_candidate(candidate_id, primitive),
            post_action_frames=(
                dataclasses.replace(post, frame_id=f"{candidate_id}-post"),
            ),
            outcome_contract=_contract(),
            observed_outcome=(candidate_id == "open-middle"),
            execution_status=ExecutionStatus.COMPLETED,
            execution_receipt_id=f"receipt-{candidate_id}-0",
        )
        for candidate_id, primitive in (
            ("open-middle", Primitive.OPEN),
            ("direct-middle", Primitive.DIRECT),
        )
    )
    dataset = ResetControlledBranchDataset(
        dataset_id="synthetic-branch-matrix-v1",
        repetitions_per_candidate=1,
        branches=branches,
    )
    assert dataset.summary()["observed_branches"] == 2
    assert dataset.summary()["candidate_executions"] == 2

    leaked = dataclasses.replace(branches[1], split="test")
    with pytest.raises(ValueError, match="crosses data splits"):
        validate_group_splits((branches[0], leaked))
