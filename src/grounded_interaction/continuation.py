"""Fixed finite-horizon execution policy used by Method V1 labels.

This is the concrete high-level time contract, not a learned planner:

* ``DIRECT`` executes the selected grounded candidate for exactly 300 control
  steps.
* ``OPEN`` executes for exactly 100 steps, reobserves using only public RGB and
  proprioception, asks one frozen proposer for at most three ``DIRECT``
  candidates, takes the first candidate in proposer order, and executes it for
  exactly 200 steps.

The learned outcome scorer is intentionally absent from every interface in
this module.  The post-OPEN choice therefore cannot accidentally call the
learned model or read a simulator predicate.  A concrete MolmoAct2/LIBERO
adapter implements :class:`BudgetedGroundedExecutor`; unit tests can supply an
identity-preserving fake through the same boundary.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from enum import Enum
from typing import Protocol, runtime_checkable

from .contracts import (
    GroundedIntervention,
    OutcomeContract,
    PolicyContext,
    Primitive,
    canonical_sha256,
)
from .execution import ExecutorReceipt, ExecutorRequest
from .method_v1_data import make_method_v1_outcome_contract

CONTINUATION_SCHEMA = "method-v1-fixed-continuation-identity-v1"
SELECTION_RULE = "first_valid_direct_in_frozen_proposer_order_v1"
INITIAL_DIRECT_STEPS = 300
OPEN_STEPS = 100
POST_OPEN_DIRECT_STEPS = 200
TOTAL_STEPS = 300
REPLAN_INTERVAL = 10
MAX_DIRECT_CANDIDATES = 3


def _clean_text(value: object, *, name: str) -> str:
    result = " ".join(str(value or "").split())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


@dataclasses.dataclass(frozen=True)
class FixedContinuationIdentity:
    """All settings that can change the post-OPEN data-generating policy."""

    proposer_id: str
    proposer_model_id: str
    proposer_revision: str
    proposer_prompt_sha256: str
    proposer_seed: int
    executor_id: str
    serializer_id: str
    selection_rule: str = SELECTION_RULE
    initial_direct_steps: int = INITIAL_DIRECT_STEPS
    open_steps: int = OPEN_STEPS
    post_open_direct_steps: int = POST_OPEN_DIRECT_STEPS
    total_steps: int = TOTAL_STEPS
    replan_interval: int = REPLAN_INTERVAL
    max_direct_candidates: int = MAX_DIRECT_CANDIDATES
    schema_version: str = CONTINUATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "proposer_id",
            "proposer_model_id",
            "proposer_revision",
            "proposer_prompt_sha256",
            "executor_id",
            "serializer_id",
        ):
            object.__setattr__(self, name, _clean_text(getattr(self, name), name=name))
        if self.schema_version != CONTINUATION_SCHEMA:
            raise ValueError("unsupported continuation identity schema")
        if self.selection_rule != SELECTION_RULE:
            raise ValueError("Method V1 continuation selection rule is frozen")
        expected = {
            "initial_direct_steps": INITIAL_DIRECT_STEPS,
            "open_steps": OPEN_STEPS,
            "post_open_direct_steps": POST_OPEN_DIRECT_STEPS,
            "total_steps": TOTAL_STEPS,
            "replan_interval": REPLAN_INTERVAL,
            "max_direct_candidates": MAX_DIRECT_CANDIDATES,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"Method V1 requires {name}={value}")
        if self.open_steps + self.post_open_direct_steps != self.total_steps:
            raise ValueError("OPEN and continuation budgets must sum to total_steps")
        if (
            not isinstance(self.proposer_seed, int)
            or isinstance(self.proposer_seed, bool)
            or self.proposer_seed < 0
        ):
            raise ValueError("proposer_seed must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    @property
    def policy_id(self) -> str:
        """Content-addressed identity stored in the OutcomeContract."""

        return canonical_sha256(self.to_dict())

    def outcome_contract(self) -> OutcomeContract:
        return make_method_v1_outcome_contract(
            continuation_policy_id=self.policy_id,
            executor_id=self.executor_id,
            serializer_id=self.serializer_id,
        )

    def validate_outcome_contract(self, contract: OutcomeContract) -> None:
        expected = self.outcome_contract()
        if contract.fingerprint() != expected.fingerprint():
            raise ValueError("outcome contract does not match continuation identity")


@runtime_checkable
class FrozenDirectContinuationProposer(Protocol):
    """Frozen public-observation proposer for the single post-OPEN decision."""

    @property
    def proposer_id(self) -> str: ...

    def propose_direct(
        self,
        context: PolicyContext,
        *,
        max_candidates: int,
        deterministic_seed: int,
    ) -> Sequence[GroundedIntervention]: ...


@dataclasses.dataclass(frozen=True)
class BudgetedExecution:
    """Public, identity-bound result of one fixed-budget VLA stage."""

    request: ExecutorRequest
    receipt: ExecutorReceipt
    previous_context: PolicyContext
    next_context: PolicyContext
    requested_control_steps: int
    control_steps_used: int
    model_seed: int
    public_trace_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.request, ExecutorRequest):
            raise TypeError("request must be an ExecutorRequest")
        if not isinstance(self.receipt, ExecutorReceipt):
            raise TypeError("receipt must be an ExecutorReceipt")
        if not isinstance(self.previous_context, PolicyContext) or not isinstance(
            self.next_context, PolicyContext
        ):
            raise TypeError("previous_context and next_context must be PolicyContext")
        for name in ("requested_control_steps", "control_steps_used", "model_seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.requested_control_steps < 1:
            raise ValueError("requested_control_steps must be positive")
        # Fixed duration prevents a private success predicate from deciding
        # when OPEN hands control to the continuation.
        if self.control_steps_used != self.requested_control_steps:
            raise ValueError("Method V1 stages must consume their exact frozen budget")
        trace_digest = str(self.public_trace_sha256)
        if len(trace_digest) != 64 or any(
            character not in "0123456789abcdef" for character in trace_digest
        ):
            raise ValueError("public_trace_sha256 must be a lowercase SHA-256 digest")
        self.request.validate_context(self.previous_context)
        self.receipt.validate_request(self.request)
        self._validate_public_transition()

    def _validate_public_transition(self) -> None:
        previous = self.previous_context
        observed = self.next_context
        if observed.prompt != previous.prompt:
            raise ValueError("execution must preserve the original task prompt")
        if observed.frames != previous.frames + self.receipt.post_frames:
            raise ValueError(
                "execution must append exactly the receipt-backed public frames"
            )
        previous_ids = {frame.frame_id for frame in previous.frames}
        if previous_ids & {frame.frame_id for frame in self.receipt.post_frames}:
            raise ValueError("post-execution public frame IDs must be new")
        latest = {
            camera: previous.latest_frame_index(camera)
            for camera in {frame.camera for frame in previous.frames}
        }
        for frame in self.receipt.post_frames:
            if frame.frame_index <= latest.get(frame.camera, -1):
                raise ValueError("post-execution frame indices must advance")
            latest[frame.camera] = frame.frame_index
        if observed.public_history[:-1] != previous.public_history:
            raise ValueError("execution must preserve prior public action history")
        if len(observed.public_history) != len(previous.public_history) + 1:
            raise ValueError("execution must append exactly one public action event")
        event = observed.public_history[-1]
        if (
            event.step_index != len(previous.public_history)
            or event.primitive is not self.request.primitive
            or event.subtask_text != self.request.subtask_text
            or event.execution_status is not self.receipt.status
        ):
            raise ValueError("public action event does not match request and receipt")

    @property
    def candidate_id(self) -> str:
        return self.request.candidate_id

    @property
    def candidate_fingerprint(self) -> str:
        return self.request.candidate_fingerprint

    def to_dict(self) -> dict[str, object]:
        return {
            "request": self.request.to_dict(),
            "receipt": self.receipt.to_dict(),
            "previous_context_fingerprint": self.previous_context.fingerprint(),
            "next_context_fingerprint": self.next_context.fingerprint(),
            "requested_control_steps": self.requested_control_steps,
            "control_steps_used": self.control_steps_used,
            "model_seed": self.model_seed,
            "public_trace_sha256": self.public_trace_sha256,
        }


@runtime_checkable
class BudgetedGroundedExecutor(Protocol):
    """Adapter for a real VLA/environment pair with a fixed step budget."""

    @property
    def executor_id(self) -> str: ...

    @property
    def serializer_id(self) -> str: ...

    @property
    def replan_interval(self) -> int: ...

    def execute_candidate(
        self,
        candidate: GroundedIntervention,
        context: PolicyContext,
        *,
        control_step_budget: int,
        model_seed: int,
    ) -> BudgetedExecution: ...


class FixedHorizonStatus(str, Enum):
    DIRECT_EXECUTED = "DIRECT_EXECUTED"
    OPEN_THEN_DIRECT_EXECUTED = "OPEN_THEN_DIRECT_EXECUTED"
    OPEN_WITH_NO_DIRECT_CANDIDATE = "OPEN_WITH_NO_DIRECT_CANDIDATE"


@dataclasses.dataclass(frozen=True)
class FixedHorizonExecution:
    """Complete public trace consumed by the final-task evaluator."""

    continuation_policy_id: str
    model_seed: int
    selected_candidate_id: str
    selected_candidate_fingerprint: str
    status: FixedHorizonStatus
    stages: tuple[BudgetedExecution, ...]
    continuation_candidate_ids: tuple[str, ...]
    continuation_candidate_fingerprints: tuple[str, ...]
    final_context: PolicyContext

    def __post_init__(self) -> None:
        for name in (
            "continuation_policy_id",
            "selected_candidate_id",
            "selected_candidate_fingerprint",
        ):
            object.__setattr__(self, name, _clean_text(getattr(self, name), name=name))
        if not isinstance(self.status, FixedHorizonStatus):
            raise TypeError("status must be a FixedHorizonStatus")
        stages = tuple(self.stages)
        if not stages or len(stages) > 2:
            raise ValueError("fixed-horizon execution must contain one or two stages")
        if stages[0].candidate_id != self.selected_candidate_id:
            raise ValueError("first stage changed the selected candidate ID")
        if stages[0].candidate_fingerprint != self.selected_candidate_fingerprint:
            raise ValueError("first stage changed the selected candidate fingerprint")
        if self.final_context != stages[-1].next_context:
            raise ValueError("final_context must be the last real reobservation")
        candidate_ids = tuple(self.continuation_candidate_ids)
        fingerprints = tuple(self.continuation_candidate_fingerprints)
        if len(candidate_ids) != len(fingerprints):
            raise ValueError("continuation candidate identities must align")
        if len(candidate_ids) > MAX_DIRECT_CANDIDATES:
            raise ValueError("continuation candidate count exceeds frozen maximum")
        if self.status is FixedHorizonStatus.DIRECT_EXECUTED:
            if len(stages) != 1 or stages[0].request.primitive is not Primitive.DIRECT:
                raise ValueError("DIRECT trace must contain one DIRECT stage")
            if candidate_ids:
                raise ValueError("DIRECT trace must not run the continuation proposer")
        elif self.status is FixedHorizonStatus.OPEN_THEN_DIRECT_EXECUTED:
            if len(stages) != 2:
                raise ValueError("OPEN-then-DIRECT trace must contain two stages")
            if [stage.request.primitive for stage in stages] != [
                Primitive.OPEN,
                Primitive.DIRECT,
            ]:
                raise ValueError("OPEN-then-DIRECT stage order changed")
            if not candidate_ids or stages[1].candidate_id != candidate_ids[0]:
                raise ValueError("continuation must execute proposer top-1 DIRECT")
            if stages[1].candidate_fingerprint != fingerprints[0]:
                raise ValueError("continuation top-1 fingerprint changed")
        else:
            if len(stages) != 1 or stages[0].request.primitive is not Primitive.OPEN:
                raise ValueError("no-candidate trace must stop after OPEN")
            if candidate_ids:
                raise ValueError("no-candidate trace cannot claim proposed candidates")
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "continuation_candidate_ids", candidate_ids)
        object.__setattr__(self, "continuation_candidate_fingerprints", fingerprints)

    @property
    def control_steps_used(self) -> int:
        return sum(stage.control_steps_used for stage in self.stages)

    def to_dict(self) -> dict[str, object]:
        return {
            "continuation_policy_id": self.continuation_policy_id,
            "model_seed": self.model_seed,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_candidate_fingerprint": self.selected_candidate_fingerprint,
            "status": self.status.value,
            "stages": [stage.to_dict() for stage in self.stages],
            "continuation_candidate_ids": list(self.continuation_candidate_ids),
            "continuation_candidate_fingerprints": list(
                self.continuation_candidate_fingerprints
            ),
            "final_context_fingerprint": self.final_context.fingerprint(),
            "control_steps_used": self.control_steps_used,
        }

    @property
    def public_trace_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


class FixedHorizonContinuation:
    """Execute Method V1's single learned choice and fixed continuation."""

    def __init__(
        self,
        *,
        identity: FixedContinuationIdentity,
        proposer: FrozenDirectContinuationProposer,
        executor: BudgetedGroundedExecutor,
    ) -> None:
        if not isinstance(identity, FixedContinuationIdentity):
            raise TypeError("identity must be a FixedContinuationIdentity")
        if not isinstance(proposer, FrozenDirectContinuationProposer):
            raise TypeError("proposer must implement FrozenDirectContinuationProposer")
        if not isinstance(executor, BudgetedGroundedExecutor):
            raise TypeError("executor must implement BudgetedGroundedExecutor")
        observed = {
            "proposer_id": proposer.proposer_id,
            "executor_id": executor.executor_id,
            "serializer_id": executor.serializer_id,
            "replan_interval": executor.replan_interval,
        }
        expected = {
            "proposer_id": identity.proposer_id,
            "executor_id": identity.executor_id,
            "serializer_id": identity.serializer_id,
            "replan_interval": identity.replan_interval,
        }
        if observed != expected:
            raise ValueError(
                "runtime identity differs from frozen continuation identity: "
                f"expected={expected}, observed={observed}"
            )
        self.identity = identity
        self.proposer = proposer
        self.executor = executor

    @staticmethod
    def _validate_initial_candidate(
        candidate: GroundedIntervention,
        context: PolicyContext,
    ) -> None:
        if not isinstance(candidate, GroundedIntervention):
            raise TypeError("selected_candidate must be a GroundedIntervention")
        if candidate.primitive not in {Primitive.DIRECT, Primitive.OPEN}:
            raise ValueError("Method V1 executes DIRECT or OPEN only")
        candidate.validate_against(context)

    @staticmethod
    def _validate_stage(
        result: BudgetedExecution,
        *,
        candidate: GroundedIntervention,
        context: PolicyContext,
        budget: int,
        model_seed: int,
    ) -> None:
        if not isinstance(result, BudgetedExecution):
            raise TypeError("executor must return BudgetedExecution")
        expected = {
            "candidate_id": candidate.candidate_id,
            "candidate_fingerprint": candidate.fingerprint(),
            "previous_context": context,
            "requested_control_steps": budget,
            "control_steps_used": budget,
            "model_seed": model_seed,
        }
        observed = {
            "candidate_id": result.candidate_id,
            "candidate_fingerprint": result.candidate_fingerprint,
            "previous_context": result.previous_context,
            "requested_control_steps": result.requested_control_steps,
            "control_steps_used": result.control_steps_used,
            "model_seed": result.model_seed,
        }
        if observed != expected:
            raise ValueError(
                "executor result changes frozen candidate or budget identity"
            )

    def _execute(
        self,
        candidate: GroundedIntervention,
        context: PolicyContext,
        *,
        budget: int,
        model_seed: int,
    ) -> BudgetedExecution:
        result = self.executor.execute_candidate(
            candidate,
            context,
            control_step_budget=budget,
            model_seed=model_seed,
        )
        self._validate_stage(
            result,
            candidate=candidate,
            context=context,
            budget=budget,
            model_seed=model_seed,
        )
        return result

    def run(
        self,
        *,
        context: PolicyContext,
        selected_candidate: GroundedIntervention,
        model_seed: int,
    ) -> FixedHorizonExecution:
        if not isinstance(context, PolicyContext):
            raise TypeError("context must be a PolicyContext")
        if (
            not isinstance(model_seed, int)
            or isinstance(model_seed, bool)
            or model_seed < 0
        ):
            raise ValueError("model_seed must be a non-negative integer")
        self._validate_initial_candidate(selected_candidate, context)

        if selected_candidate.primitive is Primitive.DIRECT:
            direct = self._execute(
                selected_candidate,
                context,
                budget=self.identity.initial_direct_steps,
                model_seed=model_seed,
            )
            return FixedHorizonExecution(
                continuation_policy_id=self.identity.policy_id,
                model_seed=model_seed,
                selected_candidate_id=selected_candidate.candidate_id,
                selected_candidate_fingerprint=selected_candidate.fingerprint(),
                status=FixedHorizonStatus.DIRECT_EXECUTED,
                stages=(direct,),
                continuation_candidate_ids=(),
                continuation_candidate_fingerprints=(),
                final_context=direct.next_context,
            )

        opened = self._execute(
            selected_candidate,
            context,
            budget=self.identity.open_steps,
            model_seed=model_seed,
        )
        proposed = tuple(
            self.proposer.propose_direct(
                opened.next_context,
                max_candidates=self.identity.max_direct_candidates,
                deterministic_seed=self.identity.proposer_seed,
            )
        )
        if len(proposed) > self.identity.max_direct_candidates:
            raise ValueError("continuation proposer exceeded max_direct_candidates")
        ids: set[str] = set()
        fingerprints: set[str] = set()
        for candidate in proposed:
            if not isinstance(candidate, GroundedIntervention):
                raise TypeError("continuation proposer returned a non-candidate value")
            if candidate.primitive is not Primitive.DIRECT:
                raise ValueError("post-OPEN continuation may propose DIRECT only")
            candidate.validate_against(opened.next_context)
            if candidate.candidate_id in ids or candidate.fingerprint() in fingerprints:
                raise ValueError("continuation proposer returned duplicate candidates")
            ids.add(candidate.candidate_id)
            fingerprints.add(candidate.fingerprint())

        if not proposed:
            return FixedHorizonExecution(
                continuation_policy_id=self.identity.policy_id,
                model_seed=model_seed,
                selected_candidate_id=selected_candidate.candidate_id,
                selected_candidate_fingerprint=selected_candidate.fingerprint(),
                status=FixedHorizonStatus.OPEN_WITH_NO_DIRECT_CANDIDATE,
                stages=(opened,),
                continuation_candidate_ids=(),
                continuation_candidate_fingerprints=(),
                final_context=opened.next_context,
            )

        # This is deliberately top-1 in the frozen proposer's order.  There is
        # no learned high-level re-score after OPEN.
        continuation_candidate = proposed[0]
        direct = self._execute(
            continuation_candidate,
            opened.next_context,
            budget=self.identity.post_open_direct_steps,
            model_seed=model_seed,
        )
        return FixedHorizonExecution(
            continuation_policy_id=self.identity.policy_id,
            model_seed=model_seed,
            selected_candidate_id=selected_candidate.candidate_id,
            selected_candidate_fingerprint=selected_candidate.fingerprint(),
            status=FixedHorizonStatus.OPEN_THEN_DIRECT_EXECUTED,
            stages=(opened, direct),
            continuation_candidate_ids=tuple(item.candidate_id for item in proposed),
            continuation_candidate_fingerprints=tuple(
                item.fingerprint() for item in proposed
            ),
            final_context=direct.next_context,
        )
