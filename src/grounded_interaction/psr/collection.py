"""Outcome-bearing collection for the frozen PSR-VLA v1 protocol.

This module is deliberately the only place where a private simulator snapshot
and a private task evaluator coexist.  Neither object is ever passed to the
policy.  The policy sees only :class:`PublicHistory`; the evaluator is invoked
only after the selected intent window and the fixed native continuation have
finished.

One :class:`PSRBranchRecord` represents one candidate that was actually
executed.  A candidate that was not executed never receives the executed
candidate's future evidence or task label.  Infrastructure failures are
retained as unlabelled records instead of being silently removed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, Self

from grounded_interaction.contracts import canonical_json_bytes, canonical_sha256

from .data import (
    PSR_BRANCH_SCHEMA,
    CollectionStatus,
    PSRBranchRecord,
    RecordKind,
)
from .molmo_backend import ActionChunk, EncodedHistory
from .types import ExecutedIntent, IntentCandidate, PublicHistory, PublicObservation

TOTAL_CONTROL_STEPS = 300
INTENT_WINDOW_STEPS = 50
EXECUTE_CHUNK_STEPS = 10
RECEIPT_SCHEMA = "psr-v1-immutable-receipt-v1"
RECEIPT_SEAL_SCHEMA = "psr-v1-immutable-receipt-seal-v1"
SelectionMode = Literal["sample_one", "all_candidates"]


class CollectionPolicy(Protocol):
    """In-process policy surface used by the collector.

    ``predict`` and ``choose`` are intentionally absent.  Result collection is
    driven by a fixed public behavior policy, not by the C head being fitted.
    """

    def reset(self) -> None: ...

    def encode(self, history: PublicHistory) -> EncodedHistory: ...

    def propose(
        self,
        encoded: EncodedHistory,
        *,
        episode_seed: int,
        global_step: int,
    ) -> list[IntentCandidate]: ...

    def act_chunk(
        self,
        history: PublicHistory,
        selected_intent: IntentCandidate,
        *,
        rng_seed: int,
        requested_steps: int,
    ) -> ActionChunk: ...


class SameResetEnvironment(Protocol):
    """Simulator interface with an explicit private snapshot boundary."""

    def observe_public(self) -> PublicObservation: ...

    def step_public(self, action: Sequence[float]) -> PublicObservation: ...

    def public_terminal(self) -> bool: ...

    def public_status(self) -> str: ...

    def capture_private_snapshot(self) -> object: ...

    def restore_private_snapshot(self, snapshot: object) -> PublicObservation: ...

    def close(self) -> None: ...


class SameResetEnvironmentFactory(Protocol):
    """Create an environment at the collector-only reset reference."""

    def create(
        self, *, initial_reset_ref: str, episode_seed: int
    ) -> SameResetEnvironment: ...


class PrivateOutcomeEvaluator(Protocol):
    """Evaluator-only final predicate; never a policy input or stop signal."""

    def final_success(self, environment: SameResetEnvironment) -> bool: ...

    def terminal_reason(self, environment: SameResetEnvironment) -> str: ...


class CollectionArtifactSink(Protocol):
    """Persist action chunks and a frozen-native future-evidence target."""

    def write_action_chunk(
        self,
        *,
        record_id: str,
        phase: Literal["intent", "continuation"],
        chunk_start_step: int,
        actions: Sequence[Sequence[float]],
    ) -> str: ...

    def write_future_evidence(
        self,
        *,
        record_id: str,
        observation: PublicObservation,
        target_encoder_id: str,
    ) -> str: ...


@dataclasses.dataclass(frozen=True)
class SnapshotCollectionPlan:
    """One frozen public decision point and its execution identities."""

    episode_id: str
    group_id: str
    split: str
    initial_reset_ref: str
    public_history: PublicHistory
    episode_seed: int
    behavior_seed: int
    selection_mode: SelectionMode
    candidate_generation_version: str
    execution_snapshot_id: str
    continuation_id: str
    target_encoder_id: str

    def __post_init__(self) -> None:
        for name in (
            "episode_id",
            "group_id",
            "initial_reset_ref",
            "candidate_generation_version",
            "execution_snapshot_id",
            "continuation_id",
            "target_encoder_id",
        ):
            value = " ".join(str(getattr(self, name) or "").split())
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        if self.split not in {"train", "validation", "calibration", "test"}:
            raise ValueError("split must be train/validation/calibration/test")
        if not isinstance(self.public_history, PublicHistory):
            raise TypeError("public_history must be PublicHistory")
        for name in ("episode_seed", "behavior_seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.selection_mode not in {"sample_one", "all_candidates"}:
            raise ValueError("selection_mode must be sample_one or all_candidates")
        if (
            self.public_history.current.control_step
            + self.public_history.remaining_control_steps
            != TOTAL_CONTROL_STEPS
        ):
            raise ValueError(
                "PSR v1 collection must preserve the original 300-step budget"
            )
        if self.public_history.remaining_control_steps == 0:
            raise ValueError("a collection decision requires remaining control budget")


@dataclasses.dataclass(frozen=True)
class BehaviorChoice:
    """A public, reproducible branch-inclusion decision."""

    candidate: IntentCandidate
    rule: str
    probability: float
    seed: int


@dataclasses.dataclass(frozen=True)
class CollectionSummary:
    records: tuple[PSRBranchRecord, ...]
    proposed_candidates: int
    selected_branches: int
    completed: int
    infrastructure_failures: int


def _stable_seed(namespace: str, payload: Mapping[str, Any]) -> int:
    body = {"schema": namespace, **dict(payload)}
    return int.from_bytes(
        hashlib.sha256(canonical_json_bytes(body)).digest()[:8], "big"
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    """Write a complete canonical record even if the OS reports a short write."""

    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise OSError("failed while writing immutable receipt bytes")
        offset += written


def _candidate_set(
    candidates: Sequence[IntentCandidate], *, native_task: str | None = None
) -> tuple[IntentCandidate, ...]:
    rows = tuple(candidates)
    if not rows:
        raise RuntimeError("candidate generation returned no candidates")
    if any(not isinstance(candidate, IntentCandidate) for candidate in rows):
        raise TypeError("candidate generation returned an invalid candidate")
    by_id = {candidate.candidate_id: candidate for candidate in rows}
    if len(by_id) != len(rows):
        raise RuntimeError("candidate generation returned duplicate candidates")
    native = [candidate for candidate in rows if candidate.execution_route == "native"]
    if len(native) != 1:
        raise RuntimeError("PSR collection requires exactly one true native candidate")
    if native_task is not None and native[0].text != " ".join(native_task.split()):
        raise RuntimeError("the native candidate must carry the original task text")
    # Canonical ordering makes artifacts identical if a proposer returns the
    # same semantic set in another list order.  Candidate IDs are never model
    # inputs; this ordering is collection provenance only.
    return tuple(by_id[key] for key in sorted(by_id))


def public_behavior_choices(
    *,
    history: PublicHistory,
    candidates: Sequence[IntentCandidate],
    episode_seed: int,
    behavior_seed: int,
    mode: SelectionMode,
) -> tuple[BehaviorChoice, ...]:
    """Select branches without outcome access and without list-order seeds."""

    canonical = _candidate_set(candidates, native_task=history.task)
    common = {
        "history": history.fingerprint,
        "candidate_ids": sorted(candidate.candidate_id for candidate in canonical),
        "episode_seed": episode_seed,
        "behavior_seed": behavior_seed,
        "decision_step": history.current.control_step,
    }
    selection_seed = _stable_seed("psr-public-behavior-selection-v1", common)
    if mode == "sample_one":
        sorted_ids = sorted(candidate.candidate_id for candidate in canonical)
        selected_id = sorted_ids[selection_seed % len(sorted_ids)]
        selected = next(
            candidate
            for candidate in canonical
            if candidate.candidate_id == selected_id
        )
        return (
            BehaviorChoice(
                candidate=selected,
                rule="public-uniform-one-v1",
                probability=1.0 / len(canonical),
                seed=selection_seed,
            ),
        )
    if mode == "all_candidates":
        return tuple(
            BehaviorChoice(
                candidate=candidate,
                rule="same-reset-all-candidates-v1",
                probability=1.0,
                seed=_stable_seed(
                    "psr-public-all-candidates-branch-v1",
                    {**common, "candidate_id": candidate.candidate_id},
                ),
            )
            for candidate in canonical
        )
    raise ValueError("unknown behavior-selection mode")


def _action_seed(
    *,
    episode_seed: int,
    decision_step: int,
    candidate_id: str,
    control_step: int,
    phase: str,
) -> int:
    return _stable_seed(
        "psr-v1-action-seed-v1",
        {
            "episode_seed": episode_seed,
            "decision_step": decision_step,
            "candidate_id": candidate_id,
            "control_step": control_step,
            "phase": phase,
        },
    )


def _record_id(plan: SnapshotCollectionPlan, candidate: IntentCandidate) -> str:
    identity = {
        "schema": "psr-v1-branch-identity-v1",
        "episode_id": plan.episode_id,
        "group_id": plan.group_id,
        "history": plan.public_history.fingerprint,
        "candidate_id": candidate.candidate_id,
        "episode_seed": plan.episode_seed,
        "behavior_seed": plan.behavior_seed,
        "execution_snapshot_id": plan.execution_snapshot_id,
        "continuation_id": plan.continuation_id,
    }
    return "psr-branch-" + canonical_sha256(identity)


def _advance_with_real_observation(
    history: PublicHistory, observation: PublicObservation
) -> PublicHistory:
    expected = history.current.control_step + 1
    if observation.control_step != expected:
        raise RuntimeError(
            f"non-contiguous public observation: expected step {expected}, "
            f"received {observation.control_step}"
        )
    return PublicHistory(
        task=history.task,
        current=observation,
        previous=history.previous,
        executed=history.executed,
        remaining_control_steps=history.remaining_control_steps - 1,
    )


def _execute_phase(
    *,
    policy: CollectionPolicy,
    environment: SameResetEnvironment,
    artifact_sink: CollectionArtifactSink,
    record_id: str,
    history: PublicHistory,
    intent: IntentCandidate,
    step_budget: int,
    episode_seed: int,
    decision_step: int,
    phase: Literal["intent", "continuation"],
    action_refs_out: list[str],
    observations_out: list[PublicObservation],
) -> PublicHistory:
    """Execute one fixed intent, reobserving and re-encoding every chunk."""

    if step_budget < 0 or step_budget > history.remaining_control_steps:
        raise ValueError("phase step budget is outside the remaining episode budget")
    executed = 0
    while executed < step_budget and not environment.public_terminal():
        requested = min(EXECUTE_CHUNK_STEPS, step_budget - executed)
        seed = _action_seed(
            episode_seed=episode_seed,
            decision_step=decision_step,
            candidate_id=intent.candidate_id,
            control_step=history.current.control_step,
            phase=phase,
        )
        # act_chunk performs a fresh encode from the actual current history.
        chunk = policy.act_chunk(
            history, intent, rng_seed=seed, requested_steps=requested
        )
        actions = tuple(tuple(float(value) for value in row) for row in chunk.actions)
        if not actions:
            raise RuntimeError("action backend returned an empty chunk")
        applied: list[tuple[float, ...]] = []
        chunk_start = history.current.control_step
        for action in actions[:requested]:
            if len(action) != 7 or any(not math.isfinite(value) for value in action):
                raise RuntimeError("action backend returned a non-finite non-7D action")
            observation = environment.step_public(action)
            history = _advance_with_real_observation(history, observation)
            applied.append(action)
            observations_out.append(observation)
            executed += 1
            if executed == step_budget or environment.public_terminal():
                break
        if not applied:
            raise RuntimeError("no action from the generated chunk was applied")
        action_refs_out.append(
            artifact_sink.write_action_chunk(
                record_id=record_id,
                phase=phase,
                chunk_start_step=chunk_start,
                actions=applied,
            )
        )
    return history


def _with_completed_intent(
    *,
    history: PublicHistory,
    decision_start: PublicObservation,
    candidate: IntentCandidate,
    action_refs: Sequence[str],
    public_status: str,
) -> PublicHistory:
    if history.current.control_step <= decision_start.control_step:
        raise RuntimeError("an intent window completed without a real control step")
    event = ExecutedIntent(
        text=candidate.text,
        execution_route=candidate.execution_route,
        start_step=decision_start.control_step,
        end_step=history.current.control_step,
        actions_ref="sha256:"
        + canonical_sha256(
            {
                "schema": "psr-v1-window-action-references-v1",
                "references": list(action_refs),
            }
        ),
        public_status=public_status,
    )
    return PublicHistory(
        task=history.task,
        current=history.current,
        previous=(*history.previous, decision_start)[-2:],
        executed=(*history.executed, event),
        remaining_control_steps=history.remaining_control_steps,
    )


def _branch_record(
    *,
    plan: SnapshotCollectionPlan,
    candidates: tuple[IntentCandidate, ...],
    choice: BehaviorChoice,
    action_refs: Sequence[str],
    observations: Sequence[PublicObservation],
    future_evidence_ref: str | None,
    final_success: bool | None,
    terminal_reason: str,
    status: CollectionStatus,
) -> PSRBranchRecord:
    valid = status is CollectionStatus.COMPLETED
    return PSRBranchRecord(
        schema_version=PSR_BRANCH_SCHEMA,
        record_kind=RecordKind.SNAPSHOT_ROLLOUTS,
        record_id=_record_id(plan, choice.candidate),
        episode_id=plan.episode_id,
        group_id=plan.group_id,
        split=plan.split,
        decision_step=plan.public_history.current.control_step,
        remaining_steps=plan.public_history.remaining_control_steps,
        public_history=plan.public_history,
        candidate_set=candidates,
        chosen_candidate_id=choice.candidate.candidate_id,
        chosen_intent_ids=choice.candidate.token_ids,
        candidate_generation_version=plan.candidate_generation_version,
        behavior_selection_rule=choice.rule,
        behavior_probability=choice.probability,
        execution_snapshot_id=plan.execution_snapshot_id,
        continuation_id=plan.continuation_id,
        target_encoder_id=plan.target_encoder_id,
        initial_reset_ref=plan.initial_reset_ref,
        actual_action_chunks=tuple(action_refs),
        actual_step_count=len(observations),
        actual_observations=tuple(observations),
        future_evidence_ref=future_evidence_ref if valid else None,
        evidence_valid=valid and future_evidence_ref is not None,
        final_success=final_success if valid else None,
        cost_valid=valid,
        terminal_reason=terminal_reason,
        seed_schedule={
            "schema": "psr-v1-collection-seeds-v1",
            "episode_seed": plan.episode_seed,
            "behavior_seed": plan.behavior_seed,
            "behavior_selection_seed": choice.seed,
            "action_seed_rule": "content-and-control-step-v1",
        },
        collection_status=status,
    )


def _collect_branch(
    *,
    plan: SnapshotCollectionPlan,
    candidates: tuple[IntentCandidate, ...],
    choice: BehaviorChoice,
    policy: CollectionPolicy,
    environment: SameResetEnvironment,
    evaluator: PrivateOutcomeEvaluator,
    artifact_sink: CollectionArtifactSink,
) -> PSRBranchRecord:
    """Run 50-step U, capture E, then native pi_c to the original horizon."""

    record_id = _record_id(plan, choice.candidate)
    history = plan.public_history
    decision_start = history.current
    action_refs: list[str] = []
    observations: list[PublicObservation] = []
    try:
        window = min(INTENT_WINDOW_STEPS, history.remaining_control_steps)
        window_ref_start = len(action_refs)
        history = _execute_phase(
            policy=policy,
            environment=environment,
            artifact_sink=artifact_sink,
            record_id=record_id,
            history=history,
            intent=choice.candidate,
            step_budget=window,
            episode_seed=plan.episode_seed,
            decision_step=decision_start.control_step,
            phase="intent",
            action_refs_out=action_refs,
            observations_out=observations,
        )
        window_action_refs = action_refs[window_ref_start:]
        # E exists only at the specified U-window boundary. If a public
        # terminal ends execution earlier, its last frame is real but is not
        # the registered future target and therefore remains masked.
        reached_evidence_boundary = (
            history.current.control_step == decision_start.control_step + window
        )
        future_evidence_ref = None
        if reached_evidence_boundary:
            future_evidence_ref = artifact_sink.write_future_evidence(
                record_id=record_id,
                observation=history.current,
                target_encoder_id=plan.target_encoder_id,
            )
        history = _with_completed_intent(
            history=history,
            decision_start=decision_start,
            candidate=choice.candidate,
            action_refs=window_action_refs,
            public_status=environment.public_status(),
        )
        native = next(
            candidate
            for candidate in candidates
            if candidate.execution_route == "native"
        )
        # No proposal, C-head query, or learned choice occurs in pi_c.
        history = _execute_phase(
            policy=policy,
            environment=environment,
            artifact_sink=artifact_sink,
            record_id=record_id,
            history=history,
            intent=native,
            step_budget=history.remaining_control_steps,
            episode_seed=plan.episode_seed,
            decision_step=decision_start.control_step,
            phase="continuation",
            action_refs_out=action_refs,
            observations_out=observations,
        )
        success = evaluator.final_success(environment)
        if not isinstance(success, bool):
            raise TypeError("private evaluator must return bool")
        reason = " ".join(evaluator.terminal_reason(environment).split())
        if not reason:
            raise ValueError("private evaluator terminal reason must be non-empty")
        return _branch_record(
            plan=plan,
            candidates=candidates,
            choice=choice,
            action_refs=action_refs,
            observations=observations,
            future_evidence_ref=future_evidence_ref,
            final_success=success,
            terminal_reason=reason,
            status=CollectionStatus.COMPLETED,
        )
    except Exception as error:  # noqa: BLE001 - collector is the audit boundary.
        return _branch_record(
            plan=plan,
            candidates=candidates,
            choice=choice,
            action_refs=action_refs,
            observations=observations,
            future_evidence_ref=None,
            final_success=None,
            terminal_reason=f"infrastructure_failure:{type(error).__name__}",
            status=CollectionStatus.INFRASTRUCTURE_FAILURE,
        )


def collect_snapshot_rollouts(
    *,
    plan: SnapshotCollectionPlan,
    policy: CollectionPolicy,
    environment_factory: SameResetEnvironmentFactory,
    evaluator: PrivateOutcomeEvaluator,
    artifact_sink: CollectionArtifactSink,
    receipt_writer: ImmutableJSONLReceiptWriter,
) -> CollectionSummary:
    """Collect actual S0 branches from one frozen public decision point.

    Candidate proposal happens once from public H.  Every included branch is
    restored to the exact same private simulator/RNG snapshot and its public
    reset observation is checked before physical execution starts.
    """

    if receipt_writer.closed:
        raise RuntimeError("receipt writer is already closed")
    environment = environment_factory.create(
        initial_reset_ref=plan.initial_reset_ref, episode_seed=plan.episode_seed
    )
    records: list[PSRBranchRecord] = []
    try:
        observed = environment.observe_public()
        if observed != plan.public_history.current:
            raise RuntimeError(
                "environment does not reproduce the frozen public history"
            )
        private_snapshot = environment.capture_private_snapshot()
        policy.reset()
        encoded = policy.encode(plan.public_history)
        candidates = _candidate_set(
            policy.propose(
                encoded,
                episode_seed=plan.episode_seed,
                global_step=plan.public_history.current.control_step,
            ),
            native_task=plan.public_history.task,
        )
        choices = public_behavior_choices(
            history=plan.public_history,
            candidates=candidates,
            episode_seed=plan.episode_seed,
            behavior_seed=plan.behavior_seed,
            mode=plan.selection_mode,
        )
        for choice in choices:
            policy.reset()
            restored = environment.restore_private_snapshot(private_snapshot)
            if restored != plan.public_history.current:
                raise RuntimeError(
                    "private snapshot restore changed the public decision state"
                )
            record = _collect_branch(
                plan=plan,
                candidates=candidates,
                choice=choice,
                policy=policy,
                environment=environment,
                evaluator=evaluator,
                artifact_sink=artifact_sink,
            )
            receipt_writer.append(record)
            records.append(record)
    finally:
        environment.close()
    return CollectionSummary(
        records=tuple(records),
        proposed_candidates=len(candidates),
        selected_branches=len(records),
        completed=sum(
            record.collection_status is CollectionStatus.COMPLETED for record in records
        ),
        infrastructure_failures=sum(
            record.collection_status is CollectionStatus.INFRASTRUCTURE_FAILURE
            for record in records
        ),
    )


class ImmutableJSONLReceiptWriter:
    """Create-once, fsyncing, hash-chained JSONL branch receipts.

    The output and its seal are opened with ``O_EXCL``.  A second collector can
    therefore neither resume ambiguously nor overwrite a consumed experiment.
    Recovery from a crash must create a new explicitly versioned output.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.seal_path = self.path.with_suffix(self.path.suffix + ".seal.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.seal_path.exists():
            raise FileExistsError(self.seal_path)
        self._fd = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        self._previous_sha256 = "0" * 64
        self._record_ids: set[str] = set()
        self._count = 0
        self.closed = False

    def append(self, record: PSRBranchRecord) -> str:
        if self.closed:
            raise RuntimeError("cannot append to a closed receipt writer")
        if not isinstance(record, PSRBranchRecord):
            raise TypeError("receipt payload must be a PSRBranchRecord")
        if record.record_id in self._record_ids:
            raise ValueError("refusing a duplicate record receipt")
        body = {
            "schema_version": RECEIPT_SCHEMA,
            "sequence": self._count,
            "previous_receipt_sha256": self._previous_sha256,
            "record": record.to_dict(),
        }
        digest = canonical_sha256(body)
        envelope = {**body, "receipt_sha256": digest}
        line = canonical_json_bytes(envelope) + b"\n"
        _write_all(self._fd, line)
        os.fsync(self._fd)
        self._record_ids.add(record.record_id)
        self._previous_sha256 = digest
        self._count += 1
        return digest

    def close(self) -> None:
        if self.closed:
            return
        os.close(self._fd)
        self.closed = True
        seal = {
            "schema_version": RECEIPT_SEAL_SCHEMA,
            "records": self._count,
            "final_receipt_sha256": self._previous_sha256,
            "jsonl_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
        }
        descriptor = os.open(
            self.seal_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            payload = canonical_json_bytes(seal) + b"\n"
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def verify_immutable_receipts(path: str | Path) -> tuple[PSRBranchRecord, ...]:
    """Verify the complete chain and create-once seal before loading records."""

    source = Path(path).expanduser().resolve()
    seal_path = source.with_suffix(source.suffix + ".seal.json")
    if not source.is_file() or not seal_path.is_file():
        raise FileNotFoundError("both immutable JSONL and its seal are required")
    lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line]
    previous = "0" * 64
    records: list[PSRBranchRecord] = []
    for sequence, line in enumerate(lines):
        envelope = json.loads(line)
        expected_keys = {
            "schema_version",
            "sequence",
            "previous_receipt_sha256",
            "record",
            "receipt_sha256",
        }
        if set(envelope) != expected_keys:
            raise ValueError("receipt envelope fields differ from schema")
        if envelope["schema_version"] != RECEIPT_SCHEMA:
            raise ValueError("receipt schema version mismatch")
        if envelope["sequence"] != sequence:
            raise ValueError("receipt sequence is not contiguous")
        if envelope["previous_receipt_sha256"] != previous:
            raise ValueError("receipt hash chain is broken")
        declared = envelope.pop("receipt_sha256")
        observed = canonical_sha256(envelope)
        if declared != observed:
            raise ValueError("receipt digest mismatch")
        previous = observed
        records.append(PSRBranchRecord.from_mapping(envelope["record"]))
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("duplicate record ID in immutable receipt log")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if set(seal) != {
        "schema_version",
        "records",
        "final_receipt_sha256",
        "jsonl_sha256",
    }:
        raise ValueError("receipt seal fields differ from schema")
    if seal["schema_version"] != RECEIPT_SEAL_SCHEMA:
        raise ValueError("receipt seal schema mismatch")
    if seal["records"] != len(records) or seal["final_receipt_sha256"] != previous:
        raise ValueError("receipt seal does not match the JSONL chain")
    if seal["jsonl_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError("receipt JSONL byte digest mismatch")
    return tuple(records)
