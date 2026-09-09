from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from grounded_interaction.psr.collection import (
    ImmutableJSONLReceiptWriter,
    SnapshotCollectionPlan,
    collect_snapshot_rollouts,
    public_behavior_choices,
    verify_immutable_receipts,
)
from grounded_interaction.psr.data import CollectionStatus
from grounded_interaction.psr.molmo_backend import ActionChunk, EncodedHistory
from grounded_interaction.psr.types import (
    IntentCandidate,
    PublicHistory,
    PublicObservation,
    RGBReference,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _observation(step: int) -> PublicObservation:
    return PublicObservation(
        agentview_rgb=RGBReference(
            frame_id=f"agent-{step}",
            image_sha256=_digest(f"agent-{step}"),
            width=32,
            height=32,
        ),
        wrist_rgb=RGBReference(
            frame_id=f"wrist-{step}",
            image_sha256=_digest(f"wrist-{step}"),
            width=32,
            height=32,
        ),
        robot_state=(float(step), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        control_step=step,
    )


class _Environment:
    def __init__(self, initial_step: int) -> None:
        self.step = initial_step
        self.restore_count = 0
        self.closed = False

    def observe_public(self) -> PublicObservation:
        return _observation(self.step)

    def step_public(self, action: Any) -> PublicObservation:
        assert len(action) == 7
        self.step += 1
        return self.observe_public()

    def public_terminal(self) -> bool:
        return self.step >= 300

    def public_status(self) -> str:
        return "control budget active" if self.step < 300 else "control budget ended"

    def capture_private_snapshot(self) -> object:
        # In a real simulator this also contains simulator and RNG state.
        return {"step": self.step, "rng": (19, 23)}

    def restore_private_snapshot(self, snapshot: object) -> PublicObservation:
        assert isinstance(snapshot, dict)
        self.step = int(snapshot["step"])
        self.restore_count += 1
        return self.observe_public()

    def close(self) -> None:
        self.closed = True


class _Factory:
    def __init__(self, initial_step: int) -> None:
        self.environment = _Environment(initial_step)
        self.calls: list[tuple[str, int]] = []

    def create(self, *, initial_reset_ref: str, episode_seed: int) -> _Environment:
        self.calls.append((initial_reset_ref, episode_seed))
        return self.environment


class _EarlyEnvironment(_Environment):
    def public_terminal(self) -> bool:
        return self.step >= 240


class _EarlyFactory(_Factory):
    def __init__(self, initial_step: int) -> None:
        self.environment = _EarlyEnvironment(initial_step)
        self.calls = []


class _Policy:
    def __init__(
        self, candidates: tuple[IntentCandidate, ...], *, expected_step: int = 225
    ) -> None:
        self.candidates = candidates
        self.expected_step = expected_step
        self.reset_calls = 0
        self.action_calls: list[tuple[str, str, int, int, int]] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def encode(self, history: PublicHistory) -> EncodedHistory:
        return EncodedHistory(
            history_identity=history.fingerprint,
            task=history.task,
            state_tokens=None,
            prefix_inputs_embeds=None,
            prefix_input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            b_positions=None,
            native_inputs={},
            current_images=(None, None),
            current_state=history.current.robot_state,
        )

    def propose(
        self, encoded: EncodedHistory, *, episode_seed: int, global_step: int
    ) -> list[IntentCandidate]:
        assert encoded.history_identity
        assert episode_seed == 31
        assert global_step == self.expected_step
        return list(self.candidates)

    def act_chunk(
        self,
        history: PublicHistory,
        selected_intent: IntentCandidate,
        *,
        rng_seed: int,
        requested_steps: int,
    ) -> ActionChunk:
        self.action_calls.append(
            (
                selected_intent.candidate_id,
                selected_intent.execution_route,
                history.current.control_step,
                requested_steps,
                rng_seed,
            )
        )
        return ActionChunk(
            actions=tuple((0.0,) * 7 for _ in range(requested_steps)),
            execution_route=selected_intent.execution_route,
            requested_steps=requested_steps,
        )


class _Artifacts:
    def __init__(self, *, fail_evidence: bool = False) -> None:
        self.fail_evidence = fail_evidence
        self.action_rows: list[tuple[str, str, int, int]] = []
        self.evidence_steps: list[int] = []

    def write_action_chunk(
        self,
        *,
        record_id: str,
        phase: str,
        chunk_start_step: int,
        actions: Any,
    ) -> str:
        self.action_rows.append(
            (record_id, phase, chunk_start_step, len(tuple(actions)))
        )
        return f"actions://{record_id}/{phase}/{chunk_start_step}"

    def write_future_evidence(
        self,
        *,
        record_id: str,
        observation: PublicObservation,
        target_encoder_id: str,
    ) -> str:
        if self.fail_evidence:
            raise RuntimeError("target encoder unavailable")
        assert target_encoder_id == "native-patches-s0"
        self.evidence_steps.append(observation.control_step)
        return f"evidence://{record_id}/{observation.control_step}"


class _Evaluator:
    def __init__(self) -> None:
        self.steps: list[int] = []

    def final_success(self, environment: _Environment) -> bool:
        self.steps.append(environment.step)
        return environment.step == 300

    def terminal_reason(self, environment: _Environment) -> str:
        assert environment.step == 300
        return "original_budget_exhausted"


class _EarlyEvaluator(_Evaluator):
    def final_success(self, environment: _Environment) -> bool:
        self.steps.append(environment.step)
        return False

    def terminal_reason(self, environment: _Environment) -> str:
        assert environment.step == 240
        return "public_terminal"


def _candidates() -> tuple[IntentCandidate, ...]:
    return (
        IntentCandidate("Inspect the package label", (11, 12), "conditioned"),
        IntentCandidate("Put the requested item in the basket", (21, 22), "native"),
    )


def _plan(
    *, mode: str = "all_candidates", decision_step: int = 225
) -> SnapshotCollectionPlan:
    return SnapshotCollectionPlan(
        episode_id="episode-31",
        group_id="hidden-family-7/step-225",
        split="train",
        initial_reset_ref="collector://reset/private-31",
        public_history=PublicHistory(
            task="Put the requested item in the basket",
            current=_observation(decision_step),
            previous=(_observation(max(0, decision_step - 50)),)
            if decision_step > 0
            else (),
            executed=(),
            remaining_control_steps=300 - decision_step,
        ),
        episode_seed=31,
        behavior_seed=17,
        selection_mode=mode,  # type: ignore[arg-type]
        candidate_generation_version="molmo-language-s0-v1",
        execution_snapshot_id="psr-s0-checkpoint-v1",
        continuation_id="native-molmoact2-pic-v1",
        target_encoder_id="native-patches-s0",
    )


def test_same_reset_branches_obey_50_then_native_to_original_300(
    tmp_path: Path,
) -> None:
    candidates = _candidates()
    policy = _Policy(candidates)
    factory = _Factory(225)
    evaluator = _Evaluator()
    artifacts = _Artifacts()
    path = tmp_path / "receipts.jsonl"
    with ImmutableJSONLReceiptWriter(path) as writer:
        summary = collect_snapshot_rollouts(
            plan=_plan(),
            policy=policy,
            environment_factory=factory,
            evaluator=evaluator,
            artifact_sink=artifacts,
            receipt_writer=writer,
        )

    assert summary.proposed_candidates == 2
    assert summary.selected_branches == summary.completed == 2
    assert summary.infrastructure_failures == 0
    assert factory.environment.restore_count == 2
    assert factory.environment.closed
    assert evaluator.steps == [300, 300]
    assert artifacts.evidence_steps == [275, 275]
    assert all(record.actual_step_count == 75 for record in summary.records)
    assert all(
        record.cost_valid and record.evidence_valid for record in summary.records
    )
    assert all(record.final_success is True for record in summary.records)
    assert all(len(record.supervision()) == 5 for record in summary.records)

    conditioned_id = candidates[0].candidate_id
    conditioned_calls = [row for row in policy.action_calls if row[0] == conditioned_id]
    assert [(row[2], row[3]) for row in conditioned_calls] == [
        (225, 10),
        (235, 10),
        (245, 10),
        (255, 10),
        (265, 10),
    ]
    calls_after_conditioned_window = [
        row for row in policy.action_calls if row[2] >= 275
    ]
    assert calls_after_conditioned_window
    assert all(row[1] == "native" for row in calls_after_conditioned_window)
    assert [row[3] for row in calls_after_conditioned_window[:3]] == [10, 10, 5]

    loaded = verify_immutable_receipts(path)
    assert loaded == summary.records
    assert all(
        "final_success" not in json.dumps(record.model_input()) for record in loaded
    )


def test_public_uniform_choice_and_seed_ignore_candidate_order() -> None:
    candidates = _candidates() + (
        IntentCandidate("Turn the package to read it", (31, 32), "conditioned"),
    )
    plan = _plan(mode="sample_one")
    first = public_behavior_choices(
        history=plan.public_history,
        candidates=candidates,
        episode_seed=plan.episode_seed,
        behavior_seed=plan.behavior_seed,
        mode="sample_one",
    )
    second = public_behavior_choices(
        history=plan.public_history,
        candidates=tuple(reversed(candidates)),
        episode_seed=plan.episode_seed,
        behavior_seed=plan.behavior_seed,
        mode="sample_one",
    )
    assert first == second
    assert first[0].probability == pytest.approx(1 / 3)
    assert first[0].rule == "public-uniform-one-v1"


def test_infrastructure_failure_keeps_trace_but_has_no_e_or_c_label(
    tmp_path: Path,
) -> None:
    policy = _Policy(_candidates())
    factory = _Factory(225)
    evaluator = _Evaluator()
    path = tmp_path / "failed.jsonl"
    with ImmutableJSONLReceiptWriter(path) as writer:
        summary = collect_snapshot_rollouts(
            plan=_plan(mode="sample_one"),
            policy=policy,
            environment_factory=factory,
            evaluator=evaluator,
            artifact_sink=_Artifacts(fail_evidence=True),
            receipt_writer=writer,
        )
    assert summary.completed == 0
    assert summary.infrastructure_failures == 1
    record = summary.records[0]
    assert record.collection_status is CollectionStatus.INFRASTRUCTURE_FAILURE
    assert record.actual_step_count == 50
    assert record.actual_action_chunks
    assert record.future_evidence_ref is None
    assert record.final_success is None
    assert record.evidence_valid is record.cost_valid is False
    assert record.supervision()["failure"] is None
    assert evaluator.steps == []
    assert verify_immutable_receipts(path) == (record,)


def test_receipt_log_is_create_once_and_tamper_evident(tmp_path: Path) -> None:
    path = tmp_path / "receipts.jsonl"
    policy = _Policy(_candidates())
    with ImmutableJSONLReceiptWriter(path) as writer:
        collect_snapshot_rollouts(
            plan=_plan(mode="sample_one"),
            policy=policy,
            environment_factory=_Factory(225),
            evaluator=_Evaluator(),
            artifact_sink=_Artifacts(),
            receipt_writer=writer,
        )
    assert len(verify_immutable_receipts(path)) == 1
    with pytest.raises(FileExistsError):
        ImmutableJSONLReceiptWriter(path)

    original = path.read_text(encoding="utf-8")
    envelope = json.loads(original)
    envelope["record"]["terminal_reason"] = "tampered"
    path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_immutable_receipts(path)


def test_plan_refuses_a_new_budget_after_a_late_decision() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="original 300-step budget"):
        dataclasses.replace(
            plan,
            public_history=dataclasses.replace(
                plan.public_history, remaining_control_steps=300
            ),
        )


def test_less_than_50_steps_uses_only_the_original_remaining_budget(
    tmp_path: Path,
) -> None:
    policy = _Policy(_candidates(), expected_step=280)
    artifacts = _Artifacts()
    path = tmp_path / "late.jsonl"
    with ImmutableJSONLReceiptWriter(path) as writer:
        summary = collect_snapshot_rollouts(
            plan=_plan(mode="sample_one", decision_step=280),
            policy=policy,
            environment_factory=_Factory(280),
            evaluator=_Evaluator(),
            artifact_sink=artifacts,
            receipt_writer=writer,
        )
    assert summary.records[0].actual_step_count == 20
    assert artifacts.evidence_steps == [300]
    assert [row[1] for row in artifacts.action_rows] == ["intent", "intent"]
    assert [row[3] for row in artifacts.action_rows] == [10, 10]


def test_early_public_terminal_keeps_cost_but_masks_missing_boundary_evidence(
    tmp_path: Path,
) -> None:
    policy = _Policy(_candidates())
    artifacts = _Artifacts()
    path = tmp_path / "early.jsonl"
    with ImmutableJSONLReceiptWriter(path) as writer:
        summary = collect_snapshot_rollouts(
            plan=_plan(mode="sample_one"),
            policy=policy,
            environment_factory=_EarlyFactory(225),
            evaluator=_EarlyEvaluator(),
            artifact_sink=artifacts,
            receipt_writer=writer,
        )
    record = summary.records[0]
    assert record.collection_status is CollectionStatus.COMPLETED
    assert record.actual_step_count == 15
    assert record.cost_valid and record.final_success is False
    assert not record.evidence_valid and record.future_evidence_ref is None
    assert artifacts.evidence_steps == []
