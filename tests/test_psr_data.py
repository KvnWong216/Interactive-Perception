from __future__ import annotations

import dataclasses

import pytest

from grounded_interaction.psr.data import (
    PSR_BRANCH_SCHEMA,
    CollectionStatus,
    PSRBranchRecord,
    RecordKind,
    collate_public_and_labels,
    validate_records,
)
from grounded_interaction.psr.types import (
    IntentCandidate,
    PublicHistory,
    PublicObservation,
    RGBReference,
    candidate_seed,
)


def _observation(step: int, character: str) -> PublicObservation:
    return PublicObservation(
        RGBReference(f"agent-{step}", character * 64, 64, 64),
        RGBReference(f"wrist-{step}", character * 64, 64, 64),
        (0.0,) * 8,
        step,
    )


def _record(**changes: object) -> PSRBranchRecord:
    candidates = (
        IntentCandidate("Inspect the package label", (1, 2), "conditioned"),
        IntentCandidate("Put the butter in the basket", (3, 4), "native"),
    )
    values: dict[str, object] = {
        "schema_version": PSR_BRANCH_SCHEMA,
        "record_kind": RecordKind.SNAPSHOT_ROLLOUTS,
        "record_id": "record-1",
        "episode_id": "episode-1",
        "group_id": "group-1",
        "split": "train",
        "decision_step": 0,
        "remaining_steps": 300,
        "public_history": PublicHistory(
            "Put the butter in the basket", _observation(0, "a"), (), (), 300
        ),
        "candidate_set": candidates,
        "chosen_candidate_id": candidates[0].candidate_id,
        "chosen_intent_ids": candidates[0].token_ids,
        "candidate_generation_version": "psr-generator-v1",
        "behavior_selection_rule": "public-uniform-v1",
        "behavior_probability": 0.5,
        "execution_snapshot_id": "psr-s0-v1",
        "continuation_id": "native-molmoact2-fixed-v1",
        "target_encoder_id": "frozen-native-patches-v1",
        "initial_reset_ref": "collector://reset/1",
        "actual_action_chunks": ("actions://chunk/1",),
        "actual_step_count": 50,
        "actual_observations": (_observation(50, "b"),),
        "future_evidence_ref": "features://episode-1/50",
        "evidence_valid": True,
        "final_success": True,
        "cost_valid": True,
        "terminal_reason": "budget_end",
        "seed_schedule": {"episode_seed": 17, "candidate_seed": 23},
        "collection_status": CollectionStatus.COMPLETED,
    }
    values.update(changes)
    return PSRBranchRecord(**values)  # type: ignore[arg-type]


def test_record_round_trip_and_label_firewall() -> None:
    record = _record()
    assert PSRBranchRecord.from_mapping(record.to_dict()) == record
    forward, labels = collate_public_and_labels((record,))
    serialized = repr(forward)
    for private_name in (
        "initial_reset_ref",
        "final_success",
        "future_evidence_ref",
        "execution_snapshot_id",
        "chosen_candidate_id",
        "candidate_id",
    ):
        assert private_name not in serialized
    assert labels[0]["failure"] is False
    changed_label = dataclasses.replace(
        record, record_id="record-2", final_success=False
    )
    assert changed_label.model_input() == record.model_input()
    assert changed_label.supervision()["failure"] is True


def test_history_and_candidate_identity_are_stable() -> None:
    record = _record()
    assert (
        PublicHistory.from_mapping(record.public_history.to_dict())
        == record.public_history
    )
    candidate = record.candidate_set[0]
    reconstructed = IntentCandidate(
        candidate.text, candidate.token_ids, candidate.execution_route
    )
    assert reconstructed.candidate_id == candidate.candidate_id
    assert candidate_seed(17, 31, 0, reconstructed) == candidate_seed(
        17, 31, 0, candidate
    )
    reordered = dataclasses.replace(
        record, candidate_set=tuple(reversed(record.candidate_set))
    )
    assert reordered.chosen_candidate_id == record.chosen_candidate_id


def test_dataset_rejects_stage_split_identity_and_exact_repeat() -> None:
    first = _record()
    second_candidate = first.candidate_set[1]
    second = dataclasses.replace(
        first,
        record_id="record-2",
        group_id="group-2",
        chosen_candidate_id=second_candidate.candidate_id,
        chosen_intent_ids=second_candidate.token_ids,
    )
    summary = validate_records(
        (first, second),
        expected_kind=RecordKind.SNAPSHOT_ROLLOUTS,
        expected_snapshot_id="psr-s0-v1",
    )
    assert summary.records == 2
    with pytest.raises(ValueError, match="record kinds"):
        validate_records(
            (
                first,
                dataclasses.replace(
                    second, record_kind=RecordKind.CLOSED_LOOP_EVALUATION
                ),
            )
        )
    with pytest.raises(ValueError, match="episode_id crosses"):
        validate_records((first, dataclasses.replace(second, split="test")))
    with pytest.raises(ValueError, match="execution_snapshot_id mismatch"):
        validate_records((first,), expected_snapshot_id="other-s0")
    with pytest.raises(ValueError, match="same seed schedule"):
        validate_records((first, dataclasses.replace(first, record_id="record-3")))
    independent_repeat = dataclasses.replace(
        first,
        record_id="record-4",
        seed_schedule={"episode_seed": 17, "candidate_seed": 24},
    )
    assert validate_records((first, independent_repeat)).records == 2


def test_infrastructure_failure_is_unlabelled() -> None:
    failure = _record(
        record_id="infra-1",
        actual_action_chunks=(),
        actual_step_count=0,
        actual_observations=(),
        future_evidence_ref=None,
        evidence_valid=False,
        final_success=None,
        cost_valid=False,
        terminal_reason="model service unavailable",
        collection_status=CollectionStatus.INFRASTRUCTURE_FAILURE,
    )
    assert failure.supervision()["cost_valid"] is False
    assert failure.supervision()["evidence_valid"] is False
    with pytest.raises(ValueError, match="audit-only"):
        dataclasses.replace(failure, final_success=False, cost_valid=True)
