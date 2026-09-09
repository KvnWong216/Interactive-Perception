"""Strict, outcome-bearing data records for the PSR V1 execution protocol.

The classes in this module deliberately keep the policy input and the label
store separate.  A :class:`PSRBranchRecord` describes one candidate that was
*actually* executed.  Other entries in ``candidate_set`` are context for the
recorded behavior policy; they never inherit the chosen candidate's future
evidence or task result.

This is a schema and validation layer, not a simulator collector.  In
particular, reset handles and outcomes are retained for audit/training while
``model_input`` returns only the public history and open-vocabulary intents.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from grounded_interaction.contracts import (
    assert_public_policy_value,
    canonical_json_bytes,
    canonical_sha256,
)

from .types import IntentCandidate, PublicHistory, PublicObservation

PSR_BRANCH_SCHEMA = "psr-v1-branch-record-v1"
PSR_SPLITS = frozenset({"train", "validation", "calibration", "test"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RecordKind(str, enum.Enum):
    """Three intentionally non-interchangeable uses of real trajectories."""

    WARMUP_DEMONSTRATIONS = "warmup_demonstrations"
    SNAPSHOT_ROLLOUTS = "snapshot_rollouts"
    CLOSED_LOOP_EVALUATION = "closed_loop_evaluation"


class CollectionStatus(str, enum.Enum):
    """Whether a trial yielded scientific labels or only an audit receipt."""

    COMPLETED = "completed"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


def _text(value: Any, name: str) -> str:
    result = " ".join(str(value or "").split())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _sha256(value: Any, name: str) -> str:
    result = str(value)
    if not _SHA256.fullmatch(result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _record_kind(value: Any) -> RecordKind:
    if isinstance(value, RecordKind):
        return value
    try:
        return RecordKind(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"record_kind must be one of {[item.value for item in RecordKind]}"
        ) from error


def _collection_status(value: Any) -> CollectionStatus:
    if isinstance(value, CollectionStatus):
        return value
    try:
        return CollectionStatus(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "collection_status must be completed or infrastructure_failure"
        ) from error


def _json_copy(value: Any, name: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("utf-8"))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise TypeError(f"{name} must be finite and JSON-compatible") from error


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_plain(child) for child in value]
    return value


def _exact(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    observed = set(value)
    if observed != keys:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(keys - observed)}, "
            f"extra={sorted(observed - keys)}"
        )
    return value


def _candidate_payload(candidate: IntentCandidate) -> dict[str, Any]:
    """Return semantic candidate content, excluding its content hash.

    ``candidate_id`` is valuable provenance but the specification explicitly
    forbids feeding text hashes to the model.  The predictor consumes the
    ordinary token IDs and the native/conditioned route only.
    """

    token_ids = getattr(candidate, "token_ids", None)
    if token_ids is None:
        token_ids = getattr(candidate, "intent_token_ids", None)
    if token_ids is None:
        raise TypeError("IntentCandidate must expose token_ids")
    payload = {
        "text": candidate.text,
        "token_ids": [int(token) for token in token_ids],
        "execution_route": candidate.execution_route,
    }
    assert_public_policy_value(payload, path="psr.candidate")
    return payload


@dataclasses.dataclass(frozen=True)
class PSRBranchRecord:
    """One actually executed open-vocabulary intent and its real result.

    The record never stores per-candidate counterfactual labels.  Same-reset
    comparisons are represented as several records sharing ``group_id`` and
    ``initial_reset_ref``, each with a different chosen candidate.
    """

    schema_version: str
    record_kind: RecordKind
    record_id: str
    episode_id: str
    group_id: str
    split: str
    decision_step: int
    remaining_steps: int
    public_history: PublicHistory
    candidate_set: tuple[IntentCandidate, ...]
    chosen_candidate_id: str
    chosen_intent_ids: tuple[int, ...]
    candidate_generation_version: str
    behavior_selection_rule: str
    behavior_probability: float
    execution_snapshot_id: str
    continuation_id: str
    target_encoder_id: str
    initial_reset_ref: str
    actual_action_chunks: tuple[str, ...]
    actual_step_count: int
    actual_observations: tuple[PublicObservation, ...]
    future_evidence_ref: str | None
    evidence_valid: bool
    final_success: bool | None
    cost_valid: bool
    terminal_reason: str
    seed_schedule: Mapping[str, Any]
    collection_status: CollectionStatus

    def __post_init__(self) -> None:
        if self.schema_version != PSR_BRANCH_SCHEMA:
            raise ValueError(f"schema_version must equal {PSR_BRANCH_SCHEMA!r}")
        object.__setattr__(self, "record_kind", _record_kind(self.record_kind))
        object.__setattr__(
            self, "collection_status", _collection_status(self.collection_status)
        )

        for name in (
            "record_id",
            "episode_id",
            "group_id",
            "chosen_candidate_id",
            "candidate_generation_version",
            "behavior_selection_rule",
            "execution_snapshot_id",
            "continuation_id",
            "target_encoder_id",
            "initial_reset_ref",
            "terminal_reason",
        ):
            value = _text(getattr(self, name), name)
            object.__setattr__(self, name, value)

        if self.split not in PSR_SPLITS:
            raise ValueError(f"split must be one of {sorted(PSR_SPLITS)}")
        for name in ("decision_step", "remaining_steps", "actual_step_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.remaining_steps > 300:
            raise ValueError("remaining_steps cannot exceed the frozen 300-step budget")
        if self.actual_step_count > self.remaining_steps:
            raise ValueError("actual_step_count cannot exceed remaining_steps")
        if self.public_history.remaining_control_steps != self.remaining_steps:
            raise ValueError("record and public history remaining-step budgets differ")
        if self.public_history.current.control_step != self.decision_step:
            raise ValueError(
                "decision_step must equal the current public observation step"
            )

        candidates = tuple(self.candidate_set)
        if not candidates or any(
            not isinstance(item, IntentCandidate) for item in candidates
        ):
            raise ValueError("candidate_set must contain IntentCandidate instances")
        candidate_ids = [item.candidate_id for item in candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate_set contains duplicate stable candidate IDs")
        if self.chosen_candidate_id not in candidate_ids:
            raise ValueError("chosen_candidate_id is absent from candidate_set")
        object.__setattr__(self, "candidate_set", candidates)
        chosen = candidates[candidate_ids.index(self.chosen_candidate_id)]
        chosen_tokens = tuple(self.chosen_intent_ids)
        if any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in chosen_tokens
        ):
            raise ValueError("chosen_intent_ids must contain non-negative integers")
        expected_tokens = tuple(_candidate_payload(chosen)["token_ids"])
        if chosen_tokens != expected_tokens:
            raise ValueError("chosen_intent_ids do not match the executed candidate")
        object.__setattr__(self, "chosen_intent_ids", chosen_tokens)

        if (
            not isinstance(self.behavior_probability, (int, float))
            or isinstance(self.behavior_probability, bool)
            or not math.isfinite(float(self.behavior_probability))
            or not 0.0 < float(self.behavior_probability) <= 1.0
        ):
            raise ValueError("behavior_probability must be finite and in (0, 1]")
        object.__setattr__(
            self, "behavior_probability", float(self.behavior_probability)
        )

        action_refs = tuple(
            _text(value, "actual action chunk reference")
            for value in self.actual_action_chunks
        )
        if len(set(action_refs)) != len(action_refs):
            raise ValueError("actual_action_chunks must be unique references")
        object.__setattr__(self, "actual_action_chunks", action_refs)
        observations = tuple(self.actual_observations)
        if any(not isinstance(item, PublicObservation) for item in observations):
            raise TypeError("actual_observations must contain PublicObservation values")
        observation_keys = [
            (
                item.control_step,
                item.agentview_rgb.image_sha256,
                item.wrist_rgb.image_sha256,
            )
            for item in observations
        ]
        if len(set(observation_keys)) != len(observation_keys):
            raise ValueError("actual_observations contains duplicate real observations")
        if [item.control_step for item in observations] != sorted(
            item.control_step for item in observations
        ):
            raise ValueError("actual_observations must be chronological")
        if any(item.control_step < self.decision_step for item in observations):
            raise ValueError("actual_observations cannot predate the decision point")
        object.__setattr__(self, "actual_observations", observations)

        if not isinstance(self.evidence_valid, bool):
            raise TypeError("evidence_valid must be a bool")
        if self.evidence_valid:
            object.__setattr__(
                self,
                "future_evidence_ref",
                _text(self.future_evidence_ref, "future_evidence_ref"),
            )
        elif self.future_evidence_ref is not None:
            raise ValueError(
                "future_evidence_ref must be null when evidence_valid=false"
            )
        if not isinstance(self.cost_valid, bool):
            raise TypeError("cost_valid must be a bool")
        if self.cost_valid:
            if not isinstance(self.final_success, bool):
                raise TypeError("final_success must be bool when cost_valid=true")
        elif self.final_success is not None:
            raise ValueError("final_success must be null when cost_valid=false")

        seed_schedule = _json_copy(self.seed_schedule, "seed_schedule")
        if not isinstance(seed_schedule, Mapping) or not seed_schedule:
            raise ValueError("seed_schedule must be a non-empty mapping")
        object.__setattr__(self, "seed_schedule", _freeze(seed_schedule))

        if self.collection_status is CollectionStatus.INFRASTRUCTURE_FAILURE:
            if self.cost_valid or self.evidence_valid or self.final_success is not None:
                raise ValueError(
                    "infrastructure failures are audit-only and cannot carry E/C labels"
                )
        elif (
            self.record_kind
            in {
                RecordKind.SNAPSHOT_ROLLOUTS,
                RecordKind.CLOSED_LOOP_EVALUATION,
            }
            and not self.cost_valid
        ):
            raise ValueError(
                "completed snapshot/evaluation records require an observed final outcome"
            )

        # Exercise the public firewall at construction time.  This catches a
        # future change that accidentally adds reset/evaluator metadata to the
        # forward payload.
        assert_public_policy_value(self.model_input(), path="psr.branch.model_input")

    @property
    def chosen_candidate(self) -> IntentCandidate:
        return next(
            candidate
            for candidate in self.candidate_set
            if candidate.candidate_id == self.chosen_candidate_id
        )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())

    def model_input(self) -> dict[str, Any]:
        """Return the exact pre-execution, public forward payload.

        It intentionally omits record/reset/split identities, behavior-policy
        metadata, chosen-candidate identity, real future observations and both
        supervision masks/labels.
        """

        result = {
            "history": self.public_history.to_dict(),
            "candidates": [_candidate_payload(item) for item in self.candidate_set],
        }
        assert_public_policy_value(result, path="psr.branch.model_input")
        return result

    def supervision(self) -> dict[str, Any]:
        """Return labels for the sole executed candidate, never alternatives."""

        if self.collection_status is CollectionStatus.INFRASTRUCTURE_FAILURE:
            return {
                "chosen_candidate_id": self.chosen_candidate_id,
                "evidence_ref": None,
                "evidence_valid": False,
                "failure": None,
                "cost_valid": False,
            }
        return {
            "chosen_candidate_id": self.chosen_candidate_id,
            "evidence_ref": self.future_evidence_ref,
            "evidence_valid": self.evidence_valid,
            "failure": None if not self.cost_valid else (not self.final_success),
            "cost_valid": self.cost_valid,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_kind": self.record_kind.value,
            "record_id": self.record_id,
            "episode_id": self.episode_id,
            "group_id": self.group_id,
            "split": self.split,
            "decision_step": self.decision_step,
            "remaining_steps": self.remaining_steps,
            "public_history": self.public_history.to_dict(),
            "candidate_set": [item.to_dict() for item in self.candidate_set],
            "chosen_candidate_id": self.chosen_candidate_id,
            "chosen_intent_ids": list(self.chosen_intent_ids),
            "candidate_generation_version": self.candidate_generation_version,
            "behavior_selection_rule": self.behavior_selection_rule,
            "behavior_probability": self.behavior_probability,
            "execution_snapshot_id": self.execution_snapshot_id,
            "continuation_id": self.continuation_id,
            "target_encoder_id": self.target_encoder_id,
            "initial_reset_ref": self.initial_reset_ref,
            "actual_action_chunks": list(self.actual_action_chunks),
            "actual_step_count": self.actual_step_count,
            "actual_observations": [
                item.to_dict() for item in self.actual_observations
            ],
            "future_evidence_ref": self.future_evidence_ref,
            "evidence_valid": self.evidence_valid,
            "final_success": self.final_success,
            "cost_valid": self.cost_valid,
            "terminal_reason": self.terminal_reason,
            "seed_schedule": _plain(self.seed_schedule),
            "collection_status": self.collection_status.value,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> PSRBranchRecord:
        item = _exact(
            value,
            {field.name for field in dataclasses.fields(cls)},
            "PSR branch record",
        )
        return cls(
            schema_version=item["schema_version"],
            record_kind=item["record_kind"],
            record_id=item["record_id"],
            episode_id=item["episode_id"],
            group_id=item["group_id"],
            split=item["split"],
            decision_step=item["decision_step"],
            remaining_steps=item["remaining_steps"],
            public_history=PublicHistory.from_mapping(item["public_history"]),
            candidate_set=tuple(
                IntentCandidate.from_mapping(v) for v in item["candidate_set"]
            ),
            chosen_candidate_id=item["chosen_candidate_id"],
            chosen_intent_ids=tuple(item["chosen_intent_ids"]),
            candidate_generation_version=item["candidate_generation_version"],
            behavior_selection_rule=item["behavior_selection_rule"],
            behavior_probability=item["behavior_probability"],
            execution_snapshot_id=item["execution_snapshot_id"],
            continuation_id=item["continuation_id"],
            target_encoder_id=item["target_encoder_id"],
            initial_reset_ref=item["initial_reset_ref"],
            actual_action_chunks=tuple(item["actual_action_chunks"]),
            actual_step_count=item["actual_step_count"],
            actual_observations=tuple(
                PublicObservation.from_mapping(v) for v in item["actual_observations"]
            ),
            future_evidence_ref=item["future_evidence_ref"],
            evidence_valid=item["evidence_valid"],
            final_success=item["final_success"],
            cost_valid=item["cost_valid"],
            terminal_reason=item["terminal_reason"],
            seed_schedule=item["seed_schedule"],
            collection_status=item["collection_status"],
        )


@dataclasses.dataclass(frozen=True)
class DatasetSummary:
    """Audit counts returned only after a complete validation pass."""

    record_kind: RecordKind
    records: int
    completed: int
    infrastructure_failures: int
    evidence_labels: int
    cost_labels: int
    groups: int
    episodes: int
    splits: Mapping[str, int]
    identity_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_kind": self.record_kind.value,
            "records": self.records,
            "completed": self.completed,
            "infrastructure_failures": self.infrastructure_failures,
            "evidence_labels": self.evidence_labels,
            "cost_labels": self.cost_labels,
            "groups": self.groups,
            "episodes": self.episodes,
            "splits": dict(self.splits),
            "identity_sha256": self.identity_sha256,
        }


def validate_records(
    records: Sequence[PSRBranchRecord],
    *,
    expected_kind: RecordKind | str | None = None,
    expected_snapshot_id: str | None = None,
    expected_continuation_id: str | None = None,
    expected_target_encoder_id: str | None = None,
) -> DatasetSummary:
    """Validate nonmixing, grouping, uniqueness, and execution identities."""

    rows = tuple(records)
    if not rows:
        raise ValueError("records must be non-empty")
    if any(not isinstance(row, PSRBranchRecord) for row in rows):
        raise TypeError("records must contain PSRBranchRecord values")

    kinds = {row.record_kind for row in rows}
    if len(kinds) != 1:
        raise ValueError("record kinds cannot be mixed in one dataset")
    kind = next(iter(kinds))
    if expected_kind is not None and kind is not _record_kind(expected_kind):
        raise ValueError(
            "dataset record kind does not match the requested training stage"
        )

    for attribute, expected in (
        ("execution_snapshot_id", expected_snapshot_id),
        ("continuation_id", expected_continuation_id),
        ("target_encoder_id", expected_target_encoder_id),
    ):
        if expected is not None:
            mismatched = [
                row.record_id for row in rows if getattr(row, attribute) != expected
            ]
            if mismatched:
                raise ValueError(f"{attribute} mismatch for records {mismatched[:3]}")

    record_ids = [row.record_id for row in rows]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("duplicate record_id")
    fingerprints = [row.fingerprint for row in rows]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("duplicate branch record")
    chosen_branches = [
        (
            row.group_id,
            row.chosen_candidate_id,
            canonical_sha256(_plain(row.seed_schedule)),
        )
        for row in rows
    ]
    if len(set(chosen_branches)) != len(chosen_branches):
        raise ValueError("duplicate candidate execution under the same seed schedule")

    # Every reset or correlated episode family is assigned to exactly one
    # split.  Group IDs are decision points; reset and episode checks protect
    # hidden-world pairs and temporal slices as well.
    for attribute in ("episode_id", "group_id", "initial_reset_ref"):
        split_by_key: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            split_by_key[getattr(row, attribute)].add(row.split)
        crossed = [key for key, splits in split_by_key.items() if len(splits) > 1]
        if crossed:
            raise ValueError(f"{attribute} crosses data splits: {crossed[:3]}")

    groups: dict[str, list[PSRBranchRecord]] = defaultdict(list)
    for row in rows:
        groups[row.group_id].append(row)
    for group_id, group in groups.items():
        reference = group[0]
        expected_candidates = {item.candidate_id for item in reference.candidate_set}
        for row in group[1:]:
            if row.public_history.fingerprint != reference.public_history.fingerprint:
                raise ValueError(
                    f"decision group {group_id} has different public histories"
                )
            if {item.candidate_id for item in row.candidate_set} != expected_candidates:
                raise ValueError(
                    f"decision group {group_id} has different frozen candidates"
                )
            for attribute in (
                "candidate_generation_version",
                "execution_snapshot_id",
                "continuation_id",
                "target_encoder_id",
                "initial_reset_ref",
            ):
                if getattr(row, attribute) != getattr(reference, attribute):
                    raise ValueError(f"decision group {group_id} mixes {attribute}")

    split_counts = Counter(row.split for row in rows)
    identity = {
        "schema": "psr-v1-dataset-identity-v1",
        "record_kind": kind.value,
        "records": sorted((row.record_id, row.fingerprint) for row in rows),
    }
    return DatasetSummary(
        record_kind=kind,
        records=len(rows),
        completed=sum(
            row.collection_status is CollectionStatus.COMPLETED for row in rows
        ),
        infrastructure_failures=sum(
            row.collection_status is CollectionStatus.INFRASTRUCTURE_FAILURE
            for row in rows
        ),
        evidence_labels=sum(row.evidence_valid for row in rows),
        cost_labels=sum(row.cost_valid for row in rows),
        groups=len(groups),
        episodes=len({row.episode_id for row in rows}),
        splits=MappingProxyType(dict(sorted(split_counts.items()))),
        identity_sha256=canonical_sha256(identity),
    )


def collate_public_and_labels(
    records: Sequence[PSRBranchRecord],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep forward inputs physically separate from E/C label dictionaries."""

    return (
        [record.model_input() for record in records],
        [record.supervision() for record in records],
    )


def load_jsonl(
    path: str,
    *,
    expected_kind: RecordKind | str | None = None,
    expected_snapshot_id: str | None = None,
    expected_continuation_id: str | None = None,
    expected_target_encoder_id: str | None = None,
) -> tuple[PSRBranchRecord, ...]:
    """Read strict JSONL records and validate them before returning any data."""

    with open(path, "r", encoding="utf-8") as handle:
        records = tuple(
            PSRBranchRecord.from_mapping(json.loads(line))
            for line in handle
            if line.strip()
        )
    validate_records(
        records,
        expected_kind=expected_kind,
        expected_snapshot_id=expected_snapshot_id,
        expected_continuation_id=expected_continuation_id,
        expected_target_encoder_id=expected_target_encoder_id,
    )
    return records
