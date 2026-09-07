"""Observed, reset-controlled branch data for grounded interventions.

One :class:`ObservedBranch` contains exactly one intervention that was actually
executed.  It never assigns outcomes to alternatives that were not run.  A
counterfactual comparison is assembled only by grouping separate, observed
branches that share an exact reset state and public pre-action context.

The dataset validator enforces two levels of grouping:

``initial_state_group``
    The opaque reset state.  All prompt variants and action forks from this
    physical state must remain in one data split.

``decision_group_id``
    One exact public context (including the prompt) from which every candidate
    is executed under the same frozen outcome contract.

Private evaluator metadata is retained for audit and labels, but it is omitted
from ``model_input`` by construction.  This module uses only the Python
standard library and has no dependency on historical pipeline packages.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from .contracts import (
    ExecutionStatus,
    GroundedIntervention,
    OutcomeContract,
    PolicyContext,
    PublicFrame,
    assert_public_policy_value,
    canonical_json_bytes,
    canonical_sha256,
)


def _clean_text(value: Any, *, name: str) -> str:
    result = " ".join(str(value or "").split())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _plain_json_copy(value: Any, *, name: str) -> Any:
    """Validate and detach an arbitrary evaluator-side JSON value."""

    try:
        encoded = canonical_json_bytes(value)
        return json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise TypeError(f"{name} must be finite and JSON-compatible") from error


def _freeze_private_value(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_private_value(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_private_value(child) for child in value)
    return value


def _plain_private_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_private_value(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_plain_private_value(child) for child in value]
    return value


@dataclasses.dataclass(frozen=True)
class ObservedBranch:
    """One real execution from an exact reset-controlled decision point.

    ``observed_outcome`` has the meaning frozen by ``outcome_contract`` and
    applies only to ``executed_intervention``.  Alternatives appear as other
    :class:`ObservedBranch` objects with their own execution receipts.

    ``reset_state_sha256`` is transport/audit metadata for verifying the reset.
    It is intentionally absent from :meth:`model_input`.

    A ``STOP`` branch is observed by closing the episode, evaluating the same
    bounded outcome, and recording a local termination receipt plus a fresh
    public observation.  It never calls the VLA and is never assigned a
    default negative label.
    """

    branch_id: str
    initial_state_group: str
    decision_group_id: str
    split: str
    reset_state_sha256: str
    repeat_index: int
    context: PolicyContext
    executed_intervention: GroundedIntervention
    post_action_frames: tuple[PublicFrame, ...]
    outcome_contract: OutcomeContract
    observed_outcome: bool
    execution_status: ExecutionStatus
    execution_receipt_id: str
    diagnostics: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    private_evaluator_metadata: Mapping[str, Any] = dataclasses.field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "branch_id",
            "initial_state_group",
            "decision_group_id",
            "split",
            "execution_receipt_id",
        ):
            object.__setattr__(self, name, _clean_text(getattr(self, name), name=name))

        digest = str(self.reset_state_sha256)
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("reset_state_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "reset_state_sha256", digest)

        if (
            not isinstance(self.repeat_index, int)
            or isinstance(self.repeat_index, bool)
            or self.repeat_index < 0
        ):
            raise ValueError("repeat_index must be a non-negative integer")
        if not isinstance(self.context, PolicyContext):
            raise TypeError("context must be a PolicyContext")
        if not isinstance(self.executed_intervention, GroundedIntervention):
            raise TypeError("executed_intervention must be a GroundedIntervention")
        self.executed_intervention.validate_against(self.context)
        if not isinstance(self.outcome_contract, OutcomeContract):
            raise TypeError("outcome_contract must be an OutcomeContract")
        if not isinstance(self.observed_outcome, bool):
            raise TypeError("observed_outcome must be a bool")
        if not isinstance(self.execution_status, ExecutionStatus):
            raise TypeError("execution_status must be an ExecutionStatus")

        post_frames = tuple(self.post_action_frames)
        if not post_frames or any(
            not isinstance(frame, PublicFrame) for frame in post_frames
        ):
            raise ValueError("post_action_frames must contain at least one PublicFrame")
        post_ids = [frame.frame_id for frame in post_frames]
        post_keys = [(frame.camera, frame.frame_index) for frame in post_frames]
        if len(set(post_ids)) != len(post_ids):
            raise ValueError("post-action frame IDs must be unique")
        if len(set(post_keys)) != len(post_keys):
            raise ValueError("post-action camera/frame_index pairs must be unique")
        if set(post_ids) & {frame.frame_id for frame in self.context.frames}:
            raise ValueError(
                "pre- and post-action observations require distinct frame IDs"
            )
        object.__setattr__(self, "post_action_frames", post_frames)

        diagnostics = _freeze_private_value(
            _plain_json_copy(self.diagnostics, name="diagnostics")
        )
        evaluator = _freeze_private_value(
            _plain_json_copy(
                self.private_evaluator_metadata,
                name="private_evaluator_metadata",
            )
        )
        if not isinstance(diagnostics, Mapping) or not isinstance(evaluator, Mapping):
            raise TypeError(
                "diagnostics and private evaluator metadata must be mappings"
            )
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "private_evaluator_metadata", evaluator)

        # This is the complete payload consumed before the intervention.  The
        # assertion ensures that no private branch field can leak through a
        # future refactor of model_input().
        assert_public_policy_value(
            self.model_input(), path="observed_branch.model_input"
        )

    @property
    def candidate_id(self) -> str:
        return self.executed_intervention.candidate_id

    @property
    def candidate_fingerprint(self) -> str:
        return self.executed_intervention.fingerprint()

    def model_input(self) -> dict[str, Any]:
        """Return only information available before executing this candidate."""

        value = {
            "context": self.context.to_dict(),
            "candidate": self.executed_intervention.policy_payload(),
        }
        assert_public_policy_value(value, path="observed_branch.model_input")
        return value

    def supervision(self) -> dict[str, Any]:
        """Return the observed label for this execution and no alternatives."""

        return {
            "executed_candidate_id": self.candidate_id,
            "executed_candidate_fingerprint": self.candidate_fingerprint,
            "outcome_contract": self.outcome_contract.to_dict(),
            "observed_outcome": self.observed_outcome,
            "post_action_frames": [
                frame.to_dict() for frame in self.post_action_frames
            ],
            "execution_status": self.execution_status.value,
            "execution_receipt_id": self.execution_receipt_id,
            "diagnostics": _plain_private_value(self.diagnostics),
        }

    def to_dict(self, *, include_private: bool = True) -> dict[str, Any]:
        value = {
            "branch_id": self.branch_id,
            "initial_state_group": self.initial_state_group,
            "decision_group_id": self.decision_group_id,
            "split": self.split,
            "reset_state_sha256": self.reset_state_sha256,
            "repeat_index": self.repeat_index,
            "model_input": self.model_input(),
            "supervision": self.supervision(),
        }
        if include_private:
            value["private_evaluator_metadata"] = _plain_private_value(
                self.private_evaluator_metadata
            )
        return value

    def fingerprint(self) -> str:
        """Hash the complete branch, including its private evaluator sidecar."""

        return canonical_sha256(self.to_dict(include_private=True))

    def public_fingerprint(self) -> str:
        """Hash only public model input plus the observed training target."""

        return canonical_sha256(self.to_dict(include_private=False))


def validate_group_splits(branches: Sequence[ObservedBranch]) -> None:
    """Keep every branch and prompt variant of one reset state in one split."""

    state_to_split: dict[str, str] = {}
    state_to_digest: dict[str, str] = {}
    for branch in branches:
        if not isinstance(branch, ObservedBranch):
            raise TypeError("branch collections may contain only ObservedBranch")
        previous_split = state_to_split.setdefault(
            branch.initial_state_group, branch.split
        )
        if previous_split != branch.split:
            raise ValueError(
                f"initial state {branch.initial_state_group!r} crosses data splits"
            )
        previous_digest = state_to_digest.setdefault(
            branch.initial_state_group, branch.reset_state_sha256
        )
        if previous_digest != branch.reset_state_sha256:
            raise ValueError(
                f"initial state {branch.initial_state_group!r} changes reset digest"
            )


def validate_reset_controlled_branches(
    branches: Sequence[ObservedBranch],
    *,
    repetitions_per_candidate: int,
) -> None:
    """Validate a complete matrix of actually executed candidate forks.

    Each exact decision group must contain at least two distinct physical
    choices, and every candidate must have receipt-backed repetitions indexed
    exactly ``0 .. repetitions_per_candidate - 1``.  This validates collection
    structure, not candidate quality or empirical performance.
    """

    records = tuple(branches)
    if not records:
        raise ValueError("a branch dataset must be non-empty")
    if (
        not isinstance(repetitions_per_candidate, int)
        or isinstance(repetitions_per_candidate, bool)
        or repetitions_per_candidate < 1
    ):
        raise ValueError("repetitions_per_candidate must be a positive integer")
    if any(not isinstance(branch, ObservedBranch) for branch in records):
        raise TypeError("branch datasets may contain only ObservedBranch")

    branch_ids = [branch.branch_id for branch in records]
    receipt_ids = [branch.execution_receipt_id for branch in records]
    if len(set(branch_ids)) != len(branch_ids):
        raise ValueError("branch_id values must be globally unique")
    if len(set(receipt_ids)) != len(receipt_ids):
        raise ValueError("execution_receipt_id values must be globally unique")

    validate_group_splits(records)
    contract_fingerprints = {
        branch.outcome_contract.fingerprint() for branch in records
    }
    if len(contract_fingerprints) != 1:
        raise ValueError("one branch dataset must use one frozen outcome contract")

    by_decision_group: dict[str, list[ObservedBranch]] = defaultdict(list)
    for branch in records:
        by_decision_group[branch.decision_group_id].append(branch)

    expected_repeats = set(range(repetitions_per_candidate))
    for group_id, rows in by_decision_group.items():
        state_ids = {row.initial_state_group for row in rows}
        splits = {row.split for row in rows}
        reset_digests = {row.reset_state_sha256 for row in rows}
        context_fingerprints = {row.context.fingerprint() for row in rows}
        group_contracts = {row.outcome_contract.fingerprint() for row in rows}
        if not (
            len(state_ids)
            == len(splits)
            == len(reset_digests)
            == len(context_fingerprints)
            == len(group_contracts)
            == 1
        ):
            raise ValueError(
                f"decision group {group_id!r} does not share one reset/context/contract"
            )

        candidate_fingerprints: dict[str, str] = {}
        repeats: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            previous = candidate_fingerprints.setdefault(
                row.candidate_id, row.candidate_fingerprint
            )
            if previous != row.candidate_fingerprint:
                raise ValueError(
                    f"candidate {row.candidate_id!r} changes within decision group {group_id!r}"
                )
            if row.repeat_index in repeats[row.candidate_id]:
                raise ValueError(
                    f"duplicate repeat for candidate {row.candidate_id!r} in {group_id!r}"
                )
            repeats[row.candidate_id].add(row.repeat_index)

        if len(candidate_fingerprints) < 2:
            raise ValueError(
                f"decision group {group_id!r} needs at least two executed candidates"
            )
        for candidate_id, observed_repeats in repeats.items():
            if observed_repeats != expected_repeats:
                raise ValueError(
                    f"candidate {candidate_id!r} in {group_id!r} has incomplete repetitions"
                )


@dataclasses.dataclass(frozen=True)
class ResetControlledBranchDataset:
    """A fingerprinted collection of complete, receipt-backed branch matrices."""

    dataset_id: str
    repetitions_per_candidate: int
    branches: tuple[ObservedBranch, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dataset_id", _clean_text(self.dataset_id, name="dataset_id")
        )
        records = tuple(self.branches)
        object.__setattr__(self, "branches", records)
        validate_reset_controlled_branches(
            records,
            repetitions_per_candidate=self.repetitions_per_candidate,
        )

    @property
    def outcome_contract(self) -> OutcomeContract:
        return self.branches[0].outcome_contract

    def summary(self) -> dict[str, Any]:
        candidates_by_group: dict[str, set[str]] = defaultdict(set)
        for branch in self.branches:
            candidates_by_group[branch.decision_group_id].add(branch.candidate_id)
        return {
            "dataset_id": self.dataset_id,
            "observed_branches": len(self.branches),
            "initial_state_groups": len(
                {branch.initial_state_group for branch in self.branches}
            ),
            "decision_groups": len(candidates_by_group),
            "candidate_executions": sum(
                len(candidates) for candidates in candidates_by_group.values()
            ),
            "repetitions_per_candidate": self.repetitions_per_candidate,
            "splits": dict(
                sorted(Counter(branch.split for branch in self.branches).items())
            ),
            "outcome_contract_fingerprint": self.outcome_contract.fingerprint(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "repetitions_per_candidate": self.repetitions_per_candidate,
            "outcome_contract": self.outcome_contract.to_dict(),
            "branches": [
                branch.to_dict(include_private=True)
                for branch in sorted(self.branches, key=lambda item: item.branch_id)
            ],
        }

    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())
