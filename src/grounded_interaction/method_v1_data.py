"""Outcome-bearing data contracts for Method V1.

Method V1 learns one quantity: final task success after executing one grounded
candidate and then following a frozen, finite-horizon continuation policy.  A
decision manifest freezes the *public* decision point and complete candidate
set once.  A schedule then forks the exact reset state for every
candidate/seed pair.  Finally, a collection attempt is either an
outcome-evaluated :class:`~grounded_interaction.data.ObservedBranch` or an
unlabelled infrastructure failure -- never both.

This module deliberately composes the existing ``PolicyContext``,
``GroundedIntervention``, ``OutcomeContract`` and ``ObservedBranch`` classes.
It does not define a second candidate or label schema.  Simulator state hashes,
split identities, and evaluator artifacts are audit data and are absent from
``DecisionGroupManifest.model_input`` by construction.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from enum import Enum
from types import MappingProxyType
from typing import Any

from .contracts import (
    GroundedIntervention,
    OutcomeContract,
    PolicyContext,
    Primitive,
    assert_public_policy_value,
    canonical_json_bytes,
    canonical_sha256,
)
from .data import ObservedBranch, validate_group_splits

METHOD_V1_MANIFEST_SCHEMA = "method-v1-decision-group-manifest-v2"
METHOD_V1_SCHEDULE_SCHEMA = "method-v1-branch-schedule-entry-v1"
METHOD_V1_ATTEMPT_SCHEMA = "method-v1-collection-attempt-v1"
METHOD_V1_OUTCOME_NAME = "full_task_with_fixed_continuation_v1"
METHOD_V1_HORIZON = 300
METHOD_V1_REPETITIONS = 2
METHOD_V1_MAX_CANDIDATES = 6
METHOD_V1_MAX_PER_PRIMITIVE = 3
METHOD_V1_FAILURE_HANDLING = (
    "policy_failures_evaluated_infrastructure_failures_unlabelled_v1"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_SPLITS = frozenset({"train", "validation", "calibration", "test"})
_ALLOWED_PRIMITIVES = frozenset({Primitive.DIRECT, Primitive.OPEN})


class InformationStratum(str, Enum):
    """Evaluator-only information condition assigned before outcomes exist.

    The stratum is a collection-design label, not a policy observation.  It
    records whether the frozen scene was constructed so that information
    acquisition is necessary, already sufficient, or physically possible but
    unhelpful for the task.  It must never be inserted into
    :meth:`DecisionGroupManifest.model_input`.
    """

    INFORMATION_NECESSARY = "INFORMATION_NECESSARY"
    INFORMATION_SUFFICIENT = "INFORMATION_SUFFICIENT"
    INFORMATION_ACTION_NO_HELP = "INFORMATION_ACTION_NO_HELP"


INFORMATION_STRATA: tuple[InformationStratum, ...] = tuple(InformationStratum)
INFORMATION_STRATUM_VALUES: tuple[str, ...] = tuple(
    item.value for item in INFORMATION_STRATA
)


def _information_stratum(value: object) -> InformationStratum:
    if isinstance(value, InformationStratum):
        return value
    if not isinstance(value, str):
        raise TypeError("information_stratum must be an InformationStratum")
    try:
        return InformationStratum(value)
    except ValueError as error:
        raise ValueError(
            f"information_stratum must be one of {list(INFORMATION_STRATUM_VALUES)}"
        ) from error


def validate_information_stratum_counts(
    value: Any,
    *,
    reset_groups: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    """Validate the pre-outcome split-by-information-condition allocation.

    Every split and every stratum must be named explicitly, including the
    all-zero calibration split.  For each split the stratum counts must sum to
    the already frozen reset-group count.  The returned plain mapping follows
    a deterministic split/stratum order and is safe to hash into identities.
    """

    if not isinstance(value, Mapping):
        raise TypeError("information_stratum_counts must be a mapping")
    if not isinstance(reset_groups, Mapping):
        raise TypeError("reset_groups must be a mapping")
    if set(value) != _ALLOWED_SPLITS:
        raise ValueError(
            f"information_stratum_counts must name exactly {sorted(_ALLOWED_SPLITS)}"
        )
    if set(reset_groups) != _ALLOWED_SPLITS:
        raise ValueError(f"reset_groups must name exactly {sorted(_ALLOWED_SPLITS)}")

    result: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "calibration", "test"):
        total = reset_groups[split]
        if not isinstance(total, int) or isinstance(total, bool) or total < 0:
            raise ValueError("reset-group counts must be non-negative integers")
        raw_counts = value[split]
        if not isinstance(raw_counts, Mapping):
            raise TypeError(f"information stratum counts for {split} must be a mapping")
        if set(raw_counts) != set(INFORMATION_STRATUM_VALUES):
            raise ValueError(
                f"information stratum counts for {split} must name exactly "
                f"{list(INFORMATION_STRATUM_VALUES)}"
            )
        counts: dict[str, int] = {}
        for stratum in INFORMATION_STRATUM_VALUES:
            count = raw_counts[stratum]
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(
                    "information stratum counts must be non-negative integers"
                )
            counts[stratum] = count
        if sum(counts.values()) != total:
            raise ValueError(
                f"information stratum counts for {split} sum to "
                f"{sum(counts.values())}, not reset_groups[{split!r}]={total}"
            )
        if total > 0 and any(count == 0 for count in counts.values()):
            raise ValueError(
                f"non-empty split {split} must include every information stratum"
            )
        result[split] = counts
    return result


def _clean_text(value: object, *, name: str) -> str:
    result = " ".join(str(value or "").split())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _require_sha256(value: object, *, name: str) -> str:
    result = str(value)
    if not _SHA256_RE.fullmatch(result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _plain_json_copy(value: Any, *, name: str) -> Any:
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


def _exact_mapping(value: Any, *, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    observed = set(value)
    if observed != keys:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(keys - observed)}, "
            f"extra={sorted(observed - keys)}"
        )
    return value


def validate_method_v1_outcome_contract(contract: OutcomeContract) -> None:
    """Require the new full-task label and reject the E1 contact label."""

    if not isinstance(contract, OutcomeContract):
        raise TypeError("contract must be an OutcomeContract")
    if contract.outcome_name != METHOD_V1_OUTCOME_NAME:
        raise ValueError(
            "Method V1 requires the full_task_with_fixed_continuation_v1 "
            "outcome contract"
        )
    if contract.horizon != METHOD_V1_HORIZON:
        raise ValueError("Method V1 outcome horizon must be exactly 300 steps")
    if contract.failure_handling != METHOD_V1_FAILURE_HANDLING:
        raise ValueError("Method V1 failure handling is not the frozen policy")


def make_method_v1_outcome_contract(
    *,
    continuation_policy_id: str,
    executor_id: str,
    serializer_id: str,
) -> OutcomeContract:
    """Build the sole outcome contract accepted by Method V1 datasets."""

    return OutcomeContract(
        outcome_name=METHOD_V1_OUTCOME_NAME,
        continuation_policy_id=continuation_policy_id,
        horizon=METHOD_V1_HORIZON,
        executor_id=executor_id,
        serializer_id=serializer_id,
        failure_handling=METHOD_V1_FAILURE_HANDLING,
    )


@dataclasses.dataclass(frozen=True)
class ProposalRejection:
    """Auditable reason for dropping one public VLM proposal.

    The rejected content is bound by a digest rather than copied into policy
    input.  Rejections are diagnostics and never become training labels.
    """

    proposal_index: int
    reason_code: str
    proposal_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.proposal_index, int)
            or isinstance(self.proposal_index, bool)
            or self.proposal_index < 0
        ):
            raise ValueError("proposal_index must be a non-negative integer")
        object.__setattr__(
            self, "reason_code", _clean_text(self.reason_code, name="reason_code")
        )
        object.__setattr__(
            self,
            "proposal_sha256",
            _require_sha256(self.proposal_sha256, name="proposal_sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_mapping(cls, value: Any) -> ProposalRejection:
        mapping = _exact_mapping(
            value,
            keys={"proposal_index", "reason_code", "proposal_sha256"},
            name="proposal rejection",
        )
        return cls(
            proposal_index=int(mapping["proposal_index"]),
            reason_code=str(mapping["reason_code"]),
            proposal_sha256=str(mapping["proposal_sha256"]),
        )


@dataclasses.dataclass(frozen=True)
class DecisionGroupManifest:
    """One frozen public decision point and its complete candidate set."""

    manifest_id: str
    experiment_id: str
    initial_state_group: str
    decision_group_id: str
    split_group_id: str
    split: str
    information_stratum: InformationStratum
    scene_id: str
    layout_id: str
    reset_state_sha256: str
    context: PolicyContext
    candidates: tuple[GroundedIntervention, ...]
    outcome_contract: OutcomeContract
    proposal_provider_id: str
    proposal_request_sha256: str
    token_provider_id: str
    token_cache_sha256: str
    configuration_sha256: str
    model_seeds: tuple[int, ...]
    repetitions_per_candidate: int = METHOD_V1_REPETITIONS
    proposal_rejections: tuple[ProposalRejection, ...] = ()
    schema_version: str = METHOD_V1_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "manifest_id",
            "experiment_id",
            "initial_state_group",
            "decision_group_id",
            "split_group_id",
            "scene_id",
            "layout_id",
            "proposal_provider_id",
            "token_provider_id",
        ):
            object.__setattr__(self, name, _clean_text(getattr(self, name), name=name))
        if self.schema_version != METHOD_V1_MANIFEST_SCHEMA:
            raise ValueError("unsupported Method V1 manifest schema")
        if self.split not in _ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {sorted(_ALLOWED_SPLITS)}")
        object.__setattr__(
            self,
            "information_stratum",
            _information_stratum(self.information_stratum),
        )
        for name in (
            "reset_state_sha256",
            "proposal_request_sha256",
            "token_cache_sha256",
            "configuration_sha256",
        ):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), name=name)
            )
        if not isinstance(self.context, PolicyContext):
            raise TypeError("context must be a PolicyContext")
        validate_method_v1_outcome_contract(self.outcome_contract)
        if (
            not isinstance(self.repetitions_per_candidate, int)
            or isinstance(self.repetitions_per_candidate, bool)
            or self.repetitions_per_candidate != METHOD_V1_REPETITIONS
        ):
            raise ValueError("Method V1 requires exactly two seeds per candidate")

        seeds = tuple(self.model_seeds)
        if len(seeds) != self.repetitions_per_candidate:
            raise ValueError("model_seeds must match repetitions_per_candidate")
        if any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
            for seed in seeds
        ):
            raise ValueError("model seeds must be non-negative integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("model seeds must be unique")
        object.__setattr__(self, "model_seeds", seeds)

        candidates = tuple(self.candidates)
        if not candidates or len(candidates) > METHOD_V1_MAX_CANDIDATES:
            raise ValueError("Method V1 requires between one and six candidates")
        if any(not isinstance(item, GroundedIntervention) for item in candidates):
            raise TypeError("candidates must contain GroundedIntervention values")
        for candidate in candidates:
            if candidate.primitive not in _ALLOWED_PRIMITIVES:
                raise ValueError("Method V1 initially enables DIRECT and OPEN only")
            candidate.validate_against(self.context)
        ids = [candidate.candidate_id for candidate in candidates]
        fingerprints = [candidate.fingerprint() for candidate in candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate_id values must be unique in one decision group")
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("duplicate grounded candidates are forbidden")
        counts = Counter(candidate.primitive for candidate in candidates)
        if any(count > METHOD_V1_MAX_PER_PRIMITIVE for count in counts.values()):
            raise ValueError("Method V1 permits at most three candidates per primitive")
        object.__setattr__(self, "candidates", candidates)

        rejections = tuple(self.proposal_rejections)
        if any(not isinstance(item, ProposalRejection) for item in rejections):
            raise TypeError("proposal_rejections must contain ProposalRejection values")
        rejection_indices = [item.proposal_index for item in rejections]
        if len(set(rejection_indices)) != len(rejection_indices):
            raise ValueError("proposal rejection indices must be unique")
        object.__setattr__(self, "proposal_rejections", rejections)

        # This is the complete scorer-visible payload.  Audit identities and
        # result fields are intentionally absent.
        assert_public_policy_value(self.model_input(), path="method_v1.model_input")

    @property
    def candidate_set_sha256(self) -> str:
        return canonical_sha256(
            [candidate.policy_payload() for candidate in self.candidates]
        )

    def model_input(self) -> dict[str, Any]:
        value = {
            "context": self.context.to_dict(),
            "candidates": [candidate.policy_payload() for candidate in self.candidates],
        }
        assert_public_policy_value(value, path="method_v1.model_input")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "experiment_id": self.experiment_id,
            "initial_state_group": self.initial_state_group,
            "decision_group_id": self.decision_group_id,
            "split_group_id": self.split_group_id,
            "split": self.split,
            "information_stratum": self.information_stratum.value,
            "scene_id": self.scene_id,
            "layout_id": self.layout_id,
            "reset_state_sha256": self.reset_state_sha256,
            "model_input": self.model_input(),
            "outcome_contract": self.outcome_contract.to_dict(),
            "proposal_provider_id": self.proposal_provider_id,
            "proposal_request_sha256": self.proposal_request_sha256,
            "token_provider_id": self.token_provider_id,
            "token_cache_sha256": self.token_cache_sha256,
            "configuration_sha256": self.configuration_sha256,
            "model_seeds": list(self.model_seeds),
            "repetitions_per_candidate": self.repetitions_per_candidate,
            "proposal_rejections": [
                item.to_dict() for item in self.proposal_rejections
            ],
            "candidate_set_sha256": self.candidate_set_sha256,
            "manifest_sha256": self.fingerprint(),
        }

    def _payload(self) -> dict[str, Any]:
        """Canonical payload excluding its derived digest."""

        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "experiment_id": self.experiment_id,
            "initial_state_group": self.initial_state_group,
            "decision_group_id": self.decision_group_id,
            "split_group_id": self.split_group_id,
            "split": self.split,
            "information_stratum": self.information_stratum.value,
            "scene_id": self.scene_id,
            "layout_id": self.layout_id,
            "reset_state_sha256": self.reset_state_sha256,
            "model_input": self.model_input(),
            "outcome_contract": self.outcome_contract.to_dict(),
            "proposal_provider_id": self.proposal_provider_id,
            "proposal_request_sha256": self.proposal_request_sha256,
            "token_provider_id": self.token_provider_id,
            "token_cache_sha256": self.token_cache_sha256,
            "configuration_sha256": self.configuration_sha256,
            "model_seeds": list(self.model_seeds),
            "repetitions_per_candidate": self.repetitions_per_candidate,
            "proposal_rejections": [
                item.to_dict() for item in self.proposal_rejections
            ],
            "candidate_set_sha256": self.candidate_set_sha256,
        }

    def fingerprint(self) -> str:
        return canonical_sha256(self._payload())

    def candidate(self, candidate_id: str) -> GroundedIntervention:
        matches = [
            item for item in self.candidates if item.candidate_id == candidate_id
        ]
        if len(matches) != 1:
            raise ValueError(f"unknown or ambiguous candidate_id {candidate_id!r}")
        return matches[0]

    @classmethod
    def from_mapping(cls, value: Any) -> DecisionGroupManifest:
        mapping = _exact_mapping(
            value,
            keys={
                "schema_version",
                "manifest_id",
                "experiment_id",
                "initial_state_group",
                "decision_group_id",
                "split_group_id",
                "split",
                "information_stratum",
                "scene_id",
                "layout_id",
                "reset_state_sha256",
                "model_input",
                "outcome_contract",
                "proposal_provider_id",
                "proposal_request_sha256",
                "token_provider_id",
                "token_cache_sha256",
                "configuration_sha256",
                "model_seeds",
                "repetitions_per_candidate",
                "proposal_rejections",
                "candidate_set_sha256",
                "manifest_sha256",
            },
            name="Method V1 decision manifest",
        )
        model_input = _exact_mapping(
            mapping["model_input"],
            keys={"context", "candidates"},
            name="Method V1 manifest model input",
        )
        result = cls(
            schema_version=str(mapping["schema_version"]),
            manifest_id=str(mapping["manifest_id"]),
            experiment_id=str(mapping["experiment_id"]),
            initial_state_group=str(mapping["initial_state_group"]),
            decision_group_id=str(mapping["decision_group_id"]),
            split_group_id=str(mapping["split_group_id"]),
            split=str(mapping["split"]),
            information_stratum=_information_stratum(mapping["information_stratum"]),
            scene_id=str(mapping["scene_id"]),
            layout_id=str(mapping["layout_id"]),
            reset_state_sha256=str(mapping["reset_state_sha256"]),
            context=PolicyContext.from_mapping(model_input["context"]),
            candidates=tuple(
                GroundedIntervention.from_mapping(item)
                for item in model_input["candidates"]
            ),
            outcome_contract=OutcomeContract.from_mapping(mapping["outcome_contract"]),
            proposal_provider_id=str(mapping["proposal_provider_id"]),
            proposal_request_sha256=str(mapping["proposal_request_sha256"]),
            token_provider_id=str(mapping["token_provider_id"]),
            token_cache_sha256=str(mapping["token_cache_sha256"]),
            configuration_sha256=str(mapping["configuration_sha256"]),
            model_seeds=tuple(int(item) for item in mapping["model_seeds"]),
            repetitions_per_candidate=int(mapping["repetitions_per_candidate"]),
            proposal_rejections=tuple(
                ProposalRejection.from_mapping(item)
                for item in mapping["proposal_rejections"]
            ),
        )
        if mapping["candidate_set_sha256"] != result.candidate_set_sha256:
            raise ValueError("candidate set digest is inconsistent")
        if mapping["manifest_sha256"] != result.fingerprint():
            raise ValueError("manifest digest is inconsistent")
        return result


@dataclasses.dataclass(frozen=True)
class BranchScheduleEntry:
    """One single-use execution of one candidate from one exact reset."""

    schedule_id: str
    execution_index: int
    manifest_id: str
    manifest_sha256: str
    decision_group_id: str
    initial_state_group: str
    split_group_id: str
    split: str
    reset_state_sha256: str
    candidate_id: str
    candidate_fingerprint: str
    repeat_index: int
    model_seed: int
    outcome_contract_sha256: str
    schema_version: str = METHOD_V1_SCHEDULE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "schedule_id",
            "manifest_id",
            "decision_group_id",
            "initial_state_group",
            "split_group_id",
            "candidate_id",
        ):
            object.__setattr__(self, name, _clean_text(getattr(self, name), name=name))
        if self.schema_version != METHOD_V1_SCHEDULE_SCHEMA:
            raise ValueError("unsupported Method V1 schedule schema")
        if self.split not in _ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {sorted(_ALLOWED_SPLITS)}")
        for name in (
            "manifest_sha256",
            "reset_state_sha256",
            "candidate_fingerprint",
            "outcome_contract_sha256",
        ):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), name=name)
            )
        for name in ("execution_index", "repeat_index", "model_seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def _payload(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def entry_id(self) -> str:
        return canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "entry_id": self.entry_id}

    @classmethod
    def from_mapping(cls, value: Any) -> BranchScheduleEntry:
        expected = {field.name for field in dataclasses.fields(cls)} | {"entry_id"}
        mapping = _exact_mapping(
            value, keys=expected, name="Method V1 branch schedule entry"
        )
        result = cls(
            **{field.name: mapping[field.name] for field in dataclasses.fields(cls)}
        )
        if mapping["entry_id"] != result.entry_id:
            raise ValueError("schedule entry digest is inconsistent")
        return result


def validate_manifest_splits(manifests: Sequence[DecisionGroupManifest]) -> None:
    """Prevent reset, hidden-variant, or scene-layout groups crossing splits."""

    records = tuple(manifests)
    if any(not isinstance(item, DecisionGroupManifest) for item in records):
        raise TypeError("manifests must contain DecisionGroupManifest values")
    ids = [item.manifest_id for item in records]
    groups = [item.decision_group_id for item in records]
    if len(set(ids)) != len(ids):
        raise ValueError("manifest_id values must be globally unique")
    if len(set(groups)) != len(groups):
        raise ValueError("decision_group_id values must be globally unique")

    split_maps: dict[str, dict[Any, str]] = {
        "initial_state_group": {},
        "split_group_id": {},
        "scene/layout": {},
    }
    reset_digests: dict[str, str] = {}
    for item in records:
        keys = {
            "initial_state_group": item.initial_state_group,
            "split_group_id": item.split_group_id,
            "scene/layout": (item.scene_id, item.layout_id),
        }
        for name, key in keys.items():
            previous = split_maps[name].setdefault(key, item.split)
            if previous != item.split:
                raise ValueError(f"{name} {key!r} crosses data splits")
        previous_digest = reset_digests.setdefault(
            item.initial_state_group, item.reset_state_sha256
        )
        if previous_digest != item.reset_state_sha256:
            raise ValueError("initial_state_group changes reset-state digest")


def manifest_information_stratum_counts(
    manifests: Sequence[DecisionGroupManifest],
) -> dict[str, dict[str, int]]:
    """Report evaluator-only stratum counts without changing policy inputs."""

    records = tuple(manifests)
    if any(not isinstance(item, DecisionGroupManifest) for item in records):
        raise TypeError("manifests must contain DecisionGroupManifest values")
    counts = {
        split: {stratum: 0 for stratum in INFORMATION_STRATUM_VALUES}
        for split in ("train", "validation", "calibration", "test")
    }
    for manifest in records:
        counts[manifest.split][manifest.information_stratum.value] += 1
    return counts


def validate_manifest_information_strata(
    manifests: Sequence[DecisionGroupManifest],
    *,
    expected_counts: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    """Require a manifest population to match its pre-registered strata exactly."""

    records = tuple(manifests)
    validate_manifest_splits(records)
    if not isinstance(expected_counts, Mapping):
        raise TypeError("expected_counts must be a mapping")
    if set(expected_counts) != _ALLOWED_SPLITS:
        raise ValueError(f"expected_counts must name exactly {sorted(_ALLOWED_SPLITS)}")
    reset_groups: dict[str, int] = {}
    for split in ("train", "validation", "calibration", "test"):
        split_counts = expected_counts[split]
        if not isinstance(split_counts, Mapping):
            raise TypeError(f"expected_counts[{split!r}] must be a mapping")
        raw_values = tuple(split_counts.values())
        if any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in raw_values
        ):
            raise ValueError("expected stratum counts must be non-negative integers")
        reset_groups[split] = sum(raw_values)
    expected = validate_information_stratum_counts(
        expected_counts,
        reset_groups=reset_groups,
    )
    observed = manifest_information_stratum_counts(records)
    if observed != expected:
        raise ValueError(
            "manifest information-stratum allocation differs from the "
            f"pre-registered counts; expected={expected}, observed={observed}"
        )
    return observed


def build_branch_schedule(
    manifests: Sequence[DecisionGroupManifest],
    *,
    schedule_id: str,
    start_index: int = 0,
) -> tuple[BranchScheduleEntry, ...]:
    """Create the complete candidate x model-seed execution matrix."""

    manifest_rows = tuple(manifests)
    if not manifest_rows:
        raise ValueError("at least one decision manifest is required")
    validate_manifest_splits(manifest_rows)
    schedule_name = _clean_text(schedule_id, name="schedule_id")
    if (
        not isinstance(start_index, int)
        or isinstance(start_index, bool)
        or start_index != 0
    ):
        raise ValueError("execution_index is schedule-local and must start at zero")
    rows: list[BranchScheduleEntry] = []
    execution_index = start_index
    for manifest in manifest_rows:
        for candidate in manifest.candidates:
            for repeat_index, model_seed in enumerate(manifest.model_seeds):
                rows.append(
                    BranchScheduleEntry(
                        schedule_id=schedule_name,
                        execution_index=execution_index,
                        manifest_id=manifest.manifest_id,
                        manifest_sha256=manifest.fingerprint(),
                        decision_group_id=manifest.decision_group_id,
                        initial_state_group=manifest.initial_state_group,
                        split_group_id=manifest.split_group_id,
                        split=manifest.split,
                        reset_state_sha256=manifest.reset_state_sha256,
                        candidate_id=candidate.candidate_id,
                        candidate_fingerprint=candidate.fingerprint(),
                        repeat_index=repeat_index,
                        model_seed=model_seed,
                        outcome_contract_sha256=manifest.outcome_contract.fingerprint(),
                    )
                )
                execution_index += 1
    validate_branch_schedule(manifest_rows, rows)
    return tuple(rows)


def validate_branch_schedule(
    manifests: Sequence[DecisionGroupManifest],
    schedule: Sequence[BranchScheduleEntry],
) -> None:
    """Verify exact, paired, non-replaceable schedule membership."""

    manifest_rows = tuple(manifests)
    schedule_rows = tuple(schedule)
    if not manifest_rows or not schedule_rows:
        raise ValueError("manifests and schedule must be non-empty")
    validate_manifest_splits(manifest_rows)
    if any(not isinstance(item, BranchScheduleEntry) for item in schedule_rows):
        raise TypeError("schedule must contain BranchScheduleEntry values")
    # Each decision freeze owns a local immutable schedule.  A dataset may
    # aggregate many such pre-outcome schedules; execution_index is therefore
    # local to schedule_id and is never presented as a global collection order.
    schedule_ids = tuple(dict.fromkeys(item.schedule_id for item in schedule_rows))
    for schedule_id in schedule_ids:
        indices = [
            item.execution_index
            for item in schedule_rows
            if item.schedule_id == schedule_id
        ]
        if indices != list(range(len(indices))):
            raise ValueError(
                "execution_index values must start at zero and be consecutive "
                "within each schedule_id"
            )
    entry_ids = [item.entry_id for item in schedule_rows]
    if len(set(entry_ids)) != len(entry_ids):
        raise ValueError("schedule entries must be globally unique")

    manifest_by_id = {item.manifest_id: item for item in manifest_rows}
    if {item.manifest_id for item in schedule_rows} != set(manifest_by_id):
        raise ValueError("schedule and manifest membership disagree")
    schedule_for_manifest: dict[str, str] = {}
    for row in schedule_rows:
        previous = schedule_for_manifest.setdefault(row.manifest_id, row.schedule_id)
        if previous != row.schedule_id:
            raise ValueError("one manifest may belong to only one frozen schedule")
    actual: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    for row in schedule_rows:
        manifest = manifest_by_id[row.manifest_id]
        candidate = manifest.candidate(row.candidate_id)
        expected_fields = {
            "manifest_sha256": manifest.fingerprint(),
            "decision_group_id": manifest.decision_group_id,
            "initial_state_group": manifest.initial_state_group,
            "split_group_id": manifest.split_group_id,
            "split": manifest.split,
            "reset_state_sha256": manifest.reset_state_sha256,
            "candidate_fingerprint": candidate.fingerprint(),
            "outcome_contract_sha256": manifest.outcome_contract.fingerprint(),
        }
        for name, expected in expected_fields.items():
            if getattr(row, name) != expected:
                raise ValueError(f"schedule row changes frozen {name}")
        if row.repeat_index >= len(manifest.model_seeds):
            raise ValueError("schedule repeat_index is outside the frozen seed list")
        if row.model_seed != manifest.model_seeds[row.repeat_index]:
            raise ValueError("schedule model seed differs from the paired seed")
        key = (row.candidate_id, row.repeat_index, row.model_seed)
        if key in actual[row.manifest_id]:
            raise ValueError("duplicate candidate/seed execution in schedule")
        actual[row.manifest_id].add(key)

    for manifest in manifest_rows:
        expected = {
            (candidate.candidate_id, repeat_index, model_seed)
            for candidate in manifest.candidates
            for repeat_index, model_seed in enumerate(manifest.model_seeds)
        }
        if actual[manifest.manifest_id] != expected:
            raise ValueError("schedule is not the complete candidate x seed matrix")


class CollectionAttemptStatus(str, Enum):
    OUTCOME_EVALUATED = "OUTCOME_EVALUATED"
    INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"


@dataclasses.dataclass(frozen=True)
class InfrastructureFailure:
    """Unlabelled failure of experiment infrastructure, not a policy outcome."""

    failure_type: str
    message: str
    diagnostics_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "failure_type", _clean_text(self.failure_type, name="failure_type")
        )
        object.__setattr__(self, "message", _clean_text(self.message, name="message"))
        object.__setattr__(
            self,
            "diagnostics_sha256",
            _require_sha256(self.diagnostics_sha256, name="diagnostics_sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_mapping(cls, value: Any) -> InfrastructureFailure:
        mapping = _exact_mapping(
            value,
            keys={"failure_type", "message", "diagnostics_sha256"},
            name="Method V1 infrastructure failure",
        )
        return cls(
            failure_type=str(mapping["failure_type"]),
            message=str(mapping["message"]),
            diagnostics_sha256=str(mapping["diagnostics_sha256"]),
        )


@dataclasses.dataclass(frozen=True)
class CollectionAttempt:
    """Single-use schedule consumption with explicit label separation."""

    attempt_id: str
    schedule_entry: BranchScheduleEntry
    status: CollectionAttemptStatus
    artifact_tree_sha256: str
    branch: ObservedBranch | None = None
    infrastructure_failure: InfrastructureFailure | None = None
    schema_version: str = METHOD_V1_ATTEMPT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "attempt_id", _clean_text(self.attempt_id, name="attempt_id")
        )
        if self.schema_version != METHOD_V1_ATTEMPT_SCHEMA:
            raise ValueError("unsupported Method V1 attempt schema")
        if not isinstance(self.schedule_entry, BranchScheduleEntry):
            raise TypeError("schedule_entry must be a BranchScheduleEntry")
        if not isinstance(self.status, CollectionAttemptStatus):
            raise TypeError("status must be a CollectionAttemptStatus")
        object.__setattr__(
            self,
            "artifact_tree_sha256",
            _require_sha256(self.artifact_tree_sha256, name="artifact_tree_sha256"),
        )
        if self.status is CollectionAttemptStatus.INFRASTRUCTURE_FAILURE:
            if self.branch is not None:
                raise ValueError(
                    "infrastructure failure must not carry an outcome label"
                )
            if not isinstance(self.infrastructure_failure, InfrastructureFailure):
                raise ValueError("infrastructure failure diagnostics are required")
            return
        if not isinstance(self.branch, ObservedBranch):
            raise TypeError("outcome-evaluated attempt requires an ObservedBranch")
        if self.infrastructure_failure is not None:
            raise ValueError(
                "outcome-evaluated attempt cannot be infrastructure failure"
            )
        self._validate_branch_identity(self.branch)

    def _validate_branch_identity(self, branch: ObservedBranch) -> None:
        row = self.schedule_entry
        expected = {
            "initial_state_group": row.initial_state_group,
            "decision_group_id": row.decision_group_id,
            "split": row.split,
            "reset_state_sha256": row.reset_state_sha256,
            "repeat_index": row.repeat_index,
            "candidate_id": row.candidate_id,
            "candidate_fingerprint": row.candidate_fingerprint,
            "outcome_contract_sha256": row.outcome_contract_sha256,
        }
        observed = {
            "initial_state_group": branch.initial_state_group,
            "decision_group_id": branch.decision_group_id,
            "split": branch.split,
            "reset_state_sha256": branch.reset_state_sha256,
            "repeat_index": branch.repeat_index,
            "candidate_id": branch.candidate_id,
            "candidate_fingerprint": branch.candidate_fingerprint,
            "outcome_contract_sha256": branch.outcome_contract.fingerprint(),
        }
        for name, expected_value in expected.items():
            if observed[name] != expected_value:
                raise ValueError(f"observed branch changes schedule field {name}")

    @property
    def has_training_label(self) -> bool:
        return self.status is CollectionAttemptStatus.OUTCOME_EVALUATED

    def to_dict(self, *, include_private_branch: bool = True) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "schedule_entry": self.schedule_entry.to_dict(),
            "status": self.status.value,
            "artifact_tree_sha256": self.artifact_tree_sha256,
            "branch": (
                self.branch.to_dict(include_private=include_private_branch)
                if self.branch is not None
                else None
            ),
            "infrastructure_failure": (
                self.infrastructure_failure.to_dict()
                if self.infrastructure_failure is not None
                else None
            ),
        }

    @classmethod
    def from_mapping(cls, value: Any) -> CollectionAttempt:
        mapping = _exact_mapping(
            value,
            keys={
                "schema_version",
                "attempt_id",
                "schedule_entry",
                "status",
                "artifact_tree_sha256",
                "branch",
                "infrastructure_failure",
            },
            name="Method V1 collection attempt",
        )
        raw_branch = mapping["branch"]
        raw_failure = mapping["infrastructure_failure"]
        return cls(
            schema_version=str(mapping["schema_version"]),
            attempt_id=str(mapping["attempt_id"]),
            schedule_entry=BranchScheduleEntry.from_mapping(mapping["schedule_entry"]),
            status=CollectionAttemptStatus(str(mapping["status"])),
            artifact_tree_sha256=str(mapping["artifact_tree_sha256"]),
            branch=(
                None
                if raw_branch is None
                else ObservedBranch.from_public_mapping(raw_branch)
            ),
            infrastructure_failure=(
                None
                if raw_failure is None
                else InfrastructureFailure.from_mapping(raw_failure)
            ),
        )


@dataclasses.dataclass(frozen=True)
class ExecutedOnlyTarget:
    """Dense candidate view with exactly one observed Bernoulli target."""

    decision_group_id: str
    candidate_ids: tuple[str, ...]
    labels: tuple[bool | None, ...]
    executed_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decision_group_id",
            _clean_text(self.decision_group_id, name="decision_group_id"),
        )
        ids = tuple(
            _clean_text(item, name="candidate_id") for item in self.candidate_ids
        )
        labels = tuple(self.labels)
        mask = tuple(self.executed_mask)
        if not ids or len(ids) != len(labels) or len(ids) != len(mask):
            raise ValueError("candidate_ids, labels, and executed_mask must align")
        if sum(bool(item) for item in mask) != 1:
            raise ValueError("executed_mask must select exactly one candidate")
        for label, executed in zip(labels, mask, strict=True):
            if executed and not isinstance(label, bool):
                raise ValueError("the executed candidate requires a boolean label")
            if not executed and label is not None:
                raise ValueError("unexecuted candidate outcomes must remain unknown")
        object.__setattr__(self, "candidate_ids", ids)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "executed_mask", mask)

    @property
    def executed_index(self) -> int:
        return self.executed_mask.index(True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_group_id": self.decision_group_id,
            "candidate_ids": list(self.candidate_ids),
            "labels": list(self.labels),
            "executed_mask": list(self.executed_mask),
        }


def executed_only_target(
    manifest: DecisionGroupManifest,
    branch: ObservedBranch,
) -> ExecutedOnlyTarget:
    """Expose one observed outcome without fabricating counterfactual labels."""

    if branch.decision_group_id != manifest.decision_group_id:
        raise ValueError("branch and manifest decision groups disagree")
    candidate_ids = tuple(item.candidate_id for item in manifest.candidates)
    try:
        index = candidate_ids.index(branch.candidate_id)
    except ValueError as error:
        raise ValueError("executed candidate is absent from the manifest") from error
    candidate = manifest.candidates[index]
    if branch.candidate_fingerprint != candidate.fingerprint():
        raise ValueError("executed candidate fingerprint differs from the manifest")
    labels: list[bool | None] = [None] * len(candidate_ids)
    mask = [False] * len(candidate_ids)
    labels[index] = branch.observed_outcome
    mask[index] = True
    return ExecutedOnlyTarget(
        decision_group_id=manifest.decision_group_id,
        candidate_ids=candidate_ids,
        labels=tuple(labels),
        executed_mask=tuple(mask),
    )


def validate_collection_attempts(
    manifests: Sequence[DecisionGroupManifest],
    schedule: Sequence[BranchScheduleEntry],
    attempts: Sequence[CollectionAttempt],
    *,
    require_complete: bool,
) -> None:
    """Validate single use, split isolation, and observed-only supervision."""

    manifest_rows = tuple(manifests)
    schedule_rows = tuple(schedule)
    attempt_rows = tuple(attempts)
    validate_branch_schedule(manifest_rows, schedule_rows)
    if any(not isinstance(item, CollectionAttempt) for item in attempt_rows):
        raise TypeError("attempts must contain CollectionAttempt values")
    schedule_by_id = {item.entry_id: item for item in schedule_rows}
    attempt_ids = [item.attempt_id for item in attempt_rows]
    entry_ids = [item.schedule_entry.entry_id for item in attempt_rows]
    if len(set(attempt_ids)) != len(attempt_ids):
        raise ValueError("attempt_id values must be globally unique")
    if len(set(entry_ids)) != len(entry_ids):
        raise ValueError("a schedule entry may be consumed at most once")
    for attempt in attempt_rows:
        expected = schedule_by_id.get(attempt.schedule_entry.entry_id)
        if expected is None or expected != attempt.schedule_entry:
            raise ValueError("attempt references an unknown or altered schedule entry")
    if require_complete and set(entry_ids) != set(schedule_by_id):
        raise ValueError("collection is incomplete")
    if require_complete and any(
        attempt.status is not CollectionAttemptStatus.OUTCOME_EVALUATED
        for attempt in attempt_rows
    ):
        raise ValueError(
            "complete collection requires an outcome-evaluated attempt for every "
            "schedule entry; infrastructure failures remain explicitly incomplete"
        )

    branches = tuple(
        attempt.branch
        for attempt in attempt_rows
        if attempt.status is CollectionAttemptStatus.OUTCOME_EVALUATED
    )
    if any(branch is None for branch in branches):  # pragma: no cover - type guard
        raise RuntimeError("outcome-evaluated attempt lost its branch")
    typed_branches = tuple(branch for branch in branches if branch is not None)
    branch_ids = [branch.branch_id for branch in typed_branches]
    receipt_ids = [branch.execution_receipt_id for branch in typed_branches]
    if len(set(branch_ids)) != len(branch_ids):
        raise ValueError("observed branch IDs must be unique")
    if len(set(receipt_ids)) != len(receipt_ids):
        raise ValueError("execution receipt IDs must be unique")
    if typed_branches:
        validate_group_splits(typed_branches)


@dataclasses.dataclass(frozen=True)
class MethodV1OutcomeDataset:
    """Manifest-backed Method V1 collection; partial collections stay explicit."""

    dataset_id: str
    manifests: tuple[DecisionGroupManifest, ...]
    schedule: tuple[BranchScheduleEntry, ...]
    attempts: tuple[CollectionAttempt, ...]
    complete: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dataset_id", _clean_text(self.dataset_id, name="dataset_id")
        )
        manifests = tuple(self.manifests)
        schedule = tuple(self.schedule)
        attempts = tuple(self.attempts)
        object.__setattr__(self, "manifests", manifests)
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "attempts", attempts)
        validate_collection_attempts(
            manifests, schedule, attempts, require_complete=self.complete
        )
        contract_ids = {
            manifest.outcome_contract.fingerprint() for manifest in manifests
        }
        if len(contract_ids) != 1:
            raise ValueError("one Method V1 dataset must use one outcome contract")

    @property
    def observed_branches(self) -> tuple[ObservedBranch, ...]:
        return tuple(
            attempt.branch
            for attempt in self.attempts
            if attempt.branch is not None
            and attempt.status is CollectionAttemptStatus.OUTCOME_EVALUATED
        )

    def executed_targets(self) -> tuple[ExecutedOnlyTarget, ...]:
        by_group = {item.decision_group_id: item for item in self.manifests}
        return tuple(
            executed_only_target(by_group[branch.decision_group_id], branch)
            for branch in self.observed_branches
        )

    def summary(self) -> dict[str, Any]:
        counts = Counter(attempt.status.value for attempt in self.attempts)
        return {
            "dataset_id": self.dataset_id,
            "decision_groups": len(self.manifests),
            "scheduled_branches": len(self.schedule),
            "consumed_attempts": len(self.attempts),
            "outcome_evaluated": counts[
                CollectionAttemptStatus.OUTCOME_EVALUATED.value
            ],
            "infrastructure_failures": counts[
                CollectionAttemptStatus.INFRASTRUCTURE_FAILURE.value
            ],
            "unconsumed": len(self.schedule) - len(self.attempts),
            "complete": self.complete,
            "splits": dict(
                sorted(Counter(item.split for item in self.manifests).items())
            ),
            "outcome_contract_sha256": self.manifests[0].outcome_contract.fingerprint(),
        }

    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "dataset_id": self.dataset_id,
                "manifests": [item.fingerprint() for item in self.manifests],
                "schedule": [item.entry_id for item in self.schedule],
                "attempts": [
                    item.to_dict(include_private_branch=False) for item in self.attempts
                ],
                "complete": self.complete,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "method-v1-outcome-dataset-v1",
            "dataset_id": self.dataset_id,
            "manifests": [item.to_dict() for item in self.manifests],
            "schedule": [item.to_dict() for item in self.schedule],
            "attempts": [
                item.to_dict(include_private_branch=False) for item in self.attempts
            ],
            "complete": self.complete,
            "summary": self.summary(),
            "dataset_sha256": self.fingerprint(),
        }

    @classmethod
    def from_mapping(cls, value: Any) -> MethodV1OutcomeDataset:
        mapping = _exact_mapping(
            value,
            keys={
                "schema_version",
                "dataset_id",
                "manifests",
                "schedule",
                "attempts",
                "complete",
                "summary",
                "dataset_sha256",
            },
            name="Method V1 outcome dataset",
        )
        if mapping["schema_version"] != "method-v1-outcome-dataset-v1":
            raise ValueError("unsupported Method V1 dataset schema")
        result = cls(
            dataset_id=str(mapping["dataset_id"]),
            manifests=tuple(
                DecisionGroupManifest.from_mapping(item)
                for item in mapping["manifests"]
            ),
            schedule=tuple(
                BranchScheduleEntry.from_mapping(item) for item in mapping["schedule"]
            ),
            attempts=tuple(
                CollectionAttempt.from_mapping(item) for item in mapping["attempts"]
            ),
            complete=bool(mapping["complete"]),
        )
        if mapping["summary"] != result.summary():
            raise ValueError("Method V1 dataset summary is inconsistent")
        if mapping["dataset_sha256"] != result.fingerprint():
            raise ValueError("Method V1 dataset digest is inconsistent")
        return result


def primitive_outcome_coverage(
    dataset: MethodV1OutcomeDataset,
) -> dict[str, Any]:
    """Audit executed primitive-by-outcome support after collection.

    This diagnostic reads only evaluator-side manifests and *observed*
    executed outcomes.  It never creates counterfactual labels, and neither
    the information stratum nor the outcome is added to scorer input.
    """

    if not isinstance(dataset, MethodV1OutcomeDataset):
        raise TypeError("dataset must be a MethodV1OutcomeDataset")
    split_order = ("train", "validation", "calibration", "test")
    primitive_order = (Primitive.DIRECT, Primitive.OPEN)
    splits: dict[str, dict[str, Any]] = {
        split: {
            "decision_groups_by_information_stratum": {
                stratum: 0 for stratum in INFORMATION_STRATUM_VALUES
            },
            "outcome_evaluated_by_information_stratum": {
                stratum: 0 for stratum in INFORMATION_STRATUM_VALUES
            },
            "primitive_by_outcome": {
                primitive.value: {"failure": 0, "success": 0}
                for primitive in primitive_order
            },
            "outcome_evaluated": 0,
        }
        for split in split_order
    }
    manifest_by_id = {item.manifest_id: item for item in dataset.manifests}
    for manifest in dataset.manifests:
        splits[manifest.split]["decision_groups_by_information_stratum"][
            manifest.information_stratum.value
        ] += 1
    for attempt in dataset.attempts:
        if attempt.status is not CollectionAttemptStatus.OUTCOME_EVALUATED:
            continue
        branch = attempt.branch
        if branch is None:  # pragma: no cover - enforced by CollectionAttempt
            raise RuntimeError("outcome-evaluated attempt lost its branch")
        manifest = manifest_by_id[attempt.schedule_entry.manifest_id]
        candidate = manifest.candidate(attempt.schedule_entry.candidate_id)
        outcome_name = "success" if branch.observed_outcome else "failure"
        row = splits[manifest.split]
        row["primitive_by_outcome"][candidate.primitive.value][outcome_name] += 1
        row["outcome_evaluated_by_information_stratum"][
            manifest.information_stratum.value
        ] += 1
        row["outcome_evaluated"] += 1
    return {
        "schema_version": "method-v1-primitive-outcome-coverage-v1",
        "splits": splits,
    }


def validate_primitive_outcome_coverage(
    dataset: MethodV1OutcomeDataset,
    *,
    required_splits: Sequence[str] = ("train", "validation", "test"),
) -> dict[str, Any]:
    """Fail closed unless DIRECT and OPEN each contain both Bernoulli labels."""

    split_names = tuple(required_splits)
    if not split_names:
        raise ValueError("required_splits must be non-empty")
    if any(split not in _ALLOWED_SPLITS for split in split_names):
        raise ValueError(
            f"required_splits must be drawn from {sorted(_ALLOWED_SPLITS)}"
        )
    if len(set(split_names)) != len(split_names):
        raise ValueError("required_splits must be unique")
    report = primitive_outcome_coverage(dataset)
    missing: list[str] = []
    for split in split_names:
        primitive_counts = report["splits"][split]["primitive_by_outcome"]
        for primitive in (Primitive.DIRECT.value, Primitive.OPEN.value):
            for outcome in ("failure", "success"):
                if primitive_counts[primitive][outcome] == 0:
                    missing.append(f"{split}:{primitive}:{outcome}")
    if missing:
        raise ValueError(
            f"primitive-by-outcome coverage is incomplete; missing={missing}"
        )
    return report
