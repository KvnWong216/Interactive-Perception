"""Train the Method-V1 grounded outcome scorer from observed branch outcomes.

This module deliberately sits *after* candidate proposal, frozen VLM feature
extraction, and reset-controlled branch collection.  It never creates labels
or fills counterfactual outcomes.  One :class:`EncodedOutcomeRecord` contains a
full candidate set plus exactly one receipt-backed executed outcome.

The training target is the Bernoulli outcome defined by the frozen
``full_task_with_fixed_continuation_v1`` execution contract.  Training and
early stopping use only ``train`` and ``validation`` splits.  An optional scalar
temperature may be fitted later on a disjoint ``calibration`` split; it cannot
change candidate ranking.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import Primitive, canonical_json_bytes, canonical_sha256
from .losses import executed_candidate_bernoulli_nll
from .model import GroundedOutcomeModel
from .selection import ExpectedSuccessSelector, SelectionDecision, ValuePrediction
from .tokens import CandidateTokenField, FrozenTokenField

try:  # Keep execution-only installations importable.
    import torch
    from torch import Tensor, nn
    from torch.utils.data import DataLoader, Dataset, Sampler
except ImportError:  # pragma: no cover - exercised in core-only environments.
    torch = None
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = Any  # type: ignore[misc,assignment]
    Dataset = object  # type: ignore[assignment]
    Sampler = object  # type: ignore[assignment]


_SHA256_HEX = frozenset("0123456789abcdef")
_CHECKPOINT_SCHEMA = "method-v1-outcome-checkpoint-v1"
_ALLOWED_SPLITS = frozenset({"train", "validation", "calibration", "test"})


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError(
            "Method-V1 outcome training requires the optional PyTorch dependency"
        )


def _clean_text(value: Any, *, name: str) -> str:
    result = " ".join(str(value or "").split())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _sha256(value: Any, *, name: str) -> str:
    result = str(value)
    if len(result) != 64 or any(character not in _SHA256_HEX for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(value: object, path: Path) -> None:
    _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(canonical_json_bytes(value) + b"\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclasses.dataclass(frozen=True)
class EncodedOutcomeRecord:
    """One observed execution aligned to a frozen full candidate token field.

    The token field must contain one context (``batch_size == 1``).  Exactly one
    valid candidate fingerprint is named as executed.  No label exists for the
    remaining candidates; collation writes NaN outside the executed mask.

    ``split_group_id`` is broader than a decision group when prompt variants or
    hidden-state variants share a physical reset.  It is the unit that must
    never cross train/validation/calibration/test partitions.
    """

    record_id: str
    experiment_id: str
    initial_state_group: str
    decision_group_id: str
    split_group_id: str
    split: str
    proposal_provider_id: str
    configuration_sha256: str
    branch_fingerprint: str
    outcome_contract_sha256: str
    token_cache_sha256: str
    field: CandidateTokenField
    executed_candidate_fingerprint: str
    observed_outcome: bool

    def __post_init__(self) -> None:
        for name in (
            "record_id",
            "experiment_id",
            "initial_state_group",
            "decision_group_id",
            "split_group_id",
            "proposal_provider_id",
        ):
            object.__setattr__(self, name, _clean_text(getattr(self, name), name=name))
        split = str(self.split).strip().lower()
        if split not in _ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {sorted(_ALLOWED_SPLITS)}")
        object.__setattr__(self, "split", split)
        for name in (
            "branch_fingerprint",
            "configuration_sha256",
            "outcome_contract_sha256",
            "token_cache_sha256",
            "executed_candidate_fingerprint",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        if not isinstance(self.field, CandidateTokenField):
            raise TypeError("field must be a CandidateTokenField")
        if self.field.batch_size != 1:
            raise ValueError("each encoded outcome record must contain one context")
        if not isinstance(self.observed_outcome, bool):
            raise TypeError("observed_outcome must be a bool")

        valid_fingerprints = tuple(
            fingerprint
            for fingerprint, valid in zip(
                self.field.candidate_fingerprints[0],
                self.field.valid_mask[0].tolist(),
                strict=True,
            )
            if valid
        )
        if valid_fingerprints.count(self.executed_candidate_fingerprint) != 1:
            raise ValueError(
                "executed candidate fingerprint must identify exactly one valid candidate"
            )

    @property
    def provider_id(self) -> str:
        return self.field.public_context.provider_id

    @property
    def context_fingerprint(self) -> str:
        return self.field.context_fingerprints[0]


class EncodedOutcomeDataset(Dataset):
    """In-memory index over disk-cached frozen features and real outcomes."""

    def __init__(self, records: Sequence[EncodedOutcomeRecord]) -> None:
        _require_torch()
        self._records = tuple(records)
        if not self._records:
            raise ValueError("encoded outcome dataset must not be empty")
        if any(
            not isinstance(record, EncodedOutcomeRecord) for record in self._records
        ):
            raise TypeError("dataset accepts EncodedOutcomeRecord values only")
        record_ids = [record.record_id for record in self._records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("record_id values must be unique")
        branch_fingerprints = [record.branch_fingerprint for record in self._records]
        if len(branch_fingerprints) != len(set(branch_fingerprints)):
            raise ValueError("branch fingerprints must be unique")
        validate_split_groups(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> EncodedOutcomeRecord:
        return self._records[index]

    @property
    def records(self) -> tuple[EncodedOutcomeRecord, ...]:
        return self._records

    def subset(self, split: str) -> EncodedOutcomeDataset:
        normalized = str(split).strip().lower()
        return EncodedOutcomeDataset(
            tuple(record for record in self._records if record.split == normalized)
        )


def encoded_records_from_method_v1_dataset(
    dataset: object,
    *,
    token_fields_by_sha256: Mapping[str, CandidateTokenField],
    splits: Iterable[str] | None = None,
) -> tuple[EncodedOutcomeRecord, ...]:
    """Join canonical Method-V1 manifests/outcomes to frozen token caches.

    ``method_v1_data`` owns outcome admission.  This function intentionally
    consumes its typed :class:`MethodV1OutcomeDataset` rather than reparsing
    arbitrary collector files.  Infrastructure failures are absent from
    ``observed_branches`` and therefore cannot silently become negative labels.
    """

    _require_torch()
    from .method_v1_data import MethodV1OutcomeDataset

    if not isinstance(dataset, MethodV1OutcomeDataset):
        raise TypeError("dataset must be a canonical MethodV1OutcomeDataset")
    allowed_splits = None if splits is None else frozenset(str(item) for item in splits)
    if allowed_splits is not None and (
        not allowed_splits or allowed_splits - _ALLOWED_SPLITS
    ):
        raise ValueError("splits must be a non-empty subset of canonical splits")
    manifests = {manifest.decision_group_id: manifest for manifest in dataset.manifests}
    records: list[EncodedOutcomeRecord] = []
    for branch in dataset.observed_branches:
        manifest = manifests[branch.decision_group_id]
        if allowed_splits is not None and manifest.split not in allowed_splits:
            continue
        try:
            field = token_fields_by_sha256[manifest.token_cache_sha256]
        except KeyError as error:
            raise ValueError(
                f"missing token cache {manifest.token_cache_sha256}"
            ) from error
        if not isinstance(field, CandidateTokenField) or field.batch_size != 1:
            raise TypeError("token cache mapping must contain one-context fields")
        if field.public_context.provider_id != manifest.token_provider_id:
            raise ValueError("token cache provider identity differs from manifest")
        if field.context_fingerprints != (manifest.context.fingerprint(),):
            raise ValueError("token cache context differs from decision manifest")
        expected_ids = tuple(
            candidate.candidate_id for candidate in manifest.candidates
        )
        expected_fingerprints = tuple(
            candidate.fingerprint() for candidate in manifest.candidates
        )
        if field.candidate_ids[0] != expected_ids:
            raise ValueError("token cache candidate IDs differ from decision manifest")
        if field.candidate_fingerprints[0] != expected_fingerprints:
            raise ValueError(
                "token cache candidate fingerprints differ from decision manifest"
            )
        records.append(
            EncodedOutcomeRecord(
                record_id=branch.branch_id,
                experiment_id=manifest.experiment_id,
                initial_state_group=manifest.initial_state_group,
                decision_group_id=manifest.decision_group_id,
                split_group_id=manifest.split_group_id,
                split=manifest.split,
                proposal_provider_id=manifest.proposal_provider_id,
                configuration_sha256=manifest.configuration_sha256,
                branch_fingerprint=branch.public_fingerprint(),
                outcome_contract_sha256=manifest.outcome_contract.fingerprint(),
                token_cache_sha256=manifest.token_cache_sha256,
                field=field,
                executed_candidate_fingerprint=branch.candidate_fingerprint,
                observed_outcome=branch.observed_outcome,
            )
        )
    if not records:
        raise ValueError("Method-V1 dataset contains no outcome-evaluated branches")
    return tuple(records)


def validate_split_groups(records: Sequence[EncodedOutcomeRecord]) -> None:
    """Reject leakage of reset/layout groups across data splits."""

    group_to_split: dict[str, str] = {}
    decision_to_split: dict[str, str] = {}
    reset_to_split: dict[str, str] = {}
    for record in records:
        previous_reset = reset_to_split.setdefault(
            record.initial_state_group, record.split
        )
        if previous_reset != record.split:
            raise ValueError(
                f"initial state group {record.initial_state_group!r} crosses data splits"
            )
        previous = group_to_split.setdefault(record.split_group_id, record.split)
        if previous != record.split:
            raise ValueError(
                f"split group {record.split_group_id!r} crosses data splits"
            )
        previous_decision = decision_to_split.setdefault(
            record.decision_group_id, record.split
        )
        if previous_decision != record.split:
            raise ValueError(
                f"decision group {record.decision_group_id!r} crosses data splits"
            )


class GroupUniformSampler(Sampler):
    """Sample physical reset groups uniformly, then one observed branch.

    Each cycle visits every group once in a random order.  The selected branch
    within that group is uniform.  Thus groups with six candidates cannot
    dominate groups with two.  ``set_epoch`` makes the sequence reproducible
    while still changing it across epochs.
    """

    def __init__(
        self,
        records: Sequence[EncodedOutcomeRecord],
        *,
        seed: int,
        samples_per_epoch: int | None = None,
    ) -> None:
        _require_torch()
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        by_group: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            if not isinstance(record, EncodedOutcomeRecord):
                raise TypeError("sampler records must be EncodedOutcomeRecord values")
            by_group[record.initial_state_group].append(index)
        if not by_group:
            raise ValueError("group-uniform sampling requires at least one group")
        if samples_per_epoch is None:
            samples_per_epoch = len(records)
        if (
            not isinstance(samples_per_epoch, int)
            or isinstance(samples_per_epoch, bool)
            or samples_per_epoch < 1
        ):
            raise ValueError("samples_per_epoch must be a positive integer")
        self._by_group = {
            key: tuple(values) for key, values in sorted(by_group.items())
        }
        self._seed = seed
        self._epoch = 0
        self._samples_per_epoch = samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self._epoch = epoch

    def __len__(self) -> int:
        return self._samples_per_epoch

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self._seed + 1_000_003 * self._epoch)
        groups = list(self._by_group)
        emitted = 0
        while emitted < self._samples_per_epoch:
            rng.shuffle(groups)
            for group_id in groups:
                if emitted >= self._samples_per_epoch:
                    break
                yield rng.choice(self._by_group[group_id])
                emitted += 1


@dataclasses.dataclass(frozen=True)
class OutcomeTrainingBatch:
    """Padded candidate fields plus receipt-backed executed-only targets."""

    field: CandidateTokenField
    observed_outcomes: Tensor
    executed_mask: Tensor
    record_ids: tuple[str, ...]
    decision_group_ids: tuple[str, ...]
    split_group_ids: tuple[str, ...]
    outcome_contract_sha256: str

    def __post_init__(self) -> None:
        _require_torch()
        if not isinstance(self.field, CandidateTokenField):
            raise TypeError("field must be a CandidateTokenField")
        if not isinstance(self.observed_outcomes, torch.Tensor):
            raise TypeError("observed_outcomes must be a tensor")
        if not isinstance(self.executed_mask, torch.Tensor):
            raise TypeError("executed_mask must be a tensor")
        expected = (self.field.batch_size, self.field.candidate_count)
        if self.observed_outcomes.shape != expected:
            raise ValueError("observed_outcomes shape does not match candidate field")
        if (
            self.executed_mask.shape != expected
            or self.executed_mask.dtype is not torch.bool
        ):
            raise ValueError("executed_mask must be a matching bool tensor")
        if bool((self.executed_mask.sum(dim=1) != 1).any()):
            raise ValueError(
                "each training row must name exactly one executed candidate"
            )
        if len(self.record_ids) != self.field.batch_size:
            raise ValueError("record IDs do not match batch size")

    def to(self, device: str | torch.device) -> OutcomeTrainingBatch:
        """Move train-time tensors without changing frozen identities."""

        _require_torch()
        context = dataclasses.replace(
            self.field.public_context,
            tokens=self.field.public_context.tokens.to(device),
            valid_mask=self.field.public_context.valid_mask.to(device),
            current_patch_mask=self.field.public_context.current_patch_mask.to(device),
            patch_xyxy=self.field.public_context.patch_xyxy.to(device),
        )
        updates: dict[str, object] = {
            "tokens": self.field.tokens.to(device),
            "valid_mask": self.field.valid_mask.to(device),
            "grounding_support": self.field.grounding_support.to(device),
            "public_context": context,
        }
        if hasattr(self.field, "public_state_values"):
            state_values = self.field.public_state_values
            state_mask = self.field.public_state_valid_mask
            updates["public_state_values"] = (
                None if state_values is None else state_values.to(device)
            )
            updates["public_state_valid_mask"] = (
                None if state_mask is None else state_mask.to(device)
            )
        field = dataclasses.replace(self.field, **updates)
        return dataclasses.replace(
            self,
            field=field,
            observed_outcomes=self.observed_outcomes.to(device),
            executed_mask=self.executed_mask.to(device),
        )

    def loss(self, prediction: object) -> Tensor:
        """Check prediction identity before computing executed-only NLL."""

        logits = getattr(prediction, "task_success_logits", None)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("prediction must expose task_success_logits")
        if tuple(getattr(prediction, "candidate_fingerprints", ())) != tuple(
            self.field.candidate_fingerprints
        ):
            raise ValueError("prediction candidate fingerprints changed")
        if tuple(getattr(prediction, "context_fingerprints", ())) != tuple(
            self.field.context_fingerprints
        ):
            raise ValueError("prediction context fingerprints changed")
        return executed_candidate_bernoulli_nll(
            logits,
            self.observed_outcomes.to(device=logits.device),
            self.executed_mask.to(device=logits.device),
        )


def collate_outcome_records(
    records: Sequence[EncodedOutcomeRecord],
) -> OutcomeTrainingBatch:
    """Pad independently cached fields while preserving all identity metadata."""

    _require_torch()
    rows = tuple(records)
    if not rows:
        raise ValueError("cannot collate an empty outcome batch")
    if any(not isinstance(row, EncodedOutcomeRecord) for row in rows):
        raise TypeError("collation accepts EncodedOutcomeRecord values only")
    providers = {row.provider_id for row in rows}
    contracts = {row.outcome_contract_sha256 for row in rows}
    context_dims = {row.field.public_context.channel_count for row in rows}
    candidate_dims = {int(row.field.tokens.shape[2]) for row in rows}
    if len(providers) != 1:
        raise ValueError("one batch cannot mix frozen feature providers")
    if len(contracts) != 1:
        raise ValueError("one batch cannot mix outcome contracts")
    if len(context_dims) != 1 or len(candidate_dims) != 1:
        raise ValueError("one batch cannot mix token dimensions")

    batch_size = len(rows)
    max_tokens = max(row.field.public_context.token_count for row in rows)
    max_candidates = max(row.field.candidate_count for row in rows)
    context_dim = next(iter(context_dims))
    candidate_dim = next(iter(candidate_dims))
    # Frozen Qwen caches may be bfloat16.  The small scorer is trained in
    # float32 by default, so conversion happens explicitly at this boundary.
    dtype = torch.float32

    context_tokens = torch.zeros(batch_size, max_tokens, context_dim, dtype=dtype)
    context_valid = torch.zeros(batch_size, max_tokens, dtype=torch.bool)
    current_patches = torch.zeros(batch_size, max_tokens, dtype=torch.bool)
    patch_xyxy = torch.zeros(batch_size, max_tokens, 4, dtype=dtype)
    candidate_tokens = torch.zeros(
        batch_size, max_candidates, candidate_dim, dtype=dtype
    )
    candidate_valid = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    support = torch.zeros(batch_size, max_candidates, max_tokens, dtype=torch.bool)
    outcomes = torch.full((batch_size, max_candidates), float("nan"), dtype=dtype)
    executed = torch.zeros(batch_size, max_candidates, dtype=torch.bool)

    camera_ids: list[tuple[str | None, ...]] = []
    frame_ids: list[tuple[str | None, ...]] = []
    candidate_ids: list[tuple[str | None, ...]] = []
    candidate_fingerprints: list[tuple[str | None, ...]] = []
    primitives: list[tuple[Primitive | None, ...]] = []
    context_fingerprints: list[str] = []
    state_values: list[Tensor] = []
    state_masks: list[Tensor] = []

    for batch_index, row in enumerate(rows):
        field = row.field
        context = field.public_context
        token_count = context.token_count
        candidate_count = field.candidate_count
        context_tokens[batch_index, :token_count] = context.tokens[0].to(dtype=dtype)
        context_valid[batch_index, :token_count] = context.valid_mask[0]
        current_patches[batch_index, :token_count] = context.current_patch_mask[0]
        patch_xyxy[batch_index, :token_count] = context.patch_xyxy[0].to(dtype=dtype)
        candidate_tokens[batch_index, :candidate_count] = field.tokens[0].to(
            dtype=dtype
        )
        candidate_valid[batch_index, :candidate_count] = field.valid_mask[0]
        support[batch_index, :candidate_count, :token_count] = field.grounding_support[
            0
        ]

        padded_cameras = tuple(context.camera_ids[0]) + (None,) * (
            max_tokens - token_count
        )
        padded_frames = tuple(context.frame_ids[0]) + (None,) * (
            max_tokens - token_count
        )
        camera_ids.append(padded_cameras)
        frame_ids.append(padded_frames)
        candidate_ids.append(
            tuple(field.candidate_ids[0]) + (None,) * (max_candidates - candidate_count)
        )
        candidate_fingerprints.append(
            tuple(field.candidate_fingerprints[0])
            + (None,) * (max_candidates - candidate_count)
        )
        primitives.append(
            tuple(field.primitives[0]) + (None,) * (max_candidates - candidate_count)
        )
        context_fingerprints.append(context.context_fingerprints[0])

        matches = [
            index
            for index, fingerprint in enumerate(field.candidate_fingerprints[0])
            if fingerprint == row.executed_candidate_fingerprint
            and bool(field.valid_mask[0, index])
        ]
        if len(matches) != 1:
            raise ValueError("executed candidate identity changed before collation")
        executed_index = matches[0]
        executed[batch_index, executed_index] = True
        outcomes[batch_index, executed_index] = float(row.observed_outcome)

        if hasattr(field, "public_state_values"):
            raw_state = field.public_state_values
            raw_mask = field.public_state_valid_mask
            if raw_state is not None and raw_mask is not None:
                state_values.append(raw_state[0].to(dtype=dtype))
                state_masks.append(raw_mask[0])

    public_context = FrozenTokenField(
        tokens=context_tokens,
        valid_mask=context_valid,
        current_patch_mask=current_patches,
        camera_ids=tuple(camera_ids),
        frame_ids=tuple(frame_ids),
        patch_xyxy=patch_xyxy,
        context_fingerprints=tuple(context_fingerprints),
        provider_id=next(iter(providers)),
    )
    field_kwargs: dict[str, object] = {}
    if hasattr(rows[0].field, "public_state_values"):
        if len(state_values) != batch_size or len(state_masks) != batch_size:
            raise ValueError("public state fields changed across cached records")
        field_kwargs = {
            "public_state_values": torch.stack(state_values),
            "public_state_valid_mask": torch.stack(state_masks).to(dtype=torch.bool),
        }
    field = CandidateTokenField(
        tokens=candidate_tokens,
        valid_mask=candidate_valid,
        candidate_ids=tuple(candidate_ids),
        candidate_fingerprints=tuple(candidate_fingerprints),
        primitives=tuple(primitives),
        grounding_support=support,
        public_context=public_context,
        **field_kwargs,
    )
    return OutcomeTrainingBatch(
        field=field,
        observed_outcomes=outcomes,
        executed_mask=executed,
        record_ids=tuple(row.record_id for row in rows),
        decision_group_ids=tuple(row.decision_group_id for row in rows),
        split_group_ids=tuple(row.split_group_id for row in rows),
        outcome_contract_sha256=next(iter(contracts)),
    )


def _read_json_mapping(path: str | Path) -> Mapping[str, object]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read strict JSON object {source}") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"{source} must contain a JSON object")
    return value


def load_method_v1_outcome_dataset(
    path: str | Path,
) -> object:
    """Load a self-contained dataset for diagnostics only.

    The JSON object is deliberately *not* receipt-backed evidence.  Canonical
    training and formal evaluation reject the returned value; use
    :func:`build_method_v1_outcome_dataset` with collector freeze and
    attempt paths for those workflows.
    """

    from .method_v1_data import MethodV1OutcomeDataset

    return MethodV1OutcomeDataset.from_mapping(_read_json_mapping(path))


_RECEIPT_ADMISSION_CAPABILITY = object()


@dataclasses.dataclass(frozen=True, init=False)
class ReceiptBackedMethodV1Dataset:
    """Immutable capability proving collector-source admission was performed.

    Instances cannot be constructed from a self-contained dataset mapping.  The
    sole constructor is :func:`build_method_v1_outcome_dataset`, which re-opens
    each decision freeze and collection attempt through collector validators and
    requires the frozen schedule to have been consumed exactly once.
    """

    _dataset: object
    evidence_sha256: str
    collection_plan_sha256: str
    scorer_verifier_auth_key_id: str
    _proposal_population_summary: Mapping[str, Any]
    freeze_receipt_sha256s: tuple[str, ...]
    source_collection_plan_path: Path
    source_freeze_dirs: tuple[Path, ...]
    source_attempt_paths: tuple[Path, ...]
    _capability: object

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError(
            "ReceiptBackedMethodV1Dataset must be created by "
            "build_method_v1_outcome_dataset"
        )

    @property
    def dataset_id(self) -> str:
        return self._validated_payload().dataset_id

    @property
    def manifests(self) -> tuple[object, ...]:
        return self._validated_payload().manifests

    @property
    def schedule(self) -> tuple[object, ...]:
        return self._validated_payload().schedule

    @property
    def attempts(self) -> tuple[object, ...]:
        return self._validated_payload().attempts

    @property
    def observed_branches(self) -> tuple[object, ...]:
        return self._validated_payload().observed_branches

    @property
    def complete(self) -> bool:
        return self._validated_payload().complete

    def fingerprint(self) -> str:
        return self._validated_payload().fingerprint()

    def summary(self) -> dict[str, Any]:
        summary = dict(self._validated_payload().summary())
        summary.update(
            {
                "receipt_backed": True,
                "schedule_fully_consumed": True,
                "admission_evidence_sha256": self.evidence_sha256,
                "collection_plan_sha256": self.collection_plan_sha256,
                "scorer_verifier_auth_key_id": self.scorer_verifier_auth_key_id,
                "proposal_population": self.proposal_population_summary,
            }
        )
        return summary

    @property
    def proposal_population_summary(self) -> dict[str, Any]:
        """Return the plan-bound proposal denominator without terminal artifacts."""

        return json.loads(
            canonical_json_bytes(self._proposal_population_summary).decode("utf-8")
        )

    def evidence_for_splits(self, splits: Iterable[str]) -> str:
        """Bind only the admitted source rows visible to one learning stage."""

        split_names = frozenset(str(value) for value in splits)
        if not split_names:
            raise ValueError("admission evidence requires at least one split")
        dataset = self._validated_payload()
        unknown = split_names - {manifest.split for manifest in dataset.manifests}
        if unknown:
            raise ValueError(
                f"admission evidence names unknown splits {sorted(unknown)}"
            )
        return _receipt_admission_evidence_sha256(
            dataset=dataset,
            freeze_receipt_sha256s=self.freeze_receipt_sha256s,
            splits=split_names,
            collection_plan_sha256=self.collection_plan_sha256,
            scorer_verifier_auth_key_id=self.scorer_verifier_auth_key_id,
        )

    def _validated_payload(self) -> Any:
        if self._capability is not _RECEIPT_ADMISSION_CAPABILITY:
            raise TypeError("dataset lacks collector receipt admission capability")
        from .method_v1_data import MethodV1OutcomeDataset

        if not isinstance(self._dataset, MethodV1OutcomeDataset):
            raise TypeError("receipt-backed dataset payload changed type")
        return self._dataset


def _new_receipt_backed_dataset(
    *,
    dataset: object,
    evidence_sha256: str,
    collection_plan_sha256: str,
    scorer_verifier_auth_key_id: str,
    proposal_population_summary: Mapping[str, Any],
    freeze_receipt_sha256s: tuple[str, ...],
    source_collection_plan_path: Path,
    source_freeze_dirs: tuple[Path, ...],
    source_attempt_paths: tuple[Path, ...],
) -> ReceiptBackedMethodV1Dataset:
    result = object.__new__(ReceiptBackedMethodV1Dataset)
    object.__setattr__(result, "_dataset", dataset)
    object.__setattr__(result, "evidence_sha256", evidence_sha256)
    object.__setattr__(result, "collection_plan_sha256", collection_plan_sha256)
    object.__setattr__(
        result, "scorer_verifier_auth_key_id", scorer_verifier_auth_key_id
    )
    object.__setattr__(
        result,
        "_proposal_population_summary",
        json.loads(canonical_json_bytes(proposal_population_summary).decode("utf-8")),
    )
    object.__setattr__(result, "freeze_receipt_sha256s", freeze_receipt_sha256s)
    object.__setattr__(
        result, "source_collection_plan_path", source_collection_plan_path
    )
    object.__setattr__(result, "source_freeze_dirs", source_freeze_dirs)
    object.__setattr__(result, "source_attempt_paths", source_attempt_paths)
    object.__setattr__(result, "_capability", _RECEIPT_ADMISSION_CAPABILITY)
    result._validated_payload()
    return result


def _collection_plan_proposal_population_summary(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe which pre-proposal resets are represented by the plan.

    Version-1 plans were frozen only after successful proposal generation, so
    they cannot establish an unconditional reset-population denominator.  The
    legacy path remains readable for old software artifacts, but reports its
    conditional scope instead of silently treating proposal-covered groups as
    the whole study population.
    """

    raw_population = document.get("proposal_population")
    if raw_population is None:
        return {
            "schema_version": "method-v1-proposal-population-summary-v1",
            "preproposal_inventory": False,
            "population_denominator_available": False,
            "metric_scope": "POST_PROPOSAL_CONDITIONAL_LEGACY_PLAN",
            "reset_inventory_sha256": None,
            "overall": None,
            "by_split": None,
        }
    if not isinstance(raw_population, Mapping):
        raise TypeError("collection-plan proposal population must be a mapping")
    raw_rows = raw_population.get("terminal_rows")
    if not isinstance(raw_rows, list):
        raise TypeError("collection-plan proposal terminal rows must be a list")

    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        population = len(rows)
        proposal_failures = sum(
            row["terminal_status"] == "PROPOSAL_FAILURE" for row in rows
        )
        choice_eligible = sum(
            row["terminal_status"] == "VALID_CHOICE_SET" for row in rows
        )
        single_primitive = sum(
            row["terminal_status"] == "VALID_SINGLE_PRIMITIVE_SET" for row in rows
        )
        proposal_covered = choice_eligible + single_primitive
        return {
            "population_group_count": population,
            "proposal_covered_group_count": proposal_covered,
            "choice_eligible_group_count": choice_eligible,
            "single_primitive_group_count": single_primitive,
            "proposal_failure_group_count": proposal_failures,
            "proposal_coverage": (
                proposal_covered / population if population else None
            ),
            "choice_eligibility_rate": (
                choice_eligible / population if population else None
            ),
        }

    rows = tuple(dict(row) for row in raw_rows)
    overall = summarize(rows)
    expected_overall = {
        "population_group_count": raw_population.get("population_group_count"),
        "proposal_covered_group_count": raw_population.get(
            "proposal_covered_group_count"
        ),
        "choice_eligible_group_count": raw_population.get(
            "choice_eligible_group_count"
        ),
    }
    if any(overall[name] != value for name, value in expected_overall.items()):
        raise ValueError("collection-plan proposal population aggregate drifted")
    by_split = {
        split: summarize(tuple(row for row in rows if row["split"] == split))
        for split in ("train", "validation", "calibration", "test")
    }
    return {
        "schema_version": "method-v1-proposal-population-summary-v1",
        "preproposal_inventory": True,
        "population_denominator_available": True,
        "metric_scope": "FULL_PREPROPOSAL_RESET_POPULATION",
        "reset_inventory_sha256": raw_population["reset_inventory_sha256"],
        "overall": overall,
        "by_split": by_split,
    }


def _receipt_admission_evidence_sha256(
    *,
    dataset: object,
    freeze_receipt_sha256s: Sequence[str],
    splits: Iterable[str],
    collection_plan_sha256: str,
    scorer_verifier_auth_key_id: str,
) -> str:
    from .method_v1_data import MethodV1OutcomeDataset

    if not isinstance(dataset, MethodV1OutcomeDataset):
        raise TypeError("admission evidence requires a MethodV1OutcomeDataset")
    selected_splits = frozenset(str(value) for value in splits)
    if not selected_splits:
        raise ValueError("admission evidence requires at least one split")
    receipts = tuple(freeze_receipt_sha256s)
    if len(receipts) != len(dataset.manifests):
        raise ValueError("freeze receipt coverage differs from decision manifests")
    manifest_receipts = tuple(zip(dataset.manifests, receipts, strict=True))
    selected_manifest_ids = {
        manifest.manifest_id
        for manifest in dataset.manifests
        if manifest.split in selected_splits
    }
    return canonical_sha256(
        {
            "schema_version": "method-v1-receipt-admission-v3",
            "dataset_id": dataset.dataset_id,
            "collection_plan_sha256": _sha256(
                collection_plan_sha256, name="collection plan SHA-256"
            ),
            "scorer_verifier_auth_key_id": _sha256(
                scorer_verifier_auth_key_id,
                name="scorer verifier authentication key ID",
            ),
            "splits": sorted(selected_splits),
            "manifests": sorted(
                (
                    {
                        "manifest_id": manifest.manifest_id,
                        "manifest_sha256": manifest.fingerprint(),
                        "freeze_receipt_sha256": receipt,
                    }
                    for manifest, receipt in manifest_receipts
                    if manifest.manifest_id in selected_manifest_ids
                ),
                key=lambda item: str(item["manifest_id"]),
            ),
            "attempts": sorted(
                (
                    {
                        "attempt_id": attempt.attempt_id,
                        "schedule_entry_id": attempt.schedule_entry.entry_id,
                        "status": attempt.status.value,
                        "artifact_tree_sha256": attempt.artifact_tree_sha256,
                    }
                    for attempt in dataset.attempts
                    if attempt.schedule_entry.manifest_id in selected_manifest_ids
                ),
                key=lambda item: str(item["schedule_entry_id"]),
            ),
            "schedule_fully_consumed": True,
        }
    )


def require_receipt_backed_dataset(dataset: object) -> Any:
    """Return the canonical payload only after checking admission capability."""

    if not isinstance(dataset, ReceiptBackedMethodV1Dataset):
        raise TypeError(
            "canonical Method-V1 training/evaluation requires a "
            "ReceiptBackedMethodV1Dataset admitted from --freeze-dir and --attempt"
        )
    return dataset._validated_payload()


def build_method_v1_outcome_dataset(
    *,
    collection_plan_path: str | Path,
    collection_config_path: str | Path,
    freeze_dirs: Sequence[str | Path],
    attempt_paths: Sequence[str | Path],
    dataset_id: str,
    require_complete: bool = False,
    admitted_splits: Iterable[str] | None = None,
) -> ReceiptBackedMethodV1Dataset:
    """Reconstruct canonical data directly from collector-owned artifacts.

    The global collection plan and all pre-outcome freezes are always verified.
    When ``admitted_splits`` is omitted, every attempt artifact tree, evaluator
    sidecar, and public execution trace is then re-opened through collector-owned
    validators.  A learning stage may instead name an explicit split subset.  In
    that mode only attempt paths from those splits are accepted or opened, which
    lets training admit train/validation outcomes without reading calibration or
    test labels.  Every schedule entry *within the admitted scope* must still be
    represented by exactly one consumed attempt; infrastructure failures remain
    unlabelled and cannot be silently omitted.
    """

    from .method_v1_data import MethodV1OutcomeDataset

    roots = tuple(Path(path).expanduser().resolve() for path in freeze_dirs)
    attempts_sources = tuple(
        Path(path).expanduser().resolve() for path in attempt_paths
    )
    if not roots:
        raise ValueError("at least one freeze directory is required")
    if not attempts_sources:
        raise ValueError("at least one collection attempt is required")
    if len(set(roots)) != len(roots):
        raise ValueError("freeze directories must be unique")
    if len(set(attempts_sources)) != len(attempts_sources):
        raise ValueError("collection attempt paths must be unique")
    from .collect_outcomes import (
        load_verified_collection_attempt,
        load_verified_collection_plan,
        load_verified_execution_claim,
    )

    collection_plan = load_verified_collection_plan(
        collection_plan_path,
        freeze_dirs=roots,
        config_path=collection_config_path,
    )
    all_freezes = collection_plan.freezes
    if admitted_splits is None:
        split_scope = frozenset(item.manifest.split for item in all_freezes)
    else:
        normalized_splits = tuple(
            str(value).strip().lower() for value in admitted_splits
        )
        if not normalized_splits:
            raise ValueError("admitted_splits must be non-empty when provided")
        if len(set(normalized_splits)) != len(normalized_splits):
            raise ValueError("admitted_splits must not contain duplicates")
        split_scope = frozenset(normalized_splits)
        if split_scope - _ALLOWED_SPLITS:
            raise ValueError(
                f"admitted_splits must be drawn from {sorted(_ALLOWED_SPLITS)}"
            )
    freezes = tuple(item for item in all_freezes if item.manifest.split in split_scope)
    if not freezes:
        raise ValueError("collection plan has no decision freezes in admitted_splits")
    manifests = tuple(item.manifest for item in freezes)
    schedule = tuple(entry for item in freezes for entry in item.schedule)
    admitted_entry_ids = {entry.entry_id for entry in schedule}
    # Fail before opening a path whose contract-level entry identity belongs to
    # an unadmitted split. The receipt and execution-claim checks below still
    # establish the exact semantic identity of every accepted artifact.
    for source in attempts_sources:
        if (
            source.name != "attempt.json"
            or source.parent.name not in admitted_entry_ids
        ):
            raise ValueError(
                "attempt path names a schedule entry outside admitted_splits"
            )
    attempts = tuple(
        load_verified_collection_attempt(path) for path in attempts_sources
    )
    freeze_by_manifest = {item.manifest.manifest_id: item for item in freezes}
    for attempt, attempt_path in zip(attempts, attempts_sources, strict=True):
        freeze = freeze_by_manifest.get(attempt.schedule_entry.manifest_id)
        if freeze is None:
            raise ValueError("attempt manifest is outside the global collection plan")
        claim = load_verified_execution_claim(
            freeze_dir=freeze.freeze_dir,
            entry=attempt.schedule_entry,
            collection_plan_sha256=collection_plan.collection_plan_sha256,
            scorer_verifier_auth_key_id=(collection_plan.scorer_verifier_auth_key_id),
        )
        expected_attempt_path = (
            Path(str(claim["output_root"])).expanduser().resolve()
            / attempt.schedule_entry.entry_id
            / "attempt.json"
        )
        if attempt_path != expected_attempt_path:
            raise ValueError(
                "attempt path differs from its collection-plan-bound execution claim"
            )
        started_path = attempt_path.parent / "started.json"
        if started_path.is_file():
            started = _read_json_mapping(started_path)
            for name in (
                "collection_plan_sha256",
                "scorer_verifier_auth_key_id",
                "selection_sha256",
            ):
                if started.get(name) != claim[name]:
                    raise ValueError(
                        f"started artifact differs from execution claim {name}"
                    )
    dataset = MethodV1OutcomeDataset(
        dataset_id=dataset_id,
        manifests=manifests,
        schedule=schedule,
        attempts=attempts,
        complete=require_complete,
    )
    scheduled_ids = {item.entry_id for item in dataset.schedule}
    consumed_ids = {item.schedule_entry.entry_id for item in dataset.attempts}
    if consumed_ids != scheduled_ids:
        missing = sorted(scheduled_ids - consumed_ids)
        extra = sorted(consumed_ids - scheduled_ids)
        raise ValueError(
            "receipt-backed admission requires every frozen schedule entry to be "
            f"consumed exactly once; missing={missing}, extra={extra}"
        )
    receipt_sha256s = tuple(item.freeze_receipt_sha256 for item in freezes)
    evidence_sha256 = _receipt_admission_evidence_sha256(
        dataset=dataset,
        freeze_receipt_sha256s=receipt_sha256s,
        splits={manifest.split for manifest in manifests},
        collection_plan_sha256=collection_plan.collection_plan_sha256,
        scorer_verifier_auth_key_id=(collection_plan.scorer_verifier_auth_key_id),
    )
    return _new_receipt_backed_dataset(
        dataset=dataset,
        evidence_sha256=evidence_sha256,
        collection_plan_sha256=collection_plan.collection_plan_sha256,
        scorer_verifier_auth_key_id=collection_plan.scorer_verifier_auth_key_id,
        proposal_population_summary=_collection_plan_proposal_population_summary(
            collection_plan.document
        ),
        freeze_receipt_sha256s=receipt_sha256s,
        source_collection_plan_path=collection_plan.plan_path,
        source_freeze_dirs=roots,
        source_attempt_paths=attempts_sources,
    )


def load_qwen_token_fields(
    manifests: Sequence[object],
    *,
    cache_root: str | Path,
) -> dict[str, CandidateTokenField]:
    """Load concrete Qwen caches bound by canonical decision manifests."""

    _require_torch()
    from .method_v1_data import DecisionGroupManifest
    from .qwen_provider import QwenFeatureCache

    cache = QwenFeatureCache(cache_root)
    result: dict[str, CandidateTokenField] = {}
    for manifest in manifests:
        if not isinstance(manifest, DecisionGroupManifest):
            raise TypeError("manifests must contain DecisionGroupManifest values")
        field = cache.load(
            manifest.token_cache_sha256,
            context=manifest.context,
            candidates=manifest.candidates,
            provider_id=manifest.token_provider_id,
        )
        if field is None:
            raise FileNotFoundError(
                f"missing Qwen feature cache {manifest.token_cache_sha256}"
            )
        result[manifest.token_cache_sha256] = field
    return result


def load_qwen_token_fields_from_roots(
    manifests: Sequence[object],
    *,
    cache_roots: Sequence[str | Path],
) -> dict[str, CandidateTokenField]:
    """Resolve each manifest against concrete ``QwenFeatureCache`` roots.

    Collector output normally places one cache at ``<freeze>/feature_cache``.
    A consolidated content-addressed cache is also supported.  Finding the
    same cache key in multiple roots is rejected instead of choosing one by
    path order, which keeps the training identity unambiguous.
    """

    _require_torch()
    from .method_v1_data import DecisionGroupManifest

    roots = tuple(Path(root).expanduser().resolve() for root in cache_roots)
    if not roots:
        raise ValueError("at least one Qwen feature-cache root is required")
    result: dict[str, CandidateTokenField] = {}
    for manifest in manifests:
        if not isinstance(manifest, DecisionGroupManifest):
            raise TypeError("manifests must contain DecisionGroupManifest values")
        key = manifest.token_cache_sha256
        matches = [
            root
            for root in roots
            if (root / "sha256" / key[:2] / f"{key}.pt").is_file()
            or (root / "sha256" / key[:2] / f"{key}.json").is_file()
        ]
        if not matches:
            raise FileNotFoundError(f"missing Qwen feature cache {key}")
        if len(matches) != 1:
            raise ValueError(
                f"Qwen feature cache {key} appears in multiple declared roots"
            )
        field = load_qwen_token_fields((manifest,), cache_root=matches[0])[key]
        result.setdefault(key, field)
    return result


def method_v1_training_records(
    dataset: object,
    *,
    cache_roots: Sequence[str | Path],
    splits: Iterable[str] = ("train", "validation"),
) -> tuple[EncodedOutcomeRecord, ...]:
    """Join canonical labels to their frozen, pre-decision Qwen features."""

    dataset = require_receipt_backed_dataset(dataset)
    allowed_splits = frozenset(str(item) for item in splits)
    if not allowed_splits or allowed_splits - {"train", "validation"}:
        raise ValueError("training records may load only train and validation splits")
    selected_manifests = tuple(
        manifest for manifest in dataset.manifests if manifest.split in allowed_splits
    )
    if not selected_manifests:
        raise ValueError("dataset has no manifests in the requested training splits")
    fields = load_qwen_token_fields_from_roots(
        selected_manifests, cache_roots=cache_roots
    )
    return encoded_records_from_method_v1_dataset(
        dataset, token_fields_by_sha256=fields, splits=allowed_splits
    )


def method_v1_training_dataset_identity(
    dataset: object,
    *,
    cache_roots: Sequence[str | Path],
) -> str:
    """Recompute the exact train/validation identity bound into checkpoints.

    Formal evaluation uses this to reject a checkpoint trained from a different
    dataset even when its feature-provider and outcome-contract names happen to
    match the held-out collection.
    """

    return _dataset_identity(
        method_v1_training_records(dataset, cache_roots=cache_roots)
    )


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """Frozen defaults for supervised Monte-Carlo outcome learning."""

    optimizer: str = "AdamW"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    effective_batch_size: int = 64
    micro_batch_size: int = 8
    gradient_clip_norm: float = 1.0
    max_epochs: int = 30
    early_stopping_metric: str = "validation_nll"
    early_stopping_patience: int = 5
    hidden_dim: int = 256
    num_heads: int = 4
    feedforward_dim: int = 1024
    use_public_state: bool = True
    seeds: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        if self.optimizer != "AdamW":
            raise ValueError("Method-V1 freezes optimizer to AdamW")
        if self.early_stopping_metric != "validation_nll":
            raise ValueError("Method-V1 early stopping metric must be validation_nll")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        for name in (
            "effective_batch_size",
            "micro_batch_size",
            "max_epochs",
            "early_stopping_patience",
            "hidden_dim",
            "num_heads",
            "feedforward_dim",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.effective_batch_size % self.micro_batch_size:
            raise ValueError(
                "effective_batch_size must be divisible by micro_batch_size"
            )
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not isinstance(self.use_public_state, bool):
            raise TypeError("use_public_state must be a bool")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive and finite")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty unique tuple")
        if any(
            not isinstance(seed, int) or isinstance(seed, bool) for seed in self.seeds
        ):
            raise TypeError("all seeds must be integers")

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.effective_batch_size // self.micro_batch_size

    def to_dict(self) -> dict[str, object]:
        value = dataclasses.asdict(self)
        value["seeds"] = list(self.seeds)
        return value


@dataclasses.dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_nll: float
    validation_nll: float
    mean_gradient_norm: float


@dataclasses.dataclass(frozen=True)
class TrainingResult:
    seed: int
    checkpoint_path: str
    checkpoint_sha256: str
    identity_path: str
    identity_sha256: str
    identity_file_sha256: str
    best_epoch: int
    best_validation_nll: float
    trainable_parameter_count: int
    epochs_completed: int
    stopped_early: bool
    history: tuple[EpochMetrics, ...]


def _seed_everything(seed: int) -> None:
    _require_torch()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _assert_training_partitions(records: Sequence[EncodedOutcomeRecord]) -> None:
    validate_split_groups(records)
    observed_splits = {record.split for record in records}
    if not {"train", "validation"}.issubset(observed_splits):
        raise ValueError("training requires non-empty train and validation splits")
    if observed_splits - {"train", "validation"}:
        raise ValueError(
            "training entry point accepts only train/validation records; "
            "calibration and test must be loaded separately"
        )
    provider_ids = {record.provider_id for record in records}
    contracts = {record.outcome_contract_sha256 for record in records}
    experiment_ids = {record.experiment_id for record in records}
    proposal_provider_ids = {record.proposal_provider_id for record in records}
    configurations = {record.configuration_sha256 for record in records}
    if len(provider_ids) != 1:
        raise ValueError("training cannot mix frozen feature providers")
    if len(contracts) != 1:
        raise ValueError("training cannot mix outcome contracts")
    if len(experiment_ids) != 1:
        raise ValueError("training cannot mix experiment identities")
    if len(proposal_provider_ids) != 1:
        raise ValueError("training cannot mix proposal providers")
    if len(configurations) != 1:
        raise ValueError("training cannot mix resolved Method-V1 configurations")


def _dataset_identity(records: Sequence[EncodedOutcomeRecord]) -> str:
    return canonical_sha256(
        {
            "schema_version": "method-v1-training-dataset-identity-v1",
            "records": [
                {
                    "record_id": record.record_id,
                    "experiment_id": record.experiment_id,
                    "initial_state_group": record.initial_state_group,
                    "decision_group_id": record.decision_group_id,
                    "split_group_id": record.split_group_id,
                    "split": record.split,
                    "branch_fingerprint": record.branch_fingerprint,
                    "proposal_provider_id": record.proposal_provider_id,
                    "configuration_sha256": record.configuration_sha256,
                    "token_cache_sha256": record.token_cache_sha256,
                    "executed_candidate_fingerprint": (
                        record.executed_candidate_fingerprint
                    ),
                    "observed_outcome": record.observed_outcome,
                }
                for record in sorted(records, key=lambda item: item.record_id)
            ],
        }
    )


def _mean_validation_nll(
    model: GroundedOutcomeModel,
    loader: DataLoader,
    *,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            moved = batch.to(device)
            losses = executed_candidate_bernoulli_nll(
                model(moved.field).task_success_logits,
                moved.observed_outcomes,
                moved.executed_mask,
                reduction="none",
            )
            total += float(losses.detach().sum().cpu())
            count += int(losses.numel())
    if count == 0:
        raise ValueError("validation loader produced no observed outcomes")
    return total / count


def train_one_seed(
    records: Sequence[EncodedOutcomeRecord],
    *,
    output_dir: str | Path,
    config: TrainingConfig,
    seed: int,
    device: str | torch.device = "cpu",
    training_admission_evidence_sha256: str | None = None,
    collection_plan_sha256: str | None = None,
    scorer_verifier_auth_key_id: str | None = None,
) -> TrainingResult:
    """Train one scorer and bind any collector admission into its identity.

    Direct unit-level callers may omit ``training_admission_evidence_sha256``.
    Such a checkpoint remains useful for diagnostics, but formal evaluation
    rejects it because it cannot prove where its supervision came from.
    """

    _require_torch()
    rows = tuple(records)
    _assert_training_partitions(rows)
    if training_admission_evidence_sha256 is not None:
        training_admission_evidence_sha256 = _sha256(
            training_admission_evidence_sha256,
            name="training admission evidence SHA-256",
        )
    if (collection_plan_sha256 is None) != (scorer_verifier_auth_key_id is None):
        raise ValueError(
            "collection plan SHA-256 and scorer verifier key ID must be supplied together"
        )
    if collection_plan_sha256 is not None:
        collection_plan_sha256 = _sha256(
            collection_plan_sha256, name="collection plan SHA-256"
        )
        scorer_verifier_auth_key_id = _sha256(
            scorer_verifier_auth_key_id,
            name="scorer verifier authentication key ID",
        )
    if seed not in config.seeds:
        raise ValueError("seed is not part of the frozen training configuration")
    train_rows = tuple(record for record in rows if record.split == "train")
    validation_rows = tuple(record for record in rows if record.split == "validation")
    train_dataset = EncodedOutcomeDataset(train_rows)
    validation_dataset = EncodedOutcomeDataset(validation_rows)

    _seed_everything(seed)
    device_value = torch.device(device)
    sample = rows[0].field
    use_public_state = config.use_public_state
    if use_public_state:
        invalid_state_records = [
            record.record_id
            for record in rows
            if record.field.public_state_values is None
            or record.field.public_state_valid_mask is None
            or not bool(record.field.public_state_valid_mask[0])
        ]
        if invalid_state_records:
            raise ValueError(
                "public-state-enabled training requires valid 9-D state for every "
                f"train/validation record; missing={invalid_state_records[:3]}"
            )
    model = GroundedOutcomeModel(
        context_dim=sample.public_context.channel_count,
        candidate_dim=int(sample.tokens.shape[2]),
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        feedforward_dim=config.feedforward_dim,
        use_public_state=use_public_state,
        public_state_dim=9,
    ).to(device_value)
    state_statistics: dict[str, list[float]] | None = None
    if use_public_state:
        state_by_reset: dict[str, Tensor] = {}
        for record in train_rows:
            state = record.field.public_state_values[0].detach().cpu()
            previous = state_by_reset.setdefault(record.initial_state_group, state)
            if not torch.equal(previous, state):
                raise ValueError(
                    "public state changes within one exact initial_state_group"
                )
        observed_states = torch.stack(
            [state_by_reset[key] for key in sorted(state_by_reset)]
        )
        mean = observed_states.mean(dim=0)
        scale = observed_states.std(dim=0, unbiased=False)
        scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
        model.set_public_state_statistics(
            mean=mean.to(device_value), scale=scale.to(device_value)
        )
        state_statistics = {
            "mean": [float(value) for value in mean],
            "scale": [float(value) for value in scale],
        }
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    sampler = GroupUniformSampler(train_rows, seed=seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.micro_batch_size,
        sampler=sampler,
        collate_fn=collate_outcome_records,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,
        collate_fn=collate_outcome_records,
        num_workers=0,
    )

    best_state: dict[str, Tensor] | None = None
    best_validation = math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[EpochMetrics] = []
    accumulation = config.gradient_accumulation_steps

    for epoch in range(config.max_epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_total = 0.0
        train_count = 0
        gradient_norms: list[float] = []
        batch_count = len(train_loader)
        accumulated_examples = 0
        for batch_index, batch in enumerate(train_loader):
            moved = batch.to(device_value)
            prediction = model(moved.field)
            raw_loss = moved.loss(prediction)
            batch_examples = moved.field.batch_size
            (raw_loss * batch_examples / config.effective_batch_size).backward()
            train_total += float(raw_loss.detach().cpu()) * moved.field.batch_size
            train_count += moved.field.batch_size
            accumulated_examples += batch_examples
            update_due = (batch_index + 1) % accumulation == 0 or (
                batch_index + 1 == batch_count
            )
            if update_due:
                if accumulated_examples != config.effective_batch_size:
                    correction = config.effective_batch_size / accumulated_examples
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )
                gradient_norms.append(float(norm.detach().cpu()))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_examples = 0

        if train_count == 0:
            raise ValueError("training loader produced no observed outcomes")
        validation_nll = _mean_validation_nll(
            model, validation_loader, device=device_value
        )
        history.append(
            EpochMetrics(
                epoch=epoch,
                train_nll=train_total / train_count,
                validation_nll=validation_nll,
                mean_gradient_norm=(
                    sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0
                ),
            )
        )
        if validation_nll < best_validation:
            best_validation = validation_nll
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    if best_state is None or best_epoch < 0:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    provider_id = rows[0].provider_id
    contract_sha = rows[0].outcome_contract_sha256
    dataset_sha = _dataset_identity(rows)
    checkpoint_identity = {
        "schema_version": _CHECKPOINT_SCHEMA,
        "seed": seed,
        "experiment_id": next(iter({row.experiment_id for row in rows})),
        "configuration_sha256": next(iter({row.configuration_sha256 for row in rows})),
        "proposal_provider_id": next(iter({row.proposal_provider_id for row in rows})),
        "provider_id": provider_id,
        "outcome_contract_sha256": contract_sha,
        "training_dataset_sha256": dataset_sha,
        "training_admission_evidence_sha256": training_admission_evidence_sha256,
        "collection_plan_sha256": collection_plan_sha256,
        "scorer_verifier_auth_key_id": scorer_verifier_auth_key_id,
        "training_split_group_ids": sorted({row.split_group_id for row in rows}),
        "model": {
            "class": "grounded_interaction.model.GroundedOutcomeModel",
            "context_dim": sample.public_context.channel_count,
            "candidate_dim": int(sample.tokens.shape[2]),
            "hidden_dim": config.hidden_dim,
            "num_heads": config.num_heads,
            "feedforward_dim": config.feedforward_dim,
            "use_public_state": use_public_state,
            "public_state_dim": 9,
            "public_state_train_statistics": state_statistics,
        },
        "training": config.to_dict(),
        "best_epoch": best_epoch,
        "best_validation_nll": best_validation,
        "trainable_parameter_count": parameter_count,
    }
    identity_sha = canonical_sha256(checkpoint_identity)
    checkpoint_payload = {
        "schema_version": _CHECKPOINT_SCHEMA,
        "identity": checkpoint_identity,
        "identity_sha256": identity_sha,
        "model_state_dict": best_state,
        "history": [dataclasses.asdict(item) for item in history],
    }
    destination = Path(output_dir).expanduser().resolve()
    checkpoint_path = destination / f"seed_{seed}" / "best.pt"
    identity_path = destination / f"seed_{seed}" / "identity.json"
    _atomic_torch_save(checkpoint_payload, checkpoint_path)
    _atomic_json_dump(
        {**checkpoint_identity, "identity_sha256": identity_sha}, identity_path
    )
    return TrainingResult(
        seed=seed,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=_file_sha256(checkpoint_path),
        identity_path=str(identity_path),
        identity_sha256=identity_sha,
        identity_file_sha256=_file_sha256(identity_path),
        best_epoch=best_epoch,
        best_validation_nll=best_validation,
        trainable_parameter_count=parameter_count,
        epochs_completed=len(history),
        stopped_early=len(history) < config.max_epochs,
        history=tuple(history),
    )


def train_all_seeds(
    records: Sequence[EncodedOutcomeRecord],
    *,
    output_dir: str | Path,
    config: TrainingConfig | None = None,
    device: str | torch.device = "cpu",
    training_admission_evidence_sha256: str | None = None,
    collection_plan_sha256: str | None = None,
    scorer_verifier_auth_key_id: str | None = None,
) -> tuple[TrainingResult, ...]:
    """Train the three configured seeds as independent deployed models."""

    if config is None:
        config = TrainingConfig()
    return tuple(
        train_one_seed(
            records,
            output_dir=output_dir,
            config=config,
            seed=seed,
            device=device,
            training_admission_evidence_sha256=training_admission_evidence_sha256,
            collection_plan_sha256=collection_plan_sha256,
            scorer_verifier_auth_key_id=scorer_verifier_auth_key_id,
        )
        for seed in config.seeds
    )


def load_checkpoint_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[GroundedOutcomeModel, Mapping[str, object]]:
    """Reconstruct an outcome model from a hash-independent local checkpoint."""

    _require_torch()
    payload = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location=device,
        weights_only=True,
    )
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != _CHECKPOINT_SCHEMA
    ):
        raise ValueError("unsupported outcome checkpoint schema")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise TypeError("checkpoint identity is missing")
    expected_identity_sha = _sha256(
        payload.get("identity_sha256"), name="checkpoint identity SHA-256"
    )
    if canonical_sha256(identity) != expected_identity_sha:
        raise ValueError("checkpoint identity SHA-256 mismatch")
    model_config = identity.get("model")
    if not isinstance(model_config, Mapping):
        raise TypeError("checkpoint model identity is missing")
    model = GroundedOutcomeModel(
        context_dim=int(model_config["context_dim"]),
        candidate_dim=int(model_config["candidate_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_heads=int(model_config["num_heads"]),
        feedforward_dim=int(model_config["feedforward_dim"]),
        use_public_state=bool(model_config.get("use_public_state", False)),
        public_state_dim=int(model_config.get("public_state_dim", 9)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, identity


def _candidate_field_to_device(
    field: CandidateTokenField, device: str | torch.device
) -> CandidateTokenField:
    """Move only detached policy-visible tensors to the scorer device."""

    _require_torch()
    context = dataclasses.replace(
        field.public_context,
        tokens=field.public_context.tokens.to(device),
        valid_mask=field.public_context.valid_mask.to(device),
        current_patch_mask=field.public_context.current_patch_mask.to(device),
        patch_xyxy=field.public_context.patch_xyxy.to(device),
    )
    return dataclasses.replace(
        field,
        tokens=field.tokens.to(device),
        valid_mask=field.valid_mask.to(device),
        grounding_support=field.grounding_support.to(device),
        public_context=context,
        public_state_values=field.public_state_values.to(device),
        public_state_valid_mask=field.public_state_valid_mask.to(device),
    )


@dataclasses.dataclass(frozen=True)
class ScorerSelectionResult:
    """Identity-bound scorer output before any physical execution."""

    manifest_sha256: str
    token_cache_sha256: str
    checkpoint_sha256: str
    checkpoint_identity_sha256: str
    outcome_contract_sha256: str
    temperature: float
    predictions: tuple[ValuePrediction, ...]
    decision: SelectionDecision

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "method-v1-scorer-selection-v1",
            "evidence_scope": "PRE_EXECUTION_MODEL_DECISION",
            "manifest_sha256": self.manifest_sha256,
            "token_cache_sha256": self.token_cache_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_identity_sha256": self.checkpoint_identity_sha256,
            "outcome_contract_sha256": self.outcome_contract_sha256,
            "temperature": self.temperature,
            "predictions": [item.to_dict() for item in self.predictions],
            "decision": self.decision.to_dict(),
        }


def select_manifest_candidate(
    manifest: object,
    *,
    cache_root: str | Path,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    temperature: float = 1.0,
    feasibility: Mapping[str, bool] | None = None,
) -> ScorerSelectionResult:
    """Score and select one candidate from a frozen public decision manifest.

    The only model inputs are the manifest-bound, pre-decision Qwen cache and
    its public state.  Outcome labels, post-action frames, simulator semantic
    IDs, and evaluator sidecars are neither accepted nor loaded here.
    """

    _require_torch()
    from .method_v1_data import DecisionGroupManifest

    if not isinstance(manifest, DecisionGroupManifest):
        raise TypeError("manifest must be a DecisionGroupManifest")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    candidate_ids = tuple(candidate.candidate_id for candidate in manifest.candidates)
    if feasibility is None:
        feasible_by_id = {candidate_id: True for candidate_id in candidate_ids}
    else:
        if set(feasibility) != set(candidate_ids):
            raise ValueError("feasibility keys must match manifest candidates exactly")
        if any(not isinstance(value, bool) for value in feasibility.values()):
            raise TypeError("feasibility values must be bool")
        feasible_by_id = dict(feasibility)

    field = load_qwen_token_fields((manifest,), cache_root=cache_root)[
        manifest.token_cache_sha256
    ]
    model, checkpoint_identity = load_checkpoint_model(checkpoint_path, device=device)
    if checkpoint_identity.get("provider_id") != manifest.token_provider_id:
        raise ValueError("checkpoint provider identity differs from manifest")
    contract_sha256 = manifest.outcome_contract.fingerprint()
    if checkpoint_identity.get("outcome_contract_sha256") != contract_sha256:
        raise ValueError("checkpoint outcome contract differs from manifest")
    expected_identity = {
        "experiment_id": manifest.experiment_id,
        "configuration_sha256": manifest.configuration_sha256,
        "proposal_provider_id": manifest.proposal_provider_id,
    }
    for name, expected in expected_identity.items():
        if checkpoint_identity.get(name) != expected:
            raise ValueError(f"checkpoint {name} differs from manifest")
    training_groups = checkpoint_identity.get("training_split_group_ids")
    if (
        not isinstance(training_groups, list)
        or not training_groups
        or any(
            not isinstance(value, str) or not value.strip() for value in training_groups
        )
        or len(set(training_groups)) != len(training_groups)
    ):
        raise ValueError("checkpoint training split-group identity is invalid")
    if manifest.split_group_id in training_groups:
        raise ValueError(
            "deployment manifest split group overlaps checkpoint training data"
        )

    moved = _candidate_field_to_device(field, device)
    with torch.no_grad():
        output = model(moved)
        probabilities = (
            torch.sigmoid(output.task_success_logits / temperature).detach().cpu()[0]
        )
    expected_fingerprints = tuple(
        candidate.fingerprint() for candidate in manifest.candidates
    )
    if output.context_fingerprints != (manifest.context.fingerprint(),):
        raise ValueError("scorer output changed the manifest context identity")
    if output.candidate_fingerprints != (expected_fingerprints,):
        raise ValueError("scorer output changed candidate identities")
    predictions = tuple(
        ValuePrediction(
            candidate_id=candidate.candidate_id,
            candidate_fingerprint=candidate.fingerprint(),
            success_probability=float(probabilities[index]),
            feasible=feasible_by_id[candidate.candidate_id],
        )
        for index, candidate in enumerate(manifest.candidates)
    )
    decision = ExpectedSuccessSelector().select(manifest.candidates, predictions)
    checkpoint_source = Path(checkpoint_path).expanduser().resolve()
    return ScorerSelectionResult(
        manifest_sha256=manifest.fingerprint(),
        token_cache_sha256=manifest.token_cache_sha256,
        checkpoint_sha256=_file_sha256(checkpoint_source),
        checkpoint_identity_sha256=canonical_sha256(checkpoint_identity),
        outcome_contract_sha256=contract_sha256,
        temperature=temperature,
        predictions=predictions,
        decision=decision,
    )


@dataclasses.dataclass(frozen=True)
class TemperatureFit:
    """One positive scalar fitted on a disjoint calibration split."""

    temperature: float
    raw_nll: float
    calibrated_nll: float
    calibration_group_count: int
    calibration_example_count: int


def _binary_nll(logits: Tensor, labels: Tensor) -> Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)


def fit_positive_temperature(
    logits: Tensor,
    labels: Tensor,
    *,
    calibration_group_ids: Sequence[str],
    training_split_group_ids: Iterable[str],
    max_iterations: int = 50,
) -> TemperatureFit:
    """Fit ``sigmoid(logit / T)`` using only independent calibration groups."""

    _require_torch()
    logits = torch.as_tensor(logits, dtype=torch.float64).flatten().detach()
    labels = torch.as_tensor(labels, dtype=torch.float64).flatten().detach()
    groups = tuple(str(value) for value in calibration_group_ids)
    if (
        logits.numel() < 1
        or labels.shape != logits.shape
        or len(groups) != logits.numel()
    ):
        raise ValueError("calibration logits, labels, and group IDs must align")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(labels).all()):
        raise ValueError("calibration inputs must be finite")
    if bool(((labels != 0) & (labels != 1)).any()):
        raise ValueError("calibration labels must be binary")
    overlap = set(groups) & {str(value) for value in training_split_group_ids}
    if overlap:
        raise ValueError("calibration split groups overlap train/validation groups")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=max_iterations,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp()
        loss = _binary_nll(logits / temperature, labels)
        loss.backward()
        return loss

    raw_nll = float(_binary_nll(logits, labels))
    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp())
    calibrated_nll = float(_binary_nll(logits / temperature, labels))
    if not math.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("temperature fitting produced a non-positive value")
    return TemperatureFit(
        temperature=temperature,
        raw_nll=raw_nll,
        calibrated_nll=calibrated_nll,
        calibration_group_count=len(set(groups)),
        calibration_example_count=len(groups),
    )


def _config_from_mapping(
    value: Mapping[str, object],
    *,
    micro_batch_size: int | None = None,
) -> TrainingConfig:
    """Load either ``experiments/method_v1.yaml`` or a flat training block.

    The paper configuration nests architecture dimensions under ``model`` and
    early stopping under ``training.early_stopping``.  AdamW and validation NLL
    are method contracts, not silently replaceable CLI choices.
    """

    if "training" in value:
        training = value["training"]
        model = value.get("model", {})
        if not isinstance(training, Mapping):
            raise TypeError("training config section must be a mapping")
        if not isinstance(model, Mapping):
            raise TypeError("model config section must be a mapping")
    else:
        training = value
        model = {}
    allowed_training = {
        "optimizer",
        "learning_rate",
        "weight_decay",
        "effective_batch_size",
        "micro_batch_size",
        "gradient_clip_norm",
        "max_epochs",
        "early_stopping",
        "early_stopping_metric",
        "early_stopping_patience",
        "hidden_dim",
        "num_heads",
        "feedforward_dim",
        "seeds",
    }
    extra = set(training) - allowed_training
    if extra:
        raise ValueError(f"unknown training config keys: {sorted(extra)}")
    normalized = dict(training)

    nested_early = normalized.pop("early_stopping", None)
    if nested_early is not None:
        if not isinstance(nested_early, Mapping) or set(nested_early) != {
            "metric",
            "patience",
        }:
            raise ValueError("early_stopping must contain exactly metric and patience")
        nested_metric = str(nested_early["metric"])
        nested_patience = nested_early["patience"]
        if (
            "early_stopping_metric" in normalized
            and normalized["early_stopping_metric"] != nested_metric
        ) or (
            "early_stopping_patience" in normalized
            and normalized["early_stopping_patience"] != nested_patience
        ):
            raise ValueError("flat and nested early-stopping values disagree")
        normalized["early_stopping_metric"] = nested_metric
        normalized["early_stopping_patience"] = nested_patience

    for name in ("hidden_dim", "num_heads", "feedforward_dim"):
        if name in model:
            if name in normalized and normalized[name] != model[name]:
                raise ValueError(f"training and model config disagree on {name}")
            normalized[name] = model[name]

    public_state = model.get("public_state")
    if public_state is not None:
        if not isinstance(public_state, Mapping):
            raise TypeError("model.public_state must be a mapping")
        if set(public_state) != {
            "enabled",
            "proprio_dim",
            "include_remaining_budget_fraction",
        }:
            raise ValueError("model.public_state fields differ from Method-V1 schema")
        if public_state["proprio_dim"] != 8 or not bool(
            public_state["include_remaining_budget_fraction"]
        ):
            raise ValueError(
                "Method-V1 public state must be 8-D proprio plus budget fraction"
            )
        normalized["use_public_state"] = bool(public_state["enabled"])

    if "seeds" in normalized:
        raw_seeds = normalized["seeds"]
        if not isinstance(raw_seeds, Sequence) or isinstance(raw_seeds, str):
            raise TypeError("training seeds must be a sequence of integers")
        normalized["seeds"] = tuple(int(seed) for seed in raw_seeds)

    effective_batch_size = int(
        normalized.get("effective_batch_size", TrainingConfig.effective_batch_size)
    )
    configured_micro = normalized.get("micro_batch_size")
    if micro_batch_size is not None:
        configured_micro = micro_batch_size
    if configured_micro is None:
        # Keep the default small enough for a 16 GB development GPU while
        # retaining an exact effective batch through gradient accumulation.
        upper = min(TrainingConfig.micro_batch_size, effective_batch_size)
        configured_micro = next(
            candidate
            for candidate in range(upper, 0, -1)
            if effective_batch_size % candidate == 0
        )
    normalized["micro_batch_size"] = configured_micro
    return TrainingConfig(**normalized)


def _load_config_mapping(path: str | Path) -> Mapping[str, object]:
    source = Path(path).expanduser().resolve()
    text = source.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - base file is JSON YAML.
            raise RuntimeError(
                "non-JSON YAML training configs require the PyYAML dependency"
            ) from error
        value = yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise TypeError("training config must be a mapping")
    return value


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train Method-V1 scorers from real cached branch outcomes"
    )
    parser.add_argument(
        "--freeze-dir",
        action="append",
        required=True,
        help="collector decision-freeze directory; repeat for every group",
    )
    parser.add_argument(
        "--collection-plan",
        required=True,
        help="immutable pre-outcome global collection-plan JSON",
    )
    parser.add_argument(
        "--attempt",
        action="append",
        required=True,
        help=(
            "collector attempt.json from train/validation only; repeat for every "
            "consumed branch in those splits"
        ),
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--cache-root",
        action="append",
        help=(
            "QwenFeatureCache root; repeat when caches are not located under "
            "each --freeze-dir"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--config",
        default="experiments/method_v1.yaml",
        help="full Method-V1 JSON/YAML experiment configuration",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        help="memory-facing batch; effective batch remains frozen in config",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    config = _config_from_mapping(
        _load_config_mapping(args.config),
        micro_batch_size=args.micro_batch_size,
    )
    dataset = build_method_v1_outcome_dataset(
        collection_plan_path=args.collection_plan,
        collection_config_path=args.config,
        freeze_dirs=tuple(args.freeze_dir),
        attempt_paths=tuple(args.attempt),
        dataset_id=args.dataset_id,
        require_complete=args.require_complete,
        admitted_splits=("train", "validation"),
    )
    from .method_v1_data import validate_primitive_outcome_coverage

    outcome_coverage = validate_primitive_outcome_coverage(
        dataset._validated_payload(),
        required_splits=("train", "validation"),
    )
    cache_roots = tuple(args.cache_root or ()) or tuple(
        str(Path(path).expanduser().resolve() / "feature_cache")
        for path in args.freeze_dir
    )
    records = method_v1_training_records(dataset, cache_roots=cache_roots)
    training_admission_evidence_sha256 = dataset.evidence_for_splits(
        ("train", "validation")
    )
    results = train_all_seeds(
        records,
        output_dir=args.output_dir,
        config=config,
        device=args.device,
        training_admission_evidence_sha256=training_admission_evidence_sha256,
        collection_plan_sha256=dataset.collection_plan_sha256,
        scorer_verifier_auth_key_id=dataset.scorer_verifier_auth_key_id,
    )
    report = {
        "schema_version": "method-v1-training-run-report-v1",
        "empirical_claim": False,
        "note": (
            "Training completion is not evidence of policy improvement; held-out "
            "closed-loop evaluation is required."
        ),
        "dataset_sha256": dataset.fingerprint(),
        "admission_evidence_sha256": dataset.evidence_sha256,
        "collection_plan_sha256": dataset.collection_plan_sha256,
        "scorer_verifier_auth_key_id": dataset.scorer_verifier_auth_key_id,
        "proposal_population": dataset.proposal_population_summary,
        "training_admission_evidence_sha256": (training_admission_evidence_sha256),
        "primitive_outcome_coverage": outcome_coverage,
        "schedule_fully_consumed": True,
        "admitted_splits": ["train", "validation"],
        "outcome_evaluated_records": len(dataset.observed_branches),
        "train_validation_records": len(records),
        "training_config": config.to_dict(),
        "runs": [dataclasses.asdict(result) for result in results],
    }
    report_path = Path(args.output_dir).expanduser().resolve() / "training_report.json"
    _atomic_json_dump(report, report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI.
    raise SystemExit(_main())
