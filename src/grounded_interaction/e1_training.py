"""Typed training ingestion for sealed E1 execution artifacts.

This module is deliberately narrower than a feature provider.  It turns one
validated, label-bearing execution tree into the exact executed-candidate
targets consumed by the existing Bernoulli outcome loss.  Candidate IDs remain
audit metadata; tensor alignment is performed only with immutable candidate
fingerprints.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .contracts import ExecutionStatus
from .data import ObservedBranch
from .e1 import E1Plan, validate_e1_artifacts
from .losses import executed_candidate_bernoulli_nll

try:  # Keep execution-only installs free of a PyTorch dependency.
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - exercised in torch-free installs.
    torch = None
    Tensor = Any  # type: ignore[misc,assignment]


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("E1 training collation requires the learned extra")


@dataclasses.dataclass(frozen=True)
class ValidatedE1TrainingRecord:
    """One typed branch admitted only after full byte and semantic validation."""

    branch: ObservedBranch
    evidence_class: str
    artifact_manifest_sha256: str
    plan_id: str
    plan_sha256: str
    trial_id: str
    outcome_contract_sha256: str
    source_directory: str

    def __post_init__(self) -> None:
        if not isinstance(self.branch, ObservedBranch):
            raise TypeError("branch must be an ObservedBranch")
        if self.evidence_class not in {
            "LIVE_MOLMOACT2_LIBERO",
            "SOFTWARE_TEST_DOUBLE",
        }:
            raise ValueError("unsupported E1 evidence class")
        for name in (
            "artifact_manifest_sha256",
            "plan_sha256",
            "outcome_contract_sha256",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.branch.outcome_contract.fingerprint() != self.outcome_contract_sha256:
            raise ValueError("training record outcome contract fingerprint changed")
        if self.branch.execution_status is not ExecutionStatus.COMPLETED:
            raise ValueError("E1 training accepts only complete-intervention outcomes")


def load_e1_training_record(
    output_dir: str | Path,
    *,
    expected_plan: E1Plan,
    expected_trial_id: str,
    require_empirical: bool = True,
) -> ValidatedE1TrainingRecord:
    """Load one outcome only after reconstructing it from the frozen trace.

    ``require_empirical`` defaults to true.  Tests must opt in explicitly when
    exercising this loader with a software backend, so fake evidence cannot
    silently enter a training dataset.
    """

    root = Path(output_dir).expanduser().resolve()
    report = validate_e1_artifacts(
        root,
        require_empirical=require_empirical,
        expected_plan=expected_plan,
        expected_trial_id=expected_trial_id,
    )
    if report["status"] != "VERIFIED_COMPLETED":
        raise ValueError("infrastructure failures do not produce training records")
    branch = ObservedBranch.from_public_mapping(
        json.loads((root / "observed_branch.public.json").read_text(encoding="utf-8"))
    )
    return ValidatedE1TrainingRecord(
        branch=branch,
        evidence_class=str(report["evidence_class"]),
        artifact_manifest_sha256=str(report["manifest_sha256"]),
        plan_id=expected_plan.plan_id,
        plan_sha256=expected_plan.frozen_plan_sha256,
        trial_id=expected_trial_id,
        outcome_contract_sha256=branch.outcome_contract.fingerprint(),
        source_directory=str(root),
    )


@dataclasses.dataclass(frozen=True)
class E1OutcomeTargetBatch:
    """Executed-only labels aligned by candidate fingerprint, never by ID text."""

    observed_outcomes: Tensor
    executed_mask: Tensor
    candidate_fingerprints: tuple[str, ...]
    record_fingerprints: tuple[str, ...]
    context_fingerprint: str
    decision_group_id: str
    outcome_contract_sha256: str

    def __post_init__(self) -> None:
        _require_torch()
        if not isinstance(self.observed_outcomes, torch.Tensor):
            raise TypeError("observed_outcomes must be a tensor")
        if not isinstance(self.executed_mask, torch.Tensor):
            raise TypeError("executed_mask must be a tensor")
        if self.executed_mask.dtype is not torch.bool:
            raise TypeError("executed_mask must have bool dtype")
        if self.observed_outcomes.shape != self.executed_mask.shape:
            raise ValueError("E1 target tensors must share one shape")
        rows, columns = self.executed_mask.shape
        if rows < 1 or columns != len(self.candidate_fingerprints):
            raise ValueError("candidate fingerprint width does not match targets")
        if len(self.record_fingerprints) != rows:
            raise ValueError("record fingerprint count does not match targets")
        if bool((self.executed_mask.sum(dim=1) != 1).any()):
            raise ValueError("each E1 training row must select one executed candidate")

    def loss(
        self,
        outcome_logits: Tensor,
        *,
        prediction_candidate_fingerprints: Sequence[Sequence[str | None]],
    ) -> Tensor:
        """Validate model-output identity, then compute executed-only NLL."""

        _require_torch()
        if not isinstance(outcome_logits, torch.Tensor):
            raise TypeError("outcome_logits must be a tensor")
        if outcome_logits.shape != self.executed_mask.shape:
            raise ValueError("outcome logits do not match the E1 target matrix")
        expected = tuple(self.candidate_fingerprints)
        observed = tuple(tuple(row) for row in prediction_candidate_fingerprints)
        if observed != tuple(expected for _ in range(outcome_logits.shape[0])):
            raise ValueError("model candidate fingerprints do not match E1 targets")
        return executed_candidate_bernoulli_nll(
            outcome_logits,
            self.observed_outcomes.to(device=outcome_logits.device),
            self.executed_mask.to(device=outcome_logits.device),
        )


def collate_e1_outcome_targets(
    records: Sequence[ValidatedE1TrainingRecord],
) -> E1OutcomeTargetBatch:
    """Collate one reset-controlled decision group into supervised rows."""

    _require_torch()
    rows = tuple(records)
    if not rows:
        raise ValueError("E1 target collation requires at least one record")
    if any(not isinstance(row, ValidatedE1TrainingRecord) for row in rows):
        raise TypeError("E1 target collation accepts validated records only")
    decision_groups = {row.branch.decision_group_id for row in rows}
    contexts = {row.branch.context.fingerprint() for row in rows}
    contracts = {row.outcome_contract_sha256 for row in rows}
    plans = {(row.plan_id, row.plan_sha256) for row in rows}
    if not (len(decision_groups) == len(contexts) == len(contracts) == len(plans) == 1):
        raise ValueError(
            "one E1 target batch must share a decision group, context, plan, and outcome contract"
        )
    fingerprints = tuple(sorted({row.branch.candidate_fingerprint for row in rows}))
    index_by_fingerprint = {value: index for index, value in enumerate(fingerprints)}
    observed = torch.full((len(rows), len(fingerprints)), float("nan"))
    executed = torch.zeros((len(rows), len(fingerprints)), dtype=torch.bool)
    record_fingerprints: list[str] = []
    for row_index, row in enumerate(rows):
        candidate_index = index_by_fingerprint[row.branch.candidate_fingerprint]
        executed[row_index, candidate_index] = True
        observed[row_index, candidate_index] = float(row.branch.observed_outcome)
        record_fingerprints.append(row.branch.public_fingerprint())
    return E1OutcomeTargetBatch(
        observed_outcomes=observed,
        executed_mask=executed,
        candidate_fingerprints=fingerprints,
        record_fingerprints=tuple(record_fingerprints),
        context_fingerprint=next(iter(contexts)),
        decision_group_id=next(iter(decision_groups)),
        outcome_contract_sha256=next(iter(contracts)),
    )
