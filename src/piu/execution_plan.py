"""Public calibrated preconditions for causal candidate execution plans."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from .calibrated_controller import INFORMATION_PRIMITIVES, TASK_PRIMITIVES


def _set(values: Sequence[bool]) -> frozenset[bool]:
    if any(not isinstance(value, bool) for value in values):
        raise TypeError("candidate execution preconditions must be boolean sets")
    return frozenset(values)


@dataclasses.dataclass(frozen=True)
class PublicExecutionContext:
    task_sufficiency: frozenset[bool]
    target_presence: frozenset[bool]
    holding_requested_target: frozenset[bool]
    spatial_reference_available: bool

    @classmethod
    def create(
        cls,
        *,
        task_sufficiency: Sequence[bool],
        target_presence: Sequence[bool],
        holding_requested_target: Sequence[bool],
        spatial_reference_available: bool,
    ) -> PublicExecutionContext:
        if not isinstance(spatial_reference_available, bool):
            raise TypeError("spatial-reference availability must be boolean")
        return cls(
            task_sufficiency=_set(task_sufficiency),
            target_presence=_set(target_presence),
            holding_requested_target=_set(holding_requested_target),
            spatial_reference_available=spatial_reference_available,
        )


@dataclasses.dataclass(frozen=True)
class CandidateEligibility:
    eligible: bool
    reason: str


def candidate_eligibility(
    candidate: Mapping[str, Any], context: PublicExecutionContext
) -> CandidateEligibility:
    """Apply the same binder-only preconditions before collecting effects.

    Effect predictions are deliberately absent: this plan exists to collect
    the outcomes that will supervise them. Ineligible candidates stay in the
    route matrix with masked effects instead of being executed outside their
    qualified public context.
    """

    primitive = " ".join(str(candidate.get("primitive", "")).split()).upper()
    if primitive in {"STOP", "REPORT_NOT_FOUND"}:
        return CandidateEligibility(True, "nonphysical terminal exact-null branch")
    if primitive in INFORMATION_PRIMITIVES:
        if context.task_sufficiency != frozenset({False}):
            return CandidateEligibility(
                False,
                "information primitive lacks singleton insufficient-evidence context",
            )
        return CandidateEligibility(
            True, "singleton insufficient-evidence context permits information fork"
        )
    if primitive not in TASK_PRIMITIVES:
        raise ValueError(f"unknown candidate primitive {primitive!r}")
    if context.task_sufficiency != frozenset({True}):
        return CandidateEligibility(
            False, "task primitive lacks singleton sufficient-evidence context"
        )
    required_holding = {
        "DIRECT": frozenset({False}),
        "PICK": frozenset({False}),
        "PLACE": frozenset({True}),
    }[primitive]
    if context.holding_requested_target != required_holding:
        return CandidateEligibility(
            False, f"{primitive} lacks singleton-compatible holding context"
        )
    if primitive in {"DIRECT", "PICK"}:
        if context.target_presence != frozenset({True}):
            return CandidateEligibility(
                False, f"{primitive} lacks singleton target-presence context"
            )
        if not context.spatial_reference_available:
            return CandidateEligibility(
                False, f"{primitive} lacks a calibrated current-frame spatial set"
            )
    return CandidateEligibility(
        True, "calibrated public binder preconditions authorize an executed fork"
    )
