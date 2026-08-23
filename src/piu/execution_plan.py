"""Public calibrated preconditions for causal candidate execution plans."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .binding_calibration import apply_binding_calibration
from .calibrated_controller import INFORMATION_PRIMITIVES, TASK_PRIMITIVES
from .executor_bridge import current_spatial_references, serialize_pi05_subtask


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


def _members(values: np.ndarray) -> list[bool]:
    membership = np.asarray(values, dtype=bool)
    if membership.shape != (2,):
        raise ValueError("binder binary prediction-set shape is invalid")
    return [
        label
        for label, included in zip((False, True), membership, strict=True)
        if included
    ]


def calibrated_candidate_plan(
    *,
    transition: Any,
    binding: Mapping[str, Any],
    calibration: Mapping[str, Any],
    feature_report: Mapping[str, Any],
    sample_index: int,
    alpha: float,
) -> dict[str, Any]:
    """Recompute binder-only eligibility and exact serialized candidate stimuli."""

    if alpha not in [
        float(value) for value in calibration["risk_contract"]["reported_alpha"]
    ]:
        raise ValueError("execution-plan alpha was not preregistered")
    calibrated = apply_binding_calibration(binding, calibration, alpha=alpha)
    camera_names = tuple(feature_report["layout"]["camera_names"])
    references = current_spatial_references(
        prediction_set=calibrated["spatial_prediction_set"][sample_index],
        patch_xy=np.asarray(binding["patch_xy"])[sample_index],
        camera_id=np.asarray(binding["camera_id"])[sample_index],
        temporal_id=np.asarray(binding["temporal_id"])[sample_index],
        camera_names=camera_names,
    )
    context = PublicExecutionContext.create(
        task_sufficiency=_members(
            calibrated["task_sufficiency_prediction_set"][sample_index]
        ),
        target_presence=_members(
            calibrated["target_presence_prediction_set"][sample_index]
        ),
        holding_requested_target=_members(
            calibrated["holding_requested_target_prediction_set"][sample_index]
        ),
        spatial_reference_available=bool(references),
    )
    rows = []
    for raw_candidate in transition.candidate_actions:
        candidate = dict(raw_candidate)
        candidate_id = str(candidate["candidate_id"])
        primitive = str(candidate["primitive"]).upper()
        eligibility = candidate_eligibility(candidate, context)
        physical = primitive not in {"STOP", "REPORT_NOT_FOUND"}
        spatial = references if primitive in {"PICK", "DIRECT"} else ()
        subtask = (
            serialize_pi05_subtask(candidate, spatial_references=spatial)
            if physical and eligibility.eligible
            else None
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "primitive": primitive,
                "eligible_for_execution": eligibility.eligible,
                "eligibility_reason": eligibility.reason,
                "structured_pi05_subtask": subtask,
                "spatial_references": [
                    {
                        **dataclasses.asdict(item),
                        "selected_patch_indices": list(item.selected_patch_indices),
                        "x_interval": list(item.x_interval),
                        "y_interval": list(item.y_interval),
                    }
                    for item in spatial
                ],
            }
        )
    return {
        "alpha": float(alpha),
        "public_execution_context": {
            "task_sufficiency": sorted(context.task_sufficiency),
            "target_presence": sorted(context.target_presence),
            "holding_requested_target": sorted(context.holding_requested_target),
            "spatial_reference_available": context.spatial_reference_available,
        },
        "candidates": rows,
    }
