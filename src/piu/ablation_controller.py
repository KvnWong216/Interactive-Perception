"""Explicit uncalibrated B3/B4 decision semantics."""

from __future__ import annotations

import dataclasses

import numpy as np

from .calibrated_controller import INFORMATION_PRIMITIVES, TASK_PRIMITIVES, DecisionKind


@dataclasses.dataclass(frozen=True)
class UncalibratedDecision:
    kind: DecisionKind
    selected_candidate_id: str | None
    selected_candidate_primitive: str | None
    reason: str


def decide_unique_argmax(
    *,
    route_logits: np.ndarray,
    candidate_valid_mask: np.ndarray,
    candidate_id: np.ndarray,
    candidate_primitive: np.ndarray,
) -> UncalibratedDecision:
    """Select a unique maximum for diagnostic B3/B4, abstaining on exact ties."""

    logits = np.asarray(route_logits, dtype=np.float64)
    valid = np.asarray(candidate_valid_mask, dtype=bool)
    identifiers = np.asarray(candidate_id).astype(str)
    primitives = np.char.upper(np.asarray(candidate_primitive).astype(str))
    if not (
        logits.ndim == 1
        and logits.shape == valid.shape == identifiers.shape == primitives.shape
    ):
        raise ValueError("uncalibrated route arrays must be matching vectors")
    if not valid.any() or not np.isfinite(logits[valid]).all():
        raise ValueError("uncalibrated decision needs finite valid route logits")
    maximum = logits[valid].max()
    winners = np.flatnonzero(valid & (logits == maximum))
    if len(winners) != 1:
        return UncalibratedDecision(
            DecisionKind.ABSTAIN,
            None,
            None,
            "uncalibrated route logits have no unique maximum",
        )
    index = int(winners[0])
    identifier = " ".join(identifiers[index].split())
    primitive = " ".join(primitives[index].split())
    if not identifier or not primitive:
        raise ValueError("uncalibrated winner lacks candidate identity")
    if primitive in INFORMATION_PRIMITIVES:
        kind = DecisionKind.INTERACT
    elif primitive in TASK_PRIMITIVES:
        kind = DecisionKind.EXECUTE
    elif primitive == "STOP":
        kind = DecisionKind.STOP
    elif primitive == "REPORT_NOT_FOUND":
        kind = DecisionKind.REPORT_NOT_FOUND
    else:
        return UncalibratedDecision(
            DecisionKind.ABSTAIN,
            None,
            None,
            f"unregistered primitive {primitive}",
        )
    return UncalibratedDecision(
        kind,
        identifier,
        primitive,
        "unique uncalibrated route-logit maximum",
    )
