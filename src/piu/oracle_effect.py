"""Evaluator-only oracle-effect decisions kept outside public methods."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

from .action_effect import EffectLabel
from .calibrated_controller import INFORMATION_PRIMITIVES, TASK_PRIMITIVES, DecisionKind


@dataclasses.dataclass(frozen=True)
class OracleEffectDecision:
    kind: DecisionKind
    candidate_id: str
    primitive: str


def decide_oracle_effect(
    labels: Sequence[EffectLabel],
    candidates: Sequence[Mapping[str, object]],
) -> OracleEffectDecision:
    """Select the unique evaluator-correct branch from an exact effect matrix."""

    candidate_by_id = {str(row.get("candidate_id")): row for row in candidates}
    if len(candidate_by_id) != len(candidates):
        raise ValueError("oracle-effect public candidate IDs are not unique")
    label_by_id = {label.candidate_id: label for label in labels}
    if set(label_by_id) != set(candidate_by_id) or len(label_by_id) != len(labels):
        raise ValueError("oracle-effect labels and public candidate matrix differ")
    winners = [label for label in labels if label.selection_correct]
    if len(winners) != 1:
        raise ValueError("oracle-effect matrix requires exactly one correct branch")
    winner = winners[0]
    candidate = candidate_by_id[winner.candidate_id]
    primitive = str(candidate.get("primitive", "")).upper()
    if primitive != winner.candidate_primitive:
        raise ValueError("oracle-effect winner primitive differs from public candidate")
    if primitive in INFORMATION_PRIMITIVES:
        kind = DecisionKind.INTERACT
    elif primitive in TASK_PRIMITIVES:
        kind = DecisionKind.EXECUTE
    elif primitive == "STOP":
        kind = DecisionKind.STOP
    elif primitive == "REPORT_NOT_FOUND":
        kind = DecisionKind.REPORT_NOT_FOUND
    else:
        raise ValueError(f"oracle-effect winner uses unknown primitive {primitive}")
    return OracleEffectDecision(kind, winner.candidate_id, primitive)
