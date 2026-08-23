"""Prompt-conditioned information acquisition and utilization research core."""

from .contracts import (
    EvaluatorSidecar,
    PublicTransition,
    Split,
    load_evaluator_sidecars,
    load_public_transitions,
    validate_group_splits,
    validate_public_sidecar_pair,
)
from .evaluation import STAGE_ORDER, aggregate_stage_evidence

__all__ = [
    "STAGE_ORDER",
    "EvaluatorSidecar",
    "PublicTransition",
    "Split",
    "aggregate_stage_evidence",
    "load_evaluator_sidecars",
    "load_public_transitions",
    "validate_group_splits",
    "validate_public_sidecar_pair",
]
