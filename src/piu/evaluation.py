"""Six-stage evidence aggregation for PIU closed-loop experiments."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

from .contracts import EvaluatorSidecar

STAGE_ORDER = (
    "L0_information_sufficiency",
    "L1_interaction_selection",
    "L2_primitive_execution",
    "L3_information_acquisition",
    "L4_information_utilization",
    "L5_task_completion",
)


def _wilson(successes: int, trials: int) -> list[float] | None:
    if trials == 0:
        return None
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (rate + z**2 / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _rate(values: Sequence[bool]) -> dict[str, Any]:
    successes = sum(values)
    trials = len(values)
    return {
        "successes": successes,
        "trials": trials,
        "rate": successes / trials if trials else None,
        "wilson_95": _wilson(successes, trials),
    }


def aggregate_stage_evidence(rows: Sequence[EvaluatorSidecar]) -> dict[str, Any]:
    """Aggregate observed stages and adjacent conditional conversions.

    Missing L0/L1 labels remain unsupported rather than being silently counted as
    failures. L4 is explicitly the target-contact utilization endpoint; it does
    not claim that a spatial target-binding prediction was measured.
    """

    if not rows:
        raise ValueError("at least one evaluator sidecar is required")
    outcomes = [row.stage_outcomes() for row in rows]
    stages: dict[str, Any] = {}
    for stage in STAGE_ORDER:
        observed = [value[stage] for value in outcomes if value[stage] is not None]
        stages[stage] = {
            **_rate([bool(value) for value in observed]),
            "unsupported": len(rows) - len(observed),
        }

    transitions: dict[str, Any] = {}
    for previous, current in pairwise(STAGE_ORDER):
        eligible = [
            value[current]
            for value in outcomes
            if value[previous] is True and value[current] is not None
        ]
        transitions[f"{current}_given_{previous}"] = _rate(
            [bool(value) for value in eligible]
        )

    frontier: Counter[str] = Counter()
    for value in outcomes:
        observed = [
            (stage, value[stage])
            for stage in STAGE_ORDER
            if value[stage] is not None
        ]
        failed = next((stage for stage, passed in observed if passed is False), None)
        frontier[failed or "no_observed_failure"] += 1

    acquired = [row for row in rows if row.information_acquired]
    utilization_after_acquisition = _rate(
        [row.target_grasp_contact for row in acquired]
    )
    wrong_contact_after_acquisition = _rate(
        [row.wrong_object_grasp_contact for row in acquired]
    )
    return {
        "schema_version": "piu.stage-evaluation.v1",
        "claim_scope": "RETROSPECTIVE_DESCRIPTIVE_EVIDENCE_ONLY",
        "groups": len(rows),
        "stages": stages,
        "adjacent_stage_transitions": transitions,
        "first_observed_failure": dict(sorted(frontier.items())),
        "acquisition_to_utilization": {
            "acquisition_successes": len(acquired),
            "target_contact_after_acquisition": utilization_after_acquisition,
            "wrong_object_contact_after_acquisition": wrong_contact_after_acquisition,
            "interpretation": (
                "Target grasp contact is a physical utilization endpoint, not a "
                "direct measurement of spatial binding accuracy."
            ),
        },
        "continuous_metrics": {
            "target_visible_pixels_post": [
                max(row.target_visible_pixels_post.values()) for row in rows
            ],
            "target_maximum_lift_m": [row.target_maximum_lift_m for row in rows],
        },
    }
