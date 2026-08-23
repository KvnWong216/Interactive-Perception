"""Prospective paired-pilot summaries and conservative formal design points."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import variance
from typing import Any

from calibrated_interaction.paired_power import (
    clopper_pearson_interval,
    clopper_pearson_lower_bound,
    smallest_prospective_group_count,
)

from .statistics import paired_binary_summary


def validate_development_episode(
    value: Mapping[str, Any], *, method_id: str, outcome: str
) -> dict[str, Any]:
    """Validate the outcome-facing portion of one public development episode."""

    if value.get("schema_version") != "piu.closed-loop-episode.v1":
        raise ValueError("paired pilot contains an unsupported episode")
    if value.get("method_id") != method_id:
        raise ValueError("paired pilot episode method differs from its arm")
    if value.get("split") != "development":
        raise ValueError("paired pilot episodes must come from development")
    if value.get("evidence_class") != "public_method":
        raise ValueError("paired pilot cannot use an oracle upper bound")
    if value.get("online_oracle_inputs") != []:
        raise ValueError("paired pilot public episodes consumed oracle inputs")
    if value.get("rollout_status") not in {
        "COMPLETE",
        "FAILED",
        "TIMEOUT",
        "ABSTAINED",
    }:
        raise ValueError("paired pilot rollout status is unsupported")
    group = " ".join(str(value.get("initial_state_group", "")).split())
    seed = value.get("simulator_seed")
    outcomes = value.get("outcomes")
    if not group or not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("paired pilot requires a group and integer simulator seed")
    if not isinstance(outcomes, Mapping) or not isinstance(outcomes.get(outcome), bool):
        raise TypeError("paired pilot primary outcome must be boolean")
    return {
        "initial_state_group": group,
        "simulator_seed": seed,
        "outcome": bool(outcomes[outcome]),
        "rollout_status": str(value["rollout_status"]),
    }


def paired_risk_difference_interval(
    *,
    treatment_only: int,
    comparator_only: int,
    trials: int,
    confidence: float,
) -> list[float]:
    """Conservative simultaneous exact-binomial bounds for the paired difference."""

    if min(treatment_only, comparator_only) < 0 or trials < 1:
        raise ValueError("invalid paired discordance counts")
    if treatment_only + comparator_only > trials:
        raise ValueError("discordant counts exceed paired trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    marginal_confidence = 1.0 - (1.0 - confidence) / 2.0
    treatment_interval = clopper_pearson_interval(
        treatment_only, trials, confidence=marginal_confidence
    )
    comparator_interval = clopper_pearson_interval(
        comparator_only, trials, confidence=marginal_confidence
    )
    return [
        max(-1.0, treatment_interval[0] - comparator_interval[1]),
        min(1.0, treatment_interval[1] - comparator_interval[0]),
    ]


def summarize_paired_pilot(
    treatment: Sequence[bool],
    comparator: Sequence[bool],
    *,
    confidence: float,
) -> dict[str, Any]:
    """Report the paired effect, discordance, variance, and a conservative CI."""

    result = paired_binary_summary(treatment, comparator)
    discordance = result["discordance"]
    differences = [
        int(left) - int(right)
        for left, right in zip(treatment, comparator, strict=True)
    ]
    result["paired_difference_sample_variance"] = (
        variance(differences) if len(differences) > 1 else 0.0
    )
    result["paired_risk_difference_conservative_interval"] = {
        "confidence": confidence,
        "bounds": paired_risk_difference_interval(
            treatment_only=int(discordance["treatment_only"]),
            comparator_only=int(discordance["comparator_only"]),
            trials=len(treatment),
            confidence=confidence,
        ),
        "method": (
            "Bonferroni simultaneous Clopper-Pearson bounds on the two "
            "discordant multinomial cell probabilities"
        ),
    }
    return result


def conservative_design_point(
    *,
    treatment_only: int,
    comparator_only: int,
    trials: int,
    joint_confidence: float,
) -> dict[str, float]:
    """Construct a lower-confidence operating point for exact paired power."""

    discordant = treatment_only + comparator_only
    if trials < 1 or min(treatment_only, comparator_only) < 0:
        raise ValueError("invalid paired pilot counts")
    if discordant > trials:
        raise ValueError("discordant counts exceed pilot trials")
    if not 0.0 < joint_confidence < 1.0:
        raise ValueError("joint confidence must lie in (0,1)")
    marginal_confidence = 1.0 - (1.0 - joint_confidence) / 2.0
    discordance_lower = clopper_pearson_lower_bound(
        discordant, trials, confidence=marginal_confidence
    )
    directional_lower = (
        0.0
        if discordant == 0
        else clopper_pearson_lower_bound(
            treatment_only,
            discordant,
            confidence=marginal_confidence,
        )
    )
    return {
        "joint_confidence": joint_confidence,
        "bonferroni_component_confidence": marginal_confidence,
        "discordant_pair_probability_lower": discordance_lower,
        "treatment_win_given_discordance_lower": directional_lower,
        "treatment_only_probability": discordance_lower * directional_lower,
        "comparator_only_probability": discordance_lower
        * (1.0 - directional_lower),
    }


def prospective_paired_design(
    treatment: Sequence[bool],
    comparator: Sequence[bool],
    *,
    alpha: float,
    target_power: float,
    report_confidence: float,
    design_joint_confidence: float,
    search_limit: int,
) -> dict[str, Any]:
    """Plan a disjoint formal test without turning pilot significance into a gate."""

    summary = summarize_paired_pilot(
        treatment, comparator, confidence=report_confidence
    )
    discordance = summary["discordance"]
    point = conservative_design_point(
        treatment_only=int(discordance["treatment_only"]),
        comparator_only=int(discordance["comparator_only"]),
        trials=len(treatment),
        joint_confidence=design_joint_confidence,
    )
    if float(summary["paired_risk_difference"]) <= 0.0:
        status = "PILOT_HAS_NO_POSITIVE_DIRECTIONAL_EFFECT"
        design = None
    elif point["treatment_win_given_discordance_lower"] <= 0.5:
        status = "PILOT_TOO_UNCERTAIN_FOR_CONSERVATIVE_DIRECTIONAL_PLAN"
        design = None
    else:
        result = smallest_prospective_group_count(
            intervention_only_probability=point["treatment_only_probability"],
            baseline_only_probability=point["comparator_only_probability"],
            alpha=alpha,
            target_power=target_power,
            search_limit=search_limit,
        )
        design = (
            None
            if result is None
            else {"prospective_group_count": result[0], "power": result[1]}
        )
        status = (
            "PROSPECTIVE_GROUP_COUNT_FROZEN"
            if design is not None
            else "NO_PLAN_WITHIN_NUMERICAL_SEARCH_BOUND"
        )
    return {
        "status": status,
        "pilot_summary": summary,
        "conservative_design_point": point,
        "design": design,
    }
