"""Exact paired analysis for sealed PIU binary outcomes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml

from .primitive_registry import wilson_interval

REQUIRED_BINARY_OUTCOMES = (
    "target_grasp_contact",
    "wrong_object_grasp_contact",
    "target_destination_final",
    "task_success",
    "abstention",
)
REQUIRED_CONTINUOUS_OUTCOMES = (
    "target_maximum_lift_m",
    "interaction_count",
    "executed_steps",
)


def exact_paired_binomial_pvalue(treatment_only: int, comparator_only: int) -> float:
    """Two-sided exact conditional test over discordant pairs (McNemar)."""

    if treatment_only < 0 or comparator_only < 0:
        raise ValueError("discordance counts cannot be negative")
    discordant = treatment_only + comparator_only
    if discordant == 0:
        return 1.0
    tail_end = min(treatment_only, comparator_only)
    tail = sum(math.comb(discordant, count) for count in range(tail_end + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_binary_summary(
    treatment: Sequence[bool], comparator: Sequence[bool]
) -> dict[str, Any]:
    if len(treatment) != len(comparator) or not treatment:
        raise ValueError("paired binary arms must have the same nonzero length")
    if any(not isinstance(item, bool) for item in (*treatment, *comparator)):
        raise TypeError("paired binary outcomes must be booleans")
    both_success = sum(left and right for left, right in zip(treatment, comparator, strict=True))
    treatment_only = sum(left and not right for left, right in zip(treatment, comparator, strict=True))
    comparator_only = sum(not left and right for left, right in zip(treatment, comparator, strict=True))
    both_failure = len(treatment) - both_success - treatment_only - comparator_only
    treatment_successes = both_success + treatment_only
    comparator_successes = both_success + comparator_only
    trials = len(treatment)
    return {
        "trials": trials,
        "treatment": {
            "successes": treatment_successes,
            "rate": treatment_successes / trials,
            "wilson_95": wilson_interval(treatment_successes, trials),
        },
        "comparator": {
            "successes": comparator_successes,
            "rate": comparator_successes / trials,
            "wilson_95": wilson_interval(comparator_successes, trials),
        },
        "paired_risk_difference": (treatment_successes - comparator_successes) / trials,
        "discordance": {
            "both_success": both_success,
            "treatment_only": treatment_only,
            "comparator_only": comparator_only,
            "both_failure": both_failure,
        },
        "exact_two_sided_paired_binomial_p": exact_paired_binomial_pvalue(
            treatment_only, comparator_only
        ),
        "interval_note": (
            "Wilson intervals describe each marginal arm; they are not an interval "
            "for the paired risk difference."
        ),
    }


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Return monotone Holm adjusted p-values in the original key space."""

    if not pvalues:
        return {}
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in pvalues.values()):
        raise ValueError("p-values must be finite and lie in [0,1]")
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    running = 0.0
    adjusted: dict[str, float] = {}
    for index, (name, value) in enumerate(ordered):
        running = max(running, (count - index) * value)
        adjusted[name] = min(1.0, running)
    return {name: adjusted[name] for name in pvalues}


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_formal_outcomes(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise ValueError("formal outcome file is empty")
    unique: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("schema_version") != "piu.formal-outcome.v1":
            raise ValueError("unsupported formal outcome schema")
        if row.get("split") != "sealed_test":
            raise ValueError("formal outcomes must come from sealed_test")
        key = (str(row.get("initial_state_group", "")), str(row.get("method_id", "")))
        if not all(key) or key in unique:
            raise ValueError("formal group/method keys must be nonempty and unique")
        unique.add(key)
        simulator_seed = row.get("simulator_seed")
        if not isinstance(simulator_seed, int) or isinstance(simulator_seed, bool):
            raise TypeError("formal outcomes require an integer simulator seed")
        expected_evidence = (
            "oracle_upper_bound" if key[1] in {"B6", "B7"} else "public_method"
        )
        if row.get("evidence_class") != expected_evidence:
            raise ValueError(
                f"formal method {key[1]} requires evidence_class {expected_evidence}"
            )
        status = row.get("rollout_status")
        if status not in {"COMPLETE", "FAILED", "TIMEOUT", "ABSTAINED"}:
            raise ValueError("formal rollout_status is missing or unsupported")
        for hash_name in (
            "source_state_sha256",
            "action_history_sha256",
            "policy_identity_sha256",
        ):
            digest = str(row.get(hash_name, ""))
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"{hash_name} must be a lowercase SHA-256")
        outcomes = row.get("outcomes")
        if not isinstance(outcomes, Mapping):
            raise TypeError("formal outcomes require an outcome mapping")
        for name in REQUIRED_BINARY_OUTCOMES:
            if not isinstance(outcomes.get(name), bool):
                raise TypeError(f"formal binary outcome {name} must be boolean")
        for name in REQUIRED_CONTINUOUS_OUTCOMES:
            value = outcomes.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"formal continuous outcome {name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"formal continuous outcome {name} must be finite and nonnegative")
        if status == "ABSTAINED" and outcomes["abstention"] is not True:
            raise ValueError("ABSTAINED rollout must retain abstention=true")
    return rows


def load_analysis_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, Mapping) or value.get("schema_version") != "piu.formal-analysis-experiment.v1":
        raise ValueError("unsupported PIU formal analysis config")
    population = value.get("population", {})
    reporting = value.get("reporting", {})
    if population.get("require_complete_paired_method_matrix") is not True:
        raise ValueError("formal analysis requires a complete paired matrix")
    required_methods = population.get("required_method_ids")
    if not isinstance(required_methods, Sequence) or isinstance(
        required_methods, (str, bytes)
    ):
        raise TypeError("formal analysis requires an explicit method ID sequence")
    if tuple(str(item) for item in required_methods) != tuple(
        f"B{index}" for index in range(9)
    ):
        raise ValueError("formal analysis method matrix must be exactly B0--B8")
    if population.get("missing_or_failed_rollout") != "retain_as_observed_failure":
        raise ValueError("formal analysis cannot drop failed rollouts")
    if reporting.get("oracle_column_separate") is not True:
        raise ValueError("formal analysis must separate the oracle column")
    if reporting.get("no_success_threshold") is not True:
        raise ValueError("formal analysis cannot add an empirical pass threshold")
    primary = value.get("primary", {})
    if (
        primary.get("test") != "exact_two_sided_paired_binomial"
        or primary.get("outcome") not in REQUIRED_BINARY_OUTCOMES
        or primary.get("treatment") not in required_methods
        or primary.get("comparator") not in required_methods
    ):
        raise ValueError("formal primary comparison is unsupported")
    secondary = value.get("secondary", {}).get("comparisons")
    if not isinstance(secondary, Sequence) or isinstance(secondary, (str, bytes)):
        raise TypeError("formal secondary comparisons must be an explicit sequence")
    comparison_ids: set[str] = set()
    for comparison in secondary:
        if not isinstance(comparison, Mapping):
            raise TypeError("formal secondary comparison must be a mapping")
        identifier = str(comparison.get("id", ""))
        if not identifier or identifier in comparison_ids:
            raise ValueError("formal secondary comparison IDs must be unique")
        comparison_ids.add(identifier)
        if (
            comparison.get("outcome") not in REQUIRED_BINARY_OUTCOMES
            or comparison.get("treatment") not in required_methods
            or comparison.get("comparator") not in required_methods
        ):
            raise ValueError(f"unsupported formal secondary comparison {identifier}")
    descriptive_references = value.get("descriptive_continuous", {}).get(
        "paired_references"
    )
    if tuple(descriptive_references or ()) != ("B0", "B1", "B3"):
        raise ValueError("formal descriptive references must remain B0, B1, and B3")
    interpretation = value.get("interpretation_contract", {})
    if (
        interpretation.get("automatic_boolean_gate") is not False
        or interpretation.get("no_p_value_only_success_claim") is not True
        or interpretation.get("no_posthoc_nonsaturated_subgroup") is not True
    ):
        raise ValueError("formal interpretation contract permits a post hoc claim")
    return dict(value)


def _comparison(
    matrix: Mapping[str, Mapping[str, Mapping[str, Any]]],
    specification: Mapping[str, Any],
) -> dict[str, Any]:
    treatment_id = str(specification["treatment"])
    comparator_id = str(specification["comparator"])
    outcome = str(specification["outcome"])
    treatment_rows = matrix[treatment_id]
    comparator_rows = matrix[comparator_id]
    if set(treatment_rows) != set(comparator_rows):
        raise ValueError(f"comparison {specification['id']} is not paired by group")
    groups = sorted(treatment_rows)
    result = paired_binary_summary(
        [bool(treatment_rows[group][outcome]) for group in groups],
        [bool(comparator_rows[group][outcome]) for group in groups],
    )
    return {
        "id": str(specification["id"]),
        "treatment_id": treatment_id,
        "comparator_id": comparator_id,
        "outcome": outcome,
        "group_ids": groups,
        **result,
    }


def analyze_formal_outcomes(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    methods = sorted({str(row["method_id"]) for row in rows})
    required_methods = sorted(
        str(item) for item in config["population"]["required_method_ids"]
    )
    if methods != required_methods:
        raise ValueError("formal outcome matrix must contain exactly B0--B8")
    groups = sorted({str(row["initial_state_group"]) for row in rows})
    matrix: dict[str, dict[str, Mapping[str, Any]]] = {method: {} for method in methods}
    evidence_class: dict[str, str] = {}
    source_state_by_group: dict[str, str] = {}
    seed_by_group: dict[str, int] = {}
    policy_identities: set[str] = set()
    for row in rows:
        method = str(row["method_id"])
        group = str(row["initial_state_group"])
        matrix[method][group] = row["outcomes"]
        source_digest = str(row["source_state_sha256"])
        previous_source = source_state_by_group.setdefault(group, source_digest)
        if previous_source != source_digest:
            raise ValueError(
                f"formal group {group} mixes different paired source states"
            )
        simulator_seed = int(row["simulator_seed"])
        previous_seed = seed_by_group.setdefault(group, simulator_seed)
        if previous_seed != simulator_seed:
            raise ValueError(f"formal group {group} mixes different simulator seeds")
        policy_identities.add(str(row["policy_identity_sha256"]))
        observed_class = str(row.get("evidence_class", ""))
        previous = evidence_class.setdefault(method, observed_class)
        if previous != observed_class:
            raise ValueError(f"method {method} mixes evidence classes")
    if len(policy_identities) != 1:
        raise ValueError("formal methods use different frozen policy identities")
    if any(set(matrix[method]) != set(groups) for method in methods):
        raise ValueError("formal outcome matrix has a missing method/group row")
    oracle_ids = {str(item) for item in config["reporting"]["oracle_methods"]}
    if any(
        oracle_id in matrix
        and evidence_class[oracle_id] != "oracle_upper_bound"
        for oracle_id in oracle_ids
    ):
        raise ValueError("declared oracle rows are not marked as an upper bound")
    if any(
        evidence_class[method] == "oracle_upper_bound" and method not in oracle_ids
        for method in methods
    ):
        raise ValueError("an undeclared method is marked as an oracle")
    primary = _comparison(matrix, config["primary"])
    secondary = [
        _comparison(matrix, item)
        for item in config["secondary"]["comparisons"]
    ]
    raw_secondary = {
        item["id"]: float(item["exact_two_sided_paired_binomial_p"])
        for item in secondary
    }
    adjusted = holm_adjust(raw_secondary)
    for item in secondary:
        item["holm_adjusted_p"] = adjusted[item["id"]]
    binary_descriptive: dict[str, Any] = {}
    for outcome in REQUIRED_BINARY_OUTCOMES:
        method_summary = {}
        for method in methods:
            values = [bool(matrix[method][group][outcome]) for group in groups]
            successes = sum(values)
            method_summary[method] = {
                "successes": successes,
                "trials": len(values),
                "rate": successes / len(values),
                "wilson_95": wilson_interval(successes, len(values)),
            }
        binary_descriptive[outcome] = method_summary
    continuous: dict[str, Any] = {}
    for outcome in config["descriptive_continuous"]["outcomes"]:
        method_summary = {}
        for method in methods:
            values = [float(matrix[method][group][outcome]) for group in groups]
            method_summary[method] = {"mean": mean(values), "median": median(values)}
        paired = {}
        references = [
            str(item)
            for item in config["descriptive_continuous"].get(
                "paired_references", [config["primary"]["comparator"]]
            )
        ]
        for reference in references:
            if reference not in matrix:
                raise ValueError(f"unknown descriptive reference {reference}")
            for method in methods:
                if method == reference:
                    continue
                differences = [
                    float(matrix[method][group][outcome])
                    - float(matrix[reference][group][outcome])
                    for group in groups
                ]
                paired[f"{method}_minus_{reference}"] = {
                    "mean": mean(differences),
                    "median": median(differences),
                }
        continuous[str(outcome)] = {"arms": method_summary, "paired": paired}
    contact_rates = {
        method: mean(
            bool(matrix[method][group]["target_grasp_contact"])
            for group in groups
        )
        for method in ("B0", "B7", "B8")
    }
    oracle_gap = contact_rates["B7"] - contact_rates["B0"]
    closed_gap = contact_rates["B8"] - contact_rates["B0"]
    gap_diagnostic = {
        "outcome": "target_grasp_contact",
        "rates": contact_rates,
        "raw_to_oracle_gap": oracle_gap,
        "ours_minus_raw": closed_gap,
        "fraction_of_raw_to_oracle_gap_closed": (
            closed_gap / oracle_gap if oracle_gap > 0.0 else None
        ),
        "inferential_test": None,
        "warning": "B7 is an evaluator-only upper bound, not a public method.",
    }
    return {
        "schema_version": "piu.formal-analysis.v1",
        "claim_scope": "SEALED_ONLY_IF_INPUT_AUTHORIZATION_IS_VALID",
        "groups": groups,
        "public_methods": [method for method in methods if method not in oracle_ids],
        "oracle_upper_bound_methods": [method for method in methods if method in oracle_ids],
        "binary_descriptive_by_method": binary_descriptive,
        "primary": primary,
        "secondary_holm_family": secondary,
        "continuous_descriptive_only": continuous,
        "oracle_binding_gap_descriptive_only": gap_diagnostic,
        "interpretation_contract": dict(config["interpretation_contract"]),
        "automatic_method_pass": None,
    }


def analyze_files(data_path: Path, config_path: Path) -> dict[str, Any]:
    report = analyze_formal_outcomes(
        load_formal_outcomes(data_path), load_analysis_config(config_path)
    )
    report["inputs"] = {
        "outcomes": {"path": str(data_path), "sha256": _sha256_text(data_path)},
        "config": {"path": str(config_path), "sha256": _sha256_text(config_path)},
    }
    return report
