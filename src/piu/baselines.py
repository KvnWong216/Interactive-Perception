"""Validation for the fixed-scenario, paired PIU baseline registry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

EXPECTED_METHOD_IDS = tuple(f"B{index}" for index in range(9))
ALLOWED_FAMILIES = frozenset({"baseline", "ablation", "proposed", "upper_bound"})


def _sequence(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(str(item) for item in value)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be non-empty and duplicate-free")
    return result


def validate_baseline_registry(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reject unfair public baselines and any oracle presented as a method."""

    if value.get("schema_version") != "piu.baseline-registry.v1":
        raise ValueError("unsupported PIU baseline registry schema")
    shared = value.get("shared_contract")
    if not isinstance(shared, Mapping):
        raise TypeError("baseline registry requires a shared contract")
    if shared.get("paired_initial_states") != "required":
        raise ValueError("baseline comparison must preserve paired initial states")
    if shared.get("failure_accounting") != "timeouts_and_abstentions_remain_in_denominator":
        raise ValueError("baseline failures and abstentions cannot be filtered")
    _sequence(shared.get("outcomes"), name="shared outcomes")
    public = set(_sequence(shared.get("public_inputs"), name="shared public inputs"))
    forbidden = set(
        _sequence(shared.get("forbidden_online"), name="shared forbidden inputs")
    )
    budgets = shared.get("option_step_budgets")
    if not isinstance(budgets, Mapping) or any(int(item) <= 0 for item in budgets.values()):
        raise ValueError("shared option budgets must be positive")
    methods = value.get("methods")
    if not isinstance(methods, Sequence) or isinstance(methods, (str, bytes)):
        raise TypeError("baseline methods must be a sequence")
    identifiers = tuple(str(row.get("id")) for row in methods)
    if identifiers != EXPECTED_METHOD_IDS:
        raise ValueError(f"baseline IDs must be exactly {EXPECTED_METHOD_IDS}")
    for row in methods:
        family = str(row.get("family"))
        if family not in ALLOWED_FAMILIES:
            raise ValueError(f"unknown baseline family {family!r}")
        if not str(row.get("status", "")).strip():
            raise ValueError(f"{row['id']} requires an evidence status")
        if row.get("same_frozen_policy") is not True:
            raise ValueError(f"{row['id']} changes the frozen policy")
        if row.get("same_option_budgets") is not True:
            raise ValueError(f"{row['id']} changes option budgets")
        if row.get("same_evaluator") is not True:
            raise ValueError(f"{row['id']} changes the evaluator")
        inputs = set(_sequence(row.get("online_inputs"), name=f"{row['id']} inputs"))
        privileged = set(row.get("online_privileged_inputs", ()))
        if family == "upper_bound":
            if not privileged:
                raise ValueError("oracle upper bounds must declare privileged inputs")
            if row.get("separate_upper_bound_column") is not True:
                raise ValueError("oracle upper bound must be reported separately")
            if row.get("eligible_for_main_method_comparison") is not False:
                raise ValueError("oracle cannot be a main-method comparison")
        else:
            if privileged:
                raise ValueError(f"public method {row['id']} declares oracle inputs")
            if not inputs <= public:
                raise ValueError(f"public method {row['id']} exceeds shared inputs")
            normalized = {item.lower() for item in inputs}
            if normalized & {item.lower() for item in forbidden}:
                raise ValueError(f"public method {row['id']} consumes evaluator inputs")
    selection = value.get("selection_contract")
    if not isinstance(selection, Mapping):
        raise TypeError("baseline selection contract is required")
    if selection.get("test_time_method_selection") != "forbidden":
        raise ValueError("test-time baseline selection must be forbidden")
    if selection.get("failed_or_missing_runs") != "never_dropped":
        raise ValueError("missing baseline runs cannot be dropped")
    if selection.get("oracle_used_for_method_selection") is not False:
        raise ValueError("oracle outcomes cannot select a public method")
    return dict(value)


def load_baseline_registry(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, Mapping):
        raise TypeError("baseline registry root must be a mapping")
    return validate_baseline_registry(value)
