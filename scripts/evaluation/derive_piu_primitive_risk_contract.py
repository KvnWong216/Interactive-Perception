#!/usr/bin/env python3
"""Derive a per-dispatch primitive rate from an external episode risk budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.primitive_registry import (
    allocate_episode_primitive_risk,
    validate_derived_primitive_risk_contract,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def reference(path: Path) -> dict[str, str]:
    return {"path": portable(path), "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allocation-config",
        type=Path,
        default=ROOT / "configs/experiments/piu_executor_risk_allocation_v1.yaml",
    )
    parser.add_argument("--external-budget", type=Path, required=True)
    parser.add_argument("--primitive", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    allocation_path = resolve(args.allocation_config)
    budget_path = resolve(args.external_budget)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("derived primitive risk contracts are immutable")
    allocation_config = yaml.safe_load(allocation_path.read_text())
    if allocation_config.get("schema_version") != "piu.executor-risk-allocation.v1":
        raise ValueError("unsupported executor risk-allocation config")
    if allocation_config.get("status") != (
        "frozen_derivation_waiting_for_external_episode_risk_budget"
    ):
        raise ValueError("executor risk-allocation derivation is not frozen")
    claims = allocation_config.get("claim_contract", {})
    if (
        claims.get("external_budget_required") is not True
        or claims.get("budget_may_be_inferred_from_retrospective_successes")
        is not False
        or claims.get("budget_may_reuse_historical_count_gate") is not False
        or claims.get("calibration_alpha_is_executor_failure_budget") is not False
        or claims.get("derived_rate_is_task_success_probability") is not False
        or claims.get("qualification_outcomes_loaded_during_derivation") is not False
        or claims.get("retrospective_pilot_may_set_design_alternative") is not False
    ):
        raise ValueError("executor risk-allocation claim firewall was weakened")
    allocation = allocation_config.get("allocation", {})
    if (
        allocation.get("method")
        != "bonferroni_union_bound_equal_per_dispatch"
        or allocation.get("dependence_assumption") != "none"
        or allocation.get("outcome_dependent_choice") is not False
    ):
        raise ValueError("unsupported executor risk allocation")
    baseline_path = resolve(allocation_config["inputs"]["baseline_registry"])
    baseline = yaml.safe_load(baseline_path.read_text())
    if baseline.get("schema_version") != "piu.baseline-registry.v1":
        raise ValueError("unsupported baseline registry")
    if baseline.get("scenario") != allocation_config["scope"]["scenario"]:
        raise ValueError("risk allocation and baseline registry use different scenarios")
    maximum_dispatches = baseline["shared_contract"]["maximum_controller_decisions"]
    if (
        not isinstance(maximum_dispatches, int)
        or isinstance(maximum_dispatches, bool)
        or maximum_dispatches < 1
    ):
        raise ValueError("baseline registry lacks a physical-dispatch bound")
    protocol_path = resolve(
        allocation_config["inputs"]["primitive_registry_protocol"]
    )
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol.get("schema_version") != "piu.primitive-registry-protocol.v1":
        raise ValueError("unsupported primitive registry protocol")
    if protocol.get("formal_qualification", {}).get("minimum_reliable_rate") is not None:
        raise ValueError("primitive registry protocol contains a hand-entered rate")
    if protocol.get("formal_qualification", {}).get("risk_allocation") != portable(
        allocation_path
    ):
        raise ValueError("primitive protocol references another risk allocation")
    budget = yaml.safe_load(budget_path.read_text())
    if budget.get("schema_version") != "piu.external-execution-risk-budget.v1":
        raise ValueError("unsupported external execution-risk budget")
    if budget.get("status") != "FROZEN_BEFORE_PRIMITIVE_QUALIFICATION_OUTCOMES":
        raise ValueError("external risk budget is not prospectively frozen")
    if budget.get("outcomes_loaded") is not False:
        raise ValueError("external risk budget may not load qualification outcomes")
    authority = " ".join(str(budget.get("authority", "")).split())
    rationale = " ".join(str(budget.get("rationale", "")).split())
    if not authority or not rationale:
        raise ValueError("external risk budget requires an authority and rationale")
    derived = allocate_episode_primitive_risk(
        maximum_episode_failure_probability=float(
            budget["maximum_episode_probability_of_any_primitive_failure"]
        ),
        maximum_physical_dispatches=maximum_dispatches,
    )
    alternative = float(
        budget["design_alternative_per_dispatch_success_probability"]
    )
    if not derived["minimum_reliable_rate"] < alternative <= 1.0:
        raise ValueError(
            "external design alternative must exceed the derived per-dispatch rate"
        )
    candidate_id = " ".join(args.candidate_id.split())
    primitive = " ".join(args.primitive.split()).upper()
    context = " ".join(args.context.split())
    if not candidate_id or not primitive or not context:
        raise ValueError("primitive, context, and candidate ID must be nonempty")
    if primitive not in protocol.get("prospective_success_contracts", {}):
        raise ValueError("primitive lacks a simulator-grounded success contract")
    result = {
        "schema_version": "piu.primitive-risk-contract.v1",
        "status": "FROZEN_BEFORE_PRIMITIVE_QUALIFICATION_OUTCOMES",
        "claim_scope": "EXECUTOR_RELIABILITY_ONLY_NOT_TASK_SUCCESS",
        "primitive": primitive,
        "context": context,
        "candidate_id": candidate_id,
        "minimum_reliable_rate": derived["minimum_reliable_rate"],
        "minimum_reliable_rate_provenance": (
            "derived_union_bound_from_external_episode_budget"
        ),
        "alpha": float(protocol["formal_qualification"]["alpha"]),
        "target_power": float(protocol["formal_qualification"]["target_power"]),
        "design_alternative_success_probability": alternative,
        "design_alternative_provenance": "external_task_owner_contract",
        "retrospective_pilot_used_for_effect_size": False,
        "risk_allocation": derived,
        "external_authority": authority,
        "external_rationale": rationale,
        "inputs": {
            "allocation_config": reference(allocation_path),
            "baseline_registry": reference(baseline_path),
            "primitive_registry_protocol": reference(protocol_path),
            "external_budget": reference(budget_path),
        },
        "outcomes_loaded": False,
        "paper_method_claim_allowed": False,
        "warning": (
            "This rate bounds primitive execution failure under the declared "
            "episode budget. It is not a probability or guarantee of task success."
        ),
    }
    validate_derived_primitive_risk_contract(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
