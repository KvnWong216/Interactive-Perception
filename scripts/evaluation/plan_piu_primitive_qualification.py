#!/usr/bin/env python3
"""Prospectively size primitive qualification from an explicit risk contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.primitive_registry import smallest_binomial_design


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--risk-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-limit", type=int, default=1000)
    args = parser.parse_args()
    registry_path = resolve(args.registry)
    contract_path = resolve(args.risk_contract)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("primitive qualification plans are immutable")
    registry = json.loads(registry_path.read_text())
    if registry.get("schema_version") != "piu.primitive-reliability-registry.v1":
        raise ValueError("unsupported primitive registry")
    contract = yaml.safe_load(contract_path.read_text())
    if contract.get("schema_version") != "piu.primitive-risk-contract.v1":
        raise ValueError("unsupported primitive risk contract")
    primitive = str(contract["primitive"])
    context = str(contract["context"])
    candidate_id = " ".join(str(contract.get("candidate_id", "")).split())
    if not candidate_id:
        raise ValueError("primitive risk contract requires an exact candidate_id")
    pilot = registry["evaluated"][primitive][context]["estimate"]
    null_rate = float(contract["minimum_reliable_rate"])
    alternative_rate = float(pilot["rate"])
    alpha = float(contract["alpha"])
    target_power = float(contract["target_power"])
    if alternative_rate > null_rate:
        design = smallest_binomial_design(
            null_success_probability=null_rate,
            alternative_success_probability=alternative_rate,
            alpha=alpha,
            target_power=target_power,
            search_limit=args.search_limit,
        )
        status = (
            "PROSPECTIVE_GROUP_COUNT_FROZEN"
            if design is not None
            else "NO_PLAN_WITHIN_NUMERICAL_SEARCH_BOUND"
        )
    else:
        design = None
        status = "PILOT_DOES_NOT_EXCEED_DECLARED_RELIABILITY_CONTRACT"
    result = {
        "schema_version": "piu.primitive-qualification-plan.v1",
        "status": status,
        "claim_scope": "DESIGN_ONLY_NO_FORMAL_OUTCOME_DATA",
        "primitive": primitive,
        "context": context,
        "candidate_id": candidate_id,
        "registry": {
            "path": portable(registry_path),
            "sha256": sha256(registry_path),
        },
        "risk_contract": {
            "path": portable(contract_path),
            "sha256": sha256(contract_path),
            "minimum_reliable_rate": null_rate,
            "provenance": contract["minimum_reliable_rate_provenance"],
        },
        "pilot": pilot,
        "pilot_role": "design_alternative_only_excluded_from_formal_test",
        "alternative_success_probability": alternative_rate,
        "alpha": alpha,
        "target_power": target_power,
        "design": design,
        "search_limit": args.search_limit,
        "test": "exact_one_sided_binomial",
        "warning": (
            "The minimum reliable rate is an external downstream risk contract, "
            "not a value selected from this pilot. Formal groups must be new."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
