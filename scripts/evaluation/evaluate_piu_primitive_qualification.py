#!/usr/bin/env python3
"""Issue an immutable primitive certificate from frozen prospective outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.primitive_registry import evaluate_frozen_binomial_design


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan_path = resolve(args.plan)
    outcomes_path = resolve(args.outcomes)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("primitive qualification certificates are immutable")
    plan = json.loads(plan_path.read_text())
    if plan.get("schema_version") != "piu.primitive-qualification-plan.v1":
        raise ValueError("unsupported primitive qualification plan")
    if plan.get("status") != "PROSPECTIVE_GROUP_COUNT_FROZEN":
        raise ValueError("primitive qualification plan has no frozen design")
    if plan.get("claim_scope") != "DESIGN_ONLY_NO_FORMAL_OUTCOME_DATA":
        raise ValueError("primitive plan claim firewall differs")
    expected = {
        "candidate_id": str(plan["candidate_id"]),
        "primitive": str(plan["primitive"]),
        "context": str(plan["context"]),
    }
    rows = []
    for line in outcomes_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema_version") != "piu.primitive-qualification-outcome.v1":
            raise ValueError("unsupported primitive qualification outcome")
        if any(str(row.get(name)) != value for name, value in expected.items()):
            raise ValueError("primitive outcome identity differs from frozen plan")
        if not isinstance(row.get("success"), bool):
            raise TypeError("primitive outcome success must be boolean")
        if row.get("evaluator_sidecar_only") is not True:
            raise ValueError("primitive outcome must come from the evaluator sidecar")
        group = " ".join(str(row.get("initial_state_group", "")).split())
        if not group:
            raise ValueError("primitive outcome lacks an initial-state group")
        rows.append((group, bool(row["success"])))
    groups = [group for group, _ in rows]
    if len(groups) != len(set(groups)):
        raise ValueError("primitive qualification groups must be unique")
    design = plan["design"]
    result = evaluate_frozen_binomial_design(
        [success for _, success in rows],
        null_success_probability=float(plan["risk_contract"]["minimum_reliable_rate"]),
        alpha=float(plan["alpha"]),
        expected_trials=int(design["trials"]),
        rejection_success_count=int(design["rejection_success_count"]),
    )
    certificate = {
        "schema_version": "piu.primitive-qualification-certificate.v1",
        "status": "FORMALLY_QUALIFIED" if result["qualified"] else "NOT_QUALIFIED",
        **expected,
        "initial_state_groups": groups,
        "complete_frozen_denominator": True,
        "plan": {"path": portable(plan_path), "sha256": sha256(plan_path)},
        "outcomes": {
            "path": portable(outcomes_path),
            "sha256": sha256(outcomes_path),
        },
        "risk_contract": plan["risk_contract"],
        "alpha": float(plan["alpha"]),
        "test": "exact_one_sided_binomial",
        "result": result,
        "paper_method_action_authorized": bool(result["qualified"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(certificate, indent=2, allow_nan=False) + "\n")
    print(json.dumps(certificate, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
