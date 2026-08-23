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
from piu.primitive_registry import (
    load_primitive_qualification_certificate,
    load_primitive_qualification_execution_receipt,
    load_primitive_qualification_plan,
    load_primitive_qualification_schedule,
    primitive_qualification_outcome,
)


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
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--outcomes-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan_path = resolve(args.plan)
    schedule_path = resolve(args.schedule)
    outcomes_path = resolve(args.outcomes_output)
    output = resolve(args.output)
    if output.exists() or outcomes_path.exists():
        raise FileExistsError("primitive qualification outputs are immutable")
    plan = load_primitive_qualification_plan(plan_path, repository_root=ROOT)
    schedule = load_primitive_qualification_schedule(
        schedule_path, repository_root=ROOT
    )
    if schedule["inputs"]["plan"]["sha256"] != sha256(plan_path):
        raise ValueError("primitive qualification schedule uses another plan")
    expected = {
        "candidate_id": str(plan["candidate_id"]),
        "primitive": str(plan["primitive"]),
        "context": str(plan["context"]),
    }
    rows = []
    schedule_digest = sha256(schedule_path)
    outcome_rows = []
    for index, entry in enumerate(schedule["entries"]):
        receipt_path = resolve(Path(entry["expected_execution_receipt"]))
        receipt = load_primitive_qualification_execution_receipt(
            receipt_path,
            schedule_path=schedule_path,
            schedule=schedule,
            execution_index=index,
            repository_root=ROOT,
        )
        success, evaluator = primitive_qualification_outcome(
            receipt, plan=plan, repository_root=ROOT
        )
        outcome_rows.append(
            {
                "schema_version": "piu.primitive-qualification-outcome.v1",
                **expected,
                "execution_index": index,
                "initial_state_group": entry["initial_state_group"],
                "simulator_seed": entry["simulator_seed"],
                "source_state_sha256": entry["source_state"]["sha256"],
                "schedule_sha256": schedule_digest,
                "execution_receipt": {
                    "path": portable(receipt_path),
                    "sha256": sha256(receipt_path),
                },
                "success": success,
                "success_derived_from_registered_evaluator_contract": True,
                "evaluator": evaluator,
            }
        )
    outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    with outcomes_path.open("x") as handle:
        handle.write("".join(json.dumps(row, allow_nan=False) + "\n" for row in outcome_rows))
    rows = [
        (row["execution_index"], row["initial_state_group"], row["success"])
        for row in outcome_rows
    ]
    groups = [group for _, group, _ in rows]
    design = plan["design"]
    result = evaluate_frozen_binomial_design(
        [success for _, _, success in rows],
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
        "schedule": {
            "path": portable(schedule_path),
            "sha256": sha256(schedule_path),
        },
        "candidate_contract": schedule["candidate_contract"],
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
    load_primitive_qualification_certificate(output, repository_root=ROOT)
    print(json.dumps(certificate, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
