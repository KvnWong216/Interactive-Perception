from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from piu.primitive_registry import (
    evaluate_frozen_binomial_design,
    exact_binomial_power,
    reliability_record,
    smallest_binomial_design,
)

ROOT = Path(__file__).resolve().parents[1]


def test_qualified_executor_map_accepts_only_formal_certificates(
    tmp_path: Path,
) -> None:
    certificate = tmp_path / "certificate.json"
    certificate.write_text(
        json.dumps(
            {
                "schema_version": "piu.primitive-qualification-certificate.v1",
                "status": "FORMALLY_QUALIFIED",
                "paper_method_action_authorized": True,
                "candidate_id": "open_drawer",
                "primitive": "OPEN",
                "result": {"qualified": True},
            }
        )
    )
    output = tmp_path / "map.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/build_piu_qualified_executor_map.py"),
            "--certificate",
            str(certificate),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(output.read_text())
    assert value["primitives"] == {"open_drawer": "OPEN"}
    assert value["paper_method_action_authorized"] is True


def test_reliability_record_has_no_pass_threshold() -> None:
    record = reliability_record([True] * 9 + [False])
    assert record["successes"] == 9
    assert record["rate"] == 0.9
    assert set(record) == {"successes", "trials", "rate", "wilson_95"}


def test_exact_primitive_power_is_one_sided_and_prospective() -> None:
    design = smallest_binomial_design(
        null_success_probability=0.5,
        alternative_success_probability=1.0,
        alpha=0.05,
        target_power=0.8,
    )
    assert design == {
        "trials": 5,
        "rejection_success_count": 5,
        "power": 1.0,
    }
    assert (
        exact_binomial_power(
            4,
            null_success_probability=0.5,
            alternative_success_probability=1.0,
            alpha=0.05,
        )["power"]
        == 0.0
    )


def test_registry_rebuild_and_explicit_risk_contract_planner(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/build_piu_primitive_registry.py"),
            "--output",
            str(registry_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    registry = json.loads(registry_path.read_text())
    assert registry["evaluated"]["OPEN"]["middle_drawer"]["estimate"]["successes"] == 9
    assert (
        registry["evaluated"]["PICK"]["visible_work_surface"]["estimate"]["successes"]
        == 10
    )
    assert (
        registry["evaluated"]["PICK"]["post_open_middle_drawer"]["estimate"][
            "successes"
        ]
        == 0
    )
    assert registry["historical_count_gates_used"] is False
    contract_path = tmp_path / "risk.yaml"
    contract_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.primitive-risk-contract.v1",
                "primitive": "PICK",
                "context": "visible_work_surface",
                "candidate_id": "pick_cream_cheese",
                "minimum_reliable_rate": 0.5,
                "minimum_reliable_rate_provenance": "synthetic_test_contract",
                "alpha": 0.05,
                "target_power": 0.8,
            }
        )
    )
    plan_path = tmp_path / "plan.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/plan_piu_primitive_qualification.py"),
            "--registry",
            str(registry_path),
            "--risk-contract",
            str(contract_path),
            "--output",
            str(plan_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(plan_path.read_text())
    assert plan["status"] == "PROSPECTIVE_GROUP_COUNT_FROZEN"
    assert plan["candidate_id"] == "pick_cream_cheese"
    assert plan["design"]["trials"] == 5


def test_frozen_primitive_design_requires_complete_boolean_denominator() -> None:
    result = evaluate_frozen_binomial_design(
        [True] * 5,
        null_success_probability=0.5,
        alpha=0.05,
        expected_trials=5,
        rejection_success_count=5,
    )
    assert result["qualified"] is True
    with pytest.raises(ValueError, match="trial count"):
        evaluate_frozen_binomial_design(
            [True] * 4,
            null_success_probability=0.5,
            alpha=0.05,
            expected_trials=5,
            rejection_success_count=5,
        )


def test_power_rejects_a_nonseparated_alternative() -> None:
    with pytest.raises(ValueError, match="exceed"):
        exact_binomial_power(
            10,
            null_success_probability=0.8,
            alternative_success_probability=0.8,
            alpha=0.05,
        )


def test_formal_primitive_certificate_uses_every_frozen_group(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": "piu.primitive-qualification-plan.v1",
                "status": "PROSPECTIVE_GROUP_COUNT_FROZEN",
                "claim_scope": "DESIGN_ONLY_NO_FORMAL_OUTCOME_DATA",
                "candidate_id": "pick_butter",
                "primitive": "PICK",
                "context": "calibrated_current_spatial_reference",
                "risk_contract": {
                    "minimum_reliable_rate": 0.5,
                    "provenance": "synthetic_test_contract",
                },
                "alpha": 0.05,
                "design": {
                    "trials": 5,
                    "rejection_success_count": 5,
                    "power": 1.0,
                },
            }
        )
    )
    outcomes = tmp_path / "outcomes.jsonl"
    rows = [
        {
            "schema_version": "piu.primitive-qualification-outcome.v1",
            "candidate_id": "pick_butter",
            "primitive": "PICK",
            "context": "calibrated_current_spatial_reference",
            "initial_state_group": f"group-{index}",
            "success": True,
            "evaluator_sidecar_only": True,
        }
        for index in range(5)
    ]
    outcomes.write_text("".join(json.dumps(row) + "\n" for row in rows))
    certificate = tmp_path / "certificate.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/evaluate_piu_primitive_qualification.py"),
            "--plan",
            str(plan),
            "--outcomes",
            str(outcomes),
            "--output",
            str(certificate),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(certificate.read_text())
    assert value["status"] == "FORMALLY_QUALIFIED"
    assert value["paper_method_action_authorized"] is True
    assert value["result"]["exact_one_sided_p_value"] == pytest.approx(0.03125)
