from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from piu.primitive_registry import (
    exact_binomial_power,
    reliability_record,
    smallest_binomial_design,
)

ROOT = Path(__file__).resolve().parents[1]


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
    assert exact_binomial_power(
        4,
        null_success_probability=0.5,
        alternative_success_probability=1.0,
        alpha=0.05,
    )["power"] == 0.0


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
    assert registry["evaluated"]["OPEN"]["middle_drawer"]["estimate"][
        "successes"
    ] == 9
    assert registry["evaluated"]["PICK"]["visible_work_surface"]["estimate"][
        "successes"
    ] == 10
    assert registry["evaluated"]["PICK"]["post_open_middle_drawer"][
        "estimate"
    ]["successes"] == 0
    assert registry["historical_count_gates_used"] is False
    contract_path = tmp_path / "risk.yaml"
    contract_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.primitive-risk-contract.v1",
                "primitive": "PICK",
                "context": "visible_work_surface",
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
    assert plan["design"]["trials"] == 5


def test_power_rejects_a_nonseparated_alternative() -> None:
    with pytest.raises(ValueError, match="exceed"):
        exact_binomial_power(
            10,
            null_success_probability=0.8,
            alternative_success_probability=0.8,
            alpha=0.05,
        )
