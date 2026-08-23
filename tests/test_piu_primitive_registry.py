from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from piu.primitive_registry import (
    allocate_episode_primitive_risk,
    evaluate_frozen_binomial_design,
    exact_binomial_power,
    reliability_record,
    load_primitive_qualification_certificate,
    load_primitive_qualification_execution_receipt,
    load_primitive_qualification_plan,
    load_primitive_qualification_schedule,
    smallest_binomial_design,
    validate_derived_primitive_risk_contract,
)
from piu_test_artifacts import write_formal_primitive_certificate

ROOT = Path(__file__).resolve().parents[1]


def test_qualified_executor_map_rejects_a_fabricated_certificate(
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
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/build_piu_qualified_executor_map.py"),
            "--certificate",
            str(certificate),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "artifact reference" in completed.stderr
    assert not output.exists()


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
        search_limit=100,
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


def test_episode_risk_allocation_is_a_dependence_free_union_bound() -> None:
    allocation = allocate_episode_primitive_risk(
        maximum_episode_failure_probability=0.2,
        maximum_physical_dispatches=8,
    )
    assert allocation["per_dispatch_failure_probability"] == pytest.approx(0.025)
    assert allocation["minimum_reliable_rate"] == pytest.approx(0.975)
    assert allocation["union_bound_maximum_episode_failure_probability"] == (
        pytest.approx(0.2)
    )
    assert allocation["dependence_assumption"] == "none"


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
    assert registry["evaluated"]["DIRECT"][
        "visible_work_surface_grasp_contact"
    ]["estimate"]["successes"] == 10
    assert (
        registry["evaluated"]["DIRECT"][
            "post_open_middle_drawer_grasp_contact"
        ]["estimate"][
            "successes"
        ]
        == 0
    )
    assert registry["historical_count_gates_used"] is False
    budget_path = tmp_path / "external_budget.yaml"
    budget_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.external-execution-risk-budget.v1",
                "status": "FROZEN_BEFORE_PRIMITIVE_QUALIFICATION_OUTCOMES",
                "maximum_episode_probability_of_any_primitive_failure": 0.8,
                "design_alternative_per_dispatch_success_probability": 1.0,
                "maximum_qualification_groups_per_primitive": 1000,
                "authority": "synthetic fixture task owner",
                "rationale": "exercise the derivation without making a claim",
                "outcomes_loaded": False,
            }
        )
    )
    contract_path = tmp_path / "risk.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/derive_piu_primitive_risk_contract.py"),
            "--external-budget",
            str(budget_path),
            "--primitive",
            "PICK",
            "--context",
            "visible_work_surface",
            "--candidate-id",
            "pick_cream_cheese",
            "--output",
            str(contract_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    contract = json.loads(contract_path.read_text())
    validate_derived_primitive_risk_contract(contract)
    assert contract["minimum_reliable_rate"] == pytest.approx(0.9)
    assert contract["outcomes_loaded"] is False
    assert contract["maximum_qualification_groups"] == 1000
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
    assert plan["risk_contract"]["provenance"] == (
        "derived_union_bound_from_external_episode_budget"
    )
    assert plan["alternative_success_probability"] == 1.0
    assert plan["retrospective_pilot_used_for_effect_size"] is False
    assert plan["maximum_qualification_groups"] == 1000
    assert plan["maximum_qualification_groups_provenance"] == (
        "external_task_owner_resource_contract"
    )
    assert plan["design"]["trials"] > 5


def test_risk_contract_rejects_a_hand_modified_minimum_rate(tmp_path: Path) -> None:
    allocation = allocate_episode_primitive_risk(
        maximum_episode_failure_probability=0.2,
        maximum_physical_dispatches=8,
    )
    value = {
        "schema_version": "piu.primitive-risk-contract.v1",
        "status": "FROZEN_BEFORE_PRIMITIVE_QUALIFICATION_OUTCOMES",
        "outcomes_loaded": False,
        "minimum_reliable_rate_provenance": (
            "derived_union_bound_from_external_episode_budget"
        ),
        "primitive": "OPEN",
        "context": "middle_drawer",
        "candidate_id": "open_middle_drawer",
        "minimum_reliable_rate": allocation["minimum_reliable_rate"] - 0.1,
        "design_alternative_success_probability": 1.0,
        "design_alternative_provenance": "external_task_owner_contract",
        "maximum_qualification_groups": 1000,
        "maximum_qualification_groups_provenance": (
            "external_task_owner_resource_contract"
        ),
        "retrospective_pilot_used_for_effect_size": False,
        "risk_allocation": allocation,
        "alpha": 0.05,
        "target_power": 0.8,
    }
    with pytest.raises(ValueError, match="differs from derivation"):
        validate_derived_primitive_risk_contract(value)


def test_external_budget_cannot_omit_qualification_group_resource_cap(
    tmp_path: Path,
) -> None:
    budget = tmp_path / "budget.yaml"
    budget.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.external-execution-risk-budget.v1",
                "status": "FROZEN_BEFORE_PRIMITIVE_QUALIFICATION_OUTCOMES",
                "maximum_episode_probability_of_any_primitive_failure": 0.8,
                "design_alternative_per_dispatch_success_probability": 1.0,
                "authority": "synthetic fixture task owner",
                "rationale": "exercise missing resource-cap rejection",
                "outcomes_loaded": False,
            }
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/derive_piu_primitive_risk_contract.py"),
            "--external-budget",
            str(budget),
            "--primitive",
            "OPEN",
            "--context",
            "middle_drawer",
            "--candidate-id",
            "open_middle_drawer",
            "--output",
            str(tmp_path / "risk.json"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "resource cap" in completed.stderr
    planner = ROOT / "scripts/evaluation/plan_piu_primitive_qualification.py"
    assert "--search-limit" not in planner.read_text()


def test_blocked_primitive_plan_is_recomputed_as_terminal_evidence(
    tmp_path: Path,
) -> None:
    budget = tmp_path / "budget.yaml"
    budget.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.external-execution-risk-budget.v1",
                "status": "FROZEN_BEFORE_PRIMITIVE_QUALIFICATION_OUTCOMES",
                "maximum_episode_probability_of_any_primitive_failure": 0.8,
                "design_alternative_per_dispatch_success_probability": 1.0,
                "maximum_qualification_groups_per_primitive": 1,
                "authority": "synthetic fixture task owner",
                "rationale": "exercise exact no-design terminal evidence",
                "outcomes_loaded": False,
            }
        )
    )
    risk = tmp_path / "risk.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/derive_piu_primitive_risk_contract.py"),
            "--external-budget",
            str(budget),
            "--primitive",
            "OPEN",
            "--context",
            "middle_drawer",
            "--candidate-id",
            "open_middle_drawer",
            "--output",
            str(risk),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = tmp_path / "plan.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/plan_piu_primitive_qualification.py"),
            "--registry",
            str(ROOT / "results/method/piu_primitive_reliability_registry_v2.json"),
            "--risk-contract",
            str(risk),
            "--output",
            str(plan),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = load_primitive_qualification_plan(plan, repository_root=ROOT)
    assert value["status"] == "NO_PLAN_WITHIN_EXTERNAL_COLLECTION_RESOURCE_CAP"
    assert value["design"] is None


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
    certificate = write_formal_primitive_certificate(
        tmp_path,
        candidate_id="pick_butter",
        primitive="PICK",
        context="calibrated_current_spatial_reference",
    )
    value = json.loads(certificate.read_text())
    assert value["status"] == "FORMALLY_QUALIFIED"
    assert value["paper_method_action_authorized"] is True
    assert value["result"]["exact_one_sided_p_value"] <= 0.05
    output = tmp_path / "qualified_map.json"
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
    qualified = json.loads(output.read_text())
    assert qualified["primitives"] == {"pick_butter": "PICK"}
    schedule_path = Path(value["schedule"]["path"])
    if not schedule_path.is_absolute():
        schedule_path = ROOT / schedule_path
    schedule = json.loads(schedule_path.read_text())
    dry_schedule_value = json.loads(json.dumps(schedule))
    dry_run_root = tmp_path / "dry_qualification_runs"
    dry_schedule_value["run_root"] = str(dry_run_root)
    for entry in dry_schedule_value["entries"]:
        run_id = hashlib.sha256(
            entry["initial_state_group"].encode()
        ).hexdigest()[:12]
        entry["expected_execution_receipt"] = str(
            dry_run_root
            / f"{entry['execution_index']:05d}_{run_id}"
            / "report.json"
        )
    dry_schedule = tmp_path / "dry_schedule.json"
    dry_schedule.write_text(json.dumps(dry_schedule_value, indent=2) + "\n")
    dry = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/pipeline/run_piu_primitive_qualification.py"),
            "--schedule",
            str(dry_schedule),
            "--execution-index",
            "0",
            "--host",
            "pi05.example.internal",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    dry_plan = json.loads(dry.stdout)
    assert dry_plan["external_server_only"] is True
    assert dry_plan["paper_method_dispatch_authorized"] is False
    assert not dry_run_root.exists()

    chained_entry = schedule["entries"][1]
    chained_receipt_path = Path(chained_entry["expected_execution_receipt"])
    if not chained_receipt_path.is_absolute():
        chained_receipt_path = ROOT / chained_receipt_path
    chained_receipt_text = chained_receipt_path.read_text()
    chained_receipt = json.loads(chained_receipt_text)
    chained_attempt_path = Path(chained_receipt["attempt"]["path"])
    if not chained_attempt_path.is_absolute():
        chained_attempt_path = ROOT / chained_attempt_path
    chained_attempt_text = chained_attempt_path.read_text()
    chained_attempt = json.loads(chained_attempt_text)
    chained_attempt["previous_receipt_sha256"] = "0" * 64
    chained_attempt_path.write_text(json.dumps(chained_attempt, indent=2) + "\n")
    chained_receipt["attempt"]["sha256"] = hashlib.sha256(
        chained_attempt_path.read_bytes()
    ).hexdigest()
    chained_receipt_path.write_text(json.dumps(chained_receipt, indent=2) + "\n")
    with pytest.raises(ValueError, match="receipt chain"):
        load_primitive_qualification_execution_receipt(
            chained_receipt_path,
            schedule_path=schedule_path,
            schedule=schedule,
            execution_index=1,
            repository_root=ROOT,
        )
    chained_attempt_path.write_text(chained_attempt_text)
    chained_receipt_path.write_text(chained_receipt_text)

    tampered_schedule = tmp_path / "tampered_schedule.json"
    schedule["entries"][1]["source_state"] = schedule["entries"][0][
        "source_state"
    ]
    tampered_schedule.write_text(json.dumps(schedule, indent=2) + "\n")
    with pytest.raises(ValueError, match="reuse an opaque source state"):
        load_primitive_qualification_schedule(
            tampered_schedule, repository_root=ROOT
        )

    outcomes_path = Path(value["outcomes"]["path"])
    if not outcomes_path.is_absolute():
        outcomes_path = ROOT / outcomes_path
    rows = [json.loads(line) for line in outcomes_path.read_text().splitlines()]
    rows[0]["success"] = False
    outcomes_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    value["outcomes"]["sha256"] = hashlib.sha256(
        outcomes_path.read_bytes()
    ).hexdigest()
    certificate.write_text(json.dumps(value, indent=2) + "\n")
    with pytest.raises(ValueError, match="evaluator recomputation"):
        load_primitive_qualification_certificate(
            certificate, repository_root=ROOT
        )
