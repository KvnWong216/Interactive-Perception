from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from piu.empirical_dag_v2 import evaluate_empirical_dag_v2
from piu.s03_execution import (
    DEFAULT_CONTRACT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SCHEDULE,
    build_record_artifact,
    build_s03_public_request,
    build_started_receipt,
    validate_s03_model_identity_contract,
    validate_s03_no_legacy_oracle_dependency,
    validate_s03_outcome_schema,
    validate_s03_receipts,
    validate_s03_runner_preflight,
    validate_s03_single_use_policy,
    write_record_and_close,
    write_started_receipt,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def frozen_preflight() -> tuple[dict, dict, dict, dict]:
    return validate_s03_runner_preflight(
        contract_path=ROOT / DEFAULT_CONTRACT,
        schedule_path=ROOT / DEFAULT_SCHEDULE,
        execution_index=0,
        repository_root=ROOT,
    )


def _tree_snapshot(root: Path) -> list[tuple[str, str]]:
    if not root.exists():
        return []
    import hashlib

    return [
        (
            str(path.relative_to(root)),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def test_validate_only_and_dry_run_never_write_or_infer() -> None:
    output_root = ROOT / DEFAULT_OUTPUT_ROOT
    before = _tree_snapshot(output_root)
    base = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/pipeline/run_piu_s03_perception_decision.py"),
        "--schedule",
        str(DEFAULT_SCHEDULE),
        "--execution-index",
        "0",
    ]
    for flag in ("--validate-only", "--dry-run"):
        completed = subprocess.run(
            [*base, flag], cwd=ROOT, check=True, capture_output=True, text=True
        )
        report = json.loads(completed.stdout)
        assert report["inference_executed"] is False
        assert report["outcome_present"] is False
        assert report["files_written"] is False
        assert report["paper_claim_ready"] is False
    assert _tree_snapshot(output_root) == before


def test_runner_rejects_invalid_index_and_schedule_sha(tmp_path: Path) -> None:
    with pytest.raises(IndexError):
        validate_s03_runner_preflight(
            contract_path=ROOT / DEFAULT_CONTRACT,
            schedule_path=ROOT / DEFAULT_SCHEDULE,
            execution_index=620,
            repository_root=ROOT,
        )
    schedule = json.loads((ROOT / DEFAULT_SCHEDULE).read_text())
    schedule["records"][0]["prompt"] = "changed"
    changed = tmp_path / "mutated-schedule.json"
    changed.write_text(json.dumps(schedule))
    with pytest.raises(ValueError, match="other than the frozen"):
        validate_s03_runner_preflight(
            contract_path=ROOT / DEFAULT_CONTRACT,
            schedule_path=changed,
            execution_index=0,
            repository_root=ROOT,
        )


def test_public_request_firewall_rejects_privileged_injection(
    frozen_preflight: tuple[dict, dict, dict, dict],
) -> None:
    _, identity, schedule, request = frozen_preflight
    changed = copy.deepcopy(schedule)
    changed["records"][0]["input_artifacts"]["simulator_instance_id"] = 7
    manifest_path = ROOT / request["manifest"]["path"]
    manifest = json.loads(manifest_path.read_text())
    with pytest.raises(ValueError, match="forbidden policy field"):
        build_s03_public_request(
            schedule=changed,
            manifest=manifest,
            schedule_path=ROOT / DEFAULT_SCHEDULE,
            manifest_path=manifest_path,
            identity_path=ROOT / request["model_identity"]["path"],
            identity=identity,
            execution_index=0,
            repository_root=ROOT,
        )


def test_identity_requires_all_fields_and_has_no_oracle_dependency(
    tmp_path: Path, frozen_preflight: tuple[dict, dict, dict, dict]
) -> None:
    _, identity, _, _ = frozen_preflight
    validate_s03_no_legacy_oracle_dependency(identity)
    changed = copy.deepcopy(identity)
    changed["backend"]["legacy_oracle_dependency"] = True
    with pytest.raises(ValueError, match="legacy Oracle"):
        validate_s03_no_legacy_oracle_dependency(changed)
    missing = copy.deepcopy(identity)
    missing.pop("calibrator")
    path = tmp_path / "missing-identity.json"
    path.write_text(json.dumps(missing))
    with pytest.raises(ValueError, match="fields differ"):
        validate_s03_model_identity_contract(path, repository_root=ROOT)


def test_infrastructure_record_schema_and_single_use_receipt_chain(
    tmp_path: Path, frozen_preflight: tuple[dict, dict, dict, dict]
) -> None:
    _, identity, schedule, request = frozen_preflight
    output_root = tmp_path / "synthetic-ledger"
    row = schedule["records"][0]
    started = build_started_receipt(
        request=request,
        output_root=output_root,
        previous_close_sha256=None,
        repository_root=ROOT,
    )
    write_started_receipt(output_root, started)
    record = build_record_artifact(
        request=request,
        schedule_row=row,
        prediction=None,
        outcome=None,
        artifacts=[],
        identity=identity,
        infrastructure_failure={
            "stage": "UNIT_TEST_AFTER_REQUEST",
            "category": "SYNTHETIC_INFRASTRUCTURE_FAILURE",
            "message": "schema-only fixture; no inference was invoked",
            "consumes_execution_index": True,
            "adjudication": "closed_without_rerun_or_replacement",
        },
        inference_executed=False,
    )
    validate_s03_outcome_schema(record, repository_root=ROOT, identity=identity)
    record_path = output_root / "records/000_s03-a-000/infrastructure_failure.json"
    write_record_and_close(
        output_root=output_root,
        record_path=record_path,
        record=record,
        started=started,
        repository_root=ROOT,
    )
    manifest_path = ROOT / request["manifest"]["path"]
    identity_path = ROOT / request["model_identity"]["path"]
    ledger = validate_s03_receipts(
        output_root,
        schedule=schedule,
        schedule_path=ROOT / DEFAULT_SCHEDULE,
        manifest_path=manifest_path,
        identity_path=identity_path,
        identity=identity,
        repository_root=ROOT,
    )
    assert ledger == {"closed": 1, "in_flight": None, "next_execution_index": 1}
    with pytest.raises(ValueError, match="next index is 1"):
        validate_s03_single_use_policy(
            output_root,
            execution_index=0,
            ledger_status=ledger,
            record_id="s03-a-000",
        )


def test_dag_recognizes_runner_without_completing_s03() -> None:
    report = evaluate_empirical_dag_v2(
        ROOT / "configs/experiments/piu_empirical_stage_dag_v2.yaml",
        repository_root=ROOT,
    )
    stages = {row["id"]: row for row in report["stages"]}
    assert report["s03_public_runner"]["status"] == (
        "FROZEN_READY_BEFORE_S03_OUTCOMES"
    )
    assert stages["S03_public_perception_decision_gate"]["status"] == (
        "READY_FOR_EXTERNAL_WORK"
    )
    assert report["paper_claim_ready"] is False
