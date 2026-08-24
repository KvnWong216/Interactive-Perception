from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from piu.empirical_dag_v2 import evaluate_empirical_dag_v2
from piu.s03_preparation import (
    MANIFEST_PATH,
    RUNBOOK_SHA256,
    S02_CERTIFICATE_SHA256,
    S02_SCHEDULE_SHA256,
    SCHEDULE_PATH,
    build_s03_input_manifest,
    build_s03_offline_schedule,
    sha256,
    validate_s03_input_manifest,
    validate_s03_offline_schedule,
)


ROOT = Path(__file__).resolve().parents[1]


def _dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def test_frozen_s03_preparation_is_complete_sanitized_and_deterministic() -> None:
    manifest_path = ROOT / MANIFEST_PATH
    schedule_path = ROOT / SCHEDULE_PATH
    manifest = validate_s03_input_manifest(manifest_path, repository_root=ROOT)
    schedule = validate_s03_offline_schedule(schedule_path, repository_root=ROOT)

    assert manifest["counts"] == {
        "total": 620,
        "A_INFORMATION_EFFECT": 124,
        "B_DECISION_ROUTING": {"ACT": 124, "OPEN": 124, "STOP": 124},
        "C_CLOSED_LOOP_TRANSITION": 124,
    }
    assert len(schedule["records"]) == 620
    assert all(row["outcome_present"] is False for row in manifest["records"])
    assert all(row["inference_executed"] is False for row in schedule["records"])
    assert {101, 104} <= {
        row["linked_s02_index"] for row in manifest["records"]
    }
    assert manifest["upstream"]["runbook"]["sha256"] == RUNBOOK_SHA256
    assert manifest["upstream"]["s02_schedule"]["sha256"] == S02_SCHEDULE_SHA256
    assert manifest["upstream"]["s02_certificate"]["sha256"] == S02_CERTIFICATE_SHA256

    rebuilt = build_s03_input_manifest(repository_root=ROOT)
    assert rebuilt == manifest
    rebuilt_schedule = build_s03_offline_schedule(
        rebuilt,
        manifest_path=manifest_path,
        manifest_sha256=sha256(manifest_path),
        repository_root=ROOT,
    )
    assert rebuilt_schedule == schedule


def test_s03_manifest_rejects_privileged_policy_field(tmp_path: Path) -> None:
    manifest = copy.deepcopy(
        json.loads((ROOT / MANIFEST_PATH).read_text())
    )
    manifest["records"][0]["policy_visible_input"]["simulator_instance_id"] = 7
    path = tmp_path / "manifest.json"
    _dump(path, manifest)
    with pytest.raises(ValueError):
        validate_s03_input_manifest(path, repository_root=ROOT)


def test_s03_manifest_rejects_filtered_failure_indices(tmp_path: Path) -> None:
    manifest = copy.deepcopy(
        json.loads((ROOT / MANIFEST_PATH).read_text())
    )
    manifest["records"][101]["linked_s02_index"] = 100
    path = tmp_path / "manifest.json"
    _dump(path, manifest)
    with pytest.raises(ValueError, match="filtered or reordered"):
        validate_s03_input_manifest(path, repository_root=ROOT)


def test_s03_schedule_rejects_predictions_and_outcome_flags(tmp_path: Path) -> None:
    schedule = copy.deepcopy(
        json.loads((ROOT / SCHEDULE_PATH).read_text())
    )
    schedule["records"][0]["prediction"] = "OPEN"
    schedule["records"][1]["outcome_present"] = True
    path = tmp_path / "schedule.json"
    _dump(path, schedule)
    with pytest.raises(ValueError):
        validate_s03_offline_schedule(path, repository_root=ROOT)


def test_canonical_v2_separates_legacy_oracle_and_public_s03_gate() -> None:
    dag = ROOT / "configs/experiments/piu_empirical_stage_dag_v2.yaml"
    report = evaluate_empirical_dag_v2(dag, repository_root=ROOT)
    stages = {row["id"]: row for row in report["stages"]}

    assert sha256(ROOT / "configs/experiments/piu_empirical_stage_dag_v1.yaml") == (
        "4f390bb0b9f0ca1216544f0da05f114f0b39029b286f74e10190f3283b3deda0"
    )
    assert stages["S02_open_primitive_qualification"]["status"] == "COMPLETE"
    assert stages["S03_oracle_development_gate"]["claim_scope"] == (
        "PRIVILEGED_LEGACY_DIAGNOSTIC_NOT_PUBLIC_METHOD_RESULT"
    )
    assert stages["S03_oracle_development_gate"]["public_input_claim_eligible"] is False
    assert "S03_oracle_development_gate" not in report["next_actionable_stages"]
    assert stages["S03_public_perception_decision_input_freeze"]["status"] == "COMPLETE"
    assert stages["S03_public_perception_decision_gate"]["status"] == (
        "READY_FOR_EXTERNAL_WORK"
    )
    assert not (ROOT / "results/method/piu_s03_perception_decision_certificate_v1.json").exists()
    assert report["paper_claim_ready"] is False
