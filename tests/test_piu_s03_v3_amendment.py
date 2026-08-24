from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from piu.s03_execution import validate_s03_single_use_policy
from piu.s03_v3_amendment import (
    CERTIFIED,
    EXECUTION_BLOCKED_INFRA,
    EXECUTION_COMPLETE_PENDING_CERTIFICATE,
    EXECUTION_IN_PROGRESS,
    FROZEN_READY_BEFORE_OUTCOMES,
    V1_BLOCKER_PATH,
    V1_BLOCKER_SHA256,
    V2_BLOCKER_PATH,
    V2_BLOCKER_SHA256,
    V2_OUTPUT_ROOT,
    V2_PARTIAL_TREE_SHA256,
    V3_CERTIFICATE_PATH,
    V3_OUTPUT_ROOT,
    classify_s03_v3_lifecycle,
    probe_s03_v3_backend_readiness,
    validate_s03_v1_seal,
    validate_s03_v2_seal,
    validate_s03_v3_execution_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "results/method/piu_s03_perception_decision_execution_plan_v3.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def frozen_v3() -> tuple[dict, dict, dict]:
    return validate_s03_v3_execution_plan(PLAN, repository_root=ROOT)


def test_missing_sentencepiece_readiness_fails_closed() -> None:
    def finder(name: str):
        if name == "sentencepiece":
            return None
        return importlib.util.find_spec(name)

    with pytest.raises(RuntimeError, match="sentencepiece"):
        probe_s03_v3_backend_readiness(
            repository_root=ROOT, module_finder=finder
        )


def test_present_sentencepiece_and_siglip_processor_readiness_passes() -> None:
    report = probe_s03_v3_backend_readiness(repository_root=ROOT)
    assert report["package_versions"]["sentencepiece"] == "0.2.1"
    assert report["package_versions"]["protobuf"] == "6.33.5"
    assert report["sentencepiece_available"] is True
    assert report["siglip_tokenizer"] == {
        "processor_class": "SiglipProcessor",
        "tokenizer_class": "SiglipTokenizer",
        "local_files_only": True,
        "initialized": True,
    }
    assert report["grounding_dino_processor_api"]["score_threshold_keyword"] == "threshold"
    assert report["model_weights_loaded"] is False
    assert report["model_prediction_executed"] is False
    assert report["outcome_written"] is False


def test_backend_readiness_preflight_is_zero_write_and_zero_outcome() -> None:
    assert not (ROOT / V3_OUTPUT_ROOT).exists()
    assert not (ROOT / V3_CERTIFICATE_PATH).exists()
    before = set(path.relative_to(ROOT) for path in ROOT.rglob("*") if path.is_file())
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/repro/preflight_piu_s03_backend_v3.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    after = set(path.relative_to(ROOT) for path in ROOT.rglob("*") if path.is_file())
    assert report["status"] == "VERIFIED"
    assert report["model_weights_loaded"] is False
    assert report["inference_executed"] is False
    assert report["outcome_present"] is False
    assert report["files_written"] is False
    assert before == after
    assert not (ROOT / V3_OUTPUT_ROOT).exists()
    assert not (ROOT / V3_CERTIFICATE_PATH).exists()


def test_lifecycle_accepts_validated_closed_one_as_execution_in_progress() -> None:
    seal = validate_s03_v2_seal(repository_root=ROOT)
    assert seal["partial_execution_tree_sha256"] == V2_PARTIAL_TREE_SHA256
    state = classify_s03_v3_lifecycle(
        {"closed": 1, "in_flight": None, "next_execution_index": 1},
        execution_authorized=True,
        certificate_present=False,
    )
    assert state == EXECUTION_IN_PROGRESS


def test_lifecycle_zero_closed_is_frozen_ready() -> None:
    assert classify_s03_v3_lifecycle(
        {"closed": 0, "in_flight": None, "next_execution_index": 0},
        execution_authorized=True,
        certificate_present=False,
    ) == FROZEN_READY_BEFORE_OUTCOMES


def test_lifecycle_closed_620_without_certificate_is_pending_certificate() -> None:
    assert classify_s03_v3_lifecycle(
        {"closed": 620, "in_flight": None, "next_execution_index": 620},
        execution_authorized=True,
        certificate_present=False,
    ) == EXECUTION_COMPLETE_PENDING_CERTIFICATE


def test_certificate_before_closed_620_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires 620"):
        classify_s03_v3_lifecycle(
            {"closed": 619, "in_flight": None, "next_execution_index": 619},
            execution_authorized=True,
            certificate_present=True,
            certificate_validated=True,
        )


def test_certified_requires_complete_and_validated_certificate() -> None:
    assert classify_s03_v3_lifecycle(
        {"closed": 620, "in_flight": None, "next_execution_index": 620},
        execution_authorized=True,
        certificate_present=True,
        certificate_validated=True,
    ) == CERTIFIED
    with pytest.raises(ValueError, match="was not validated"):
        classify_s03_v3_lifecycle(
            {"closed": 620, "in_flight": None, "next_execution_index": 620},
            execution_authorized=True,
            certificate_present=True,
        )


def test_infrastructure_blocker_is_distinct_from_outcome_completion() -> None:
    assert classify_s03_v3_lifecycle(
        {"closed": 1, "in_flight": None, "next_execution_index": 1},
        execution_authorized=True,
        certificate_present=False,
        infrastructure_blocker_present=True,
    ) == EXECUTION_BLOCKED_INFRA


def test_pre_authorization_receipt_is_rejected_as_leakage() -> None:
    with pytest.raises(ValueError, match="before execution authorization"):
        classify_s03_v3_lifecycle(
            {"closed": 1, "in_flight": None, "next_execution_index": 1},
            execution_authorized=False,
            certificate_present=False,
        )


def test_v1_v2_index_zero_remain_immutable_and_single_use() -> None:
    v1 = validate_s03_v1_seal(repository_root=ROOT)
    v2 = validate_s03_v2_seal(repository_root=ROOT)
    assert _sha256(ROOT / V1_BLOCKER_PATH) == V1_BLOCKER_SHA256
    assert _sha256(ROOT / V2_BLOCKER_PATH) == V2_BLOCKER_SHA256
    assert v1["execution_index_0_rerun_eligible"] is False
    assert v2["execution_index_0_rerun_eligible"] is False
    with pytest.raises(ValueError, match="next index is 1"):
        validate_s03_single_use_policy(
            ROOT / V2_OUTPUT_ROOT,
            execution_index=0,
            ledger_status={"closed": 1, "in_flight": None, "next_execution_index": 1},
            record_id="s03-a-000",
        )


def test_v3_plan_keeps_all_records_thresholds_and_failure_indices(
    frozen_v3: tuple[dict, dict, dict],
) -> None:
    plan, identity, model = frozen_v3
    assert plan["record_count"] == 620
    assert len(plan["logical_record_binding"]["ordered_record_ids"]) == 620
    assert len(set(plan["logical_record_binding"]["ordered_record_ids"])) == 620
    assert plan["thresholds"]["A_INFORMATION_EFFECT"] == {
        "success_threshold": 76,
        "denominator": 124,
    }
    assert plan["thresholds"]["B_DECISION_ROUTING"]["ACT"]["success_threshold"] == 55
    assert plan["thresholds"]["C_CLOSED_LOOP_TRANSITION"] == {
        "integrity_threshold": 124,
        "main_success_threshold": 76,
        "denominator": 124,
    }
    assert plan["s02_indices_101_and_104_included"] is True
    assert plan["record_filtering"] is False
    assert plan["record_replacement"] is False
    assert plan["record_cherry_picking"] is False
    assert identity["legacy_oracle_actionable"] is False
    assert model["backend"]["legacy_oracle_dependency"] is False


def test_privileged_input_firewall_is_inherited_unchanged(
    frozen_v3: tuple[dict, dict, dict],
) -> None:
    _, _, model = frozen_v3
    firewall = model["policy_input_firewall"]
    assert firewall["online_oracle_inputs"] == []
    assert {
        "simulator_semantic_id",
        "simulator_instance_id",
        "simulator_segmentation",
        "target_mask",
        "object_pose",
        "oracle_target_marker",
        "container_membership",
        "evaluator_only_fields",
    }.issubset(firewall["forbidden_fields"])
    assert model["execution_scope"]["legacy_oracle"] is False
    assert model["execution_scope"]["pi05_action_calls"] is False


def test_v3_runner_validate_only_does_not_create_execution_root() -> None:
    assert not (ROOT / V3_OUTPUT_ROOT).exists()
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/pipeline/run_piu_s03_perception_decision_v3.py"),
            "--execution-plan",
            str(PLAN),
            "--execution-index",
            "0",
            "--validate-only",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["lifecycle_state"] == FROZEN_READY_BEFORE_OUTCOMES
    assert report["inference_executed"] is False
    assert report["outcome_present"] is False
    assert report["files_written"] is False
    assert not (ROOT / V3_OUTPUT_ROOT).exists()
