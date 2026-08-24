from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from piu.empirical_dag_v3 import evaluate_empirical_dag_v3
from piu.empirical_dag_v4 import evaluate_empirical_dag_v4
from piu.s03_v2_amendment import (
    V1_BLOCKER_SHA256,
    V1_PARTIAL_TREE_SHA256,
    V2_CERTIFICATE_PATH,
    V2_OUTPUT_ROOT,
    V2_PLAN_PATH,
    _validate_s03_v2_model_identity,
    validate_s03_v1_seal,
)
from piu.s03_v3_amendment import validate_s03_v2_seal


ROOT = Path(__file__).resolve().parents[1]


def _snapshot(root: Path) -> list[tuple[str, str]]:
    if not root.exists():
        return []
    return [
        (str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


@pytest.fixture(scope="module")
def frozen_v2() -> tuple[dict, dict, dict]:
    plan = json.loads((ROOT / V2_PLAN_PATH).read_text())
    identity = json.loads(
        (ROOT / "configs/experiments/piu_s03_runner_identity_v2.json").read_text()
    )
    _, model = _validate_s03_v2_model_identity(
        ROOT / "configs/experiments/piu_s03_model_identity_v2.json",
        repository_root=ROOT,
    )
    validate_s03_v2_seal(repository_root=ROOT)
    return plan, identity, model


def test_v1_is_sealed_without_outcome_or_rerun() -> None:
    seal = validate_s03_v1_seal(repository_root=ROOT)
    assert seal == {
        "status": "SEALED_INFRASTRUCTURE_BLOCKED",
        "execution_index_0_consumed": True,
        "execution_index_0_rerun_eligible": False,
        "model_task_outcomes": 0,
        "certificate_present": False,
        "partial_execution_tree_sha256": V1_PARTIAL_TREE_SHA256,
    }
    blocker = json.loads(
        (ROOT / "results/diagnostics/piu_s03_execution_blocker_v1.json").read_text()
    )
    assert hashlib.sha256(
        (ROOT / "results/diagnostics/piu_s03_execution_blocker_v1.json").read_bytes()
    ).hexdigest() == V1_BLOCKER_SHA256
    assert blocker["failed_record"]["execution_index"] == 0
    assert blocker["failed_record"]["outcome_generated"] is False


def test_v2_binds_all_logical_records_and_inherits_firewall(
    frozen_v2: tuple[dict, dict, dict],
) -> None:
    plan, identity, model = frozen_v2
    assert plan["record_count"] == 620
    assert len(plan["logical_record_binding"]["ordered_record_ids"]) == 620
    assert plan["logical_record_binding"]["counts"] == {
        "total": 620,
        "A_INFORMATION_EFFECT": 124,
        "B_DECISION_ROUTING": {"ACT": 124, "OPEN": 124, "STOP": 124},
        "C_CLOSED_LOOP_TRANSITION": 124,
    }
    assert plan["s02_indices_101_and_104_included"] is True
    assert plan["thresholds"]["A_INFORMATION_EFFECT"]["success_threshold"] == 76
    assert plan["thresholds"]["B_DECISION_ROUTING"]["ACT"]["success_threshold"] == 55
    assert plan["thresholds"]["C_CLOSED_LOOP_TRANSITION"]["integrity_threshold"] == 124
    base = json.loads(
        (ROOT / "configs/experiments/piu_s03_model_identity_v1.json").read_text()
    )
    assert model["policy_input_firewall"] == base["policy_input_firewall"]
    assert model["checkpoint"]["digest"] == base["checkpoint"]["digest"]
    assert model["calibrator"] == base["calibrator"]
    assert identity["runtime"]["grounding_dino_processor_api"]["score_threshold_keyword"] == "threshold"
    assert identity["legacy_oracle_actionable"] is False


def test_v2_seal_validates_the_exact_consumed_history() -> None:
    seal = validate_s03_v2_seal(repository_root=ROOT)
    assert seal["status"] == "SEALED_INFRASTRUCTURE_BLOCKED"
    assert seal["execution_index_0_consumed"] is True
    assert seal["execution_index_0_rerun_eligible"] is False
    assert seal["model_task_outcomes"] == 0
    assert seal["certificate_present"] is False
    assert seal["metrics_status"] == "UNAVAILABLE_NOT_PASS_FAIL"
    assert seal["ambiguous_rate_interpretable"] is False


def test_sealed_v2_runner_rejects_reuse_without_writing() -> None:
    output_root = ROOT / V2_OUTPUT_ROOT
    before = _snapshot(output_root)
    completed = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/pipeline/run_piu_s03_perception_decision_v2.py"),
            "--execution-plan",
            str(V2_PLAN_PATH),
            "--execution-index",
            "0",
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "S03 v2 outcomes or certificate exist before the v2 freeze" in completed.stderr
    assert _snapshot(output_root) == before
    assert len(before) == 67
    assert not (ROOT / V2_CERTIFICATE_PATH).exists()


def test_dag_v3_is_historical_and_dag_v4_keeps_gate_incomplete() -> None:
    with pytest.raises(
        ValueError, match="S03 v2 outcomes or certificate exist before the v2 freeze"
    ):
        evaluate_empirical_dag_v3(
            ROOT / "configs/experiments/piu_empirical_stage_dag_v3.yaml",
            repository_root=ROOT,
        )
    report = evaluate_empirical_dag_v4(
        ROOT / "configs/experiments/piu_empirical_stage_dag_v4.yaml",
        repository_root=ROOT,
    )
    assert report["status"] == "INCOMPLETE"
    assert report["s03_v2"]["status"] == "SEALED_INFRASTRUCTURE_BLOCKED"
    assert report["s03_v3_runner"]["status"] == "FROZEN_READY_BEFORE_OUTCOMES"
    assert report["s03_v3_runner"]["outcomes_generated"] == 0
    assert report["legacy_oracle_actionable"] is False
    assert report["paper_claim_ready"] is False
