"""Evaluate the additive S03 public-v3 execution amendment."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .empirical_dag_v2 import evaluate_empirical_dag_v2
from .s03_execution import validate_s03_receipts
from .s03_preparation import sha256, validate_s03_input_manifest, validate_s03_offline_schedule
from .s03_v3_amendment import (
    V3_CERTIFICATE_PATH,
    V3_BLOCKER_PATH,
    V3_OUTPUT_ROOT,
    classify_s03_v3_lifecycle,
    validate_s03_v2_seal,
    validate_s03_v3_execution_plan,
)


def _resolve(path: str | Path, *, repository_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repository_root / value


def _verify_reference(value: Any, label: str, *, repository_root: Path) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exact path/SHA-256")
    path = _resolve(str(value["path"]), repository_root=repository_root)
    if not path.is_file() or sha256(path) != value["sha256"]:
        raise ValueError(f"{label} differs from its frozen bytes")
    return path


def evaluate_empirical_dag_v4(
    amendment_path: Path, *, repository_root: Path
) -> dict[str, Any]:
    config = yaml.safe_load(amendment_path.read_text())
    if not isinstance(config, Mapping) or set(config) != {
        "schema_version",
        "id",
        "status",
        "base_dag_v2",
        "historical_dag_v3",
        "v3_amendment_note",
        "v2_blocker",
        "v3_runtime_readiness",
        "v3_runner_identity",
        "v3_execution_plan",
        "legacy_oracle_actionable",
        "paper_claim_ready",
    }:
        raise ValueError("S03 DAG v4 fields differ")
    if (
        config["schema_version"]
        != "piu.empirical-stage-dag-execution-amendment.v3"
        or config["status"] != "frozen_s03_public_v3_before_outcomes"
        or config["legacy_oracle_actionable"] is not False
        or config["paper_claim_ready"] is not False
    ):
        raise ValueError("S03 DAG v4 crossed the pre-outcome claim boundary")
    base_path = _verify_reference(
        config["base_dag_v2"], "DAG v4 base DAG v2", repository_root=repository_root
    )
    _verify_reference(
        config["historical_dag_v3"],
        "DAG v4 historical DAG v3",
        repository_root=repository_root,
    )
    note = _verify_reference(
        config["v3_amendment_note"],
        "DAG v4 v3 amendment note",
        repository_root=repository_root,
    )
    blocker = _verify_reference(
        config["v2_blocker"], "DAG v4 v2 blocker", repository_root=repository_root
    )
    readiness = _verify_reference(
        config["v3_runtime_readiness"],
        "DAG v4 runtime readiness",
        repository_root=repository_root,
    )
    runner = _verify_reference(
        config["v3_runner_identity"],
        "DAG v4 runner identity",
        repository_root=repository_root,
    )
    plan_path = _verify_reference(
        config["v3_execution_plan"],
        "DAG v4 execution plan",
        repository_root=repository_root,
    )
    base = evaluate_empirical_dag_v2(base_path, repository_root=repository_root)
    v2_seal = validate_s03_v2_seal(repository_root=repository_root)
    plan, identity, model = validate_s03_v3_execution_plan(
        plan_path, repository_root=repository_root
    )
    if identity["runtime_readiness"]["path"] != str(readiness.relative_to(repository_root)):
        raise ValueError("DAG v4 readiness and runner identity differ")
    if plan["runner_identity"]["path"] != str(runner.relative_to(repository_root)):
        raise ValueError("DAG v4 runner identity and plan differ")
    schedule_path = _resolve(
        plan["parent_logical_schedule"]["path"], repository_root=repository_root
    )
    manifest_path = _resolve(
        plan["logical_manifest"]["path"], repository_root=repository_root
    )
    model_path = _resolve(
        identity["model_identity"]["path"], repository_root=repository_root
    )
    schedule = validate_s03_offline_schedule(
        schedule_path, repository_root=repository_root
    )
    validate_s03_input_manifest(manifest_path, repository_root=repository_root)
    ledger = validate_s03_receipts(
        repository_root / V3_OUTPUT_ROOT,
        schedule=schedule,
        schedule_path=schedule_path,
        manifest_path=manifest_path,
        identity_path=model_path,
        identity=model,
        repository_root=repository_root,
    )
    certificate_exists = (repository_root / V3_CERTIFICATE_PATH).exists()
    lifecycle = classify_s03_v3_lifecycle(
        ledger,
        execution_authorized=True,
        certificate_present=certificate_exists,
        certificate_validated=False,
        infrastructure_blocker_present=(repository_root / V3_BLOCKER_PATH).exists(),
    )
    if certificate_exists:
        raise ValueError("DAG v4 requires the future v3 certificate validator")
    stages = list(base["stages"])
    stages.extend(
        [
            {
                "id": "S03_public_execution_v1_seal",
                "status": "COMPLETE",
                "claim_scope": "INFRASTRUCTURE_FAILURE_AUDIT_NOT_PERFORMANCE_EVIDENCE",
                "artifacts": [],
            },
            {
                "id": "S03_public_execution_v2_seal",
                "status": "COMPLETE",
                "claim_scope": "INFRASTRUCTURE_FAILURE_AUDIT_NOT_PERFORMANCE_EVIDENCE",
                "artifacts": [
                    {"path": str(blocker.relative_to(repository_root)), "status": "VALID"},
                    {"path": str(note.relative_to(repository_root)), "status": "VALID"},
                ],
            },
            {
                "id": "S03_public_execution_v3_runtime_readiness",
                "status": "COMPLETE",
                "claim_scope": "PROCESSOR_CONFIG_READINESS_NOT_PERFORMANCE_EVIDENCE",
                "artifacts": [
                    {"path": str(readiness.relative_to(repository_root)), "status": "VALID"}
                ],
            },
            {
                "id": "S03_public_execution_v3_freeze",
                "status": "COMPLETE",
                "claim_scope": "PRE_OUTCOME_EXECUTION_FREEZE_NOT_PERFORMANCE_EVIDENCE",
                "artifacts": [
                    {"path": str(runner.relative_to(repository_root)), "status": "VALID"},
                    {"path": str(plan_path.relative_to(repository_root)), "status": "VALID"},
                ],
            },
            {
                "id": "S03_public_perception_decision_gate_v3",
                "status": (
                    "READY_FOR_EXTERNAL_WORK"
                    if lifecycle == "FROZEN_READY_BEFORE_OUTCOMES"
                    else "PARTIAL"
                ),
                "claim_scope": "FUTURE_PUBLIC_INPUT_OUTCOME_CERTIFICATE",
                "artifacts": [],
            },
        ]
    )
    return {
        "status": "INCOMPLETE",
        "canonical_dag_version": "v4_s03_public_execution_v3_amendment",
        "base_dag_status": base["status"],
        "s03_v2": v2_seal,
        "s03_v3_runner": {
            "status": lifecycle,
            "identity_id": identity["identity_id"],
            "record_count": plan["record_count"],
            "outcomes_generated": 0,
            "certificate_present": False,
            "ledger": ledger,
        },
        "legacy_oracle_actionable": False,
        "paper_claim_ready": False,
        "next_actionable_stages": ["S03_public_perception_decision_gate_v3"],
        "stages": stages,
    }
