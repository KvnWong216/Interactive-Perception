"""Evaluate the additive S03 execution-v2 amendment over canonical DAG v2."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .empirical_dag_v2 import evaluate_empirical_dag_v2
from .s03_preparation import sha256
from .s03_v2_amendment import (
    validate_s03_v1_seal,
    validate_s03_v2_execution_plan,
)


def _resolve(path: str | Path, *, repository_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repository_root / value


def _verify_reference(value: Any, *, repository_root: Path) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError("DAG v3 artifact reference must contain exact path/SHA-256")
    path = _resolve(str(value["path"]), repository_root=repository_root)
    if not path.is_file() or sha256(path) != value["sha256"]:
        raise ValueError("DAG v3 artifact differs from its frozen bytes")
    return path


def evaluate_empirical_dag_v3(
    amendment_path: Path, *, repository_root: Path
) -> dict[str, Any]:
    config = yaml.safe_load(amendment_path.read_text())
    if not isinstance(config, Mapping) or config.get("schema_version") != "piu.empirical-stage-dag-execution-amendment.v2":
        raise ValueError("unsupported S03 execution DAG amendment")
    base_path = _verify_reference(config.get("base_dag_v2"), repository_root=repository_root)
    base_report = evaluate_empirical_dag_v2(base_path, repository_root=repository_root)
    seal_note = _verify_reference(config.get("v1_seal_note"), repository_root=repository_root)
    identity_path = _verify_reference(config.get("v2_runner_identity"), repository_root=repository_root)
    plan_path = _verify_reference(config.get("v2_execution_plan"), repository_root=repository_root)
    seal = validate_s03_v1_seal(repository_root=repository_root)
    plan, identity, _ = validate_s03_v2_execution_plan(plan_path, repository_root=repository_root)
    if plan["runner_identity"]["path"] != str(identity_path.relative_to(repository_root)):
        raise ValueError("DAG v3 runner identity and execution plan differ")
    if config.get("legacy_oracle_actionable") is not False or config.get("paper_claim_ready") is not False:
        raise ValueError("DAG v3 crossed the public pre-outcome claim boundary")
    stages = list(base_report["stages"])
    stages.extend([
        {
            "id": "S03_public_execution_v1_seal",
            "status": "COMPLETE",
            "claim_scope": "INFRASTRUCTURE_FAILURE_AUDIT_NOT_PERFORMANCE_EVIDENCE",
            "artifacts": [{"path": str(seal_note.relative_to(repository_root)), "status": "VALID"}],
        },
        {
            "id": "S03_public_execution_v2_freeze",
            "status": "COMPLETE",
            "claim_scope": "PRE_OUTCOME_EXECUTION_FREEZE_NOT_PERFORMANCE_EVIDENCE",
            "artifacts": [
                {"path": str(identity_path.relative_to(repository_root)), "status": "VALID"},
                {"path": str(plan_path.relative_to(repository_root)), "status": "VALID"},
            ],
        },
    ])
    return {
        "status": "INCOMPLETE",
        "canonical_dag_version": "v3_s03_public_execution_v2_amendment",
        "base_dag_status": base_report["status"],
        "s03_v1": seal,
        "s03_v2_runner": {
            "status": "FROZEN_READY_BEFORE_S03_V2_OUTCOMES",
            "identity_id": identity["identity_id"],
            "record_count": plan["record_count"],
            "outcomes_generated": 0,
            "certificate_present": False,
        },
        "legacy_oracle_actionable": False,
        "paper_claim_ready": False,
        "next_actionable_stages": ["S03_public_perception_decision_gate_v2"],
        "stages": stages,
    }
