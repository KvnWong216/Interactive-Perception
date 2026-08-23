from __future__ import annotations

import json
import hashlib
from pathlib import Path

import yaml

from piu.empirical_dag import evaluate_empirical_dag, validate_artifact_rule
from piu.policy_identity import expected_server_metadata, load_checkpoint_identity


ROOT = Path(__file__).resolve().parents[1]


def _split(root: Path) -> Path:
    roles = (
        "train",
        "development",
        "calibration_temperature",
        "calibration_conformal",
        "effect_calibration_temperature",
        "effect_calibration_conformal",
        "sealed_test",
        "primitive_qualification",
        "oracle_formal",
    )
    path = root / "split.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "piu.group-split-manifest.v1",
                "status": "FROZEN_BEFORE_COLLECTION",
                "allocation_method": "prospective_without_outcome_access",
                "scenario": "fixed drawer",
                "assignments": [
                    {
                        "initial_state_group": f"g-{role}",
                        "seed": 100 + index,
                        "split_role": role,
                    }
                    for index, role in enumerate(roles)
                ],
            }
        )
    )
    return path


def _dag(root: Path, stages: list[dict]) -> Path:
    path = root / "dag.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.empirical-stage-dag.v1",
                "split_manifest": {
                    "id": "split",
                    "path": "split.json",
                    "format": "json",
                    "schema_version": "piu.group-split-manifest.v1",
                },
                "required_split_roles": [
                    "train",
                    "development",
                    "calibration_temperature",
                    "calibration_conformal",
                    "effect_calibration_temperature",
                    "effect_calibration_conformal",
                    "sealed_test",
                    "oracle_formal",
                ],
                "stages": stages,
            },
            sort_keys=False,
        )
    )
    return path


def test_empirical_dag_reports_only_unblocked_stage_as_actionable(tmp_path: Path) -> None:
    _split(tmp_path)
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps({"schema_version": "wrong"}))
    dag = _dag(
        tmp_path,
        [
            {
                "id": "first",
                "depends_on": [],
                "artifacts": [
                    {
                        "id": "missing",
                        "path": "missing.json",
                        "format": "json",
                        "schema_version": "piu.first.v1",
                    }
                ],
            },
            {
                "id": "second",
                "depends_on": ["first"],
                "artifacts": [
                    {
                        "id": "stale",
                        "path": "stale.json",
                        "format": "json",
                        "schema_version": "piu.second.v1",
                    }
                ],
            },
        ],
    )
    report = evaluate_empirical_dag(dag, repository_root=tmp_path)
    assert report["next_actionable_stages"] == ["first"]
    assert [row["status"] for row in report["stages"]] == [
        "READY_FOR_EXTERNAL_WORK",
        "WAITING_FOR_PREDECESSOR",
    ]
    assert report["stages"][1]["artifacts"][0]["status"] == "INVALID"


def test_negative_gate_is_valid_terminal_evidence_not_corruption(tmp_path: Path) -> None:
    split = _split(tmp_path)
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "schema_version": "piu.gate.v1",
                "status": "NEGATIVE",
                "split": {
                    "path": "split.json",
                    "sha256": __import__("hashlib").sha256(split.read_bytes()).hexdigest(),
                },
            }
        )
    )
    dag = _dag(
        tmp_path,
        [
            {
                "id": "gate",
                "depends_on": [],
                "artifacts": [
                    {
                        "id": "gate-result",
                        "path": "gate.json",
                        "format": "json",
                        "schema_version": "piu.gate.v1",
                        "state_field": "status",
                        "success_values": ["POSITIVE"],
                        "terminal_values": {"NEGATIVE": "causal gate failed"},
                    }
                ],
            },
            {
                "id": "training",
                "depends_on": ["gate"],
                "artifacts": [],
            },
        ],
    )
    report = evaluate_empirical_dag(dag, repository_root=tmp_path)
    assert report["stages"][0]["status"] == "TERMINAL_BLOCKED"
    assert report["stages"][0]["artifacts"][0]["errors"] == []
    assert report["stages"][1]["status"] == "TERMINAL_BLOCKED"
    assert report["paper_claim_ready"] is False


def test_external_budget_validator_rejects_infeasible_owner_values(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.baseline-registry.v1",
                "shared_contract": {"maximum_controller_decisions": 8},
            }
        )
    )
    budget = tmp_path / "budget.yaml"
    budget.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.external-execution-risk-budget.v1",
                "status": "FROZEN_BEFORE_PRIMITIVE_QUALIFICATION_OUTCOMES",
                "maximum_episode_probability_of_any_primitive_failure": 0.08,
                "design_alternative_per_dispatch_success_probability": 0.5,
                "maximum_qualification_groups_per_primitive": 20,
                "authority": "task owner",
                "rationale": "prospective risk and collection resource decision",
                "outcomes_loaded": False,
            }
        )
    )
    row = validate_artifact_rule(
        {
            "id": "budget",
            "path": "budget.yaml",
            "format": "yaml",
            "schema_version": "piu.external-execution-risk-budget.v1",
            "validator": "external_execution_risk_budget",
            "baseline_registry": "registry.yaml",
        },
        repository_root=tmp_path,
        split_manifest=None,
    )
    assert row["status"] == "INVALID"
    assert "must exceed the derived" in row["errors"][0]


def test_external_endpoint_validator_rechecks_identity_and_probe_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n")
    identity_path = (
        ROOT / "results/diagnostics/pi05_libero_checkpoint_identity_v1.json"
    )
    identity = load_checkpoint_identity(identity_path)
    value = {
        "schema_version": "piu.external-pi05-check.v1",
        "status": "PASS",
        "endpoint": {"host": "pi05.internal", "port": 8002},
        "identity": {
            **expected_server_metadata(identity),
            "capabilities": ["action_chunks"],
            "server_session_id": "a" * 32,
        },
        "checkpoint_identity": {
            "path": str(identity_path),
            "sha256": hashlib.sha256(identity_path.read_bytes()).hexdigest(),
        },
        "action_probe": {
            "source_report": {
                "path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
            "keyframe": "00_before",
            "shape": [1, 7],
            "finite": True,
            "elapsed_seconds": 0.1,
        },
    }
    endpoint = tmp_path / "endpoint.json"
    endpoint.write_text(json.dumps(value))
    rule = {
        "id": "endpoint",
        "path": "endpoint.json",
        "format": "json",
        "schema_version": "piu.external-pi05-check.v1",
        "validator": "external_pi05_endpoint",
        "checkpoint_identity": str(identity_path),
    }
    assert (
        validate_artifact_rule(
            rule, repository_root=tmp_path, split_manifest=None
        )["status"]
        == "VALID"
    )
    value["identity"]["policy_config"] = "another_policy"
    endpoint.write_text(json.dumps(value))
    rejected = validate_artifact_rule(
        rule, repository_root=tmp_path, split_manifest=None
    )
    assert rejected["status"] == "INVALID"
    assert "identity differs" in rejected["errors"][0]
