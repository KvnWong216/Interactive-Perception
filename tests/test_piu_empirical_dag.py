from __future__ import annotations

import json
from pathlib import Path

import yaml

from piu.empirical_dag import evaluate_empirical_dag


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
