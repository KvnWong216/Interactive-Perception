"""Compose and validate the S03 public-input amendment without changing DAG v1."""

from __future__ import annotations

import copy
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .empirical_dag import evaluate_empirical_dag, sha256
from .s03_preparation import (
    validate_s03_input_manifest,
    validate_s03_offline_schedule,
    validate_s03_public_outcome_certificate,
)


_CUSTOM_VALIDATORS = {
    "s03_public_input_manifest",
    "s03_public_offline_schedule",
    "s03_public_outcome_certificate",
}


def _resolve(path: str | Path, *, repository_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repository_root / value


def _portable(path: Path, *, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def compose_empirical_dag_v2(
    amendment_path: Path, *, repository_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = yaml.safe_load(amendment_path.read_text())
    if (
        not isinstance(raw, Mapping)
        or raw.get("schema_version") != "piu.empirical-stage-dag-amendment.v1"
    ):
        raise ValueError("unsupported PIU empirical DAG amendment")
    base_ref = raw.get("base_dag")
    if not isinstance(base_ref, Mapping):
        raise TypeError("empirical DAG amendment requires a base DAG reference")
    base_path = _resolve(str(base_ref.get("path", "")), repository_root=repository_root)
    expected_sha = str(base_ref.get("sha256", ""))
    if not base_path.is_file() or sha256(base_path) != expected_sha:
        raise ValueError("empirical DAG amendment base differs from its frozen hash")
    base = yaml.safe_load(base_path.read_text())
    if not isinstance(base, Mapping) or base.get("schema_version") != "piu.empirical-stage-dag.v1":
        raise ValueError("empirical DAG amendment base schema is unsupported")

    config = copy.deepcopy(dict(base))
    stages = config.get("stages")
    if not isinstance(stages, list):
        raise TypeError("empirical DAG base stages must be a list")
    by_id = {str(stage.get("id", "")): stage for stage in stages}
    updates = raw.get("stage_updates", [])
    if not isinstance(updates, list):
        raise TypeError("empirical DAG stage updates must be a list")
    for update in updates:
        if not isinstance(update, Mapping):
            raise TypeError("empirical DAG stage update must be a mapping")
        stage_id = str(update.get("id", ""))
        if stage_id not in by_id:
            raise ValueError(f"empirical DAG update references unknown stage {stage_id!r}")
        by_id[stage_id].update(
            {key: copy.deepcopy(value) for key, value in update.items() if key != "id"}
        )

    insertions = raw.get("stage_insertions", [])
    if not isinstance(insertions, list):
        raise TypeError("empirical DAG stage insertions must be a list")
    for insertion in insertions:
        if not isinstance(insertion, Mapping):
            raise TypeError("empirical DAG insertion must be a mapping")
        after = str(insertion.get("after", ""))
        inserted = insertion.get("stages")
        if after not in by_id or not isinstance(inserted, list) or not inserted:
            raise ValueError("empirical DAG insertion has an invalid anchor or stages")
        position = next(
            index for index, stage in enumerate(stages) if stage["id"] == after
        ) + 1
        for stage in inserted:
            if not isinstance(stage, Mapping):
                raise TypeError("inserted empirical DAG stage must be a mapping")
            stage_id = str(stage.get("id", ""))
            if not stage_id or stage_id in by_id:
                raise ValueError("inserted empirical DAG stage ID is empty or duplicated")
            value = copy.deepcopy(dict(stage))
            stages.insert(position, value)
            position += 1
            by_id[stage_id] = value

    config["id"] = str(raw.get("id", config.get("id", "")))
    config["status"] = str(raw.get("status", config.get("status", "")))
    _validate_public_s03_dag_contract(stages)
    provenance = {
        "path": _portable(amendment_path, repository_root=repository_root),
        "sha256": sha256(amendment_path),
        "base_dag": {
            "path": _portable(base_path, repository_root=repository_root),
            "sha256": expected_sha,
        },
    }
    return config, provenance


def _validate_public_s03_dag_contract(stages: list[dict[str, Any]]) -> None:
    by_id = {str(stage.get("id", "")): stage for stage in stages}
    legacy = by_id.get("S03_oracle_development_gate")
    freeze = by_id.get("S03_public_perception_decision_input_freeze")
    gate = by_id.get("S03_public_perception_decision_gate")
    if legacy is None or freeze is None or gate is None:
        raise ValueError("canonical v2 DAG must contain legacy and public S03 paths")
    if (
        legacy.get("path_status") != "LEGACY_PRIVILEGED_ORACLE_DIAGNOSTIC"
        or legacy.get("public_input_claim_eligible") is not False
    ):
        raise ValueError("legacy Oracle S03 is not excluded from the public-input claim")
    if list(freeze.get("depends_on", ())) != ["S02_open_primitive_qualification"]:
        raise ValueError("public S03 input freeze must depend on qualified OPEN")
    freeze_artifacts = list(freeze.get("artifacts", ()))
    if [row.get("validator") for row in freeze_artifacts] != [
        "s03_public_input_manifest",
        "s03_public_offline_schedule",
    ]:
        raise ValueError("public S03 input freeze lacks its two static validators")
    if list(gate.get("depends_on", ())) != [
        "S03_public_perception_decision_input_freeze"
    ]:
        raise ValueError("public S03 outcome gate must depend on its input freeze")
    gate_artifacts = list(gate.get("artifacts", ()))
    if (
        len(gate_artifacts) != 1
        or gate_artifacts[0].get("id") != "s03_public_outcome_certificate"
        or gate_artifacts[0].get("validator") != "s03_public_outcome_certificate"
    ):
        raise ValueError("public S03 completion must require its validated certificate")


def _run_custom_validator(
    validator: str,
    path: Path,
    rule: Mapping[str, Any],
    *,
    repository_root: Path,
) -> None:
    if validator == "s03_public_input_manifest":
        validate_s03_input_manifest(path, repository_root=repository_root)
    elif validator == "s03_public_offline_schedule":
        validate_s03_offline_schedule(path, repository_root=repository_root)
    elif validator == "s03_public_outcome_certificate":
        schedule_path = _resolve(
            str(rule.get("input_schedule", "")), repository_root=repository_root
        )
        validate_s03_public_outcome_certificate(
            path,
            schedule_path=schedule_path,
            repository_root=repository_root,
        )
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown v2 custom validator {validator!r}")


def _recompute_statuses(report: dict[str, Any]) -> None:
    statuses: dict[str, str] = {}
    next_actionable = []
    for stage in report["stages"]:
        dependencies = list(stage["depends_on"])
        dependency_status = {name: statuses[name] for name in dependencies}
        inherited_terminal = any(
            value == "TERMINAL_BLOCKED" for value in dependency_status.values()
        )
        dependency_complete = all(value == "COMPLETE" for value in dependency_status.values())
        artifact_statuses = [row["status"] for row in stage["artifacts"]]
        if inherited_terminal or (
            dependency_complete and "TERMINAL_BLOCKED" in artifact_statuses
        ):
            status = "TERMINAL_BLOCKED"
        elif not dependency_complete:
            status = "WAITING_FOR_PREDECESSOR"
        elif "INVALID" in artifact_statuses:
            status = "INVALID"
        elif artifact_statuses and all(value == "VALID" for value in artifact_statuses):
            status = "COMPLETE"
        elif any(value == "VALID" for value in artifact_statuses):
            status = "PARTIAL"
        else:
            status = "READY_FOR_EXTERNAL_WORK"
        stage["dependency_status"] = dependency_status
        stage["status"] = status
        statuses[stage["id"]] = status
        if status in {"READY_FOR_EXTERNAL_WORK", "PARTIAL", "INVALID"}:
            next_actionable.append(stage["id"])
    all_complete = bool(report["stages"]) and all(
        stage["status"] == "COMPLETE" for stage in report["stages"]
    )
    report["next_actionable_stages"] = next_actionable
    report["status"] = "COMPLETE" if all_complete else "INCOMPLETE"
    report["empirical_pipeline_complete"] = all_complete
    report["paper_claim_ready"] = all_complete
    report["paper_claim_ready_reason"] = (
        "every hash-bound empirical stage validated"
        if all_complete
        else "one or more external empirical stages remain incomplete"
    )


def evaluate_empirical_dag_v2(
    amendment_path: Path, *, repository_root: Path
) -> dict[str, Any]:
    config, provenance = compose_empirical_dag_v2(
        amendment_path, repository_root=repository_root
    )
    evaluation_config = copy.deepcopy(config)
    custom_rules: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for stage in evaluation_config["stages"]:
        for rule in stage.get("artifacts", ()):
            validator = str(rule.get("validator", ""))
            if validator in _CUSTOM_VALIDATORS:
                custom_rules[str(rule["id"])] = (validator, copy.deepcopy(rule))
                rule.pop("validator")

    with tempfile.TemporaryDirectory(prefix="piu-s03-dag-v2-") as temp_dir:
        composed_path = Path(temp_dir) / "composed_dag_v1.yaml"
        composed_path.write_text(yaml.safe_dump(evaluation_config, sort_keys=False))
        report = evaluate_empirical_dag(
            composed_path, repository_root=repository_root
        )

    for stage in report["stages"]:
        for artifact in stage["artifacts"]:
            custom = custom_rules.get(str(artifact["id"]))
            if custom is None or artifact["status"] == "PENDING":
                continue
            validator, rule = custom
            path = _resolve(str(rule["path"]), repository_root=repository_root)
            try:
                _run_custom_validator(
                    validator, path, rule, repository_root=repository_root
                )
            except Exception as exc:
                artifact["status"] = "INVALID"
                artifact["errors"].append(str(exc))
    _recompute_statuses(report)
    for stage in report["stages"]:
        if stage["id"] == "S03_oracle_development_gate":
            stage["path_status"] = "LEGACY_PRIVILEGED_ORACLE_DIAGNOSTIC"
            stage["public_input_claim_eligible"] = False
            stage["execution_authorized_by_public_s03_design"] = False
    report["next_actionable_stages"] = [
        stage_id
        for stage_id in report["next_actionable_stages"]
        if stage_id != "S03_oracle_development_gate"
    ]
    report["non_actionable_legacy_diagnostic_stages"] = [
        "S03_oracle_development_gate"
    ]
    report["dag"] = provenance
    report["canonical_dag_version"] = "v2_s03_public_input_amendment"
    return report
