"""Machine-verifiable readiness DAG for the external PIU experiment."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .spatial_prefix import validate_feature_arrays
from .splits import (
    load_split_manifest,
    role_to_split,
    validate_learning_collection_budget,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path, *, repository_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repository_root / value


def _field(value: Any, dotted: str) -> Any:
    current = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(dotted)
        current = current[part]
    return current


def _verify_references(
    value: Any, *, repository_root: Path, location: str = "artifact"
) -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            relative = str(value.get("path", ""))
            expected = str(value.get("sha256", ""))
            path = _resolve(relative, repository_root=repository_root)
            if not _SHA256.fullmatch(expected):
                errors.append(f"{location} has malformed SHA-256")
            elif not path.is_file():
                errors.append(f"{location} references missing file: {relative}")
            elif sha256(path) != expected:
                errors.append(f"{location} reference hash differs: {relative}")
        for name, child in value.items():
            errors.extend(
                _verify_references(
                    child,
                    repository_root=repository_root,
                    location=f"{location}.{name}",
                )
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            errors.extend(
                _verify_references(
                    child,
                    repository_root=repository_root,
                    location=f"{location}[{index}]",
                )
            )
    return errors


def _assigned_groups(split_manifest: Mapping[str, Any], role: str) -> set[str]:
    return {
        str(row["initial_state_group"])
        for row in split_manifest["assignments"]
        if row["split_role"] == role
    }


def _validate_group_binding(
    value: Mapping[str, Any],
    rule: Mapping[str, Any],
    *,
    split_manifest: Mapping[str, Any] | None,
) -> list[str]:
    if split_manifest is None:
        return ["group binding requires a valid split manifest"]
    errors = []
    role = str(rule["split_role"])
    try:
        observed = {str(item) for item in _field(value, str(rule["field"]))}
    except (KeyError, TypeError):
        return [f"missing group field {rule['field']}"]
    assigned = _assigned_groups(split_manifest, role)
    if not assigned:
        errors.append(f"split manifest has no groups for {role}")
    elif observed != assigned:
        errors.append(
            f"groups at {rule['field']} differ from complete {role} allocation"
        )
    return errors


def _validate_npz_role(
    value: Mapping[str, Any],
    role: str,
    *,
    split_manifest: Mapping[str, Any] | None,
    repository_root: Path,
) -> list[str]:
    if split_manifest is None:
        return ["feature role validation requires a valid split manifest"]
    reference = value.get("output")
    if not isinstance(reference, Mapping):
        return ["feature report lacks output reference"]
    path = _resolve(str(reference.get("path", "")), repository_root=repository_root)
    if not path.is_file():
        return ["feature report output is missing"]
    try:
        with np.load(path, allow_pickle=False) as store:
            arrays = {name: np.asarray(store[name]) for name in store.files}
        validate_feature_arrays(arrays)
        observed_groups = set(arrays["initial_state_group"].astype(str).tolist())
        observed_splits = set(arrays["split"].astype(str).tolist())
    except Exception as exc:  # format failures are evidence-invalid, not crashes
        return [f"invalid spatial-prefix cache: {exc}"]
    expected_groups = _assigned_groups(split_manifest, role)
    expected_split = role_to_split(role).value
    errors = []
    if observed_groups != expected_groups:
        errors.append(f"feature groups differ from complete {role} allocation")
    if observed_splits != {expected_split}:
        errors.append(f"feature split differs from role {role}")
    return errors


def _load_value(path: Path, format_name: str) -> tuple[Any, list[str]]:
    try:
        if format_name == "json":
            return json.loads(path.read_text()), []
        if format_name == "yaml":
            return yaml.safe_load(path.read_text()), []
        if format_name == "jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            if not rows:
                raise ValueError("JSONL is empty")
            return rows, []
    except Exception as exc:
        return None, [f"cannot parse {format_name}: {exc}"]
    return None, [f"unsupported artifact format {format_name!r}"]


def validate_artifact_rule(
    rule: Mapping[str, Any],
    *,
    repository_root: Path,
    split_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    relative = str(rule.get("path", ""))
    path = _resolve(relative, repository_root=repository_root)
    row: dict[str, Any] = {
        "id": str(rule.get("id", relative)),
        "path": relative,
        "status": "PENDING",
        "sha256": None,
        "errors": [],
        "terminal_reason": None,
    }
    if not path.is_file():
        return row
    row["sha256"] = sha256(path)
    value, errors = _load_value(path, str(rule.get("format", "json")))
    if errors:
        row.update(status="INVALID", errors=errors)
        return row
    values = value if isinstance(value, list) else [value]
    expected_schema = rule.get("schema_version")
    if expected_schema is not None:
        for index, item in enumerate(values):
            if not isinstance(item, Mapping) or item.get("schema_version") != expected_schema:
                errors.append(f"schema mismatch at row {index}")
    if rule.get("verify_references", True) and isinstance(value, (Mapping, list)):
        errors.extend(_verify_references(value, repository_root=repository_root))
    if isinstance(value, Mapping):
        if rule.get("validator") == "learning_collection_budget":
            try:
                validate_learning_collection_budget(value)
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        for dotted, expected in dict(rule.get("expected", {})).items():
            try:
                observed = _field(value, str(dotted))
            except KeyError:
                errors.append(f"missing required field {dotted}")
            else:
                if observed != expected:
                    errors.append(f"field {dotted} differs from frozen contract")
        for binding in rule.get("group_bindings", ()):
            errors.extend(
                _validate_group_binding(
                    value, binding, split_manifest=split_manifest
                )
            )
        role = rule.get("npz_group_role")
        if role is not None:
            errors.extend(
                _validate_npz_role(
                    value,
                    str(role),
                    split_manifest=split_manifest,
                    repository_root=repository_root,
                )
            )
        state_field = rule.get("state_field")
        if state_field is not None:
            try:
                state = _field(value, str(state_field))
            except KeyError:
                errors.append(f"missing state field {state_field}")
            else:
                success = list(rule.get("success_values", ()))
                terminal = dict(rule.get("terminal_values", {}))
                if state in terminal:
                    row["terminal_reason"] = str(terminal[state])
                elif state not in success:
                    errors.append(f"unrecognized state {state!r} at {state_field}")
    if errors:
        row.update(status="INVALID", errors=errors)
    elif row["terminal_reason"] is not None:
        row["status"] = "TERMINAL_BLOCKED"
    else:
        row["status"] = "VALID"
    return row


def _validate_dag(stages: Sequence[Mapping[str, Any]]) -> None:
    ids = [str(stage.get("id", "")) for stage in stages]
    if not ids or any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("empirical DAG stage IDs must be nonempty and unique")
    known: set[str] = set()
    artifact_ids: set[str] = set()
    for stage in stages:
        dependencies = [str(item) for item in stage.get("depends_on", ())]
        if len(set(dependencies)) != len(dependencies) or not set(dependencies) <= known:
            raise ValueError("empirical DAG must be topologically ordered")
        for artifact in stage.get("artifacts", ()):
            artifact_id = str(artifact.get("id", ""))
            if not artifact_id or artifact_id in artifact_ids:
                raise ValueError("empirical DAG artifact IDs must be nonempty and unique")
            artifact_ids.add(artifact_id)
        known.add(str(stage["id"]))


def evaluate_empirical_dag(
    dag_path: Path, *, repository_root: Path
) -> dict[str, Any]:
    config = yaml.safe_load(dag_path.read_text())
    if (
        not isinstance(config, Mapping)
        or config.get("schema_version") != "piu.empirical-stage-dag.v1"
    ):
        raise ValueError("unsupported PIU empirical stage DAG")
    stages = config.get("stages")
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        raise TypeError("empirical DAG stages must be a sequence")
    _validate_dag(stages)
    split_rule = config.get("split_manifest")
    if not isinstance(split_rule, Mapping):
        raise TypeError("empirical DAG requires one split-manifest rule")
    split_row = validate_artifact_rule(
        split_rule, repository_root=repository_root, split_manifest=None
    )
    split_manifest = None
    if split_row["status"] == "VALID":
        split_manifest = load_split_manifest(
            _resolve(str(split_rule["path"]), repository_root=repository_root)
        )
        observed_roles = {row["split_role"] for row in split_manifest["assignments"]}
        required_roles = set(config.get("required_split_roles", ()))
        if not required_roles <= observed_roles:
            missing = sorted(required_roles - observed_roles)
            split_row["status"] = "INVALID"
            split_row["errors"].append(
                f"split manifest lacks required mainline roles {missing}"
            )
            split_manifest = None
    statuses: dict[str, str] = {}
    rows = []
    next_actionable = []
    for stage in stages:
        stage_id = str(stage["id"])
        dependencies = [str(item) for item in stage.get("depends_on", ())]
        artifact_rows = []
        for rule in stage.get("artifacts", ()):
            if str(rule.get("id", "")) == str(split_rule.get("id", "")):
                artifact_rows.append(dict(split_row))
            else:
                artifact_rows.append(
                    validate_artifact_rule(
                        rule,
                        repository_root=repository_root,
                        split_manifest=split_manifest,
                    )
                )
        dependency_status = {name: statuses[name] for name in dependencies}
        inherited_terminal = any(
            status == "TERMINAL_BLOCKED" for status in dependency_status.values()
        )
        dependency_complete = all(
            status == "COMPLETE" for status in dependency_status.values()
        )
        artifact_statuses = [row["status"] for row in artifact_rows]
        if inherited_terminal or (
            dependency_complete and "TERMINAL_BLOCKED" in artifact_statuses
        ):
            status = "TERMINAL_BLOCKED"
        elif not dependency_complete:
            status = "WAITING_FOR_PREDECESSOR"
        elif "INVALID" in artifact_statuses:
            status = "INVALID"
        elif artifact_statuses and all(item == "VALID" for item in artifact_statuses):
            status = "COMPLETE"
        elif any(item == "VALID" for item in artifact_statuses):
            status = "PARTIAL"
        else:
            status = "READY_FOR_EXTERNAL_WORK"
        statuses[stage_id] = status
        if status in {"READY_FOR_EXTERNAL_WORK", "PARTIAL", "INVALID"}:
            next_actionable.append(stage_id)
        rows.append(
            {
                "id": stage_id,
                "title": str(stage.get("title", stage_id)),
                "depends_on": dependencies,
                "dependency_status": dependency_status,
                "status": status,
                "artifacts": artifact_rows,
                "claim_scope": str(stage.get("claim_scope", "NOT_PERFORMANCE_EVIDENCE")),
            }
        )
    all_complete = bool(rows) and all(row["status"] == "COMPLETE" for row in rows)
    return {
        "schema_version": "piu.empirical-stage-status.v1",
        "dag": {"path": str(dag_path), "sha256": sha256(dag_path)},
        "status": "COMPLETE" if all_complete else "INCOMPLETE",
        "empirical_pipeline_complete": all_complete,
        "paper_claim_ready": all_complete,
        "paper_claim_ready_reason": (
            "every hash-bound empirical stage validated"
            if all_complete
            else "one or more external empirical stages remain incomplete"
        ),
        "next_actionable_stages": next_actionable,
        "stages": rows,
        "local_gpu_actions_performed": False,
    }
