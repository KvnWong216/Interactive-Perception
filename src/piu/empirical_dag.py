"""Machine-verifiable readiness DAG for the external PIU experiment."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .spatial_prefix import validate_feature_arrays
from .formal_attempt import (
    load_formal_schedule,
    validate_attempt_close,
    validate_attempt_ticket,
)
from .formal_design import validate_development_episode
from .contracts import load_public_transitions
from .policy_identity import load_checkpoint_identity, validate_server_metadata
from .primitive_registry import validate_external_execution_risk_budget
from .splits import (
    load_split_manifest,
    role_to_split,
    validate_learning_collection_budget,
)
from .statistics import load_formal_outcomes

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
        if format_name == "text":
            value = path.read_text()
            if not value:
                raise ValueError("text artifact is empty")
            return value, []
    except Exception as exc:
        return None, [f"cannot parse {format_name}: {exc}"]
    return None, [f"unsupported artifact format {format_name!r}"]


def _configured_path(
    rule: Mapping[str, Any], name: str, *, repository_root: Path
) -> Path:
    raw = rule.get(name)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"validator requires configured path {name}")
    return _resolve(raw, repository_root=repository_root)


def _referenced_path(
    value: Any, name: str, *, repository_root: Path
) -> Path:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an artifact reference")
    path = _resolve(str(value.get("path", "")), repository_root=repository_root)
    if not path.is_file() or value.get("sha256") != sha256(path):
        raise ValueError(f"{name} differs from its content hash")
    return path


def _portable(path: Path, *, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root))
    except ValueError:
        return str(path.resolve())


def _validate_external_budget(
    value: Mapping[str, Any], rule: Mapping[str, Any], *, repository_root: Path
) -> None:
    registry_path = _configured_path(
        rule, "baseline_registry", repository_root=repository_root
    )
    registry = yaml.safe_load(registry_path.read_text())
    if registry.get("schema_version") != "piu.baseline-registry.v1":
        raise ValueError("external risk budget baseline registry is unsupported")
    maximum = registry.get("shared_contract", {}).get(
        "maximum_controller_decisions"
    )
    validate_external_execution_risk_budget(
        value, maximum_physical_dispatches=maximum
    )


def _validate_external_pi05_endpoint(
    value: Mapping[str, Any], rule: Mapping[str, Any], *, repository_root: Path
) -> None:
    if set(value) != {
        "schema_version",
        "status",
        "endpoint",
        "identity",
        "checkpoint_identity",
        "action_probe",
    }:
        raise ValueError("endpoint check fields are not closed")
    identity_path = _configured_path(
        rule, "checkpoint_identity", repository_root=repository_root
    )
    identity = load_checkpoint_identity(identity_path)
    reference = value.get("checkpoint_identity")
    if not isinstance(reference, Mapping) or (
        _resolve(str(reference.get("path", "")), repository_root=repository_root)
        .resolve()
        != identity_path.resolve()
        or reference.get("sha256") != sha256(identity_path)
    ):
        raise ValueError("endpoint check uses another checkpoint identity")
    metadata = value.get("identity")
    if not isinstance(metadata, Mapping):
        raise TypeError("endpoint check lacks server metadata")
    validate_server_metadata(metadata, identity)
    endpoint = value.get("endpoint")
    if not isinstance(endpoint, Mapping):
        raise TypeError("endpoint check lacks endpoint identity")
    host = " ".join(str(endpoint.get("host", "")).split())
    port = endpoint.get("port")
    if (
        not host
        or not isinstance(port, int)
        or isinstance(port, bool)
        or not 1 <= port <= 65535
    ):
        raise ValueError("endpoint host/port are invalid")
    probe = value.get("action_probe")
    if not isinstance(probe, Mapping) or probe.get("finite") is not True:
        raise ValueError("endpoint check lacks a finite action probe")
    if set(probe) != {
        "source_report",
        "keyframe",
        "shape",
        "finite",
        "elapsed_seconds",
    }:
        raise ValueError("endpoint action probe fields are not closed")
    if not " ".join(str(probe.get("keyframe", "")).split()):
        raise ValueError("endpoint action probe lacks a keyframe identity")
    shape = probe.get("shape")
    elapsed = probe.get("elapsed_seconds")
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 1
            for item in shape
        )
        or not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ValueError("endpoint action probe shape/timing are invalid")
    if not isinstance(probe.get("source_report"), Mapping):
        raise TypeError("endpoint action probe lacks hash-bound source report")


def _validate_prompted_vlm_identity(value: Mapping[str, Any]) -> None:
    metadata = value.get("server_metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("prompted-VLM identity lacks server metadata")
    if metadata.get("schema_version") != "piu.prompted-vlm-router-server.v1":
        raise ValueError("prompted-VLM server metadata schema is unsupported")
    if any(
        not " ".join(str(metadata.get(name, "")).split())
        for name in ("model_id", "revision")
    ):
        raise ValueError("prompted-VLM identity requires model ID and revision")
    capabilities = metadata.get("capabilities")
    if (
        not isinstance(capabilities, list)
        or "public_candidate_routing_v1" not in capabilities
        or any(not isinstance(item, str) for item in capabilities)
    ):
        raise ValueError("prompted-VLM identity lacks public routing capability")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_prompted_vlm_probe(
    value: Mapping[str, Any], *, repository_root: Path
) -> None:
    if (
        value.get("method_id") != "B1"
        or value.get("split") != "development"
        or value.get("evaluator_labels_loaded") is not False
        or value.get("online_oracle_inputs") != []
        or value.get("manual_confidence_threshold") is not None
        or value.get("paper_method_claim_allowed") is not False
    ):
        raise ValueError("prompted-VLM probe crossed its public development firewall")
    external = value.get("external_router")
    if not isinstance(external, Mapping):
        raise TypeError("prompted-VLM probe lacks external router provenance")
    identity_ref = external.get("identity")
    if not isinstance(identity_ref, Mapping):
        raise TypeError("prompted-VLM probe lacks router identity reference")
    identity_path = _resolve(
        str(identity_ref.get("path", "")), repository_root=repository_root
    )
    identity = json.loads(identity_path.read_text())
    _validate_prompted_vlm_identity(identity)
    if external.get("server_metadata") != identity["server_metadata"]:
        raise ValueError("prompted-VLM probe server differs from frozen identity")
    response = external.get("response")
    if not isinstance(response, Mapping) or set(response) != {
        "schema_version",
        "request_sha256",
        "selected_candidate_id",
    }:
        raise ValueError("prompted-VLM probe lacks the exact bounded response")
    if (
        response.get("schema_version") != "piu.prompted-vlm-router-response.v1"
        or response.get("request_sha256") != external.get("request_sha256")
        or _canonical_sha256(response) != external.get("response_sha256")
    ):
        raise ValueError("prompted-VLM response transcript differs from its hashes")
    decisions = value.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != 1:
        raise ValueError("prompted-VLM probe must contain one decision")
    decision = decisions[0]
    if not isinstance(decision, Mapping):
        raise TypeError("prompted-VLM probe decision must be a mapping")
    response_id = " ".join(str(response.get("selected_candidate_id", "")).split())
    candidates = decision.get("public_candidates")
    if not isinstance(candidates, list):
        raise TypeError("prompted-VLM probe lacks its public candidates")
    matches = [row for row in candidates if row.get("candidate_id") == response_id]
    selected_id = decision.get("selected_candidate_id")
    expected_selected = (
        response_id
        if len(matches) == 1 and decision.get("decision_kind") != "ABSTAIN"
        else None
    )
    if selected_id != expected_selected:
        raise ValueError("prompted-VLM decision differs from the bounded response")
    inputs = value.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {"public_transition"}:
        raise TypeError("prompted-VLM probe lacks closed public input provenance")
    public_path = _referenced_path(
        inputs["public_transition"],
        "prompted-VLM public transition",
        repository_root=repository_root,
    )
    public_rows = [
        row
        for row in load_public_transitions(public_path)
        if row.sample_id == decision.get("sample_id")
    ]
    if (
        len(public_rows) != 1
        or public_rows[0].split.value != "development"
        or public_rows[0].initial_state_group
        != decision.get("initial_state_group")
        or [dict(item) for item in public_rows[0].candidate_actions] != candidates
        or value.get("initial_state_groups")
        != [public_rows[0].initial_state_group]
    ):
        raise ValueError("prompted-VLM probe differs from its public transition")


def _validate_development_episode_arm(
    values: Sequence[Any],
    rule: Mapping[str, Any],
    *,
    repository_root: Path,
    split_manifest: Mapping[str, Any] | None,
) -> None:
    if split_manifest is None:
        raise ValueError("development episode validation requires a split manifest")
    method_id = str(rule.get("method_id", ""))
    registry_path = _configured_path(
        rule, "baseline_registry", repository_root=repository_root
    )
    registry = yaml.safe_load(registry_path.read_text())
    identity_path = _resolve(
        str(registry.get("shared_contract", {}).get("checkpoint_identity", "")),
        repository_root=repository_root,
    )
    identity_digest = sha256(identity_path)
    assignments = {
        str(row["initial_state_group"]): int(row["seed"])
        for row in split_manifest["assignments"]
        if row["split_role"] == "development"
    }
    groups: set[str] = set()
    source_hashes: set[str] = set()
    for item in values:
        if not isinstance(item, Mapping):
            raise TypeError("development episode rows must be mappings")
        row = validate_development_episode(
            item, method_id=method_id, outcome="task_success"
        )
        group = str(row["initial_state_group"])
        if group in groups:
            raise ValueError("development episode arm duplicates a group")
        source = item.get("source_state")
        identity = item.get("policy_identity")
        if not isinstance(source, Mapping) or not isinstance(identity, Mapping):
            raise TypeError("development episode lacks source/policy provenance")
        source_digest = str(source.get("sha256", ""))
        if (
            group not in assignments
            or row["simulator_seed"] != assignments[group]
            or identity.get("sha256") != identity_digest
            or source_digest in source_hashes
        ):
            raise ValueError(
                "development episode differs from its group, seed, policy, or state"
            )
        groups.add(group)
        source_hashes.add(source_digest)
    if groups != set(assignments):
        raise ValueError("development episode arm differs from complete allocation")


def _validate_scheduled_outcomes(
    path: Path,
    rule: Mapping[str, Any],
    *, repository_root: Path,
) -> None:
    schedule_path = _configured_path(
        rule, "formal_schedule", repository_root=repository_root
    )
    schedule = load_formal_schedule(schedule_path, repository_root=repository_root)
    rows = load_formal_outcomes(path)
    selected_method = rule.get("method_id")
    entries = [
        item
        for item in schedule["entries"]
        if selected_method is None or item["method_id"] == selected_method
    ]
    expected = {
        (str(item["initial_state_group"]), str(item["method_id"])): item
        for item in entries
    }
    observed = {
        (str(item["initial_state_group"]), str(item["method_id"])): item
        for item in rows
    }
    if observed.keys() != expected.keys() or len(rows) != len(expected):
        raise ValueError("formal outcomes differ from the complete scheduled denominator")
    policy = schedule["inputs"]["policy_identity"]["sha256"]
    for key, row in observed.items():
        entry = expected[key]
        if (
            row["simulator_seed"] != entry["simulator_seed"]
            or row["source_state_sha256"] != entry["source_state"]["sha256"]
            or row["policy_identity_sha256"] != policy
        ):
            raise ValueError("formal outcome identity differs from its schedule entry")
        episode_path = _referenced_path(
            row.get("episode"), "formal outcome episode", repository_root=repository_root
        )
        authorization_path = _referenced_path(
            row.get("sealed_authorization"),
            "formal row authorization",
            repository_root=repository_root,
        )
        ticket_path = _referenced_path(
            row.get("formal_attempt_ticket"),
            "formal attempt ticket",
            repository_root=repository_root,
        )
        close_path = _referenced_path(
            row.get("formal_attempt_close"),
            "formal attempt close",
            repository_root=repository_root,
        )
        episode = json.loads(episode_path.read_text())
        if (
            episode.get("schema_version") != "piu.closed-loop-episode.v1"
            or episode.get("method_id") != row["method_id"]
            or episode.get("initial_state_group") != row["initial_state_group"]
            or episode.get("simulator_seed") != row["simulator_seed"]
            or episode.get("split") != "sealed_test"
            or episode.get("evidence_class") != row["evidence_class"]
            or episode.get("rollout_status") != row["rollout_status"]
            or episode.get("outcomes") != row["outcomes"]
        ):
            raise ValueError("formal outcome row differs from its sealed episode")
        source_path = _referenced_path(
            episode.get("source_state"),
            "formal episode source state",
            repository_root=repository_root,
        )
        history_path = _referenced_path(
            episode.get("public_action_history"),
            "formal episode action history",
            repository_root=repository_root,
        )
        identity_path = _referenced_path(
            episode.get("policy_identity"),
            "formal episode policy identity",
            repository_root=repository_root,
        )
        if (
            sha256(source_path) != row["source_state_sha256"]
            or sha256(history_path) != row["action_history_sha256"]
            or sha256(identity_path) != row["policy_identity_sha256"]
        ):
            raise ValueError("formal episode provenance differs from its outcome row")
        ticket, ticket_schedule = validate_attempt_ticket(
            ticket_path,
            repository_root=repository_root,
            method_id=str(row["method_id"]),
            initial_state_group=str(row["initial_state_group"]),
            simulator_seed=int(row["simulator_seed"]),
            source_state=source_path,
            output_dir=episode_path.parent,
            allow_closed=True,
        )
        if (
            ticket_schedule != schedule
            or ticket.get("entry") != entry
            or ticket.get("execution_index") != entry["execution_index"]
        ):
            raise ValueError("formal attempt ticket differs from the scheduled cell")
        validate_attempt_close(
            close_path, ticket_path=ticket_path, episode_path=episode_path
        )
        authorization = json.loads(authorization_path.read_text())
        row_output = _resolve(
            str(authorization.get("single_use_output", "")),
            repository_root=repository_root,
        )
        if not row_output.is_file() or load_formal_outcomes(row_output) != [row]:
            raise ValueError("formal matrix row differs from its single-use artifact")
        expected_authorization = {
            "episode_sha256": sha256(episode_path),
            "source_state_sha256": row["source_state_sha256"],
            "action_history_sha256": row["action_history_sha256"],
            "policy_identity_sha256": row["policy_identity_sha256"],
            "method_id": row["method_id"],
            "formal_attempt_close_sha256": sha256(close_path),
            "single_use_output": _portable(
                row_output, repository_root=repository_root
            ),
        }
        if (
            authorization.get("schema_version")
            != "piu.formal-row-sealed-authorization.v1"
            or any(
                authorization.get(name) != required
                for name, required in expected_authorization.items()
            )
        ):
            raise ValueError("formal row authorization differs from the scheduled row")


def _validate_evidence_bound_svg(
    value: str, rule: Mapping[str, Any], *, repository_root: Path
) -> None:
    table_path = _configured_path(
        rule, "evidence_table", repository_root=repository_root
    )
    marker = f"evidence-table-sha256:{sha256(table_path)}"
    if (
        not value.startswith("<svg")
        or not value.rstrip().endswith("</svg>")
        or 'role="img"' not in value
        or value.count(marker) != 1
    ):
        raise ValueError("paper SVG is not uniquely bound to the evidence table")


def _validate_custom(
    value: Any,
    path: Path,
    rule: Mapping[str, Any],
    *,
    repository_root: Path,
    split_manifest: Mapping[str, Any] | None,
) -> None:
    validator = rule.get("validator")
    if validator in {None, "learning_collection_budget"}:
        return
    if validator == "external_execution_risk_budget":
        _validate_external_budget(value, rule, repository_root=repository_root)
    elif validator == "external_pi05_endpoint":
        _validate_external_pi05_endpoint(value, rule, repository_root=repository_root)
    elif validator == "prompted_vlm_identity":
        _validate_prompted_vlm_identity(value)
    elif validator == "prompted_vlm_probe":
        _validate_prompted_vlm_probe(value, repository_root=repository_root)
    elif validator == "development_episode_arm":
        _validate_development_episode_arm(
            value,
            rule,
            repository_root=repository_root,
            split_manifest=split_manifest,
        )
    elif validator == "formal_schedule":
        load_formal_schedule(path, repository_root=repository_root)
    elif validator == "scheduled_formal_outcomes":
        _validate_scheduled_outcomes(path, rule, repository_root=repository_root)
    elif validator == "evidence_bound_svg":
        if not isinstance(value, str):
            raise TypeError("evidence-bound SVG must be a text artifact")
        _validate_evidence_bound_svg(value, rule, repository_root=repository_root)
    else:
        raise ValueError(f"unknown empirical artifact validator {validator!r}")


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
    try:
        _validate_custom(
            value,
            path,
            rule,
            repository_root=repository_root,
            split_manifest=split_manifest,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        errors.append(str(exc))
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
