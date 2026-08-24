"""Fail-closed execution contract for the frozen public-input S03 schedule.

Validation and request construction are side-effect free.  The helpers that
write receipts or invoke the backend are called only by the canonical runner
after its explicit outcome-write authorization flag has been supplied.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .s03_preparation import (
    S02_OUTCOME_INDEX_PATH,
    S02_OUTCOME_INDEX_SHA256,
    canonical_sha256,
    sha256,
    validate_s03_input_manifest,
    validate_s03_offline_schedule,
)


MODEL_IDENTITY_SCHEMA = "piu.s03-model-identity.v1"
RUNNER_CONTRACT_SCHEMA = "piu.s03-runner-contract.v1"
REQUEST_SCHEMA = "piu.s03-public-inference-request.v1"
RECORD_SCHEMA = "piu.s03-record-artifact.v1"
RECEIPT_SCHEMA = "piu.s03-execution-receipt.v1"

DEFAULT_SCHEDULE = Path(
    "results/method/piu_s03_perception_decision_schedule_v1.json"
)
DEFAULT_CONTRACT = Path("configs/experiments/piu_s03_runner_contract_v1.json")
DEFAULT_IDENTITY = Path("configs/experiments/piu_s03_model_identity_v1.json")
DEFAULT_OUTPUT_ROOT = Path("runs/piu_s03_perception_decision_v1")

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_FORBIDDEN_POLICY_KEYS = {
    "simulator_semantic_id",
    "simulator_instance_id",
    "simulator_segmentation",
    "target_mask",
    "object_pose",
    "oracle_target_marker",
    "oracle_target_location",
    "container_membership",
    "evaluator_only_fields",
    "task_predicate",
    "reward",
    "success",
    "failure",
    "expected_route",
    "route_label",
    "sufficiency_label",
    "revealed_label",
}
_RECORD_KEYS = {
    "schema_version",
    "record_id",
    "schedule_index",
    "linked_s02_index",
    "subtest",
    "stratum",
    "schedule",
    "manifest",
    "manifest_record_sha256",
    "public_input_digest",
    "request_sha256",
    "model_identity",
    "status",
    "inference_executed",
    "outcome_present",
    "prediction",
    "outcome",
    "artifacts",
    "infrastructure_failure",
    "privileged_input_violation",
    "online_oracle_inputs",
    "physical_dispatch_executed",
    "s02_artifacts_modified",
    "paper_claim_ready",
    "created_at",
}
_RECEIPT_KEYS = {
    "schema_version",
    "receipt_phase",
    "status",
    "execution_index",
    "record_id",
    "schedule",
    "manifest",
    "model_identity",
    "public_input_digest",
    "request_sha256",
    "single_use_output_dir",
    "previous_close_sha256",
    "record_artifact",
    "outcome_present",
    "inference_executed",
    "consumes_execution_index",
    "created_at",
}


def _resolve(path: str | Path, *, repository_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repository_root / value


def _portable(path: Path, *, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} must be a mapping")
    return value


def _load_json(path: Path, location: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    return dict(_mapping(value, location))


def _reference(path: Path, *, repository_root: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _portable(path, repository_root=repository_root),
        "sha256": sha256(path),
    }


def _verify_reference(
    value: Any, location: str, *, repository_root: Path
) -> Path:
    ref = _mapping(value, location)
    if set(ref) != {"path", "sha256"}:
        raise ValueError(f"{location} must be an exact path/SHA-256 reference")
    digest = str(ref.get("sha256", ""))
    path = _resolve(str(ref.get("path", "")), repository_root=repository_root)
    if not _SHA256.fullmatch(digest) or not path.is_file() or sha256(path) != digest:
        raise ValueError(f"{location} differs from its frozen bytes")
    return path


def _validate_timestamp(value: Any, location: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{location} must be an RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{location} is not an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{location} must include a timezone")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _scan_policy_input(value: Any, *, location: str = "policy_request") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normalized_key(str(key))
            if normalized in _FORBIDDEN_POLICY_KEYS:
                raise ValueError(f"{location} contains forbidden policy field {key!r}")
            _scan_policy_input(child, location=f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _scan_policy_input(child, location=f"{location}[{index}]")


def _verify_schema_document(
    ref: Any, expected_id: str, *, repository_root: Path
) -> Path:
    path = _verify_reference(ref, f"schema {expected_id}", repository_root=repository_root)
    value = _load_json(path, f"schema {expected_id}")
    if value.get("$id") != expected_id or value.get("additionalProperties") is not False:
        raise ValueError(f"schema document is not the strict frozen {expected_id}")
    return path


def _validate_named_artifacts(
    value: Any, location: str, *, repository_root: Path
) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must contain at least one artifact")
    result: list[dict[str, str]] = []
    ids: set[str] = set()
    for index, raw in enumerate(value):
        row = _mapping(raw, f"{location}[{index}]")
        if set(row) != {"artifact_id", "path", "sha256"}:
            raise ValueError(f"{location}[{index}] has unexpected fields")
        artifact_id = str(row["artifact_id"])
        if not artifact_id or artifact_id in ids:
            raise ValueError(f"{location} artifact IDs must be unique")
        ids.add(artifact_id)
        _verify_reference(
            {"path": row["path"], "sha256": row["sha256"]},
            f"{location}[{index}]",
            repository_root=repository_root,
        )
        result.append(dict(row))
    return result


def validate_s03_model_identity_contract(
    path: Path, *, repository_root: Path
) -> dict[str, Any]:
    """Validate every model, checkpoint, calibrator, code, and sampling stamp."""

    identity = _load_json(path, "S03 model identity")
    required = {
        "schema_version",
        "status",
        "identity_id",
        "model_id",
        "backend",
        "checkpoint",
        "calibrator",
        "code",
        "runtime",
        "sampling",
        "policy_input_firewall",
        "execution_scope",
        "paper_claim_ready",
    }
    if set(identity) != required:
        raise ValueError("S03 model identity fields differ from its frozen schema")
    if (
        identity["schema_version"] != MODEL_IDENTITY_SCHEMA
        or identity["status"] != "FROZEN_BEFORE_S03_OUTCOMES"
        or identity["paper_claim_ready"] is not False
        or not str(identity["identity_id"])
        or not str(identity["model_id"])
    ):
        raise ValueError("S03 model identity crossed the pre-outcome boundary")

    backend = _mapping(identity["backend"], "S03 backend")
    if set(backend) != {
        "backend_id",
        "entrypoint",
        "report_schema",
        "model_paths",
        "action_registry",
        "public_outcome_adapter",
        "legacy_oracle_dependency",
    }:
        raise ValueError("S03 backend identity fields differ")
    _verify_reference(backend["entrypoint"], "S03 backend entrypoint", repository_root=repository_root)
    _verify_reference(backend["action_registry"], "S03 action registry", repository_root=repository_root)
    model_paths = _mapping(backend["model_paths"], "S03 backend model paths")
    if set(model_paths) != {"grounding_dino", "sam", "dinov2", "siglip", "qwen"}:
        raise ValueError("S03 backend model path set differs")
    if any(not _resolve(str(value), repository_root=repository_root).is_dir() for value in model_paths.values()):
        raise FileNotFoundError("one or more S03 backend model directories are missing")
    if backend["report_schema"] != "interaction-uncertainty.qwen-observation-pipeline.v0":
        raise ValueError("S03 backend report schema differs")
    if backend["legacy_oracle_dependency"] is not False:
        raise ValueError("legacy Oracle dependency is forbidden for public S03")
    adapter = _mapping(backend["public_outcome_adapter"], "public outcome adapter")
    if adapter != {
        "adapter_id": "no_frozen_six_frame_critic_input_v1",
        "status": "CONSERVATIVE_AMBIGUOUS_ONLY",
        "prediction_set": ["AMBIGUOUS"],
        "rationale": "frozen_schedule_exposes_no_complete_calibrated_rgb_critic_input",
    }:
        raise ValueError("public outcome adapter is not the frozen fail-closed identity")

    checkpoint = _mapping(identity["checkpoint"], "S03 checkpoint identity")
    if set(checkpoint) != {"algorithm", "artifacts", "digest"} or checkpoint["algorithm"] != (
        "sha256_of_canonical_json_ordered_artifact_refs"
    ):
        raise ValueError("S03 checkpoint identity is malformed")
    checkpoint_artifacts = _validate_named_artifacts(
        checkpoint["artifacts"], "S03 checkpoint artifacts", repository_root=repository_root
    )
    if checkpoint["digest"] != canonical_sha256(checkpoint_artifacts):
        raise ValueError("S03 checkpoint aggregate digest differs")

    calibrator = _mapping(identity["calibrator"], "S03 calibrator identity")
    if set(calibrator) != {
        "calibrator_id", "version", "status", "artifact", "confidence_threshold"
    }:
        raise ValueError("S03 calibrator identity fields differ")
    if calibrator != {
        "calibrator_id": "none_public_qwen_pipeline_v0",
        "version": "0",
        "status": "UNCALIBRATED_DEVELOPMENT_ONLY",
        "artifact": None,
        "confidence_threshold": None,
    }:
        raise ValueError("S03 must disclose its absent calibrator without a threshold")

    code = _mapping(identity["code"], "S03 code identity")
    if set(code) != {"commit", "algorithm", "artifacts", "digest"}:
        raise ValueError("S03 code identity fields differ")
    commit = str(code["commit"])
    if not _COMMIT.fullmatch(commit):
        raise ValueError("S03 code identity has no exact source commit")
    code_artifacts = _validate_named_artifacts(
        code["artifacts"], "S03 code artifacts", repository_root=repository_root
    )
    if code["algorithm"] != "sha256_of_canonical_json_ordered_artifact_refs" or code[
        "digest"
    ] != canonical_sha256(code_artifacts):
        raise ValueError("S03 code aggregate digest differs")
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=repository_root,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("S03 code commit is not present in this repository") from exc
    for row in code_artifacts:
        committed = subprocess.run(
            ["git", "show", f"{commit}:{row['path']}"],
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != row["sha256"]:
            raise ValueError(f"S03 code artifact is not byte-bound by {commit}: {row['path']}")

    runtime = _mapping(identity["runtime"], "S03 runtime identity")
    if set(runtime) != {"device", "precision", "max_proposals_per_view", "offline_only"}:
        raise ValueError("S03 runtime identity fields differ")
    if (
        runtime["device"] not in {"cuda", "cpu"}
        or runtime["precision"] != "backend_frozen_defaults"
        or not isinstance(runtime["max_proposals_per_view"], int)
        or runtime["max_proposals_per_view"] <= 0
        or runtime["offline_only"] is not True
    ):
        raise ValueError("S03 runtime identity is invalid")
    sampling = _mapping(identity["sampling"], "S03 sampling identity")
    if sampling != {
        "do_sample": False,
        "temperature": None,
        "seed_base": 26082300,
        "per_record_seed_rule": "seed_base_plus_schedule_index",
    }:
        raise ValueError("S03 sampling identity differs from deterministic decoding")

    firewall = _mapping(identity["policy_input_firewall"], "S03 policy firewall")
    if (
        set(firewall) != {"allowed_fields", "forbidden_fields", "online_oracle_inputs"}
        or firewall.get("online_oracle_inputs") != []
        or sorted(map(_normalized_key, firewall.get("forbidden_fields", [])))
        != sorted(_FORBIDDEN_POLICY_KEYS)
    ):
        raise ValueError("S03 identity does not freeze the complete policy firewall")
    scope = _mapping(identity["execution_scope"], "S03 execution scope")
    if scope != {
        "physical_rollout": False,
        "pi05_action_calls": False,
        "environment_steps": False,
        "post_decision_dispatch": False,
        "legacy_oracle": False,
        "formal_method_claim": False,
    }:
        raise ValueError("S03 identity permits an operation outside offline public inference")
    return identity


def validate_s03_no_legacy_oracle_dependency(identity: Mapping[str, Any]) -> None:
    backend = _mapping(identity.get("backend"), "S03 backend")
    scope = _mapping(identity.get("execution_scope"), "S03 execution scope")
    paths = [
        str(backend.get("entrypoint", {}).get("path", "")),
        str(backend.get("backend_id", "")),
        *map(str, _mapping(backend.get("model_paths"), "model paths").values()),
    ]
    if backend.get("legacy_oracle_dependency") is not False or scope.get("legacy_oracle") is not False:
        raise ValueError("legacy Oracle is actionable in the S03 runner identity")
    forbidden_path_parts = ("run_oracle", "oracle_target_prompt", "instance_seg")
    if any(any(part in value.lower() for part in forbidden_path_parts) for value in paths):
        raise ValueError("S03 backend references a legacy Oracle implementation")


def validate_s03_runner_contract(
    path: Path, *, repository_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _load_json(path, "S03 runner contract")
    required = {
        "schema_version",
        "status",
        "runner_id",
        "frozen_inputs",
        "model_identity",
        "schemas",
        "runner",
        "output",
        "policies",
        "paper_claim_ready",
    }
    if set(contract) != required:
        raise ValueError("S03 runner contract fields differ")
    if (
        contract["schema_version"] != RUNNER_CONTRACT_SCHEMA
        or contract["status"] != "FROZEN_READY_BEFORE_S03_OUTCOMES"
        or contract["paper_claim_ready"] is not False
    ):
        raise ValueError("S03 runner contract crossed the pre-outcome boundary")
    frozen = _mapping(contract["frozen_inputs"], "S03 runner frozen inputs")
    if set(frozen) != {"schedule", "manifest", "dag_v2", "runbook", "s02_schedule", "s02_certificate", "s02_outcome_index"}:
        raise ValueError("S03 runner frozen inputs differ")
    paths = {
        name: _verify_reference(ref, f"S03 frozen {name}", repository_root=repository_root)
        for name, ref in frozen.items()
    }
    schedule = validate_s03_offline_schedule(paths["schedule"], repository_root=repository_root)
    manifest = validate_s03_input_manifest(paths["manifest"], repository_root=repository_root)
    if schedule["input_manifest"] != frozen["manifest"]:
        raise ValueError("S03 runner schedule and manifest references differ")
    for name in ("runbook", "dag_v2", "s02_schedule", "s02_certificate", "s02_outcome_index"):
        schedule_name = "dag_amendment" if name == "dag_v2" else name
        if schedule["upstream"][schedule_name]["sha256"] != frozen[name]["sha256"]:
            raise ValueError(f"S03 runner {name} differs from the frozen schedule")

    identity_path = _verify_reference(
        contract["model_identity"], "S03 model identity", repository_root=repository_root
    )
    identity = validate_s03_model_identity_contract(identity_path, repository_root=repository_root)
    validate_s03_no_legacy_oracle_dependency(identity)
    schemas = _mapping(contract["schemas"], "S03 schemas")
    if set(schemas) != {"model_identity", "record_artifact", "receipt"}:
        raise ValueError("S03 runner schema references differ")
    for name, expected in (
        ("model_identity", MODEL_IDENTITY_SCHEMA),
        ("record_artifact", RECORD_SCHEMA),
        ("receipt", RECEIPT_SCHEMA),
    ):
        _verify_schema_document(schemas[name], expected, repository_root=repository_root)
    runner = _mapping(contract["runner"], "S03 runner identity")
    if set(runner) != {"entrypoint", "implementation", "validator"}:
        raise ValueError("S03 runner source references differ")
    for name, ref in runner.items():
        _verify_reference(ref, f"S03 runner {name}", repository_root=repository_root)
    output = _mapping(contract["output"], "S03 output contract")
    if output != {
        "root": str(DEFAULT_OUTPUT_ROOT),
        "receipt_directory": "_receipts",
        "record_directory": "records",
        "record_artifact_name": "record.json",
        "infrastructure_failure_artifact_name": "infrastructure_failure.json",
    }:
        raise ValueError("S03 canonical output layout differs")
    policies = _mapping(contract["policies"], "S03 execution policies")
    if policies != {
        "ordered_execution": True,
        "single_use": True,
        "rerun": False,
        "skip": False,
        "replace": False,
        "infrastructure_failure_consumes_index_after_started_receipt": True,
        "validate_only_writes": False,
        "dry_run_writes": False,
        "outcome_write_requires_explicit_flag": True,
    }:
        raise ValueError("S03 single-use policy differs")
    return contract, identity


def model_identity_stamp(identity: Mapping[str, Any], schedule_index: int) -> dict[str, Any]:
    return {
        "identity_id": identity["identity_id"],
        "model_id": identity["model_id"],
        "checkpoint_digest": identity["checkpoint"]["digest"],
        "calibrator_id": identity["calibrator"]["calibrator_id"],
        "calibrator_version": identity["calibrator"]["version"],
        "code_commit": identity["code"]["commit"],
        "backend_id": identity["backend"]["backend_id"],
        "seed": identity["sampling"]["seed_base"] + schedule_index,
        "sampling": dict(identity["sampling"]),
    }


def build_s03_public_request(
    *,
    schedule: Mapping[str, Any],
    manifest: Mapping[str, Any],
    schedule_path: Path,
    manifest_path: Path,
    identity_path: Path,
    identity: Mapping[str, Any],
    execution_index: int,
    repository_root: Path,
) -> dict[str, Any]:
    records = schedule["records"]
    if isinstance(execution_index, bool) or not isinstance(execution_index, int) or not 0 <= execution_index < len(records):
        raise IndexError(f"S03 execution index must be in [0, {len(records) - 1}]")
    row = records[execution_index]
    manifest_row = manifest["records"][execution_index]
    if row["record_id"] != manifest_row["record_id"]:
        raise ValueError("S03 schedule/manifest record IDs differ")
    _scan_policy_input(row["input_artifacts"])
    if set(row["input_artifacts"]) != {
        "prompt",
        "candidate_registry",
        "observation_sequence",
    }:
        raise ValueError("S03 scheduled policy input contains an unknown field")
    policy_request = {
        "prompt": row["input_artifacts"]["prompt"],
        "candidate_registry": dict(row["input_artifacts"]["candidate_registry"]),
        "observation_sequence": list(row["input_artifacts"]["observation_sequence"]),
        "online_oracle_inputs": [],
    }
    _scan_policy_input({key: value for key, value in policy_request.items() if key != "online_oracle_inputs"})
    if canonical_sha256(row["input_artifacts"]) != row["public_input_digest"]:
        raise ValueError("S03 request input digest differs from the schedule")
    request = {
        "schema_version": REQUEST_SCHEMA,
        "schedule": _reference(schedule_path, repository_root=repository_root),
        "manifest": _reference(manifest_path, repository_root=repository_root),
        "model_identity": _reference(identity_path, repository_root=repository_root),
        "execution_index": execution_index,
        "record_id": row["record_id"],
        "manifest_record_sha256": row["manifest_record_sha256"],
        "public_input_digest": row["public_input_digest"],
        "model_identity_stamp": model_identity_stamp(identity, execution_index),
        "policy_request": policy_request,
        "execution_scope": {
            "offline_inference_only": True,
            "physical_dispatch": False,
            "pi05_action_calls": False,
            "environment_steps": False,
            "evaluator_private_inputs_visible_to_policy": False,
        },
    }
    return request


def validate_s03_runner_preflight(
    *,
    contract_path: Path,
    schedule_path: Path,
    execution_index: int,
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract, identity = validate_s03_runner_contract(
        contract_path, repository_root=repository_root
    )
    expected_schedule = _resolve(
        contract["frozen_inputs"]["schedule"]["path"], repository_root=repository_root
    )
    if schedule_path.resolve() != expected_schedule.resolve() or sha256(schedule_path) != contract[
        "frozen_inputs"
    ]["schedule"]["sha256"]:
        raise ValueError("runner was given a schedule other than the frozen public S03 schedule")
    schedule = validate_s03_offline_schedule(schedule_path, repository_root=repository_root)
    manifest_path = _resolve(schedule["input_manifest"]["path"], repository_root=repository_root)
    manifest = validate_s03_input_manifest(manifest_path, repository_root=repository_root)
    identity_path = _resolve(contract["model_identity"]["path"], repository_root=repository_root)
    request = build_s03_public_request(
        schedule=schedule,
        manifest=manifest,
        schedule_path=schedule_path,
        manifest_path=manifest_path,
        identity_path=identity_path,
        identity=identity,
        execution_index=execution_index,
        repository_root=repository_root,
    )
    return contract, identity, schedule, request


def _validate_identity_stamp(
    stamp: Any, *, identity: Mapping[str, Any], schedule_index: int
) -> None:
    if stamp != model_identity_stamp(identity, schedule_index):
        raise ValueError("S03 record model/checkpoint/calibrator identity differs")


def _validate_artifact_references(
    values: Any, *, repository_root: Path
) -> None:
    if not isinstance(values, list):
        raise TypeError("S03 record artifacts must be a list")
    seen: set[str] = set()
    for index, ref in enumerate(values):
        path = _verify_reference(ref, f"S03 record artifact[{index}]", repository_root=repository_root)
        portable = _portable(path, repository_root=repository_root)
        if portable in seen:
            raise ValueError("S03 record artifact references are duplicated")
        seen.add(portable)


def existing_artifact_references(
    root: Path, *, repository_root: Path
) -> list[dict[str, str]]:
    """Hash every already-materialized file under a record directory."""

    if not root.exists():
        return []
    return [
        _reference(path, repository_root=repository_root)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def validate_s03_outcome_schema(
    value_or_path: Mapping[str, Any] | Path,
    *,
    repository_root: Path,
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = (
        _load_json(value_or_path, "S03 record artifact")
        if isinstance(value_or_path, Path)
        else dict(_mapping(value_or_path, "S03 record artifact"))
    )
    if set(record) != _RECORD_KEYS or record.get("schema_version") != RECORD_SCHEMA:
        raise ValueError("S03 record artifact fields differ from the frozen schema")
    index = record.get("schedule_index")
    linked = record.get("linked_s02_index")
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < 620
        or isinstance(linked, bool)
        or not isinstance(linked, int)
        or not 0 <= linked < 124
    ):
        raise ValueError("S03 record indices are invalid")
    for name in ("schedule", "manifest"):
        _verify_reference(record[name], f"S03 record {name}", repository_root=repository_root)
    for name in ("manifest_record_sha256", "public_input_digest", "request_sha256"):
        if not _SHA256.fullmatch(str(record.get(name, ""))):
            raise ValueError(f"S03 record {name} is not SHA-256")
    if identity is not None:
        _validate_identity_stamp(record["model_identity"], identity=identity, schedule_index=index)
    elif not isinstance(record["model_identity"], Mapping):
        raise TypeError("S03 record model identity stamp is missing")
    _validate_timestamp(record["created_at"], "S03 record created_at")
    if (
        record["privileged_input_violation"] is not False
        or record["online_oracle_inputs"] != []
        or record["physical_dispatch_executed"] is not False
        or record["s02_artifacts_modified"] is not False
        or record["paper_claim_ready"] is not False
    ):
        raise ValueError("S03 record violates the public offline claim boundary")
    _validate_artifact_references(record["artifacts"], repository_root=repository_root)
    status = record["status"]
    if status == "EVALUATED":
        if (
            record["inference_executed"] is not True
            or record["outcome_present"] is not True
            or not isinstance(record["prediction"], Mapping)
            or not isinstance(record["outcome"], Mapping)
            or record["infrastructure_failure"] is not None
        ):
            raise ValueError("evaluated S03 record is incomplete")
    elif status == "INFRASTRUCTURE_FAILURE":
        failure = _mapping(record["infrastructure_failure"], "S03 infrastructure failure")
        if (
            set(failure) != {"stage", "category", "message", "consumes_execution_index", "adjudication"}
            or failure["consumes_execution_index"] is not True
            or record["outcome_present"] is not False
            or record["outcome"] is not None
            or (record["prediction"] is not None and not isinstance(record["prediction"], Mapping))
        ):
            raise ValueError("S03 infrastructure failure is not separately adjudicated")
    else:
        raise ValueError("S03 record has an unknown status")
    return record


def _receipt_paths(output_root: Path, index: int) -> tuple[Path, Path]:
    ledger = output_root / "_receipts"
    return ledger / f"{index:03d}.started.json", ledger / f"{index:03d}.closed.json"


def _record_dir(output_root: Path, index: int, record_id: str) -> Path:
    return output_root / "records" / f"{index:03d}_{record_id}"


def validate_s03_receipts(
    output_root: Path,
    *,
    schedule: Mapping[str, Any],
    schedule_path: Path,
    manifest_path: Path,
    identity_path: Path,
    identity: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    if not output_root.exists():
        return {"closed": 0, "in_flight": None, "next_execution_index": 0}
    ledger = output_root / "_receipts"
    records_root = output_root / "records"
    if not ledger.is_dir() or not records_root.is_dir():
        raise ValueError("S03 output root is partial or uses a noncanonical layout")
    expected_names = {
        path.name
        for index in range(620)
        for path in _receipt_paths(output_root, index)
    }
    unexpected = [path.name for path in ledger.iterdir() if not path.is_file() or path.name not in expected_names]
    if unexpected:
        raise ValueError(f"unexpected S03 receipt ledger entries: {sorted(unexpected)}")
    schedule_ref = _reference(schedule_path, repository_root=repository_root)
    manifest_ref = _reference(manifest_path, repository_root=repository_root)
    identity_ref = _reference(identity_path, repository_root=repository_root)
    manifest = validate_s03_input_manifest(
        manifest_path, repository_root=repository_root
    )
    previous_close: str | None = None
    closed = 0
    in_flight: int | None = None
    seen_records: set[str] = set()
    for index, row in enumerate(schedule["records"]):
        started_path, closed_path = _receipt_paths(output_root, index)
        if not started_path.exists():
            if closed_path.exists():
                raise ValueError("S03 close receipt exists without a start receipt")
            if any(_receipt_paths(output_root, later)[0].exists() or _receipt_paths(output_root, later)[1].exists() for later in range(index + 1, 620)):
                raise ValueError("S03 receipt ledger skipped or reordered an index")
            break
        started = _load_json(started_path, "S03 started receipt")
        if set(started) != _RECEIPT_KEYS or started != {
            "schema_version": RECEIPT_SCHEMA,
            "receipt_phase": "STARTED",
            "status": "STARTED",
            "execution_index": index,
            "record_id": row["record_id"],
            "schedule": schedule_ref,
            "manifest": manifest_ref,
            "model_identity": identity_ref,
            "public_input_digest": row["public_input_digest"],
            "request_sha256": started.get("request_sha256"),
            "single_use_output_dir": str(_record_dir(output_root, index, row["record_id"]).resolve()),
            "previous_close_sha256": previous_close,
            "record_artifact": None,
            "outcome_present": False,
            "inference_executed": False,
            "consumes_execution_index": True,
            "created_at": started.get("created_at"),
        }:
            raise ValueError(f"S03 started receipt differs at index {index}")
        if not _SHA256.fullmatch(str(started["request_sha256"])):
            raise ValueError("S03 started receipt request digest is invalid")
        expected_request = build_s03_public_request(
            schedule=schedule,
            manifest=manifest,
            schedule_path=schedule_path,
            manifest_path=manifest_path,
            identity_path=identity_path,
            identity=identity,
            execution_index=index,
            repository_root=repository_root,
        )
        if started["request_sha256"] != canonical_sha256(expected_request):
            raise ValueError("S03 started receipt does not bind its exact public request")
        _validate_timestamp(started["created_at"], "S03 started receipt created_at")
        if row["record_id"] in seen_records:
            raise ValueError("S03 receipt ledger duplicates a record ID")
        seen_records.add(row["record_id"])
        if not closed_path.exists():
            in_flight = index
            if any(_receipt_paths(output_root, later)[0].exists() or _receipt_paths(output_root, later)[1].exists() for later in range(index + 1, 620)):
                raise ValueError("S03 ledger advanced beyond an unclosed attempt")
            break
        closed_value = _load_json(closed_path, "S03 close receipt")
        if set(closed_value) != _RECEIPT_KEYS:
            raise ValueError("S03 close receipt fields differ")
        if (
            closed_value["schema_version"] != RECEIPT_SCHEMA
            or closed_value["receipt_phase"] != "CLOSED"
            or closed_value["status"] not in {"CLOSED_WITH_OUTCOME", "CLOSED_INFRASTRUCTURE_FAILURE"}
            or closed_value["execution_index"] != index
            or closed_value["record_id"] != row["record_id"]
            or closed_value["schedule"] != schedule_ref
            or closed_value["manifest"] != manifest_ref
            or closed_value["model_identity"] != identity_ref
            or closed_value["public_input_digest"] != row["public_input_digest"]
            or closed_value["request_sha256"] != started["request_sha256"]
            or closed_value["single_use_output_dir"] != started["single_use_output_dir"]
            or closed_value["previous_close_sha256"] != previous_close
            or closed_value["consumes_execution_index"] is not True
        ):
            raise ValueError(f"S03 close receipt differs at index {index}")
        _validate_timestamp(closed_value["created_at"], "S03 close receipt created_at")
        record_path = _verify_reference(
            closed_value["record_artifact"], "S03 close record artifact", repository_root=repository_root
        )
        record = validate_s03_outcome_schema(record_path, repository_root=repository_root, identity=identity)
        if (
            record["schedule_index"] != index
            or record["record_id"] != row["record_id"]
            or record["request_sha256"] != started["request_sha256"]
        ):
            raise ValueError("S03 close receipt references another record")
        expected_status = "CLOSED_WITH_OUTCOME" if record["status"] == "EVALUATED" else "CLOSED_INFRASTRUCTURE_FAILURE"
        if closed_value["status"] != expected_status or closed_value["outcome_present"] != record["outcome_present"] or closed_value["inference_executed"] != record["inference_executed"]:
            raise ValueError("S03 close receipt state differs from its record artifact")
        previous_close = sha256(closed_path)
        closed += 1
    return {
        "closed": closed,
        "in_flight": in_flight,
        "next_execution_index": in_flight if in_flight is not None else closed,
    }


def validate_s03_single_use_policy(
    output_root: Path,
    *,
    execution_index: int,
    ledger_status: Mapping[str, Any],
    record_id: str,
) -> None:
    if execution_index != ledger_status["next_execution_index"]:
        raise ValueError(
            f"S03 requires exact order; next index is {ledger_status['next_execution_index']}"
        )
    started, closed = _receipt_paths(output_root, execution_index)
    record_dir = _record_dir(output_root, execution_index, record_id)
    if started.exists() or closed.exists() or record_dir.exists():
        raise FileExistsError("S03 execution index was already started and cannot rerun")


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def build_started_receipt(
    *,
    request: Mapping[str, Any],
    output_root: Path,
    previous_close_sha256: str | None,
    repository_root: Path,
) -> dict[str, Any]:
    index = int(request["execution_index"])
    return {
        "schema_version": RECEIPT_SCHEMA,
        "receipt_phase": "STARTED",
        "status": "STARTED",
        "execution_index": index,
        "record_id": request["record_id"],
        "schedule": request["schedule"],
        "manifest": request["manifest"],
        "model_identity": request["model_identity"],
        "public_input_digest": request["public_input_digest"],
        "request_sha256": canonical_sha256(request),
        "single_use_output_dir": str(_record_dir(output_root, index, request["record_id"]).resolve()),
        "previous_close_sha256": previous_close_sha256,
        "record_artifact": None,
        "outcome_present": False,
        "inference_executed": False,
        "consumes_execution_index": True,
        "created_at": utc_now(),
    }


def _route(report: Mapping[str, Any]) -> str:
    action = str(_mapping(report.get("selected_action"), "public selected action").get("action", ""))
    return {"ACT": "ACT", "OPEN_CONTAINER": "OPEN", "STOP": "STOP"}.get(action, "ABSTAIN")


def _entropy(report: Mapping[str, Any]) -> float:
    field = _mapping(report.get("prompt_conditioned_interaction_field"), "public belief")
    belief = _mapping(field.get("target_location_belief"), "target location belief")
    probabilities = [float(value) for value in belief.values()]
    if not probabilities or any(not math.isfinite(value) or value < 0 for value in probabilities):
        raise ValueError("public belief contains invalid probabilities")
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("public belief has no probability mass")
    normalized = [value / total for value in probabilities]
    raw = -sum(value * math.log(value) for value in normalized if value > 0)
    return raw / math.log(len(normalized)) if len(normalized) > 1 else 0.0


def _run_infer(
    *,
    observation: Mapping[str, Any],
    prompt: str,
    identity: Mapping[str, Any],
    output_dir: Path,
    repository_root: Path,
    previous_report: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    backend = identity["backend"]
    runtime = identity["runtime"]
    rgb = observation["rgb"]
    output = output_dir / "report.json"
    assets = output_dir / "assets"
    command = [
        sys.executable,
        str(_resolve(backend["entrypoint"]["path"], repository_root=repository_root)),
        "--agentview", str(_resolve(rgb["agentview"]["path"], repository_root=repository_root)),
        "--wrist", str(_resolve(rgb["wrist"]["path"], repository_root=repository_root)),
        "--prompt", prompt,
        "--asset-dir", str(assets),
        "--output", str(output),
        "--device", runtime["device"],
        "--max-proposals-per-view", str(runtime["max_proposals_per_view"]),
        "--action-registry", str(_resolve(backend["action_registry"]["path"], repository_root=repository_root)),
    ]
    for flag, name in (
        ("--grounding-model", "grounding_dino"),
        ("--sam-model", "sam"),
        ("--dino-model", "dinov2"),
        ("--siglip-model", "siglip"),
        ("--qwen-model", "qwen"),
    ):
        command.extend([flag, str(_resolve(backend["model_paths"][name], repository_root=repository_root))])
    history = observation["public_action_history"]
    if history["artifact"] is not None:
        command.extend(["--history", str(_resolve(history["artifact"]["path"], repository_root=repository_root))])
    if previous_report is not None:
        command.extend(
            [
                "--previous-report", str(previous_report),
                "--executed-action", "OPEN_CONTAINER",
                "--observed-outcome", "AMBIGUOUS",
            ]
        )
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        command,
        cwd=repository_root,
        env=environment,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    log = output_dir / "backend_log.json"
    _write_new_json(
        log,
        {"command": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr},
    )
    report = _load_json(output, "S03 public backend report")
    if report.get("schema_version") != backend["report_schema"] or report.get("online_oracle_inputs") != []:
        raise ValueError("S03 backend report violates its public report contract")
    return report, existing_artifact_references(
        output_dir, repository_root=repository_root
    )


def _private_s02_label(
    manifest_row: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    receipt_path = _verify_reference(
        {"path": manifest_row["s02_receipt_provenance"]["path"], "sha256": manifest_row["s02_receipt_provenance"]["sha256"]},
        "S02 evaluator receipt",
        repository_root=repository_root,
    )
    receipt = _load_json(receipt_path, "S02 execution receipt")
    semantic_path = _verify_reference(receipt["semantic_report"], "S02 semantic report", repository_root=repository_root)
    semantic = _load_json(semantic_path, "S02 semantic report")
    evaluator = _mapping(semantic.get("evaluator"), "S02 evaluator-only subtree")
    visibility = _mapping(evaluator.get("target_visibility_pixels"), "S02 target visibility")
    initial = sum(int(value) for value in _mapping(visibility.get("initial"), "initial visibility").values())
    final = sum(int(value) for value in _mapping(visibility.get("final"), "final visibility").values())
    outcome_index_path = _resolve(S02_OUTCOME_INDEX_PATH, repository_root=repository_root)
    if sha256(outcome_index_path) != S02_OUTCOME_INDEX_SHA256:
        raise ValueError("S02 evaluator-only outcome index differs from its frozen bytes")
    outcome_rows = [json.loads(line) for line in outcome_index_path.read_text().splitlines() if line.strip()]
    linked_index = int(manifest_row["linked_s02_index"])
    matching = [row for row in outcome_rows if row.get("execution_index") == linked_index]
    if len(matching) != 1 or matching[0].get("execution_receipt", {}).get("sha256") != sha256(receipt_path):
        raise ValueError("S02 evaluator-only outcome join is not one-to-one and hash-bound")
    return {
        "sources": [
            _reference(semantic_path, repository_root=repository_root),
            _reference(outcome_index_path, repository_root=repository_root),
        ],
        "joined_after_public_report_hashes_frozen": True,
        "pre_target_revealed": initial > 0,
        "post_target_revealed": final > 0,
        "pre_information_sufficient": None,
        "post_information_sufficient": None,
        "sufficiency_status": "UNAVAILABLE_NO_FROZEN_SCHEMA",
        "open_execution_success": matching[0].get("success") is True,
    }


def execute_s03_record_backend(
    *,
    request: Mapping[str, Any],
    schedule_row: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
    identity: Mapping[str, Any],
    record_dir: Path,
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    """Run public offline inference only; never dispatch a selected action."""

    policy = request["policy_request"]
    observations = policy["observation_sequence"]
    reports: list[dict[str, Any]] = []
    report_refs: list[dict[str, str]] = []
    for position, observation in enumerate(observations):
        phase = str(observation["phase"])
        previous = record_dir / "pre" / "report.json" if schedule_row["subtest"] == "C_CLOSED_LOOP_TRANSITION" and phase == "post" else None
        report, refs = _run_infer(
            observation=observation,
            prompt=policy["prompt"],
            identity=identity,
            output_dir=record_dir / phase,
            repository_root=repository_root,
            previous_report=previous,
        )
        reports.append(report)
        report_refs.extend(refs)
    public_report_hashes = [ref["sha256"] for ref in report_refs if ref["path"].endswith("report.json")]
    prediction = {
        "public_report_hashes": public_report_hashes,
        "routes": [_route(report) for report in reports],
        "route_sets": [
            list(report["selected_action"].get("target_prediction_set", []))
            for report in reports
        ],
        "public_outcome_prediction_set": list(identity["backend"]["public_outcome_adapter"]["prediction_set"]),
        "online_oracle_inputs": [],
    }
    private = _private_s02_label(manifest_row, repository_root=repository_root)
    report_refs.extend(private["sources"])
    subtest = schedule_row["subtest"]
    failures: list[str] = []
    if subtest == "A_INFORMATION_EFFECT":
        pre_entropy, post_entropy = map(_entropy, reports)
        if not post_entropy < pre_entropy:
            failures.append("NO_ENTROPY_REDUCTION")
        failures.append("SUFFICIENCY_NOT_IMPROVED")
        if not (not private["pre_target_revealed"] and private["post_target_revealed"]):
            failures.append("TARGET_NOT_REVEALED")
        if not private["open_execution_success"]:
            failures.append("OPEN_EXECUTION_FAILURE")
        outcome = {
            "validator_role": "information_effect_input",
            "entropy_pre": pre_entropy,
            "entropy_post": post_entropy,
            "delta_location_entropy": pre_entropy - post_entropy,
            "evaluator_private": private,
            "primary_positive": False,
            "failure_categories": sorted(set(failures)),
        }
    elif subtest == "B_DECISION_ROUTING":
        expected = schedule_row["stratum"]
        predicted = prediction["routes"][0]
        coverage = False
        correct = predicted == expected and not (predicted == "STOP" and not coverage)
        if not correct:
            failures.append({"ACT": "VISIBLE_NOT_ACT", "OPEN": "HIDDEN_NOT_OPEN", "STOP": "ABSENT_NOT_STOP"}[expected])
        if predicted == "ABSTAIN":
            failures.append("ABSTAINED")
        if predicted == "STOP" and not coverage:
            failures.append("STOP_WITHOUT_PUBLIC_COVERAGE")
        outcome = {
            "validator_role": "decision_routing_input",
            "expected_route_evaluator_only": expected,
            "predicted_route": predicted,
            "public_coverage_certificate_present": coverage,
            "correct": correct,
            "failure_categories": sorted(set(failures)),
        }
    else:
        pre_route, post_route = prediction["routes"]
        if pre_route != "OPEN":
            failures.append("PRE_ROUTE_NOT_OPEN")
        if not private["open_execution_success"]:
            failures.append("OPEN_EXECUTION_FAILURE")
        failures.append("PUBLIC_OUTCOME_AMBIGUOUS")
        if post_route != "ACT":
            failures.append("POST_ROUTE_NOT_ACT")
        outcome = {
            "validator_role": "closed_loop_transition_input",
            "transition_chain_integrity": True,
            "pre_route": pre_route,
            "public_outcome_prediction_set": ["AMBIGUOUS"],
            "post_route": post_route,
            "evaluator_private": private,
            "primary_positive": False,
            "failure_categories": sorted(set(failures)),
        }
    return prediction, outcome, report_refs


def build_record_artifact(
    *,
    request: Mapping[str, Any],
    schedule_row: Mapping[str, Any],
    prediction: Mapping[str, Any] | None,
    outcome: Mapping[str, Any] | None,
    artifacts: list[dict[str, str]],
    identity: Mapping[str, Any],
    infrastructure_failure: Mapping[str, Any] | None = None,
    inference_executed: bool,
) -> dict[str, Any]:
    evaluated = infrastructure_failure is None
    return {
        "schema_version": RECORD_SCHEMA,
        "record_id": request["record_id"],
        "schedule_index": request["execution_index"],
        "linked_s02_index": schedule_row["linked_s02_index"],
        "subtest": schedule_row["subtest"],
        "stratum": schedule_row["stratum"],
        "schedule": request["schedule"],
        "manifest": request["manifest"],
        "manifest_record_sha256": request["manifest_record_sha256"],
        "public_input_digest": request["public_input_digest"],
        "request_sha256": canonical_sha256(request),
        "model_identity": model_identity_stamp(identity, request["execution_index"]),
        "status": "EVALUATED" if evaluated else "INFRASTRUCTURE_FAILURE",
        "inference_executed": inference_executed,
        "outcome_present": evaluated,
        "prediction": dict(prediction) if prediction is not None else None,
        "outcome": dict(outcome) if outcome is not None else None,
        "artifacts": artifacts,
        "infrastructure_failure": dict(infrastructure_failure) if infrastructure_failure is not None else None,
        "privileged_input_violation": False,
        "online_oracle_inputs": [],
        "physical_dispatch_executed": False,
        "s02_artifacts_modified": False,
        "paper_claim_ready": False,
        "created_at": utc_now(),
    }


def build_close_receipt(
    *,
    started: Mapping[str, Any],
    record_path: Path,
    record: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA,
        "receipt_phase": "CLOSED",
        "status": "CLOSED_WITH_OUTCOME" if record["status"] == "EVALUATED" else "CLOSED_INFRASTRUCTURE_FAILURE",
        "execution_index": started["execution_index"],
        "record_id": started["record_id"],
        "schedule": started["schedule"],
        "manifest": started["manifest"],
        "model_identity": started["model_identity"],
        "public_input_digest": started["public_input_digest"],
        "request_sha256": started["request_sha256"],
        "single_use_output_dir": started["single_use_output_dir"],
        "previous_close_sha256": started["previous_close_sha256"],
        "record_artifact": _reference(record_path, repository_root=repository_root),
        "outcome_present": record["outcome_present"],
        "inference_executed": record["inference_executed"],
        "consumes_execution_index": True,
        "created_at": utc_now(),
    }


def write_started_receipt(output_root: Path, receipt: Mapping[str, Any]) -> Path:
    started, _ = _receipt_paths(output_root, int(receipt["execution_index"]))
    _write_new_json(started, receipt)
    return started


def write_record_and_close(
    *,
    output_root: Path,
    record_path: Path,
    record: Mapping[str, Any],
    started: Mapping[str, Any],
    repository_root: Path,
) -> tuple[Path, Path]:
    _write_new_json(record_path, record)
    _, close_path = _receipt_paths(output_root, int(record["schedule_index"]))
    close = build_close_receipt(
        started=started, record_path=record_path, record=record, repository_root=repository_root
    )
    _write_new_json(close_path, close)
    return record_path, close_path
