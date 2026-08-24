"""Prospective S03 public-v2 execution amendment and static validators.

This module never invokes the perception backend.  It seals the consumed v1
index, validates the new runner identity, and binds a fresh execution ledger
to the unchanged 620-record logical schedule.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from interaction_uncertainty.grounding_dino_compat import (
    COMPATIBILITY_WRAPPER_VERSION,
    grounding_dino_post_process_identity,
)

from .s03_execution import (
    build_s03_public_request,
    validate_s03_no_legacy_oracle_dependency,
)
from .s03_preparation import (
    canonical_sha256,
    sha256,
    validate_s03_input_manifest,
    validate_s03_offline_schedule,
)


EXECUTION_VERSION = "s03_public_v2"
V1_BLOCKER_PATH = Path("results/diagnostics/piu_s03_execution_blocker_v1.json")
V1_BLOCKER_SHA256 = "9e742a7a20cc28622da4b95c9e75079e1e8700d707aec4fa7d935952d89e6048"
V1_OUTPUT_ROOT = Path("runs/piu_s03_perception_decision_v1")
V1_PARTIAL_TREE_SHA256 = "eee3900ad0e490e027f18436f979bc72977c015ae27870e80d7d4ac278b265ad"
V1_CERTIFICATE_PATH = Path("results/method/piu_s03_perception_decision_certificate_v1.json")
PARENT_SCHEDULE_PATH = Path("results/method/piu_s03_perception_decision_schedule_v1.json")
PARENT_SCHEDULE_SHA256 = "26faff39586b8071a130253079f2cf3478c88649c6f08276ce7469644123a701"
LOGICAL_MANIFEST_PATH = Path("results/method/piu_s03_perception_decision_input_manifest_v1.json")
LOGICAL_MANIFEST_SHA256 = "757a56c4848d66190b310ef21bd97db2ea954d13c6103f105f93db6868a27433"
V2_IDENTITY_PATH = Path("configs/experiments/piu_s03_runner_identity_v2.json")
V2_MODEL_IDENTITY_PATH = Path("configs/experiments/piu_s03_model_identity_v2.json")
V2_PLAN_PATH = Path("results/method/piu_s03_perception_decision_execution_plan_v2.json")
V2_OUTPUT_ROOT = Path("runs/piu_s03_perception_decision_v2")
V2_CERTIFICATE_PATH = Path("results/method/piu_s03_perception_decision_certificate_v2.json")
V1_MODEL_IDENTITY_PATH = Path("configs/experiments/piu_s03_model_identity_v1.json")
V1_MODEL_IDENTITY_SHA256 = "16903c60054573ccb116dc0d6e544ab7b51178e763ea1b32fe64c65e4556ebda"

_V1_FILE_SHA256 = {
    "_receipts/000.closed.json": "85cd3ffbb97a617b03c6b5fe663e8b31a67b93a18310edf27ca348b1cf2aebce",
    "_receipts/000.started.json": "24b8b5c72f15534217ca19ad5b0e5be20b5b1334089275a9923b9bc498e253a5",
    "records/000_s03-a-000/infrastructure_failure.json": "452226d3a8a514ececd2083466ecd436e77804f7ad3385d429200921e931ba55",
}
_THRESHOLDS = {
    "A_INFORMATION_EFFECT": {"success_threshold": 76, "denominator": 124},
    "B_DECISION_ROUTING": {
        "ACT": {"success_threshold": 55, "denominator": 124},
        "OPEN": {"success_threshold": 55, "denominator": 124},
        "STOP": {"success_threshold": 55, "denominator": 124},
    },
    "C_CLOSED_LOOP_TRANSITION": {
        "integrity_threshold": 124,
        "main_success_threshold": 76,
        "denominator": 124,
    },
}


def _resolve(path: str | Path, *, repository_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repository_root / value


def _portable(path: Path, *, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_mapping(path: Path, location: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} must be a mapping")
    return dict(value)


def _verify_reference(
    value: Any, location: str, *, repository_root: Path
) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{location} must be an exact path/SHA-256 reference")
    path = _resolve(str(value["path"]), repository_root=repository_root)
    if not path.is_file() or sha256(path) != value["sha256"]:
        raise ValueError(f"{location} differs from its frozen bytes")
    return path


def _reference(path: Path, *, repository_root: Path) -> dict[str, str]:
    resolved = _resolve(path, repository_root=repository_root)
    return {
        "path": _portable(resolved, repository_root=repository_root),
        "sha256": sha256(resolved),
    }


def _tree_rows(root: Path) -> list[dict[str, str]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def validate_s03_v1_seal(*, repository_root: Path) -> dict[str, Any]:
    """Prove that the three-file v1 failure ledger is still byte-identical."""

    blocker_path = repository_root / V1_BLOCKER_PATH
    if sha256(blocker_path) != V1_BLOCKER_SHA256:
        raise ValueError("S03 v1 blocker diagnostic changed after sealing")
    blocker = _load_mapping(blocker_path, "S03 v1 blocker")
    root = repository_root / V1_OUTPUT_ROOT
    rows = _tree_rows(root)
    if {row["path"]: row["sha256"] for row in rows} != _V1_FILE_SHA256:
        raise ValueError("S03 v1 immutable receipt/failure file set changed")
    if canonical_sha256(rows) != V1_PARTIAL_TREE_SHA256:
        raise ValueError("S03 v1 partial execution tree changed")
    ledger = blocker.get("ledger")
    failed = blocker.get("failed_record")
    if not isinstance(ledger, Mapping) or not isinstance(failed, Mapping):
        raise ValueError("S03 v1 blocker lacks its immutable ledger")
    if (
        ledger.get("records_started") != 1
        or ledger.get("records_closed") != 1
        or ledger.get("records_with_outcome") != 0
        or ledger.get("reruns") != 0
        or ledger.get("skips") != 0
        or failed.get("execution_index") != 0
        or failed.get("index_consumed") is not True
        or failed.get("prediction_generated") is not False
        or failed.get("outcome_generated") is not False
    ):
        raise ValueError("S03 v1 blocker no longer seals the consumed index-0 failure")
    if (repository_root / V1_CERTIFICATE_PATH).exists():
        raise ValueError("S03 v1 must not acquire a certificate after infrastructure stop")
    return {
        "status": "SEALED_INFRASTRUCTURE_BLOCKED",
        "execution_index_0_consumed": True,
        "execution_index_0_rerun_eligible": False,
        "model_task_outcomes": 0,
        "certificate_present": False,
        "partial_execution_tree_sha256": V1_PARTIAL_TREE_SHA256,
    }


def _validate_code_identity(
    code: Any, *, repository_root: Path
) -> list[dict[str, str]]:
    if not isinstance(code, Mapping) or set(code) != {
        "commit",
        "algorithm",
        "artifacts",
        "digest",
    }:
        raise ValueError("S03 v2 runner code identity fields differ")
    commit = str(code["commit"])
    artifacts = code["artifacts"]
    if (
        code["algorithm"] != "sha256_of_canonical_json_ordered_artifact_refs"
        or not isinstance(artifacts, list)
        or not artifacts
        or code["digest"] != canonical_sha256(artifacts)
    ):
        raise ValueError("S03 v2 runner code identity digest differs")
    subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repository_root,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    ids: set[str] = set()
    for row in artifacts:
        if not isinstance(row, Mapping) or set(row) != {"artifact_id", "path", "sha256"}:
            raise ValueError("S03 v2 code artifact reference is malformed")
        if row["artifact_id"] in ids:
            raise ValueError("S03 v2 code artifact IDs are duplicated")
        ids.add(str(row["artifact_id"]))
        path = _verify_reference(
            {"path": row["path"], "sha256": row["sha256"]},
            "S03 v2 code artifact",
            repository_root=repository_root,
        )
        committed = subprocess.run(
            ["git", "show", f"{commit}:{row['path']}"],
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != row["sha256"] or sha256(path) != row["sha256"]:
            raise ValueError("S03 v2 code artifact is not bound to the frozen source commit")
    return [dict(row) for row in artifacts]


def _validate_s03_v2_model_identity(
    path: Path, *, repository_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and expand the compact v2 amendment over frozen model v1."""

    amendment = _load_mapping(path, "S03 v2 model identity amendment")
    required = {
        "schema_version",
        "status",
        "identity_id",
        "model_id",
        "base_model_identity",
        "backend_id",
        "backend_entrypoint",
        "code",
        "transformers_version",
        "grounding_dino_processor_api",
        "compatibility_wrapper_version",
        "checkpoint_changed",
        "calibrator_changed",
        "policy_input_firewall_changed",
        "legacy_oracle_dependency",
        "outcomes_generated",
        "paper_claim_ready",
    }
    if set(amendment) != required:
        raise ValueError("S03 v2 model identity amendment fields differ")
    if (
        amendment["schema_version"] != "piu.s03-model-identity-amendment.v2"
        or amendment["status"] != "FROZEN_BEFORE_S03_V2_OUTCOMES"
        or amendment["identity_id"] != "piu_s03_public_perception_v2"
        or amendment["model_id"] != "public_rgb_dino_sam_siglip_qwen25vl_pipeline_v0"
        or amendment["checkpoint_changed"] is not False
        or amendment["calibrator_changed"] is not False
        or amendment["policy_input_firewall_changed"] is not False
        or amendment["legacy_oracle_dependency"] is not False
        or amendment["outcomes_generated"] != 0
        or amendment["paper_claim_ready"] is not False
    ):
        raise ValueError("S03 v2 model identity crossed the pre-outcome amendment boundary")
    base_path = _verify_reference(
        amendment["base_model_identity"],
        "S03 v1 base model identity",
        repository_root=repository_root,
    )
    if (
        _portable(base_path, repository_root=repository_root) != str(V1_MODEL_IDENTITY_PATH)
        or sha256(base_path) != V1_MODEL_IDENTITY_SHA256
    ):
        raise ValueError("S03 v2 model amendment changed its frozen model/checkpoint parent")
    base = _load_mapping(base_path, "S03 v1 base model identity")
    checkpoint = base.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("algorithm") != (
        "sha256_of_canonical_json_ordered_artifact_refs"
    ) or checkpoint.get("digest") != canonical_sha256(checkpoint.get("artifacts")):
        raise ValueError("S03 v1 checkpoint aggregate identity is malformed")
    for row in checkpoint["artifacts"]:
        _verify_reference(
            {"path": row["path"], "sha256": row["sha256"]},
            "S03 inherited checkpoint artifact",
            repository_root=repository_root,
        )
    backend_entrypoint = _verify_reference(
        amendment["backend_entrypoint"],
        "S03 v2 backend entrypoint",
        repository_root=repository_root,
    )
    _validate_code_identity(amendment["code"], repository_root=repository_root)
    import transformers

    installed_api = grounding_dino_post_process_identity(
        transformers.GroundingDinoProcessor.post_process_grounded_object_detection
    )
    if (
        amendment["transformers_version"] != transformers.__version__
        or amendment["grounding_dino_processor_api"] != installed_api
        or amendment["compatibility_wrapper_version"] != COMPATIBILITY_WRAPPER_VERSION
    ):
        raise ValueError("S03 v2 model amendment differs from the installed Grounding DINO API")
    expanded = copy.deepcopy(base)
    expanded["identity_id"] = amendment["identity_id"]
    expanded["backend"]["backend_id"] = amendment["backend_id"]
    expanded["backend"]["entrypoint"] = {
        "path": _portable(backend_entrypoint, repository_root=repository_root),
        "sha256": sha256(backend_entrypoint),
    }
    expanded["code"] = copy.deepcopy(amendment["code"])
    expanded["paper_claim_ready"] = False
    validate_s03_no_legacy_oracle_dependency(expanded)
    return amendment, expanded


def validate_s03_v2_runner_identity(
    path: Path, *, repository_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _load_mapping(path, "S03 v2 runner identity")
    required = {
        "schema_version",
        "status",
        "execution_version",
        "identity_id",
        "parent_logical_schedule",
        "logical_manifest",
        "v1_seal",
        "model_identity",
        "model_checkpoint_calibrator_identity",
        "runtime",
        "code",
        "output",
        "policies",
        "execution_scope",
        "outcomes_generated",
        "certificate_present",
        "legacy_oracle_actionable",
        "paper_claim_ready",
    }
    if set(identity) != required:
        raise ValueError("S03 v2 runner identity fields differ")
    if (
        identity["schema_version"] != "piu.s03-runner-identity.v2"
        or identity["status"] != "FROZEN_BEFORE_S03_V2_OUTCOMES"
        or identity["execution_version"] != EXECUTION_VERSION
        or identity["identity_id"] != "piu_s03_public_offline_single_use_v2"
        or identity["outcomes_generated"] != 0
        or identity["certificate_present"] is not False
        or identity["legacy_oracle_actionable"] is not False
        or identity["paper_claim_ready"] is not False
    ):
        raise ValueError("S03 v2 runner identity crossed the pre-outcome boundary")
    schedule_path = _verify_reference(
        identity["parent_logical_schedule"], "S03 v2 parent schedule", repository_root=repository_root
    )
    manifest_path = _verify_reference(
        identity["logical_manifest"], "S03 v2 logical manifest", repository_root=repository_root
    )
    if sha256(schedule_path) != PARENT_SCHEDULE_SHA256 or sha256(manifest_path) != LOGICAL_MANIFEST_SHA256:
        raise ValueError("S03 v2 identity changed the frozen logical records")
    validate_s03_offline_schedule(schedule_path, repository_root=repository_root)
    validate_s03_input_manifest(manifest_path, repository_root=repository_root)
    seal = identity["v1_seal"]
    if not isinstance(seal, Mapping) or set(seal) != {
        "blocker",
        "seal_note",
        "partial_execution_tree_sha256",
        "execution_index_0_consumed",
        "execution_index_0_rerun_eligible",
        "model_task_outcomes",
        "certificate_present",
        "paper_claim_eligible",
    }:
        raise ValueError("S03 v2 identity lacks the exact v1 seal")
    blocker_path = _verify_reference(seal["blocker"], "S03 v1 blocker", repository_root=repository_root)
    _verify_reference(seal["seal_note"], "S03 v1 seal note", repository_root=repository_root)
    if (
        sha256(blocker_path) != V1_BLOCKER_SHA256
        or seal["partial_execution_tree_sha256"] != V1_PARTIAL_TREE_SHA256
        or seal["execution_index_0_consumed"] is not True
        or seal["execution_index_0_rerun_eligible"] is not False
        or seal["model_task_outcomes"] != 0
        or seal["certificate_present"] is not False
        or seal["paper_claim_eligible"] is not False
    ):
        raise ValueError("S03 v1 seal state differs from the immutable blocker")
    validate_s03_v1_seal(repository_root=repository_root)
    model_path = _verify_reference(
        identity["model_identity"], "S03 v2 model identity", repository_root=repository_root
    )
    _, model_identity = _validate_s03_v2_model_identity(
        model_path, repository_root=repository_root
    )
    validate_s03_no_legacy_oracle_dependency(model_identity)
    stamp = identity["model_checkpoint_calibrator_identity"]
    if stamp != {
        "identity_id": model_identity["identity_id"],
        "model_id": model_identity["model_id"],
        "checkpoint_digest": model_identity["checkpoint"]["digest"],
        "calibrator_id": model_identity["calibrator"]["calibrator_id"],
        "calibrator_version": model_identity["calibrator"]["version"],
        "calibrator_status": model_identity["calibrator"]["status"],
    }:
        raise ValueError("S03 v2 checkpoint/calibrator stamp differs")
    runtime = identity["runtime"]
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "transformers_version",
        "grounding_dino_processor_api",
        "compatibility_wrapper_version",
    }:
        raise ValueError("S03 v2 runtime identity fields differ")
    import transformers

    installed = grounding_dino_post_process_identity(
        transformers.GroundingDinoProcessor.post_process_grounded_object_detection
    )
    if (
        runtime["transformers_version"] != transformers.__version__
        or runtime["grounding_dino_processor_api"] != installed
        or runtime["compatibility_wrapper_version"] != COMPATIBILITY_WRAPPER_VERSION
    ):
        raise ValueError("S03 v2 Grounding DINO runtime identity differs")
    _validate_code_identity(identity["code"], repository_root=repository_root)
    if identity["output"] != {
        "root": str(V2_OUTPUT_ROOT),
        "certificate": str(V2_CERTIFICATE_PATH),
        "receipt_directory": "_receipts",
        "record_directory": "records",
    }:
        raise ValueError("S03 v2 output identity differs")
    if identity["policies"] != {
        "all_620_logical_records_once": True,
        "ordered_execution": True,
        "single_use": True,
        "rerun": False,
        "skip": False,
        "replace": False,
        "cherry_pick": False,
        "v1_index_0_reuse": False,
        "outcome_write_requires_explicit_flag": True,
    }:
        raise ValueError("S03 v2 prospective execution policies differ")
    if identity["execution_scope"] != {
        "offline_public_input_inference_only": True,
        "physical_rollout": False,
        "pi05_action_calls": False,
        "environment_steps": False,
        "legacy_oracle": False,
        "privileged_policy_inputs": False,
        "formal_method_claim": False,
    }:
        raise ValueError("S03 v2 identity permits an out-of-scope operation")
    if (repository_root / V2_OUTPUT_ROOT).exists() or (repository_root / V2_CERTIFICATE_PATH).exists():
        raise ValueError("S03 v2 outcomes or certificate exist before the v2 freeze")
    return identity, model_identity


def build_s03_v2_execution_plan(
    *, runner_identity_path: Path, repository_root: Path
) -> dict[str, Any]:
    identity, _ = validate_s03_v2_runner_identity(
        runner_identity_path, repository_root=repository_root
    )
    schedule_path = _resolve(
        identity["parent_logical_schedule"]["path"], repository_root=repository_root
    )
    schedule = validate_s03_offline_schedule(schedule_path, repository_root=repository_root)
    record_bindings = [
        {
            "execution_index": index,
            "record_id": row["record_id"],
            "parent_schedule_index": row["schedule_index"],
            "parent_schedule_record_sha256": canonical_sha256(row),
            "linked_s02_index": row["linked_s02_index"],
            "subtest": row["subtest"],
            "stratum": row["stratum"],
            "public_input_digest": row["public_input_digest"],
        }
        for index, row in enumerate(schedule["records"])
    ]
    return {
        "schema_version": "piu.s03-execution-plan.v2",
        "status": "FROZEN_BEFORE_S03_V2_OUTCOMES",
        "execution_version": EXECUTION_VERSION,
        "claim_scope": "PRE_OUTCOME_EXECUTION_AMENDMENT_NOT_PERFORMANCE_EVIDENCE",
        "parent_logical_schedule": _reference(schedule_path, repository_root=repository_root),
        "logical_manifest": identity["logical_manifest"],
        "runner_identity": _reference(runner_identity_path, repository_root=repository_root),
        "v1_history": {
            "blocker": identity["v1_seal"]["blocker"],
            "execution_index_0_status": "INFRASTRUCTURE_BLOCKED_CONSUMED_NO_OUTCOME",
            "excluded_from_v2_outcome_statistics": True,
            "v1_index_0_rerun": False,
        },
        "execution_rule": "all_620_logical_records_once_in_parent_order_under_v2_identity",
        "record_count": 620,
        "thresholds": _THRESHOLDS,
        "logical_record_binding": {
            "algorithm": "parent_schedule_sha256_plus_ordered_record_binding_sha256",
            "ordered_record_ids": [row["record_id"] for row in record_bindings],
            "ordered_record_binding_sha256": canonical_sha256(record_bindings),
            "counts": copy.deepcopy(schedule["counts"]),
            "each_s02_index_multiplicity": 5,
        },
        "s02_indices_101_and_104_included": True,
        "record_filtering": False,
        "record_replacement": False,
        "record_cherry_picking": False,
        "inference_executed": False,
        "outcome_present": False,
        "predictions_present": False,
        "certificate_present": False,
        "legacy_oracle_used": False,
        "paper_claim_ready": False,
    }


def validate_s03_v2_execution_plan(
    path: Path, *, repository_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = _load_mapping(path, "S03 v2 execution plan")
    identity_path = _verify_reference(
        plan.get("runner_identity"), "S03 v2 runner identity", repository_root=repository_root
    )
    identity, model_identity = validate_s03_v2_runner_identity(
        identity_path, repository_root=repository_root
    )
    expected = build_s03_v2_execution_plan(
        runner_identity_path=identity_path, repository_root=repository_root
    )
    if plan != expected:
        raise ValueError("S03 v2 execution plan differs from its deterministic 620-record binding")
    schedule_path = _resolve(
        plan["parent_logical_schedule"]["path"], repository_root=repository_root
    )
    schedule = validate_s03_offline_schedule(schedule_path, repository_root=repository_root)
    records = schedule["records"]
    binding = plan["logical_record_binding"]
    if not isinstance(binding, Mapping) or len(records) != 620:
        raise ValueError("S03 v2 execution plan must bind exactly 620 parent records")
    if binding["ordered_record_ids"] != [row["record_id"] for row in records]:
        raise ValueError("S03 v2 execution order changed")
    if len(set(binding["ordered_record_ids"])) != 620:
        raise ValueError("S03 v2 record IDs are duplicated")
    for subtest, stratum in (
        ("A_INFORMATION_EFFECT", "PAIRED_PRE_POST"),
        ("B_DECISION_ROUTING", "ACT"),
        ("B_DECISION_ROUTING", "OPEN"),
        ("B_DECISION_ROUTING", "STOP"),
        ("C_CLOSED_LOOP_TRANSITION", "OBSERVE_OPEN_REOBSERVE"),
    ):
        indices = [
            row["linked_s02_index"]
            for row in records
            if row["subtest"] == subtest and row["stratum"] == stratum
        ]
        if indices != list(range(124)) or 101 not in indices or 104 not in indices:
            raise ValueError("S03 v2 filtered or reordered a logical subtest/stratum")
    return plan, identity, model_identity


def validate_s03_v2_runner_preflight(
    *, plan_path: Path, execution_index: int, repository_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan, identity, model_identity = validate_s03_v2_execution_plan(
        plan_path, repository_root=repository_root
    )
    if isinstance(execution_index, bool) or not isinstance(execution_index, int) or not 0 <= execution_index < 620:
        raise IndexError("S03 v2 execution index must be in [0, 619]")
    schedule_path = _resolve(
        plan["parent_logical_schedule"]["path"], repository_root=repository_root
    )
    schedule = validate_s03_offline_schedule(schedule_path, repository_root=repository_root)
    manifest_path = _resolve(
        plan["logical_manifest"]["path"], repository_root=repository_root
    )
    manifest = validate_s03_input_manifest(manifest_path, repository_root=repository_root)
    model_identity_path = _resolve(
        identity["model_identity"]["path"], repository_root=repository_root
    )
    request = build_s03_public_request(
        schedule=schedule,
        manifest=manifest,
        schedule_path=schedule_path,
        manifest_path=manifest_path,
        identity_path=model_identity_path,
        identity=model_identity,
        execution_index=execution_index,
        repository_root=repository_root,
    )
    if plan["logical_record_binding"]["ordered_record_ids"][execution_index] != request["record_id"]:
        raise ValueError("S03 v2 plan/request record binding differs")
    return plan, identity, model_identity, schedule, request
