"""Prospective S03 public-v3 execution amendment and lifecycle validation.

This module deliberately separates immutable pre-outcome identity validation
from append-only execution-ledger validation.  Importing it never loads model
weights, executes inference, or writes an outcome artifact.
"""

from __future__ import annotations

import copy
import importlib.metadata
import importlib.util
import json
import platform
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from interaction_uncertainty.grounding_dino_compat import (
    COMPATIBILITY_WRAPPER_VERSION,
    grounding_dino_post_process_identity,
)

from .s03_execution import (
    build_s03_public_request,
    validate_s03_no_legacy_oracle_dependency,
    validate_s03_receipts,
)
from .s03_preparation import (
    canonical_sha256,
    sha256,
    validate_s03_input_manifest,
    validate_s03_offline_schedule,
)
from .s03_v2_amendment import (
    _validate_s03_v2_model_identity,
    validate_s03_v1_seal,
)


EXECUTION_VERSION = "s03_public_v3"
RECORD_COUNT = 620

PARENT_SCHEDULE_PATH = Path(
    "results/method/piu_s03_perception_decision_schedule_v1.json"
)
PARENT_SCHEDULE_SHA256 = (
    "26faff39586b8071a130253079f2cf3478c88649c6f08276ce7469644123a701"
)
LOGICAL_MANIFEST_PATH = Path(
    "results/method/piu_s03_perception_decision_input_manifest_v1.json"
)
LOGICAL_MANIFEST_SHA256 = (
    "757a56c4848d66190b310ef21bd97db2ea954d13c6103f105f93db6868a27433"
)
V1_BLOCKER_PATH = Path("results/diagnostics/piu_s03_execution_blocker_v1.json")
V1_BLOCKER_SHA256 = (
    "9e742a7a20cc28622da4b95c9e75079e1e8700d707aec4fa7d935952d89e6048"
)
V1_PARTIAL_TREE_SHA256 = (
    "eee3900ad0e490e027f18436f979bc72977c015ae27870e80d7d4ac278b265ad"
)
V2_BLOCKER_PATH = Path(
    "results/diagnostics/piu_s03_v2_execution_blocker_v1.json"
)
V2_BLOCKER_SHA256 = (
    "6bc48e54e0bb114fef8fffc3bcdf56144143c16842fc6b372cd429a341b710b8"
)
V2_PARTIAL_TREE_SHA256 = (
    "017220a2c71d0378f8a6d21cfbab37896ada45e41410fc9bd6c4bb5dea236d25"
)
V2_PLAN_PATH = Path(
    "results/method/piu_s03_perception_decision_execution_plan_v2.json"
)
V2_PLAN_SHA256 = (
    "b6ff3764f3ccb298f679a810e62613d9561ee0b9801d8258c6297bb3cbc8ca6c"
)
V2_RUNNER_IDENTITY_PATH = Path("configs/experiments/piu_s03_runner_identity_v2.json")
V2_RUNNER_IDENTITY_SHA256 = (
    "3c00bb8bbcf442ed173e67e6e461bb985c71fe44aba3c498e5e82b8bfcd508bd"
)
V2_MODEL_IDENTITY_PATH = Path("configs/experiments/piu_s03_model_identity_v2.json")
V2_MODEL_IDENTITY_SHA256 = (
    "87dc3c296560a688c6b334602ea7c6a937a749e9de8c5e36e215092af4f413d2"
)
V2_OUTPUT_ROOT = Path("runs/piu_s03_perception_decision_v2")
V2_CERTIFICATE_PATH = Path(
    "results/method/piu_s03_perception_decision_certificate_v2.json"
)

V3_RUNTIME_READINESS_PATH = Path(
    "configs/experiments/piu_s03_runtime_readiness_v3.json"
)
V3_MODEL_IDENTITY_PATH = Path("configs/experiments/piu_s03_model_identity_v3.json")
V3_RUNNER_IDENTITY_PATH = Path("configs/experiments/piu_s03_runner_identity_v3.json")
V3_PLAN_PATH = Path(
    "results/method/piu_s03_perception_decision_execution_plan_v3.json"
)
V3_OUTPUT_ROOT = Path("runs/piu_s03_perception_decision_v3")
V3_CERTIFICATE_PATH = Path(
    "results/method/piu_s03_perception_decision_certificate_v3.json"
)
V3_BLOCKER_PATH = Path("results/diagnostics/piu_s03_v3_execution_blocker_v1.json")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_DISTRIBUTIONS = (
    "accelerate",
    "numpy",
    "pillow",
    "protobuf",
    "sentencepiece",
    "torch",
    "torchvision",
    "transformers",
)
_REQUIRED_IMPORTS = (
    "accelerate",
    "google.protobuf",
    "numpy",
    "PIL",
    "sentencepiece",
    "torch",
    "torchvision",
    "transformers",
)
_MODEL_PATHS = {
    "grounding_dino": Path("checkpoints/perception/grounding-dino-tiny"),
    "sam": Path("checkpoints/perception/sam-vit-base"),
    "dinov2": Path("checkpoints/perception/dinov2-small"),
    "siglip": Path("checkpoints/perception/siglip-base-patch16-224"),
    "qwen": Path("checkpoints/perception/qwen2.5-vl-3b-instruct"),
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

FROZEN_READY_BEFORE_OUTCOMES = "FROZEN_READY_BEFORE_OUTCOMES"
EXECUTION_IN_PROGRESS = "EXECUTION_IN_PROGRESS"
EXECUTION_BLOCKED_INFRA = "EXECUTION_BLOCKED_INFRA"
EXECUTION_COMPLETE_PENDING_CERTIFICATE = "EXECUTION_COMPLETE_PENDING_CERTIFICATE"
CERTIFIED = "CERTIFIED"


def _resolve(path: str | Path, *, repository_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repository_root / value


def _portable(path: Path, *, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _reference(path: Path, *, repository_root: Path) -> dict[str, str]:
    return {
        "path": _portable(path, repository_root=repository_root),
        "sha256": sha256(path),
    }


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    return dict(value)


def _verify_reference(
    value: Any, label: str, *, repository_root: Path
) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain an exact path/SHA-256 reference")
    path = _resolve(str(value["path"]), repository_root=repository_root)
    if (
        not path.is_file()
        or not _SHA256.fullmatch(str(value["sha256"]))
        or sha256(path) != value["sha256"]
    ):
        raise ValueError(f"{label} differs from its frozen bytes")
    return path


def _tree_identity(path: Path, *, repository_root: Path) -> dict[str, Any]:
    rows = [
        {
            "path": _portable(item, repository_root=repository_root),
            "sha256": sha256(item),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    return {
        "algorithm": "sha256_of_canonical_json_ordered_relative_path_and_sha256_rows",
        "file_count": len(rows),
        "root_sha256": canonical_sha256(rows),
    }


def _validate_code_identity(value: Any, *, repository_root: Path) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "freeze_parent_commit",
        "algorithm",
        "artifacts",
        "digest",
    }:
        raise ValueError("S03 v3 code identity fields differ")
    artifacts = value["artifacts"]
    if (
        value["freeze_parent_commit"]
        != "6ebcfd092e6a7fe875c9c3f2d26a540490434590"
        or value["algorithm"]
        != "sha256_of_canonical_json_ordered_artifact_refs"
        or not isinstance(artifacts, list)
        or value["digest"] != canonical_sha256(artifacts)
    ):
        raise ValueError("S03 v3 code identity digest differs")
    identifiers: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "artifact_id",
            "path",
            "sha256",
        }:
            raise ValueError("S03 v3 code artifact reference differs")
        artifact_id = str(artifact["artifact_id"])
        if artifact_id in identifiers:
            raise ValueError("S03 v3 code identity duplicates an artifact ID")
        identifiers.add(artifact_id)
        _verify_reference(
            {"path": artifact["path"], "sha256": artifact["sha256"]},
            f"S03 v3 code artifact {artifact_id}",
            repository_root=repository_root,
        )


def _package_versions() -> dict[str, str]:
    return {
        distribution: importlib.metadata.version(distribution)
        for distribution in _REQUIRED_DISTRIBUTIONS
    }


def probe_s03_v3_backend_readiness(
    *,
    repository_root: Path,
    module_finder: Callable[[str], Any] = importlib.util.find_spec,
    require_empty_execution_boundary: bool = True,
) -> dict[str, Any]:
    """Load processors/configs only; never instantiate model weights or predict."""

    missing = [name for name in _REQUIRED_IMPORTS if module_finder(name) is None]
    if missing:
        raise RuntimeError(
            "S03 v3 backend dependency readiness failed; missing imports: "
            + ", ".join(missing)
        )
    output_root = repository_root / V3_OUTPUT_ROOT
    certificate = repository_root / V3_CERTIFICATE_PATH
    if require_empty_execution_boundary and (output_root.exists() or certificate.exists()):
        raise RuntimeError(
            "S03 v3 backend readiness is pre-outcome-only; output/certificate already exists"
        )

    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoProcessor,
        GroundingDinoProcessor,
        SamProcessor,
        SiglipProcessor,
    )

    roots = {
        name: repository_root / relative for name, relative in _MODEL_PATHS.items()
    }
    processors = {
        "grounding_dino": GroundingDinoProcessor.from_pretrained(
            roots["grounding_dino"], local_files_only=True
        ),
        "sam": SamProcessor.from_pretrained(roots["sam"], local_files_only=True),
        "dinov2": AutoImageProcessor.from_pretrained(
            roots["dinov2"], local_files_only=True, use_fast=False
        ),
        "siglip": SiglipProcessor.from_pretrained(
            roots["siglip"], local_files_only=True, use_fast=False
        ),
        "qwen": AutoProcessor.from_pretrained(
            roots["qwen"], local_files_only=True, use_fast=False
        ),
    }
    configs = {
        name: AutoConfig.from_pretrained(path, local_files_only=True)
        for name, path in roots.items()
    }
    import torch
    import torchvision
    import transformers

    grounding_identity = grounding_dino_post_process_identity(
        transformers.GroundingDinoProcessor.post_process_grounded_object_detection
    )
    if grounding_identity["compatibility_wrapper_version"] != COMPATIBILITY_WRAPPER_VERSION:
        raise RuntimeError("S03 v3 Grounding DINO compatibility wrapper differs")
    siglip_tokenizer = getattr(processors["siglip"], "tokenizer", None)
    if type(siglip_tokenizer).__name__ != "SiglipTokenizer":
        raise RuntimeError("S03 v3 SigLIP tokenizer did not initialize")
    if require_empty_execution_boundary and (output_root.exists() or certificate.exists()):
        raise RuntimeError("S03 v3 readiness probe created an outcome boundary")
    return {
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "runtime_build_versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "sentencepiece_available": True,
        "siglip_tokenizer": {
            "processor_class": type(processors["siglip"]).__name__,
            "tokenizer_class": type(siglip_tokenizer).__name__,
            "local_files_only": True,
            "initialized": True,
        },
        "processor_classes": {
            name: type(processor).__name__
            for name, processor in processors.items()
        },
        "config_classes": {
            name: type(config).__name__ for name, config in configs.items()
        },
        "grounding_dino_processor_api": grounding_identity,
        "all_required_processors_and_configs_loaded": True,
        "model_weights_loaded": False,
        "model_prediction_executed": False,
        "outcome_written": False,
    }


def build_s03_v3_runtime_readiness(*, repository_root: Path) -> dict[str, Any]:
    base_model = repository_root / V2_MODEL_IDENTITY_PATH
    if sha256(base_model) != V2_MODEL_IDENTITY_SHA256:
        raise ValueError("S03 v2 model identity changed before v3 readiness freeze")
    _, expanded = _validate_s03_v2_model_identity(
        base_model, repository_root=repository_root
    )
    probe = probe_s03_v3_backend_readiness(repository_root=repository_root)
    return {
        "schema_version": "piu.s03-runtime-readiness.v3",
        "status": "FROZEN_READY_BEFORE_S03_V3_OUTCOMES",
        "execution_version": EXECUTION_VERSION,
        "dependency_declarations": [
            _reference(repository_root / "pyproject.toml", repository_root=repository_root),
            _reference(repository_root / "uv.lock", repository_root=repository_root),
        ],
        "backend_readiness": probe,
        "model_checkpoint_calibrator_identity": {
            "identity_id": "piu_s03_public_perception_v3",
            "model_id": expanded["model_id"],
            "checkpoint_digest": expanded["checkpoint"]["digest"],
            "calibrator_id": expanded["calibrator"]["calibrator_id"],
            "calibrator_version": expanded["calibrator"]["version"],
            "calibrator_status": expanded["calibrator"]["status"],
        },
        "execution_boundary": {
            "output_root": str(V3_OUTPUT_ROOT),
            "certificate": str(V3_CERTIFICATE_PATH),
            "output_root_absent_at_freeze": True,
            "certificate_absent_at_freeze": True,
            "inference_executed": False,
            "outcomes_generated": 0,
        },
        "legacy_oracle_dependency": False,
        "paper_claim_ready": False,
    }


def validate_s03_v3_runtime_readiness(
    path: Path,
    *,
    repository_root: Path,
    rerun_processor_probe: bool = True,
) -> dict[str, Any]:
    value = _load_mapping(path, "S03 v3 runtime readiness")
    if set(value) != {
        "schema_version",
        "status",
        "execution_version",
        "dependency_declarations",
        "backend_readiness",
        "model_checkpoint_calibrator_identity",
        "execution_boundary",
        "legacy_oracle_dependency",
        "paper_claim_ready",
    }:
        raise ValueError("S03 v3 runtime readiness fields differ")
    if (
        value["schema_version"] != "piu.s03-runtime-readiness.v3"
        or value["status"] != "FROZEN_READY_BEFORE_S03_V3_OUTCOMES"
        or value["execution_version"] != EXECUTION_VERSION
        or value["legacy_oracle_dependency"] is not False
        or value["paper_claim_ready"] is not False
    ):
        raise ValueError("S03 v3 runtime readiness crossed the pre-outcome boundary")
    declarations = value["dependency_declarations"]
    if not isinstance(declarations, list) or len(declarations) != 2:
        raise ValueError("S03 v3 dependency declaration references differ")
    for reference, expected in zip(declarations, ("pyproject.toml", "uv.lock")):
        resolved = _verify_reference(
            reference, "S03 v3 dependency declaration", repository_root=repository_root
        )
        if _portable(resolved, repository_root=repository_root) != expected:
            raise ValueError("S03 v3 dependency declaration path differs")
    readiness = value["backend_readiness"]
    if not isinstance(readiness, Mapping):
        raise TypeError("S03 v3 backend readiness is missing")
    if (
        readiness.get("sentencepiece_available") is not True
        or readiness.get("all_required_processors_and_configs_loaded") is not True
        or readiness.get("model_weights_loaded") is not False
        or readiness.get("model_prediction_executed") is not False
        or readiness.get("outcome_written") is not False
        or readiness.get("siglip_tokenizer", {}).get("initialized") is not True
    ):
        raise ValueError("S03 v3 backend readiness is incomplete")
    if readiness.get("package_versions") != _package_versions():
        raise ValueError("S03 v3 installed package versions differ")
    if rerun_processor_probe:
        current = probe_s03_v3_backend_readiness(
            repository_root=repository_root,
            require_empty_execution_boundary=not (
                (repository_root / V3_OUTPUT_ROOT).exists()
                or (repository_root / V3_CERTIFICATE_PATH).exists()
            ),
        )
        if current != readiness:
            raise ValueError("S03 v3 live processor/config readiness differs")
    boundary = value["execution_boundary"]
    if boundary != {
        "output_root": str(V3_OUTPUT_ROOT),
        "certificate": str(V3_CERTIFICATE_PATH),
        "output_root_absent_at_freeze": True,
        "certificate_absent_at_freeze": True,
        "inference_executed": False,
        "outcomes_generated": 0,
    }:
        raise ValueError("S03 v3 readiness boundary differs")
    return value


def validate_s03_v2_seal(*, repository_root: Path) -> dict[str, Any]:
    """Validate immutable v2 history without invoking its faulty prefreeze gate."""

    validate_s03_v1_seal(repository_root=repository_root)
    blocker_path = repository_root / V2_BLOCKER_PATH
    if sha256(blocker_path) != V2_BLOCKER_SHA256:
        raise ValueError("S03 v2 blocker bytes changed")
    blocker = _load_mapping(blocker_path, "S03 v2 blocker")
    if (
        blocker.get("status") != "FORMAL_EXECUTION_STOPPED_INFRASTRUCTURE_FAILURE"
        or blocker.get("ledger", {}).get("records_started") != 1
        or blocker.get("ledger", {}).get("records_closed") != 1
        or blocker.get("ledger", {}).get("records_with_prediction") != 0
        or blocker.get("ledger", {}).get("records_with_outcome") != 0
        or blocker.get("failed_record", {}).get("execution_index") != 0
        or blocker.get("failed_record", {}).get("index_consumed") is not True
        or blocker.get("failed_record", {}).get("rerun_eligible") is not False
        or blocker.get("stop_decision", {}).get("certificate_generated") is not False
        or blocker.get("paper_claim_ready") is not False
    ):
        raise ValueError("S03 v2 blocker no longer seals the exact infrastructure stop")
    for path, expected in (
        (V2_PLAN_PATH, V2_PLAN_SHA256),
        (V2_RUNNER_IDENTITY_PATH, V2_RUNNER_IDENTITY_SHA256),
        (V2_MODEL_IDENTITY_PATH, V2_MODEL_IDENTITY_SHA256),
    ):
        if sha256(repository_root / path) != expected:
            raise ValueError(f"S03 v2 frozen artifact changed: {path}")
    tree = _tree_identity(repository_root / V2_OUTPUT_ROOT, repository_root=repository_root)
    if tree != {
        "algorithm": "sha256_of_canonical_json_ordered_relative_path_and_sha256_rows",
        "file_count": 67,
        "root_sha256": V2_PARTIAL_TREE_SHA256,
    }:
        raise ValueError("S03 v2 partial execution tree changed")
    if (repository_root / V2_CERTIFICATE_PATH).exists():
        raise ValueError("S03 v2 certificate must remain absent")
    _, expanded = _validate_s03_v2_model_identity(
        repository_root / V2_MODEL_IDENTITY_PATH, repository_root=repository_root
    )
    schedule = validate_s03_offline_schedule(
        repository_root / PARENT_SCHEDULE_PATH, repository_root=repository_root
    )
    ledger = validate_s03_receipts(
        repository_root / V2_OUTPUT_ROOT,
        schedule=schedule,
        schedule_path=repository_root / PARENT_SCHEDULE_PATH,
        manifest_path=repository_root / LOGICAL_MANIFEST_PATH,
        identity_path=repository_root / V2_MODEL_IDENTITY_PATH,
        identity=expanded,
        repository_root=repository_root,
    )
    if ledger != {"closed": 1, "in_flight": None, "next_execution_index": 1}:
        raise ValueError("S03 v2 receipt chain differs from its sealed state")
    return {
        "status": "SEALED_INFRASTRUCTURE_BLOCKED",
        "execution_index_0_consumed": True,
        "execution_index_0_rerun_eligible": False,
        "model_task_outcomes": 0,
        "certificate_present": False,
        "partial_execution_tree_sha256": V2_PARTIAL_TREE_SHA256,
        "metrics_status": "UNAVAILABLE_NOT_PASS_FAIL",
        "ambiguous_generated": 0,
        "ambiguous_rate_interpretable": False,
    }


def classify_s03_v3_lifecycle(
    ledger_status: Mapping[str, Any],
    *,
    execution_authorized: bool,
    certificate_present: bool,
    certificate_validated: bool = False,
    infrastructure_blocker_present: bool = False,
) -> str:
    """Classify a validated append-only receipt chain."""

    if set(ledger_status) != {"closed", "in_flight", "next_execution_index"}:
        raise ValueError("S03 v3 ledger status fields differ")
    closed = ledger_status["closed"]
    in_flight = ledger_status["in_flight"]
    next_index = ledger_status["next_execution_index"]
    if (
        isinstance(closed, bool)
        or not isinstance(closed, int)
        or not 0 <= closed <= RECORD_COUNT
        or (in_flight is not None and (isinstance(in_flight, bool) or not isinstance(in_flight, int)))
        or isinstance(next_index, bool)
        or not isinstance(next_index, int)
    ):
        raise ValueError("S03 v3 ledger counts are invalid")
    expected_next = in_flight if in_flight is not None else closed
    if next_index != expected_next:
        raise ValueError("S03 v3 ledger next index differs from its receipt chain")
    if in_flight is not None and (in_flight != closed or not 0 <= in_flight < RECORD_COUNT):
        raise ValueError("S03 v3 in-flight receipt is not the next exact index")
    execution_started = closed > 0 or in_flight is not None
    if not execution_authorized and (
        execution_started or certificate_present or infrastructure_blocker_present
    ):
        raise ValueError("S03 v3 execution artifacts exist before execution authorization")
    if certificate_present:
        if infrastructure_blocker_present:
            raise ValueError("blocked S03 v3 execution cannot have a certificate")
        if closed != RECORD_COUNT or in_flight is not None:
            raise ValueError("S03 v3 certificate requires 620 valid closed receipts")
        if not certificate_validated:
            raise ValueError("S03 v3 certificate exists but was not validated")
        return CERTIFIED
    if certificate_validated:
        raise ValueError("S03 v3 certificate cannot validate when absent")
    if infrastructure_blocker_present:
        if not execution_started:
            raise ValueError("S03 v3 infrastructure blocker cannot precede execution")
        return EXECUTION_BLOCKED_INFRA
    if closed == RECORD_COUNT:
        if in_flight is not None:
            raise ValueError("S03 v3 cannot start receipt 620")
        return EXECUTION_COMPLETE_PENDING_CERTIFICATE
    if execution_started:
        return EXECUTION_IN_PROGRESS
    return FROZEN_READY_BEFORE_OUTCOMES


def validate_s03_v3_model_identity(
    path: Path, *, repository_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    amendment = _load_mapping(path, "S03 v3 model identity")
    if set(amendment) != {
        "schema_version",
        "status",
        "identity_id",
        "model_id",
        "base_model_identity",
        "runtime_readiness",
        "backend_id",
        "checkpoint_changed",
        "calibrator_changed",
        "policy_input_firewall_changed",
        "legacy_oracle_dependency",
        "outcomes_generated",
        "paper_claim_ready",
    }:
        raise ValueError("S03 v3 model identity fields differ")
    if (
        amendment["schema_version"] != "piu.s03-model-identity-amendment.v3"
        or amendment["status"] != "FROZEN_BEFORE_S03_V3_OUTCOMES"
        or amendment["identity_id"] != "piu_s03_public_perception_v3"
        or amendment["model_id"]
        != "public_rgb_dino_sam_siglip_qwen25vl_pipeline_v0"
        or amendment["backend_id"]
        != "interaction_uncertainty_public_rgb_pipeline_v0_grounding_dino_compat_runtime_ready_v3"
        or amendment["checkpoint_changed"] is not False
        or amendment["calibrator_changed"] is not False
        or amendment["policy_input_firewall_changed"] is not False
        or amendment["legacy_oracle_dependency"] is not False
        or amendment["outcomes_generated"] != 0
        or amendment["paper_claim_ready"] is not False
    ):
        raise ValueError("S03 v3 model identity crossed the pre-outcome boundary")
    base_path = _verify_reference(
        amendment["base_model_identity"],
        "S03 v3 base model identity",
        repository_root=repository_root,
    )
    if (
        _portable(base_path, repository_root=repository_root)
        != str(V2_MODEL_IDENTITY_PATH)
        or sha256(base_path) != V2_MODEL_IDENTITY_SHA256
    ):
        raise ValueError("S03 v3 model identity changed its v2 parent")
    readiness_path = _verify_reference(
        amendment["runtime_readiness"],
        "S03 v3 runtime readiness",
        repository_root=repository_root,
    )
    validate_s03_v3_runtime_readiness(
        readiness_path, repository_root=repository_root
    )
    _, expanded = _validate_s03_v2_model_identity(
        base_path, repository_root=repository_root
    )
    expanded = copy.deepcopy(expanded)
    expanded["identity_id"] = amendment["identity_id"]
    expanded["backend"]["backend_id"] = amendment["backend_id"]
    expanded["paper_claim_ready"] = False
    validate_s03_no_legacy_oracle_dependency(expanded)
    return amendment, expanded


def validate_s03_v3_runner_identity(
    path: Path, *, repository_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _load_mapping(path, "S03 v3 runner identity")
    if set(identity) != {
        "schema_version",
        "status",
        "execution_version",
        "identity_id",
        "parent_logical_schedule",
        "logical_manifest",
        "v1_seal",
        "v2_seal",
        "runtime_readiness",
        "model_identity",
        "model_checkpoint_calibrator_identity",
        "code",
        "output",
        "lifecycle_states",
        "policies",
        "execution_scope",
        "outcomes_generated",
        "certificate_present",
        "legacy_oracle_actionable",
        "paper_claim_ready",
    }:
        raise ValueError("S03 v3 runner identity fields differ")
    if (
        identity["schema_version"] != "piu.s03-runner-identity.v3"
        or identity["status"] != "FROZEN_BEFORE_S03_V3_OUTCOMES"
        or identity["execution_version"] != EXECUTION_VERSION
        or identity["identity_id"] != "piu_s03_public_offline_single_use_v3"
        or identity["outcomes_generated"] != 0
        or identity["certificate_present"] is not False
        or identity["legacy_oracle_actionable"] is not False
        or identity["paper_claim_ready"] is not False
    ):
        raise ValueError("S03 v3 runner identity crossed the pre-outcome boundary")
    schedule_path = _verify_reference(
        identity["parent_logical_schedule"],
        "S03 v3 parent schedule",
        repository_root=repository_root,
    )
    manifest_path = _verify_reference(
        identity["logical_manifest"],
        "S03 v3 logical manifest",
        repository_root=repository_root,
    )
    if (
        sha256(schedule_path) != PARENT_SCHEDULE_SHA256
        or sha256(manifest_path) != LOGICAL_MANIFEST_SHA256
    ):
        raise ValueError("S03 v3 identity changed frozen logical inputs")
    validate_s03_offline_schedule(schedule_path, repository_root=repository_root)
    validate_s03_input_manifest(manifest_path, repository_root=repository_root)
    if identity["v1_seal"] != {
        "blocker": {"path": str(V1_BLOCKER_PATH), "sha256": V1_BLOCKER_SHA256},
        "partial_execution_tree_sha256": V1_PARTIAL_TREE_SHA256,
        "index_0_consumed": True,
        "index_0_rerun_eligible": False,
        "model_task_outcomes": 0,
        "certificate_present": False,
    }:
        raise ValueError("S03 v3 identity changed the v1 seal")
    if identity["v2_seal"] != {
        "blocker": {"path": str(V2_BLOCKER_PATH), "sha256": V2_BLOCKER_SHA256},
        "partial_execution_tree_sha256": V2_PARTIAL_TREE_SHA256,
        "index_0_consumed": True,
        "index_0_rerun_eligible": False,
        "model_task_outcomes": 0,
        "certificate_present": False,
        "metrics_status": "UNAVAILABLE_NOT_PASS_FAIL",
        "ambiguous_generated": 0,
        "ambiguous_rate_interpretable": False,
    }:
        raise ValueError("S03 v3 identity changed the v2 seal")
    validate_s03_v1_seal(repository_root=repository_root)
    validate_s03_v2_seal(repository_root=repository_root)
    readiness_path = _verify_reference(
        identity["runtime_readiness"],
        "S03 v3 runtime readiness",
        repository_root=repository_root,
    )
    readiness = validate_s03_v3_runtime_readiness(
        readiness_path, repository_root=repository_root
    )
    model_path = _verify_reference(
        identity["model_identity"],
        "S03 v3 model identity",
        repository_root=repository_root,
    )
    _, model = validate_s03_v3_model_identity(
        model_path, repository_root=repository_root
    )
    if identity["model_checkpoint_calibrator_identity"] != readiness[
        "model_checkpoint_calibrator_identity"
    ]:
        raise ValueError("S03 v3 model/checkpoint/calibrator stamp differs")
    _validate_code_identity(identity["code"], repository_root=repository_root)
    if identity["output"] != {
        "root": str(V3_OUTPUT_ROOT),
        "certificate": str(V3_CERTIFICATE_PATH),
        "infrastructure_blocker": str(V3_BLOCKER_PATH),
        "receipt_directory": "_receipts",
        "record_directory": "records",
    }:
        raise ValueError("S03 v3 output identity differs")
    if identity["lifecycle_states"] != [
        FROZEN_READY_BEFORE_OUTCOMES,
        EXECUTION_IN_PROGRESS,
        EXECUTION_BLOCKED_INFRA,
        EXECUTION_COMPLETE_PENDING_CERTIFICATE,
        CERTIFIED,
    ]:
        raise ValueError("S03 v3 lifecycle states differ")
    if identity["policies"] != {
        "all_620_logical_records_once": True,
        "ordered_execution": True,
        "single_use": True,
        "rerun": False,
        "skip": False,
        "replace": False,
        "cherry_pick": False,
        "v1_v2_index_0_reuse": False,
        "outcome_write_requires_explicit_flag": True,
        "valid_in_progress_receipts_allowed": True,
        "certificate_requires_620_valid_closed_receipts": True,
    }:
        raise ValueError("S03 v3 execution policies differ")
    if identity["execution_scope"] != {
        "offline_public_input_inference_only": True,
        "physical_rollout": False,
        "pi05_action_calls": False,
        "environment_steps": False,
        "legacy_oracle": False,
        "privileged_policy_inputs": False,
        "formal_method_claim": False,
    }:
        raise ValueError("S03 v3 identity permits an out-of-scope operation")
    return identity, model


def build_s03_v3_execution_plan(
    *, runner_identity_path: Path, repository_root: Path
) -> dict[str, Any]:
    identity, _ = validate_s03_v3_runner_identity(
        runner_identity_path, repository_root=repository_root
    )
    schedule_path = _resolve(
        identity["parent_logical_schedule"]["path"], repository_root=repository_root
    )
    schedule = validate_s03_offline_schedule(
        schedule_path, repository_root=repository_root
    )
    bindings = [
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
        "schema_version": "piu.s03-execution-plan.v3",
        "status": "FROZEN_BEFORE_S03_V3_OUTCOMES",
        "execution_version": EXECUTION_VERSION,
        "claim_scope": "PRE_OUTCOME_EXECUTION_AMENDMENT_NOT_PERFORMANCE_EVIDENCE",
        "parent_logical_schedule": _reference(
            schedule_path, repository_root=repository_root
        ),
        "logical_manifest": identity["logical_manifest"],
        "runner_identity": _reference(
            runner_identity_path, repository_root=repository_root
        ),
        "runtime_readiness": identity["runtime_readiness"],
        "historical_infrastructure_attempts": {
            "v1": identity["v1_seal"],
            "v2": identity["v2_seal"],
            "excluded_from_v3_performance_statistics": True,
            "v1_v2_index_0_rerun": False,
        },
        "execution_rule": "all_620_logical_records_once_in_parent_order_under_v3_identity",
        "record_count": RECORD_COUNT,
        "thresholds": copy.deepcopy(_THRESHOLDS),
        "logical_record_binding": {
            "algorithm": "parent_schedule_sha256_plus_ordered_record_binding_sha256",
            "ordered_record_ids": [row["record_id"] for row in bindings],
            "ordered_record_binding_sha256": canonical_sha256(bindings),
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


def validate_s03_v3_execution_plan(
    path: Path, *, repository_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = _load_mapping(path, "S03 v3 execution plan")
    runner_path = _verify_reference(
        plan.get("runner_identity"),
        "S03 v3 runner identity",
        repository_root=repository_root,
    )
    identity, model = validate_s03_v3_runner_identity(
        runner_path, repository_root=repository_root
    )
    expected = build_s03_v3_execution_plan(
        runner_identity_path=runner_path, repository_root=repository_root
    )
    if plan != expected:
        raise ValueError("S03 v3 execution plan differs from its deterministic binding")
    schedule = validate_s03_offline_schedule(
        repository_root / PARENT_SCHEDULE_PATH, repository_root=repository_root
    )
    for subtest, stratum in (
        ("A_INFORMATION_EFFECT", "PAIRED_PRE_POST"),
        ("B_DECISION_ROUTING", "ACT"),
        ("B_DECISION_ROUTING", "OPEN"),
        ("B_DECISION_ROUTING", "STOP"),
        ("C_CLOSED_LOOP_TRANSITION", "OBSERVE_OPEN_REOBSERVE"),
    ):
        indices = [
            row["linked_s02_index"]
            for row in schedule["records"]
            if row["subtest"] == subtest and row["stratum"] == stratum
        ]
        if indices != list(range(124)) or 101 not in indices or 104 not in indices:
            raise ValueError("S03 v3 filtered or reordered a logical stratum")
    return plan, identity, model


def validate_s03_v3_runner_preflight(
    *, plan_path: Path, execution_index: int, repository_root: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
]:
    plan, identity, model = validate_s03_v3_execution_plan(
        plan_path, repository_root=repository_root
    )
    if (
        isinstance(execution_index, bool)
        or not isinstance(execution_index, int)
        or not 0 <= execution_index < RECORD_COUNT
    ):
        raise IndexError("S03 v3 execution index must be in [0, 619]")
    schedule_path = repository_root / PARENT_SCHEDULE_PATH
    manifest_path = repository_root / LOGICAL_MANIFEST_PATH
    schedule = validate_s03_offline_schedule(
        schedule_path, repository_root=repository_root
    )
    manifest = validate_s03_input_manifest(
        manifest_path, repository_root=repository_root
    )
    model_path = _resolve(
        identity["model_identity"]["path"], repository_root=repository_root
    )
    request = build_s03_public_request(
        schedule=schedule,
        manifest=manifest,
        schedule_path=schedule_path,
        manifest_path=manifest_path,
        identity_path=model_path,
        identity=model,
        execution_index=execution_index,
        repository_root=repository_root,
    )
    output_root = repository_root / V3_OUTPUT_ROOT
    ledger = validate_s03_receipts(
        output_root,
        schedule=schedule,
        schedule_path=schedule_path,
        manifest_path=manifest_path,
        identity_path=model_path,
        identity=model,
        repository_root=repository_root,
    )
    certificate_present = (repository_root / V3_CERTIFICATE_PATH).exists()
    blocker_present = (repository_root / V3_BLOCKER_PATH).exists()
    state = classify_s03_v3_lifecycle(
        ledger,
        execution_authorized=True,
        certificate_present=certificate_present,
        certificate_validated=False,
        infrastructure_blocker_present=blocker_present,
    )
    if certificate_present:
        raise ValueError("S03 v3 certified execution requires the future certificate validator")
    return plan, identity, model, schedule, request, ledger, state
