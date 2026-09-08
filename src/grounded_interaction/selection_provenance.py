"""Immutable provenance for one deployed Method-V1 scorer decision.

A bare list of probabilities is not execution evidence: its checkpoint hashes
could be invented and its argmax recomputed without ever reading a model.  This
module freezes the output of the real scorer together with the checkpoint,
checkpoint-identity sidecar, detached Qwen cache files, calibration choice, and
decision-manifest identities that produced it.  Immediately before consuming a
physical execution slot, the collector both re-hashes those files and reruns the
bound scorer.  Re-hashing a forged JSON decision can therefore never substitute
for checkpoint inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import canonical_json_bytes, canonical_sha256
from .method_v1_data import (
    BranchScheduleEntry,
    DecisionGroupManifest,
    validate_branch_schedule,
)
from .qwen_provider import (
    QwenFeatureCache,
    candidate_feature_cache_key,
    qwen_cache_artifact_sha256,
)
from .selection import ExpectedSuccessSelector, ValuePrediction

FROZEN_SELECTION_SCHEMA = "method-v1-frozen-scorer-selection-v4"
SCORER_SELECTION_SCHEMA = "method-v1-scorer-selection-v1"
CALIBRATOR_SCHEMA = "method-v1-temperature-calibrator-v1"
IDENTITY_TEMPERATURE = "IDENTITY_TEMPERATURE"
FROZEN_TEMPERATURE = "FROZEN_TEMPERATURE_CALIBRATOR"
SCORER_REPLAY_ABS_TOLERANCE = 1e-6
_SHA256_HEX = frozenset("0123456789abcdef")
_CHECKPOINT_IDENTITY_KEYS = {
    "schema_version",
    "seed",
    "experiment_id",
    "configuration_sha256",
    "proposal_provider_id",
    "provider_id",
    "outcome_contract_sha256",
    "training_dataset_sha256",
    "training_admission_evidence_sha256",
    "collection_plan_sha256",
    "scorer_verifier_auth_key_id",
    "training_split_group_ids",
    "model",
    "training",
    "best_epoch",
    "best_validation_nll",
    "trainable_parameter_count",
}


def _require_sha256(value: object, *, name: str) -> str:
    result = str(value)
    if len(result) != 64 or any(character not in _SHA256_HEX for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _exact_mapping(
    value: object,
    *,
    keys: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} fields differ from its frozen schema")
    return value


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")


def _positive_temperature(value: object) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("selection temperature must be positive and finite")
    return result


def _checkpoint_identity(path: Path) -> tuple[dict[str, Any], str, str]:
    document = _read_json(path, name="checkpoint identity")
    if "identity_sha256" not in document:
        raise ValueError("checkpoint identity sidecar is missing identity_sha256")
    identity_sha256 = _require_sha256(
        document["identity_sha256"], name="checkpoint identity SHA-256"
    )
    identity = {
        key: value for key, value in document.items() if key != "identity_sha256"
    }
    if set(identity) != _CHECKPOINT_IDENTITY_KEYS:
        raise ValueError("outcome checkpoint identity fields differ from schema")
    if identity.get("schema_version") != "method-v1-outcome-checkpoint-v1":
        raise ValueError("unsupported outcome checkpoint identity schema")
    if canonical_sha256(identity) != identity_sha256:
        raise ValueError("checkpoint identity sidecar digest mismatch")
    return identity, identity_sha256, _file_sha256(path)


@dataclass(frozen=True)
class CheckpointTrainingProvenance:
    """Receipt-admission identities carried by a deployable checkpoint.

    These fields do not attest an untrusted host. They make accidental use of
    an unaudited, test-trained, or cross-plan checkpoint fail closed and keep
    the training-data identity visible through scorer replay and execution
    receipts.
    """

    training_dataset_sha256: str
    training_admission_evidence_sha256: str
    collection_plan_sha256: str
    scorer_verifier_auth_key_id: str
    training_split_group_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_dataset_sha256": self.training_dataset_sha256,
            "training_admission_evidence_sha256": (
                self.training_admission_evidence_sha256
            ),
            "collection_plan_sha256": self.collection_plan_sha256,
            "scorer_verifier_auth_key_id": self.scorer_verifier_auth_key_id,
            "training_split_group_ids": list(self.training_split_group_ids),
        }


def _validate_checkpoint_manifest_identity(
    identity: Mapping[str, Any], manifest: DecisionGroupManifest
) -> CheckpointTrainingProvenance:
    """Reject unaudited training, identity drift, and split-group leakage."""

    expected = {
        "experiment_id": manifest.experiment_id,
        "configuration_sha256": manifest.configuration_sha256,
        "proposal_provider_id": manifest.proposal_provider_id,
        "provider_id": manifest.token_provider_id,
        "outcome_contract_sha256": manifest.outcome_contract.fingerprint(),
    }
    for field, expected_value in expected.items():
        if identity.get(field) != expected_value:
            raise ValueError(f"checkpoint identity {field} differs from manifest")
    training_dataset_sha256 = _require_sha256(
        identity.get("training_dataset_sha256"),
        name="checkpoint training dataset SHA-256",
    )
    training_admission_evidence_sha256 = _require_sha256(
        identity.get("training_admission_evidence_sha256"),
        name="checkpoint training-admission evidence SHA-256",
    )
    collection_plan_sha256 = _require_sha256(
        identity.get("collection_plan_sha256"),
        name="checkpoint training collection-plan SHA-256",
    )
    scorer_verifier_auth_key_id = _require_sha256(
        identity.get("scorer_verifier_auth_key_id"),
        name="checkpoint scorer-verifier authentication key ID",
    )
    raw_groups = identity.get("training_split_group_ids")
    if (
        not isinstance(raw_groups, list)
        or not raw_groups
        or any(not isinstance(value, str) or not value.strip() for value in raw_groups)
        or len(set(raw_groups)) != len(raw_groups)
    ):
        raise ValueError("checkpoint training split-group identity is invalid")
    groups = tuple(raw_groups)
    if manifest.split_group_id in groups:
        raise ValueError(
            "deployment manifest split group overlaps checkpoint training data"
        )
    return CheckpointTrainingProvenance(
        training_dataset_sha256=training_dataset_sha256,
        training_admission_evidence_sha256=training_admission_evidence_sha256,
        collection_plan_sha256=collection_plan_sha256,
        scorer_verifier_auth_key_id=scorer_verifier_auth_key_id,
        training_split_group_ids=groups,
    )


def _cache_paths(cache_root: Path, key: str) -> tuple[Path, Path]:
    key = _require_sha256(key, name="token cache key")
    directory = cache_root / "sha256" / key[:2]
    return directory / f"{key}.pt", directory / f"{key}.json"


def _validate_cache_sidecar(
    *,
    tensor_path: Path,
    sidecar_path: Path,
    manifest: DecisionGroupManifest,
) -> dict[str, str]:
    sidecar = _exact_mapping(
        _read_json(sidecar_path, name="Qwen feature-cache sidecar"),
        keys={
            "schema",
            "logical_key",
            "provider_id",
            "context_fingerprint",
            "candidate_ids",
            "candidate_fingerprints",
            "primitives",
            "camera_ids",
            "frame_ids",
            "tensor_sha256",
            "tensor_artifact_sha256",
        },
        name="Qwen feature-cache sidecar",
    )
    logical_key = candidate_feature_cache_key(
        provider_id=manifest.token_provider_id,
        context=manifest.context,
        candidates=manifest.candidates,
    )
    expected = {
        "schema": QwenFeatureCache.SIDECAR_SCHEMA,
        "logical_key": logical_key,
        "provider_id": manifest.token_provider_id,
        "context_fingerprint": manifest.context.fingerprint(),
        "candidate_ids": [candidate.candidate_id for candidate in manifest.candidates],
        "candidate_fingerprints": [
            candidate.fingerprint() for candidate in manifest.candidates
        ],
        "primitives": [candidate.primitive.value for candidate in manifest.candidates],
    }
    for field, expected_value in expected.items():
        if sidecar[field] != expected_value:
            raise ValueError(f"Qwen feature-cache sidecar changed {field}")
    tensor_sha256 = sidecar["tensor_sha256"]
    if not isinstance(tensor_sha256, Mapping) or set(tensor_sha256) != (
        QwenFeatureCache._tensor_names()
    ):
        raise ValueError("Qwen feature-cache tensor digest fields changed")
    for field, digest in tensor_sha256.items():
        if not str(field).strip():
            raise ValueError("Qwen feature-cache tensor name must be non-empty")
        _require_sha256(digest, name=f"Qwen feature-cache tensor {field!s}")
    tensor_artifact_sha256 = _file_sha256(tensor_path)
    if sidecar["tensor_artifact_sha256"] != tensor_artifact_sha256:
        raise ValueError("Qwen feature-cache serialized tensor digest changed")
    sidecar_artifact_sha256 = _file_sha256(sidecar_path)
    artifact_sha256 = qwen_cache_artifact_sha256(
        logical_key=logical_key,
        tensor_artifact_sha256=tensor_artifact_sha256,
        sidecar_artifact_sha256=sidecar_artifact_sha256,
    )
    if artifact_sha256 != manifest.token_cache_sha256:
        raise ValueError("Qwen feature-cache artifact identity changed")
    return {
        "logical_key": logical_key,
        "artifact_sha256": artifact_sha256,
        "tensor_artifact_sha256": tensor_artifact_sha256,
        "sidecar_artifact_sha256": sidecar_artifact_sha256,
    }


def _selection_payload(
    value: object,
    *,
    manifest: DecisionGroupManifest,
    entry: BranchScheduleEntry,
) -> tuple[Mapping[str, Any], float]:
    expected_entry = {
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest.fingerprint(),
        "decision_group_id": manifest.decision_group_id,
        "initial_state_group": manifest.initial_state_group,
        "split_group_id": manifest.split_group_id,
        "split": manifest.split,
        "reset_state_sha256": manifest.reset_state_sha256,
        "outcome_contract_sha256": manifest.outcome_contract.fingerprint(),
    }
    for field, expected in expected_entry.items():
        if getattr(entry, field) != expected:
            raise ValueError(f"schedule entry changed manifest-bound field {field}")
    selection = _exact_mapping(
        value,
        keys={
            "schema_version",
            "evidence_scope",
            "manifest_sha256",
            "token_cache_sha256",
            "checkpoint_sha256",
            "checkpoint_identity_sha256",
            "outcome_contract_sha256",
            "temperature",
            "predictions",
            "decision",
        },
        name="scorer selection",
    )
    if (
        selection["schema_version"] != SCORER_SELECTION_SCHEMA
        or selection["evidence_scope"] != "PRE_EXECUTION_MODEL_DECISION"
    ):
        raise ValueError("unsupported scorer selection schema")
    expected_bindings = {
        "manifest_sha256": manifest.fingerprint(),
        "token_cache_sha256": manifest.token_cache_sha256,
        "outcome_contract_sha256": manifest.outcome_contract.fingerprint(),
    }
    for field, expected in expected_bindings.items():
        if selection[field] != expected:
            raise ValueError(f"scorer selection changed {field}")
    _require_sha256(selection["checkpoint_sha256"], name="selection checkpoint")
    _require_sha256(
        selection["checkpoint_identity_sha256"],
        name="selection checkpoint identity",
    )
    temperature = _positive_temperature(selection["temperature"])
    raw_predictions = selection["predictions"]
    if not isinstance(raw_predictions, list):
        raise TypeError("selection predictions must be a list")
    predictions: list[ValuePrediction] = []
    for raw_prediction in raw_predictions:
        prediction = _exact_mapping(
            raw_prediction,
            keys={
                "candidate_id",
                "candidate_fingerprint",
                "success_probability",
                "feasible",
            },
            name="selection prediction",
        )
        predictions.append(
            ValuePrediction(
                candidate_id=str(prediction["candidate_id"]),
                candidate_fingerprint=str(prediction["candidate_fingerprint"]),
                success_probability=float(prediction["success_probability"]),
                feasible=prediction["feasible"],
            )
        )
    decision = ExpectedSuccessSelector().select(manifest.candidates, predictions)
    if (
        not isinstance(selection["decision"], Mapping)
        or dict(selection["decision"]) != decision.to_dict()
    ):
        raise ValueError("selection decision does not match its predictions")
    if decision.candidate is None:
        raise ValueError("an abstaining selection cannot execute a schedule entry")
    if (
        decision.candidate.candidate_id != entry.candidate_id
        or decision.candidate.fingerprint() != entry.candidate_fingerprint
    ):
        raise ValueError("schedule entry differs from scorer selection")
    return selection, temperature


def _calibrator_document(
    path: Path,
    *,
    temperature: float,
    checkpoint_sha256: str,
    checkpoint_identity_sha256: str,
    outcome_contract_sha256: str,
) -> tuple[dict[str, Any], str, str]:
    document = _read_json(path, name="temperature calibrator")
    calibrator = _exact_mapping(
        document,
        keys={
            "schema_version",
            "method",
            "temperature",
            "checkpoint_sha256",
            "checkpoint_identity_sha256",
            "outcome_contract_sha256",
            "calibration_dataset_sha256",
            "calibration_split_group_ids_sha256",
            "fit_config_sha256",
            "calibration_group_count",
            "calibration_example_count",
            "identity_sha256",
        },
        name="temperature calibrator",
    )
    if (
        calibrator["schema_version"] != CALIBRATOR_SCHEMA
        or calibrator["method"] != "positive_scalar_temperature"
    ):
        raise ValueError("unsupported temperature calibrator")
    stored_identity = _require_sha256(
        calibrator["identity_sha256"], name="calibrator identity"
    )
    body = {key: value for key, value in calibrator.items() if key != "identity_sha256"}
    if canonical_sha256(body) != stored_identity:
        raise ValueError("temperature calibrator identity digest mismatch")
    if _positive_temperature(calibrator["temperature"]) != temperature:
        raise ValueError("selection temperature differs from frozen calibrator")
    expected = {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_identity_sha256": checkpoint_identity_sha256,
        "outcome_contract_sha256": outcome_contract_sha256,
    }
    for field, expected_value in expected.items():
        if calibrator[field] != expected_value:
            raise ValueError(f"temperature calibrator changed {field}")
    for field in (
        "calibration_dataset_sha256",
        "calibration_split_group_ids_sha256",
        "fit_config_sha256",
    ):
        _require_sha256(calibrator[field], name=field)
    for field in ("calibration_group_count", "calibration_example_count"):
        count = calibrator[field]
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"{field} must be a positive integer")
    return dict(calibrator), stored_identity, _file_sha256(path)


def _calibration_provenance(
    *,
    selection_temperature: float,
    calibrator_path: Path | None,
    expected_calibrator_sha256: str | None,
    checkpoint_sha256: str,
    checkpoint_identity_sha256: str,
    outcome_contract_sha256: str,
) -> dict[str, Any]:
    if calibrator_path is None:
        if selection_temperature != 1.0:
            raise ValueError(
                "non-identity temperature requires a frozen calibrator artifact"
            )
        if expected_calibrator_sha256 is not None:
            raise ValueError("identity temperature cannot name a calibrator digest")
        return {
            "mode": IDENTITY_TEMPERATURE,
            "temperature": 1.0,
            "artifact_path": None,
            "artifact_file_sha256": None,
            "artifact_identity_sha256": None,
        }
    calibrator_path = calibrator_path.expanduser().resolve()
    _calibrator, identity_sha256, file_sha256 = _calibrator_document(
        calibrator_path,
        temperature=selection_temperature,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_identity_sha256=checkpoint_identity_sha256,
        outcome_contract_sha256=outcome_contract_sha256,
    )
    if expected_calibrator_sha256 is not None and file_sha256 != _require_sha256(
        expected_calibrator_sha256, name="expected calibrator file"
    ):
        raise ValueError("actual calibrator file differs from frozen expectation")
    return {
        "mode": FROZEN_TEMPERATURE,
        "temperature": selection_temperature,
        "artifact_path": str(calibrator_path),
        "artifact_file_sha256": file_sha256,
        "artifact_identity_sha256": identity_sha256,
    }


def _build_frozen_selection_artifact(
    selection_value: object,
    *,
    manifest: DecisionGroupManifest,
    entry: BranchScheduleEntry,
    checkpoint_path: str | Path,
    checkpoint_identity_path: str | Path,
    cache_root: str | Path,
    expected_checkpoint_sha256: str,
    expected_checkpoint_identity_sha256: str,
    calibrator_path: str | Path | None = None,
    expected_calibrator_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind a scorer result to actual frozen files and decision identities.

    This is deliberately private: a caller-provided score is only a proposed
    serialization.  :func:`verify_frozen_scorer_selection` must replay the bound
    checkpoint before the artifact is admissible for physical execution.
    """

    if not isinstance(manifest, DecisionGroupManifest):
        raise TypeError("manifest must be a DecisionGroupManifest")
    if not isinstance(entry, BranchScheduleEntry):
        raise TypeError("entry must be a BranchScheduleEntry")
    selection, temperature = _selection_payload(
        selection_value, manifest=manifest, entry=entry
    )
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    checkpoint_identity = Path(checkpoint_identity_path).expanduser().resolve()
    actual_checkpoint_sha256 = _file_sha256(checkpoint)
    expected_checkpoint_sha256 = _require_sha256(
        expected_checkpoint_sha256, name="expected checkpoint"
    )
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("actual checkpoint differs from frozen expectation")
    identity, actual_identity_sha256, identity_file_sha256 = _checkpoint_identity(
        checkpoint_identity
    )
    expected_checkpoint_identity_sha256 = _require_sha256(
        expected_checkpoint_identity_sha256,
        name="expected checkpoint identity",
    )
    if actual_identity_sha256 != expected_checkpoint_identity_sha256:
        raise ValueError("actual checkpoint identity differs from frozen expectation")
    if selection["checkpoint_sha256"] != actual_checkpoint_sha256:
        raise ValueError("selection does not name the actual checkpoint file")
    if selection["checkpoint_identity_sha256"] != actual_identity_sha256:
        raise ValueError("selection does not name the actual checkpoint identity")
    expected_contract = manifest.outcome_contract.fingerprint()
    training_provenance = _validate_checkpoint_manifest_identity(identity, manifest)

    cache_root_path = Path(cache_root).expanduser().resolve()
    cache_tensor, cache_sidecar = _cache_paths(
        cache_root_path, manifest.token_cache_sha256
    )
    cache_identity = _validate_cache_sidecar(
        tensor_path=cache_tensor,
        sidecar_path=cache_sidecar,
        manifest=manifest,
    )
    cache = {
        "root": str(cache_root_path),
        "logical_key": cache_identity["logical_key"],
        "artifact_sha256": cache_identity["artifact_sha256"],
        "tensor_path": str(cache_tensor),
        "tensor_artifact_sha256": cache_identity["tensor_artifact_sha256"],
        "sidecar_path": str(cache_sidecar),
        "sidecar_artifact_sha256": cache_identity["sidecar_artifact_sha256"],
    }
    calibration = _calibration_provenance(
        selection_temperature=temperature,
        calibrator_path=(None if calibrator_path is None else Path(calibrator_path)),
        expected_calibrator_sha256=expected_calibrator_sha256,
        checkpoint_sha256=actual_checkpoint_sha256,
        checkpoint_identity_sha256=actual_identity_sha256,
        outcome_contract_sha256=expected_contract,
    )
    body = {
        "schema_version": FROZEN_SELECTION_SCHEMA,
        "evidence_scope": "PRE_EXECUTION_MODEL_DECISION",
        "producer": ("grounded_interaction.train_outcomes.select_manifest_candidate"),
        "bindings": {
            "manifest_sha256": manifest.fingerprint(),
            "experiment_id": manifest.experiment_id,
            "configuration_sha256": manifest.configuration_sha256,
            "proposal_provider_id": manifest.proposal_provider_id,
            "token_cache_sha256": manifest.token_cache_sha256,
            "token_provider_id": manifest.token_provider_id,
            "outcome_contract_sha256": expected_contract,
            "schedule_entry_id": entry.entry_id,
            "candidate_id": entry.candidate_id,
            "candidate_fingerprint": entry.candidate_fingerprint,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": actual_checkpoint_sha256,
            "expected_sha256": expected_checkpoint_sha256,
            "identity_path": str(checkpoint_identity),
            "identity_file_sha256": identity_file_sha256,
            "identity_sha256": actual_identity_sha256,
            "expected_identity_sha256": expected_checkpoint_identity_sha256,
        },
        "training_provenance": training_provenance.to_dict(),
        "cache": cache,
        "calibration": calibration,
        "selection": dict(selection),
        "selection_sha256": canonical_sha256(selection),
    }
    return {**body, "artifact_sha256": canonical_sha256(body)}


@dataclass(frozen=True)
class VerifiedFrozenSelection:
    artifact_sha256: str
    selection_sha256: str
    checkpoint_sha256: str
    checkpoint_identity_sha256: str
    training_dataset_sha256: str
    training_admission_evidence_sha256: str
    training_collection_plan_sha256: str
    scorer_verifier_auth_key_id: str
    candidate_id: str
    candidate_fingerprint: str
    temperature: float


def verify_frozen_scorer_selection(
    path: str | Path,
    *,
    manifest: DecisionGroupManifest,
    entry: BranchScheduleEntry,
) -> VerifiedFrozenSelection:
    """Re-hash and replay every frozen scorer input before execution.

    The JSON probabilities are never trusted as evidence.  After validating
    all immutable identities, this function runs the named checkpoint against
    the named physical Qwen cache on CPU and compares every candidate score,
    feasibility bit, and the resulting argmax with the frozen declaration.
    """

    artifact_path = Path(path).expanduser().resolve()
    artifact = _exact_mapping(
        _read_json(artifact_path, name="frozen scorer selection"),
        keys={
            "schema_version",
            "evidence_scope",
            "producer",
            "bindings",
            "checkpoint",
            "training_provenance",
            "cache",
            "calibration",
            "selection",
            "selection_sha256",
            "artifact_sha256",
        },
        name="frozen scorer selection",
    )
    if (
        artifact["schema_version"] != FROZEN_SELECTION_SCHEMA
        or artifact["evidence_scope"] != "PRE_EXECUTION_MODEL_DECISION"
        or artifact["producer"]
        != "grounded_interaction.train_outcomes.select_manifest_candidate"
    ):
        raise ValueError("unsupported frozen scorer selection provenance")
    body = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    artifact_sha256 = _require_sha256(
        artifact["artifact_sha256"], name="frozen selection artifact"
    )
    if canonical_sha256(body) != artifact_sha256:
        raise ValueError("frozen scorer selection artifact digest mismatch")

    bindings = _exact_mapping(
        artifact["bindings"],
        keys={
            "manifest_sha256",
            "experiment_id",
            "configuration_sha256",
            "proposal_provider_id",
            "token_cache_sha256",
            "token_provider_id",
            "outcome_contract_sha256",
            "schedule_entry_id",
            "candidate_id",
            "candidate_fingerprint",
        },
        name="frozen selection bindings",
    )
    expected_bindings = {
        "manifest_sha256": manifest.fingerprint(),
        "experiment_id": manifest.experiment_id,
        "configuration_sha256": manifest.configuration_sha256,
        "proposal_provider_id": manifest.proposal_provider_id,
        "token_cache_sha256": manifest.token_cache_sha256,
        "token_provider_id": manifest.token_provider_id,
        "outcome_contract_sha256": manifest.outcome_contract.fingerprint(),
        "schedule_entry_id": entry.entry_id,
        "candidate_id": entry.candidate_id,
        "candidate_fingerprint": entry.candidate_fingerprint,
    }
    if dict(bindings) != expected_bindings:
        raise ValueError("frozen selection bindings differ from execution entry")

    selection, temperature = _selection_payload(
        artifact["selection"], manifest=manifest, entry=entry
    )
    selection_sha256 = _require_sha256(
        artifact["selection_sha256"], name="selection payload"
    )
    if canonical_sha256(selection) != selection_sha256:
        raise ValueError("selection payload digest mismatch")

    checkpoint = _exact_mapping(
        artifact["checkpoint"],
        keys={
            "path",
            "sha256",
            "expected_sha256",
            "identity_path",
            "identity_file_sha256",
            "identity_sha256",
            "expected_identity_sha256",
        },
        name="selection checkpoint provenance",
    )
    checkpoint_path = Path(str(checkpoint["path"])).expanduser().resolve()
    actual_checkpoint_sha256 = _file_sha256(checkpoint_path)
    for field in ("sha256", "expected_sha256"):
        if _require_sha256(checkpoint[field], name=f"checkpoint {field}") != (
            actual_checkpoint_sha256
        ):
            raise ValueError("checkpoint file changed after selection freeze")
    identity_path = Path(str(checkpoint["identity_path"])).expanduser().resolve()
    identity, identity_sha256, identity_file_sha256 = _checkpoint_identity(
        identity_path
    )
    if (
        _require_sha256(
            checkpoint["identity_file_sha256"], name="checkpoint identity file"
        )
        != identity_file_sha256
    ):
        raise ValueError("checkpoint identity file changed after selection freeze")
    for field in ("identity_sha256", "expected_identity_sha256"):
        if _require_sha256(checkpoint[field], name=f"checkpoint {field}") != (
            identity_sha256
        ):
            raise ValueError("checkpoint identity changed after selection freeze")
    if selection["checkpoint_sha256"] != actual_checkpoint_sha256:
        raise ValueError("selection checkpoint binding changed")
    if selection["checkpoint_identity_sha256"] != identity_sha256:
        raise ValueError("selection checkpoint identity binding changed")
    training_provenance = _validate_checkpoint_manifest_identity(identity, manifest)
    frozen_training_provenance = _exact_mapping(
        artifact["training_provenance"],
        keys={
            "training_dataset_sha256",
            "training_admission_evidence_sha256",
            "collection_plan_sha256",
            "scorer_verifier_auth_key_id",
            "training_split_group_ids",
        },
        name="checkpoint training provenance",
    )
    if dict(frozen_training_provenance) != training_provenance.to_dict():
        raise ValueError(
            "checkpoint training provenance changed after selection freeze"
        )

    cache = _exact_mapping(
        artifact["cache"],
        keys={
            "root",
            "logical_key",
            "artifact_sha256",
            "tensor_path",
            "tensor_artifact_sha256",
            "sidecar_path",
            "sidecar_artifact_sha256",
        },
        name="selection cache provenance",
    )
    cache_root = Path(str(cache["root"])).expanduser().resolve()
    expected_tensor, expected_sidecar = _cache_paths(
        cache_root, manifest.token_cache_sha256
    )
    if (
        Path(str(cache["tensor_path"])).expanduser().resolve() != expected_tensor
        or Path(str(cache["sidecar_path"])).expanduser().resolve() != expected_sidecar
    ):
        raise ValueError("selection cache paths changed")
    if _file_sha256(expected_tensor) != _require_sha256(
        cache["tensor_artifact_sha256"], name="cache tensor artifact"
    ):
        raise ValueError("Qwen feature-cache tensor file changed after selection")
    if _file_sha256(expected_sidecar) != _require_sha256(
        cache["sidecar_artifact_sha256"], name="cache sidecar artifact"
    ):
        raise ValueError("Qwen feature-cache sidecar changed after selection")
    cache_identity = _validate_cache_sidecar(
        tensor_path=expected_tensor,
        sidecar_path=expected_sidecar,
        manifest=manifest,
    )
    for field in (
        "logical_key",
        "artifact_sha256",
        "tensor_artifact_sha256",
        "sidecar_artifact_sha256",
    ):
        if cache[field] != cache_identity[field]:
            raise ValueError(f"Qwen feature-cache changed {field}")

    calibration = _exact_mapping(
        artifact["calibration"],
        keys={
            "mode",
            "temperature",
            "artifact_path",
            "artifact_file_sha256",
            "artifact_identity_sha256",
        },
        name="selection calibration provenance",
    )
    if _positive_temperature(calibration["temperature"]) != temperature:
        raise ValueError("selection and provenance temperatures differ")
    if calibration["mode"] == IDENTITY_TEMPERATURE:
        if temperature != 1.0 or any(
            calibration[field] is not None
            for field in (
                "artifact_path",
                "artifact_file_sha256",
                "artifact_identity_sha256",
            )
        ):
            raise ValueError("identity-temperature provenance is inconsistent")
    elif calibration["mode"] == FROZEN_TEMPERATURE:
        calibrator_path = Path(str(calibration["artifact_path"])).expanduser().resolve()
        _, calibrator_identity, calibrator_file = _calibrator_document(
            calibrator_path,
            temperature=temperature,
            checkpoint_sha256=actual_checkpoint_sha256,
            checkpoint_identity_sha256=identity_sha256,
            outcome_contract_sha256=manifest.outcome_contract.fingerprint(),
        )
        if calibrator_file != _require_sha256(
            calibration["artifact_file_sha256"], name="calibrator file"
        ) or calibrator_identity != _require_sha256(
            calibration["artifact_identity_sha256"], name="calibrator identity"
        ):
            raise ValueError("temperature calibrator changed after selection freeze")
    else:
        raise ValueError("unknown selection calibration mode")

    from .train_outcomes import select_manifest_candidate

    replayed = select_manifest_candidate(
        manifest,
        cache_root=cache_root,
        checkpoint_path=checkpoint_path,
        device="cpu",
        temperature=temperature,
    ).to_dict()
    replayed_selection, replayed_temperature = _selection_payload(
        replayed, manifest=manifest, entry=entry
    )
    if replayed_temperature != temperature:
        raise ValueError("checkpoint replay changed selection temperature")
    for field in (
        "schema_version",
        "evidence_scope",
        "manifest_sha256",
        "token_cache_sha256",
        "checkpoint_sha256",
        "checkpoint_identity_sha256",
        "outcome_contract_sha256",
        "temperature",
    ):
        if replayed_selection[field] != selection[field]:
            raise ValueError(f"checkpoint replay changed selection field {field}")
    declared_decision = selection["decision"]
    replayed_decision = replayed_selection["decision"]
    for field in (
        "status",
        "candidate_id",
        "candidate_fingerprint",
        "reason",
    ):
        if replayed_decision[field] != declared_decision[field]:
            raise ValueError(f"checkpoint replay changed decision field {field}")
    if not math.isclose(
        float(declared_decision["success_probability"]),
        float(replayed_decision["success_probability"]),
        rel_tol=0.0,
        abs_tol=SCORER_REPLAY_ABS_TOLERANCE,
    ):
        raise ValueError("declared decision score differs from checkpoint replay")
    declared_predictions = selection["predictions"]
    replayed_predictions = replayed_selection["predictions"]
    if len(declared_predictions) != len(replayed_predictions):
        raise ValueError("checkpoint replay changed candidate score count")
    for declared, actual in zip(
        declared_predictions, replayed_predictions, strict=True
    ):
        for field in ("candidate_id", "candidate_fingerprint", "feasible"):
            if declared[field] != actual[field]:
                raise ValueError(
                    f"checkpoint replay changed candidate prediction field {field}"
                )
        if not math.isclose(
            float(declared["success_probability"]),
            float(actual["success_probability"]),
            rel_tol=0.0,
            abs_tol=SCORER_REPLAY_ABS_TOLERANCE,
        ):
            raise ValueError("declared candidate score differs from checkpoint replay")

    return VerifiedFrozenSelection(
        artifact_sha256=artifact_sha256,
        selection_sha256=selection_sha256,
        checkpoint_sha256=actual_checkpoint_sha256,
        checkpoint_identity_sha256=identity_sha256,
        training_dataset_sha256=training_provenance.training_dataset_sha256,
        training_admission_evidence_sha256=(
            training_provenance.training_admission_evidence_sha256
        ),
        training_collection_plan_sha256=(training_provenance.collection_plan_sha256),
        scorer_verifier_auth_key_id=(training_provenance.scorer_verifier_auth_key_id),
        candidate_id=entry.candidate_id,
        candidate_fingerprint=entry.candidate_fingerprint,
        temperature=temperature,
    )


def _score_manifest_once(
    *,
    manifest: DecisionGroupManifest,
    cache_root: str | Path,
    checkpoint_path: str | Path,
    checkpoint_identity_path: str | Path,
    expected_checkpoint_sha256: str,
    expected_checkpoint_identity_sha256: str,
    device: str = "cpu",
    temperature: float = 1.0,
    calibrator_path: str | Path | None = None,
    expected_calibrator_sha256: str | None = None,
) -> Any:
    """Validate immutable inputs, then run the outcome scorer exactly once."""

    # Validate immutable deployment inputs before loading the scorer or cache.
    # This prevents a mistyped "expected" digest from producing a plausible
    # decision that is rejected only after expensive model inference.
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    expected_checkpoint_sha256 = _require_sha256(
        expected_checkpoint_sha256, name="expected checkpoint"
    )
    if _file_sha256(checkpoint) != expected_checkpoint_sha256:
        raise ValueError("actual checkpoint differs from frozen expectation")
    identity_path = Path(checkpoint_identity_path).expanduser().resolve()
    identity, identity_sha256, _ = _checkpoint_identity(identity_path)
    expected_checkpoint_identity_sha256 = _require_sha256(
        expected_checkpoint_identity_sha256,
        name="expected checkpoint identity",
    )
    if identity_sha256 != expected_checkpoint_identity_sha256:
        raise ValueError("actual checkpoint identity differs from frozen expectation")
    _validate_checkpoint_manifest_identity(identity, manifest)
    cache_tensor, cache_sidecar = _cache_paths(
        Path(cache_root).expanduser().resolve(), manifest.token_cache_sha256
    )
    _file_sha256(cache_tensor)
    _validate_cache_sidecar(
        tensor_path=cache_tensor,
        sidecar_path=cache_sidecar,
        manifest=manifest,
    )
    _calibration_provenance(
        selection_temperature=_positive_temperature(temperature),
        calibrator_path=(None if calibrator_path is None else Path(calibrator_path)),
        expected_calibrator_sha256=expected_calibrator_sha256,
        checkpoint_sha256=expected_checkpoint_sha256,
        checkpoint_identity_sha256=expected_checkpoint_identity_sha256,
        outcome_contract_sha256=manifest.outcome_contract.fingerprint(),
    )

    from .train_outcomes import select_manifest_candidate

    return select_manifest_candidate(
        manifest,
        cache_root=cache_root,
        checkpoint_path=checkpoint_path,
        device=device,
        temperature=temperature,
    )


def _freeze_scorer_result(
    selection_value: object,
    *,
    manifest: DecisionGroupManifest,
    entry: BranchScheduleEntry,
    cache_root: str | Path,
    checkpoint_path: str | Path,
    checkpoint_identity_path: str | Path,
    expected_checkpoint_sha256: str,
    expected_checkpoint_identity_sha256: str,
    output_path: str | Path,
    calibrator_path: str | Path | None = None,
    expected_calibrator_sha256: str | None = None,
) -> VerifiedFrozenSelection:
    """Freeze a previously computed scorer result without running it again."""

    artifact = _build_frozen_selection_artifact(
        selection_value,
        manifest=manifest,
        entry=entry,
        checkpoint_path=checkpoint_path,
        checkpoint_identity_path=checkpoint_identity_path,
        cache_root=cache_root,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_checkpoint_identity_sha256=expected_checkpoint_identity_sha256,
        calibrator_path=calibrator_path,
        expected_calibrator_sha256=expected_calibrator_sha256,
    )
    destination = Path(output_path).expanduser().resolve()
    _write_json_once(destination, artifact)
    return verify_frozen_scorer_selection(destination, manifest=manifest, entry=entry)


def resolve_selected_schedule_entry(
    *,
    manifest: DecisionGroupManifest,
    schedule: Sequence[BranchScheduleEntry],
    candidate_id: str,
    candidate_fingerprint: str,
    model_seed: int,
) -> BranchScheduleEntry:
    """Resolve one scorer argmax to exactly one frozen execution row.

    The complete local schedule is validated before matching. Consequently a
    missing, duplicated, filtered, or outcome-augmented row fails closed rather
    than changing which physical branch will be consumed.
    """

    if not isinstance(manifest, DecisionGroupManifest):
        raise TypeError("manifest must be a DecisionGroupManifest")
    rows = tuple(schedule)
    validate_branch_schedule((manifest,), rows)
    if (
        not isinstance(model_seed, int)
        or isinstance(model_seed, bool)
        or model_seed < 0
    ):
        raise ValueError("model_seed must be a non-negative integer")
    candidates = [
        candidate
        for candidate in manifest.candidates
        if candidate.candidate_id == candidate_id
        and candidate.fingerprint() == candidate_fingerprint
    ]
    if len(candidates) != 1:
        raise ValueError("scorer decision does not identify one manifest candidate")
    matches = [
        row
        for row in rows
        if row.manifest_id == manifest.manifest_id
        and row.manifest_sha256 == manifest.fingerprint()
        and row.candidate_id == candidate_id
        and row.candidate_fingerprint == candidate_fingerprint
        and row.model_seed == model_seed
    ]
    if len(matches) != 1:
        raise ValueError(
            "selected candidate and model seed must identify exactly one "
            "frozen schedule entry"
        )
    return matches[0]


def create_frozen_scorer_selection(
    *,
    manifest: DecisionGroupManifest,
    entry: BranchScheduleEntry,
    cache_root: str | Path,
    checkpoint_path: str | Path,
    checkpoint_identity_path: str | Path,
    expected_checkpoint_sha256: str,
    expected_checkpoint_identity_sha256: str,
    output_path: str | Path,
    device: str = "cpu",
    temperature: float = 1.0,
    calibrator_path: str | Path | None = None,
    expected_calibrator_sha256: str | None = None,
) -> VerifiedFrozenSelection:
    """Run the scorer once, then freeze a caller-selected schedule row.

    This is the backwards-compatible execution-index path. The frozen builder
    still verifies that the row is the scorer's deterministic argmax.
    """

    result = _score_manifest_once(
        manifest=manifest,
        cache_root=cache_root,
        checkpoint_path=checkpoint_path,
        checkpoint_identity_path=checkpoint_identity_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_checkpoint_identity_sha256=expected_checkpoint_identity_sha256,
        device=device,
        temperature=temperature,
        calibrator_path=calibrator_path,
        expected_calibrator_sha256=expected_calibrator_sha256,
    )
    return _freeze_scorer_result(
        result.to_dict(),
        manifest=manifest,
        entry=entry,
        cache_root=cache_root,
        checkpoint_path=checkpoint_path,
        checkpoint_identity_path=checkpoint_identity_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_checkpoint_identity_sha256=expected_checkpoint_identity_sha256,
        output_path=output_path,
        calibrator_path=calibrator_path,
        expected_calibrator_sha256=expected_calibrator_sha256,
    )


def create_frozen_scorer_selection_for_model_seed(
    *,
    manifest: DecisionGroupManifest,
    schedule: Sequence[BranchScheduleEntry],
    model_seed: int,
    cache_root: str | Path,
    checkpoint_path: str | Path,
    checkpoint_identity_path: str | Path,
    expected_checkpoint_sha256: str,
    expected_checkpoint_identity_sha256: str,
    output_path: str | Path,
    device: str = "cpu",
    temperature: float = 1.0,
    calibrator_path: str | Path | None = None,
    expected_calibrator_sha256: str | None = None,
) -> tuple[VerifiedFrozenSelection, BranchScheduleEntry]:
    """Score all candidates once and resolve the selected frozen schedule row.

    No outcome, post-action observation, evaluator sidecar, or branch receipt is
    accepted by this interface. ``model_seed`` selects only the pre-registered
    repetition of the model-selected candidate; it does not affect scoring.
    """

    rows = tuple(schedule)
    # Reject malformed schedules before loading the model. This proves that
    # every candidate/seed branch was frozen before the decision.
    validate_branch_schedule((manifest,), rows)
    result = _score_manifest_once(
        manifest=manifest,
        cache_root=cache_root,
        checkpoint_path=checkpoint_path,
        checkpoint_identity_path=checkpoint_identity_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_checkpoint_identity_sha256=expected_checkpoint_identity_sha256,
        device=device,
        temperature=temperature,
        calibrator_path=calibrator_path,
        expected_calibrator_sha256=expected_calibrator_sha256,
    )
    candidate = result.decision.candidate
    if candidate is None:
        raise ValueError("an abstaining scorer decision has no executable schedule row")
    entry = resolve_selected_schedule_entry(
        manifest=manifest,
        schedule=rows,
        candidate_id=candidate.candidate_id,
        candidate_fingerprint=candidate.fingerprint(),
        model_seed=model_seed,
    )
    verified = _freeze_scorer_result(
        result.to_dict(),
        manifest=manifest,
        entry=entry,
        cache_root=cache_root,
        checkpoint_path=checkpoint_path,
        checkpoint_identity_path=checkpoint_identity_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_checkpoint_identity_sha256=expected_checkpoint_identity_sha256,
        output_path=output_path,
        calibrator_path=calibrator_path,
        expected_calibrator_sha256=expected_calibrator_sha256,
    )
    return verified, entry


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run and freeze one provenance-bound Method-V1 scorer decision"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    row_selector = parser.add_mutually_exclusive_group(required=True)
    row_selector.add_argument(
        "--model-seed",
        type=int,
        help=(
            "score the complete manifest once and automatically resolve the "
            "selected candidate's row for this frozen repetition seed"
        ),
    )
    row_selector.add_argument(
        "--execution-index",
        type=int,
        help="legacy explicit row; it must already match the scorer's argmax",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-identity", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-checkpoint-identity-sha256", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--calibrator", type=Path)
    parser.add_argument("--expected-calibrator-sha256")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = DecisionGroupManifest.from_mapping(
        _read_json(args.manifest.resolve(), name="decision manifest")
    )
    schedule_document = _read_json(args.schedule.resolve(), name="branch schedule")
    entries = schedule_document.get("entries")
    if not isinstance(entries, list):
        raise TypeError("branch schedule entries must be a list")
    if any(not isinstance(value, Mapping) for value in entries):
        raise TypeError("branch schedule entries must be JSON objects")
    schedule = tuple(BranchScheduleEntry.from_mapping(value) for value in entries)
    validate_branch_schedule((manifest,), schedule)
    common = {
        "manifest": manifest,
        "cache_root": args.cache_root,
        "checkpoint_path": args.checkpoint,
        "checkpoint_identity_path": args.checkpoint_identity,
        "expected_checkpoint_sha256": args.expected_checkpoint_sha256,
        "expected_checkpoint_identity_sha256": (
            args.expected_checkpoint_identity_sha256
        ),
        "output_path": args.output,
        "device": args.device,
        "temperature": args.temperature,
        "calibrator_path": args.calibrator,
        "expected_calibrator_sha256": args.expected_calibrator_sha256,
    }
    if args.model_seed is not None:
        verified, entry = create_frozen_scorer_selection_for_model_seed(
            schedule=schedule,
            model_seed=args.model_seed,
            **common,
        )
    else:
        matching = [
            entry for entry in schedule if entry.execution_index == args.execution_index
        ]
        if len(matching) != 1:
            raise ValueError("execution-index must identify exactly one schedule entry")
        entry = matching[0]
        verified = create_frozen_scorer_selection(entry=entry, **common)
    print(
        json.dumps(
            {**verified.__dict__, "execution_index": entry.execution_index},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the API
    raise SystemExit(_main())
