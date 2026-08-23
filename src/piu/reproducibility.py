"""Hash-bound offline release and explicit external-gate audit."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(str(item) for item in value)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be non-empty and duplicate-free")
    if any(Path(item).is_absolute() or ".." in Path(item).parts for item in result):
        raise ValueError(f"{name} must contain repository-relative paths")
    return result


def _validate_parse(path: Path) -> None:
    if path.suffix == ".py":
        ast.parse(path.read_text(), filename=str(path))
    elif path.suffix in {".yaml", ".yml"}:
        if yaml.safe_load(path.read_text()) is None:
            raise ValueError(f"empty YAML artifact: {path}")
    elif path.suffix == ".json":
        json.loads(path.read_text())


def _inventory_digest(inventory: Sequence[Mapping[str, str]]) -> str:
    canonical = json.dumps(list(inventory), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def audit_repro_manifest(
    manifest_path: Path,
    *,
    repository_root: Path,
    reference_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = yaml.safe_load(manifest_path.read_text())
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != "piu.offline-repro-manifest.v1":
        raise ValueError("unsupported PIU reproducibility manifest")
    resource = manifest.get("resource_contract", {})
    if int(resource.get("local_gpu_memory_mib_max", -1)) != 1500:
        raise ValueError("repro manifest must preserve the 1500 MiB GPU cap")
    if resource.get("local_pi05_checkpoint_load_allowed") is not False:
        raise ValueError("local pi0.5 checkpoint loading must remain forbidden")
    if resource.get("pi05_execution_location") != "identified_external_server_only":
        raise ValueError("pi0.5 execution must remain external-only")
    claim = manifest.get("claim_contract", {})
    if any(
        claim.get(name) is not False
        for name in (
            "synthetic_tests_are_performance_evidence",
            "missing_external_gate_can_be_imputed",
            "old_256_pixel_labels_can_train_main_method",
            "unsealed_results_can_enter_main_table",
            "unqualified_primitive_can_dispatch",
        )
    ):
        raise ValueError("repro manifest weakens a claim firewall")
    required = _files(manifest.get("required_files"), name="required files")
    inventory = []
    errors = []
    for relative in sorted(required):
        path = repository_root / relative
        if not path.is_file():
            errors.append(f"missing required file: {relative}")
            continue
        try:
            _validate_parse(path)
        except (SyntaxError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid required file {relative}: {exc}")
            continue
        inventory.append(
            {"path": relative, "sha256": sha256_file(path), "bytes": str(path.stat().st_size)}
        )
    entrypoints = _files(manifest.get("python_entrypoints"), name="Python entrypoints")
    entrypoint_rows = []
    for relative in entrypoints:
        path = repository_root / relative
        executable = path.is_file() and os.access(path, os.X_OK)
        syntax_valid = False
        if path.is_file():
            try:
                ast.parse(path.read_text(), filename=str(path))
                syntax_valid = True
            except SyntaxError as exc:
                errors.append(f"invalid entrypoint {relative}: {exc}")
        if not executable:
            errors.append(f"entrypoint is not executable: {relative}")
        entrypoint_rows.append(
            {"path": relative, "executable": executable, "syntax_valid": syntax_valid}
        )
    external_rows = []
    for gate in manifest.get("external_empirical_gates", ()):
        relative = str(gate["artifact"])
        path = repository_root / relative
        external_rows.append(
            {
                "id": str(gate["id"]),
                "artifact": relative,
                "status": "PRESENT_UNVERIFIED" if path.is_file() else "PENDING_EXTERNAL",
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    inventory_sha = _inventory_digest(inventory)
    reference_match: bool | None = None
    if reference_report is not None:
        expected = str(reference_report.get("offline_inventory_sha256", ""))
        reference_match = bool(expected) and expected == inventory_sha
        if not reference_match:
            errors.append("offline inventory differs from the reference release")
    offline_ready = not errors and len(inventory) == len(required)
    empirical_ready = offline_ready and all(
        row["status"] == "PRESENT_UNVERIFIED" for row in external_rows
    )
    return {
        "schema_version": "piu.offline-repro-audit.v1",
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "mode": "VERIFY_REFERENCE" if reference_report is not None else "CREATE_RELEASE_LOCK",
        "offline_ready": offline_ready,
        "empirical_ready": empirical_ready,
        "paper_claim_ready": False,
        "paper_claim_ready_reason": (
            "external artifacts are only presence-checked here and require their own "
            "identity, split, authorization, and evaluator validation"
        ),
        "local_gpu_actions_performed": False,
        "offline_inventory": inventory,
        "offline_inventory_sha256": inventory_sha,
        "reference_inventory_match": reference_match,
        "entrypoints": entrypoint_rows,
        "external_empirical_gates": external_rows,
        "errors": errors,
    }
