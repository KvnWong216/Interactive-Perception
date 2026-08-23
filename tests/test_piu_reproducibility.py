from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from piu.reproducibility import audit_repro_manifest, validate_repro_lock

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/experiments/piu_offline_repro_v3.yaml"


def test_repository_offline_release_is_ready_but_empirical_gates_are_not() -> None:
    report = audit_repro_manifest(MANIFEST, repository_root=ROOT)
    assert report["offline_ready"] is True
    assert report["empirical_ready"] is False
    assert report["paper_claim_ready"] is False
    assert report["local_gpu_actions_performed"] is False
    assert all(row["syntax_valid"] for row in report["entrypoints"])


def test_reference_inventory_detects_source_drift(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n")
    entrypoint = tmp_path / "run.py"
    entrypoint.write_text("#!/usr/bin/env python3\n")
    entrypoint.chmod(0o755)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.offline-repro-manifest.v1",
                "resource_contract": {
                    "local_gpu_memory_mib_max": 1500,
                    "local_pi05_checkpoint_load_allowed": False,
                    "pi05_execution_location": "identified_external_server_only",
                },
                "required_files": ["module.py"],
                "python_entrypoints": ["run.py"],
                "external_empirical_gates": [],
                "claim_contract": {
                    "synthetic_tests_are_performance_evidence": False,
                    "missing_external_gate_can_be_imputed": False,
                    "old_256_pixel_labels_can_train_main_method": False,
                    "unsealed_results_can_enter_main_table": False,
                    "unqualified_primitive_can_dispatch": False,
                },
            }
        )
    )
    first = audit_repro_manifest(manifest, repository_root=tmp_path)
    source.write_text("VALUE = 2\n")
    second = audit_repro_manifest(
        manifest, repository_root=tmp_path, reference_report=first
    )
    assert second["offline_ready"] is False
    assert second["reference_inventory_match"] is False


def test_release_lock_rejects_current_file_drift(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n")
    entrypoint = tmp_path / "run.py"
    entrypoint.write_text("#!/usr/bin/env python3\n")
    entrypoint.chmod(0o755)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.offline-repro-manifest.v1",
                "resource_contract": {
                    "local_gpu_memory_mib_max": 1500,
                    "local_pi05_checkpoint_load_allowed": False,
                    "pi05_execution_location": "identified_external_server_only",
                },
                "required_files": ["module.py"],
                "python_entrypoints": ["run.py"],
                "external_empirical_gates": [],
                "claim_contract": {
                    "synthetic_tests_are_performance_evidence": False,
                    "missing_external_gate_can_be_imputed": False,
                    "old_256_pixel_labels_can_train_main_method": False,
                    "unsealed_results_can_enter_main_table": False,
                    "unqualified_primitive_can_dispatch": False,
                },
            }
        )
    )
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(audit_repro_manifest(manifest, repository_root=tmp_path))
    )
    validate_repro_lock(lock, manifest_path=manifest, repository_root=tmp_path)
    source.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="differs from lock"):
        validate_repro_lock(lock, manifest_path=manifest, repository_root=tmp_path)


def test_release_audit_rejects_an_untracked_local_import(tmp_path: Path) -> None:
    package = tmp_path / "src/piu"
    package.mkdir(parents=True)
    main = package / "main.py"
    main.write_text("from piu.helper import VALUE\n")
    helper = package / "helper.py"
    helper.write_text("VALUE = 1\n")
    entrypoint = tmp_path / "run.py"
    entrypoint.write_text("#!/usr/bin/env python3\n")
    entrypoint.chmod(0o755)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.offline-repro-manifest.v1",
                "resource_contract": {
                    "local_gpu_memory_mib_max": 1500,
                    "local_pi05_checkpoint_load_allowed": False,
                    "pi05_execution_location": "identified_external_server_only",
                },
                "required_files": ["src/piu/main.py"],
                "python_entrypoints": ["run.py"],
                "external_empirical_gates": [],
                "claim_contract": {
                    "synthetic_tests_are_performance_evidence": False,
                    "missing_external_gate_can_be_imputed": False,
                    "old_256_pixel_labels_can_train_main_method": False,
                    "unsealed_results_can_enter_main_table": False,
                    "unqualified_primitive_can_dispatch": False,
                },
            }
        )
    )
    report = audit_repro_manifest(manifest, repository_root=tmp_path)
    assert report["offline_ready"] is False
    assert any(
        "src/piu/main.py -> src/piu/helper.py" in error
        for error in report["errors"]
    )
