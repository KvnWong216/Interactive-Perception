from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
IDENTITY_MODULE = ROOT / "scripts/infra/checkpoint_identity.py"
CHECK_MODULE = ROOT / "scripts/infra/check_external_pi05.py"
COMPUTE_CONTRACT = ROOT / "configs/experiments/piu_empirical_compute_contract_v1.yaml"
CHECKPOINT_IDENTITY = (
    ROOT / "results/diagnostics/pi05_libero_checkpoint_identity_v1.json"
)


def load_script(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_identity_is_deterministic_and_content_sensitive(tmp_path) -> None:
    identity = load_script("checkpoint_identity", IDENTITY_MODULE)
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "params/weights").write_bytes(b"abc")
    (checkpoint / "config.json").write_text("{}")
    first = identity.checkpoint_identity(checkpoint, chunk_bytes=2)
    second = identity.checkpoint_identity(checkpoint, chunk_bytes=7)
    assert first == second
    assert first["file_count"] == 2
    assert first["total_bytes"] == 5
    (checkpoint / "params/weights").write_bytes(b"abd")
    assert identity.checkpoint_identity(checkpoint)["sha256"] != first["sha256"]


def test_external_endpoint_metadata_requires_exact_checkpoint_identity() -> None:
    checker = load_script("external_pi05_check", CHECK_MODULE)
    checkpoint = {
        "schema_version": "piu.checkpoint-tree-sha256.v1",
        "sha256": "abc123",
        "file_count": 7,
        "total_bytes": 42,
    }
    identity = {"policy_config": "pi05_libero", "checkpoint": checkpoint}
    metadata = {
        "schema_version": "piu.identified-pi05-server.v1",
        "policy_config": "pi05_libero",
        "environment": "LIBERO",
        "checkpoint": checkpoint,
    }
    checker.validate_metadata(metadata, identity)
    checker.validate_metadata(
        {
            **metadata,
            "capabilities": ["action_chunks", "spatial_prefix_v1"],
            "server_session_id": "a" * 32,
        },
        identity,
    )
    try:
        checker.validate_metadata(
            {**metadata, "server_session_id": "not-a-session"}, identity
        )
    except ValueError as error:
        assert "identity mismatch" in str(error)
    else:
        raise AssertionError("malformed server session identity was accepted")
    mismatched = {**metadata, "policy_config": "pi05_base"}
    try:
        checker.validate_metadata(mismatched, identity)
    except ValueError as error:
        assert "identity mismatch" in str(error)
    else:
        raise AssertionError("mismatched policy identity was accepted")


def test_action_probe_packet_reuses_exact_retained_policy_keyframe() -> None:
    checker = load_script("external_pi05_probe", CHECK_MODULE)
    report = ROOT / "runs/paper_cycle_executor_v2/seed1400/open_butter/report.json"
    packet = checker.packet_from_report(report, "00_before")
    assert packet.image.shape == (224, 224, 3)
    assert packet.wrist_image.shape == (224, 224, 3)
    assert packet.state.shape == (8,)
    assert packet.prompt == "Open the middle layer of the drawer"


def _runtime(*, gpu_name: str = "NVIDIA GeForce RTX 4080 SUPER") -> dict:
    return {
        "schema_version": "piu.identified-policy-runtime.v1",
        "server_process_id": 1234,
        "python_version": "3.11.15",
        "jax_version": "0.5.3",
        "jaxlib_version": "0.5.3",
        "platform": "Linux-test",
        "cuda_visible_devices": "0",
        "xla_python_client_mem_fraction": 0.85,
        "jax_platform_version": "cuda 12030",
        "gpu": {
            "physical_index": 0,
            "visible_index": 0,
            "name": gpu_name,
            "memory_total_mib": 16376,
            "driver_version": "595.84",
        },
    }


def _endpoint(tmp_path: Path, mode: str) -> dict:
    source = tmp_path / "source.json"
    source.write_text("{}\n")
    identity = json.loads(CHECKPOINT_IDENTITY.read_text())
    runtime = _runtime()
    local = mode == "local_identified_server"
    return {
        "schema_version": "piu.external-pi05-check.v2",
        "status": "PASS",
        "endpoint": {"host": "127.0.0.1" if local else "pi05.internal", "port": 8002},
        "identity": {
            "schema_version": "piu.identified-pi05-server.v1",
            "policy_config": "pi05_libero",
            "environment": "LIBERO",
            "checkpoint": identity["checkpoint"],
            "capabilities": ["action_chunks", "spatial_prefix_v1"],
            "server_session_id": "a" * 32,
            "runtime_identity": runtime,
        },
        "checkpoint_identity": {
            "path": str(CHECKPOINT_IDENTITY),
            "sha256": hashlib.sha256(CHECKPOINT_IDENTITY.read_bytes()).hexdigest(),
        },
        "compute_provenance": {
            "schema_version": "piu.endpoint-compute-provenance.v1",
            "compute_contract": {
                "path": str(COMPUTE_CONTRACT),
                "sha256": hashlib.sha256(COMPUTE_CONTRACT.read_bytes()).hexdigest(),
            },
            "semantic_contract": "identified_out_of_process_frozen_policy_endpoint",
            "deployment_mode": mode,
            "local_gpu_used": local,
            "server_out_of_process": True,
            "runtime_identity": runtime,
            "policy_weights_modified": False,
            "quantization_used": False,
            "pruning_used": False,
            "dtype_override_used": False,
            "cpu_offload_used": False,
            "qualification_outcomes_loaded": False,
        },
        "action_probe": {
            "source_report": {
                "path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
            "keyframe": "00_before",
            "shape": [1, 7],
            "finite": True,
            "elapsed_seconds": 0.1,
        },
    }


def _validate(value: dict) -> None:
    from piu.compute_provenance import validate_external_pi05_endpoint_artifact

    validate_external_pi05_endpoint_artifact(
        value,
        checkpoint_identity_path=CHECKPOINT_IDENTITY,
        compute_contract_path=COMPUTE_CONTRACT,
        repository_root=ROOT,
    )


def test_local_identified_server_exact_checkpoint_is_accepted(tmp_path: Path) -> None:
    _validate(_endpoint(tmp_path, "local_identified_server"))


def test_remote_identified_server_exact_checkpoint_is_accepted(tmp_path: Path) -> None:
    _validate(_endpoint(tmp_path, "remote_identified_server"))


def test_local_deployment_cannot_claim_local_gpu_false(tmp_path: Path) -> None:
    value = _endpoint(tmp_path, "local_identified_server")
    value["compute_provenance"]["local_gpu_used"] = False
    try:
        _validate(value)
    except ValueError as error:
        assert "local_gpu_used contradicts" in str(error)
    else:
        raise AssertionError("false local-GPU provenance was accepted")


def test_endpoint_checkpoint_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    value = _endpoint(tmp_path, "remote_identified_server")
    value["identity"]["checkpoint"]["sha256"] = "0" * 64
    try:
        _validate(value)
    except ValueError as error:
        assert "identity differs" in str(error)
    else:
        raise AssertionError("checkpoint mismatch was accepted")


def test_endpoint_deployment_mode_is_required(tmp_path: Path) -> None:
    value = _endpoint(tmp_path, "remote_identified_server")
    del value["compute_provenance"]["deployment_mode"]
    try:
        _validate(value)
    except ValueError as error:
        assert "fields are not closed" in str(error)
    else:
        raise AssertionError("endpoint without a deployment mode was accepted")


def test_empirical_amendment_preserves_historical_1500_mib_contract() -> None:
    import yaml
    from piu.compute_provenance import load_empirical_compute_contract

    contract = load_empirical_compute_contract(COMPUTE_CONTRACT, repository_root=ROOT)
    retained = yaml.safe_load(
        (ROOT / "configs/experiments/piu_offline_repro_v1.yaml").read_text()
    )
    assert contract["offline_replay_contract"]["local_gpu_memory_mib_max"] == 1500
    assert retained["resource_contract"]["local_gpu_memory_mib_max"] == 1500
    assert retained["resource_contract"]["local_pi05_checkpoint_load_allowed"] is False
    assert contract["deployment_contracts"]["local_identified_server"]["local_gpu_used"] is True
    assert hashlib.sha256(
        (ROOT / "configs/experiments/piu_offline_repro_v1.yaml").read_bytes()
    ).hexdigest() == "3c5575f1b0894777851ca420126ac6447e9f19ca01963300282540d5b104f111"
    assert hashlib.sha256(
        (ROOT / "results/diagnostics/piu_offline_repro_preflight_v1.json").read_bytes()
    ).hexdigest() == "18f47a0dd6a64c5b2aefb4559ea47d857c0d522b9c43a7e53912bcd13a3b1c72"
