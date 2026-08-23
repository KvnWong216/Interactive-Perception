"""Prospective compute provenance for an identified frozen pi0.5 endpoint."""

from __future__ import annotations

import hashlib
import ipaddress
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .policy_identity import load_checkpoint_identity, validate_server_metadata


COMPUTE_CONTRACT_SCHEMA = "piu.empirical-compute-contract.v1"
ENDPOINT_CHECK_SCHEMA = "piu.external-pi05-check.v2"
RUNTIME_SCHEMA = "piu.identified-policy-runtime.v1"
SEMANTIC_CONTRACT = "identified_out_of_process_frozen_policy_endpoint"
DEPLOYMENT_MODES = frozenset(
    {"local_identified_server", "remote_identified_server"}
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path, *, repository_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repository_root / value


def _reference(
    value: Any,
    *,
    expected_path: Path,
    repository_root: Path,
    name: str,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a hash-bound artifact reference")
    observed_path = _resolve(
        str(value.get("path", "")), repository_root=repository_root
    )
    if (
        observed_path.resolve() != expected_path.resolve()
        or value.get("sha256") != sha256(expected_path)
    ):
        raise ValueError(f"{name} differs from its frozen content")


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def load_empirical_compute_contract(
    path: Path, *, repository_root: Path
) -> dict[str, Any]:
    """Load and validate the prospective contract without reading outcomes."""

    value = yaml.safe_load(path.read_text())
    expected_fields = {
        "schema_version",
        "id",
        "status",
        "semantic_contract",
        "policy_contract",
        "offline_replay_contract",
        "deployment_contracts",
        "qualification_outcomes_loaded",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("empirical compute-contract fields are not closed")
    if value.get("schema_version") != COMPUTE_CONTRACT_SCHEMA:
        raise ValueError("unsupported empirical compute contract")
    if value.get("status") != "FROZEN_BEFORE_EMPIRICAL_POLICY_EXECUTION":
        raise ValueError("empirical compute contract is not prospectively frozen")
    if value.get("semantic_contract") != SEMANTIC_CONTRACT:
        raise ValueError("empirical compute semantic contract changed")
    if value.get("qualification_outcomes_loaded") is not False:
        raise ValueError("compute amendment may not load qualification outcomes")

    policy = value.get("policy_contract")
    if not isinstance(policy, Mapping) or set(policy) != {
        "policy_config",
        "environment",
        "checkpoint_identity",
        "weights_frozen",
        "quantization_allowed",
        "pruning_allowed",
        "dtype_override_allowed",
        "cpu_offload_allowed",
        "public_observation_contract_preserved",
        "simulator_privileged_inputs_allowed",
        "action_interface",
        "finite_action_probe_required",
        "endpoint_artifact_immutable",
    }:
        raise ValueError("empirical policy-contract fields are not closed")
    if (
        policy.get("policy_config") != "pi05_libero"
        or policy.get("environment") != "LIBERO"
        or policy.get("weights_frozen") is not True
        or any(
            policy.get(name) is not False
            for name in (
                "quantization_allowed",
                "pruning_allowed",
                "dtype_override_allowed",
                "cpu_offload_allowed",
                "simulator_privileged_inputs_allowed",
            )
        )
        or policy.get("public_observation_contract_preserved") is not True
        or policy.get("action_interface") != "action_chunks"
        or policy.get("finite_action_probe_required") is not True
        or policy.get("endpoint_artifact_immutable") is not True
    ):
        raise ValueError("empirical policy contract weakens the frozen executor")
    identity_path = _resolve(
        str(policy.get("checkpoint_identity", "")), repository_root=repository_root
    )
    identity = load_checkpoint_identity(identity_path)
    if identity.get("policy_config") != policy.get("policy_config"):
        raise ValueError("compute contract policy config differs from identity")

    offline = value.get("offline_replay_contract")
    if not isinstance(offline, Mapping) or set(offline) != {
        "manifest",
        "historical_contract_retained",
        "local_gpu_memory_mib_max",
        "local_model_device",
        "local_pi05_checkpoint_load_allowed",
        "local_simulator_gpu_render_allowed",
        "pi05_execution_location",
        "local_gpu_used",
    }:
        raise ValueError("offline replay contract fields are not closed")
    offline_path = _resolve(
        str(offline.get("manifest", "")), repository_root=repository_root
    )
    offline_manifest = yaml.safe_load(offline_path.read_text())
    retained = offline_manifest.get("resource_contract", {})
    required_offline = {
        "local_gpu_memory_mib_max": 1500,
        "local_model_device": "cpu",
        "local_pi05_checkpoint_load_allowed": False,
        "local_simulator_gpu_render_allowed": False,
        "pi05_execution_location": "identified_external_server_only",
    }
    if (
        offline.get("historical_contract_retained") is not True
        or offline.get("local_gpu_used") is not False
        or any(offline.get(name) != expected for name, expected in required_offline.items())
        or any(retained.get(name) != expected for name, expected in required_offline.items())
    ):
        raise ValueError("offline retained 1500 MiB contract was rewritten")

    deployments = value.get("deployment_contracts")
    if not isinstance(deployments, Mapping) or set(deployments) != DEPLOYMENT_MODES:
        raise ValueError("compute contract must define local and remote deployments")
    local = deployments["local_identified_server"]
    remote = deployments["remote_identified_server"]
    if not isinstance(local, Mapping) or set(local) != {
        "local_gpu_used",
        "server_process",
        "endpoint_host",
        "gpu_physical_index",
        "gpu_name",
        "gpu_memory_total_mib",
        "xla_python_client_mem_fraction",
    }:
        raise ValueError("local identified deployment fields are not closed")
    fraction = local.get("xla_python_client_mem_fraction")
    if (
        local.get("local_gpu_used") is not True
        or local.get("server_process") != "independent"
        or local.get("endpoint_host") != "127.0.0.1"
        or local.get("gpu_physical_index") != 0
        or local.get("gpu_name") != "NVIDIA GeForce RTX 4080 SUPER"
        or local.get("gpu_memory_total_mib") != 16376
        or not isinstance(fraction, (int, float))
        or isinstance(fraction, bool)
        or not math.isclose(float(fraction), 0.85, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("local identified deployment differs from authorization")
    if not isinstance(remote, Mapping) or set(remote) != {
        "local_gpu_used",
        "server_process",
        "tunneled_loopback_allowed",
    }:
        raise ValueError("remote identified deployment fields are not closed")
    if (
        remote.get("local_gpu_used") is not False
        or remote.get("server_process") != "independent"
        or remote.get("tunneled_loopback_allowed") is not True
    ):
        raise ValueError("remote identified deployment contract is malformed")
    return dict(value)


def validate_runtime_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "server_process_id",
        "python_version",
        "jax_version",
        "jaxlib_version",
        "platform",
        "cuda_visible_devices",
        "xla_python_client_mem_fraction",
        "jax_platform_version",
        "gpu",
    }:
        raise ValueError("identified runtime fields are not closed")
    if value.get("schema_version") != RUNTIME_SCHEMA:
        raise ValueError("identified runtime schema is unsupported")
    if (
        not isinstance(value.get("server_process_id"), int)
        or isinstance(value.get("server_process_id"), bool)
        or value["server_process_id"] < 1
    ):
        raise ValueError("identified runtime lacks a server process ID")
    for name in (
        "python_version",
        "jax_version",
        "jaxlib_version",
        "platform",
        "cuda_visible_devices",
        "jax_platform_version",
    ):
        if not " ".join(str(value.get(name, "")).split()):
            raise ValueError(f"identified runtime lacks {name}")
    fraction = value.get("xla_python_client_mem_fraction")
    if (
        not isinstance(fraction, (int, float))
        or isinstance(fraction, bool)
        or not 0.0 < float(fraction) <= 1.0
    ):
        raise ValueError("identified runtime has an invalid XLA fraction")
    gpu = value.get("gpu")
    if not isinstance(gpu, Mapping) or set(gpu) != {
        "physical_index",
        "visible_index",
        "name",
        "memory_total_mib",
        "driver_version",
    }:
        raise ValueError("identified runtime GPU fields are not closed")
    if any(
        not isinstance(gpu.get(name), int)
        or isinstance(gpu.get(name), bool)
        or gpu[name] < 0
        for name in ("physical_index", "visible_index")
    ) or (
        not isinstance(gpu.get("memory_total_mib"), int)
        or isinstance(gpu.get("memory_total_mib"), bool)
        or gpu["memory_total_mib"] < 1
    ):
        raise ValueError("identified runtime GPU numeric identity is invalid")
    for name in ("name", "driver_version"):
        if not " ".join(str(gpu.get(name, "")).split()):
            raise ValueError(f"identified runtime GPU lacks {name}")
    return dict(value)


def validate_external_pi05_endpoint_artifact(
    value: Mapping[str, Any],
    *,
    checkpoint_identity_path: Path,
    compute_contract_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Validate a v2 endpoint probe including deployment and runtime truth."""

    if set(value) != {
        "schema_version",
        "status",
        "endpoint",
        "identity",
        "checkpoint_identity",
        "compute_provenance",
        "action_probe",
    }:
        raise ValueError("endpoint check fields are not closed")
    if value.get("schema_version") != ENDPOINT_CHECK_SCHEMA:
        raise ValueError("unsupported endpoint-check schema")
    if value.get("status") != "PASS":
        raise ValueError("endpoint check did not pass")
    identity = load_checkpoint_identity(checkpoint_identity_path)
    _reference(
        value.get("checkpoint_identity"),
        expected_path=checkpoint_identity_path,
        repository_root=repository_root,
        name="endpoint checkpoint identity",
    )
    metadata = value.get("identity")
    if not isinstance(metadata, Mapping):
        raise TypeError("endpoint check lacks server metadata")
    validate_server_metadata(metadata, identity)
    runtime = validate_runtime_identity(metadata.get("runtime_identity"))

    endpoint = value.get("endpoint")
    if not isinstance(endpoint, Mapping) or set(endpoint) != {"host", "port"}:
        raise ValueError("endpoint host/port fields are not closed")
    host = " ".join(str(endpoint.get("host", "")).split())
    port = endpoint.get("port")
    if (
        not host
        or not isinstance(port, int)
        or isinstance(port, bool)
        or not 1 <= port <= 65535
    ):
        raise ValueError("endpoint host/port are invalid")

    contract = load_empirical_compute_contract(
        compute_contract_path, repository_root=repository_root
    )
    provenance = value.get("compute_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "schema_version",
        "compute_contract",
        "semantic_contract",
        "deployment_mode",
        "local_gpu_used",
        "server_out_of_process",
        "runtime_identity",
        "policy_weights_modified",
        "quantization_used",
        "pruning_used",
        "dtype_override_used",
        "cpu_offload_used",
        "qualification_outcomes_loaded",
    }:
        raise ValueError("endpoint compute-provenance fields are not closed")
    if provenance.get("schema_version") != "piu.endpoint-compute-provenance.v1":
        raise ValueError("endpoint compute-provenance schema is unsupported")
    _reference(
        provenance.get("compute_contract"),
        expected_path=compute_contract_path,
        repository_root=repository_root,
        name="endpoint compute contract",
    )
    mode = provenance.get("deployment_mode")
    if mode not in DEPLOYMENT_MODES:
        raise ValueError("endpoint deployment mode is missing or unsupported")
    deployment = contract["deployment_contracts"][mode]
    if provenance.get("local_gpu_used") is not deployment["local_gpu_used"]:
        raise ValueError("endpoint local_gpu_used contradicts deployment mode")
    if (
        provenance.get("semantic_contract") != SEMANTIC_CONTRACT
        or provenance.get("server_out_of_process") is not True
        or provenance.get("runtime_identity") != runtime
        or any(
            provenance.get(name) is not False
            for name in (
                "policy_weights_modified",
                "quantization_used",
                "pruning_used",
                "dtype_override_used",
                "cpu_offload_used",
                "qualification_outcomes_loaded",
            )
        )
    ):
        raise ValueError("endpoint compute provenance weakens the frozen contract")
    if mode == "local_identified_server":
        local = contract["deployment_contracts"][mode]
        gpu = runtime["gpu"]
        if not _is_loopback(host):
            raise ValueError("local identified server must use a loopback endpoint")
        if (
            gpu["physical_index"] != local["gpu_physical_index"]
            or gpu["name"] != local["gpu_name"]
            or gpu["memory_total_mib"] != local["gpu_memory_total_mib"]
            or not math.isclose(
                float(runtime["xla_python_client_mem_fraction"]),
                float(local["xla_python_client_mem_fraction"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("local runtime differs from the authorized GPU contract")

    probe = value.get("action_probe")
    if not isinstance(probe, Mapping) or set(probe) != {
        "source_report",
        "keyframe",
        "shape",
        "finite",
        "elapsed_seconds",
    }:
        raise ValueError("endpoint action-probe fields are not closed")
    if probe.get("finite") is not True:
        raise ValueError("endpoint check lacks a finite action probe")
    source = probe.get("source_report")
    if not isinstance(source, Mapping):
        raise TypeError("endpoint action probe lacks a hash-bound source report")
    source_path = _resolve(
        str(source.get("path", "")), repository_root=repository_root
    )
    if not source_path.is_file() or source.get("sha256") != sha256(source_path):
        raise ValueError("endpoint action-probe source differs from its hash")
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
    return dict(value)
