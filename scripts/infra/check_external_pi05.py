#!/usr/bin/env python3
"""Validate an identified external pi0.5 endpoint and optionally sample once."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from interactive_perception.policy_client import (
    ObservationPacket,
    OpenPiWebsocketPolicy,
)
from piu.policy_identity import validate_server_metadata
from piu.compute_provenance import (
    DEPLOYMENT_MODES,
    SEMANTIC_CONTRACT,
    load_empirical_compute_contract,
    validate_external_pi05_endpoint_artifact,
)

SERVER_SCHEMA = "piu.identified-pi05-server.v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def wait_for_endpoint(host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            connection = http.client.HTTPConnection(host, port, timeout=1.0)
            connection.request("GET", "/healthz")
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status == 200:
                return
        except (OSError, http.client.HTTPException):
            time.sleep(0.5)
    raise TimeoutError(f"pi0.5 endpoint did not become ready at {host}:{port}")


def validate_metadata(
    metadata: dict[str, Any], expected_identity: dict[str, Any]
) -> None:
    try:
        validate_server_metadata(metadata, expected_identity)
    except ValueError as error:
        raise ValueError(
            "external policy identity mismatch:\n"
            f"received={json.dumps(metadata, sort_keys=True)}\n"
            f"expected={json.dumps(expected_identity, sort_keys=True)}"
        ) from error


def packet_from_report(path: Path, keyframe_name: str) -> ObservationPacket:
    report = json.loads(path.read_text())
    rows = [
        row for row in report["controller"]["keyframes"] if row["name"] == keyframe_name
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one {keyframe_name!r} keyframe in {path}")
    row = rows[0]
    return ObservationPacket(
        image=np.asarray(Image.open(ROOT / row["image_paths"]["agentview"])),
        wrist_image=np.asarray(Image.open(ROOT / row["image_paths"]["wrist"])),
        state=np.asarray(row["public_robot_state"], dtype=np.float64),
        prompt=str(report["prompt"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--api-key")
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument(
        "--compute-contract",
        type=Path,
        default=ROOT / "configs/experiments/piu_empirical_compute_contract_v1.yaml",
    )
    parser.add_argument(
        "--deployment-mode", choices=sorted(DEPLOYMENT_MODES), required=True
    )
    parser.add_argument("--probe-report", type=Path)
    parser.add_argument("--probe-keyframe", default="00_before")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    identity_path = (
        args.identity if args.identity.is_absolute() else ROOT / args.identity
    )
    identity = json.loads(identity_path.read_text())
    compute_contract_path = (
        args.compute_contract
        if args.compute_contract.is_absolute()
        else ROOT / args.compute_contract
    )
    compute_contract = load_empirical_compute_contract(
        compute_contract_path, repository_root=ROOT
    )
    wait_for_endpoint(args.host, args.port, args.timeout)
    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port, api_key=args.api_key)
    metadata = policy.server_metadata
    validate_metadata(metadata, identity)
    result: dict[str, Any] = {
        "schema_version": "piu.external-pi05-check.v2",
        "status": "PASS",
        "endpoint": {"host": args.host, "port": args.port},
        "identity": metadata,
        "checkpoint_identity": {
            "path": portable(identity_path),
            "sha256": sha256(identity_path),
        },
        "compute_provenance": {
            "schema_version": "piu.endpoint-compute-provenance.v1",
            "compute_contract": {
                "path": portable(compute_contract_path),
                "sha256": sha256(compute_contract_path),
            },
            "semantic_contract": SEMANTIC_CONTRACT,
            "deployment_mode": args.deployment_mode,
            "local_gpu_used": compute_contract["deployment_contracts"][
                args.deployment_mode
            ]["local_gpu_used"],
            "server_out_of_process": True,
            "runtime_identity": metadata.get("runtime_identity"),
            "policy_weights_modified": False,
            "quantization_used": False,
            "pruning_used": False,
            "dtype_override_used": False,
            "cpu_offload_used": False,
            "qualification_outcomes_loaded": False,
        },
        "action_probe": None,
    }
    if args.probe_report is not None:
        report_path = (
            args.probe_report
            if args.probe_report.is_absolute()
            else ROOT / args.probe_report
        )
        started = time.monotonic()
        action = policy.sample_chunks(
            packet_from_report(report_path, args.probe_keyframe), 1
        )[0]
        result["action_probe"] = {
            "source_report": {
                "path": portable(report_path),
                "sha256": sha256(report_path),
            },
            "keyframe": args.probe_keyframe,
            "shape": list(action.shape),
            "finite": bool(np.isfinite(action).all()),
            "elapsed_seconds": time.monotonic() - started,
        }
        if result["action_probe"]["finite"] is not True:
            raise ValueError("identified endpoint returned a non-finite action")
    if args.output is not None and args.probe_report is None:
        raise ValueError("canonical endpoint checks require one finite action probe")
    if args.probe_report is not None:
        validate_external_pi05_endpoint_artifact(
            result,
            checkpoint_identity_path=identity_path,
            compute_contract_path=compute_contract_path,
            repository_root=ROOT,
        )
    output = None
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        if output.exists():
            raise FileExistsError(f"endpoint check is immutable: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result, "output": str(output) if output else None}, indent=2))


if __name__ == "__main__":
    main()
