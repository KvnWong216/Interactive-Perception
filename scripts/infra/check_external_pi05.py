#!/usr/bin/env python3
"""Validate an identified external pi0.5 endpoint and optionally sample once."""

from __future__ import annotations

import argparse
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

SERVER_SCHEMA = "piu.identified-pi05-server.v1"


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
    wait_for_endpoint(args.host, args.port, args.timeout)
    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port, api_key=args.api_key)
    metadata = policy.server_metadata
    validate_metadata(metadata, identity)
    result: dict[str, Any] = {
        "schema_version": "piu.external-pi05-check.v1",
        "status": "PASS",
        "endpoint": {"host": args.host, "port": args.port},
        "identity": metadata,
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
            "source_report": str(report_path.resolve().relative_to(ROOT)),
            "keyframe": args.probe_keyframe,
            "shape": list(action.shape),
            "finite": bool(np.isfinite(action).all()),
            "elapsed_seconds": time.monotonic() - started,
        }
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
