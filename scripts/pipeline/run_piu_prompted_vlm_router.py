#!/usr/bin/env python3
"""Run one B1 public-input prompted-VLM routing decision via an external service."""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.calibrated_controller import (
    INFORMATION_PRIMITIVES,
    TASK_PRIMITIVES,
    DecisionKind,
)
from piu.contracts import load_public_transitions
from piu.executor_bridge import serialize_pi05_subtask


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def public_observation_packet(value: dict[str, Any]) -> dict[str, Any]:
    images = {}
    for camera, artifact in value["images"].items():
        path = resolve(Path(artifact["path"]))
        if not path.is_file() or sha256(path) != artifact.get("sha256"):
            raise ValueError("B1 public image differs from its file hash")
        pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        pixel_digest = hashlib.sha256(
            np.ascontiguousarray(pixels).tobytes()
        ).hexdigest()
        declared = artifact.get("pixel_sha256")
        if declared is not None and declared != pixel_digest:
            raise ValueError("B1 public image differs from its pixel hash")
        images[str(camera)] = {
            "mime_type": "image/png",
            "file_sha256": sha256(path),
            "pixel_sha256": pixel_digest,
            "base64": base64.b64encode(path.read_bytes()).decode(),
        }
    return {
        "images": images,
        "public_robot_state": [float(item) for item in value["public_robot_state"]],
    }


def request_json(
    *,
    host: str,
    port: int,
    method: str,
    path: str,
    body: dict[str, Any] | None,
    timeout: float,
    api_key: str | None,
) -> dict[str, Any]:
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    headers = {"Accept": "application/json"}
    payload = None
    if body is not None:
        payload = json.dumps(body, separators=(",", ":")).encode()
        headers["Content-Type"] = "application/json"
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        raw = response.read()
    finally:
        connection.close()
    if response.status != 200:
        raise RuntimeError(f"external prompted-VLM service returned HTTP {response.status}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("external prompted-VLM response must be a JSON object")
    return value


def decision_kind(primitive: str) -> DecisionKind:
    if primitive in INFORMATION_PRIMITIVES:
        return DecisionKind.INTERACT
    if primitive in TASK_PRIMITIVES:
        return DecisionKind.EXECUTE
    if primitive == "STOP":
        return DecisionKind.STOP
    if primitive == "REPORT_NOT_FOUND":
        return DecisionKind.REPORT_NOT_FOUND
    return DecisionKind.ABSTAIN


def verify_sealed(
    path: Path, *, public: Path, identity: Path, output: Path
) -> None:
    value = json.loads(path.read_text())
    if value.get("schema_version") != "piu.prompted-vlm-sealed-authorization.v1":
        raise ValueError("unsupported prompted-VLM sealed authorization")
    expected = {
        "public_transition_sha256": sha256(public),
        "router_identity_sha256": sha256(identity),
        "method_id": "B1",
        "single_use_output": portable(output),
    }
    for name, required in expected.items():
        if value.get(name) != required:
            raise ValueError(f"prompted-VLM authorization differs at {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-transition", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--router-identity", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--api-key")
    parser.add_argument(
        "--expected-split", choices=("development", "sealed_test"), required=True
    )
    parser.add_argument("--sealed-authorization", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "public_transition",
        "router_identity",
        "sealed_authorization",
        "output",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, resolve(value))
    if args.timeout <= 0:
        raise ValueError("prompted-VLM timeout must be positive")
    if args.output.exists():
        raise FileExistsError("prompted-VLM controller reports are immutable")
    identity = json.loads(args.router_identity.read_text())
    if identity.get("schema_version") != "piu.prompted-vlm-router-identity.v1":
        raise ValueError("unsupported prompted-VLM router identity")
    expected_metadata = identity.get("server_metadata")
    if not isinstance(expected_metadata, dict) or not expected_metadata:
        raise ValueError("prompted-VLM identity requires exact server metadata")
    rows = [
        row
        for row in load_public_transitions(args.public_transition)
        if row.sample_id == args.sample_id
    ]
    if len(rows) != 1:
        raise ValueError("B1 sample ID must select one public transition")
    public = rows[0]
    if public.split.value != args.expected_split:
        raise ValueError("B1 public split differs")
    if args.expected_split == "sealed_test":
        if args.sealed_authorization is None:
            raise ValueError("sealed B1 inference requires authorization")
        verify_sealed(
            args.sealed_authorization,
            public=args.public_transition,
            identity=args.router_identity,
            output=args.output,
        )
    elif args.sealed_authorization is not None:
        raise ValueError("development B1 inference cannot use sealed authorization")
    candidates = [dict(row) for row in public.candidate_actions]
    request = {
        "schema_version": "piu.prompted-vlm-router-request.v1",
        "prompt": public.prompt,
        "observation": public_observation_packet(
            dict(public.observations["post_interaction"])
        ),
        "public_action_history": dict(public.public_action_history),
        "candidate_actions": candidates,
        "allowed_candidate_ids": [str(row["candidate_id"]) for row in candidates],
        "response_contract": {
            "return_only": ["selected_candidate_id"],
            "chain_of_thought": "forbidden",
            "confidence_threshold": None,
        },
    }
    metadata = request_json(
        host=args.host,
        port=args.port,
        method="GET",
        path="/metadata",
        body=None,
        timeout=args.timeout,
        api_key=args.api_key,
    )
    if metadata != expected_metadata:
        raise ValueError("external prompted-VLM server identity mismatch")
    request_digest = canonical_sha256(request)
    response = request_json(
        host=args.host,
        port=args.port,
        method="POST",
        path="/route",
        body=request,
        timeout=args.timeout,
        api_key=args.api_key,
    )
    if response.get("schema_version") != "piu.prompted-vlm-router-response.v1":
        raise ValueError("unsupported prompted-VLM response schema")
    if response.get("request_sha256") != request_digest:
        raise ValueError("prompted-VLM response is bound to another request")
    if set(response) != {
        "schema_version",
        "request_sha256",
        "selected_candidate_id",
    }:
        raise ValueError("prompted-VLM response contains undeclared fields")
    selected_id = " ".join(str(response.get("selected_candidate_id", "")).split())
    selected_rows = [
        row for row in candidates if str(row["candidate_id"]) == selected_id
    ]
    if len(selected_rows) == 1:
        selected = selected_rows[0]
        primitive = str(selected["primitive"]).upper()
        kind = decision_kind(primitive)
        if kind is DecisionKind.ABSTAIN:
            selected = None
            selected_id = ""
            primitive = ""
            reason = "prompted VLM selected a candidate with unknown primitive"
        else:
            reason = "external frozen prompted VLM selected one public candidate"
    else:
        selected = None
        selected_id = ""
        primitive = ""
        kind = DecisionKind.ABSTAIN
        reason = "prompted VLM returned no unique registered candidate"
    structured = (
        None
        if selected is None
        else serialize_pi05_subtask(selected, spatial_references=())
    )
    result = {
        "schema_version": "piu.prompted-vlm-router-report.v1",
        "claim_scope": "PUBLIC_PROMPTED_VLM_BASELINE_DECISION_NOT_RESULT",
        "method_id": "B1",
        "split": public.split.value,
        "initial_state_groups": [public.initial_state_group],
        "evaluator_labels_loaded": False,
        "online_oracle_inputs": [],
        "decisions": [
            {
                "sample_id": public.sample_id,
                "initial_state_group": public.initial_state_group,
                "decision_kind": kind.value,
                "selected_candidate_id": selected_id or None,
                "selected_candidate_primitive": primitive or None,
                "selected_candidate": selected,
                "public_candidates": candidates,
                "structured_pi05_subtask": structured,
                "reason": reason,
            }
        ],
        "external_router": {
            "endpoint": f"{args.host}:{args.port}",
            "identity": {
                "path": portable(args.router_identity),
                "sha256": sha256(args.router_identity),
            },
            "server_metadata": metadata,
            "request_sha256": request_digest,
            "response_sha256": canonical_sha256(response),
        },
        "manual_confidence_threshold": None,
        "paper_method_claim_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
