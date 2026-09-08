"""Isolated Qwen2.5-VL proposal service for the legacy LIBERO process.

Only public task text, completed public action history, and one public RGB
image cross this boundary. Feature extraction is performed offline in the
Qwen environment; this service is used online only for the single post-OPEN
DIRECT proposal. It never receives simulator state, evaluator fields, or
outcomes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .contracts import Primitive, canonical_sha256
from .proposals import (
    METHOD_V1_ENABLED_PRIMITIVES,
    METHOD_V1_MAX_CANDIDATES,
    METHOD_V1_MAX_PER_PRIMITIVE,
    proposal_request_text,
    qwen_proposal_prompt_contract,
)
from .qwen_provider import (
    Qwen25VLRuntime,
    QwenProposalGeneration,
    QwenProviderIdentity,
)
from .rgb import canonical_rgb_sha256, decode_rgb_png, encode_rgb_png

QWEN_PROPOSAL_HTTP_SCHEMA = "method-v1-qwen-proposal-http-v2"


class QwenProposalServiceError(RuntimeError):
    pass


class QwenProposalTransportError(QwenProposalServiceError):
    pass


class QwenProposalProtocolError(QwenProposalServiceError):
    pass


class QwenProposalHTTPClient:
    """ProposalRuntime implementation backed by a pinned Qwen service."""

    def __init__(
        self,
        base_url: str,
        *,
        expected_provider_id: str,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.base_url = str(base_url).strip().rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        self.expected_provider_id = " ".join(str(expected_provider_id).split())
        if not self.expected_provider_id:
            raise ValueError("expected_provider_id must be non-empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)

    @property
    def provider_id(self) -> str:
        return self.expected_provider_id

    def _exchange(
        self, path: str, *, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        data = None
        headers: dict[str, str] = {}
        method = "GET"
        if payload is not None:
            data = json.dumps(
                dict(payload),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            headers["Content-Type"] = "application/json"
            method = "POST"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                raw = response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            raise QwenProposalTransportError(str(error)) from error
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise QwenProposalProtocolError(
                "Qwen service returned invalid JSON"
            ) from error
        if not isinstance(value, dict):
            raise QwenProposalProtocolError("Qwen service response must be an object")
        if value.get("status") == "error":
            raise QwenProposalServiceError(str(value.get("message", "Qwen failure")))
        return value

    def health(self) -> dict[str, Any]:
        result = self._exchange("/health")
        if (
            result.get("status") != "ok"
            or result.get("schema_version") != QWEN_PROPOSAL_HTTP_SCHEMA
            or result.get("provider_id") != self.expected_provider_id
        ):
            raise QwenProposalProtocolError("Qwen service identity mismatch")
        return result

    def ready(self) -> dict[str, Any]:
        """Force model loading and verify readiness before outcome collection."""

        result = self._exchange("/ready")
        if (
            result.get("status") != "ok"
            or result.get("schema_version") != QWEN_PROPOSAL_HTTP_SCHEMA
            or result.get("provider_id") != self.expected_provider_id
            or result.get("model_loaded") is not True
        ):
            raise QwenProposalProtocolError("Qwen service is not model-ready")
        return result

    def generate_proposal_json(
        self,
        *,
        task_prompt: str,
        public_history_text: str,
        image: Any,
        camera_label: str,
        frame_id: str,
        enabled_primitives: Sequence[Primitive] = METHOD_V1_ENABLED_PRIMITIVES,
        max_candidates: int = METHOD_V1_MAX_CANDIDATES,
        max_per_primitive: int = METHOD_V1_MAX_PER_PRIMITIVE,
    ) -> QwenProposalGeneration:
        self.health()
        contract = qwen_proposal_prompt_contract(
            enabled_primitives=enabled_primitives,
            max_candidates=max_candidates,
            max_per_primitive=max_per_primitive,
        )
        image_digest = canonical_rgb_sha256(image)
        public_request = {
            "schema_version": QWEN_PROPOSAL_HTTP_SCHEMA,
            "provider_id": self.expected_provider_id,
            "task_prompt": str(task_prompt),
            "public_history_text": str(public_history_text),
            "camera_label": str(camera_label),
            "frame_id": str(frame_id),
            "image_sha256": image_digest,
            "enabled_primitives": [item.value for item in enabled_primitives],
            "max_candidates": max_candidates,
            "max_per_primitive": max_per_primitive,
            "prompt_contract_sha256": contract.fingerprint,
        }
        request_id = canonical_sha256(public_request)
        response = self._exchange(
            "/propose",
            payload={
                **public_request,
                "request_id": request_id,
                "image_png_base64": base64.b64encode(encode_rgb_png(image)).decode(
                    "ascii"
                ),
            },
        )
        if (
            response.get("request_id") != request_id
            or response.get("provider_id") != self.expected_provider_id
            or response.get("prompt_contract_sha256") != contract.fingerprint
        ):
            raise QwenProposalProtocolError("proposal response identity mismatch")
        try:
            result = QwenProposalGeneration(
                raw_json=str(response["raw_json"]),
                processed_width=int(response["processed_width"]),
                processed_height=int(response["processed_height"]),
                prompt_contract_sha256=str(response["prompt_contract_sha256"]),
                rendered_user_prompt_sha256=str(
                    response["rendered_user_prompt_sha256"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise QwenProposalProtocolError(
                "proposal response fields are invalid"
            ) from error
        expected_user_text = proposal_request_text(
            task_prompt=task_prompt,
            public_history_text=public_history_text,
            camera_label=camera_label,
            frame_id=frame_id,
            processed_width=result.processed_width,
            processed_height=result.processed_height,
            enabled_primitives=enabled_primitives,
            max_candidates=max_candidates,
            max_per_primitive=max_per_primitive,
        )
        if (
            result.rendered_user_prompt_sha256
            != hashlib.sha256(expected_user_text.encode("utf-8")).hexdigest()
        ):
            raise QwenProposalProtocolError("rendered user prompt identity mismatch")
        return result


class _QwenHandler(BaseHTTPRequestHandler):
    server: _QwenServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send(self, status: int, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(
            dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path not in {"/health", "/ready"}:
            self._send(404, {"status": "error", "message": "not found"})
            return
        if self.path == "/ready":
            try:
                self.server.runtime.ensure_ready()
            except Exception as error:  # noqa: BLE001
                self._send(503, {"status": "error", "message": str(error)})
                return
        self._send(
            200,
            {
                "status": "ok",
                "schema_version": QWEN_PROPOSAL_HTTP_SCHEMA,
                "provider_id": self.server.runtime.provider_id,
                "identity": self.server.runtime.identity.to_dict(),
                "model_loaded": bool(getattr(self.server.runtime, "is_loaded", False)),
            },
        )

    def do_POST(self) -> None:
        if self.path != "/propose":
            self._send(404, {"status": "error", "message": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(value, dict):
                raise TypeError("request must be an object")
            required = {
                "schema_version",
                "provider_id",
                "task_prompt",
                "public_history_text",
                "camera_label",
                "frame_id",
                "image_sha256",
                "enabled_primitives",
                "max_candidates",
                "max_per_primitive",
                "prompt_contract_sha256",
                "request_id",
                "image_png_base64",
            }
            if set(value) != required:
                raise ValueError("request fields differ from frozen schema")
            if (
                value["schema_version"] != QWEN_PROPOSAL_HTTP_SCHEMA
                or value["provider_id"] != self.server.runtime.provider_id
            ):
                raise ValueError("request identity mismatch")
            image = decode_rgb_png(base64.b64decode(value["image_png_base64"]))
            if canonical_rgb_sha256(image) != value["image_sha256"]:
                raise ValueError("public RGB digest mismatch")
            if not isinstance(value["enabled_primitives"], list):
                raise TypeError("enabled_primitives must be an array")
            enabled_primitives = tuple(
                Primitive(item) for item in value["enabled_primitives"]
            )
            contract = qwen_proposal_prompt_contract(
                enabled_primitives=enabled_primitives,
                max_candidates=value["max_candidates"],
                max_per_primitive=value["max_per_primitive"],
            )
            if value["prompt_contract_sha256"] != contract.fingerprint:
                raise ValueError("proposal prompt contract identity mismatch")
            public = {
                key: value[key]
                for key in (
                    "schema_version",
                    "provider_id",
                    "task_prompt",
                    "public_history_text",
                    "camera_label",
                    "frame_id",
                    "image_sha256",
                    "enabled_primitives",
                    "max_candidates",
                    "max_per_primitive",
                    "prompt_contract_sha256",
                )
            }
            if canonical_sha256(public) != value["request_id"]:
                raise ValueError("proposal request digest mismatch")
            with self.server.generation_lock:
                result = self.server.runtime.generate_proposal_json(
                    task_prompt=str(value["task_prompt"]),
                    public_history_text=str(value["public_history_text"]),
                    image=image,
                    camera_label=str(value["camera_label"]),
                    frame_id=str(value["frame_id"]),
                    enabled_primitives=enabled_primitives,
                    max_candidates=int(value["max_candidates"]),
                    max_per_primitive=int(value["max_per_primitive"]),
                )
            if result.prompt_contract_sha256 != contract.fingerprint:
                raise ValueError("runtime returned a different prompt contract")
            self._send(
                200,
                {
                    "status": "ok",
                    "schema_version": QWEN_PROPOSAL_HTTP_SCHEMA,
                    "provider_id": self.server.runtime.provider_id,
                    "request_id": value["request_id"],
                    "raw_json": result.raw_json,
                    "processed_width": result.processed_width,
                    "processed_height": result.processed_height,
                    "prompt_contract_sha256": result.prompt_contract_sha256,
                    "rendered_user_prompt_sha256": (result.rendered_user_prompt_sha256),
                },
            )
        # This boundary converts backend and schema failures into one closed
        # protocol error; callers never continue from a partial proposal.
        except Exception as error:  # noqa: BLE001
            self._send(422, {"status": "error", "message": str(error)})


class _QwenServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], *, runtime: Qwen25VLRuntime) -> None:
        self.runtime = runtime
        self.generation_lock = threading.Lock()
        super().__init__(address, _QwenHandler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8004)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    identity = QwenProviderIdentity()
    runtime = Qwen25VLRuntime(
        identity=identity,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
    )
    server = _QwenServer((args.host, args.port), runtime=runtime)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
