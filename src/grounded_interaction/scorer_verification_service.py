"""Modern-PyTorch HTTP service for replaying frozen Method-V1 selections."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import platform
import threading
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .contracts import canonical_sha256
from .method_v1_data import (
    BranchScheduleEntry,
    DecisionGroupManifest,
    validate_branch_schedule,
)
from .scorer_verification_client import SCORER_VERIFICATION_HTTP_SCHEMA
from .selection_provenance import (
    VerifiedFrozenSelection,
    verify_frozen_scorer_selection,
)

SCORER_REPLAY_SOURCE_NAMES = (
    "contracts.py",
    "data.py",
    "losses.py",
    "method_v1_data.py",
    "model.py",
    "proposals.py",
    "qwen_provider.py",
    "selection.py",
    "selection_provenance.py",
    "tokens.py",
    "train_outcomes.py",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _auth_mac(key: bytes, value: object) -> str:
    return hmac.new(key, _canonical_bytes(value), hashlib.sha256).hexdigest()


def _load_auth_key(path: str | Path) -> bytes:
    key = Path(path).expanduser().resolve().read_bytes()
    if len(key) < 32:
        raise ValueError("scorer verifier authentication key must contain >=32 bytes")
    return key


def _service_identity(auth_key: bytes) -> dict[str, str]:
    import torch

    try:
        major, minor = (
            int(value)
            for value in str(torch.__version__).split("+", 1)[0].split(".")[:2]
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("cannot parse scorer-service PyTorch version") from error
    if (major, minor) < (2, 11):
        raise RuntimeError("scorer verification requires PyTorch >=2.11")
    identity = {
        "schema_version": SCORER_VERIFICATION_HTTP_SCHEMA,
        "service_source_sha256": _file_sha256(Path(__file__)),
        "replay_source_sha256": _replay_source_sha256(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "key_id": hashlib.sha256(auth_key).hexdigest(),
    }
    return identity


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replay_source_sha256() -> str:
    root = Path(__file__).parent
    return canonical_sha256(
        {
            "schema_version": "method-v1-scorer-replay-source-v1",
            "files": [
                {"name": name, "sha256": _file_sha256(root / name)}
                for name in SCORER_REPLAY_SOURCE_NAMES
            ],
        }
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_inputs(
    files: Mapping[str, Any], *, entry_id: str
) -> tuple[Path, DecisionGroupManifest, BranchScheduleEntry]:
    if set(files) != {"selection", "manifest", "schedule"}:
        raise ValueError("verification files differ from frozen schema")
    paths: dict[str, Path] = {}
    for name in ("selection", "manifest", "schedule"):
        record = files[name]
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ValueError(f"{name} file identity is malformed")
        path = Path(str(record["path"])).expanduser().resolve()
        if _file_sha256(path) != record["sha256"]:
            raise ValueError(f"{name} file SHA-256 mismatch")
        paths[name] = path

    manifest = DecisionGroupManifest.from_mapping(_read_json(paths["manifest"]))
    schedule_doc = _read_json(paths["schedule"])
    if not isinstance(schedule_doc, Mapping):
        raise TypeError("schedule file must be an object")
    required = {
        "schema_version",
        "schedule_id",
        "manifest_sha256",
        "entries",
        "schedule_sha256",
    }
    if set(schedule_doc) != required:
        raise ValueError("schedule file fields differ from frozen schema")
    schedule_body = {
        key: value for key, value in schedule_doc.items() if key != "schedule_sha256"
    }
    if canonical_sha256(schedule_body) != schedule_doc["schedule_sha256"]:
        raise ValueError("schedule file digest mismatch")
    if schedule_doc["manifest_sha256"] != manifest.fingerprint():
        raise ValueError("schedule file manifest identity mismatch")
    entries = tuple(
        BranchScheduleEntry.from_mapping(item) for item in schedule_doc["entries"]
    )
    validate_branch_schedule((manifest,), entries)
    matches = [item for item in entries if item.entry_id == entry_id]
    if len(matches) != 1:
        raise ValueError("entry_id is absent or ambiguous in frozen schedule")
    return paths["selection"], manifest, matches[0]


class _VerificationHandler(BaseHTTPRequestHandler):
    server: _VerificationServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send(self, status: int, payload: Mapping[str, Any]) -> None:
        raw = json.dumps(
            dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/health":
            self._send(404, {"status": "error", "message": "not found"})
            return
        try:
            query = urllib.parse.parse_qs(parsed.query, strict_parsing=True)
        except ValueError:
            query = {}
        if set(query) != {"client_nonce"} or len(query["client_nonce"]) != 1:
            self._send(422, {"status": "error", "message": "health nonce is required"})
            return
        client_nonce = query["client_nonce"][0]
        if len(client_nonce) != 64 or any(
            character not in "0123456789abcdef" for character in client_nonce
        ):
            self._send(422, {"status": "error", "message": "health nonce is invalid"})
            return
        try:
            self.server.assert_runtime_identity_current()
        except RuntimeError as error:
            self._send(503, {"status": "error", "message": str(error)})
            return
        identity = self.server.service_identity
        body = {
            "status": "ok",
            "schema_version": SCORER_VERIFICATION_HTTP_SCHEMA,
            "service_source_sha256": identity["service_source_sha256"],
            "service_identity": identity,
            "service_id": canonical_sha256(identity),
            "client_nonce": client_nonce,
        }
        self._send(200, {**body, "response_mac": _auth_mac(self.server.auth_key, body)})

    def do_POST(self) -> None:
        if self.path != "/verify":
            self._send(404, {"status": "error", "message": "not found"})
            return
        try:
            self.server.assert_runtime_identity_current()
            length = int(self.headers.get("Content-Length", "0"))
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            required = {
                "schema_version",
                "entry_id",
                "candidate_id",
                "candidate_fingerprint",
                "files",
                "client_nonce",
                "request_id",
                "request_mac",
            }
            if not isinstance(value, Mapping) or set(value) != required:
                raise ValueError("verification request fields differ from schema")
            if value["schema_version"] != SCORER_VERIFICATION_HTTP_SCHEMA:
                raise ValueError("verification request schema mismatch")
            request_body = {
                key: value[key]
                for key in (
                    "schema_version",
                    "entry_id",
                    "candidate_id",
                    "candidate_fingerprint",
                    "files",
                    "client_nonce",
                )
            }
            if canonical_sha256(request_body) != value["request_id"]:
                raise ValueError("verification request digest mismatch")
            authenticated_request = {**request_body, "request_id": value["request_id"]}
            expected_request_mac = _auth_mac(
                self.server.auth_key, authenticated_request
            )
            if not isinstance(value["request_mac"], str) or not hmac.compare_digest(
                value["request_mac"], expected_request_mac
            ):
                raise ValueError("verification request authentication failed")
            selection_path, manifest, entry = _load_inputs(
                value["files"], entry_id=str(value["entry_id"])
            )
            if (
                value["candidate_id"] != entry.candidate_id
                or value["candidate_fingerprint"] != entry.candidate_fingerprint
            ):
                raise ValueError("request candidate identity differs from schedule")
            with self.server.verification_lock:
                verified = self.server.verifier(
                    selection_path, manifest=manifest, entry=entry
                )
            verified_payload = asdict(verified)
            response_body = {
                "status": "ok",
                "schema_version": SCORER_VERIFICATION_HTTP_SCHEMA,
                "request_id": value["request_id"],
                "entry_id": entry.entry_id,
                "candidate_id": entry.candidate_id,
                "candidate_fingerprint": entry.candidate_fingerprint,
                "files": value["files"],
                "client_nonce": value["client_nonce"],
                "service_source_sha256": self.server.service_identity[
                    "service_source_sha256"
                ],
                "service_identity": self.server.service_identity,
                "service_id": canonical_sha256(self.server.service_identity),
                "verified": verified_payload,
            }
            authenticated_response = {
                **response_body,
                "response_id": canonical_sha256(response_body),
            }
            self._send(
                200,
                {
                    **authenticated_response,
                    "response_mac": _auth_mac(
                        self.server.auth_key, authenticated_response
                    ),
                },
            )
        except Exception as error:  # noqa: BLE001 -- fail-closed process boundary
            self._send(422, {"status": "error", "message": str(error)})


class _VerificationServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        *,
        auth_key: bytes,
        verifier: Callable[
            ..., VerifiedFrozenSelection
        ] = verify_frozen_scorer_selection,
    ) -> None:
        self.verifier = verifier
        self.verification_lock = threading.Lock()
        if len(auth_key) < 32:
            raise ValueError(
                "scorer verifier authentication key must contain >=32 bytes"
            )
        self.auth_key = bytes(auth_key)
        self.service_identity = _service_identity(self.auth_key)
        super().__init__(address, _VerificationHandler)

    def assert_runtime_identity_current(self) -> None:
        if _service_identity(self.auth_key) != self.service_identity:
            raise RuntimeError(
                "scorer verifier source or runtime changed after startup"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--auth-key-file", required=True, type=Path)
    args = parser.parse_args(argv)
    server = _VerificationServer(
        (args.host, args.port), auth_key=_load_auth_key(args.auth_key_file)
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
