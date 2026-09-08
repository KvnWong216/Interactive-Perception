"""Standard-library client for isolated frozen-scorer verification.

This module deliberately does not import the scorer, Qwen provider, PyTorch,
or cache replay code.  It is safe to import in the legacy LIBERO process.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCORER_VERIFICATION_HTTP_SCHEMA = "method-v1-scorer-verification-http-v3"
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _load_auth_key(path: str | Path) -> bytes:
    key = Path(path).expanduser().resolve().read_bytes()
    if len(key) < 32:
        raise ValueError("scorer verifier authentication key must contain >=32 bytes")
    return key


def _auth_mac(key: bytes, value: object) -> str:
    return hmac.new(key, _canonical_bytes(value), hashlib.sha256).hexdigest()


def expected_verifier_source_sha256() -> str:
    """Digest of the colocated service implementation expected by this client."""

    return _file_sha256(Path(__file__).with_name("scorer_verification_service.py"))


def expected_replay_source_sha256() -> str:
    """Digest every local source file involved in checkpoint/cache replay."""

    root = Path(__file__).parent
    return _canonical_sha256(
        {
            "schema_version": "method-v1-scorer-replay-source-v1",
            "files": [
                {"name": name, "sha256": _file_sha256(root / name)}
                for name in SCORER_REPLAY_SOURCE_NAMES
            ],
        }
    )


def _require_sha256(value: object, *, name: str) -> str:
    result = str(value)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ScorerVerificationProtocolError(f"{name} is not a lowercase SHA-256")
    return result


class ScorerVerificationError(RuntimeError):
    """Base class for a verification boundary failure."""


class ScorerVerificationTransportError(ScorerVerificationError):
    """The isolated verifier could not be reached."""


class ScorerVerificationProtocolError(ScorerVerificationError):
    """The verifier returned an unbound or malformed response."""


class ScorerVerificationHTTPClient:
    """Verify a frozen decision without loading its model in this process."""

    def __init__(
        self,
        base_url: str,
        *,
        auth_key_path: str | Path,
        timeout_seconds: float = 180.0,
        expected_service_source_sha256: str | None = None,
    ) -> None:
        self.base_url = str(base_url).strip().rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self._auth_key = _load_auth_key(auth_key_path)
        self.expected_key_id = hashlib.sha256(self._auth_key).hexdigest()
        self.expected_service_source_sha256 = _require_sha256(
            expected_service_source_sha256 or expected_verifier_source_sha256(),
            name="expected verifier source",
        )
        self.expected_replay_source_sha256 = expected_replay_source_sha256()

    def _exchange(
        self, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = None
        headers: dict[str, str] = {}
        method = "GET"
        if payload is not None:
            data = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
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
        except urllib.error.HTTPError as error:
            try:
                failure = json.loads(error.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                failure = None
            if isinstance(failure, dict) and failure.get("status") == "error":
                raise ScorerVerificationError(
                    str(failure.get("message", "verification failed"))
                ) from error
            raise ScorerVerificationTransportError(str(error)) from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise ScorerVerificationTransportError(str(error)) from error
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ScorerVerificationProtocolError(
                "verifier returned invalid JSON"
            ) from error
        if not isinstance(result, dict):
            raise ScorerVerificationProtocolError("verifier response must be an object")
        if result.get("status") == "error":
            raise ScorerVerificationError(
                str(result.get("message", "verification failed"))
            )
        return result

    def health(self) -> dict[str, Any]:
        client_nonce = secrets.token_hex(32)
        result = self._exchange(f"/health?client_nonce={client_nonce}")
        if set(result) != {
            "status",
            "schema_version",
            "service_source_sha256",
            "service_identity",
            "service_id",
            "client_nonce",
            "response_mac",
        }:
            raise ScorerVerificationProtocolError(
                "scorer verifier health fields differ from schema"
            )
        if (
            result.get("status") != "ok"
            or result.get("schema_version") != SCORER_VERIFICATION_HTTP_SCHEMA
            or result.get("service_source_sha256")
            != self.expected_service_source_sha256
            or result.get("client_nonce") != client_nonce
        ):
            raise ScorerVerificationProtocolError("scorer verifier identity mismatch")
        self._validate_service_identity(result)
        self._validate_response_mac(result)
        return result

    def _validate_response_mac(self, response: dict[str, Any]) -> None:
        observed = response.get("response_mac")
        if not isinstance(observed, str):
            raise ScorerVerificationProtocolError("verifier response MAC is missing")
        body = {key: value for key, value in response.items() if key != "response_mac"}
        if not hmac.compare_digest(observed, _auth_mac(self._auth_key, body)):
            raise ScorerVerificationProtocolError(
                "verifier response authentication failed"
            )

    def _validate_service_identity(self, value: dict[str, Any]) -> None:
        identity = value.get("service_identity")
        service_id = value.get("service_id")
        if not isinstance(identity, dict) or set(identity) != {
            "schema_version",
            "service_source_sha256",
            "replay_source_sha256",
            "python_version",
            "torch_version",
            "key_id",
        }:
            raise ScorerVerificationProtocolError(
                "verifier runtime identity is malformed"
            )
        if (
            identity["schema_version"] != SCORER_VERIFICATION_HTTP_SCHEMA
            or identity["service_source_sha256"] != self.expected_service_source_sha256
            or identity["replay_source_sha256"] != self.expected_replay_source_sha256
            or identity["key_id"] != self.expected_key_id
            or _canonical_sha256(identity) != service_id
        ):
            raise ScorerVerificationProtocolError("verifier runtime identity mismatch")
        _require_sha256(identity["replay_source_sha256"], name="scorer replay source")
        if not all(
            isinstance(identity[field], str) and identity[field]
            for field in ("python_version", "torch_version")
        ):
            raise ScorerVerificationProtocolError(
                "verifier runtime versions are invalid"
            )

    def verify(
        self,
        *,
        selection_path: str | Path,
        manifest_path: str | Path,
        schedule_path: str | Path,
        entry_id: str,
    ) -> dict[str, Any]:
        paths = {
            "selection": Path(selection_path).expanduser().resolve(),
            "manifest": Path(manifest_path).expanduser().resolve(),
            "schedule": Path(schedule_path).expanduser().resolve(),
        }
        files = {
            name: {"path": str(path), "sha256": _file_sha256(path)}
            for name, path in paths.items()
        }
        selection_doc = json.loads(paths["selection"].read_text(encoding="utf-8"))
        if not isinstance(selection_doc, dict):
            raise ScorerVerificationProtocolError(
                "local frozen selection artifact is malformed"
            )
        schedule_doc = json.loads(paths["schedule"].read_text(encoding="utf-8"))
        if not isinstance(schedule_doc, dict) or not isinstance(
            schedule_doc.get("entries"), list
        ):
            raise ScorerVerificationProtocolError("local schedule file is malformed")
        matches = [
            row
            for row in schedule_doc["entries"]
            if isinstance(row, dict) and row.get("entry_id") == entry_id
        ]
        if len(matches) != 1:
            raise ScorerVerificationProtocolError(
                "entry_id is absent or ambiguous in local schedule"
            )
        candidate_id = matches[0].get("candidate_id")
        candidate_fingerprint = matches[0].get("candidate_fingerprint")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ScorerVerificationProtocolError("schedule candidate_id is invalid")
        _require_sha256(candidate_fingerprint, name="schedule candidate fingerprint")
        body = {
            "schema_version": SCORER_VERIFICATION_HTTP_SCHEMA,
            "entry_id": str(entry_id),
            "candidate_id": candidate_id,
            "candidate_fingerprint": candidate_fingerprint,
            "files": files,
            "client_nonce": secrets.token_hex(32),
        }
        request_id = _canonical_sha256(body)
        authenticated_request = {**body, "request_id": request_id}
        response = self._exchange(
            "/verify",
            {
                **authenticated_request,
                "request_mac": _auth_mac(self._auth_key, authenticated_request),
            },
        )
        expected_keys = {
            "status",
            "schema_version",
            "request_id",
            "entry_id",
            "files",
            "candidate_id",
            "candidate_fingerprint",
            "service_source_sha256",
            "service_identity",
            "service_id",
            "verified",
            "response_id",
            "client_nonce",
            "response_mac",
        }
        if set(response) != expected_keys:
            raise ScorerVerificationProtocolError(
                "verifier response fields differ from schema"
            )
        if (
            response["status"] != "ok"
            or response["schema_version"] != SCORER_VERIFICATION_HTTP_SCHEMA
            or response["request_id"] != request_id
            or response["entry_id"] != str(entry_id)
            or response["candidate_id"] != candidate_id
            or response["candidate_fingerprint"] != candidate_fingerprint
            or response["files"] != files
            or response["client_nonce"] != body["client_nonce"]
            or response["service_source_sha256"] != self.expected_service_source_sha256
        ):
            raise ScorerVerificationProtocolError("verifier response identity mismatch")
        self._validate_service_identity(response)
        self._validate_response_mac(response)
        response_body = {
            key: value
            for key, value in response.items()
            if key not in {"response_id", "response_mac"}
        }
        if _canonical_sha256(response_body) != response["response_id"]:
            raise ScorerVerificationProtocolError("verifier response digest mismatch")
        verified = response["verified"]
        if not isinstance(verified, dict) or set(verified) != {
            "artifact_sha256",
            "selection_sha256",
            "checkpoint_sha256",
            "checkpoint_identity_sha256",
            "training_dataset_sha256",
            "training_admission_evidence_sha256",
            "training_collection_plan_sha256",
            "scorer_verifier_auth_key_id",
            "candidate_id",
            "candidate_fingerprint",
            "temperature",
        }:
            raise ScorerVerificationProtocolError(
                "verified selection fields are invalid"
            )
        for field in (
            "artifact_sha256",
            "selection_sha256",
            "checkpoint_sha256",
            "checkpoint_identity_sha256",
            "training_dataset_sha256",
            "training_admission_evidence_sha256",
            "training_collection_plan_sha256",
            "scorer_verifier_auth_key_id",
            "candidate_fingerprint",
        ):
            _require_sha256(verified[field], name=f"verified {field}")
        if (
            verified["candidate_id"] != candidate_id
            or verified["candidate_fingerprint"] != candidate_fingerprint
            or not isinstance(verified["temperature"], (int, float))
            or isinstance(verified["temperature"], bool)
            or not math.isfinite(float(verified["temperature"]))
            or float(verified["temperature"]) <= 0
        ):
            raise ScorerVerificationProtocolError(
                "verified selection identity is invalid"
            )
        try:
            local_bindings = selection_doc["bindings"]
            local_checkpoint = selection_doc["checkpoint"]
            local_training = selection_doc["training_provenance"]
            expected_verified = {
                "artifact_sha256": selection_doc["artifact_sha256"],
                "selection_sha256": selection_doc["selection_sha256"],
                "checkpoint_sha256": local_checkpoint["sha256"],
                "checkpoint_identity_sha256": local_checkpoint["identity_sha256"],
                "training_dataset_sha256": local_training["training_dataset_sha256"],
                "training_admission_evidence_sha256": local_training[
                    "training_admission_evidence_sha256"
                ],
                "training_collection_plan_sha256": local_training[
                    "collection_plan_sha256"
                ],
                "scorer_verifier_auth_key_id": local_training[
                    "scorer_verifier_auth_key_id"
                ],
                "candidate_id": local_bindings["candidate_id"],
                "candidate_fingerprint": local_bindings["candidate_fingerprint"],
            }
        except (KeyError, TypeError) as error:
            raise ScorerVerificationProtocolError(
                "local frozen selection bindings are malformed"
            ) from error
        if any(verified[field] != value for field, value in expected_verified.items()):
            raise ScorerVerificationProtocolError(
                "verified result differs from the local frozen selection artifact"
            )
        for name, path in paths.items():
            if _file_sha256(path) != files[name]["sha256"]:
                raise ScorerVerificationProtocolError(
                    f"{name} file changed during scorer verification"
                )
        return verified
