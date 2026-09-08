from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from test_method_v1_selection_provenance import _frozen_inputs

from grounded_interaction.contracts import canonical_json_bytes, canonical_sha256
from grounded_interaction.scorer_verification_client import (
    SCORER_VERIFICATION_HTTP_SCHEMA,
    ScorerVerificationHTTPClient,
    ScorerVerificationProtocolError,
)
from grounded_interaction.scorer_verification_service import _VerificationServer
from grounded_interaction.selection_provenance import VerifiedFrozenSelection

AUTH_KEY = b"method-v1-test-authentication-key!!"


def _auth_key_file(tmp_path: Path) -> Path:
    path = tmp_path / "scorer-auth.key"
    path.write_bytes(AUTH_KEY)
    return path


def _write(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def test_client_import_does_not_load_torch_or_qwen_replay() -> None:
    code = (
        "import sys; import grounded_interaction.scorer_verification_client; "
        "assert 'torch' not in sys.modules; "
        "assert 'grounded_interaction.selection_provenance' not in sys.modules; "
        "assert 'grounded_interaction.qwen_provider' not in sys.modules"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)


def _shared_files(tmp_path: Path):
    inputs = _frozen_inputs(tmp_path)
    manifest = inputs["manifest"]
    schedule = tuple(inputs["schedule"])
    manifest_path = tmp_path / "decision_manifest.json"
    schedule_path = tmp_path / "branch_schedule.json"
    selection_path = tmp_path / "selection.json"
    _write(manifest_path, manifest.to_dict())
    body = {
        "schema_version": "method-v1-branch-schedule-file-v1",
        "schedule_id": schedule[0].schedule_id,
        "manifest_sha256": manifest.fingerprint(),
        "entries": [entry.to_dict() for entry in schedule],
    }
    _write(schedule_path, {**body, "schedule_sha256": canonical_sha256(body)})
    _write(
        selection_path,
        {
            "artifact_sha256": "a" * 64,
            "selection_sha256": "b" * 64,
            "checkpoint": {
                "sha256": "c" * 64,
                "identity_sha256": "d" * 64,
            },
            "training_provenance": {
                "training_dataset_sha256": "e" * 64,
                "training_admission_evidence_sha256": "f" * 64,
                "collection_plan_sha256": "1" * 64,
                "scorer_verifier_auth_key_id": "2" * 64,
                "training_split_group_ids": ["training-family-a"],
            },
            "bindings": {
                "candidate_id": schedule[0].candidate_id,
                "candidate_fingerprint": schedule[0].candidate_fingerprint,
            },
        },
    )
    return selection_path, manifest_path, schedule_path, schedule[0]


class _RunningServer:
    def __init__(self, verifier):
        self.server = _VerificationServer(
            ("127.0.0.1", 0), auth_key=AUTH_KEY, verifier=verifier
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def test_isolated_verifier_binds_all_files_entry_and_response(tmp_path: Path) -> None:
    selection, manifest, schedule, entry = _shared_files(tmp_path)
    calls = []

    def verifier(path, *, manifest, entry):
        calls.append((Path(path), manifest.fingerprint(), entry.entry_id))
        return VerifiedFrozenSelection(
            artifact_sha256="a" * 64,
            selection_sha256="b" * 64,
            checkpoint_sha256="c" * 64,
            checkpoint_identity_sha256="d" * 64,
            training_dataset_sha256="e" * 64,
            training_admission_evidence_sha256="f" * 64,
            training_collection_plan_sha256="1" * 64,
            scorer_verifier_auth_key_id="2" * 64,
            candidate_id=entry.candidate_id,
            candidate_fingerprint=entry.candidate_fingerprint,
            temperature=1.0,
        )

    with _RunningServer(verifier) as endpoint:
        client = ScorerVerificationHTTPClient(
            endpoint, auth_key_path=_auth_key_file(tmp_path)
        )
        assert client.health()["schema_version"] == SCORER_VERIFICATION_HTTP_SCHEMA
        result = client.verify(
            selection_path=selection,
            manifest_path=manifest,
            schedule_path=schedule,
            entry_id=entry.entry_id,
        )
    assert result["artifact_sha256"] == "a" * 64
    assert calls == [(selection.resolve(), entry.manifest_sha256, entry.entry_id)]


def test_client_rejects_divergent_replay_source_identity(tmp_path: Path) -> None:
    with _RunningServer(lambda *args, **kwargs: pytest.fail("unused")) as endpoint:
        client = ScorerVerificationHTTPClient(
            endpoint, auth_key_path=_auth_key_file(tmp_path)
        )
        client.expected_replay_source_sha256 = "0" * 64
        with pytest.raises(
            ScorerVerificationProtocolError, match="runtime identity mismatch"
        ):
            client.health()


def test_health_rejects_client_without_matching_secret(tmp_path: Path) -> None:
    wrong_key = tmp_path / "wrong-auth.key"
    wrong_key.write_bytes(b"x" * 32)
    with _RunningServer(lambda *args, **kwargs: pytest.fail("unused")) as endpoint:
        client = ScorerVerificationHTTPClient(endpoint, auth_key_path=wrong_key)
        with pytest.raises(
            ScorerVerificationProtocolError, match="runtime identity mismatch"
        ):
            client.health()


def test_authentication_key_must_be_at_least_32_bytes(tmp_path: Path) -> None:
    short_key = tmp_path / "short.key"
    short_key.write_bytes(b"too-short")
    with pytest.raises(ValueError, match=">=32 bytes"):
        ScorerVerificationHTTPClient("http://127.0.0.1:1", auth_key_path=short_key)


def test_client_cross_checks_verified_result_against_local_artifact(
    tmp_path: Path,
) -> None:
    selection, manifest, schedule, entry = _shared_files(tmp_path)
    document = json.loads(selection.read_text(encoding="utf-8"))
    document["artifact_sha256"] = "9" * 64
    _write(selection, document)

    def verifier(path, *, manifest, entry):
        return VerifiedFrozenSelection(
            artifact_sha256="a" * 64,
            selection_sha256="b" * 64,
            checkpoint_sha256="c" * 64,
            checkpoint_identity_sha256="d" * 64,
            training_dataset_sha256="e" * 64,
            training_admission_evidence_sha256="f" * 64,
            training_collection_plan_sha256="1" * 64,
            scorer_verifier_auth_key_id="2" * 64,
            candidate_id=entry.candidate_id,
            candidate_fingerprint=entry.candidate_fingerprint,
            temperature=1.0,
        )

    with _RunningServer(verifier) as endpoint:
        client = ScorerVerificationHTTPClient(
            endpoint, auth_key_path=_auth_key_file(tmp_path)
        )
        with pytest.raises(
            ScorerVerificationProtocolError, match="local frozen selection"
        ):
            client.verify(
                selection_path=selection,
                manifest_path=manifest,
                schedule_path=schedule,
                entry_id=entry.entry_id,
            )


def test_verifier_fails_closed_when_shared_artifact_changes(tmp_path: Path) -> None:
    selection, manifest, schedule, entry = _shared_files(tmp_path)
    client = ScorerVerificationHTTPClient(
        "http://127.0.0.1:1", auth_key_path=_auth_key_file(tmp_path)
    )
    original_exchange = client._exchange

    def mutate_then_exchange(path, payload=None):
        if path == "/verify":
            selection.write_text('{"tampered":true}\n', encoding="utf-8")
        return original_exchange(path, payload)

    with _RunningServer(
        lambda *args, **kwargs: pytest.fail("replay must not run")
    ) as endpoint:
        client.base_url = endpoint
        client._exchange = mutate_then_exchange  # type: ignore[method-assign]
        with pytest.raises(Exception, match="selection file SHA-256 mismatch"):
            client.verify(
                selection_path=selection,
                manifest_path=manifest,
                schedule_path=schedule,
                entry_id=entry.entry_id,
            )


def test_client_rehashes_shared_files_after_verifier_response(tmp_path: Path) -> None:
    selection, manifest, schedule, entry = _shared_files(tmp_path)

    def verifier(path, *, manifest, entry):
        return VerifiedFrozenSelection(
            artifact_sha256="a" * 64,
            selection_sha256="b" * 64,
            checkpoint_sha256="c" * 64,
            checkpoint_identity_sha256="d" * 64,
            training_dataset_sha256="e" * 64,
            training_admission_evidence_sha256="f" * 64,
            training_collection_plan_sha256="1" * 64,
            scorer_verifier_auth_key_id="2" * 64,
            candidate_id=entry.candidate_id,
            candidate_fingerprint=entry.candidate_fingerprint,
            temperature=1.0,
        )

    with _RunningServer(verifier) as endpoint:
        client = ScorerVerificationHTTPClient(
            endpoint, auth_key_path=_auth_key_file(tmp_path)
        )
        exchange = client._exchange

        def mutate_after_response(path, payload=None):
            response = exchange(path, payload)
            if path == "/verify":
                manifest.write_text('{"changed":true}\n', encoding="utf-8")
            return response

        client._exchange = mutate_after_response  # type: ignore[method-assign]
        with pytest.raises(
            ScorerVerificationProtocolError,
            match="manifest file changed during scorer verification",
        ):
            client.verify(
                selection_path=selection,
                manifest_path=manifest,
                schedule_path=schedule,
                entry_id=entry.entry_id,
            )


def test_client_rejects_unbound_success_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection, manifest, schedule, entry = _shared_files(tmp_path)
    client = ScorerVerificationHTTPClient(
        "http://127.0.0.1:1", auth_key_path=_auth_key_file(tmp_path)
    )

    def forged(path, payload=None):
        assert payload is not None
        identity = {
            "schema_version": SCORER_VERIFICATION_HTTP_SCHEMA,
            "service_source_sha256": client.expected_service_source_sha256,
            "replay_source_sha256": client.expected_replay_source_sha256,
            "python_version": "3.13.0",
            "torch_version": "2.11.0",
            "key_id": client.expected_key_id,
        }
        body = {
            "status": "ok",
            "schema_version": SCORER_VERIFICATION_HTTP_SCHEMA,
            "request_id": payload["request_id"],
            "entry_id": payload["entry_id"],
            "candidate_id": payload["candidate_id"],
            "candidate_fingerprint": payload["candidate_fingerprint"],
            "files": payload["files"],
            "client_nonce": payload["client_nonce"],
            "service_source_sha256": client.expected_service_source_sha256,
            "service_identity": identity,
            "service_id": canonical_sha256(identity),
            "verified": {
                "artifact_sha256": "a" * 64,
                "selection_sha256": "b" * 64,
                "checkpoint_sha256": "c" * 64,
                "checkpoint_identity_sha256": "d" * 64,
                "candidate_id": payload["candidate_id"],
                "candidate_fingerprint": payload["candidate_fingerprint"],
                "temperature": 1.0,
            },
        }
        response_id = canonical_sha256(body)
        return {**body, "response_id": response_id, "response_mac": "0" * 64}

    monkeypatch.setattr(client, "_exchange", forged)
    with pytest.raises(ScorerVerificationProtocolError, match="authentication failed"):
        client.verify(
            selection_path=selection,
            manifest_path=manifest,
            schedule_path=schedule,
            entry_id=entry.entry_id,
        )
