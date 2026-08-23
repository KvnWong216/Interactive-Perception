from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
IDENTITY_MODULE = ROOT / "scripts/infra/checkpoint_identity.py"
CHECK_MODULE = ROOT / "scripts/infra/check_external_pi05.py"


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
        {**metadata, "capabilities": ["action_chunks", "spatial_prefix_v1"]},
        identity,
    )
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
