"""One exact identity contract for every external frozen pi0.5 consumer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CHECKPOINT_SCHEMA = "piu.pi05-checkpoint-identity.v1"
SERVER_SCHEMA = "piu.identified-pi05-server.v1"


def load_checkpoint_identity(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported frozen pi0.5 checkpoint identity")
    checkpoint = value.get("checkpoint")
    if (
        value.get("policy_config") != "pi05_libero"
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema_version") != "piu.checkpoint-tree-sha256.v1"
    ):
        raise ValueError("malformed frozen pi0.5 checkpoint identity")
    return dict(value)


def expected_server_metadata(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SERVER_SCHEMA,
        "policy_config": identity["policy_config"],
        "environment": "LIBERO",
        "checkpoint": identity["checkpoint"],
    }


def validate_server_metadata(
    metadata: Mapping[str, Any], identity: Mapping[str, Any]
) -> tuple[str, ...]:
    expected = expected_server_metadata(identity)
    received = {name: metadata.get(name) for name in expected}
    capabilities = metadata.get("capabilities", ["action_chunks"])
    if (
        received != expected
        or not isinstance(capabilities, list)
        or any(not isinstance(item, str) for item in capabilities)
        or "action_chunks" not in capabilities
        or set(metadata) - {*expected, "capabilities"}
    ):
        raise ValueError("external pi0.5 server identity differs from frozen policy")
    return tuple(capabilities)
