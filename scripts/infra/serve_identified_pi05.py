#!/usr/bin/env python3
"""Serve pi05_libero with auditable checkpoint identity metadata."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/infra"))

from checkpoint_identity import checkpoint_identity  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
from openpi.training import config as training_config  # noqa: E402

SERVER_SCHEMA = "piu.identified-pi05-server.v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-config", default="pi05_libero")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    identity = checkpoint_identity(checkpoint)
    logging.info("Checkpoint identity: %s", identity)
    policy = policy_config.create_trained_policy(
        training_config.get_config(args.policy_config), checkpoint
    )
    metadata = {
        "schema_version": SERVER_SCHEMA,
        "policy_config": args.policy_config,
        "environment": "LIBERO",
        "checkpoint": identity,
    }
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
