#!/usr/bin/env python3
"""Generate an immutable content identity for a local pi0.5 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from checkpoint_identity import checkpoint_identity

ROOT = Path(__file__).resolve().parents[2]


def portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"checkpoint identity is immutable: {output}")
    result = {
        "schema_version": "piu.pi05-checkpoint-identity.v1",
        "policy_config": "pi05_libero",
        "checkpoint": checkpoint_identity(checkpoint),
        "source_checkpoint_path": portable(checkpoint),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result, "output": portable(output)}, indent=2))


if __name__ == "__main__":
    main()
