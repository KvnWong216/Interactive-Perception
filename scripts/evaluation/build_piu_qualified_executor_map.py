#!/usr/bin/env python3
"""Assemble immutable formal primitive certificates into an executor map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    certificates = [resolve(path) for path in args.certificate]
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("qualified-executor maps are immutable")
    candidates = {}
    primitives = {}
    for path in certificates:
        value = json.loads(path.read_text())
        if (
            value.get("schema_version") != "piu.primitive-qualification-certificate.v1"
            or value.get("status") != "FORMALLY_QUALIFIED"
            or value.get("paper_method_action_authorized") is not True
            or value.get("result", {}).get("qualified") is not True
        ):
            raise ValueError(f"certificate does not authorize a method action: {path}")
        candidate_id = " ".join(str(value.get("candidate_id", "")).split())
        primitive = " ".join(str(value.get("primitive", "")).split()).upper()
        if not candidate_id or not primitive or candidate_id in candidates:
            raise ValueError("qualification candidates must be nonempty and unique")
        candidates[candidate_id] = {"path": portable(path), "sha256": sha256(path)}
        primitives[candidate_id] = primitive
    result = {
        "schema_version": "piu.qualified-executor-map.v1",
        "status": "FORMALLY_QUALIFIED_CANDIDATES_ONLY",
        "candidates": candidates,
        "primitives": primitives,
        "paper_method_action_authorized": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": portable(output), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
