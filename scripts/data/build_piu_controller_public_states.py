#!/usr/bin/env python3
"""Derive PIU controller state sets from public calibrated temporal memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.temporal_memory import PublicTemporalMemory


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
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    memory_path = resolve(args.memory)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("controller public-state artifacts are immutable")
    rows = [json.loads(line) for line in memory_path.read_text().splitlines() if line]
    if not rows:
        raise ValueError("public controller-memory file is empty")
    seen: set[str] = set()
    derived = []
    for index, row in enumerate(rows):
        if row.get("schema_version") != "piu.public-controller-memory.v2":
            raise ValueError("unsupported public controller-memory row")
        if (
            row.get("public_inputs_only") is not True
            or row.get("online_oracle_inputs") != []
        ):
            raise ValueError("controller memory must be public-input only")
        sample_id = " ".join(str(row.get("sample_id", "")).split())
        group = " ".join(str(row.get("initial_state_group", "")).split())
        if not sample_id or not group or sample_id in seen:
            raise ValueError("controller memory identities must be nonempty and unique")
        seen.add(sample_id)
        memory = PublicTemporalMemory.from_mapping(row.get("memory", {}))
        derived.append(
            {
                "schema_version": "piu.controller-public-state-sets.v1",
                "sample_id": sample_id,
                "initial_state_group": group,
                "public_inputs_only": True,
                "online_oracle_inputs": [],
                "state_sets": {
                    "search_coverage_sufficient": [memory.search_coverage_sufficient],
                },
                "derivation": {
                    "schema": "piu.public-temporal-memory.v2",
                    "source_path": portable(memory_path),
                    "source_sha256": sha256(memory_path),
                    "source_row_index": index,
                },
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in derived)
    )
    print(json.dumps({"output": portable(output), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
