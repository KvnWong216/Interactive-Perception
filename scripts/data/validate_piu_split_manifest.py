#!/usr/bin/env python3
"""Validate and hash-lock a prospective PIU group split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.splits import load_split_manifest


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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = resolve(args.manifest)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("split validation reports are immutable")
    manifest = load_split_manifest(manifest_path)
    counts = Counter(row["split_role"] for row in manifest["assignments"])
    report = {
        "schema_version": "piu.group-split-validation.v1",
        "status": "PASS",
        "manifest": {
            "path": portable(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "scenario": manifest["scenario"],
        "groups": len(manifest["assignments"]),
        "role_counts": dict(sorted(counts.items())),
        "outcomes_loaded": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
