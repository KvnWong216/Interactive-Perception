#!/usr/bin/env python3
"""Verify an immutable JSONL calibration dataset against its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    manifest = json.loads(manifest_path.read_text())
    dataset = root / manifest["dataset"]
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if digest != manifest["dataset_sha256"]:
        raise SystemExit(
            f"dataset hash mismatch: manifest={manifest['dataset_sha256']} actual={digest}"
        )
    rows = sum(bool(line) for line in dataset.read_text().splitlines())
    if rows != int(manifest["samples"]):
        raise SystemExit(
            f"dataset row mismatch: manifest={manifest['samples']} actual={rows}"
        )
    artifact = manifest.get("frozen_artifact")
    expected_artifact_sha = manifest.get("frozen_artifact_sha256_before_audit")
    if artifact and expected_artifact_sha:
        actual_artifact_sha = hashlib.sha256((root / artifact).read_bytes()).hexdigest()
        if actual_artifact_sha != expected_artifact_sha:
            raise SystemExit("frozen artifact hash changed after audit collection")
    try:
        display = dataset.relative_to(root)
    except ValueError:
        display = dataset
    print(f"verified {display}: {rows} rows, sha256={digest}")


if __name__ == "__main__":
    main()
