#!/usr/bin/env python3
"""Freeze or byte-verify S03 public inputs without running inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.s03_preparation import (
    MANIFEST_PATH,
    SCHEDULE_PATH,
    build_s03_input_manifest,
    build_s03_offline_schedule,
    render_json,
    sha256,
    validate_s03_input_manifest,
    validate_s03_offline_schedule,
)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--schedule", type=Path, default=SCHEDULE_PATH)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="regenerate in memory and compare existing artifacts byte-for-byte",
    )
    args = parser.parse_args()
    manifest_path = resolve(args.manifest)
    schedule_path = resolve(args.schedule)

    manifest = build_s03_input_manifest(repository_root=ROOT)
    manifest_bytes = render_json(manifest)
    manifest_digest = __import__("hashlib").sha256(manifest_bytes).hexdigest()
    schedule = build_s03_offline_schedule(
        manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_digest,
        repository_root=ROOT,
    )
    schedule_bytes = render_json(schedule)

    if args.verify:
        if not manifest_path.is_file() or not schedule_path.is_file():
            raise FileNotFoundError("S03 preparation artifacts are not both present")
        if manifest_path.read_bytes() != manifest_bytes:
            raise ValueError("S03 input manifest differs byte-for-byte")
        if schedule_path.read_bytes() != schedule_bytes:
            raise ValueError("S03 offline schedule differs byte-for-byte")
        status = "VERIFIED"
    else:
        if manifest_path.exists() or schedule_path.exists():
            raise FileExistsError("S03 preparation artifacts are immutable")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        schedule_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(manifest_bytes)
        schedule_path.write_bytes(schedule_bytes)
        status = "FROZEN_BEFORE_S03_OUTCOMES"

    validate_s03_input_manifest(manifest_path, repository_root=ROOT)
    validate_s03_offline_schedule(schedule_path, repository_root=ROOT)
    print(
        json.dumps(
            {
                "status": status,
                "manifest": {
                    "path": str(manifest_path.relative_to(ROOT)),
                    "sha256": sha256(manifest_path),
                },
                "schedule": {
                    "path": str(schedule_path.relative_to(ROOT)),
                    "sha256": sha256(schedule_path),
                },
                "counts": manifest["counts"],
                "s02_indices_101_and_104_included": True,
                "inference_executed": False,
                "rollout_executed": False,
                "outcome_present": False,
                "paper_claim_ready": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
