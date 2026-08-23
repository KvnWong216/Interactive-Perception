#!/usr/bin/env python3
"""Create or verify the hash-bound PIU offline release inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.reproducibility import audit_repro_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs/experiments/piu_offline_repro_v3.yaml",
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output = args.output if args.output.is_absolute() else ROOT / args.output
    reference = None
    if args.reference is not None:
        reference_path = args.reference if args.reference.is_absolute() else ROOT / args.reference
        reference = json.loads(reference_path.read_text())
    if output.exists() and not args.force:
        raise FileExistsError(f"repro audit already exists: {output}")
    report = audit_repro_manifest(
        manifest, repository_root=ROOT, reference_report=reference
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("offline_ready", "empirical_ready", "paper_claim_ready", "errors")}, indent=2))
    if not report["offline_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
