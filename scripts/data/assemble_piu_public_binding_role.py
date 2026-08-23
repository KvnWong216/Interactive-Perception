#!/usr/bin/env python3
"""Assemble immutable public transitions and binding labels for one split role."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.dataset_assembly import assemble_public_binding_role


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, action="append", required=True)
    parser.add_argument("--binding-label", type=Path, action="append", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-role", required=True)
    parser.add_argument("--output-public", type=Path, required=True)
    parser.add_argument("--output-labels", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    report = assemble_public_binding_role(
        public_sources=[resolve(path) for path in args.public],
        binding_label_sources=[resolve(path) for path in args.binding_label],
        split_manifest_path=resolve(args.split_manifest),
        split_role=args.split_role,
        output_public=resolve(args.output_public),
        output_labels=resolve(args.output_labels),
        output_manifest=resolve(args.output_manifest),
        repository_root=ROOT,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
