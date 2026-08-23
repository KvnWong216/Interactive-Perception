#!/usr/bin/env python3
"""Validate separate PIU records and aggregate the six-stage evidence chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.contracts import (
    load_evaluator_sidecars,
    load_public_transitions,
    validate_public_sidecar_pair,
)
from piu.evaluation import aggregate_stage_evidence


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    public_path = args.public if args.public.is_absolute() else ROOT / args.public
    evaluator_path = (
        args.evaluator if args.evaluator.is_absolute() else ROOT / args.evaluator
    )
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    if output_path.exists():
        raise FileExistsError(f"evaluation reports are immutable: {output_path}")
    public = load_public_transitions(public_path)
    evaluator = load_evaluator_sidecars(evaluator_path)
    by_id = {row.sample_id: row for row in evaluator}
    if len(by_id) != len(evaluator) or {row.sample_id for row in public} != set(by_id):
        raise ValueError("public/evaluator sample sets differ or contain duplicates")
    for row in public:
        validate_public_sidecar_pair(row, by_id[row.sample_id])
    report = aggregate_stage_evidence(evaluator)
    report["inputs"] = {
        "public": {
            "path": portable(public_path),
            "sha256": sha256(public_path),
        },
        "evaluator": {
            "path": portable(evaluator_path),
            "sha256": sha256(evaluator_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
