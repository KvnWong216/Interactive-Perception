#!/usr/bin/env python3
"""Fit a semantic-intent split-conformal artifact from held-out JSONL data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interactive_perception.semantic_conformal import SemanticConformalCalibrator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--minimum-examples", type=int, default=30)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    if len(rows) < args.minimum_examples:
        raise SystemExit(
            f"G4 NOT-GO: {len(rows)} calibration examples; need at least "
            f"{args.minimum_examples} independent held-out examples"
        )
    examples = [(row["primitive_evidence"], str(row["true_intent"])) for row in rows]
    calibrator = SemanticConformalCalibrator.fit(
        examples,
        alpha=args.alpha,
        policy_id=args.policy_id,
        split_id=args.split_id,
    )
    artifact = calibrator.to_dict()
    artifact["dataset"] = str(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"G4 GO: wrote {args.output}")


if __name__ == "__main__":
    main()
