#!/usr/bin/env python3
"""Merge disjoint T01 reveal runs and apply the frozen reliability gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interactive_perception.capability_gate import CapabilityGate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-reliability", type=float, default=0.9)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    reports = [json.loads(path.read_text()) for path in args.inputs]
    rows = [row for report in reports for row in report["rows"]]
    seeds = [int(row["seed"]) for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError("input reports contain overlapping seeds")
    route_successes = sum(row["prediction_set"] == ["REMOVE_OCCLUDER"] for row in rows)
    reveal_successes = sum(bool(row["reveal_success"]) for row in rows)
    route_gate = CapabilityGate(
        route_successes, len(rows), args.confidence, args.required_reliability
    ).to_dict()
    route_gate["interpretation"] = "correct singleton conformal route on the target scene"
    reveal_gate = CapabilityGate(
        reveal_successes, len(rows), args.confidence, args.required_reliability
    ).to_dict()
    report = {
        "schema_version": "interactive-perception.t01-conformal-reveal-summary.v1",
        "claim_scope": "route and physically reveal the hidden target",
        "non_claim": "retrieve the revealed target and complete the final placement task",
        "required_reliability": args.required_reliability,
        "confidence": args.confidence,
        "test_seeds": sorted(seeds),
        "test_seed_count": len(seeds),
        "route_gate": route_gate,
        "reveal_gate": reveal_gate,
        "pipeline_go": bool(route_gate["passed"] and reveal_gate["passed"]),
        "failure_seeds": [row["seed"] for row in rows if not row["reveal_success"]],
        "controller_joint_reads": 0,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.inputs
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
