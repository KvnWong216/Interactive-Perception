#!/usr/bin/env python3
"""Apply an explicit reliability requirement to a pure-policy result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interactive_perception.capability_gate import CapabilityGate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--required-reliability", type=float, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    task = result["tasks"][args.task_id]
    gate = CapabilityGate(
        successes=int(task["successes"]),
        trials=int(task["episodes"]),
        confidence=args.confidence,
        required_reliability=args.required_reliability,
    )
    report = gate.to_dict()
    text = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    if not gate.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
