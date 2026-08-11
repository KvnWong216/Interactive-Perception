#!/usr/bin/env python3
"""Combine separately produced arm summaries without rerunning episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = {}
    for path in args.summaries:
        value = json.loads(path.read_text(encoding="utf-8"))
        arm = str(value.get("arm", "unknown"))
        if arm in arms:
            raise SystemExit(f"duplicate arm {arm!r}")
        arms[arm] = {
            "source": str(path),
            "policy": value.get("policy"),
            "seeds": value.get("seeds"),
            "conditions": value.get("conditions", {}),
        }
    required = {"monolithic", "fixed-rule", "uncertainty-router"}
    missing = required - set(arms)
    if missing:
        raise SystemExit(f"missing method arms: {sorted(missing)}")
    report = {"schema_version": "interactive-perception.method-comparison.v1", "arms": arms}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
