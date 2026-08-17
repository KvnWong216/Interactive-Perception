#!/usr/bin/env python3
"""Render the strict RSS method-paper decision from its gate registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interactive_perception.product_gate import summarize_product_gates  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", type=Path, default=ROOT / "benchmarks/rss_v1/gates.yaml"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/rss_method_gate_v1.json"
    )
    args = parser.parse_args()
    summary = summarize_product_gates(yaml.safe_load(args.spec.read_text()))
    method_go = summary.pop("final_product_go")
    report = {
        **summary,
        "schema_version": "interactive-perception.rss-method-gate-summary.v1",
        "rss_method_go": method_go,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"RSS method paper: {'GO' if report['rss_method_go'] else 'NOT-GO'}")
    print(f"Blocking gates: {', '.join(report['blocking_failures']) or 'none'}")


if __name__ == "__main__":
    main()
