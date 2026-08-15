#!/usr/bin/env python3
"""Render the strict final-product GO/NOT-GO decision from its frozen registry."""

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
        "--spec",
        type=Path,
        default=ROOT / "benchmarks/final_product_v1/gates.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/final_product_gate_v1.json",
    )
    args = parser.parse_args()

    summary = summarize_product_gates(yaml.safe_load(args.spec.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    decision = "GO" if summary["final_product_go"] else "NOT-GO"
    print(f"Final product: {decision}")
    print(f"Blocking gates: {', '.join(summary['blocking_failures']) or 'none'}")


if __name__ == "__main__":
    main()
