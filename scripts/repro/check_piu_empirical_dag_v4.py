#!/usr/bin/env python3
"""Check the S03 public execution-v3 amendment without inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.empirical_dag_v4 import evaluate_empirical_dag_v4  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dag",
        type=Path,
        default=ROOT / "configs/experiments/piu_empirical_stage_dag_v4.yaml",
    )
    args = parser.parse_args()
    path = args.dag if args.dag.is_absolute() else ROOT / args.dag
    print(json.dumps(evaluate_empirical_dag_v4(path, repository_root=ROOT), indent=2))


if __name__ == "__main__":
    main()
