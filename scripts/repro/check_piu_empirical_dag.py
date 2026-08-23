#!/usr/bin/env python3
"""Validate and summarize the hash-bound PIU external experiment DAG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.empirical_dag import evaluate_empirical_dag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dag",
        type=Path,
        default=ROOT / "configs/experiments/piu_empirical_stage_dag_v1.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dag = args.dag if args.dag.is_absolute() else ROOT / args.dag
    report = evaluate_empirical_dag(dag, repository_root=ROOT)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        if output.exists():
            raise FileExistsError("empirical DAG status reports are immutable")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "paper_claim_ready": report["paper_claim_ready"],
                "next_actionable_stages": report["next_actionable_stages"],
                "stages": [
                    {"id": row["id"], "status": row["status"]}
                    for row in report["stages"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
