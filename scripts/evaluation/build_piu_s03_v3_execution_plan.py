#!/usr/bin/env python3
"""Create or byte-verify the prospective S03 public-v3 execution plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.s03_v3_amendment import (  # noqa: E402
    V3_PLAN_PATH,
    V3_RUNNER_IDENTITY_PATH,
    build_s03_v3_execution_plan,
    validate_s03_v3_execution_plan,
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-identity", type=Path, default=V3_RUNNER_IDENTITY_PATH)
    parser.add_argument("--output", type=Path, default=V3_PLAN_PATH)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    runner = _resolve(args.runner_identity)
    output = _resolve(args.output)
    value = build_s03_v3_execution_plan(
        runner_identity_path=runner, repository_root=ROOT
    )
    rendered = (json.dumps(value, indent=2) + "\n").encode()
    if args.verify:
        if not output.is_file() or output.read_bytes() != rendered:
            raise ValueError("S03 v3 execution plan differs byte-for-byte")
        status = "VERIFIED"
    else:
        if output.exists():
            raise FileExistsError("S03 v3 execution plan is immutable")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(rendered)
        status = "FROZEN_BEFORE_S03_V3_OUTCOMES"
    validate_s03_v3_execution_plan(output, repository_root=ROOT)
    print(
        json.dumps(
            {
                "status": status,
                "path": str(output.relative_to(ROOT)),
                "record_count": value["record_count"],
                "inference_executed": False,
                "outcome_present": False,
                "paper_claim_ready": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
