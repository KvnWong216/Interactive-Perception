#!/usr/bin/env python3
"""Analyze the complete prospective oracle target-prompt causal cohort."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.oracle_formal import (
    analyze_oracle_formal_schedule,
    portable,
    resolve,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schedule_path = resolve(args.schedule, repository_root=ROOT)
    output = resolve(args.output, repository_root=ROOT)
    if output.exists():
        raise FileExistsError("oracle formal result is immutable")
    result = analyze_oracle_formal_schedule(
        schedule_path, repository_root=ROOT
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pending: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".pending",
            delete=False,
        ) as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
            pending = Path(handle.name)
        os.link(pending, output)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "output": portable(output, repository_root=ROOT),
                "status": result["status"],
                "groups": result["groups"],
                "primary": result["primary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
