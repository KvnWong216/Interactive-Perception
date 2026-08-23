#!/usr/bin/env python3
"""Run the preregistered paired PIU analysis on a complete sealed matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.statistics import analyze_files


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiments/piu_formal_analysis_v1.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sealed-authorization", type=Path, required=True)
    args = parser.parse_args()
    outcomes = args.outcomes if args.outcomes.is_absolute() else ROOT / args.outcomes
    config = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    authorization_path = (
        args.sealed_authorization
        if args.sealed_authorization.is_absolute()
        else ROOT / args.sealed_authorization
    )
    if output.exists():
        raise FileExistsError(f"formal reports are immutable: {output}")
    authorization = json.loads(authorization_path.read_text())
    if authorization.get("schema_version") != "piu.formal-analysis-sealed-authorization.v1":
        raise ValueError("unsupported formal-analysis sealed authorization")
    expected = {
        "outcomes_sha256": sha256(outcomes),
        "config_sha256": sha256(config),
        "single_use_output": portable(output),
    }
    for name, value in expected.items():
        if authorization.get(name) != value:
            raise ValueError(f"formal-analysis authorization differs at {name}")
    report = analyze_files(outcomes, config)
    report["sealed_authorization"] = {
        "path": portable(authorization_path),
        "sha256": sha256(authorization_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
