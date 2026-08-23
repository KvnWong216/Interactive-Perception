#!/usr/bin/env python3
"""Build or byte-verify the PIU public claim-semantics audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from calibrated_interaction.provenance import build_claim_audit_report


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "configs/experiments/method_provenance_v1.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/diagnostics/piu_public_claim_audit_v3.json",
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.verify and args.force:
        raise ValueError("--verify and --force are mutually exclusive")
    registry = resolve(args.registry)
    output = resolve(args.output)
    rendered = json.dumps(
        build_claim_audit_report(registry, repository_root=ROOT),
        indent=2,
    ) + "\n"
    if args.verify:
        if not output.is_file() or output.read_text() != rendered:
            raise ValueError(f"public claim audit differs: {output}")
        print(json.dumps({"status": "VERIFIED", "output": str(output)}, indent=2))
        return
    if output.exists() and not args.force:
        raise FileExistsError(f"public claim audit is immutable: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    print(json.dumps({"status": "CREATED", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
