#!/usr/bin/env python3
"""Build or byte-verify immutable PIU paper SVG figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.paper_figures import render_figures


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiments/piu_paper_figures_v1.yaml",
    )
    parser.add_argument("--method-output", type=Path)
    parser.add_argument("--evidence-output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text())
    method_path = resolve(
        args.method_output or Path(config["outputs"]["method"])
    )
    evidence_path = resolve(
        args.evidence_output or Path(config["outputs"]["evidence_boundary"])
    )
    if method_path.resolve() == evidence_path.resolve():
        raise ValueError("paper figure outputs must be distinct")
    rendered = render_figures(config_path, repository_root=ROOT)
    paths = {"method": method_path, "evidence_boundary": evidence_path}
    if args.verify:
        for name, path in paths.items():
            if not path.is_file() or path.read_text() != rendered[name]:
                raise ValueError(f"generated paper figure differs: {path}")
        print(json.dumps({"status": "VERIFIED", "outputs": {name: str(path) for name, path in paths.items()}}, indent=2))
        return
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "paper figure outputs are immutable: "
            + ", ".join(str(path) for path in existing)
        )
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered[name])
    print(json.dumps({"status": "CREATED", "outputs": {name: str(path) for name, path in paths.items()}}, indent=2))


if __name__ == "__main__":
    main()
