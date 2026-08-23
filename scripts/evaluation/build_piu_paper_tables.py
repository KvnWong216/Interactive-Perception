#!/usr/bin/env python3
"""Build or verify immutable PIU paper tables from admissible artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.paper_artifacts import build_evidence_tables, render_markdown


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiments/piu_robustness_reporting_v1.yaml",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="compare existing outputs byte-for-byte without modifying them",
    )
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text())
    if args.output_json is None:
        args.output_json = Path(config["outputs"]["json"])
    if args.output_markdown is None:
        args.output_markdown = Path(config["outputs"]["markdown"])
    output_json = resolve(args.output_json)
    output_markdown = resolve(args.output_markdown)
    if output_json.resolve() == output_markdown.resolve():
        raise ValueError("JSON and Markdown outputs must be different files")
    value = build_evidence_tables(config_path, repository_root=ROOT)
    json_text = json.dumps(value, indent=2) + "\n"
    markdown_text = render_markdown(value)
    if args.verify:
        for path, expected in (
            (output_json, json_text),
            (output_markdown, markdown_text),
        ):
            if not path.exists() or path.read_text() != expected:
                raise ValueError(f"generated paper artifact differs: {path}")
        print(
            json.dumps(
                {
                    "status": "VERIFIED",
                    "output_json": str(output_json),
                    "output_markdown": str(output_markdown),
                    "main_table_evidence_complete": value[
                        "main_table_evidence_complete"
                    ],
                },
                indent=2,
            )
        )
        return
    existing = [path for path in (output_json, output_markdown) if path.exists()]
    if existing:
        raise FileExistsError(
            "paper table outputs are immutable: "
            + ", ".join(str(path) for path in existing)
        )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json_text)
    output_markdown.write_text(markdown_text)
    print(
        json.dumps(
            {
                "status": "CREATED",
                "output_json": str(output_json),
                "output_markdown": str(output_markdown),
                "main_table_evidence_complete": value[
                    "main_table_evidence_complete"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
