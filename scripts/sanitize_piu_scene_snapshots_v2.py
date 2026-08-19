#!/usr/bin/env python3
"""Derive oracle-clean V3 indices from V1 RGB snapshots without rerendering.

The RGB files and public robot state in V1 were captured before any evaluator
replay and are independent of ``visual_queries``.  V1 is retired because its
index copied the teacher ``interaction_target`` into that policy metadata.  This
script verifies the immutable visual evidence, removes that field from policy
inputs, and moves it to the physically separate teacher-label index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "benchmarks/piu_v1/scene_disjoint_protocol.yaml",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "results/audits/piu_v1_policy_query_leak_audit.json",
    )
    parser.add_argument(
        "--split",
        choices=("prototype_train", "conformal_calibration"),
        required=True,
    )
    args = parser.parse_args()
    for name in ("protocol", "audit"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    stem = {
        "prototype_train": "prototype_train",
        "conformal_calibration": "conformal_calibration",
    }[args.split]
    source = ROOT / f"data/piu_v1/{stem}_v1.jsonl"
    source_labels = ROOT / f"data/piu_v1/labels/{stem}_v1.jsonl"
    output = ROOT / f"data/piu_v1/{stem}_v3.jsonl"
    output_labels = ROOT / f"data/piu_v1/labels/{stem}_v3.jsonl"
    output_manifest = output.with_suffix(".manifest.json")
    label_manifest = output_labels.with_suffix(".manifest.json")
    for path in (output, output_labels, output_manifest, label_manifest):
        if path.exists():
            raise FileExistsError(f"immutable V3 output already exists: {path}")

    protocol = yaml.safe_load(args.protocol.read_text())
    scenarios = {
        item["id"]: item
        for item in protocol["splits"][args.split]["scenarios"]
    }
    rows = read_jsonl(source)
    labels = {row["sample_id"]: row for row in read_jsonl(source_labels)}
    if len(rows) != len(labels):
        raise ValueError("V1 public and label row counts differ")

    clean_rows: list[dict[str, Any]] = []
    clean_labels: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        scenario = scenarios[str(row["scenario_id"])]
        if row.get("online_oracle_inputs"):
            raise ValueError(f"V1 row has online oracle inputs: {sample_id}")
        for view, relative in row["policy_inputs"]["image_paths"].items():
            path = ROOT / relative
            if digest(path) != row["policy_inputs"]["image_sha256"][view]:
                raise ValueError(f"image hash mismatch: {sample_id} {view}")
        expected_queries = [
            scenario["target"],
            scenario["destination"],
            scenario.get("interaction_target", "object"),
        ]
        if row["visual_queries"] != expected_queries:
            raise ValueError(f"unexpected V1 query contract: {sample_id}")
        sanitized = dict(row)
        sanitized["schema_version"] = "interaction-uncertainty.piu-scene-snapshot.v3"
        sanitized["visual_queries"] = [scenario["target"], scenario["destination"]]
        sanitized["candidate_actions"] = ["DIRECT_ACT", "OPEN_TO_INSPECT", "ABSTAIN"]
        sanitized["sanitization"] = {
            "operation": "remove teacher interaction_target from policy metadata",
            "rgb_and_public_state_unchanged": True,
            "source_v1_sha256": digest(source),
        }
        clean_rows.append(sanitized)

        teacher = dict(labels[sample_id])
        teacher["schema_version"] = "interaction-uncertainty.piu-scene-label.v3"
        teacher["interaction_target"] = scenario.get("interaction_target")
        clean_labels.append(teacher)

    write_jsonl(output, clean_rows)
    write_jsonl(output_labels, clean_labels)
    common = {
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "protocol": str(args.protocol.relative_to(ROOT)),
        "protocol_sha256": digest(args.protocol),
        "policy_query_leak_audit": str(args.audit.relative_to(ROOT)),
        "policy_query_leak_audit_sha256": digest(args.audit),
        "split": args.split,
        "seed_block": protocol["splits"][args.split]["seed_block"],
        "samples": len(clean_rows),
        "source_v1": str(source.relative_to(ROOT)),
        "source_v1_sha256": digest(source),
        "visual_evidence_reused": True,
        "visual_evidence_independent_of_queries": True,
    }
    input_report = {
        "schema_version": "interaction-uncertainty.piu-scene-snapshot-manifest.v3",
        **common,
        "dataset": str(output.relative_to(ROOT)),
        "dataset_sha256": digest(output),
        "label_file": str(output_labels.relative_to(ROOT)),
        "label_file_in_policy_index": False,
        "policy_inputs": ["agentview RGB", "wrist RGB", "public robot state", "full prompt"],
        "policy_visual_queries": ["prompt target", "prompt destination"],
        "online_oracle_inputs": [],
        "class_counts_hidden_from_policy": dict(
            Counter(row["target_location"] for row in clean_labels)
        ),
    }
    label_report = {
        "schema_version": "interaction-uncertainty.piu-scene-label-manifest.v3",
        **common,
        "labels": str(output_labels.relative_to(ROOT)),
        "labels_sha256": digest(output_labels),
        "source_dataset": str(output.relative_to(ROOT)),
        "source_dataset_sha256": digest(output),
        "evaluator_only_fields": [
            "instance segmentation",
            "target semantic ID",
            "interaction target",
        ],
    }
    output_manifest.write_text(json.dumps(input_report, indent=2) + "\n")
    label_manifest.write_text(json.dumps(label_report, indent=2) + "\n")
    print(json.dumps(input_report, indent=2))


if __name__ == "__main__":
    main()
