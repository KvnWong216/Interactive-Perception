#!/usr/bin/env python3
"""Audit PIU object proposals against physically separate teacher masks/roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def keyed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["sample_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate sample_id")
    return result


def tokens(value: str) -> set[str]:
    return {
        token
        for token in value.lower().replace("_", " ").split()
        if len(token) > 2 and token not in {"the", "with", "into", "layer"}
    }


def rle_decode(value: dict[str, Any]) -> np.ndarray:
    values: list[int] = []
    current = int(value.get("starts_with", 0))
    for length in value["counts"]:
        values.extend([current] * int(length))
        current = 1 - current
    return np.asarray(values, dtype=bool).reshape(value["size"])


def overlap_fraction(proposal: np.ndarray, target: np.ndarray) -> float:
    denominator = int(target.sum())
    if denominator == 0:
        return 0.0
    return float(np.logical_and(proposal, target).sum() / denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visible-overlap-threshold", type=float, default=0.25)
    args = parser.parse_args()
    for name in ("snapshots", "labels", "scenes", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output.exists():
        raise FileExistsError(f"immutable audit output already exists: {args.output}")

    snapshots = keyed(read_jsonl(args.snapshots))
    labels = keyed(read_jsonl(args.labels))
    scenes = keyed(read_jsonl(args.scenes))
    if set(snapshots) != set(labels) or set(snapshots) != set(scenes):
        raise ValueError("snapshot, label, and scene IDs differ")

    rows = []
    for sample_id, snapshot in snapshots.items():
        label = labels[sample_id]
        scene = scenes[sample_id]
        teacher = str(label.get("interaction_target") or "").strip().lower()
        public_queries = {str(value).strip().lower() for value in snapshot["visual_queries"]}
        frontend_queries = {str(value).strip().lower() for value in scene["visual_queries"]}
        query_leak = bool(teacher and (teacher in public_queries or teacher in frontend_queries))
        maximum_target_overlap = 0.0
        interaction_nodes = 0
        teacher_terms = tokens(teacher)
        target_masks = {
            "agentview": rle_decode(label["target_mask_policy_resolution_rle"]["agentview"]),
            "wrist": rle_decode(
                label["target_mask_policy_resolution_rle"]["robot0_eye_in_hand"]
            ),
        }
        for node in scene["objects"]:
            label_terms = set().union(
                *(tokens(value) for value in node["label_candidates"])
            )
            if teacher_terms & label_terms:
                interaction_nodes += 1
            if label["target_location"] == "visible_workspace":
                maximum_target_overlap = max(
                    maximum_target_overlap,
                    overlap_fraction(rle_decode(node["mask_rle"]), target_masks[node["view"]]),
                )
        if label["target_location"] == "visible_workspace":
            truth_node_proposed = maximum_target_overlap >= args.visible_overlap_threshold
        else:
            truth_node_proposed = interaction_nodes > 0
        rows.append(
            {
                "sample_id": sample_id,
                "scenario_id": snapshot["scenario_id"],
                "target_location": label["target_location"],
                "query_leak": query_leak,
                "objects": len(scene["objects"]),
                "maximum_target_overlap": maximum_target_overlap,
                "teacher_interaction_nodes": interaction_nodes,
                "truth_node_proposed": truth_node_proposed,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scenario_id"])].append(row)
    report = {
        "schema_version": "interaction-uncertainty.piu-object-frontend-audit.v1",
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "sources": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
            for name, path in (
                ("snapshots", args.snapshots),
                ("labels", args.labels),
                ("scenes", args.scenes),
            )
        },
        "samples": len(rows),
        "class_counts": dict(Counter(row["target_location"] for row in rows)),
        "query_leaks": sum(row["query_leak"] for row in rows),
        "truth_node_proposed": sum(row["truth_node_proposed"] for row in rows),
        "truth_node_proposal_rate": float(np.mean([row["truth_node_proposed"] for row in rows])),
        "mean_objects": float(np.mean([row["objects"] for row in rows])),
        "per_scenario": {
            scenario: {
                "samples": len(values),
                "query_leaks": sum(row["query_leak"] for row in values),
                "truth_node_proposal_rate": float(
                    np.mean([row["truth_node_proposed"] for row in values])
                ),
                "mean_maximum_target_overlap": float(
                    np.mean([row["maximum_target_overlap"] for row in values])
                ),
                "mean_teacher_interaction_nodes": float(
                    np.mean([row["teacher_interaction_nodes"] for row in values])
                ),
            }
            for scenario, values in grouped.items()
        },
        "rows": rows,
        "online_oracle_inputs": [],
        "teacher_use": "offline proposal recall audit only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "samples": report["samples"],
                "query_leaks": report["query_leaks"],
                "truth_node_proposal_rate": report["truth_node_proposal_rate"],
                "per_scenario": report["per_scenario"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
