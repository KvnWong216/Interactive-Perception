#!/usr/bin/env python3
"""Build a leakage-separated retrospective drawer transition dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.contracts import (
    EvaluatorSidecar,
    PublicTransition,
    validate_public_sidecar_pair,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def keyframe(report: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [
        row for row in report["controller"]["keyframes"] if row["name"] == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one keyframe {name!r}")
    return matches[0]


def public_observation(frame: dict[str, Any]) -> dict[str, Any]:
    images = {}
    for camera, relative in frame["image_paths"].items():
        path = ROOT / relative
        expected = frame["image_sha256"][camera]
        if sha256(path) != expected:
            raise ValueError(f"image hash mismatch: {path}")
        images[camera] = {"path": relative, "sha256": expected}
    return {
        "images": images,
        "public_robot_state": [float(value) for value in frame["public_robot_state"]],
    }


def report_reference(path: Path) -> dict[str, str]:
    return {"path": portable(path), "sha256": sha256(path)}


def build_rows(summary_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = read_json(summary_path)
    if (
        summary.get("schema_version")
        != "calibrated-interaction.paper-cycle-summary.v2"
    ):
        raise ValueError("unexpected paper-cycle schema")
    public_rows: list[dict[str, Any]] = []
    evaluator_rows: list[dict[str, Any]] = []
    for retained in summary["per_seed"]:
        seed = int(retained["seed"])
        group = f"original_drawer_seed{seed}"
        sample = f"{group}_open_then_direct"
        open_path = ROOT / retained["reports"]["open_closed_drawer"]["path"]
        direct_path = ROOT / retained["reports"]["direct_after_actual_open"]["path"]
        if sha256(open_path) != retained["reports"]["open_closed_drawer"]["sha256"]:
            raise ValueError(f"retained OPEN report changed: {open_path}")
        if sha256(direct_path) != retained["reports"]["direct_after_actual_open"]["sha256"]:
            raise ValueError(f"retained DIRECT report changed: {direct_path}")
        open_report = read_json(open_path)
        direct_report = read_json(direct_path)
        if open_report["seed"] != seed or direct_report["seed"] != seed:
            raise ValueError("seed/report mismatch")
        if open_report["controller"].get("online_oracle_inputs"):
            raise ValueError("OPEN source unexpectedly used online oracle inputs")
        if direct_report["controller"].get("online_oracle_inputs"):
            raise ValueError("DIRECT source unexpectedly used online oracle inputs")

        history_path = ROOT / open_report["controller"]["action_history"]
        public = {
            "schema_version": "piu.public-transition.v1",
            "sample_id": sample,
            "initial_state_group": group,
            "split": "retrospective_development",
            "prompt": "Place the butter in the basket",
            "observations": {
                "pre_interaction": public_observation(
                    keyframe(open_report, "00_before")
                ),
                "post_interaction": public_observation(
                    keyframe(direct_report, "00_before")
                ),
            },
            "public_action_history": {
                "path": portable(history_path),
                "sha256": sha256(history_path),
            },
            "candidate_actions": [
                {
                    "candidate_id": "open_middle_drawer",
                    "primitive": "OPEN",
                    "target": "middle drawer",
                    "purpose": "inspect for task-relevant evidence",
                },
                {
                    "candidate_id": "direct_butter_to_basket",
                    "primitive": "DIRECT",
                    "target": "butter",
                    "reference": "basket",
                },
            ],
            "online_oracle_inputs": [],
        }

        open_evaluator = open_report["evaluator"]
        direct_evaluator = direct_report["evaluator"]
        target_name = str(direct_evaluator["target_object"])
        wrong_names = [
            name for name in direct_evaluator["objects"] if name != target_name
        ]
        pre_pixels = {
            str(camera): int(count)
            for camera, count in open_evaluator["target_visibility_pixels"][
                "initial"
            ].items()
        }
        post_pixels = {
            str(camera): int(count)
            for camera, count in direct_evaluator["target_visibility_pixels"][
                "initial"
            ].items()
        }
        target_object = direct_evaluator["objects"][target_name]
        sidecar = {
            "schema_version": "piu.evaluator-sidecar.v1",
            "sample_id": sample,
            "initial_state_group": group,
            "split": "retrospective_development",
            "sufficiency_decision_correct": None,
            "interaction_selection_correct": None,
            "primitive_execution_success": bool(
                retained["open_closed_drawer"]["drawer_open"]
            ),
            "target_visible_pixels_pre": pre_pixels,
            "target_visible_pixels_post": post_pixels,
            "information_acquired": max(pre_pixels.values()) == 0
            and max(post_pixels.values()) > 0,
            "target_identity_resolved": None,
            "target_grasp_contact": target_object["grasp_contact_steps"] > 0,
            "wrong_object_grasp_contact": any(
                direct_evaluator["objects"][name]["grasp_contact_steps"] > 0
                for name in wrong_names
            ),
            "target_maximum_lift_m": float(target_object["maximum_lift_m"]),
            "target_destination_final": bool(
                retained["direct_after_actual_open"]["target_destination_final"]
            ),
            "task_success": bool(
                retained["direct_after_actual_open"]["task_success"]
            ),
            "provenance": {
                "claim_scope": "RETROSPECTIVE_DEVELOPMENT_ONLY",
                "open_report": report_reference(open_path),
                "direct_report": report_reference(direct_path),
                "summary": report_reference(summary_path),
                "evaluator_only_fields": [
                    "instance segmentation pixel counts",
                    "articulated joint state",
                    "object contact",
                    "object lift",
                    "destination predicate",
                    "task predicate",
                ],
                "binding_label_available": False,
            },
        }
        parsed_public = PublicTransition.from_mapping(public)
        parsed_sidecar = EvaluatorSidecar.from_mapping(sidecar)
        validate_public_sidecar_pair(parsed_public, parsed_sidecar)
        public_rows.append(public)
        evaluator_rows.append(sidecar)
    return public_rows, evaluator_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "results/method/original_drawer_paper_cycle_v2.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/piu/drawer_binding_sprint_v1",
    )
    args = parser.parse_args()
    summary = args.summary if args.summary.is_absolute() else ROOT / args.summary
    output = (
        args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    )
    public_path = output / "public_transitions.jsonl"
    sidecar_path = output / "evaluator_sidecars.jsonl"
    manifest_path = output / "manifest.json"
    for path in (public_path, sidecar_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"dataset assets are immutable: {path}")
    public_rows, evaluator_rows = build_rows(summary)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(public_path, public_rows)
    write_jsonl(sidecar_path, evaluator_rows)
    manifest = {
        "schema_version": "piu.drawer-binding-dataset-manifest.v1",
        "claim_scope": "RETROSPECTIVE_DEVELOPMENT_ONLY",
        "scenario": "unchanged T01D hidden-butter drawer",
        "groups": len(public_rows),
        "splits": {"retrospective_development": len(public_rows)},
        "public_transitions": {
            "path": portable(public_path),
            "sha256": sha256(public_path),
        },
        "evaluator_sidecars": {
            "path": portable(sidecar_path),
            "sha256": sha256(sidecar_path),
        },
        "source_summary": report_reference(summary),
        "policy_evaluator_storage_separated": True,
        "binding_training_labels_available": False,
        "allowed_uses": [
            "pipeline validation",
            "descriptive six-stage failure decomposition",
            "external frozen-prefix extraction rehearsal",
        ],
        "forbidden_uses": [
            "formal method comparison",
            "calibration",
            "sealed evaluation",
            "target-binding supervision",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
