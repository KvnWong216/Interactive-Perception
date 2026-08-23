#!/usr/bin/env python3
"""Export one prospective public transition from a qualified PIU dispatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.contracts import PublicTransition
from piu.primitive_registry import load_primitive_qualification_certificate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def keyframe(report: dict[str, Any], name: str) -> dict[str, Any]:
    rows = [row for row in report["controller"]["keyframes"] if row["name"] == name]
    if len(rows) != 1:
        raise ValueError(f"execution report needs exactly one {name} keyframe")
    return rows[0]


def public_observation(frame: dict[str, Any]) -> dict[str, Any]:
    images = {}
    for camera, relative in frame["image_paths"].items():
        path = resolve(Path(relative))
        expected = str(frame["image_sha256"][camera])
        if sha256(path) != expected:
            raise ValueError(f"public keyframe image hash differs: {path}")
        pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        pixel_digest = hashlib.sha256(
            np.ascontiguousarray(pixels).tobytes()
        ).hexdigest()
        declared_pixel = frame.get("image_pixel_sha256", {}).get(camera)
        if declared_pixel is not None and declared_pixel != pixel_digest:
            raise ValueError(f"public keyframe pixel hash differs: {path}")
        images[str(camera)] = {
            "path": portable(path),
            "sha256": expected,
            "pixel_sha256": pixel_digest,
        }
    return {
        "images": images,
        "public_robot_state": [float(value) for value in frame["public_robot_state"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch-receipt", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--initial-state-group", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "development", "calibration", "sealed_test"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = resolve(args.dispatch_receipt)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("prospective public transitions are immutable")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema_version") != "piu.executor-dispatch.v1":
        raise ValueError("unsupported PIU dispatch receipt")
    if receipt.get("physical_action_dispatched") is not True:
        raise ValueError("a non-physical decision cannot create an executed transition")
    if receipt.get("evaluator_fields_copied") != []:
        raise ValueError("dispatch receipt crosses the evaluator firewall")
    selected = receipt.get("selected_candidate")
    candidates = receipt.get("public_candidates")
    if not isinstance(selected, dict) or not isinstance(candidates, list):
        raise TypeError("dispatch receipt lacks public candidate payloads")
    matches = [
        row
        for row in candidates
        if row.get("candidate_id") == selected.get("candidate_id")
        and str(row.get("primitive", "")).upper()
        == str(selected.get("primitive", "")).upper()
    ]
    if len(matches) != 1:
        raise ValueError("selected candidate is not unique in public candidate set")
    report_path = resolve(Path(receipt["execution_report"]["path"]))
    if sha256(report_path) != receipt["execution_report"]["sha256"]:
        raise ValueError("qualified execution report differs from dispatch receipt")
    report = json.loads(report_path.read_text())
    if report.get("controller", {}).get("online_oracle_inputs"):
        raise ValueError("public transition source used online oracle inputs")
    qualification_path = resolve(Path(receipt["primitive_qualification"]["path"]))
    if sha256(qualification_path) != receipt["primitive_qualification"]["sha256"]:
        raise ValueError("primitive certificate differs from dispatch receipt")
    qualification = load_primitive_qualification_certificate(
        qualification_path, repository_root=ROOT
    )
    if qualification.get("status") != "FORMALLY_QUALIFIED":
        raise ValueError("public transition source primitive was not qualified")
    if qualification.get("candidate_id") != selected.get("candidate_id"):
        raise ValueError("primitive certificate candidate differs from dispatch")
    controller_path = resolve(Path(receipt["controller_report"]["path"]))
    if sha256(controller_path) != receipt["controller_report"]["sha256"]:
        raise ValueError("controller report differs from dispatch receipt")
    history_path = resolve(Path(report["controller"]["action_history"]))
    transition = {
        "schema_version": "piu.public-transition.v1",
        "sample_id": " ".join(args.sample_id.split()),
        "initial_state_group": " ".join(args.initial_state_group.split()),
        "split": args.split,
        "prompt": str(receipt["task_prompt"]),
        "observations": {
            "pre_interaction": public_observation(keyframe(report, "00_before")),
            "post_interaction": public_observation(
                keyframe(report, "05_returned_home")
            ),
        },
        "public_action_history": {
            "last_executed_candidate": selected,
            "low_level_actions": {
                "path": portable(history_path),
                "sha256": sha256(history_path),
            },
            "dispatch_receipt": {
                "path": portable(receipt_path),
                "sha256": sha256(receipt_path),
            },
        },
        "candidate_actions": candidates,
        "online_oracle_inputs": [],
    }
    PublicTransition.from_mapping(transition)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(transition, sort_keys=True) + "\n")
    print(json.dumps({"output": portable(output), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
