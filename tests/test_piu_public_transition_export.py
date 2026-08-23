from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from piu.contracts import PublicTransition

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_qualified_dispatch_exports_only_public_transition_fields(
    tmp_path: Path,
) -> None:
    images = {}
    for name in ("pre_agent", "pre_wrist", "post_agent", "post_wrist"):
        path = tmp_path / f"{name}.png"
        Image.fromarray(np.full((4, 4, 3), len(name), dtype=np.uint8)).save(path)
        images[name] = path
    history = tmp_path / "actions.json"
    history.write_text("[]\n")
    execution = tmp_path / "execution.json"
    execution.write_text(
        json.dumps(
            {
                "controller": {
                    "online_oracle_inputs": [],
                    "action_history": str(history),
                    "keyframes": [
                        {
                            "name": "00_before",
                            "image_paths": {
                                "agentview": str(images["pre_agent"]),
                                "wrist": str(images["pre_wrist"]),
                            },
                            "image_sha256": {
                                "agentview": _sha256(images["pre_agent"]),
                                "wrist": _sha256(images["pre_wrist"]),
                            },
                            "public_robot_state": [0.0] * 8,
                        },
                        {
                            "name": "05_returned_home",
                            "image_paths": {
                                "agentview": str(images["post_agent"]),
                                "wrist": str(images["post_wrist"]),
                            },
                            "image_sha256": {
                                "agentview": _sha256(images["post_agent"]),
                                "wrist": _sha256(images["post_wrist"]),
                            },
                            "public_robot_state": [0.1] * 8,
                        },
                    ],
                },
                "evaluator": {"target_pose": [1.0, 2.0, 3.0]},
            }
        )
    )
    controller = tmp_path / "controller.json"
    controller.write_text("{}\n")
    qualification = tmp_path / "qualification.json"
    qualification.write_text(
        json.dumps(
            {
                "schema_version": "piu.primitive-qualification-certificate.v1",
                "status": "FORMALLY_QUALIFIED",
                "candidate_id": "open_middle_drawer",
                "primitive": "OPEN",
            }
        )
    )
    candidate = {
        "candidate_id": "open_middle_drawer",
        "primitive": "OPEN",
        "target": "middle drawer",
    }
    receipt = tmp_path / "dispatch.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "piu.executor-dispatch.v1",
                "physical_action_dispatched": True,
                "task_prompt": "Place the butter in the basket",
                "selected_candidate": candidate,
                "public_candidates": [candidate],
                "evaluator_fields_copied": [],
                "execution_report": {
                    "path": str(execution),
                    "sha256": _sha256(execution),
                },
                "controller_report": {
                    "path": str(controller),
                    "sha256": _sha256(controller),
                },
                "primitive_qualification": {
                    "path": str(qualification),
                    "sha256": _sha256(qualification),
                },
            }
        )
    )
    output = tmp_path / "public.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/data/export_piu_public_transition.py"),
            "--dispatch-receipt",
            str(receipt),
            "--sample-id",
            "sample",
            "--initial-state-group",
            "group",
            "--split",
            "train",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    raw = json.loads(output.read_text())
    parsed = PublicTransition.from_mapping(raw)
    assert parsed.public_action_history["last_executed_candidate"] == candidate
    assert "evaluator" not in raw
    assert "target_pose" not in output.read_text()
    assert all(
        "pixel_sha256" in image
        for observation in raw["observations"].values()
        for image in observation["images"].values()
    )
