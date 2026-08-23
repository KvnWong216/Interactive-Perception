from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from piu_test_artifacts import write_formal_primitive_certificate

ROOT = Path(__file__).resolve().parents[1]


def test_dispatch_plan_uses_external_pi05_and_frozen_budget(tmp_path: Path) -> None:
    report = tmp_path / "controller.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "piu.calibrated-controller-report.v1",
                "evaluator_labels_loaded": False,
                "decisions": [
                    {
                        "sample_id": "sample",
                        "initial_state_group": "synthetic-group",
                        "decision_kind": "EXECUTE",
                        "selected_candidate_id": "pick_butter",
                        "selected_candidate_primitive": "PICK",
                        "selected_candidate": {
                            "candidate_id": "pick_butter",
                            "primitive": "PICK",
                            "target": "butter",
                        },
                        "public_candidates": [
                            {
                                "candidate_id": "pick_butter",
                                "primitive": "PICK",
                                "target": "butter",
                            }
                        ],
                        "reason": "singleton sets",
                        "structured_pi05_subtask": (
                            "Pick up the butter at agentview normalized box "
                            "x=[0.1000,0.2000], y=[0.3000,0.4000]."
                        ),
                        "spatial_references": [
                            {
                                "camera": "agentview",
                                "selected_patch_indices": [0],
                                "x_interval": [0.1, 0.2],
                                "y_interval": [0.3, 0.4],
                            }
                        ],
                    }
                ],
            }
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/pipeline/execute_piu_controller_decision.py"),
            "--controller-report",
            str(report),
            "--sample-id",
            "sample",
            "--scenario-config",
            "configs/scenarios/original_drawer.yaml",
            "--seed",
            "3000",
            "--host",
            "127.0.0.1",
            "--run-dir",
            str(tmp_path / "run"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)
    assert plan["primitive"] == "PICK"
    assert plan["external_server_only"] is True
    assert plan["primitive_formally_qualified"] is False
    assert plan["physical_dispatch_allowed"] is False
    assert "--external-server" in plan["command"]
    assert plan["command"][plan["command"].index("--steps") + 1] == "400"
    assert "--preserve-grasp" in plan["command"]
    assert not (tmp_path / "run").exists()

    certificate = write_formal_primitive_certificate(
        tmp_path / "qualification",
        candidate_id="pick_butter",
        primitive="PICK",
    )
    qualified = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/pipeline/execute_piu_controller_decision.py"),
            "--controller-report",
            str(report),
            "--sample-id",
            "sample",
            "--scenario-config",
            "configs/scenarios/original_drawer.yaml",
            "--primitive-qualification",
            str(certificate),
            "--seed",
            "3000",
            "--host",
            "127.0.0.1",
            "--run-dir",
            str(tmp_path / "qualified_run"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    qualified_plan = json.loads(qualified.stdout)
    assert qualified_plan["primitive_formally_qualified"] is True
    assert qualified_plan["physical_dispatch_allowed"] is True

    mismatched = json.loads(report.read_text())
    decision = mismatched["decisions"][0]
    decision["selected_candidate"]["target"] = "cream cheese"
    decision["public_candidates"][0]["target"] = "cream cheese"
    decision["structured_pi05_subtask"] = (
        "Pick up the cream cheese at agentview normalized box "
        "x=[0.1000,0.2000], y=[0.3000,0.4000]."
    )
    report.write_text(json.dumps(mismatched))
    rejected = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/pipeline/execute_piu_controller_decision.py"),
            "--controller-report",
            str(report),
            "--sample-id",
            "sample",
            "--scenario-config",
            "configs/scenarios/original_drawer.yaml",
            "--primitive-qualification",
            str(certificate),
            "--seed",
            "3000",
            "--host",
            "127.0.0.1",
            "--run-dir",
            str(tmp_path / "mismatched_run"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "exact executor contract" in rejected.stderr


def test_live_dispatch_refuses_an_unqualified_primitive(tmp_path: Path) -> None:
    report = tmp_path / "controller.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "piu.calibrated-controller-report.v1",
                "evaluator_labels_loaded": False,
                "decisions": [
                    {
                        "sample_id": "sample",
                        "initial_state_group": "synthetic-group",
                        "decision_kind": "INTERACT",
                        "selected_candidate_id": "open_middle_drawer",
                        "selected_candidate_primitive": "OPEN",
                        "selected_candidate": {
                            "candidate_id": "open_middle_drawer",
                            "primitive": "OPEN",
                            "target": "middle drawer",
                        },
                        "public_candidates": [
                            {
                                "candidate_id": "open_middle_drawer",
                                "primitive": "OPEN",
                                "target": "middle drawer",
                            }
                        ],
                        "reason": "singleton sets",
                        "structured_pi05_subtask": "Open the middle drawer.",
                        "spatial_references": [],
                    }
                ],
            }
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/pipeline/execute_piu_controller_decision.py"),
            "--controller-report",
            str(report),
            "--sample-id",
            "sample",
            "--scenario-config",
            "configs/scenarios/original_drawer.yaml",
            "--seed",
            "3000",
            "--host",
            "127.0.0.1",
            "--run-dir",
            str(tmp_path / "run"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "formally qualified primitive certificate" in completed.stderr
    assert not (tmp_path / "run").exists()
