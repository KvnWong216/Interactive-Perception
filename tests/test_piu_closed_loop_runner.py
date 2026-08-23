from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_closed_loop_dry_run_exposes_complete_first_decision_chain(
    tmp_path: Path,
) -> None:
    candidate_set = tmp_path / "candidates.jsonl"
    candidate_set.write_text(
        json.dumps(
            {
                "schema_version": "piu.public-candidate-set.v1",
                "sample_id": "episode",
                "initial_state_group": "group",
                "split": "development",
                "candidates": [{"candidate_id": "open_drawer", "primitive": "OPEN"}],
            }
        )
        + "\n"
    )
    files = {}
    for name in (
        "binder_checkpoint",
        "binder_training_report",
        "binder_calibration",
        "effect_checkpoint",
        "effect_training_report",
        "effect_calibration",
    ):
        path = tmp_path / name
        path.write_text("placeholder")
        files[name] = path
    command = [
        sys.executable,
        str(ROOT / "scripts/pipeline/run_piu_closed_loop.py"),
        "--scenario-config",
        str(ROOT / "configs/scenarios/original_drawer.yaml"),
        "--candidate-set",
        str(candidate_set),
        "--initial-sample-id",
        "episode",
        "--seed",
        "7",
        "--binder-checkpoint",
        str(files["binder_checkpoint"]),
        "--binder-training-report",
        str(files["binder_training_report"]),
        "--binder-calibration",
        str(files["binder_calibration"]),
        "--effect-checkpoint",
        str(files["effect_checkpoint"]),
        "--effect-training-report",
        str(files["effect_training_report"]),
        "--effect-calibration",
        str(files["effect_calibration"]),
        "--host",
        "pi05.example.internal",
        "--output-dir",
        str(tmp_path / "episode_output"),
        "--dry-run",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)
    assert plan["schema_version"] == "piu.closed-loop-plan.v1"
    assert plan["local_pi05_loaded"] is False
    assert len(plan["first_decision_commands"]) == 6
    scripts = [Path(row[1]).name for row in plan["first_decision_commands"]]
    assert scripts == [
        "capture_piu_initial_observation.py",
        "extract_piu_spatial_prefix_features_remote.py",
        "predict_piu_target_binding_online.py",
        "update_piu_temporal_memory.py",
        "build_piu_controller_public_states.py",
        "run_piu_calibrated_controller.py",
    ]
    assert not (tmp_path / "episode_output").exists()

    uncalibrated_command = []
    skip_next = False
    for item in command:
        if skip_next:
            skip_next = False
            continue
        if item in {"--binder-calibration", "--effect-calibration"}:
            skip_next = True
            continue
        uncalibrated_command.append(item)
    uncalibrated_command.extend(["--method-id", "B3"])
    uncalibrated = subprocess.run(
        uncalibrated_command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    uncalibrated_plan = json.loads(uncalibrated.stdout)
    uncalibrated_scripts = [
        Path(row[1]).name for row in uncalibrated_plan["first_decision_commands"]
    ]
    assert uncalibrated_scripts[-1] == "run_piu_uncalibrated_ablation_controller.py"
    assert "update_piu_temporal_memory.py" not in uncalibrated_scripts
