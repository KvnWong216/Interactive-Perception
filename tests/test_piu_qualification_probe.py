from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from piu.primitive_registry import load_qualification_controller_decision
from piu_test_artifacts import write_formal_primitive_certificate

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/evaluation/build_piu_primitive_qualification_probe.py"
SCHEDULE_BUILDER = (
    ROOT / "scripts/evaluation/build_piu_primitive_qualification_schedule.py"
)


def _qualification_group(directory: Path, candidate_id: str) -> str:
    split = yaml.safe_load((directory / f"{candidate_id}_split.yaml").read_text())
    return next(
        row["initial_state_group"]
        for row in split["assignments"]
        if row["split_role"] == "primitive_qualification"
    )


def _qualification_groups(directory: Path, candidate_id: str) -> list[str]:
    split = yaml.safe_load((directory / f"{candidate_id}_split.yaml").read_text())
    return [
        row["initial_state_group"]
        for row in split["assignments"]
        if row["split_role"] == "primitive_qualification"
    ]


def test_open_probe_breaks_pretraining_cycle_without_method_claim(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "qualification"
    write_formal_primitive_certificate(
        fixture, candidate_id="open_middle_drawer", primitive="OPEN"
    )
    plan = fixture / "open_middle_drawer_plan.json"
    groups = _qualification_groups(fixture, "open_middle_drawer")
    group = groups[0]
    candidate = {
        "candidate_id": "open_middle_drawer",
        "primitive": "OPEN",
        "target": "middle drawer",
        "purpose": "inspect inside",
        "required_capability": "OPEN",
    }
    candidate_set = tmp_path / "candidates.jsonl"
    candidate_set.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": "piu.public-candidate-set.v1",
                    "sample_id": f"open-probe::{item}",
                    "initial_state_group": item,
                    "split": "primitive_qualification",
                    "public_inputs_only": True,
                    "online_oracle_inputs": [],
                    "candidates": [candidate],
                }
            )
            + "\n"
            for item in groups
        )
    )
    output = tmp_path / "probe.json"
    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--plan",
            str(plan),
            "--candidate-set",
            str(candidate_set),
            "--sample-id",
            f"open-probe::{group}",
            "--initial-state-group",
            group,
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["selection_source"] == (
        "preregistered_executor_probe_not_method_decision"
    )
    assert report["trained_model_loaded"] is False
    assert report["calibration_loaded"] is False
    assert report["paper_method_selection_claim_allowed"] is False
    assert report["online_oracle_inputs"] == []
    loaded = load_qualification_controller_decision(
        output,
        candidate_id="open_middle_drawer",
        primitive="OPEN",
        initial_state_group=group,
        repository_root=ROOT,
    )
    assert loaded["spatial_reference_mode"] == "none"
    assert loaded["structured_subtask"] == "Open the middle drawer."

    widened = json.loads(output.read_text())
    widened["decisions"].append(dict(widened["decisions"][0]))
    widened_path = tmp_path / "widened_probe.json"
    widened_path.write_text(json.dumps(widened, indent=2) + "\n")
    with pytest.raises(ValueError, match="one decision"):
        load_qualification_controller_decision(
            widened_path,
            candidate_id="open_middle_drawer",
            primitive="OPEN",
            initial_state_group=group,
            repository_root=ROOT,
        )

    report_paths = {group: output}
    for item in groups[1:]:
        copied = json.loads(output.read_text())
        copied["decisions"][0]["sample_id"] = f"open-probe::{item}"
        copied["decisions"][0]["initial_state_group"] = item
        path = tmp_path / f"{item}_probe.json"
        path.write_text(json.dumps(copied, indent=2) + "\n")
        report_paths[item] = path
    schedule = tmp_path / "probe_schedule.json"
    command = [
        sys.executable,
        str(SCHEDULE_BUILDER),
        "--plan",
        str(plan),
        "--split-manifest",
        str(fixture / "open_middle_drawer_split.yaml"),
        "--run-root",
        str(tmp_path / "probe_runs"),
        "--output",
        str(schedule),
    ]
    for item in groups:
        command.extend(
            [
                "--state",
                item,
                str(fixture / f"{item}.npz"),
                "--controller-report",
                item,
                str(report_paths[item]),
            ]
        )
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    frozen = json.loads(schedule.read_text())
    assert len(frozen["entries"]) == len(groups)
    assert frozen["candidate_contract"]["spatial_reference_mode"] == "none"

    candidate_set.write_text(candidate_set.read_text() + "\n")
    with pytest.raises(ValueError, match="artifact differs"):
        load_qualification_controller_decision(
            output,
            candidate_id="open_middle_drawer",
            primitive="OPEN",
            initial_state_group=group,
            repository_root=ROOT,
        )


def test_model_free_probe_cannot_bypass_pick_spatial_binding(tmp_path: Path) -> None:
    fixture = tmp_path / "qualification"
    write_formal_primitive_certificate(
        fixture, candidate_id="pick_butter", primitive="PICK"
    )
    group = _qualification_group(fixture, "pick_butter")
    candidate_set = tmp_path / "candidates.jsonl"
    candidate_set.write_text(
        json.dumps(
            {
                "schema_version": "piu.public-candidate-set.v1",
                "sample_id": "pick-probe",
                "initial_state_group": group,
                "split": "primitive_qualification",
                "public_inputs_only": True,
                "online_oracle_inputs": [],
                "candidates": [
                    {
                        "candidate_id": "pick_butter",
                        "primitive": "PICK",
                        "target": "butter",
                    }
                ],
            }
        )
        + "\n"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--plan",
            str(fixture / "pick_butter_plan.json"),
            "--candidate-set",
            str(candidate_set),
            "--sample-id",
            "pick-probe",
            "--initial-state-group",
            group,
            "--output",
            str(tmp_path / "probe.json"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "OPEN-only" in completed.stderr
    assert not (tmp_path / "probe.json").exists()
