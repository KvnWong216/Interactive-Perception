from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from piu.oracle_formal import (
    analyze_oracle_formal_schedule,
    load_oracle_formal_group_receipt,
    load_oracle_formal_schedule,
)
from piu.policy_identity import load_checkpoint_identity
from piu_test_artifacts import (
    expected_server_metadata,
    write_formal_primitive_certificate,
)

ROOT = Path(__file__).resolve().parents[1]
STATE_BUILDER = ROOT / "scripts/evaluation/build_oracle_formal_initial_states.py"
SCHEDULE_BUILDER = ROOT / "scripts/evaluation/build_oracle_formal_schedule.py"
RUNNER = ROOT / "scripts/pipeline/run_oracle_formal_group.py"
CLOSER = ROOT / "scripts/evaluation/close_oracle_formal_group_failure.py"
ANALYZER = ROOT / "scripts/evaluation/analyze_oracle_formal_experiment.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference(path: Path) -> dict[str, str]:
    try:
        portable = str(path.resolve().relative_to(ROOT))
    except ValueError:
        portable = str(path.resolve())
    return {"path": portable, "sha256": _sha256(path)}


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _write_formal_report(
    *,
    path: Path,
    entry: dict,
    schedule: dict,
    config: dict,
    endpoint: dict,
    arm: str,
    source_state: Path,
    contact: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    asset = path.parent / "initial.bin"
    asset.write_bytes(f"paired-{entry['initial_state_group']}".encode())
    actions = path.parent / "actions.json"
    actions.write_text("[]\n")
    prompt = {
        "source_open": schedule["source_open_subtask"],
        "oracle_target_prompt": config["execution"]["prompt"],
        "raw_post_open_direct": "Place the butter in the basket",
    }[arm]
    role = {
        "source_open": "PIU_ORACLE_FORMAL_OPEN_SOURCE",
        "oracle_target_prompt": "PIU_ORACLE_FORMAL_ORACLE",
        "raw_post_open_direct": "PIU_ORACLE_FORMAL_BASELINE",
    }[arm]
    oracle = arm == "oracle_target_prompt"
    controller = {
        "server_mode": "external",
        "server_metadata": endpoint["identity"],
        "online_oracle_inputs": (
            ["declared target identity", "online instance segmentation"]
            if oracle
            else []
        ),
        "oracle_visual_prompt": None,
        "policy_calls": 1,
        "action_history": _portable(actions),
        "source_initial_state_transport": {
            "path": _portable(source_state),
            "sha256": _sha256(source_state),
            "state_key": entry["state_key"],
        },
        "keyframes": [
            {
                "name": "00_before",
                "image_paths": {"agentview": _portable(asset)},
                "image_sha256": {"agentview": _sha256(asset)},
            }
        ],
    }
    if arm == "source_open":
        final_state = Path(entry["expected_post_open_state"])
        controller["opaque_state_transport"] = _portable(final_state)
        controller["opaque_state_transport_sha256"] = _sha256(final_state)
    if oracle:
        controller["oracle_visual_prompt"] = {
            "style": schedule["selected_style"],
            "source_initial_state": {
                "path": _portable(source_state),
                "sha256": _sha256(source_state),
                "state_key": entry["state_key"],
            },
            "policy_call_audit": [
                {"visible_pixels": {"agentview": 1, "wrist": 0}}
            ],
            "keyframes": [
                {
                    "name": "00_first_policy_call",
                    "image_paths": {"agentview": _portable(asset)},
                    "image_sha256": {"agentview": _sha256(asset)},
                }
            ],
        }
    path.write_text(
        json.dumps(
            {
                "schema_version": "piu.semantic-option.v2",
                "claim_scope": (
                    "EVALUATOR_ONLY_ORACLE_UPPER_BOUND"
                    if oracle
                    else "PUBLIC_INPUT_EXECUTION"
                ),
                "seed": entry["simulator_seed"],
                "role": role,
                "prompt": prompt,
                "controller": controller,
                "evaluator": {
                    "target_in_destination_final": contact,
                    "task_success": contact,
                    "objects": {
                        config["execution"]["target_object"]: {
                            "grasp_contact_steps": int(contact),
                            "maximum_lift_m": 0.1 if contact else 0.0,
                        },
                        config["execution"]["wrong_object"]: {
                            "grasp_contact_steps": 0,
                            "maximum_lift_m": 0.0,
                        },
                    },
                },
            },
            indent=2,
        )
        + "\n"
    )


def _formal_fixture(tmp_path: Path) -> dict[str, Path]:
    certificate = write_formal_primitive_certificate(
        tmp_path / "qualification",
        candidate_id="open_middle_drawer",
        primitive="OPEN",
    )
    config = yaml.safe_load(
        (
            ROOT
            / "configs/experiments/original_drawer_oracle_target_prompt_pilot_v2.yaml"
        ).read_text()
    )
    config_path = tmp_path / "oracle_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    formal_config = yaml.safe_load(
        (ROOT / "configs/experiments/piu_oracle_formal_v1.yaml").read_text()
    )
    formal_config["oracle_pilot_protocol"] = str(config_path)
    formal_config["run_root"] = str(tmp_path / "formal_runs")
    formal_config_path = tmp_path / "oracle_formal_config.yaml"
    formal_config_path.write_text(yaml.safe_dump(formal_config, sort_keys=False))
    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "calibrated-interaction.oracle-target-prompt-result.v1"
                ),
                "status": "INDEPENDENT_DEVELOPMENT_PILOT_COMPLETE",
                "formal_method_claim": False,
                "automatic_method_branch": False,
                "confirmation": {"selected_style": "box"},
            },
            indent=2,
        )
        + "\n"
    )
    plan_path = tmp_path / "formal_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "calibrated-interaction.oracle-formal-test-plan.v1"
                ),
                "status": "PROSPECTIVE_GROUP_COUNT_FROZEN",
                "claim_scope": "DESIGN_ONLY_NO_FORMAL_OUTCOME_DATA",
                "protocol": _reference(config_path),
                "pilot": {**_reference(pilot_path), "trials": 5},
                "alpha": config["formal_followup"]["alpha"],
                "target_power": config["formal_followup"]["target_power"],
                "prospective_group_count": 2,
                "test": config["formal_followup"]["test"],
                "warning": (
                    "Pilot effect estimates determine design only. Pilot groups "
                    "are excluded from the formal p-value and effect estimate."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    split_path = tmp_path / "split.yaml"
    roles = (
        "train",
        "development",
        "calibration_temperature",
        "calibration_conformal",
        "sealed_test",
    )
    assignments = [
        {
            "initial_state_group": f"core-{role}",
            "seed": 41000 + index,
            "split_role": role,
        }
        for index, role in enumerate(roles)
    ]
    assignments.extend(
        [
            {
                "initial_state_group": "oracle-formal-a",
                "seed": 42001,
                "split_role": "oracle_formal",
            },
            {
                "initial_state_group": "oracle-formal-b",
                "seed": 42002,
                "split_role": "oracle_formal",
            },
        ]
    )
    split_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.group-split-manifest.v1",
                "status": "FROZEN_BEFORE_COLLECTION",
                "allocation_method": "prospective_without_outcome_access",
                "scenario": "original_cluttered_drawer",
                "assignments": assignments,
            },
            sort_keys=False,
        )
    )
    state_paths = {}
    for index, group in enumerate(("oracle-formal-a", "oracle-formal-b")):
        state_path = tmp_path / f"{group}.npz"
        np.savez_compressed(state_path, state=np.asarray([index, index + 0.25]))
        state_paths[group] = state_path
    states_path = tmp_path / "formal_states.json"
    subprocess.run(
        [
            sys.executable,
            str(STATE_BUILDER),
            "--split-manifest",
            str(split_path),
            "--state",
            "oracle-formal-a",
            str(state_paths["oracle-formal-a"]),
            "--state",
            "oracle-formal-b",
            str(state_paths["oracle-formal-b"]),
            "--output",
            str(states_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    schedule_path = tmp_path / "formal_schedule.json"
    subprocess.run(
        [
            sys.executable,
            str(SCHEDULE_BUILDER),
            "--formal-plan",
            str(plan_path),
            "--split-manifest",
            str(split_path),
            "--initial-state-manifest",
            str(states_path),
            "--open-certificate",
            str(certificate),
            "--config",
            str(config_path),
            "--formal-execution-config",
            str(formal_config_path),
            "--output",
            str(schedule_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    identity_path = ROOT / "results/diagnostics/pi05_libero_checkpoint_identity_v1.json"
    identity = load_checkpoint_identity(identity_path)
    server_metadata = expected_server_metadata(identity)
    server_metadata["server_session_id"] = "a" * 32
    endpoint_path = tmp_path / "endpoint.json"
    endpoint_path.write_text(
        json.dumps(
            {
                "schema_version": "piu.external-pi05-check.v1",
                "status": "PASS",
                "endpoint": {"host": "pi05.example", "port": 8002},
                "identity": server_metadata,
                "action_probe": {"finite": True, "shape": [1, 7]},
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "certificate": certificate,
        "config": config_path,
        "formal_config": formal_config_path,
        "pilot": pilot_path,
        "plan": plan_path,
        "split": split_path,
        "states": states_path,
        "schedule": schedule_path,
        "endpoint": endpoint_path,
        "state_a": state_paths["oracle-formal-a"],
        "state_b": state_paths["oracle-formal-b"],
    }


def test_oracle_formal_schedule_is_disjoint_qualified_and_dry_runnable(
    tmp_path: Path,
) -> None:
    files = _formal_fixture(tmp_path)
    schedule = load_oracle_formal_schedule(
        files["schedule"], repository_root=ROOT
    )
    assert schedule["primary_outcome"] == "target_grasp_contact"
    assert schedule["source_open_candidate"]["candidate_id"] == (
        "open_middle_drawer"
    )
    assert len(schedule["entries"]) == 2
    assert all(set(row["arm_order"]) == set(schedule["arms"]) for row in schedule["entries"])
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--schedule",
            str(files["schedule"]),
            "--execution-index",
            "0",
            "--endpoint-check",
            str(files["endpoint"]),
            "--host",
            "pi05.example",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    execution = json.loads(completed.stdout)
    assert execution["outputs_created"] is False
    assert "--external-server" in execution["commands"]["source_open"]
    assert "--oracle-target-visual-prompt box" in execution["commands"][
        "oracle_target_prompt"
    ]
    with pytest.raises(ValueError, match="complete denominator"):
        analyze_oracle_formal_schedule(files["schedule"], repository_root=ROOT)


def test_oracle_formal_interruption_closes_as_paired_failure(tmp_path: Path) -> None:
    files = _formal_fixture(tmp_path)
    schedule = json.loads(files["schedule"].read_text())
    restarted = json.loads(files["endpoint"].read_text())
    restarted["identity"]["server_session_id"] = "b" * 32
    restarted_endpoint = tmp_path / "endpoint_after_restart.json"
    restarted_endpoint.write_text(json.dumps(restarted, indent=2) + "\n")
    previous = None
    for entry in schedule["entries"]:
        endpoint_path = (
            files["endpoint"]
            if entry["execution_index"] == 0
            else restarted_endpoint
        )
        started_path = Path(entry["expected_started_ticket"])
        started_path.parent.mkdir(parents=True)
        started_path.write_text(
            json.dumps(
                {
                    "schema_version": "piu.oracle-formal-group-start.v1",
                    "status": "STARTED_SINGLE_USE",
                    "claim_scope": "FORMAL_ORACLE_ATTEMPT_NO_OUTCOMES_LOADED",
                    "outcomes_loaded": False,
                    "execution_index": entry["execution_index"],
                    "entry": entry,
                    "schedule_sha256": _sha256(files["schedule"]),
                    "endpoint_check_sha256": _sha256(endpoint_path),
                    "previous_receipt_sha256": previous,
                },
                indent=2,
            )
            + "\n"
        )
        subprocess.run(
            [
                sys.executable,
                str(CLOSER),
                "--schedule",
                str(files["schedule"]),
                "--execution-index",
                str(entry["execution_index"]),
                "--endpoint-check",
                str(endpoint_path),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        receipt_path = Path(entry["expected_group_receipt"])
        receipt = load_oracle_formal_group_receipt(
            receipt_path,
            schedule_path=files["schedule"],
            schedule=schedule,
            execution_index=entry["execution_index"],
            repository_root=ROOT,
        )
        assert receipt["source_open_status"] == "INTERRUPTED_UNVERIFIED"
        assert all(
            not values["target_grasp_contact"]
            for values in receipt["derived_outcomes"].values()
        )
        previous = _sha256(receipt_path)
    output = tmp_path / "formal_result.json"
    subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--schedule",
            str(files["schedule"]),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text())
    assert result["status"] == "FORMAL_ORACLE_CAUSAL_EFFECT_NOT_SUPPORTED"
    assert result["complete_frozen_denominator"] is True
    assert result["primary"]["exact_two_sided_paired_binomial_p"] == 1.0
    assert result["formal_method_claim"] is False
    assert result["server_session_count"] == 2


def test_oracle_formal_complete_reports_are_recomputed_not_trusted(
    tmp_path: Path,
) -> None:
    files = _formal_fixture(tmp_path)
    schedule = json.loads(files["schedule"].read_text())
    config = yaml.safe_load(files["config"].read_text())
    endpoint = json.loads(files["endpoint"].read_text())
    previous = None
    for entry in schedule["entries"]:
        started_path = Path(entry["expected_started_ticket"])
        started_path.parent.mkdir(parents=True)
        started_path.write_text(
            json.dumps(
                {
                    "schema_version": "piu.oracle-formal-group-start.v1",
                    "status": "STARTED_SINGLE_USE",
                    "claim_scope": "FORMAL_ORACLE_ATTEMPT_NO_OUTCOMES_LOADED",
                    "outcomes_loaded": False,
                    "execution_index": entry["execution_index"],
                    "entry": entry,
                    "schedule_sha256": _sha256(files["schedule"]),
                    "endpoint_check_sha256": _sha256(files["endpoint"]),
                    "previous_receipt_sha256": previous,
                },
                indent=2,
            )
            + "\n"
        )
        post_open = Path(entry["expected_post_open_state"])
        post_open.parent.mkdir(parents=True)
        np.savez_compressed(post_open, state=np.asarray([9.0, 10.0]))
        open_report = Path(entry["expected_open_report"])
        _write_formal_report(
            path=open_report,
            entry=entry,
            schedule=schedule,
            config=config,
            endpoint=endpoint,
            arm="source_open",
            source_state=Path(entry["source_state"]["path"]),
            contact=False,
        )
        arm_reports = {}
        for arm in schedule["arms"]:
            arm_path = Path(entry["expected_arm_reports"][arm])
            _write_formal_report(
                path=arm_path,
                entry=entry,
                schedule=schedule,
                config=config,
                endpoint=endpoint,
                arm=arm,
                source_state=post_open,
                contact=arm == "oracle_target_prompt",
            )
            arm_reports[arm] = _reference(arm_path)
        receipt_path = Path(entry["expected_group_receipt"])
        receipt = {
            "schema_version": "piu.oracle-formal-group-receipt.v1",
            "status": "CLOSED_SINGLE_USE",
            "claim_scope": "FORMAL_ORACLE_INTENTION_TO_TREAT",
            "execution_index": entry["execution_index"],
            "initial_state_group": entry["initial_state_group"],
            "schedule_sha256": _sha256(files["schedule"]),
            "started_ticket_sha256": _sha256(started_path),
            "endpoint_check": _reference(files["endpoint"]),
            "server_session_id": endpoint["identity"]["server_session_id"],
            "source_open_status": "COMPLETE",
            "arm_status": {arm: "COMPLETE" for arm in schedule["arms"]},
            "reports": {"source_open": _reference(open_report), **arm_reports},
            "post_open_state_sha256": _sha256(post_open),
            "derived_outcomes": {
                arm: {
                    "target_grasp_contact": arm == "oracle_target_prompt",
                    "wrong_object_grasp_contact": False,
                    "target_destination_final": arm == "oracle_target_prompt",
                    "task_success": arm == "oracle_target_prompt",
                    "target_maximum_lift_m": (
                        0.1 if arm == "oracle_target_prompt" else 0.0
                    ),
                }
                for arm in schedule["arms"]
            },
            "errors": {"source_open": None, **{arm: None for arm in schedule["arms"]}},
            "outcomes_entered_manually": False,
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        load_oracle_formal_group_receipt(
            receipt_path,
            schedule_path=files["schedule"],
            schedule=schedule,
            execution_index=entry["execution_index"],
            repository_root=ROOT,
        )
        previous = _sha256(receipt_path)
    result = analyze_oracle_formal_schedule(files["schedule"], repository_root=ROOT)
    assert result["primary"]["paired_risk_difference"] == 1.0
    assert result["primary"]["treatment"]["successes"] == 2
    assert result["primary"]["comparator"]["successes"] == 0
    tampered = json.loads(Path(schedule["entries"][0]["expected_group_receipt"]).read_text())
    tampered["derived_outcomes"]["raw_post_open_direct"][
        "target_grasp_contact"
    ] = True
    Path(schedule["entries"][0]["expected_group_receipt"]).write_text(
        json.dumps(tampered, indent=2) + "\n"
    )
    with pytest.raises(ValueError, match="recomputation"):
        analyze_oracle_formal_schedule(files["schedule"], repository_root=ROOT)


def test_oracle_formal_rejects_pilot_seed_reuse(tmp_path: Path) -> None:
    files = _formal_fixture(tmp_path)
    split = yaml.safe_load(files["split"].read_text())
    next(
        row for row in split["assignments"] if row["split_role"] == "oracle_formal"
    )["seed"] = 1409
    tampered = tmp_path / "pilot_seed_split.yaml"
    tampered.write_text(yaml.safe_dump(split, sort_keys=False))
    tampered_states = tmp_path / "pilot_seed_states.json"
    subprocess.run(
        [
            sys.executable,
            str(STATE_BUILDER),
            "--split-manifest",
            str(tampered),
            "--state",
            "oracle-formal-a",
            str(files["state_a"]),
            "--state",
            "oracle-formal-b",
            str(files["state_b"]),
            "--output",
            str(tampered_states),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCHEDULE_BUILDER),
            "--formal-plan",
            str(files["plan"]),
            "--split-manifest",
            str(tampered),
            "--initial-state-manifest",
            str(tampered_states),
            "--open-certificate",
            str(files["certificate"]),
            "--config",
            str(files["config"]),
            "--formal-execution-config",
            str(files["formal_config"]),
            "--output",
            str(tmp_path / "invalid_schedule.json"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "preflight/pilot seed" in completed.stderr
    assert not (tmp_path / "invalid_schedule.json").exists()

    qualification_split = yaml.safe_load(files["split"].read_text())
    next(
        row
        for row in qualification_split["assignments"]
        if row["split_role"] == "oracle_formal"
    )["seed"] = 30000
    qualification_split_path = tmp_path / "qualification_seed_split.yaml"
    qualification_split_path.write_text(
        yaml.safe_dump(qualification_split, sort_keys=False)
    )
    qualification_states = tmp_path / "qualification_seed_states.json"
    subprocess.run(
        [
            sys.executable,
            str(STATE_BUILDER),
            "--split-manifest",
            str(qualification_split_path),
            "--state",
            "oracle-formal-a",
            str(files["state_a"]),
            "--state",
            "oracle-formal-b",
            str(files["state_b"]),
            "--output",
            str(qualification_states),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    qualification_reuse = subprocess.run(
        [
            sys.executable,
            str(SCHEDULE_BUILDER),
            "--formal-plan",
            str(files["plan"]),
            "--split-manifest",
            str(qualification_split_path),
            "--initial-state-manifest",
            str(qualification_states),
            "--open-certificate",
            str(files["certificate"]),
            "--config",
            str(files["config"]),
            "--formal-execution-config",
            str(files["formal_config"]),
            "--output",
            str(tmp_path / "qualification_reuse_schedule.json"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert qualification_reuse.returncode != 0
    assert "qualification data" in qualification_reuse.stderr
