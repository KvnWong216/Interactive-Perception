from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/original_drawer_oracle_target_prompt_pilot_v2.yaml"
PREFLIGHT = ROOT / "results/diagnostics/original_drawer_oracle_prompt_preflight_v3.json"
AUDIT = ROOT / "results/diagnostics/original_drawer_threshold_audit_v1.json"
RUNNER = ROOT / "scripts/pipeline/run_oracle_target_prompt_gate.py"
SUMMARIZER = ROOT / "scripts/evaluation/summarize_oracle_target_prompt_gate.py"
EXECUTOR = ROOT / "scripts/pipeline/execute.py"
SCHEDULE_BUILDER = (
    ROOT / "scripts/evaluation/build_oracle_target_prompt_schedule.py"
)
SCHEDULE_CONFIG = (
    ROOT
    / "configs/experiments/original_drawer_oracle_target_prompt_schedule_v1.yaml"
)
FORMAL_PLANNER = ROOT / "scripts/evaluation/plan_oracle_paired_test.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_script(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_pilot_has_no_small_count_pass_fail_gate() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    assert config["schema_version"].endswith("pilot.v2")
    assert config["execution"]["target_presence_minimum_pixels"] == 1
    assert config["confirmation"]["automatic_method_branch"] is False
    assert "development_gate" not in config["confirmation"]
    assert "prefer_style_order" not in config["screen"]
    assert set(config["screen"]["seeds"]).isdisjoint(config["confirmation"]["seeds"])


def test_v2_preflight_uses_nonempty_masks_and_hashed_sources() -> None:
    result = json.loads(PREFLIGHT.read_text())
    assert result["status"] == "PASS"
    assert result["policy_server_contacted"] is False
    assert result["target_presence_minimum_pixels"] == 1
    assert result["experiment"]["sha256"] == sha256(CONFIG)
    assert result["eligible_seeds"] == list(range(1400, 1408))
    assert result["excluded_seeds"] == [1408, 1409]
    for row in result["rows"]:
        source = ROOT / row["source_initial_state"]["path"]
        assert sha256(source) == row["source_initial_state"]["sha256"]


def test_v2_runner_uses_external_server_and_new_report_root() -> None:
    runner = load_script("oracle_pilot_runner_v2", RUNNER)
    config = runner.load_config(CONFIG)
    command, report = runner.command_for(
        config=config,
        phase="screen",
        style="point",
        seed=1400,
        host="pi05.example",
        port=8002,
        server_timeout=30.0,
    )
    assert "--external-server" in command
    schema = command.index("--report-schema") + 1
    assert command[schema] == "v2"
    minimum = command.index("--oracle-minimum-visible-pixels") + 1
    assert command[minimum] == "1"
    assert report == (
        ROOT / "runs/oracle_target_prompt_pilot_v2/screen/point/seed1400/report.json"
    )


def test_v2_endpoint_gate_requires_session_and_finite_action_probe(
    tmp_path: Path,
) -> None:
    runner = load_script("oracle_pilot_endpoint_v2", RUNNER)
    identity_path = (
        ROOT / "results/diagnostics/pi05_libero_checkpoint_identity_v1.json"
    )
    identity = json.loads(identity_path.read_text())
    value = {
        "schema_version": "piu.external-pi05-check.v1",
        "status": "PASS",
        "endpoint": {"host": "pi05.example", "port": 8002},
        "identity": {
            "schema_version": "piu.identified-pi05-server.v1",
            "policy_config": identity["policy_config"],
            "environment": "LIBERO",
            "checkpoint": identity["checkpoint"],
            "capabilities": ["action_chunks", "spatial_prefix_v1"],
            "server_session_id": "a" * 32,
        },
        "action_probe": None,
    }
    endpoint = tmp_path / "endpoint.json"
    endpoint.write_text(json.dumps(value))
    try:
        runner.validate_endpoint_check(
            endpoint,
            host="pi05.example",
            port=8002,
            identity_path=identity_path,
            require_session=True,
        )
    except ValueError as error:
        assert "finite action probe" in str(error)
    else:
        raise AssertionError("metadata-only endpoint check was accepted")
    value["action_probe"] = {"finite": True}
    endpoint.write_text(json.dumps(value))
    runner.validate_endpoint_check(
        endpoint,
        host="pi05.example",
        port=8002,
        identity_path=identity_path,
        require_session=True,
    )


def test_v2_screen_schedule_is_preoutcome_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    for output in (first, second):
        subprocess.run(
            [
                sys.executable,
                str(SCHEDULE_BUILDER),
                "--phase",
                "screen",
                "--output",
                str(output),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    assert first.read_bytes() == second.read_bytes()
    schedule = json.loads(first.read_text())
    assert schedule["outcomes_loaded"] is False
    assert schedule["selected_style"] is None
    assert schedule["schedule_protocol"]["sha256"] == sha256(SCHEDULE_CONFIG)
    entries = schedule["entries"]
    assert [row["execution_index"] for row in entries] == list(range(9))
    assert {(row["style"], row["seed"]) for row in entries} == {
        (style, seed)
        for style in ("box", "point", "spotlight")
        for seed in (1400, 1403, 1406)
    }
    dry_run = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--phase",
            "screen",
            "--host",
            "not-contacted.invalid",
            "--schedule",
            str(first),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plans = json.loads(dry_run.stdout)
    assert [row["execution_index"] for row in plans] == list(range(9))
    assert all(row["status"] == "DRY_RUN" for row in plans)
    tampered = json.loads(first.read_text())
    tampered["entries"][0], tampered["entries"][1] = (
        tampered["entries"][1],
        tampered["entries"][0],
    )
    tampered["entries"][0]["execution_index"] = 0
    tampered["entries"][1]["execution_index"] = 1
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered))
    rejected = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--phase",
            "screen",
            "--host",
            "not-contacted.invalid",
            "--schedule",
            str(tampered_path),
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "hash permutation" in rejected.stderr


def test_v2_style_selection_has_no_manual_preference_order() -> None:
    summarizer = load_script("oracle_pilot_summarizer_v2", SUMMARIZER)
    config = yaml.safe_load(CONFIG.read_text())
    rows = []
    for style, contacts, changed in (
        ("box", 2, 400),
        ("point", 2, 100),
        ("spotlight", 1, 1000),
    ):
        for index in range(3):
            rows.append(
                {
                    "style": style,
                    "target_grasp_contact": index < contacts,
                    "target_destination": False,
                    "target_destination_final": False,
                    "task_success": False,
                    "wrong_object_contact": False,
                    "initial_changed_rgb_pixels": changed,
                }
            )
    selected, aggregates = summarizer.select_style(config, rows)
    assert selected == "point"
    assert aggregates["point"]["target_grasp_contact"]["successes"] == 2


def test_threshold_audit_proves_retained_conclusions_are_invariant() -> None:
    result = json.loads(AUDIT.read_text())
    assert result["visibility"]["threshold_invariant_integer_interval_raw_pixels"] == [
        1,
        447,
    ]
    target = result["target_manipulation"]
    assert target["post_open_butter"]["grasp_contact_successes"] == 0
    assert target["visible_cream_cheese_control"]["grasp_contact_successes"] == 10
    assert (
        target["visible_cream_cheese_control"][
            "minimum_maximum_lift_among_contact_runs_m"
        ]
        > 0.15
    )
    assert result["drawer_open"]["successes"] == 9


def test_v2_executor_has_no_binary_lift_threshold() -> None:
    source = EXECUTOR.read_text()
    assert 'default="v2"' in source
    assert '"target_pick_threshold": None' in source
    assert 'if metric_contract_version == "v1"' in source


def test_oracle_power_planner_hash_binds_one_frozen_resource_cap(
    tmp_path: Path,
) -> None:
    pilot = tmp_path / "pilot.json"
    pilot.write_text(
        json.dumps(
            {
                "status": "INDEPENDENT_DEVELOPMENT_PILOT_COMPLETE",
                "confirmation": {
                    "aggregates": {"target_grasp_contact": {"trials": 5}},
                    "paired_target_grasp_contact_comparison": {
                        "left_only": 5,
                        "right_only": 0,
                    },
                },
            }
        )
    )
    output = tmp_path / "plan.json"
    subprocess.run(
        [
            sys.executable,
            str(FORMAL_PLANNER),
            "--pilot",
            str(pilot),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(output.read_text())
    resource = ROOT / "configs/experiments/piu_formal_analysis_v1.yaml"
    assert plan["resource_contract"] == {
        "path": "configs/experiments/piu_formal_analysis_v1.yaml",
        "sha256": sha256(resource),
        "numerical_search_limit": 200,
    }
    assert plan["search_limit"] == 200
    assert plan["search_limit_role"] == (
        "numerical resource bound; never a success threshold"
    )
    assert "--search-limit" not in FORMAL_PLANNER.read_text()
