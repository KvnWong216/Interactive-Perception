from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/original_drawer_oracle_target_prompt_gate_v1.yaml"
PREFLIGHT = ROOT / "results/diagnostics/original_drawer_oracle_prompt_preflight_v1.json"
RUNNER = ROOT / "scripts/pipeline/run_oracle_target_prompt_gate.py"
SUMMARIZER = ROOT / "scripts/evaluation/summarize_oracle_target_prompt_gate.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_script(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_oracle_gate_is_external_only_and_group_partitioned() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    assert config["resource_contract"] == {
        "policy_server": "external_only",
        "local_policy_server_allowed": False,
        "local_gpu_memory_mib_max": 1500,
    }
    screen = set(config["screen"]["seeds"])
    confirmation = set(config["confirmation"]["seeds"])
    eligible = set(config["preflight"]["expected_eligible_seeds"])
    assert screen.isdisjoint(confirmation)
    assert screen | confirmation == eligible
    assert set(config["preflight"]["source_seeds"]) - eligible == {1408, 1409}


def test_retained_oracle_preflight_is_policy_free_and_source_hashed() -> None:
    result = json.loads(PREFLIGHT.read_text())
    assert result["status"] == "PASS"
    assert result["policy_server_contacted"] is False
    assert result["policy_actions_sampled"] is False
    assert result["experiment"]["sha256"] == sha256(CONFIG)
    assert result["eligible_seeds"] == list(range(1400, 1408))
    assert result["excluded_seeds"] == [1408, 1409]
    expected_keys = sorted(
        [
            "observation/image",
            "observation/state",
            "observation/wrist_image",
            "prompt",
        ]
    )
    for row in result["rows"]:
        source = ROOT / row["source_initial_state"]["path"]
        assert sha256(source) == row["source_initial_state"]["sha256"]
        assert row["packet"]["serialized_keys"] == expected_keys
        assert row["packet"]["agentview_shape"] == [224, 224, 3]
        assert row["packet"]["wrist_shape"] == [224, 224, 3]
        assert row["packet"]["state_shape"] == [8]


def test_runner_never_constructs_a_local_policy_server_command() -> None:
    runner = load_script("oracle_gate_runner", RUNNER)
    config = runner.load_config(CONFIG)
    command, report = runner.command_for(
        config=config,
        phase="screen",
        style="box",
        seed=1400,
        host="frozen-policy.example",
        port=8002,
        server_timeout=30.0,
    )
    assert "--external-server" in command
    assert command[command.index("--host") + 1] == "frozen-policy.example"
    assert "--gpu" not in command
    assert report == (
        ROOT / "runs/oracle_target_prompt_gate_v1/screen/box/seed1400/report.json"
    )


def test_style_selection_obeys_preregistered_lexicographic_gate() -> None:
    summarizer = load_script("oracle_gate_summarizer", SUMMARIZER)
    config = yaml.safe_load(CONFIG.read_text())
    rows = []
    outcomes = {
        "box": [(True, False, False), (True, False, False), (False, False, False)],
        "point": [(True, True, False), (True, False, False), (False, False, False)],
        "spotlight": [
            (True, True, True),
            (True, False, False),
            (False, False, False),
        ],
    }
    for style, style_rows in outcomes.items():
        for target_pick, destination, wrong_contact in style_rows:
            rows.append(
                {
                    "style": style,
                    "target_pick": target_pick,
                    "target_destination": destination,
                    "target_destination_final": destination,
                    "task_success": destination,
                    "wrong_object_contact": wrong_contact,
                }
            )
    selected, aggregates = summarizer.select_style(config, rows)
    assert selected == "point"
    assert aggregates["point"]["target_pick"]["successes"] == 2
