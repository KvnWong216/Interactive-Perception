from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from piu.statistics import (
    analyze_formal_outcomes,
    exact_paired_binomial_pvalue,
    holm_adjust,
    load_analysis_config,
    load_formal_outcomes,
    paired_binary_summary,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/piu_formal_analysis_v1.yaml"


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_exact_paired_test_reproduces_eight_one_sided_discordances() -> None:
    assert exact_paired_binomial_pvalue(8, 0) == pytest.approx(0.0078125)
    summary = paired_binary_summary([True] * 8, [False] * 8)
    assert summary["paired_risk_difference"] == 1.0
    assert summary["discordance"]["treatment_only"] == 8


def test_holm_is_monotone_and_preserves_names() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.20})


def _row(group: str, method: str, success: bool) -> dict[str, object]:
    return {
        "schema_version": "piu.formal-outcome.v1",
        "initial_state_group": group,
        "simulator_seed": 17,
        "method_id": method,
        "split": "sealed_test",
        "rollout_status": "COMPLETE",
        "evidence_class": "oracle_upper_bound"
        if method in {"B6", "B7"}
        else "public_method",
        "source_state_sha256": "1" * 64,
        "action_history_sha256": "2" * 64,
        "policy_identity_sha256": "3" * 64,
        "outcomes": {
            "target_grasp_contact": success,
            "wrong_object_grasp_contact": False,
            "target_destination_final": success,
            "task_success": False,
            "abstention": False,
            "target_maximum_lift_m": 0.1 if success else 0.0,
            "interaction_count": 1,
            "executed_steps": 10,
        },
    }


def test_complete_formal_matrix_is_paired_and_oracle_separated(tmp_path: Path) -> None:
    methods = [f"B{index}" for index in range(9)]
    rows = [
        _row(group, method, method in {"B6", "B7", "B8"})
        for group in ("g0", "g1")
        for method in methods
    ]
    path = tmp_path / "sealed.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    loaded = load_formal_outcomes(path)
    report = analyze_formal_outcomes(loaded, load_analysis_config(CONFIG))
    assert report["groups"] == ["g0", "g1"]
    assert report["primary"]["paired_risk_difference"] == 1.0
    assert report["oracle_upper_bound_methods"] == ["B6", "B7"]
    assert report["oracle_binding_gap_descriptive_only"][
        "fraction_of_raw_to_oracle_gap_closed"
    ] == 1.0
    assert "B8_minus_B1" in report["continuous_descriptive_only"][
        "interaction_count"
    ]["paired"]
    assert report["automatic_method_pass"] is None


def test_formal_analysis_rejects_missing_pairs() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    rows = [_row("g0", "B0", False), _row("g0", "B8", True), _row("g1", "B8", True)]
    with pytest.raises(ValueError, match="exactly B0--B8"):
        analyze_formal_outcomes(rows, config)


def test_formal_analysis_rejects_unpaired_source_state_hashes() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    rows = [_row("g0", f"B{index}", index == 8) for index in range(9)]
    rows[-1]["source_state_sha256"] = "3" * 64
    with pytest.raises(ValueError, match="different paired source states"):
        analyze_formal_outcomes(rows, config)


def test_formal_analysis_rejects_policy_identity_drift() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    rows = [_row("g0", f"B{index}", index == 8) for index in range(9)]
    rows[-1]["policy_identity_sha256"] = "4" * 64
    with pytest.raises(ValueError, match="different frozen policy identities"):
        analyze_formal_outcomes(rows, config)


def test_formal_analysis_cli_requires_hash_bound_single_use_authorization(
    tmp_path: Path,
) -> None:
    outcomes = tmp_path / "sealed.jsonl"
    rows = [_row("g0", f"B{index}", index == 8) for index in range(9)]
    outcomes.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "analysis.json"
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema_version": "piu.formal-analysis-sealed-authorization.v1",
                "outcomes_sha256": hashlib.sha256(outcomes.read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                "single_use_output": str(output.resolve()),
            }
        )
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/analyze_piu_formal_experiment.py"),
            "--outcomes",
            str(outcomes),
            "--config",
            str(CONFIG),
            "--sealed-authorization",
            str(authorization),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["primary"]["treatment_id"] == "B8"
    assert (
        report["sealed_authorization"]["sha256"]
        == hashlib.sha256(authorization.read_bytes()).hexdigest()
    )


def test_authorized_episode_rows_assemble_only_as_complete_frozen_matrix(
    tmp_path: Path,
) -> None:
    source = tmp_path / "state.npz"
    np.savez_compressed(source, state=np.asarray([1.0], dtype=np.float64))
    history = tmp_path / "history.json"
    history.write_text("[]\n")
    identity = ROOT / "results/diagnostics/pi05_libero_checkpoint_identity_v1.json"
    split_manifest = tmp_path / "splits.yaml"
    roles = (
        "train",
        "development",
        "calibration_temperature",
        "calibration_conformal",
        "sealed_test",
    )
    split_manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.group-split-manifest.v1",
                "status": "FROZEN_BEFORE_COLLECTION",
                "allocation_method": "prospective_without_outcome_access",
                "scenario": "fixed drawer",
                "assignments": [
                    {
                        "initial_state_group": (
                            "sealed_group" if role == "sealed_test" else role
                        ),
                        "seed": 17 if role == "sealed_test" else index,
                        "split_role": role,
                    }
                    for index, role in enumerate(roles)
                ],
            }
        )
    )
    matrix = tmp_path / "matrix.jsonl"
    formal_schedule = tmp_path / "formal_schedule.json"
    repro_lock = ROOT / "results/diagnostics/piu_offline_repro_preflight_v1.json"
    baseline_registry = ROOT / "configs/experiments/piu_baselines_v1.yaml"
    scenario_config = ROOT / "configs/scenarios/original_drawer.yaml"
    formal_schedule.write_text(
        json.dumps(
            {
                "schema_version": "piu.formal-execution-schedule.v1",
                "status": "FROZEN_BEFORE_FORMAL_OUTCOME_COLLECTION",
                "outcomes_loaded": False,
                "inputs": {
                    "offline_repro_lock": {
                        **_artifact(repro_lock),
                        "manifest_sha256": hashlib.sha256(
                            (
                                ROOT
                                / "configs/experiments/piu_offline_repro_v1.yaml"
                            ).read_bytes()
                        ).hexdigest(),
                    },
                    "split_manifest": {
                        "path": str(split_manifest),
                        "sha256": hashlib.sha256(
                            split_manifest.read_bytes()
                        ).hexdigest(),
                    },
                    "policy_identity": {
                        "path": str(identity),
                        "sha256": hashlib.sha256(identity.read_bytes()).hexdigest(),
                    },
                    "baseline_registry": _artifact(baseline_registry),
                    "scenario_config": _artifact(scenario_config),
                },
                "shared_execution_contract": {
                    "maximum_controller_decisions": 8,
                    "interpretation": (
                        "resource_cap_not_learned_decision_threshold"
                    ),
                },
                "entries": [
                    {
                        "execution_index": index,
                        "initial_state_group": "sealed_group",
                        "simulator_seed": 17,
                        "method_id": f"B{index}",
                        "state_key": "state",
                        "source_state": _artifact(source),
                    }
                    for index in range(9)
                ],
            }
        )
    )
    ledger = tmp_path / "ledger"
    row_paths = []
    for index in range(9):
        method = f"B{index}"
        run_dir = tmp_path / f"{method}_run"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/evaluation/begin_piu_formal_attempt.py"),
                "--schedule",
                str(formal_schedule),
                "--ledger-dir",
                str(ledger),
                "--run-output-dir",
                str(run_dir),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        ticket = ledger / f"{index:05d}.started.json"
        run_dir.mkdir()
        episode_path = run_dir / "episode.json"
        episode_path.write_text(
            json.dumps(
                {
                    "schema_version": "piu.closed-loop-episode.v1",
                    "method_id": method,
                    "initial_state_group": "sealed_group",
                    "simulator_seed": 17,
                    "split": "sealed_test",
                    "evidence_class": (
                        "oracle_upper_bound"
                        if method in {"B6", "B7"}
                        else "public_method"
                    ),
                    "rollout_status": "COMPLETE",
                    "source_state": _artifact(source),
                    "public_action_history": _artifact(history),
                    "policy_identity": _artifact(identity),
                    "outcomes": _row("sealed_group", method, index == 8)["outcomes"],
                    "online_oracle_inputs": (
                        ["evaluator upper bound"] if method in {"B6", "B7"} else []
                    ),
                    "formal_attempt_ticket": _artifact(ticket),
                }
            )
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/evaluation/close_piu_formal_attempt.py"),
                "--ticket",
                str(ticket),
                "--episode",
                str(episode_path),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        close = ledger / f"{index:05d}.closed.json"
        row_path = tmp_path / f"{method}.jsonl"
        authorization = tmp_path / f"{method}_authorization.json"
        authorization.write_text(
            json.dumps(
                {
                    "schema_version": "piu.formal-row-sealed-authorization.v1",
                    "episode_sha256": hashlib.sha256(
                        episode_path.read_bytes()
                    ).hexdigest(),
                    "source_state_sha256": hashlib.sha256(
                        source.read_bytes()
                    ).hexdigest(),
                    "action_history_sha256": hashlib.sha256(
                        history.read_bytes()
                    ).hexdigest(),
                    "policy_identity_sha256": hashlib.sha256(
                        identity.read_bytes()
                    ).hexdigest(),
                    "method_id": method,
                    "formal_attempt_close_sha256": hashlib.sha256(
                        close.read_bytes()
                    ).hexdigest(),
                    "single_use_output": str(row_path),
                }
            )
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/evaluation/export_piu_formal_outcome.py"),
                "--episode",
                str(episode_path),
                "--formal-attempt-close",
                str(close),
                "--sealed-authorization",
                str(authorization),
                "--output",
                str(row_path),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        row_paths.append(row_path)
    wrong_schedule = tmp_path / "wrong_formal_schedule.json"
    wrong_value = json.loads(formal_schedule.read_text())
    wrong_value["entries"][0]["source_state"]["sha256"] = "f" * 64
    wrong_schedule.write_text(json.dumps(wrong_value))
    wrong_matrix = tmp_path / "wrong_matrix.jsonl"
    wrong_authorization = tmp_path / "wrong_matrix_authorization.json"
    wrong_authorization.write_text(
        json.dumps(
            {
                "schema_version": "piu.formal-matrix-sealed-authorization.v1",
                "row_sha256_sorted": sorted(
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in row_paths
                ),
                "split_manifest_sha256": hashlib.sha256(
                    split_manifest.read_bytes()
                ).hexdigest(),
                "formal_schedule_sha256": hashlib.sha256(
                    wrong_schedule.read_bytes()
                ).hexdigest(),
                "single_use_output": str(wrong_matrix),
            }
        )
    )
    rejected = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/assemble_piu_formal_outcomes.py"),
            "--rows",
            *(str(path) for path in row_paths),
            "--split-manifest",
            str(split_manifest),
            "--formal-schedule",
            str(wrong_schedule),
            "--sealed-authorization",
            str(wrong_authorization),
            "--output",
            str(wrong_matrix),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "formal schedule source state differs" in rejected.stderr
    matrix_authorization = tmp_path / "matrix_authorization.json"
    matrix_authorization.write_text(
        json.dumps(
            {
                "schema_version": "piu.formal-matrix-sealed-authorization.v1",
                "row_sha256_sorted": sorted(
                    hashlib.sha256(path.read_bytes()).hexdigest() for path in row_paths
                ),
                "split_manifest_sha256": hashlib.sha256(
                    split_manifest.read_bytes()
                ).hexdigest(),
                "formal_schedule_sha256": hashlib.sha256(
                    formal_schedule.read_bytes()
                ).hexdigest(),
                "single_use_output": str(matrix),
            }
        )
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluation/assemble_piu_formal_outcomes.py"),
            "--rows",
            *(str(path) for path in row_paths),
            "--split-manifest",
            str(split_manifest),
            "--formal-schedule",
            str(formal_schedule),
            "--sealed-authorization",
            str(matrix_authorization),
            "--output",
            str(matrix),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert len(load_formal_outcomes(matrix)) == 9
