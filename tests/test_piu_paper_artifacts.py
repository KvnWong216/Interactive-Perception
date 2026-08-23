from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from piu.paper_artifacts import (
    build_evidence_tables,
    load_reporting_config,
    render_markdown,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/piu_robustness_reporting_v1.yaml"


def _reference(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_repository_tables_preserve_pending_as_missing_not_zero() -> None:
    value = build_evidence_tables(CONFIG, repository_root=ROOT)
    assert value["scope_interpretation"] == (
        "same_scenario_controlled_stress_not_broad_ood"
    )
    assert value["artifacts"]["retained_negative"]["status"] == "AVAILABLE"
    assert value["artifacts"]["closed_loop_sealed"]["status"] == "PENDING"
    assert value["main_table_evidence_complete"] is False
    assert value["automatic_method_success"] is None
    assert value["paper_method_claim_allowed"] is False
    assert value["missing_values_encoded_as_zero"] is False
    assert all(
        row["task_success"] == "PENDING"
        for row in value["formal_method_table"]
    )
    markdown = render_markdown(value)
    assert "`PENDING` means no admissible artifact exists" in markdown
    assert "B6 | oracle_upper_bound" in markdown
    assert "Automatic method success: `null`" in markdown


def test_reporting_config_rejects_an_automatic_success_threshold(
    tmp_path: Path,
) -> None:
    value = yaml.safe_load(CONFIG.read_text())
    value["claim_contract"]["automatic_success_threshold"] = 0.8
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(value, sort_keys=False))
    with pytest.raises(ValueError, match="unsupported result claims"):
        load_reporting_config(broken, repository_root=ROOT)


def test_available_development_reports_are_loaded_without_becoming_sealed(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"evidence\n")
    reference = _reference(payload)
    binding_metrics = {
        "spatial_nll": 1.0,
        "point_hit": 0.5,
        "target_probability_mass": 0.25,
        "presence_brier": 0.2,
        "sufficiency_brier": None,
        "holding_brier": None,
    }
    config = yaml.safe_load(CONFIG.read_text())
    binding_report_path = tmp_path / "binding.json"
    binding_report_path.write_text(
        json.dumps(
            {
                "schema_version": "piu.target-binder-training.v1",
                "paper_method_claim_allowed": False,
                "sealed_test_loaded": False,
                "calibration_loaded": False,
                "inputs": {"features": reference},
                "config": reference,
                "checkpoint": reference,
                "development_predictions": reference,
                "development_ablations": {
                    name: {"development_metrics": dict(binding_metrics)}
                    for name in config["development_only"][
                        "binding_input_ablations"
                    ]["order"]
                },
            }
        )
    )
    effect_report_path = tmp_path / "effect.json"
    effect_report_path.write_text(
        json.dumps(
            {
                "schema_version": "piu.action-effect-training.v1",
                "paper_method_claim_allowed": False,
                "sealed_test_loaded": False,
                "calibration_loaded": False,
                "inputs": {"features": reference},
                "variants": {
                    name: {
                        "selected_trial": 0,
                        "trials": [
                            {
                                "trial": 0,
                                "development_metrics": {
                                    "route_nll": 0.3,
                                    "route_top1_accuracy": 0.75,
                                    "macro_supported_factor_brier": (
                                        None if name == "route_only" else 0.1
                                    ),
                                },
                            }
                        ],
                        "checkpoint": reference,
                        "development_predictions": reference,
                    }
                    for name in config["development_only"][
                        "effect_training_variants"
                    ]["order"]
                },
            }
        )
    )
    config["development_only"]["binding_input_ablations"]["source"] = str(
        binding_report_path
    )
    config["development_only"]["effect_training_variants"]["source"] = str(
        effect_report_path
    )
    config_path = tmp_path / "reporting.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    value = build_evidence_tables(config_path, repository_root=ROOT)
    assert value["artifacts"]["binding_development"]["status"] == "AVAILABLE"
    assert value["binding_development_ablations"][0]["spatial_nll"] == 1.0
    assert value["binding_development_ablations"][0]["sufficiency_brier"] == (
        "UNSUPPORTED"
    )
    assert value["effect_development_variants"][0][
        "macro_supported_factor_brier"
    ] == "UNSUPPORTED"
    assert value["main_table_evidence_complete"] is False


def test_paper_table_cli_is_immutable_and_verifiable(tmp_path: Path) -> None:
    output_json = tmp_path / "tables.json"
    output_markdown = tmp_path / "tables.md"
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluation/build_piu_paper_tables.py"),
        "--config",
        str(CONFIG),
        "--output-json",
        str(output_json),
        "--output-markdown",
        str(output_markdown),
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    value = json.loads(output_json.read_text())
    assert value["local_gpu_actions_performed"] is False
    verified = subprocess.run(
        [*command, "--verify"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(verified.stdout)["status"] == "VERIFIED"
    duplicate = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert duplicate.returncode != 0
    assert "immutable" in duplicate.stderr
