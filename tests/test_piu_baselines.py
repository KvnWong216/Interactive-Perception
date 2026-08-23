from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from piu.baselines import load_baseline_registry, validate_baseline_registry

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/experiments/piu_baselines_v1.yaml"


def test_repository_baselines_are_paired_and_oracle_separated() -> None:
    registry = load_baseline_registry(REGISTRY)
    by_id = {row["id"]: row for row in registry["methods"]}
    assert tuple(by_id) == tuple(f"B{index}" for index in range(9))
    assert by_id["B8"]["family"] == "proposed"
    assert by_id["B6"]["family"] == "upper_bound"
    assert by_id["B7"]["eligible_for_main_method_comparison"] is False
    assert "public_candidate_descriptions" in by_id["B1"]["online_inputs"]
    assert all(row["same_evaluator"] for row in registry["methods"])
    assert (ROOT / registry["scenario"]).is_file()


def test_public_baseline_cannot_add_an_oracle_input() -> None:
    registry = yaml.safe_load(REGISTRY.read_text())
    registry["methods"][4]["online_privileged_inputs"] = ["target_instance_mask"]
    with pytest.raises(ValueError, match="declares oracle"):
        validate_baseline_registry(registry)


def test_oracle_cannot_be_mislabeled_as_a_comparable_method() -> None:
    registry = yaml.safe_load(REGISTRY.read_text())
    registry["methods"][7]["eligible_for_main_method_comparison"] = True
    with pytest.raises(ValueError, match="cannot be a main-method"):
        validate_baseline_registry(registry)
