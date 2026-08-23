from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from calibrated_interaction.provenance import load_and_validate_registry

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/experiments/method_provenance_v1.yaml"
STAGE1 = ROOT / "configs/experiments/candidate_interaction_stage1.yaml"


def test_repository_method_provenance_is_claim_safe() -> None:
    registry = load_and_validate_registry(REGISTRY)
    by_id = {row["id"]: row for row in registry["entries"]}
    assert by_id["historical_pick_lift_3cm"]["claim_use"] == "historical_only"
    assert by_id["oracle_marker_geometry"]["claim_use"] == "diagnostic_only"
    assert by_id["conformal_threshold"]["provenance"] == ("fit_on_isolated_calibration")
    assert len(registry["tracked_protocols"]) == 1


def test_unsupported_heuristic_cannot_be_promoted_to_main_method(
    tmp_path: Path,
) -> None:
    registry = yaml.safe_load(REGISTRY.read_text())
    row = next(
        entry
        for entry in registry["entries"]
        if entry["id"] == "historical_pick_lift_3cm"
    )
    row["claim_use"] = "main_method_online"
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(registry))
    with pytest.raises(ValueError, match="cannot control a main-method action"):
        load_and_validate_registry(broken)


def test_active_learned_package_never_imports_legacy_heuristics() -> None:
    for path in (ROOT / "src/calibrated_interaction").glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(
            name == "interaction_uncertainty"
            or name.startswith("interaction_uncertainty.")
            for name in imports
        ), f"{path} imports frozen Heuristic V0"


def test_fixed_four_token_pooling_is_rejected_pilot_only() -> None:
    config = yaml.safe_load(STAGE1.read_text())
    assert config["status"] == "rejected_development_pilot"
    assert config["paper_method_claim_allowed"] is False
    pooling = config["model"]["context_pooling"]
    assert pooling["type"] == "fixed_hand_designed"
    assert pooling["spatial_token_indices_retained"] is False
    assert pooling["main_method_allowed"] is False
    assert config["model"]["candidate_pooling"]["main_method_allowed"] is False
