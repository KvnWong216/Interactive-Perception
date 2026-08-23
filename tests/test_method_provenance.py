from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from calibrated_interaction.provenance import (
    audit_claim_surfaces,
    load_and_validate_registry,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/experiments/method_provenance_v1.yaml"
STAGE1 = ROOT / "configs/experiments/candidate_interaction_stage1.yaml"


def test_repository_method_provenance_is_claim_safe() -> None:
    registry = load_and_validate_registry(REGISTRY)
    by_id = {row["id"]: row for row in registry["entries"]}
    assert by_id["historical_pick_lift_3cm"]["claim_use"] == "historical_only"
    assert by_id["oracle_marker_geometry"]["claim_use"] == "diagnostic_only"
    assert by_id["conformal_threshold"]["provenance"] == ("fit_on_isolated_calibration")
    assert by_id["public_candidate_enumeration"]["provenance"] == (
        "protocol_identifier"
    )
    assert by_id["public_post_observation_memory"]["claim_use"] == (
        "main_method_online"
    )
    assert by_id["canonical_public_observation_identity"]["claim_use"] == (
        "main_method_online"
    )
    assert by_id["calibrated_post_observation_holding_belief"]["claim_use"] == (
        "main_method_online"
    )
    assert by_id["action_effect_causal_state_alignment"]["claim_use"] == (
        "main_method_online"
    )
    assert by_id["sealed_primary_paired_test"]["claim_use"] == "main_evaluator"
    assert by_id["formal_pilot_confidence_levels"]["provenance"] == (
        "formal_statistical_procedure"
    )
    assert by_id["formal_execution_permutation"]["claim_use"] == "main_evaluator"
    assert by_id["formal_initial_state_cohort"]["provenance"] == (
        "protocol_identifier"
    )
    assert by_id["formal_offline_release_lock"]["provenance"] == (
        "protocol_identifier"
    )
    assert by_id["public_executed_action_history"]["provenance"] == (
        "protocol_identifier"
    )
    assert by_id["learned_effect_to_route_bridge"]["claim_use"] == (
        "main_method_online"
    )
    assert by_id["calibrated_spatial_subtask_bridge"]["claim_use"] == (
        "main_method_online"
    )
    assert by_id["formally_qualified_primitive_certificate"]["claim_use"] == (
        "main_evaluator"
    )
    assert by_id["uncalibrated_ablation_unique_argmax"]["claim_use"] == (
        "baseline_only"
    )
    assert by_id["prompted_vlm_router_protocol"]["claim_use"] == "baseline_only"
    assert by_id["calibrated_no_spatial_bridge_ablation"]["claim_use"] == (
        "baseline_only"
    )
    assert by_id["oracle_effect_branch_selection"]["provenance"] == (
        "oracle_intervention"
    )
    assert by_id["paper_table_missingness_contract"]["claim_use"] == (
        "main_evaluator"
    )
    assert by_id["same_scenario_prompt_stress"]["provenance"] == (
        "protocol_identifier"
    )
    assert by_id["public_claim_surface_semantics"]["claim_use"] == (
        "main_evaluator"
    )
    assert by_id["oracle_formal_attempted_open_pairing"]["claim_use"] == (
        "diagnostic_only"
    )
    assert len(registry["tracked_protocols"]) == 11


def test_claim_surface_audit_rejects_retired_direct_as_pick_semantics(
    tmp_path: Path,
) -> None:
    surface = tmp_path / "paper.md"
    surface.write_text(
        "Visible-object executor control | pick 10/10. "
        "Missing evidence is PENDING."
    )
    registry = {
        "claim_surfaces": [
            {
                "path": "paper.md",
                "required_fragments": ["Missing evidence is PENDING"],
            }
        ],
        "retired_claim_fragments": [
            {
                "id": "direct_control_called_pick_qualification",
                "fragment": "visible-object executor control | pick 10/10",
                "replacement": "Report compound DIRECT contact.",
            }
        ],
    }
    with pytest.raises(ValueError, match="retired claim semantics"):
        audit_claim_surfaces(registry, repository_root=tmp_path)


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
