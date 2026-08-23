from __future__ import annotations

import json
from pathlib import Path

from calibrated_interaction.provenance import build_claim_audit_report

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/experiments/method_provenance_v1.yaml"
REPORT = ROOT / "results/diagnostics/piu_public_claim_audit_v3.json"


def test_public_claim_audit_is_current_and_evidence_bound() -> None:
    retained = json.loads(REPORT.read_text())
    rebuilt = build_claim_audit_report(REGISTRY, repository_root=ROOT)
    assert retained == rebuilt
    assert retained["status"] == "PASS"
    assert retained["retired_fragments_checked"] >= 7
    assert len(retained["claim_surfaces"]) == 4
    assert len(retained["numeric_protocols"]) == 12
    assert all(
        row["unclassified_numeric_paths"] == []
        for row in retained["numeric_protocols"]
    )
    binding = next(
        row
        for row in retained["numeric_protocols"]
        if row["path"].endswith("piu_binding_adapter_v1.yaml")
    )
    controls = {row["field"]: row["value"] for row in binding["controls"]}
    assert controls["compute.torch_threads"] == 2
    assert controls["model_search.maximum_parameter_count"] == 2_000_000
    assert all(
        row["retired_fragments_matched"] == []
        for row in retained["claim_surfaces"]
    )
    assert retained["claim_boundary"] == {
        "retired_interpretation_present": False,
        "compound_direct_relabelled_as_pick_or_place": False,
        "historical_visibility_marker_used_as_recognition_threshold": False,
        "missing_successor_evidence_imputed": False,
        "audit_is_performance_evidence": False,
    }
