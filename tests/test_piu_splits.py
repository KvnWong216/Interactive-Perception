from __future__ import annotations

import pytest

from piu.splits import assignment_for, role_to_split, validate_split_manifest


def _manifest() -> dict:
    roles = (
        "train",
        "development",
        "calibration_temperature",
        "calibration_conformal",
        "sealed_test",
    )
    return {
        "schema_version": "piu.group-split-manifest.v1",
        "status": "FROZEN_BEFORE_COLLECTION",
        "allocation_method": "prospective_without_outcome_access",
        "scenario": "fixed drawer",
        "assignments": [
            {
                "initial_state_group": f"g{index}",
                "seed": 100 + index,
                "split_role": role,
            }
            for index, role in enumerate(roles)
        ],
    }


def test_split_manifest_is_group_disjoint_and_role_complete() -> None:
    manifest = validate_split_manifest(_manifest())
    assert assignment_for(manifest, "g2")["split_role"] == ("calibration_temperature")
    assert role_to_split("calibration_conformal").value == "calibration"


def test_split_manifest_rejects_duplicate_seed_and_missing_role() -> None:
    value = _manifest()
    value["assignments"][1]["seed"] = value["assignments"][0]["seed"]
    with pytest.raises(ValueError, match="unique"):
        validate_split_manifest(value)
    value = _manifest()
    value["assignments"].pop()
    with pytest.raises(ValueError, match="every isolated split role"):
        validate_split_manifest(value)
