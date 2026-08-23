from __future__ import annotations

import pytest

from piu.contracts import Split
from piu.splits import (
    assignment_for,
    effect_collection_role,
    role_to_split,
    validate_split_manifest,
)


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
    assert role_to_split("effect_calibration_temperature").value == "calibration"


def test_split_manifest_rejects_duplicate_seed_and_missing_role() -> None:
    value = _manifest()
    value["assignments"][1]["seed"] = value["assignments"][0]["seed"]
    with pytest.raises(ValueError, match="unique"):
        validate_split_manifest(value)
    value = _manifest()
    value["assignments"].pop()
    with pytest.raises(ValueError, match="every isolated split role"):
        validate_split_manifest(value)


def test_split_manifest_can_reserve_disjoint_primitive_qualification_groups() -> None:
    value = _manifest()
    value["assignments"].append(
        {
            "initial_state_group": "primitive-q0",
            "seed": 500,
            "split_role": "primitive_qualification",
        }
    )
    manifest = validate_split_manifest(value)
    assert assignment_for(manifest, "primitive-q0")["split_role"] == (
        "primitive_qualification"
    )
    assert role_to_split("primitive_qualification") is Split.PRIMITIVE_QUALIFICATION


def test_split_manifest_can_freeze_a_purpose_specific_role_cohort() -> None:
    value = {
        "schema_version": "piu.group-split-manifest.v1",
        "status": "FROZEN_BEFORE_COLLECTION",
        "allocation_method": "prospective_without_outcome_access",
        "scenario": "fixed drawer",
        "required_roles": ["oracle_formal"],
        "assignments": [
            {
                "initial_state_group": "oracle-g0",
                "seed": 600,
                "split_role": "oracle_formal",
            }
        ],
    }
    manifest = validate_split_manifest(value)
    assert manifest["required_roles"] == ["oracle_formal"]


def test_effect_calibration_cannot_reuse_binder_calibration_groups() -> None:
    value = _manifest()
    value["assignments"].extend(
        [
            {
                "initial_state_group": "effect-temperature",
                "seed": 500,
                "split_role": "effect_calibration_temperature",
            },
            {
                "initial_state_group": "effect-conformal",
                "seed": 501,
                "split_role": "effect_calibration_conformal",
            },
        ]
    )
    manifest = validate_split_manifest(value)
    assert (
        effect_collection_role(
            manifest,
            initial_state_group="effect-temperature",
            split=Split.CALIBRATION,
        )
        == "effect_calibration_temperature"
    )
    with pytest.raises(ValueError, match="independent effect-calibration"):
        effect_collection_role(
            manifest,
            initial_state_group="g2",
            split=Split.CALIBRATION,
        )
