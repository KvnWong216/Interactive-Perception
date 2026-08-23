"""Prospectively frozen initial-state group split contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .contracts import Split

SPLIT_ROLES = (
    "train",
    "development",
    "calibration_temperature",
    "calibration_conformal",
    "sealed_test",
)
EFFECT_CALIBRATION_ROLES = (
    "effect_calibration_temperature",
    "effect_calibration_conformal",
)
LEARNING_SPLIT_ROLES = (
    "train",
    "development",
    "calibration_temperature",
    "calibration_conformal",
    *EFFECT_CALIBRATION_ROLES,
)
OPTIONAL_SPLIT_ROLES = (
    "primitive_qualification",
    "oracle_formal",
    *EFFECT_CALIBRATION_ROLES,
)


def validate_split_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != "piu.group-split-manifest.v1":
        raise ValueError("unsupported PIU group-split manifest")
    if value.get("status") != "FROZEN_BEFORE_COLLECTION":
        raise ValueError("group splits must be frozen before outcome collection")
    if value.get("allocation_method") != "prospective_without_outcome_access":
        raise ValueError("group split allocation may not inspect outcomes")
    scenario = " ".join(str(value.get("scenario", "")).split())
    if not scenario:
        raise ValueError("split manifest requires one fixed scenario")
    allowed_roles = {*SPLIT_ROLES, *OPTIONAL_SPLIT_ROLES}
    declared_required = value.get("required_roles", SPLIT_ROLES)
    if (
        not isinstance(declared_required, Sequence)
        or isinstance(declared_required, (str, bytes))
    ):
        raise TypeError("split manifest required_roles must be a sequence")
    required_roles = tuple(str(role) for role in declared_required)
    if (
        not required_roles
        or len(set(required_roles)) != len(required_roles)
        or not set(required_roles) <= allowed_roles
    ):
        raise ValueError("split manifest required_roles are malformed")
    rows = value.get("assignments")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise TypeError("split manifest assignments must be a nonempty sequence")
    groups: set[str] = set()
    seeds: set[int] = set()
    roles: set[str] = set()
    normalized = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("split assignment rows must be mappings")
        group = " ".join(str(row.get("initial_state_group", "")).split())
        seed = int(row["seed"])
        role = str(row.get("split_role", ""))
        if not group or group in groups or seed in seeds:
            raise ValueError("split groups and simulator seeds must be unique")
        if role not in allowed_roles:
            raise ValueError(f"unsupported split role {role!r}")
        groups.add(group)
        seeds.add(seed)
        roles.add(role)
        normalized.append(
            {"initial_state_group": group, "seed": seed, "split_role": role}
        )
    if not set(required_roles) <= roles:
        raise ValueError("split manifest must allocate every isolated split role")
    return {
        **dict(value),
        "scenario": scenario,
        "required_roles": list(required_roles),
        "assignments": normalized,
    }


def load_split_manifest(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, Mapping):
        raise TypeError("split manifest root must be a mapping")
    return validate_split_manifest(value)


def validate_learning_collection_budget(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an outcome-free external resource allocation for learned data."""

    if value.get("schema_version") != "piu.learning-collection-budget.v1":
        raise ValueError("unsupported learning collection budget")
    if value.get("status") != "FROZEN_BEFORE_SUCCESSOR_COLLECTION":
        raise ValueError("learning collection budget is not prospectively frozen")
    if (
        value.get("outcomes_loaded") is not False
        or value.get("model_predictions_loaded") is not False
    ):
        raise ValueError("learning collection budget inspected outcomes or predictions")
    counts = value.get("groups_per_role")
    if not isinstance(counts, Mapping) or set(counts) != set(LEARNING_SPLIT_ROLES):
        raise ValueError("learning budget must allocate every learned-data role")
    normalized_counts = {}
    for role in LEARNING_SPLIT_ROLES:
        count = counts[role]
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("learning group counts must be positive integers")
        normalized_counts[role] = count
    seed_start = value.get("seed_start")
    if not isinstance(seed_start, int) or isinstance(seed_start, bool) or seed_start < 0:
        raise ValueError("learning budget requires a nonnegative seed start")
    authority = " ".join(str(value.get("authority", "")).split())
    rationale = " ".join(str(value.get("rationale", "")).split())
    scenario = " ".join(str(value.get("scenario", "")).split())
    if not authority or not rationale or not scenario:
        raise ValueError("learning budget requires external authority and rationale")
    return {
        **dict(value),
        "groups_per_role": normalized_counts,
        "seed_start": seed_start,
        "authority": authority,
        "rationale": rationale,
        "scenario": scenario,
    }


def load_learning_collection_budget(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, Mapping):
        raise TypeError("learning collection budget root must be a mapping")
    return validate_learning_collection_budget(value)


def assignment_for(
    manifest: Mapping[str, Any], initial_state_group: str
) -> dict[str, Any]:
    matches = [
        row
        for row in manifest["assignments"]
        if row["initial_state_group"] == initial_state_group
    ]
    if len(matches) != 1:
        raise ValueError(
            "initial-state group is absent or duplicated in split manifest"
        )
    return dict(matches[0])


def role_to_split(role: str) -> Split:
    if role.startswith("calibration_") or role.startswith("effect_calibration_"):
        return Split.CALIBRATION
    return Split(role)


def effect_collection_role(
    manifest: Mapping[str, Any], *, initial_state_group: str, split: Split
) -> str:
    """Validate the prospective role of one counterfactual-effect decision."""

    assignment = assignment_for(manifest, initial_state_group)
    role = assignment["split_role"]
    if role_to_split(role) is not split:
        raise ValueError("transition split differs from its prospective group role")
    if split is Split.CALIBRATION and role not in EFFECT_CALIBRATION_ROLES:
        raise ValueError(
            "effect collection requires independent effect-calibration groups"
        )
    if split not in {Split.TRAIN, Split.DEVELOPMENT, Split.CALIBRATION, Split.SEALED_TEST}:
        raise ValueError("effect collection received a non-mainline split")
    return role
