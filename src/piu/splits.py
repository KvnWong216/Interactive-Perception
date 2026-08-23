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
OPTIONAL_SPLIT_ROLES = ("primitive_qualification", "oracle_formal")


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
        if role not in {*SPLIT_ROLES, *OPTIONAL_SPLIT_ROLES}:
            raise ValueError(f"unsupported split role {role!r}")
        groups.add(group)
        seeds.add(seed)
        roles.add(role)
        normalized.append(
            {"initial_state_group": group, "seed": seed, "split_role": role}
        )
    if not set(SPLIT_ROLES) <= roles:
        raise ValueError("split manifest must allocate every isolated split role")
    return {**dict(value), "scenario": scenario, "assignments": normalized}


def load_split_manifest(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, Mapping):
        raise TypeError("split manifest root must be a mapping")
    return validate_split_manifest(value)


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
    if role.startswith("calibration_"):
        return Split.CALIBRATION
    return Split(role)
