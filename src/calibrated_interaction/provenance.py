"""Machine-checkable provenance rules for research-method constants.

The registry is intentionally stricter than a configuration schema.  It asks
whether a number or rule is allowed to affect a public-input method claim, not
merely whether the value has the right type.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

PROVENANCE_KINDS = frozenset(
    {
        "learned_on_training_split",
        "fit_on_isolated_calibration",
        "simulator_or_physics_contract",
        "frozen_policy_contract",
        "user_risk_contract",
        "formal_statistical_procedure",
        "safety_or_resource_budget",
        "numerical_optimization",
        "protocol_identifier",
        "oracle_intervention",
        "baseline_only_heuristic",
        "unsupported_heuristic",
    }
)

CLAIM_USES = frozenset(
    {
        "main_method_online",
        "main_evaluator",
        "diagnostic_only",
        "baseline_only",
        "historical_only",
        "rejected_method_only",
        "safety_only",
    }
)

_MAIN_ONLINE_ALLOWED = frozenset(
    {
        "learned_on_training_split",
        "fit_on_isolated_calibration",
        "simulator_or_physics_contract",
        "frozen_policy_contract",
        "user_risk_contract",
        # Discrete, preregistered type/serialization rules (for example, STOP
        # versus ABSTAIN) are method definitions rather than fitted scores.
        # They remain individually auditable in the registry and may not hide
        # an unsupported numeric threshold or hand-weighted utility.
        "protocol_identifier",
    }
)
_MAIN_EVALUATOR_ALLOWED = frozenset(
    {
        "simulator_or_physics_contract",
        "formal_statistical_procedure",
        "protocol_identifier",
    }
)


def _numeric_paths(value: Any, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.update(_numeric_paths(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.update(_numeric_paths(child, f"{prefix}[{index}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        paths.add(prefix)
    return paths


def _matches_prefix(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith((f"{prefix}[", f"{prefix}."))


def load_and_validate_registry(path: Path) -> dict[str, Any]:
    """Load a provenance registry and reject claim-source violations."""

    registry = yaml.safe_load(path.read_text())
    if registry.get("schema_version") != "calibrated-interaction.provenance.v1":
        raise ValueError(f"unsupported provenance schema in {path}")
    entries = registry.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("provenance registry must contain entries")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TypeError("every provenance entry must be a mapping")
        missing = {
            "id",
            "location",
            "provenance",
            "claim_use",
            "rationale",
            "evidence",
        } - set(entry)
        if missing:
            raise ValueError(f"provenance entry is missing {sorted(missing)}")
        identifier = str(entry["id"])
        if identifier in seen:
            raise ValueError(f"duplicate provenance id {identifier!r}")
        seen.add(identifier)
        provenance = str(entry["provenance"])
        claim_use = str(entry["claim_use"])
        if provenance not in PROVENANCE_KINDS:
            raise ValueError(f"{identifier}: unknown provenance {provenance!r}")
        if claim_use not in CLAIM_USES:
            raise ValueError(f"{identifier}: unknown claim_use {claim_use!r}")
        if not str(entry["rationale"]).strip() or not str(entry["evidence"]).strip():
            raise ValueError(f"{identifier}: rationale and evidence are required")
        if claim_use == "main_method_online" and provenance not in _MAIN_ONLINE_ALLOWED:
            raise ValueError(
                f"{identifier}: {provenance} cannot control a main-method action"
            )
        if claim_use == "main_evaluator" and provenance not in _MAIN_EVALUATOR_ALLOWED:
            raise ValueError(
                f"{identifier}: {provenance} cannot define a main evaluator outcome"
            )
        if provenance == "unsupported_heuristic" and claim_use not in {
            "historical_only",
            "rejected_method_only",
        }:
            raise ValueError(
                f"{identifier}: unsupported heuristic must be historical or rejected"
            )
        if provenance == "oracle_intervention" and claim_use != "diagnostic_only":
            raise ValueError(f"{identifier}: oracle intervention must be diagnostic")
        if provenance == "baseline_only_heuristic" and claim_use != "baseline_only":
            raise ValueError(
                f"{identifier}: baseline heuristic must stay baseline-only"
            )

    repository_root = path.resolve().parents[2]
    if not (repository_root / ".git").exists():
        repository_root = Path.cwd()

    for tracked in registry.get("tracked_protocols", []):
        protocol = repository_root / str(tracked["path"])
        config = yaml.safe_load(protocol.read_text())
        controls = dict(tracked.get("numeric_controls", {}))
        identifiers = tuple(
            str(value) for value in tracked.get("numeric_identifiers", [])
        )
        numeric = _numeric_paths(config)
        unclassified = {
            numeric_path
            for numeric_path in numeric
            if numeric_path not in controls
            and not any(_matches_prefix(numeric_path, prefix) for prefix in identifiers)
        }
        if unclassified:
            raise ValueError(
                f"{protocol}: unclassified numeric controls {sorted(unclassified)}"
            )
        unknown_entries = set(controls.values()) - seen
        if unknown_entries:
            raise ValueError(
                f"{protocol}: controls reference unknown entries "
                f"{sorted(unknown_entries)}"
            )

    deprecated = registry.get("deprecated_protocols", [])
    for row in deprecated:
        protocol = repository_root / str(row["path"])
        if not protocol.is_file():
            raise ValueError(f"deprecated protocol does not exist: {protocol}")
        if row.get("paper_claim_allowed") is not False:
            raise ValueError(
                f"deprecated protocol must forbid paper claims: {protocol}"
            )
    return registry
