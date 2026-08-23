"""Machine-checkable provenance rules for research-method constants.

The registry is intentionally stricter than a configuration schema.  It asks
whether a number or rule is allowed to affect a public-input method claim, not
merely whether the value has the right type.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from hashlib import sha256
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


def _lookup_dotted(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


def _normalized_claim_text(value: str) -> str:
    """Normalize prose without weakening exact semantic-fragment checks."""

    return " ".join(value.casefold().split())


def audit_claim_surfaces(
    registry: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    """Reject retired interpretations on public claim surfaces.

    This is intentionally a semantic regression list, not a general-purpose
    prose linter. Each forbidden fragment corresponds to a previously observed
    claim error with an explicit replacement in the provenance registry.
    """

    surfaces = registry.get("claim_surfaces", [])
    retired = registry.get("retired_claim_fragments", [])
    if not isinstance(surfaces, list) or not surfaces:
        raise ValueError("provenance registry must declare public claim surfaces")
    if not isinstance(retired, list) or not retired:
        raise ValueError("provenance registry must declare retired claim fragments")

    retired_ids: set[str] = set()
    retired_rows: list[tuple[str, str]] = []
    for row in retired:
        if not isinstance(row, Mapping):
            raise TypeError("every retired claim fragment must be a mapping")
        missing = {"id", "fragment", "replacement"} - set(row)
        if missing:
            raise ValueError(f"retired claim fragment is missing {sorted(missing)}")
        identifier = str(row["id"])
        if identifier in retired_ids:
            raise ValueError(f"duplicate retired claim fragment id {identifier!r}")
        retired_ids.add(identifier)
        fragment = _normalized_claim_text(str(row["fragment"]))
        if not fragment or not str(row["replacement"]).strip():
            raise ValueError(f"{identifier}: fragment and replacement are required")
        retired_rows.append((identifier, fragment))

    reports: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for surface in surfaces:
        if not isinstance(surface, Mapping):
            raise TypeError("every claim surface must be a mapping")
        path_text = str(surface.get("path", ""))
        if not path_text or path_text in seen_paths:
            raise ValueError(f"invalid or duplicate claim surface {path_text!r}")
        seen_paths.add(path_text)
        path = repository_root / path_text
        if not path.is_file():
            raise ValueError(f"claim surface does not exist: {path}")
        raw = path.read_text()
        normalized = _normalized_claim_text(raw)
        forbidden_matches = [
            identifier
            for identifier, fragment in retired_rows
            if fragment in normalized
        ]
        if forbidden_matches:
            raise ValueError(
                f"{path}: retired claim semantics {sorted(forbidden_matches)}"
            )
        required = surface.get("required_fragments", [])
        if not isinstance(required, list) or not required:
            raise ValueError(f"{path}: required_fragments must be nonempty")
        missing_required = [
            str(fragment)
            for fragment in required
            if _normalized_claim_text(str(fragment)) not in normalized
        ]
        if missing_required:
            raise ValueError(
                f"{path}: missing claim-boundary fragments {missing_required}"
            )
        reports.append(
            {
                "path": path_text,
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "required_fragments_verified": len(required),
                "retired_fragments_matched": [],
            }
        )
    return {
        "claim_surfaces": reports,
        "retired_fragments_checked": len(retired_rows),
    }


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
        raw_identifiers = tracked.get("numeric_identifiers", {})
        if isinstance(raw_identifiers, Mapping):
            identifier_controls = {
                str(key): str(value) for key, value in raw_identifiers.items()
            }
        elif isinstance(raw_identifiers, list):
            identifier_controls = {str(value): None for value in raw_identifiers}
        else:
            raise TypeError(f"{protocol}: numeric_identifiers must be a mapping")
        identifiers = tuple(identifier_controls)
        numeric = _numeric_paths(config)
        unclassified = {
            numeric_path
            for numeric_path in numeric
            if not any(
                _matches_prefix(numeric_path, control_path)
                for control_path in controls
            )
            and not any(_matches_prefix(numeric_path, prefix) for prefix in identifiers)
        }
        if unclassified:
            raise ValueError(
                f"{protocol}: unclassified numeric controls {sorted(unclassified)}"
            )
        unused_controls = {
            control_path
            for control_path in (*controls, *identifier_controls)
            if not any(_matches_prefix(path, control_path) for path in numeric)
        }
        if unused_controls:
            raise ValueError(
                f"{protocol}: numeric declarations match no value "
                f"{sorted(unused_controls)}"
            )
        unknown_entries = (
            set(controls.values())
            | {value for value in identifier_controls.values() if value is not None}
        ) - seen
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
    audit_claim_surfaces(registry, repository_root=repository_root)
    return registry


def build_claim_audit_report(
    registry_path: Path, *, repository_root: Path
) -> dict[str, Any]:
    """Build a deterministic evidence-bound report for the public prose audit."""

    registry = load_and_validate_registry(registry_path)
    surface_report = audit_claim_surfaces(
        registry, repository_root=repository_root
    )
    entries = registry["entries"]
    provenance_counts = Counter(str(row["provenance"]) for row in entries)
    claim_use_counts = Counter(str(row["claim_use"]) for row in entries)
    numeric_protocols: list[dict[str, Any]] = []
    for tracked in registry.get("tracked_protocols", []):
        protocol_path = repository_root / str(tracked["path"])
        protocol = yaml.safe_load(protocol_path.read_text())
        controls = [
            {
                "field": str(field),
                "value": _lookup_dotted(protocol, str(field)),
                "provenance_id": str(provenance_id),
            }
            for field, provenance_id in dict(
                tracked.get("numeric_controls", {})
            ).items()
        ]
        identifiers = [
            {
                "field": str(field),
                "value": _lookup_dotted(protocol, str(field)),
                "provenance_id": str(provenance_id),
            }
            for field, provenance_id in dict(
                tracked.get("numeric_identifiers", {})
            ).items()
        ]
        numeric_protocols.append(
            {
                "path": str(tracked["path"]),
                "sha256": sha256(protocol_path.read_bytes()).hexdigest(),
                "controls": controls,
                "identifiers": identifiers,
                "unclassified_numeric_paths": [],
            }
        )
    return {
        "schema_version": "piu.public-claim-audit.v1",
        "status": "PASS",
        "registry": {
            "path": str(registry_path.resolve().relative_to(repository_root.resolve())),
            "sha256": sha256(registry_path.read_bytes()).hexdigest(),
            "entry_count": len(entries),
            "provenance_counts": dict(sorted(provenance_counts.items())),
            "claim_use_counts": dict(sorted(claim_use_counts.items())),
            "tracked_protocol_count": len(registry.get("tracked_protocols", [])),
            "deprecated_protocol_count": len(
                registry.get("deprecated_protocols", [])
            ),
        },
        **surface_report,
        "numeric_protocols": numeric_protocols,
        "claim_boundary": {
            "retired_interpretation_present": False,
            "compound_direct_relabelled_as_pick_or_place": False,
            "historical_visibility_marker_used_as_recognition_threshold": False,
            "missing_successor_evidence_imputed": False,
            "audit_is_performance_evidence": False,
        },
    }
