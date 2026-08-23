"""Immutable, split-bound assembly of prospective PIU role datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .action_effect import load_effect_labels
from .binding_data import load_binding_labels
from .contracts import load_public_transitions
from .splits import assignment_for, load_split_manifest, role_to_split


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path, *, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def artifact(path: Path, *, repository_root: Path) -> dict[str, str]:
    return {
        "path": portable(path, repository_root=repository_root),
        "sha256": sha256(path),
    }


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"JSONL source is empty or malformed: {path}")
    return [dict(row) for row in rows]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _validate_role_groups(
    *, groups: set[str], split_role: str, split_manifest: Mapping[str, Any]
) -> str:
    expected_split = role_to_split(split_role).value
    if not groups:
        raise ValueError("role dataset has no initial-state groups")
    for group in groups:
        assignment = assignment_for(split_manifest, group)
        if assignment["split_role"] != split_role:
            raise ValueError(
                f"group {group!r} belongs to {assignment['split_role']!r}, "
                f"not {split_role!r}"
            )
    return expected_split


def assemble_public_binding_role(
    *,
    public_sources: Sequence[Path],
    binding_label_sources: Sequence[Path],
    split_manifest_path: Path,
    split_role: str,
    output_public: Path,
    output_labels: Path,
    output_manifest: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Assemble exact public/label rows for one prospectively assigned role."""

    outputs = (output_public, output_labels, output_manifest)
    if any(path.exists() for path in outputs):
        raise FileExistsError("public-binding role outputs are immutable")
    if not public_sources or not binding_label_sources:
        raise ValueError("public and binding-label sources must be nonempty")
    split_manifest = load_split_manifest(split_manifest_path)
    public_rows = [row for path in public_sources for row in _jsonl_rows(path)]
    label_rows = [row for path in binding_label_sources for row in _jsonl_rows(path)]
    public = [row for path in public_sources for row in load_public_transitions(path)]
    labels = [row for path in binding_label_sources for row in load_binding_labels(path)]
    public_ids = [row.sample_id for row in public]
    label_ids = [row.sample_id for row in labels]
    if len(set(public_ids)) != len(public_ids) or len(set(label_ids)) != len(label_ids):
        raise ValueError("role dataset contains duplicate sample IDs")
    if set(public_ids) != set(label_ids):
        raise ValueError("public transitions and binding labels have different samples")
    public_by_id = {row.sample_id: row for row in public}
    label_by_id = {row.sample_id: row for row in labels}
    groups = {row.initial_state_group for row in public}
    expected_split = _validate_role_groups(
        groups=groups, split_role=split_role, split_manifest=split_manifest
    )
    for sample_id in public_ids:
        transition = public_by_id[sample_id]
        label = label_by_id[sample_id]
        if (
            transition.split.value != expected_split
            or label.split.value != expected_split
            or transition.initial_state_group != label.initial_state_group
        ):
            raise ValueError("public/binding row group or split differs from role")
    public_order = sorted(public_rows, key=lambda row: str(row["sample_id"]))
    label_order = sorted(label_rows, key=lambda row: str(row["sample_id"]))
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_public.parent.mkdir(parents=True, exist_ok=True)
    output_labels.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_public, public_order)
    _write_jsonl(output_labels, label_order)
    manifest = {
        "schema_version": "piu.public-binding-role-dataset.v1",
        "claim_scope": "PROSPECTIVE_ROLE_DATA_NOT_PERFORMANCE_EVIDENCE",
        "split_role": split_role,
        "split": expected_split,
        "initial_state_groups": sorted(groups),
        "samples": len(public_ids),
        "split_manifest": artifact(split_manifest_path, repository_root=repository_root),
        "sources": {
            "public": [artifact(path, repository_root=repository_root) for path in public_sources],
            "binding_labels": [
                artifact(path, repository_root=repository_root)
                for path in binding_label_sources
            ],
        },
        "outputs": {
            "public": artifact(output_public, repository_root=repository_root),
            "binding_labels": artifact(output_labels, repository_root=repository_root),
        },
        "sample_join_exact": True,
        "evaluator_labels_in_public_output": False,
        "paper_method_claim_allowed": False,
    }
    output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _verified_label_manifest(
    path: Path, *, repository_root: Path
) -> tuple[Path, dict[str, Any]]:
    value = json.loads(path.read_text())
    if value.get("schema_version") != "piu.action-effect-label-manifest.v1":
        raise ValueError("unsupported action-effect label manifest")
    reference = value.get("labels", {})
    label_path = Path(str(reference.get("path", "")))
    if not label_path.is_absolute():
        label_path = repository_root / label_path
    if not label_path.is_file() or sha256(label_path) != reference.get("sha256"):
        raise ValueError("effect labels differ from their source manifest")
    return label_path, value


def _verified_reference(
    value: Any, *, name: str, repository_root: Path
) -> Path:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an artifact reference")
    path = Path(str(value.get("path", "")))
    if not path.is_absolute():
        path = repository_root / path
    if not path.is_file() or sha256(path) != value.get("sha256"):
        raise ValueError(f"{name} differs from its content hash")
    return path


def validate_public_binding_role_dataset(
    path: Path, *, repository_root: Path
) -> dict[str, Any]:
    """Recompute an assembled public/binding role manifest from its sources."""

    value = json.loads(path.read_text())
    if value.get("schema_version") != "piu.public-binding-role-dataset.v1":
        raise ValueError("unsupported public-binding role dataset")
    split_role = str(value.get("split_role", ""))
    split_path = _verified_reference(
        value.get("split_manifest"),
        name="public-binding split manifest",
        repository_root=repository_root,
    )
    sources = value.get("sources")
    outputs = value.get("outputs")
    if not isinstance(sources, Mapping) or not isinstance(outputs, Mapping):
        raise TypeError("public-binding dataset lacks source/output provenance")
    public_refs = sources.get("public")
    label_refs = sources.get("binding_labels")
    if (
        not isinstance(public_refs, Sequence)
        or isinstance(public_refs, (str, bytes))
        or not public_refs
        or not isinstance(label_refs, Sequence)
        or isinstance(label_refs, (str, bytes))
        or not label_refs
    ):
        raise TypeError("public-binding dataset source lists are malformed")
    public_sources = [
        _verified_reference(
            item,
            name=f"public-binding public source {index}",
            repository_root=repository_root,
        )
        for index, item in enumerate(public_refs)
    ]
    label_sources = [
        _verified_reference(
            item,
            name=f"public-binding label source {index}",
            repository_root=repository_root,
        )
        for index, item in enumerate(label_refs)
    ]
    output_public = _verified_reference(
        outputs.get("public"),
        name="public-binding public output",
        repository_root=repository_root,
    )
    output_labels = _verified_reference(
        outputs.get("binding_labels"),
        name="public-binding label output",
        repository_root=repository_root,
    )
    source_public_rows = [
        row for source in public_sources for row in _jsonl_rows(source)
    ]
    source_label_rows = [
        row for source in label_sources for row in _jsonl_rows(source)
    ]
    if _jsonl_rows(output_public) != sorted(
        source_public_rows, key=lambda row: str(row["sample_id"])
    ) or _jsonl_rows(output_labels) != sorted(
        source_label_rows, key=lambda row: str(row["sample_id"])
    ):
        raise ValueError("public-binding outputs differ from immutable source union")
    public = load_public_transitions(output_public)
    labels = load_binding_labels(output_labels)
    public_ids = [row.sample_id for row in public]
    label_ids = [row.sample_id for row in labels]
    if (
        len(set(public_ids)) != len(public_ids)
        or len(set(label_ids)) != len(label_ids)
        or set(public_ids) != set(label_ids)
    ):
        raise ValueError("public-binding output sample join is not exact")
    split = load_split_manifest(split_path)
    groups = {row.initial_state_group for row in public}
    expected_split = _validate_role_groups(
        groups=groups, split_role=split_role, split_manifest=split
    )
    labels_by_id = {row.sample_id: row for row in labels}
    if any(
        row.split.value != expected_split
        or labels_by_id[row.sample_id].split.value != expected_split
        or labels_by_id[row.sample_id].initial_state_group
        != row.initial_state_group
        for row in public
    ):
        raise ValueError("public-binding output group/split join differs")
    expected = {
        "schema_version": "piu.public-binding-role-dataset.v1",
        "claim_scope": "PROSPECTIVE_ROLE_DATA_NOT_PERFORMANCE_EVIDENCE",
        "split_role": split_role,
        "split": expected_split,
        "initial_state_groups": sorted(groups),
        "samples": len(public_ids),
        "split_manifest": artifact(split_path, repository_root=repository_root),
        "sources": {
            "public": [
                artifact(item, repository_root=repository_root)
                for item in public_sources
            ],
            "binding_labels": [
                artifact(item, repository_root=repository_root)
                for item in label_sources
            ],
        },
        "outputs": {
            "public": artifact(output_public, repository_root=repository_root),
            "binding_labels": artifact(
                output_labels, repository_root=repository_root
            ),
        },
        "sample_join_exact": True,
        "evaluator_labels_in_public_output": False,
        "paper_method_claim_allowed": False,
    }
    if value != expected:
        raise ValueError("public-binding manifest differs from exact recomputation")
    return value


def validate_action_effect_role_dataset(
    path: Path, *, repository_root: Path
) -> dict[str, Any]:
    """Recompute a candidate-complete action-effect role manifest."""

    value = json.loads(path.read_text())
    if value.get("schema_version") != "piu.action-effect-role-dataset.v1":
        raise ValueError("unsupported action-effect role dataset")
    split_role = str(value.get("split_role", ""))
    if split_role not in {
        "train",
        "development",
        "effect_calibration_temperature",
        "effect_calibration_conformal",
    }:
        raise ValueError("action-effect role dataset has an unsupported role")
    split_path = _verified_reference(
        value.get("split_manifest"),
        name="action-effect split manifest",
        repository_root=repository_root,
    )
    binding_path = _verified_reference(
        value.get("public_binding_manifest"),
        name="action-effect public-binding manifest",
        repository_root=repository_root,
    )
    binding = validate_public_binding_role_dataset(
        binding_path, repository_root=repository_root
    )
    public_path = _verified_reference(
        value.get("public"),
        name="action-effect public rows",
        repository_root=repository_root,
    )
    if (
        binding.get("split_role") != split_role
        or binding.get("outputs", {}).get("public")
        != artifact(public_path, repository_root=repository_root)
    ):
        raise ValueError("action-effect public rows differ from binding dataset")
    source_refs = value.get("source_effect_manifests")
    if (
        not isinstance(source_refs, Sequence)
        or isinstance(source_refs, (str, bytes))
        or not source_refs
    ):
        raise TypeError("action-effect source manifests are malformed")
    source_manifests = [
        _verified_reference(
            item,
            name=f"action-effect source manifest {index}",
            repository_root=repository_root,
        )
        for index, item in enumerate(source_refs)
    ]
    source_rows = []
    label_paths = []
    for source in source_manifests:
        label_path, _ = _verified_label_manifest(
            source, repository_root=repository_root
        )
        label_paths.append(label_path)
        source_rows.extend(_jsonl_rows(label_path))
    labels_path = _verified_reference(
        value.get("labels"),
        name="action-effect labels output",
        repository_root=repository_root,
    )
    expected_label_rows = sorted(
        source_rows,
        key=lambda row: (str(row["sample_id"]), str(row["candidate_id"])),
    )
    if _jsonl_rows(labels_path) != expected_label_rows:
        raise ValueError("action-effect labels differ from immutable source union")
    split = load_split_manifest(split_path)
    public = load_public_transitions(public_path)
    groups = {row.initial_state_group for row in public}
    expected_split = _validate_role_groups(
        groups=groups, split_role=split_role, split_manifest=split
    )
    labels = [
        row
        for label_path in label_paths
        for row in load_effect_labels(label_path)
    ]
    keys = [(row.sample_id, row.candidate_id) for row in labels]
    expected_keys = {
        (transition.sample_id, str(candidate["candidate_id"]))
        for transition in public
        for candidate in transition.candidate_actions
    }
    if len(set(keys)) != len(keys) or set(keys) != expected_keys:
        raise ValueError("action-effect labels do not equal the public candidate matrix")
    public_by_id = {row.sample_id: row for row in public}
    correct_by_sample: dict[str, int] = {}
    for row in labels:
        transition = public_by_id[row.sample_id]
        if (
            row.initial_state_group != transition.initial_state_group
            or row.split.value != expected_split
        ):
            raise ValueError("action-effect group/split differs from public rows")
        correct_by_sample[row.sample_id] = correct_by_sample.get(
            row.sample_id, 0
        ) + int(row.selection_correct)
    if set(correct_by_sample) != set(public_by_id) or set(
        correct_by_sample.values()
    ) != {1}:
        raise ValueError("action-effect decisions lack exactly one correct route")
    expected = {
        "schema_version": "piu.action-effect-role-dataset.v1",
        "claim_scope": "EXECUTED_COUNTERFACTUAL_ROLE_DATA_NOT_PERFORMANCE_EVIDENCE",
        "split_role": split_role,
        "split": expected_split,
        "initial_state_groups": sorted(groups),
        "samples": len(public),
        "candidate_rows": len(labels),
        "split_manifest": artifact(split_path, repository_root=repository_root),
        "public_binding_manifest": artifact(
            binding_path, repository_root=repository_root
        ),
        "public": artifact(public_path, repository_root=repository_root),
        "source_effect_manifests": [
            artifact(item, repository_root=repository_root)
            for item in source_manifests
        ],
        "labels": artifact(labels_path, repository_root=repository_root),
        "exact_candidate_matrix": True,
        "one_correct_route_per_sample": True,
        "post_outcomes_excluded_from_decision_features": True,
        "paper_method_claim_allowed": False,
    }
    if value != expected:
        raise ValueError("action-effect manifest differs from exact recomputation")
    return value


def assemble_action_effect_role(
    *,
    public_path: Path,
    public_binding_manifest_path: Path,
    effect_label_manifests: Sequence[Path],
    split_manifest_path: Path,
    split_role: str,
    output_labels: Path,
    output_manifest: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Assemble one exact candidate matrix from per-decision executed forks."""

    if output_labels.exists() or output_manifest.exists():
        raise FileExistsError("action-effect role outputs are immutable")
    if not effect_label_manifests:
        raise ValueError("effect-label manifests must be nonempty")
    allowed = {
        "train",
        "development",
        "effect_calibration_temperature",
        "effect_calibration_conformal",
    }
    if split_role not in allowed:
        raise ValueError("action-effect assembly received an unsupported split role")
    binding_manifest = json.loads(public_binding_manifest_path.read_text())
    if binding_manifest.get("schema_version") != "piu.public-binding-role-dataset.v1":
        raise ValueError("unsupported public-binding role manifest")
    public_ref = binding_manifest.get("outputs", {}).get("public", {})
    if (
        public_ref.get("sha256") != sha256(public_path)
        or binding_manifest.get("split_role") != split_role
    ):
        raise ValueError("public role dataset differs from its binding manifest")
    split_manifest = load_split_manifest(split_manifest_path)
    public = load_public_transitions(public_path)
    expected_split = _validate_role_groups(
        groups={row.initial_state_group for row in public},
        split_role=split_role,
        split_manifest=split_manifest,
    )
    if any(row.split.value != expected_split for row in public):
        raise ValueError("public transition split differs from effect role")
    label_paths = []
    source_manifests = []
    source_values = []
    for manifest_path in effect_label_manifests:
        label_path, value = _verified_label_manifest(
            manifest_path, repository_root=repository_root
        )
        label_paths.append(label_path)
        source_manifests.append(artifact(manifest_path, repository_root=repository_root))
        source_values.extend(_jsonl_rows(label_path))
    labels = [row for path in label_paths for row in load_effect_labels(path)]
    keys = [(row.sample_id, row.candidate_id) for row in labels]
    if len(set(keys)) != len(keys):
        raise ValueError("assembled effect labels contain duplicate candidate keys")
    expected_keys = {
        (transition.sample_id, str(candidate["candidate_id"]))
        for transition in public
        for candidate in transition.candidate_actions
    }
    if set(keys) != expected_keys:
        raise ValueError("effect labels do not equal the public candidate matrix")
    by_sample: dict[str, int] = {}
    public_by_id = {row.sample_id: row for row in public}
    for row in labels:
        transition = public_by_id[row.sample_id]
        if (
            row.initial_state_group != transition.initial_state_group
            or row.split.value != expected_split
        ):
            raise ValueError("effect label group or split differs from public role")
        by_sample[row.sample_id] = by_sample.get(row.sample_id, 0) + int(
            row.selection_correct
        )
    if set(by_sample) != set(public_by_id) or set(by_sample.values()) != {1}:
        raise ValueError("every public decision requires exactly one correct route")
    output_labels.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        source_values,
        key=lambda row: (str(row["sample_id"]), str(row["candidate_id"])),
    )
    _write_jsonl(output_labels, ordered)
    manifest = {
        "schema_version": "piu.action-effect-role-dataset.v1",
        "claim_scope": "EXECUTED_COUNTERFACTUAL_ROLE_DATA_NOT_PERFORMANCE_EVIDENCE",
        "split_role": split_role,
        "split": expected_split,
        "initial_state_groups": sorted(
            {transition.initial_state_group for transition in public}
        ),
        "samples": len(public),
        "candidate_rows": len(labels),
        "split_manifest": artifact(split_manifest_path, repository_root=repository_root),
        "public_binding_manifest": artifact(
            public_binding_manifest_path, repository_root=repository_root
        ),
        "public": artifact(public_path, repository_root=repository_root),
        "source_effect_manifests": source_manifests,
        "labels": artifact(output_labels, repository_root=repository_root),
        "exact_candidate_matrix": True,
        "one_correct_route_per_sample": True,
        "post_outcomes_excluded_from_decision_features": True,
        "paper_method_claim_allowed": False,
    }
    output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
