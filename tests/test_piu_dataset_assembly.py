from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from piu.dataset_assembly import (
    assemble_action_effect_role,
    assemble_public_binding_role,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _split(path: Path) -> None:
    roles = (
        ("group", "train"),
        ("development", "development"),
        ("binder-temperature", "calibration_temperature"),
        ("binder-conformal", "calibration_conformal"),
        ("effect-temperature", "effect_calibration_temperature"),
        ("effect-conformal", "effect_calibration_conformal"),
        ("sealed", "sealed_test"),
    )
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "piu.group-split-manifest.v1",
                "status": "FROZEN_BEFORE_COLLECTION",
                "allocation_method": "prospective_without_outcome_access",
                "scenario": "fixed drawer",
                "assignments": [
                    {
                        "initial_state_group": group,
                        "seed": 100 + index,
                        "split_role": role,
                    }
                    for index, (group, role) in enumerate(roles)
                ],
            },
            sort_keys=False,
        )
    )


def _public() -> dict:
    observation = {
        "images": {"agentview": {"sha256": "a" * 64}},
        "public_robot_state": [0.0],
    }
    return {
        "schema_version": "piu.public-transition.v1",
        "sample_id": "sample",
        "initial_state_group": "group",
        "split": "train",
        "prompt": "inspect drawer",
        "observations": {
            "pre_interaction": observation,
            "post_interaction": observation,
        },
        "public_action_history": {
            "initial_observation": True,
            "last_executed_candidate": None,
        },
        "candidate_actions": [
            {"candidate_id": "stop", "primitive": "STOP", "target": "task"}
        ],
        "online_oracle_inputs": [],
    }


def _binding() -> dict:
    empty = {"size": [2, 2], "counts": [4], "starts_with": 0}
    return {
        "schema_version": "piu.binding-label.v1",
        "sample_id": "sample",
        "initial_state_group": "group",
        "split": "train",
        "target_mask_policy_resolution_rle": {
            "pre_interaction": {"agentview": empty},
            "post_interaction": {"agentview": empty},
        },
        "target_present_post": False,
        "task_sufficient_post": False,
        "holding_requested_target_post": False,
        "region_confirmed_empty_post": None,
        "task_complete_post": False,
        "executed_action": "INITIAL_OBSERVATION",
        "simulator_teacher_only": True,
    }


def _effect(digest: str) -> dict:
    return {
        "schema_version": "piu.action-effect-label.v1",
        "sample_id": "sample",
        "initial_state_group": "group",
        "split": "train",
        "candidate_id": "stop",
        "candidate_primitive": "STOP",
        "decision_observation_sha256": digest,
        "outcome_observation_sha256": {"pre": digest, "post": digest},
        "selection_correct": True,
        "eligible_for_execution": True,
        "executed": False,
        "exact_null_transition": True,
        "factors": {
            "execution_succeeded": None,
            "task_progress_succeeded": False,
            "task_relevant_change": False,
            "target_revealed": False,
            "identity_resolved_post": None,
            "candidate_rejected": False,
            "region_confirmed_empty": False,
            "task_information_sufficient_post": None,
        },
        "simulator_teacher_only": True,
    }


def test_role_assemblers_bind_split_and_exact_candidate_matrix(tmp_path: Path) -> None:
    split = tmp_path / "split.yaml"
    public_source = tmp_path / "public_source.jsonl"
    label_source = tmp_path / "binding_source.jsonl"
    _split(split)
    _write_jsonl(public_source, [_public()])
    _write_jsonl(label_source, [_binding()])
    public = tmp_path / "train/public.jsonl"
    binding = tmp_path / "train/binding.jsonl"
    binding_manifest = tmp_path / "train/public_binding_manifest.json"
    report = assemble_public_binding_role(
        public_sources=[public_source],
        binding_label_sources=[label_source],
        split_manifest_path=split,
        split_role="train",
        output_public=public,
        output_labels=binding,
        output_manifest=binding_manifest,
        repository_root=tmp_path,
    )
    assert report["sample_join_exact"] is True
    digest = hashlib.sha256(
        json.dumps(
            {
                "images": {
                    "agentview": {"kind": "legacy_file", "sha256": "a" * 64}
                },
                "public_robot_state": [0.0],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    effect_source = tmp_path / "effect_source.jsonl"
    _write_jsonl(effect_source, [_effect(digest)])
    source_manifest = tmp_path / "effect_source.manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": "piu.action-effect-label-manifest.v1",
                "labels": {
                    "path": str(effect_source.relative_to(tmp_path)),
                    "sha256": hashlib.sha256(effect_source.read_bytes()).hexdigest(),
                },
            }
        )
    )
    output = tmp_path / "train/effects.jsonl"
    manifest = tmp_path / "train/effects.manifest.json"
    result = assemble_action_effect_role(
        public_path=public,
        public_binding_manifest_path=binding_manifest,
        effect_label_manifests=[source_manifest],
        split_manifest_path=split,
        split_role="train",
        output_labels=output,
        output_manifest=manifest,
        repository_root=tmp_path,
    )
    assert result["exact_candidate_matrix"] is True
    assert result["one_correct_route_per_sample"] is True
