#!/usr/bin/env python3
"""Run label-free calibrated PIU decisions from public cached features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.action_effect import CandidateConditionedEffectPredictor, build_effect_inputs
from piu.binding_calibration import apply_binding_calibration
from piu.calibrated_controller import (
    ControllerBeliefSets,
    DecisionKind,
    decide_calibrated_sample,
)
from piu.contracts import Split
from piu.effect_calibration import apply_effect_calibration
from piu.executor_bridge import (
    current_spatial_references,
    load_public_candidate,
    serialize_pi05_subtask,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_npz(path: Path, report_path: Path, *, schema: str) -> dict[str, np.ndarray]:
    report = json.loads(report_path.read_text())
    if report.get("schema_version") != schema:
        raise ValueError(f"unsupported controller input report {schema}")
    if sha256(path) != report["output"]["sha256"]:
        raise ValueError("controller input differs from its report")
    with np.load(path) as store:
        return {name: np.asarray(store[name]) for name in store.files}


def _boolean_set(values: object) -> frozenset[bool]:
    if not isinstance(values, list) or any(
        not isinstance(item, bool) for item in values
    ):
        raise TypeError("public controller state sets must be boolean lists")
    return frozenset(values)


def load_public_state_sets(path: Path) -> dict[str, dict[str, frozenset[bool]]]:
    result = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("schema_version") != "piu.controller-public-state-sets.v1":
            raise ValueError("unsupported controller public-state schema")
        if (
            row.get("public_inputs_only") is not True
            or row.get("online_oracle_inputs") != []
        ):
            raise ValueError("controller state sets must be public-input only")
        sample_id = " ".join(str(row.get("sample_id", "")).split())
        if not sample_id or sample_id in result:
            raise ValueError("controller state sample IDs must be nonempty and unique")
        state_sets = row.get("state_sets", {})
        if set(state_sets) != {"search_coverage_sufficient"}:
            raise ValueError("controller public-state set family differs")
        result[sample_id] = {
            name: _boolean_set(state_sets[name]) for name in state_sets
        }
    if not result:
        raise ValueError("controller public-state file is empty")
    return result


def verify_sealed_authorization(args: argparse.Namespace, output: Path) -> None:
    if args.sealed_authorization is None:
        raise ValueError("sealed controller inference requires authorization")
    value = json.loads(args.sealed_authorization.read_text())
    if value.get("schema_version") != "piu.controller-sealed-authorization.v1":
        raise ValueError("invalid sealed controller authorization")
    expected = {
        "checkpoint_sha256": sha256(args.checkpoint),
        "feature_sha256": sha256(args.features),
        "binding_prediction_sha256": sha256(args.binding_predictions),
        "binder_calibration_sha256": sha256(args.binder_calibration),
        "effect_calibration_sha256": sha256(args.effect_calibration),
        "public_state_sets_sha256": sha256(args.public_state_sets),
        "method_id": args.method_id,
        "single_use_output": portable(output),
    }
    for name, required in expected.items():
        if value.get(name) != required:
            raise ValueError(f"sealed controller authorization differs at {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method-id", choices=("B5", "B8"), default="B8")
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--binding-predictions", type=Path, required=True)
    parser.add_argument("--binding-report", type=Path, required=True)
    parser.add_argument("--binder-calibration", type=Path, required=True)
    parser.add_argument("--effect-calibration", type=Path, required=True)
    parser.add_argument("--public-state-sets", type=Path, required=True)
    parser.add_argument(
        "--expected-split",
        choices=("train", "development", "calibration", "sealed_test"),
        required=True,
    )
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--sealed-authorization", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "checkpoint",
        "training_report",
        "features",
        "feature_report",
        "binding_predictions",
        "binding_report",
        "binder_calibration",
        "effect_calibration",
        "public_state_sets",
        "output",
    ):
        setattr(args, name, resolve(getattr(args, name)))
    if args.sealed_authorization is not None:
        args.sealed_authorization = resolve(args.sealed_authorization)
    if args.output.exists():
        raise FileExistsError("calibrated controller reports are immutable")
    expected_split = Split(args.expected_split)
    if expected_split is Split.SEALED_TEST:
        verify_sealed_authorization(args, args.output)
    elif args.sealed_authorization is not None:
        raise ValueError("non-sealed controller cannot consume sealed authorization")
    features = load_npz(
        args.features, args.feature_report, schema="piu.spatial-prefix-features.v1"
    )
    binding = load_npz(
        args.binding_predictions,
        args.binding_report,
        schema="piu.target-binder-online-predictions.v1",
    )
    training_report = json.loads(args.training_report.read_text())
    if training_report.get("schema_version") != "piu.action-effect-training.v1":
        raise ValueError("unsupported effect training report")
    effect_calibration = json.loads(args.effect_calibration.read_text())
    if effect_calibration.get("schema_version") != "piu.action-effect-calibration.v1":
        raise ValueError("unsupported effect calibration artifact")
    binder_calibration = json.loads(args.binder_calibration.read_text())
    if binder_calibration.get("schema_version") != "piu.target-binder-calibration.v1":
        raise ValueError("unsupported binder calibration artifact")
    binding_report = json.loads(args.binding_report.read_text())
    binder_checkpoint_hash = binding_report["inputs"]["checkpoint"]["sha256"]
    if binder_calibration["checkpoint_sha256"] != binder_checkpoint_hash:
        raise ValueError(
            "online binding scores and calibration use different checkpoints"
        )

    import torch

    if torch.cuda.is_available():
        raise RuntimeError("CUDA remained visible after controller isolation")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "piu.action-effect-checkpoint.v1":
        raise ValueError("unsupported effect checkpoint")
    variant = str(checkpoint["variant"])
    if (
        sha256(args.checkpoint)
        != training_report["variants"][variant]["checkpoint"]["sha256"]
    ):
        raise ValueError("controller checkpoint differs from training report")
    if (
        sha256(args.checkpoint) != effect_calibration["checkpoint_sha256"]
        or effect_calibration["variant"] != variant
    ):
        raise ValueError("controller calibration differs from its checkpoint")
    values = build_effect_inputs(
        feature_arrays=features,
        binding_predictions=binding,
        action_vocabulary=checkpoint["action_vocabulary"],
    )
    if set(values.split) != {expected_split}:
        raise ValueError("controller input split mismatch")
    groups = set(values.initial_state_group)
    model_selection = training_report["initial_state_groups"]
    if expected_split in {Split.TRAIN, Split.DEVELOPMENT}:
        if groups - set(model_selection[expected_split.value]):
            raise ValueError("controller groups differ from the training report")
    elif groups & set(model_selection["train"] + model_selection["development"]):
        raise ValueError("controller evaluation groups overlap model selection")
    if expected_split is Split.SEALED_TEST:
        calibration_groups = set(
            effect_calibration["initial_state_groups"]["temperature"]
            + effect_calibration["initial_state_groups"]["conformal"]
        )
        if groups & calibration_groups:
            raise ValueError("sealed controller groups overlap calibration")
        binder_calibration_groups = set(
            binder_calibration["initial_state_groups"]["temperature"]
            + binder_calibration["initial_state_groups"]["conformal"]
        )
        if groups & binder_calibration_groups:
            raise ValueError("sealed controller groups overlap binder calibration")
    public_states = load_public_state_sets(args.public_state_sets)
    if set(public_states) != set(values.sample_id):
        raise ValueError("controller public states and feature samples differ")
    model = CandidateConditionedEffectPredictor(**checkpoint["model"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.cpu().eval()
    with torch.no_grad():
        outputs = model(
            torch.as_tensor(values.belief_token),
            torch.as_tensor(values.candidate_prompt_tokens),
            candidate_prompt_valid_mask=torch.as_tensor(
                values.candidate_prompt_valid_mask
            ),
            candidate_valid_mask=torch.as_tensor(values.candidate_valid_mask),
            candidate_action_ids=torch.as_tensor(values.candidate_action_id),
            effect_backprop_to_shared=variant == "joint_effect",
            route_use_predicted_effects=variant != "route_only",
        )
    scores = {
        "route_logits": outputs["route_logits"].cpu().numpy(),
        "candidate_valid_mask": values.candidate_valid_mask,
        "factor_logits": outputs["factor_logits"].cpu().numpy(),
    }
    alpha = float(
        effect_calibration["primary_alpha"] if args.alpha is None else args.alpha
    )
    if alpha not in [float(item) for item in effect_calibration["reported_alpha"]]:
        raise ValueError("controller alpha was not preregistered in calibration")
    if alpha not in [
        float(item) for item in binder_calibration["risk_contract"]["reported_alpha"]
    ]:
        raise ValueError("controller alpha was not fitted for the binder")
    calibrated = apply_effect_calibration(scores, effect_calibration, alpha=alpha)
    calibrated_binding = apply_binding_calibration(
        binding, binder_calibration, alpha=alpha
    )
    feature_report = json.loads(args.feature_report.read_text())
    camera_names = tuple(feature_report["layout"]["camera_names"])
    decisions = []

    def members(name: str, sample_index: int) -> list[bool]:
        membership = calibrated_binding[name][sample_index]
        return [
            label
            for label, included in zip((False, True), membership, strict=True)
            if included
        ]

    for index, sample_id in enumerate(values.sample_id):
        public_candidates = []
        for candidate_index in np.flatnonzero(values.candidate_valid_mask[index]):
            candidate = load_public_candidate(
                values.candidate_payload[index, candidate_index]
            )
            if (
                str(candidate["candidate_id"])
                != values.candidate_id[index, candidate_index]
                or str(candidate["primitive"]).upper()
                != values.candidate_primitive[index, candidate_index].upper()
            ):
                raise ValueError(
                    "public candidate payload identity differs from arrays"
                )
            public_candidates.append(candidate)
        public_candidate_by_id = {
            str(candidate["candidate_id"]): candidate for candidate in public_candidates
        }
        if len(public_candidate_by_id) != len(public_candidates):
            raise ValueError("public candidate payload IDs are not unique")
        spatial_references = current_spatial_references(
            prediction_set=calibrated_binding["spatial_prediction_set"][index],
            patch_xy=binding["patch_xy"][index],
            camera_id=binding["camera_id"][index],
            temporal_id=binding["temporal_id"][index],
            camera_names=camera_names,
        )
        bridged_spatial_references = (
            spatial_references if args.method_id == "B8" else ()
        )
        state = public_states[sample_id]
        beliefs = ControllerBeliefSets.from_mapping(
            {
                "task_sufficiency": members("task_sufficiency_prediction_set", index),
                "target_presence": members("target_presence_prediction_set", index),
                "spatial_reference_available": [bool(spatial_references)],
                "holding_requested_target": members(
                    "holding_requested_target_prediction_set", index
                ),
                "search_coverage_sufficient": list(state["search_coverage_sufficient"]),
                "task_complete": members("task_complete_prediction_set", index),
            }
        )
        decision = decide_calibrated_sample(
            candidate_id=values.candidate_id,
            candidate_primitive=values.candidate_primitive,
            candidate_valid_mask=values.candidate_valid_mask,
            calibrated=calibrated,
            sample_index=index,
            belief_sets=beliefs,
            require_spatial_reference=args.method_id == "B8",
        )
        structured_subtask = None
        selected_primitive = None
        selected_candidate = None
        if decision.kind in {DecisionKind.EXECUTE, DecisionKind.INTERACT}:
            candidate_index = next(
                candidate_index
                for candidate_index in np.flatnonzero(
                    values.candidate_valid_mask[index]
                )
                if values.candidate_id[index, candidate_index]
                == decision.selected_candidate_id
            )
            selected_candidate = public_candidate_by_id[
                str(decision.selected_candidate_id)
            ]
            selected_primitive = str(selected_candidate["primitive"]).upper()
            structured_subtask = serialize_pi05_subtask(
                selected_candidate,
                spatial_references=(
                    bridged_spatial_references
                    if selected_primitive in {"PICK", "DIRECT"}
                    else ()
                ),
            )
        decisions.append(
            {
                "sample_id": sample_id,
                "initial_state_group": values.initial_state_group[index],
                "decision_kind": decision.kind.value,
                "selected_candidate_id": decision.selected_candidate_id,
                "selected_candidate_primitive": selected_primitive,
                "selected_candidate": selected_candidate,
                "public_candidates": public_candidates,
                "route_prediction_set": list(decision.route_prediction_set),
                "reason": decision.reason,
                "diagnostic_values": dict(decision.diagnostic_values),
                "structured_pi05_subtask": structured_subtask,
                "spatial_references": [
                    {
                        "camera": item.camera,
                        "selected_patch_indices": list(item.selected_patch_indices),
                        "x_interval": list(item.x_interval),
                        "y_interval": list(item.y_interval),
                    }
                    for item in bridged_spatial_references
                ],
            }
        )
    report = {
        "schema_version": "piu.calibrated-controller-report.v1",
        "claim_scope": "PUBLIC_INPUT_DECISIONS_NOT_PHYSICAL_OUTCOMES",
        "method_id": args.method_id,
        "split": expected_split.value,
        "initial_state_groups": sorted(groups),
        "alpha": alpha,
        "variant": variant,
        "spatial_bridge_enabled": args.method_id == "B8",
        "inputs": {
            name: {"path": portable(path), "sha256": sha256(path)}
            for name, path in {
                "checkpoint": args.checkpoint,
                "training_report": args.training_report,
                "features": args.features,
                "feature_report": args.feature_report,
                "binding_predictions": args.binding_predictions,
                "binding_report": args.binding_report,
                "binder_calibration": args.binder_calibration,
                "effect_calibration": args.effect_calibration,
                "public_state_sets": args.public_state_sets,
            }.items()
        },
        "evaluator_labels_loaded": False,
        "cuda_visible_to_controller": torch.cuda.is_available(),
        "decisions": decisions,
        "paper_method_claim_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
