#!/usr/bin/env python3
"""Run label-free B3/B4 route decisions without a calibration gate."""

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

from piu.ablation_controller import decide_unique_argmax
from piu.action_effect import CandidateConditionedEffectPredictor, build_effect_inputs
from piu.contracts import Split
from piu.executor_bridge import load_public_candidate, serialize_pi05_subtask


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
        raise ValueError(f"unsupported ablation-controller input {schema}")
    if sha256(path) != report["output"]["sha256"]:
        raise ValueError("ablation-controller input differs from its report")
    with np.load(path) as store:
        return {name: np.asarray(store[name]) for name in store.files}


def verify_sealed_authorization(args: argparse.Namespace) -> None:
    if args.sealed_authorization is None:
        raise ValueError("sealed uncalibrated controller requires authorization")
    value = json.loads(args.sealed_authorization.read_text())
    if (
        value.get("schema_version")
        != "piu.uncalibrated-controller-sealed-authorization.v1"
    ):
        raise ValueError("invalid uncalibrated-controller authorization")
    expected = {
        "method_id": args.method_id,
        "checkpoint_sha256": sha256(args.checkpoint),
        "feature_sha256": sha256(args.features),
        "binding_prediction_sha256": sha256(args.binding_predictions),
        "single_use_output": portable(args.output),
    }
    for name, required in expected.items():
        if value.get(name) != required:
            raise ValueError(f"uncalibrated authorization differs at {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-id", choices=("B3", "B4"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--binding-predictions", type=Path, required=True)
    parser.add_argument("--binding-report", type=Path, required=True)
    parser.add_argument(
        "--expected-split",
        choices=("train", "development", "calibration", "sealed_test"),
        required=True,
    )
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
        "output",
    ):
        setattr(args, name, resolve(getattr(args, name)))
    if args.sealed_authorization is not None:
        args.sealed_authorization = resolve(args.sealed_authorization)
    if args.output.exists():
        raise FileExistsError("uncalibrated controller reports are immutable")
    expected_split = Split(args.expected_split)
    if expected_split is Split.SEALED_TEST:
        verify_sealed_authorization(args)
    elif args.sealed_authorization is not None:
        raise ValueError("non-sealed ablation cannot consume sealed authorization")
    features = load_npz(
        args.features, args.feature_report, schema="piu.spatial-prefix-features.v1"
    )
    binding = load_npz(
        args.binding_predictions,
        args.binding_report,
        schema="piu.target-binder-online-predictions.v1",
    )
    training = json.loads(args.training_report.read_text())
    if training.get("schema_version") != "piu.action-effect-training.v1":
        raise ValueError("unsupported action-effect training report")

    import torch

    if torch.cuda.is_available():
        raise RuntimeError("CUDA remained visible during ablation inference")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    expected_variant = "route_only" if args.method_id == "B3" else "joint_effect"
    if checkpoint.get("variant") != expected_variant:
        raise ValueError("B3/B4 checkpoint variant differs from method registry")
    if (
        sha256(args.checkpoint)
        != training["variants"][expected_variant]["checkpoint"]["sha256"]
    ):
        raise ValueError("ablation checkpoint differs from training report")
    values = build_effect_inputs(
        feature_arrays=features,
        binding_predictions=binding,
        action_vocabulary=checkpoint["action_vocabulary"],
    )
    if set(values.split) != {expected_split}:
        raise ValueError("ablation-controller input split mismatch")
    groups = set(values.initial_state_group)
    model_groups = training["initial_state_groups"]
    if expected_split in {Split.TRAIN, Split.DEVELOPMENT}:
        if groups - set(model_groups[expected_split.value]):
            raise ValueError("ablation groups differ from training report")
    elif groups & set(model_groups["train"] + model_groups["development"]):
        raise ValueError("ablation evaluation overlaps model-selection groups")
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
            effect_backprop_to_shared=expected_variant == "joint_effect",
            route_use_predicted_effects=expected_variant != "route_only",
        )
    route_logits = outputs["route_logits"].cpu().numpy()
    decisions = []
    for index, sample_id in enumerate(values.sample_id):
        decision = decide_unique_argmax(
            route_logits=route_logits[index],
            candidate_valid_mask=values.candidate_valid_mask[index],
            candidate_id=values.candidate_id[index],
            candidate_primitive=values.candidate_primitive[index],
        )
        public_candidates = [
            load_public_candidate(values.candidate_payload[index, candidate_index])
            for candidate_index in np.flatnonzero(values.candidate_valid_mask[index])
        ]
        selected = next(
            (
                candidate
                for candidate in public_candidates
                if candidate["candidate_id"] == decision.selected_candidate_id
            ),
            None,
        )
        structured = (
            serialize_pi05_subtask(selected, spatial_references=())
            if selected is not None
            else None
        )
        decisions.append(
            {
                "sample_id": sample_id,
                "initial_state_group": values.initial_state_group[index],
                "decision_kind": decision.kind.value,
                "selected_candidate_id": decision.selected_candidate_id,
                "selected_candidate_primitive": decision.selected_candidate_primitive,
                "selected_candidate": selected,
                "public_candidates": public_candidates,
                "reason": decision.reason,
                "structured_pi05_subtask": structured,
                "spatial_references": [],
            }
        )
    report = {
        "schema_version": "piu.uncalibrated-ablation-controller-report.v1",
        "method_id": args.method_id,
        "variant": expected_variant,
        "claim_scope": "PUBLIC_INPUT_ABLATION_NOT_CALIBRATED_PERFORMANCE",
        "split": expected_split.value,
        "initial_state_groups": sorted(groups),
        "inputs": {
            name: {"path": portable(path), "sha256": sha256(path)}
            for name, path in {
                "checkpoint": args.checkpoint,
                "training_report": args.training_report,
                "features": args.features,
                "feature_report": args.feature_report,
                "binding_predictions": args.binding_predictions,
                "binding_report": args.binding_report,
            }.items()
        },
        "decision_rule": "unique_argmax_exact_ties_abstain",
        "effect_channel_used": args.method_id == "B4",
        "calibration_loaded": False,
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
