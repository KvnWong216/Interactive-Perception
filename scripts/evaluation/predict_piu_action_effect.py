#!/usr/bin/env python3
"""Run one frozen PIU route/effect ablation on an isolated CPU split."""

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

from piu.action_effect import (
    CandidateConditionedEffectPredictor,
    join_effect_features,
    load_effect_labels,
)
from piu.contracts import Split
from piu.effect_training import tensor_batch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_npz(path: Path, report_path: Path, *, schema: str):
    report = json.loads(report_path.read_text())
    if report.get("schema_version") != schema:
        raise ValueError(f"unsupported prediction input report {schema}")
    if sha256(path) != report["output"]["sha256"]:
        raise ValueError("prediction input differs from its report")
    with np.load(path) as store:
        arrays = {name: np.asarray(store[name]) for name in store.files}
    return arrays, report


def verify_sealed_authorization(args, output: Path) -> dict:
    if args.sealed_authorization is None:
        raise ValueError("sealed action-effect inference requires authorization")
    value = json.loads(args.sealed_authorization.read_text())
    if value.get("schema_version") != "piu.action-effect-sealed-authorization.v1":
        raise ValueError("invalid action-effect sealed authorization")
    expected = {
        "checkpoint_sha256": sha256(args.checkpoint),
        "feature_sha256": sha256(args.features),
        "binding_prediction_sha256": sha256(args.binding_predictions),
        "effect_label_sha256": sha256(args.labels),
        "single_use_output": portable(output),
    }
    for name, required in expected.items():
        if value.get(name) != required:
            raise ValueError(f"effect sealed authorization differs at {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--binding-predictions", type=Path, required=True)
    parser.add_argument("--binding-report", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--expected-split",
        choices=("train", "development", "calibration", "sealed_test"),
        required=True,
    )
    parser.add_argument(
        "--calibration-role", choices=("temperature", "conformal")
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
        "labels",
        "output",
    ):
        setattr(args, name, resolve(getattr(args, name)))
    if args.sealed_authorization is not None:
        args.sealed_authorization = resolve(args.sealed_authorization)
    report_path = args.output.with_suffix(".json")
    if args.output.exists() or report_path.exists():
        raise FileExistsError("action-effect prediction outputs are immutable")
    expected_split = Split(args.expected_split)
    if expected_split is Split.CALIBRATION and args.calibration_role is None:
        raise ValueError("calibration effect predictions require a role")
    if expected_split is not Split.CALIBRATION and args.calibration_role is not None:
        raise ValueError("only calibration predictions have a calibration role")
    if expected_split is Split.SEALED_TEST:
        authorization = verify_sealed_authorization(args, args.output)
    else:
        if args.sealed_authorization is not None:
            raise ValueError("non-sealed inference cannot consume authorization")
        authorization = None
    feature_values, _feature_report = load_npz(
        args.features,
        args.feature_report,
        schema="piu.spatial-prefix-features.v1",
    )
    binding_values, _binding_report = load_npz(
        args.binding_predictions,
        args.binding_report,
        schema="piu.target-binder-online-predictions.v1",
    )
    training_report = json.loads(args.training_report.read_text())
    if training_report.get("schema_version") != "piu.action-effect-training.v1":
        raise ValueError("unsupported action-effect training report")

    import torch

    if torch.cuda.is_available():
        raise RuntimeError("CUDA remained visible after effect inference isolation")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "piu.action-effect-checkpoint.v1":
        raise ValueError("unsupported action-effect checkpoint")
    variant = str(checkpoint["variant"])
    if sha256(args.checkpoint) != training_report["variants"][variant]["checkpoint"][
        "sha256"
    ]:
        raise ValueError("effect checkpoint differs from training report")
    bundle = join_effect_features(
        feature_arrays=feature_values,
        binding_predictions=binding_values,
        labels=load_effect_labels(args.labels),
        action_vocabulary=checkpoint["action_vocabulary"],
    )
    if set(bundle.split) != {expected_split}:
        raise ValueError("effect prediction bundle split mismatch")
    groups = set(bundle.initial_state_group)
    contract = training_report["initial_state_groups"]
    if expected_split in {Split.TRAIN, Split.DEVELOPMENT}:
        if groups - set(contract[expected_split.value]):
            raise ValueError("effect model-selection groups differ from report")
    elif groups & set(contract["train"] + contract["development"]):
        raise ValueError("effect evaluation groups overlap model-selection groups")
    model = CandidateConditionedEffectPredictor(**checkpoint["model"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.cpu().eval()
    batch = tensor_batch(bundle)
    with torch.no_grad():
        outputs = model(
            batch["belief_token"],
            batch["candidate_prompt_tokens"],
            candidate_prompt_valid_mask=batch["candidate_prompt_valid_mask"],
            candidate_valid_mask=batch["candidate_valid_mask"],
            candidate_action_ids=batch["candidate_action_id"],
            effect_backprop_to_shared=variant == "joint_effect",
            route_use_predicted_effects=variant != "route_only",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_id=np.asarray(bundle.sample_id),
        initial_state_group=np.asarray(bundle.initial_state_group),
        split=np.asarray([value.value for value in bundle.split]),
        candidate_id=bundle.candidate_id,
        candidate_primitive=bundle.candidate_primitive,
        candidate_payload=bundle.candidate_payload,
        candidate_valid_mask=bundle.candidate_valid_mask,
        route_logits=outputs["route_logits"].cpu().numpy(),
        route_target=bundle.route_target,
        factor_logits=outputs["factor_logits"].cpu().numpy(),
        factor_target=bundle.effect_target,
        factor_support_mask=bundle.effect_support_mask,
    )
    report = {
        "schema_version": "piu.action-effect-predictions.v1",
        "claim_scope": (
            "CALIBRATION_ONLY_NOT_EVALUATION_EVIDENCE"
            if expected_split is Split.CALIBRATION
            else (
                "SEALED_TEST_EVALUATION"
                if expected_split is Split.SEALED_TEST
                else "MODEL_SELECTION_FEATURES_NOT_TEST_EVIDENCE"
            )
        ),
        "variant": variant,
        "split": expected_split.value,
        "calibration_role": args.calibration_role,
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
                "labels": args.labels,
            }.items()
        },
        "output": {"path": portable(args.output), "sha256": sha256(args.output)},
        "cuda_visible_to_predictor": torch.cuda.is_available(),
        "sealed_authorization": (
            None
            if authorization is None
            else {
                "path": portable(args.sealed_authorization),
                "sha256": sha256(args.sealed_authorization),
            }
        ),
        "paper_method_claim_allowed": False,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
