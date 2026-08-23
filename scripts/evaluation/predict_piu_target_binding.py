#!/usr/bin/env python3
"""Run a frozen PIU binder on one isolated evaluation split using CPU."""

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

from piu.binding_data import join_binding_features, load_binding_labels
from piu.binding_training import tensor_batch
from piu.contracts import Split
from piu.target_binding import PromptConditionedTargetBinder


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


def verify_sealed_authorization(
    path: Path,
    *,
    checkpoint: Path,
    features: Path,
    labels: Path,
    output: Path,
) -> dict:
    value = json.loads(path.read_text())
    if value.get("schema_version") != "piu.sealed-test-authorization.v1":
        raise ValueError("invalid sealed-test authorization schema")
    expected = {
        "checkpoint_sha256": sha256(checkpoint),
        "feature_sha256": sha256(features),
        "label_sha256": sha256(labels),
        "single_use_output": portable(output),
    }
    for name, required in expected.items():
        if value.get(name) != required:
            raise ValueError(f"sealed-test authorization differs at {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--expected-split",
        choices=("train", "development", "calibration", "sealed_test"),
        required=True,
    )
    parser.add_argument("--calibration-role", choices=("temperature", "conformal"))
    parser.add_argument("--sealed-authorization", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "checkpoint",
        "training_report",
        "features",
        "feature_report",
        "labels",
        "output",
    ):
        setattr(args, name, resolve(getattr(args, name)))
    if args.sealed_authorization is not None:
        args.sealed_authorization = resolve(args.sealed_authorization)
    report_path = args.output.with_suffix(".json")
    if args.output.exists() or report_path.exists():
        raise FileExistsError("binding prediction outputs are immutable")
    expected_split = Split(args.expected_split)
    if expected_split is Split.CALIBRATION and args.calibration_role is None:
        raise ValueError("calibration predictions require an isolated role")
    if expected_split is not Split.CALIBRATION and args.calibration_role is not None:
        raise ValueError("only calibration predictions have a calibration role")
    if expected_split is Split.SEALED_TEST:
        if args.sealed_authorization is None:
            raise ValueError("sealed test requires a frozen authorization manifest")
        authorization = verify_sealed_authorization(
            args.sealed_authorization,
            checkpoint=args.checkpoint,
            features=args.features,
            labels=args.labels,
            output=args.output,
        )
    else:
        if args.sealed_authorization is not None:
            raise ValueError(
                "calibration inference cannot consume sealed authorization"
            )
        authorization = None

    training_report = json.loads(args.training_report.read_text())
    if training_report.get("schema_version") != "piu.target-binder-training.v1":
        raise ValueError("unsupported target-binder training report")
    if sha256(args.checkpoint) != training_report["checkpoint"]["sha256"]:
        raise ValueError("checkpoint differs from the training report")
    feature_report = json.loads(args.feature_report.read_text())
    if sha256(args.features) != feature_report["output"]["sha256"]:
        raise ValueError("feature cache differs from its report")
    with np.load(args.features) as store:
        feature_arrays = {name: np.asarray(store[name]) for name in store.files}

    import torch

    if torch.cuda.is_available():
        raise RuntimeError("CUDA remained visible after CPU inference isolation")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "piu.target-binder-checkpoint.v1":
        raise ValueError("unsupported target-binder checkpoint")
    bundle = join_binding_features(
        feature_arrays=feature_arrays,
        feature_report=feature_report,
        labels=load_binding_labels(args.labels),
        action_vocabulary=checkpoint["action_vocabulary"],
    )
    observed_splits = set(bundle.split)
    if observed_splits != {expected_split}:
        raise ValueError("prediction bundle does not match its declared split")
    group_contract = training_report["initial_state_groups"]
    bundle_groups = set(bundle.initial_state_group)
    if expected_split in {Split.TRAIN, Split.DEVELOPMENT}:
        if bundle_groups - set(group_contract[expected_split.value]):
            raise ValueError(
                "model-selection feature groups differ from training report"
            )
    else:
        prior_groups = set(group_contract["train"] + group_contract["development"])
        if prior_groups & bundle_groups:
            raise ValueError("evaluation groups overlap model-selection groups")
    model = PromptConditionedTargetBinder(**checkpoint["model"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.cpu().eval()
    batch = tensor_batch(bundle)
    with torch.no_grad():
        outputs = model(
            batch["image_tokens"],
            batch["prompt_tokens"],
            patch_xy=batch["patch_xy"],
            camera_ids=batch["camera_ids"],
            temporal_ids=batch["temporal_ids"],
            executed_action_ids=batch["executed_action_ids"],
            image_valid_mask=batch["image_valid_mask"],
            prompt_valid_mask=batch["prompt_valid_mask"],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_id=np.asarray(bundle.sample_id),
        initial_state_group=np.asarray(bundle.initial_state_group),
        split=np.asarray([value.value for value in bundle.split]),
        image_valid_mask=bundle.image_valid_mask,
        spatial_logits=outputs["spatial_logits"].cpu().numpy(),
        target_token=outputs["target_token"].cpu().numpy(),
        target_present_logit=outputs["target_present_logit"].cpu().numpy(),
        task_sufficiency_logit=outputs["task_sufficiency_logit"].cpu().numpy(),
        holding_requested_target_logit=outputs["holding_requested_target_logit"]
        .cpu()
        .numpy(),
        region_confirmed_empty_logit=outputs["region_confirmed_empty_logit"]
        .cpu()
        .numpy(),
        task_complete_logit=outputs["task_complete_logit"].cpu().numpy(),
        patch_target=bundle.patch_target,
        target_present=bundle.target_present,
        task_sufficient=bundle.task_sufficient,
        task_sufficient_mask=bundle.task_sufficient_mask,
        holding_requested_target=bundle.holding_requested_target,
        holding_requested_target_mask=bundle.holding_requested_target_mask,
        region_confirmed_empty=bundle.region_confirmed_empty,
        region_confirmed_empty_mask=bundle.region_confirmed_empty_mask,
        task_complete=bundle.task_complete,
        task_complete_mask=bundle.task_complete_mask,
    )
    report = {
        "schema_version": "piu.target-binder-predictions.v1",
        "claim_scope": (
            "CALIBRATION_ONLY_NOT_EVALUATION_EVIDENCE"
            if expected_split is Split.CALIBRATION
            else (
                "SEALED_TEST_EVALUATION"
                if expected_split is Split.SEALED_TEST
                else "MODEL_SELECTION_FEATURES_NOT_TEST_EVIDENCE"
            )
        ),
        "split": expected_split.value,
        "calibration_role": args.calibration_role,
        "initial_state_groups": sorted(set(bundle.initial_state_group)),
        "inputs": {
            name: {"path": portable(path), "sha256": sha256(path)}
            for name, path in {
                "checkpoint": args.checkpoint,
                "training_report": args.training_report,
                "features": args.features,
                "feature_report": args.feature_report,
                "labels": args.labels,
            }.items()
        },
        "output": {"path": portable(args.output), "sha256": sha256(args.output)},
        "samples": len(bundle.sample_id),
        "cuda_visible_to_predictor": torch.cuda.is_available(),
        "policy_observation_contains_evaluator_masks": False,
        "artifact_contains_evaluator_labels": True,
        "sealed_authorization": (
            None
            if authorization is None
            else {
                "path": portable(args.sealed_authorization),
                "sha256": sha256(args.sealed_authorization),
            }
        ),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
