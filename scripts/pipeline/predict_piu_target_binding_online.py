#!/usr/bin/env python3
"""Run frozen target binding without loading evaluator labels."""

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

from piu.binding_data import build_binding_inputs
from piu.contracts import Split
from piu.target_binding import PromptConditionedTargetBinder


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def verify_authorization(args: argparse.Namespace) -> None:
    if args.sealed_authorization is None:
        raise ValueError("sealed online binder inference requires authorization")
    value = json.loads(args.sealed_authorization.read_text())
    if value.get("schema_version") != "piu.binder-online-sealed-authorization.v1":
        raise ValueError("invalid online binder sealed authorization")
    expected = {
        "checkpoint_sha256": sha256(args.checkpoint),
        "feature_sha256": sha256(args.features),
        "single_use_output": portable(args.output),
    }
    for name, required in expected.items():
        if value.get(name) != required:
            raise ValueError(f"online binder authorization differs at {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
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
        "output",
    ):
        setattr(args, name, resolve(getattr(args, name)))
    if args.sealed_authorization is not None:
        args.sealed_authorization = resolve(args.sealed_authorization)
    report_path = args.output.with_suffix(".json")
    if args.output.exists() or report_path.exists():
        raise FileExistsError("online binder predictions are immutable")
    expected_split = Split(args.expected_split)
    if expected_split is Split.SEALED_TEST:
        verify_authorization(args)
    elif args.sealed_authorization is not None:
        raise ValueError("non-sealed binder inference cannot consume authorization")
    training_report = json.loads(args.training_report.read_text())
    if training_report.get("schema_version") != "piu.target-binder-training.v1":
        raise ValueError("unsupported target-binder training report")
    if sha256(args.checkpoint) != training_report["checkpoint"]["sha256"]:
        raise ValueError("online binder checkpoint differs from training report")
    feature_report = json.loads(args.feature_report.read_text())
    if feature_report.get("schema_version") != "piu.spatial-prefix-features.v1":
        raise ValueError("unsupported spatial-prefix feature report")
    if sha256(args.features) != feature_report["output"]["sha256"]:
        raise ValueError("online binder features differ from report")
    with np.load(args.features) as store:
        feature_arrays = {name: np.asarray(store[name]) for name in store.files}

    import torch

    if torch.cuda.is_available():
        raise RuntimeError("CUDA remained visible after online binder isolation")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "piu.target-binder-checkpoint.v1":
        raise ValueError("unsupported target-binder checkpoint")
    values = build_binding_inputs(
        feature_arrays=feature_arrays,
        feature_report=feature_report,
        action_vocabulary=checkpoint["action_vocabulary"],
    )
    if set(values.split) != {expected_split}:
        raise ValueError("online binder input split mismatch")
    groups = set(values.initial_state_group)
    group_contract = training_report["initial_state_groups"]
    if expected_split in {Split.TRAIN, Split.DEVELOPMENT}:
        if groups - set(group_contract[expected_split.value]):
            raise ValueError("online binder groups differ from training report")
    elif groups & set(group_contract["train"] + group_contract["development"]):
        raise ValueError("online binder evaluation groups overlap model selection")
    model = PromptConditionedTargetBinder(**checkpoint["model"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.cpu().eval()
    with torch.no_grad():
        outputs = model(
            torch.as_tensor(values.image_tokens),
            torch.as_tensor(values.prompt_tokens),
            patch_xy=torch.as_tensor(values.patch_xy),
            camera_ids=torch.as_tensor(values.camera_id),
            temporal_ids=torch.as_tensor(values.temporal_id),
            executed_action_ids=torch.as_tensor(values.executed_action_id),
            image_valid_mask=torch.as_tensor(values.image_valid_mask),
            prompt_valid_mask=torch.as_tensor(values.prompt_valid_mask),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_id=np.asarray(values.sample_id),
        initial_state_group=np.asarray(values.initial_state_group),
        split=np.asarray([value.value for value in values.split]),
        image_valid_mask=values.image_valid_mask,
        patch_xy=values.patch_xy,
        camera_id=values.camera_id,
        temporal_id=values.temporal_id,
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
    )
    report = {
        "schema_version": "piu.target-binder-online-predictions.v1",
        "claim_scope": "PUBLIC_INPUT_BINDING_SCORES_NOT_EVALUATOR_EVIDENCE",
        "split": expected_split.value,
        "initial_state_groups": sorted(groups),
        "inputs": {
            name: {"path": portable(path), "sha256": sha256(path)}
            for name, path in {
                "checkpoint": args.checkpoint,
                "training_report": args.training_report,
                "features": args.features,
                "feature_report": args.feature_report,
            }.items()
        },
        "output": {"path": portable(args.output), "sha256": sha256(args.output)},
        "evaluator_labels_loaded": False,
        "cuda_visible_to_predictor": torch.cuda.is_available(),
        "paper_method_claim_allowed": False,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
