#!/usr/bin/env python3
"""Evaluate a frozen calibrated PIU target binder on one sealed prediction file."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.binding_calibration import (
    MondrianBinaryLAC,
    binary_probability_metrics,
    prediction_set_metrics,
    sigmoid,
    spatial_prediction_set_metrics,
    spatial_prediction_sets,
    spatial_probabilities,
    spatial_probability_metrics,
)


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


def evaluate_binary(
    *,
    section: dict,
    logits: np.ndarray,
    truth: np.ndarray,
    alphas: list[float],
) -> dict:
    if section.get("status") != "SUPPORTED":
        raise ValueError("requested binary head has no frozen calibrator")
    temperature = float(section["temperature"])
    probability = sigmoid(np.asarray(logits, dtype=np.float64) / temperature)
    conformal = {}
    for alpha in alphas:
        fitted = MondrianBinaryLAC.from_dict(
            section["conformal"][str(alpha)]["calibrator"]
        )
        if fitted.alpha != alpha:
            raise ValueError("binary conformal alpha mismatch")
        conformal[str(alpha)] = prediction_set_metrics(
            fitted.predict(probability), truth
        )
    return {
        "proper_scores": binary_probability_metrics(
            logits, truth, temperature=temperature
        ),
        "conformal": conformal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-report", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("predictions", "prediction_report", "calibration", "output"):
        setattr(args, name, resolve(getattr(args, name)))
    if args.output.exists():
        raise FileExistsError("sealed binding evaluation is immutable")
    prediction_report = json.loads(args.prediction_report.read_text())
    if prediction_report.get("schema_version") != "piu.target-binder-predictions.v1":
        raise ValueError("unsupported binder prediction report")
    if prediction_report.get("split") != "sealed_test":
        raise ValueError("formal binding evaluator accepts only the sealed test")
    if sha256(args.predictions) != prediction_report["output"]["sha256"]:
        raise ValueError("sealed predictions differ from their report")
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("schema_version") != "piu.target-binder-calibration.v1":
        raise ValueError("unsupported target-binder calibration artifact")
    checkpoint_hash = prediction_report["inputs"]["checkpoint"]["sha256"]
    if checkpoint_hash != calibration["checkpoint_sha256"]:
        raise ValueError("sealed predictions and calibration use different checkpoints")
    with np.load(args.predictions) as store:
        values = {name: np.asarray(store[name]) for name in store.files}
    required = {
        "sample_id",
        "initial_state_group",
        "split",
        "image_valid_mask",
        "spatial_logits",
        "target_present_logit",
        "task_sufficiency_logit",
        "patch_target",
        "target_present",
        "task_sufficient",
        "task_sufficient_mask",
    }
    if set(values) != required:
        raise ValueError("sealed prediction arrays differ from the frozen schema")
    if set(values["split"].astype(str)) != {"sealed_test"}:
        raise ValueError("sealed prediction file contains another split")
    sealed_groups = set(values["initial_state_group"].astype(str))
    calibration_groups = set(
        calibration["initial_state_groups"]["temperature"]
        + calibration["initial_state_groups"]["conformal"]
    )
    if sealed_groups & calibration_groups:
        raise ValueError("sealed-test groups overlap calibration groups")
    alphas = [
        float(value) for value in calibration["risk_contract"]["reported_alpha"]
    ]

    spatial_temperature = float(calibration["spatial"]["temperature"])
    spatial_probability = spatial_probabilities(
        values["spatial_logits"],
        values["image_valid_mask"],
        temperature=spatial_temperature,
    )
    spatial_sets = {}
    for alpha in alphas:
        fitted = calibration["spatial"]["conformal"][str(alpha)]["calibrator"]
        spatial_sets[str(alpha)] = spatial_prediction_set_metrics(
            spatial_prediction_sets(
                spatial_probability, values["image_valid_mask"], fitted
            ),
            values["patch_target"],
        )
    presence = evaluate_binary(
        section=calibration["target_presence"],
        logits=values["target_present_logit"],
        truth=values["target_present"],
        alphas=alphas,
    )
    sufficiency_mask = values["task_sufficient_mask"].astype(bool)
    if calibration["task_sufficiency"].get("status") == "SUPPORTED":
        if not sufficiency_mask.any():
            raise ValueError("sealed test has no supported task-sufficiency labels")
        sufficiency = {
            "status": "SUPPORTED",
            **evaluate_binary(
                section=calibration["task_sufficiency"],
                logits=values["task_sufficiency_logit"][sufficiency_mask],
                truth=values["task_sufficient"][sufficiency_mask],
                alphas=alphas,
            ),
        }
    else:
        sufficiency = {
            "status": "UNSUPPORTED",
            "reason": calibration["task_sufficiency"]["reason"],
            "sealed_labeled": int(sufficiency_mask.sum()),
        }
    result = {
        "schema_version": "piu.target-binder-sealed-evaluation.v1",
        "claim_scope": "SEALED_TARGET_BINDING_EVIDENCE",
        "inputs": {
            name: {"path": portable(path), "sha256": sha256(path)}
            for name, path in {
                "predictions": args.predictions,
                "prediction_report": args.prediction_report,
                "calibration": args.calibration,
            }.items()
        },
        "samples": len(values["sample_id"]),
        "initial_state_groups": sorted(sealed_groups),
        "primary_alpha": calibration["primary_alpha"],
        "spatial": {
            "proper_scores": spatial_probability_metrics(
                values["spatial_logits"],
                values["image_valid_mask"],
                values["patch_target"],
                temperature=spatial_temperature,
            ),
            "conformal": spatial_sets,
        },
        "target_presence": presence,
        "task_sufficiency": sufficiency,
        "calibration_groups_loaded_for_model_selection": False,
        "sealed_test_opened": True,
        "paper_method_claim_allowed": False,
        "paper_method_claim_blocker": (
            "Target-binding metrics alone do not establish a causal frozen-VLA "
            "execution improvement."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
