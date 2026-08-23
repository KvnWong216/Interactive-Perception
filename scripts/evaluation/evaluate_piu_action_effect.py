#!/usr/bin/env python3
"""Evaluate frozen calibrated PIU route/effect scores on one sealed split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.action_effect import EFFECT_FACTORS
from piu.binding_calibration import (
    binary_probability_metrics,
    multiclass_prediction_set_metrics,
    multiclass_probability_metrics,
    prediction_set_metrics,
)
from piu.effect_calibration import apply_effect_calibration


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
        raise FileExistsError("sealed action-effect evaluations are immutable")
    prediction_report = json.loads(args.prediction_report.read_text())
    if prediction_report.get("schema_version") != "piu.action-effect-predictions.v1":
        raise ValueError("unsupported action-effect prediction report")
    if prediction_report.get("split") != "sealed_test":
        raise ValueError("formal action-effect evaluator accepts only sealed test")
    if sha256(args.predictions) != prediction_report["output"]["sha256"]:
        raise ValueError("sealed effect predictions differ from report")
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("schema_version") != "piu.action-effect-calibration.v1":
        raise ValueError("unsupported action-effect calibration artifact")
    checkpoint_hash = prediction_report["inputs"]["checkpoint"]["sha256"]
    if checkpoint_hash != calibration["checkpoint_sha256"]:
        raise ValueError("sealed effect predictions use another checkpoint")
    if prediction_report["variant"] != calibration["variant"]:
        raise ValueError("sealed effect predictions use another ablation")
    with np.load(args.predictions) as store:
        values = {name: np.asarray(store[name]) for name in store.files}
    sealed_groups = set(values["initial_state_group"].astype(str))
    calibration_groups = set(
        calibration["initial_state_groups"]["temperature"]
        + calibration["initial_state_groups"]["conformal"]
    )
    if sealed_groups & calibration_groups:
        raise ValueError("sealed effect groups overlap calibration groups")
    valid = values["candidate_valid_mask"].astype(bool)
    alphas = [float(alpha) for alpha in calibration["reported_alpha"]]
    evaluations = {}
    for alpha in alphas:
        applied = apply_effect_calibration(
            {
                name: values[name]
                for name in (
                    "route_logits",
                    "candidate_valid_mask",
                    "factor_logits",
                )
            },
            calibration,
            alpha=alpha,
        )
        factor_sets = {}
        for factor_index, factor_name in enumerate(EFFECT_FACTORS):
            section = calibration["factors"][factor_name]
            support = values["factor_support_mask"][:, :, factor_index].astype(
                bool
            ) & valid
            if section.get("status") != "SUPPORTED":
                factor_sets[factor_name] = {
                    "status": "UNSUPPORTED",
                    "sealed_labeled": int(support.sum()),
                }
                continue
            logits = values["factor_logits"][:, :, factor_index][support]
            truth = values["factor_target"][:, :, factor_index][support]
            sets = applied["factor_prediction_sets"][:, :, factor_index][support]
            factor_sets[factor_name] = {
                "status": "SUPPORTED",
                "proper_scores": binary_probability_metrics(
                    logits,
                    truth,
                    temperature=float(section["temperature"]),
                ),
                "prediction_sets": prediction_set_metrics(sets, truth),
            }
        evaluations[str(alpha)] = {
            "route_prediction_sets": multiclass_prediction_set_metrics(
                applied["route_prediction_set"], values["route_target"]
            ),
            "factors": factor_sets,
        }
    result = {
        "schema_version": "piu.action-effect-sealed-evaluation.v1",
        "claim_scope": "SEALED_ROUTE_EFFECT_EVIDENCE",
        "inputs": {
            name: {"path": portable(path), "sha256": sha256(path)}
            for name, path in {
                "predictions": args.predictions,
                "prediction_report": args.prediction_report,
                "calibration": args.calibration,
            }.items()
        },
        "variant": calibration["variant"],
        "samples": len(values["sample_id"]),
        "initial_state_groups": sorted(sealed_groups),
        "route_proper_scores": multiclass_probability_metrics(
            values["route_logits"],
            valid,
            values["route_target"],
            temperature=float(calibration["route"]["temperature"]),
        ),
        "risk_levels": evaluations,
        "primary_alpha": calibration["primary_alpha"],
        "sealed_test_opened": True,
        "paper_method_claim_allowed": False,
        "paper_method_claim_blocker": (
            "Route/effect scores alone do not establish a calibrated physical "
            "closed-loop improvement over baselines."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
