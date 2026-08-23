#!/usr/bin/env python3
"""Fit PIU binder temperatures and conformal sets on disjoint calibration roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.binding_calibration import fit_binding_calibration


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


def load_predictions(path: Path, report_path: Path, *, role: str):
    report = json.loads(report_path.read_text())
    if report.get("schema_version") != "piu.target-binder-predictions.v1":
        raise ValueError("unsupported binder prediction report")
    if report.get("split") != "calibration" or report.get("calibration_role") != role:
        raise ValueError(f"prediction report is not the {role} calibration role")
    if sha256(path) != report["output"]["sha256"]:
        raise ValueError("prediction artifact differs from its report")
    with np.load(path) as store:
        arrays = {name: np.asarray(store[name]) for name in store.files}
    required = {
        "sample_id",
        "initial_state_group",
        "split",
        "image_valid_mask",
        "spatial_logits",
        "target_token",
        "target_present_logit",
        "task_sufficiency_logit",
        "holding_requested_target_logit",
        "patch_target",
        "target_present",
        "task_sufficient",
        "task_sufficient_mask",
        "holding_requested_target",
        "holding_requested_target_mask",
        "region_confirmed_empty_logit",
        "region_confirmed_empty",
        "region_confirmed_empty_mask",
        "task_complete_logit",
        "task_complete",
        "task_complete_mask",
    }
    if set(arrays) != required:
        raise ValueError(f"prediction arrays differ: {sorted(set(arrays) ^ required)}")
    count = len(arrays["sample_id"])
    if count < 1 or any(len(value) != count for value in arrays.values()):
        raise ValueError("prediction arrays have inconsistent sample axes")
    if set(arrays["split"].astype(str)) != {"calibration"}:
        raise ValueError("prediction arrays contain a non-calibration split")
    return arrays, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--temperature-predictions", type=Path, required=True)
    parser.add_argument("--temperature-report", type=Path, required=True)
    parser.add_argument("--conformal-predictions", type=Path, required=True)
    parser.add_argument("--conformal-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "config",
        "temperature_predictions",
        "temperature_report",
        "conformal_predictions",
        "conformal_report",
        "output",
    ):
        setattr(args, name, resolve(getattr(args, name)))
    if args.output.exists():
        raise FileExistsError("binding calibration artifacts are immutable")
    config = yaml.safe_load(args.config.read_text())
    if config.get("schema_version") != "piu.binding-calibration-experiment.v1":
        raise ValueError("unsupported binding calibration config")
    alphas = [float(value) for value in config["risk_contract"]["reported_alpha"]]
    primary_alpha = float(config["risk_contract"]["primary_alpha"])
    if primary_alpha not in alphas or any(not 0.0 < value < 1.0 for value in alphas):
        raise ValueError("invalid predeclared calibration risk contract")
    temperature, temperature_report = load_predictions(
        args.temperature_predictions,
        args.temperature_report,
        role="temperature",
    )
    conformal, conformal_report = load_predictions(
        args.conformal_predictions,
        args.conformal_report,
        role="conformal",
    )
    temperature_groups = set(temperature["initial_state_group"].astype(str))
    conformal_groups = set(conformal["initial_state_group"].astype(str))
    if temperature_groups & conformal_groups:
        raise ValueError("temperature/conformal calibration groups overlap")
    temperature_checkpoint = temperature_report["inputs"]["checkpoint"]["sha256"]
    conformal_checkpoint = conformal_report["inputs"]["checkpoint"]["sha256"]
    if temperature_checkpoint != conformal_checkpoint:
        raise ValueError("calibration roles use different binder checkpoints")

    fitted = fit_binding_calibration(
        temperature_values=temperature,
        conformal_values=conformal,
        alphas=alphas,
    )
    artifact = {
        "schema_version": "piu.target-binder-calibration.v1",
        "claim_scope": "CALIBRATION_ARTIFACT_NOT_SEALED_TEST_EVIDENCE",
        "config": {"path": portable(args.config), "sha256": sha256(args.config)},
        "inputs": {
            name: {"path": portable(path), "sha256": sha256(path)}
            for name, path in {
                "temperature_predictions": args.temperature_predictions,
                "temperature_report": args.temperature_report,
                "conformal_predictions": args.conformal_predictions,
                "conformal_report": args.conformal_report,
            }.items()
        },
        "checkpoint_sha256": temperature_checkpoint,
        "initial_state_groups": {
            "temperature": sorted(temperature_groups),
            "conformal": sorted(conformal_groups),
        },
        "risk_contract": config["risk_contract"],
        "primary_alpha": primary_alpha,
        **fitted,
        "model_selection_groups_loaded": False,
        "sealed_test_loaded": False,
        "paper_method_claim_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
