#!/usr/bin/env python3
"""Fit route/effect calibration on two group-disjoint calibration roles."""

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

from piu.effect_calibration import fit_effect_calibration


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


def load_role(path: Path, report_path: Path, *, role: str):
    report = json.loads(report_path.read_text())
    if report.get("schema_version") != "piu.action-effect-predictions.v1":
        raise ValueError("unsupported action-effect prediction report")
    if report.get("split") != "calibration" or report.get("calibration_role") != role:
        raise ValueError(f"effect predictions are not the {role} calibration role")
    if sha256(path) != report["output"]["sha256"]:
        raise ValueError("effect prediction artifact differs from its report")
    with np.load(path) as store:
        arrays = {name: np.asarray(store[name]) for name in store.files}
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
        raise FileExistsError("action-effect calibration artifact is immutable")
    config = yaml.safe_load(args.config.read_text())
    if config.get("schema_version") != "piu.action-effect-calibration-experiment.v1":
        raise ValueError("unsupported action-effect calibration config")
    alphas = [float(value) for value in config["risk_contract"]["reported_alpha"]]
    primary = float(config["risk_contract"]["primary_alpha"])
    if primary not in alphas:
        raise ValueError("primary effect risk must be in the reported risk grid")
    temperature, temperature_report = load_role(
        args.temperature_predictions, args.temperature_report, role="temperature"
    )
    conformal, conformal_report = load_role(
        args.conformal_predictions, args.conformal_report, role="conformal"
    )
    if temperature_report["variant"] != conformal_report["variant"]:
        raise ValueError("effect calibration roles use different ablations")
    temperature_checkpoint = temperature_report["inputs"]["checkpoint"]["sha256"]
    if temperature_checkpoint != conformal_report["inputs"]["checkpoint"]["sha256"]:
        raise ValueError("effect calibration roles use different checkpoints")
    temperature_groups = set(temperature["initial_state_group"].astype(str))
    conformal_groups = set(conformal["initial_state_group"].astype(str))
    if temperature_groups & conformal_groups:
        raise ValueError("effect temperature/conformal groups overlap")
    fitted = fit_effect_calibration(
        temperature_values=temperature,
        conformal_values=conformal,
        alphas=alphas,
    )
    artifact = {
        **fitted,
        "claim_scope": "CALIBRATION_ARTIFACT_NOT_SEALED_TEST_EVIDENCE",
        "variant": temperature_report["variant"],
        "checkpoint_sha256": temperature_checkpoint,
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
        "initial_state_groups": {
            "temperature": sorted(temperature_groups),
            "conformal": sorted(conformal_groups),
        },
        "primary_alpha": primary,
        "controller_contract": config["controller_contract"],
        "model_selection_loaded": False,
        "sealed_test_loaded": False,
        "paper_method_claim_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
