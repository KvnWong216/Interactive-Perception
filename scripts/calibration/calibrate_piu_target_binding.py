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

from piu.binding_calibration import (
    MondrianBinaryLAC,
    binary_probability_metrics,
    fit_binary_temperature,
    fit_spatial_temperature,
    prediction_set_metrics,
    sigmoid,
    spatial_conformal_fit,
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


def binary_calibrator(
    *,
    name: str,
    temperature_logits: np.ndarray,
    temperature_truth: np.ndarray,
    conformal_logits: np.ndarray,
    conformal_truth: np.ndarray,
    alphas: list[float],
) -> dict:
    temperature = fit_binary_temperature(temperature_logits, temperature_truth)
    conformal_probability = sigmoid(conformal_logits / temperature)
    sets = {}
    for alpha in alphas:
        calibrator = MondrianBinaryLAC.fit(
            conformal_probability, conformal_truth, alpha=alpha
        )
        sets[str(alpha)] = {
            "calibrator": calibrator.to_dict(),
            "calibration_diagnostic": prediction_set_metrics(
                calibrator.predict(conformal_probability), conformal_truth
            ),
        }
    return {
        "status": "SUPPORTED",
        "name": name,
        "temperature": temperature,
        "temperature_fit_metrics": {
            "before": binary_probability_metrics(
                temperature_logits, temperature_truth, temperature=1.0
            ),
            "after": binary_probability_metrics(
                temperature_logits, temperature_truth, temperature=temperature
            ),
        },
        "conformal": sets,
    }


def optional_binary_calibrator(
    *,
    name: str,
    temperature: dict[str, np.ndarray],
    conformal: dict[str, np.ndarray],
    alphas: list[float],
) -> dict:
    """Fit one nullable verifier only when both isolated roles span both classes."""

    temperature_mask = temperature[f"{name}_mask"].astype(bool)
    conformal_mask = conformal[f"{name}_mask"].astype(bool)
    supported = set(temperature[name][temperature_mask].tolist()) == {
        0.0,
        1.0,
    } and set(conformal[name][conformal_mask].tolist()) == {0.0, 1.0}
    if not supported:
        return {
            "status": "UNSUPPORTED",
            "reason": (
                "both calibration roles require labeled positive and negative examples"
            ),
            "temperature_labeled": int(temperature_mask.sum()),
            "conformal_labeled": int(conformal_mask.sum()),
        }
    return binary_calibrator(
        name=name,
        temperature_logits=temperature[f"{name}_logit"][temperature_mask],
        temperature_truth=temperature[name][temperature_mask],
        conformal_logits=conformal[f"{name}_logit"][conformal_mask],
        conformal_truth=conformal[name][conformal_mask],
        alphas=alphas,
    )


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

    spatial_temperature = fit_spatial_temperature(
        temperature["spatial_logits"],
        temperature["image_valid_mask"],
        temperature["patch_target"],
    )
    conformal_spatial_probability = spatial_probabilities(
        conformal["spatial_logits"],
        conformal["image_valid_mask"],
        temperature=spatial_temperature,
    )
    spatial_sets = {}
    for alpha in alphas:
        fitted = spatial_conformal_fit(
            conformal_spatial_probability,
            conformal["patch_target"],
            alpha=alpha,
        )
        predicted = spatial_prediction_sets(
            conformal_spatial_probability,
            conformal["image_valid_mask"],
            fitted,
        )
        spatial_sets[str(alpha)] = {
            "calibrator": fitted,
            "calibration_diagnostic": spatial_prediction_set_metrics(
                predicted, conformal["patch_target"]
            ),
        }
    presence = binary_calibrator(
        name="target_presence",
        temperature_logits=temperature["target_present_logit"],
        temperature_truth=temperature["target_present"],
        conformal_logits=conformal["target_present_logit"],
        conformal_truth=conformal["target_present"],
        alphas=alphas,
    )
    temperature_sufficiency_mask = temperature["task_sufficient_mask"].astype(bool)
    conformal_sufficiency_mask = conformal["task_sufficient_mask"].astype(bool)
    supported_sufficiency = set(
        temperature["task_sufficient"][temperature_sufficiency_mask].tolist()
    ) == {0.0, 1.0} and set(
        conformal["task_sufficient"][conformal_sufficiency_mask].tolist()
    ) == {0.0, 1.0}
    if supported_sufficiency:
        sufficiency = binary_calibrator(
            name="task_sufficiency",
            temperature_logits=temperature["task_sufficiency_logit"][
                temperature_sufficiency_mask
            ],
            temperature_truth=temperature["task_sufficient"][
                temperature_sufficiency_mask
            ],
            conformal_logits=conformal["task_sufficiency_logit"][
                conformal_sufficiency_mask
            ],
            conformal_truth=conformal["task_sufficient"][conformal_sufficiency_mask],
            alphas=alphas,
        )
    else:
        sufficiency = {
            "status": "UNSUPPORTED",
            "reason": "both calibration roles require labeled positive and negative examples",
            "temperature_labeled": int(temperature_sufficiency_mask.sum()),
            "conformal_labeled": int(conformal_sufficiency_mask.sum()),
        }
    temperature_holding_mask = temperature["holding_requested_target_mask"].astype(bool)
    conformal_holding_mask = conformal["holding_requested_target_mask"].astype(bool)
    supported_holding = set(
        temperature["holding_requested_target"][temperature_holding_mask].tolist()
    ) == {0.0, 1.0} and set(
        conformal["holding_requested_target"][conformal_holding_mask].tolist()
    ) == {0.0, 1.0}
    if supported_holding:
        holding = binary_calibrator(
            name="holding_requested_target",
            temperature_logits=temperature["holding_requested_target_logit"][
                temperature_holding_mask
            ],
            temperature_truth=temperature["holding_requested_target"][
                temperature_holding_mask
            ],
            conformal_logits=conformal["holding_requested_target_logit"][
                conformal_holding_mask
            ],
            conformal_truth=conformal["holding_requested_target"][
                conformal_holding_mask
            ],
            alphas=alphas,
        )
    else:
        holding = {
            "status": "UNSUPPORTED",
            "reason": "both calibration roles require labeled positive and negative examples",
            "temperature_labeled": int(temperature_holding_mask.sum()),
            "conformal_labeled": int(conformal_holding_mask.sum()),
        }
    region_empty = optional_binary_calibrator(
        name="region_confirmed_empty",
        temperature=temperature,
        conformal=conformal,
        alphas=alphas,
    )
    task_complete = optional_binary_calibrator(
        name="task_complete",
        temperature=temperature,
        conformal=conformal,
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
        "spatial": {
            "status": "SUPPORTED",
            "temperature": spatial_temperature,
            "temperature_fit_metrics": {
                "before": spatial_probability_metrics(
                    temperature["spatial_logits"],
                    temperature["image_valid_mask"],
                    temperature["patch_target"],
                    temperature=1.0,
                ),
                "after": spatial_probability_metrics(
                    temperature["spatial_logits"],
                    temperature["image_valid_mask"],
                    temperature["patch_target"],
                    temperature=spatial_temperature,
                ),
            },
            "conformal": spatial_sets,
        },
        "target_presence": presence,
        "task_sufficiency": sufficiency,
        "holding_requested_target": holding,
        "region_confirmed_empty": region_empty,
        "task_complete": task_complete,
        "temperature_conformal_groups_disjoint": True,
        "model_selection_groups_loaded": False,
        "sealed_test_loaded": False,
        "paper_method_claim_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
