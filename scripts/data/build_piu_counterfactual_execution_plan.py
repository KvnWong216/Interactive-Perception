#!/usr/bin/env python3
"""Build public calibrated eligibility and exact prompts for candidate forks."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.binding_calibration import apply_binding_calibration
from piu.contracts import load_public_transitions, public_observation_sha256
from piu.execution_plan import PublicExecutionContext, candidate_eligibility
from piu.executor_bridge import current_spatial_references, serialize_pi05_subtask


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load_npz(path: Path, report_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    report = json.loads(report_path.read_text())
    if report.get("schema_version") != "piu.target-binder-online-predictions.v1":
        raise ValueError("unsupported online binder report")
    if sha256(path) != report.get("output", {}).get("sha256"):
        raise ValueError("online binder predictions differ from their report")
    with np.load(path) as store:
        return {name: np.asarray(store[name]) for name in store.files}, report


def members(values: np.ndarray) -> list[bool]:
    membership = np.asarray(values, dtype=bool)
    if membership.shape != (2,):
        raise ValueError("binder binary prediction-set shape is invalid")
    return [
        label
        for label, included in zip((False, True), membership, strict=True)
        if included
    ]


def verify_sealed(
    path: Path,
    *,
    transition: Path,
    predictions: Path,
    calibration: Path,
    output: Path,
) -> None:
    value = json.loads(path.read_text())
    if (
        value.get("schema_version")
        != "piu.counterfactual-execution-plan-sealed-authorization.v1"
    ):
        raise ValueError("unsupported execution-plan sealed authorization")
    expected = {
        "public_transition_sha256": sha256(transition),
        "binding_prediction_sha256": sha256(predictions),
        "binder_calibration_sha256": sha256(calibration),
        "single_use_output": portable(output),
    }
    for name, required in expected.items():
        if value.get(name) != required:
            raise ValueError(f"execution-plan authorization differs at {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-transition", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--binding-predictions", type=Path, required=True)
    parser.add_argument("--binding-report", type=Path, required=True)
    parser.add_argument("--binder-calibration", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--sealed-authorization", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "public_transition",
        "binding_predictions",
        "binding_report",
        "binder_calibration",
        "feature_report",
        "sealed_authorization",
        "output",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, resolve(value))
    if args.output.exists():
        raise FileExistsError("counterfactual execution plans are immutable")
    transitions = [
        row
        for row in load_public_transitions(args.public_transition)
        if row.sample_id == args.sample_id
    ]
    if len(transitions) != 1:
        raise ValueError("execution-plan sample must select one public transition")
    transition = transitions[0]
    if transition.split.value == "calibration":
        raise ValueError("effect collection cannot consume calibration groups")
    if transition.split.value == "sealed_test":
        if args.sealed_authorization is None:
            raise ValueError("sealed execution plans require authorization")
        verify_sealed(
            args.sealed_authorization,
            transition=args.public_transition,
            predictions=args.binding_predictions,
            calibration=args.binder_calibration,
            output=args.output,
        )
    elif args.sealed_authorization is not None:
        raise ValueError("non-sealed execution plan cannot use sealed authorization")
    binding, binding_report = load_npz(
        args.binding_predictions, args.binding_report
    )
    calibration = json.loads(args.binder_calibration.read_text())
    if calibration.get("schema_version") != "piu.target-binder-calibration.v1":
        raise ValueError("unsupported binder calibration artifact")
    checkpoint_sha256 = binding_report.get("inputs", {}).get("checkpoint", {}).get(
        "sha256"
    )
    if calibration.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("execution-plan binder predictions/calibration differ")
    feature_report = json.loads(args.feature_report.read_text())
    if feature_report.get("schema_version") != "piu.spatial-prefix-features.v1":
        raise ValueError("unsupported spatial-prefix feature report")
    if binding_report.get("inputs", {}).get("feature_report", {}).get(
        "sha256"
    ) != sha256(args.feature_report):
        raise ValueError("execution-plan feature report differs from binder input")
    sample_ids = np.asarray(binding["sample_id"]).astype(str)
    indices = np.flatnonzero(sample_ids == transition.sample_id)
    if len(indices) != 1:
        raise ValueError("execution-plan binder sample is absent or duplicated")
    index = int(indices[0])
    if (
        str(np.asarray(binding["initial_state_group"]).astype(str)[index])
        != transition.initial_state_group
        or str(np.asarray(binding["split"]).astype(str)[index])
        != transition.split.value
    ):
        raise ValueError("execution-plan binder group/split differs")
    alpha = float(
        calibration["risk_contract"]["primary_alpha"]
        if args.alpha is None
        else args.alpha
    )
    if alpha not in [
        float(value) for value in calibration["risk_contract"]["reported_alpha"]
    ]:
        raise ValueError("execution-plan alpha was not preregistered")
    calibrated = apply_binding_calibration(binding, calibration, alpha=alpha)
    camera_names = tuple(feature_report["layout"]["camera_names"])
    references = current_spatial_references(
        prediction_set=calibrated["spatial_prediction_set"][index],
        patch_xy=binding["patch_xy"][index],
        camera_id=binding["camera_id"][index],
        temporal_id=binding["temporal_id"][index],
        camera_names=camera_names,
    )
    context = PublicExecutionContext.create(
        task_sufficiency=members(
            calibrated["task_sufficiency_prediction_set"][index]
        ),
        target_presence=members(
            calibrated["target_presence_prediction_set"][index]
        ),
        holding_requested_target=members(
            calibrated["holding_requested_target_prediction_set"][index]
        ),
        spatial_reference_available=bool(references),
    )
    rows = []
    for candidate in transition.candidate_actions:
        candidate = dict(candidate)
        candidate_id = str(candidate["candidate_id"])
        primitive = str(candidate["primitive"]).upper()
        eligibility = candidate_eligibility(candidate, context)
        physical = primitive not in {"STOP", "REPORT_NOT_FOUND"}
        spatial = references if primitive in {"PICK", "DIRECT"} else ()
        subtask = (
            serialize_pi05_subtask(candidate, spatial_references=spatial)
            if physical and eligibility.eligible
            else None
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "primitive": primitive,
                "eligible_for_execution": eligibility.eligible,
                "eligibility_reason": eligibility.reason,
                "structured_pi05_subtask": subtask,
                "spatial_references": [
                    {
                        **dataclasses.asdict(item),
                        "selected_patch_indices": list(item.selected_patch_indices),
                        "x_interval": list(item.x_interval),
                        "y_interval": list(item.y_interval),
                    }
                    for item in spatial
                ],
            }
        )
    decision_digest = public_observation_sha256(
        transition.observations["post_interaction"]
    )
    result = {
        "schema_version": "piu.counterfactual-execution-plan.v1",
        "claim_scope": "PUBLIC_BINDER_PRECONDITIONS_NOT_EFFECT_OUTCOMES",
        "decision_sample_id": transition.sample_id,
        "initial_state_group": transition.initial_state_group,
        "split": transition.split.value,
        "decision_observation_sha256": decision_digest,
        "alpha": alpha,
        "public_execution_context": {
            "task_sufficiency": sorted(context.task_sufficiency),
            "target_presence": sorted(context.target_presence),
            "holding_requested_target": sorted(context.holding_requested_target),
            "spatial_reference_available": context.spatial_reference_available,
        },
        "candidates": rows,
        "public_inputs_only": True,
        "online_oracle_inputs": [],
        "inputs": {
            name: {"path": portable(path), "sha256": sha256(path)}
            for name, path in {
                "public_transition": args.public_transition,
                "binding_predictions": args.binding_predictions,
                "binding_report": args.binding_report,
                "binder_calibration": args.binder_calibration,
                "feature_report": args.feature_report,
            }.items()
        },
        "paper_method_claim_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
