#!/usr/bin/env python3
"""Update PIU search memory from calibrated post-observation verifier sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.binding_calibration import apply_binding_calibration
from piu.calibrated_controller import INFORMATION_PRIMITIVES
from piu.contracts import (
    load_public_transitions,
    public_observation_sha256,
)
from piu.temporal_memory import PublicObservationEvent, PublicTemporalMemory


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def prediction_membership(value: np.ndarray) -> tuple[bool, ...]:
    membership = np.asarray(value, dtype=bool)
    if membership.shape != (2,):
        raise ValueError("binary verifier membership must have shape [2]")
    return tuple(
        label
        for label, included in zip((False, True), membership, strict=True)
        if included
    )


def load_previous(path: Path) -> tuple[dict, PublicTemporalMemory]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != 1:
        raise ValueError("previous controller memory must contain exactly one row")
    row = rows[0]
    if row.get("schema_version") != "piu.public-controller-memory.v2":
        raise ValueError("unsupported previous controller-memory schema")
    if (
        row.get("public_inputs_only") is not True
        or row.get("online_oracle_inputs") != []
    ):
        raise ValueError("previous memory crosses the public/evaluator firewall")
    return row, PublicTemporalMemory.from_mapping(row.get("memory", {}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-transition", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--binding-predictions", type=Path, required=True)
    parser.add_argument("--binding-report", type=Path, required=True)
    parser.add_argument("--binder-calibration", type=Path, required=True)
    parser.add_argument("--previous-memory", type=Path)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "public_transition",
        "binding_predictions",
        "binding_report",
        "binder_calibration",
        "previous_memory",
        "output",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, resolve(value))
    if args.output.exists():
        raise FileExistsError("public temporal-memory artifacts are immutable")

    public_matches = [
        row
        for row in load_public_transitions(args.public_transition)
        if row.sample_id == args.sample_id
    ]
    if len(public_matches) != 1:
        raise ValueError("sample ID must select one public transition")
    transition = public_matches[0]
    report = json.loads(args.binding_report.read_text())
    if report.get("schema_version") != "piu.target-binder-online-predictions.v1":
        raise ValueError("memory update requires label-free online binder predictions")
    if sha256(args.binding_predictions) != report["output"]["sha256"]:
        raise ValueError("online binder predictions differ from their report")
    calibration = json.loads(args.binder_calibration.read_text())
    if calibration.get("schema_version") != "piu.target-binder-calibration.v1":
        raise ValueError("unsupported binder calibration artifact")
    if calibration["checkpoint_sha256"] != report["inputs"]["checkpoint"]["sha256"]:
        raise ValueError("binder prediction and calibration checkpoints differ")
    alpha = float(calibration["primary_alpha"] if args.alpha is None else args.alpha)
    if alpha not in [
        float(item) for item in calibration["risk_contract"]["reported_alpha"]
    ]:
        raise ValueError("memory alpha was not preregistered in binder calibration")
    with np.load(args.binding_predictions) as store:
        arrays = {name: np.asarray(store[name]) for name in store.files}
    sample_ids = arrays["sample_id"].astype(str)
    indices = np.flatnonzero(sample_ids == args.sample_id)
    if len(indices) != 1:
        raise ValueError("sample ID must select one online binder prediction")
    index = int(indices[0])
    if str(arrays["initial_state_group"][index]) != transition.initial_state_group:
        raise ValueError("binder prediction and public transition groups differ")
    calibrated = apply_binding_calibration(arrays, calibration, alpha=alpha)

    sources = tuple(
        str(candidate["candidate_id"])
        for candidate in transition.candidate_actions
        if str(candidate.get("primitive", "")).upper() in INFORMATION_PRIMITIVES
    )
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("public candidates require unique information-source IDs")
    pre_digest = public_observation_sha256(transition.observations["pre_interaction"])
    post_digest = public_observation_sha256(transition.observations["post_interaction"])
    initial = transition.public_action_history.get("initial_observation") is True
    previous_input = None
    if initial:
        if args.previous_memory is not None:
            raise ValueError("initial observation cannot extend previous memory")
        if pre_digest != post_digest:
            raise ValueError("initial observation must be an exact public null pair")
        memory = PublicTemporalMemory(sources, post_digest)
    else:
        if args.previous_memory is None:
            raise ValueError("post-action memory update requires previous memory")
        previous_row, memory = load_previous(args.previous_memory)
        if previous_row.get("initial_state_group") != transition.initial_state_group:
            raise ValueError("previous memory and transition groups differ")
        if memory.registered_information_sources != sources:
            raise ValueError(
                "registered public information sources changed mid-episode"
            )
        if memory.current_observation_sha256 != pre_digest:
            raise ValueError("transition pre-observation does not continue memory head")
        candidate = transition.public_action_history.get("last_executed_candidate")
        if not isinstance(candidate, dict):
            raise TypeError("post-action transition lacks executed public candidate")
        candidate_id = " ".join(str(candidate.get("candidate_id", "")).split())
        primitive = " ".join(str(candidate.get("primitive", "")).split()).upper()
        candidate_keys = {
            (
                " ".join(str(row.get("candidate_id", "")).split()),
                " ".join(str(row.get("primitive", "")).split()).upper(),
            )
            for row in transition.candidate_actions
        }
        if not candidate_id or (candidate_id, primitive) not in candidate_keys:
            raise ValueError("executed candidate differs from the public candidate set")
        information_source = (
            candidate_id if primitive in INFORMATION_PRIMITIVES else None
        )
        region_set = (
            prediction_membership(
                calibrated["region_confirmed_empty_prediction_set"][index]
            )
            if information_source is not None
            else ()
        )
        task_set = prediction_membership(
            calibrated["task_complete_prediction_set"][index]
        )
        memory = memory.append(
            PublicObservationEvent.create(
                step=len(memory.events),
                candidate_id=candidate_id,
                primitive=primitive,
                information_source_id=information_source,
                region_confirmed_empty_set=region_set,
                task_complete_set=task_set,
                post_observation_sha256=post_digest,
            )
        )
        previous_input = {
            "path": portable(args.previous_memory),
            "sha256": sha256(args.previous_memory),
        }

    output_row = {
        "schema_version": "piu.public-controller-memory.v2",
        "sample_id": transition.sample_id,
        "initial_state_group": transition.initial_state_group,
        "split": transition.split.value,
        "public_inputs_only": True,
        "online_oracle_inputs": [],
        "alpha": alpha,
        "memory": memory.to_public_history(),
        "inputs": {
            "public_transition": {
                "path": portable(args.public_transition),
                "sha256": sha256(args.public_transition),
            },
            "binding_predictions": {
                "path": portable(args.binding_predictions),
                "sha256": sha256(args.binding_predictions),
            },
            "binding_report": {
                "path": portable(args.binding_report),
                "sha256": sha256(args.binding_report),
            },
            "binder_calibration": {
                "path": portable(args.binder_calibration),
                "sha256": sha256(args.binder_calibration),
            },
            "previous_memory": previous_input,
        },
        "verifier_semantics": "calibrated_post_observation_not_predicted_effect",
        "evaluator_labels_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_row, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"output": portable(args.output), "sha256": sha256(args.output)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
