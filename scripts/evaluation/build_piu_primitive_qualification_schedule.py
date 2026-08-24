#!/usr/bin/env python3
"""Freeze exact new groups and execution order for primitive qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.formal_states import validate_state_archive
from piu.primitive_registry import (
    load_primitive_qualification_plan,
    load_primitive_qualification_schedule,
    primitive_qualification_permutation_key,
)
from piu.reproducibility import validate_repro_lock
from piu.splits import load_split_manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def artifact(path: Path) -> dict[str, str]:
    return {"path": portable(path), "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--state",
        action="append",
        nargs=2,
        metavar=("INITIAL_STATE_GROUP", "NPZ_PATH"),
        required=True,
    )
    parser.add_argument(
        "--controller-report",
        action="append",
        nargs=2,
        metavar=("INITIAL_STATE_GROUP", "REPORT_PATH"),
        required=True,
    )
    parser.add_argument("--state-key", default="state")
    parser.add_argument(
        "--baseline-registry",
        type=Path,
        default=ROOT / "configs/experiments/piu_baselines_v1.yaml",
    )
    parser.add_argument(
        "--scenario-config",
        type=Path,
        default=ROOT / "configs/scenarios/original_drawer.yaml",
    )
    parser.add_argument(
        "--repro-manifest",
        type=Path,
        default=ROOT / "configs/experiments/piu_offline_repro_v4.yaml",
    )
    parser.add_argument(
        "--repro-lock",
        type=Path,
        default=ROOT / "results/diagnostics/piu_offline_repro_preflight_v4.json",
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan_path = resolve(args.plan)
    split_path = resolve(args.split_manifest)
    baseline_path = resolve(args.baseline_registry)
    scenario_path = resolve(args.scenario_config)
    repro_manifest_path = resolve(args.repro_manifest)
    repro_lock_path = resolve(args.repro_lock)
    run_root = resolve(args.run_root)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("primitive qualification schedules are immutable")
    if run_root.exists():
        raise FileExistsError("qualification run root must be absent at schedule freeze")
    plan = load_primitive_qualification_plan(plan_path, repository_root=ROOT)
    validate_repro_lock(
        repro_lock_path,
        manifest_path=repro_manifest_path,
        repository_root=ROOT,
    )
    split = load_split_manifest(split_path)
    qualification = {
        str(row["initial_state_group"]): int(row["seed"])
        for row in split["assignments"]
        if row["split_role"] == "primitive_qualification"
    }
    if len(qualification) != int(plan["design"]["trials"]):
        raise ValueError("primitive qualification groups differ from frozen design")
    supplied = {}
    for raw_group, raw_path in args.state:
        group = " ".join(raw_group.split())
        if not group or group in supplied:
            raise ValueError("qualification state groups must be nonempty and unique")
        supplied[group] = resolve(Path(raw_path))
    if set(supplied) != set(qualification):
        raise ValueError("qualification states differ from the reserved split groups")
    controller_reports = {}
    for raw_group, raw_path in args.controller_report:
        group = " ".join(raw_group.split())
        if not group or group in controller_reports:
            raise ValueError("qualification controller groups must be unique")
        controller_reports[group] = resolve(Path(raw_path))
    if set(controller_reports) != set(qualification):
        raise ValueError("qualification controllers differ from reserved groups")
    state_key = " ".join(args.state_key.split())
    if not state_key:
        raise ValueError("qualification state key must be nonempty")
    registry_path = resolve(Path(plan["registry"]["path"]))
    pilot_seeds = set(json.loads(registry_path.read_text()).get("seeds", ()))
    if pilot_seeds & set(qualification.values()):
        raise ValueError("qualification split reuses retrospective pilot seeds")
    baseline = yaml.safe_load(baseline_path.read_text())
    scenario = yaml.safe_load(scenario_path.read_text())
    if baseline.get("schema_version") != "piu.baseline-registry.v1":
        raise ValueError("unsupported qualification baseline registry")
    if scenario.get("schema_version") != "piu.scenario.v1":
        raise ValueError("unsupported qualification scenario config")
    if resolve(Path(baseline["scenario"])).resolve() != resolve(
        Path(scenario["scene"]["bddl"])
    ).resolve():
        raise ValueError("qualification baseline and scenario differ")
    identity_path = resolve(
        Path(baseline["shared_contract"]["checkpoint_identity"])
    )
    plan_digest = sha256(plan_path)
    risk_path = resolve(Path(plan["risk_contract"]["path"]))
    namespace = f"{plan['candidate_id']}::{plan['primitive']}::{plan['context']}"
    rows = []
    state_digests = set()
    candidate_contract = None
    for group, seed in qualification.items():
        state_path = supplied[group]
        shape, dtype = validate_state_archive(state_path, state_key=state_key)
        digest = sha256(state_path)
        if digest in state_digests:
            raise ValueError("qualification groups reuse an opaque source state")
        state_digests.add(digest)
        key = primitive_qualification_permutation_key(
            namespace=namespace,
            plan_sha256=plan_digest,
            initial_state_group=group,
            simulator_seed=seed,
            source_state_sha256=digest,
        )
        controller_path = controller_reports[group]
        from piu.primitive_registry import load_qualification_controller_decision

        controller = load_qualification_controller_decision(
            controller_path,
            candidate_id=str(plan["candidate_id"]),
            primitive=str(plan["primitive"]),
            initial_state_group=group,
            repository_root=ROOT,
        )
        observed_contract = {
            "selected_candidate": controller["candidate"],
            "spatial_reference_mode": controller["spatial_reference_mode"],
            "serializer": "src/piu/executor_bridge.py:serialize_pi05_subtask",
        }
        if candidate_contract is None:
            candidate_contract = observed_contract
        elif observed_contract != candidate_contract:
            raise ValueError("qualification candidate contract changes across groups")
        rows.append(
            {
                "permutation_key": key,
                "initial_state_group": group,
                "simulator_seed": seed,
                "candidate_id": plan["candidate_id"],
                "primitive": plan["primitive"],
                "context": plan["context"],
                "group_id": group,
                "scenario": scenario["id"],
                "state_key": state_key,
                "controller_report": artifact(controller_path),
                "structured_subtask_sha256": controller[
                    "structured_subtask_sha256"
                ],
                "subtask_prompt": controller["structured_subtask"],
                "public_observation_sha256": controller[
                    "public_observation_sha256"
                ],
                "rollout_executed": False,
                "outcome_loaded": False,
                "pre_outcome_only": True,
                "source_state": {
                    "path": portable(state_path),
                    "sha256": digest,
                    "shape": shape,
                    "dtype": dtype,
                },
            }
        )
    rows.sort(key=lambda row: (row["permutation_key"], row["initial_state_group"]))
    for index, row in enumerate(rows):
        row["execution_index"] = index
        run_id = hashlib.sha256(row["initial_state_group"].encode()).hexdigest()[:12]
        row["expected_execution_receipt"] = portable(
            run_root / f"{index:05d}_{run_id}" / "report.json"
        )
    result = {
        "schema_version": "piu.primitive-qualification-schedule.v1",
        "status": "FROZEN_BEFORE_PRIMITIVE_QUALIFICATION_OUTCOMES",
        "claim_scope": "EXECUTOR_QUALIFICATION_ONLY_NOT_TASK_SUCCESS",
        "outcomes_loaded": False,
        "rollout_executed": False,
        "pre_outcome_only": True,
        "execution_receipts_present_at_freeze": False,
        "permutation_namespace": namespace,
        "candidate_contract": candidate_contract,
        "inputs": {
            "plan": artifact(plan_path),
            "risk_contract": artifact(risk_path),
            "split_manifest": artifact(split_path),
            "baseline_registry": artifact(baseline_path),
            "scenario_config": artifact(scenario_path),
            "policy_identity": artifact(identity_path),
            "offline_repro_lock": {
                **artifact(repro_lock_path),
                "manifest_path": portable(repro_manifest_path),
                "manifest_sha256": sha256(repro_manifest_path),
            },
        },
        "seed_allocation": split.get(
            "allocation",
            {
                "rule": "preexisting_frozen_split_manifest",
                "rationale": "synthetic or retained schedule fixture",
                "seed_start": min(qualification.values()),
                "seed_end": max(qualification.values()),
                "count": len(qualification),
                "replacement_after_capture": "prohibited",
                "replacement_after_rollout": "prohibited",
                "rollout_executed": False,
                "outcomes_loaded": False,
                "pre_outcome_only": True,
            },
        ),
        "run_root": portable(run_root),
        "entries": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    load_primitive_qualification_schedule(output, repository_root=ROOT)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
