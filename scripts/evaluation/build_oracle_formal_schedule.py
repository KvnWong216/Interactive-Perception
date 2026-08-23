#!/usr/bin/env python3
"""Freeze attempted-OPEN plus paired oracle/baseline formal executions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.executor_bridge import serialize_pi05_subtask
from piu.oracle_formal import (
    ARMS,
    SCHEDULE_SCHEMA,
    artifact,
    load_oracle_formal_initial_states,
    load_oracle_formal_plan,
    load_oracle_formal_schedule,
    portable,
    resolve,
    sha256,
)
from piu.primitive_registry import load_primitive_qualification_certificate
from piu.reproducibility import validate_repro_lock
from piu.splits import load_split_manifest


def local(path: Path) -> Path:
    return resolve(path, repository_root=ROOT)


def keyed_order(namespace: str, binding: str, names: list[str]) -> list[str]:
    return sorted(
        names,
        key=lambda name: hashlib.sha256(
            f"{namespace}\0{binding}\0{name}".encode()
        ).hexdigest(),
    )


def publish_validated(value: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pending: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".pending",
            delete=False,
        ) as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            pending = Path(handle.name)
        load_oracle_formal_schedule(pending, repository_root=ROOT)
        os.link(pending, output)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-plan", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--initial-state-manifest", type=Path, required=True)
    parser.add_argument("--open-certificate", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/experiments/original_drawer_oracle_target_prompt_pilot_v2.yaml",
    )
    parser.add_argument(
        "--scenario-config",
        type=Path,
        default=ROOT / "configs/scenarios/original_drawer.yaml",
    )
    parser.add_argument(
        "--formal-execution-config",
        type=Path,
        default=ROOT / "configs/experiments/piu_oracle_formal_v1.yaml",
    )
    parser.add_argument(
        "--baseline-registry",
        type=Path,
        default=ROOT / "configs/experiments/piu_baselines_v1.yaml",
    )
    parser.add_argument(
        "--repro-manifest",
        type=Path,
        default=ROOT / "configs/experiments/piu_offline_repro_v1.yaml",
    )
    parser.add_argument(
        "--repro-lock",
        type=Path,
        default=ROOT / "results/diagnostics/piu_offline_repro_preflight_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan_path = local(args.formal_plan)
    split_path = local(args.split_manifest)
    state_manifest_path = local(args.initial_state_manifest)
    certificate_path = local(args.open_certificate)
    config_path = local(args.config)
    formal_protocol_path = local(args.formal_execution_config)
    scenario_path = local(args.scenario_config)
    baseline_path = local(args.baseline_registry)
    repro_manifest_path = local(args.repro_manifest)
    repro_lock_path = local(args.repro_lock)
    output = local(args.output)
    if output.exists():
        raise FileExistsError("oracle formal schedules are immutable")
    validate_repro_lock(
        repro_lock_path,
        manifest_path=repro_manifest_path,
        repository_root=ROOT,
    )
    plan, config, pilot = load_oracle_formal_plan(
        plan_path, repository_root=ROOT
    )
    if plan["status"] != "PROSPECTIVE_GROUP_COUNT_FROZEN":
        raise ValueError("blocked oracle formal plan cannot be scheduled")
    if sha256(config_path) != plan["protocol"]["sha256"]:
        raise ValueError("requested config differs from the oracle formal plan")
    pilot_path = local(Path(plan["pilot"]["path"]))
    split = load_split_manifest(split_path)
    states = load_oracle_formal_initial_states(
        state_manifest_path, repository_root=ROOT
    )
    if states["split_manifest"]["sha256"] != sha256(split_path):
        raise ValueError("oracle formal states use another split")
    scenario = yaml.safe_load(scenario_path.read_text())
    baseline = yaml.safe_load(baseline_path.read_text())
    formal = yaml.safe_load(formal_protocol_path.read_text())
    if scenario.get("schema_version") != "piu.scenario.v1":
        raise ValueError("unsupported oracle formal scenario")
    if baseline.get("schema_version") != "piu.baseline-registry.v1":
        raise ValueError("unsupported oracle formal baseline registry")
    if (
        formal.get("schema_version") != "piu.oracle-formal-experiment.v1"
        or formal.get("status") != "frozen_before_formal_oracle_outcomes"
        or local(Path(formal["oracle_pilot_protocol"])).resolve()
        != config_path.resolve()
        or formal.get("scenario_config")
        != portable(scenario_path, repository_root=ROOT)
    ):
        raise ValueError("unsupported oracle formal execution protocol")
    if local(Path(baseline["scenario"])).resolve() != local(
        Path(scenario["scene"]["bddl"])
    ).resolve():
        raise ValueError("oracle formal baseline registry uses another scenario")
    if split.get("scenario") != scenario.get("id"):
        raise ValueError("oracle formal split uses another scenario")
    if config.get("scenario_config") != portable(scenario_path, repository_root=ROOT):
        raise ValueError("oracle formal config uses another scenario")
    identity_path = local(Path(config["resource_contract"]["checkpoint_identity"]))
    certificate = load_primitive_qualification_certificate(
        certificate_path, repository_root=ROOT
    )
    if (
        certificate.get("status") != "FORMALLY_QUALIFIED"
        or certificate.get("paper_method_action_authorized") is not True
        or certificate.get("primitive") != formal["source_primitive"]
        or certificate.get("candidate_id") != formal["source_candidate_id"]
    ):
        raise ValueError("oracle formal source OPEN is not exactly qualified")
    candidate = certificate["candidate_contract"]["selected_candidate"]
    source_subtask = serialize_pi05_subtask(candidate, spatial_references=())
    if (
        not source_subtask
        or certificate["candidate_contract"]["spatial_reference_mode"] != "none"
    ):
        raise ValueError("oracle formal source OPEN is not a non-spatial stimulus")
    state_rows = {row["initial_state_group"]: row for row in states["states"]}
    expected_count = int(plan["prospective_group_count"])
    if len(state_rows) != expected_count:
        raise ValueError("oracle formal state count differs from the power plan")
    prior_oracle_seeds = {int(seed) for seed in config["preflight"]["source_seeds"]}
    formal_seeds = {int(row["simulator_seed"]) for row in states["states"]}
    qualification_schedule_path = local(Path(certificate["schedule"]["path"]))
    qualification_schedule = json.loads(qualification_schedule_path.read_text())
    qualification_groups = {
        str(row["initial_state_group"]) for row in qualification_schedule["entries"]
    }
    qualification_seeds = {
        int(row["simulator_seed"]) for row in qualification_schedule["entries"]
    }
    if prior_oracle_seeds & formal_seeds:
        raise ValueError("oracle formal states reuse a preflight/pilot seed")
    if qualification_groups & set(state_rows) or qualification_seeds & formal_seeds:
        raise ValueError("oracle formal states reuse OPEN qualification data")
    schedule_contract = formal["execution_schedule"]
    if (
        schedule_contract.get("method")
        != "sha256_keyed_outcome_independent_permutation"
        or schedule_contract.get("group_order_randomized") is not True
        or schedule_contract.get("within_group_arm_order_randomized") is not True
        or schedule_contract.get("outcomes_loaded") is not False
    ):
        raise ValueError("oracle formal schedule contract is not outcome-independent")
    binding = "\0".join(
        sha256(path)
        for path in (
            plan_path,
            pilot_path,
            state_manifest_path,
            split_path,
            config_path,
            formal_protocol_path,
            scenario_path,
            baseline_path,
            identity_path,
            certificate_path,
            repro_lock_path,
        )
    )
    namespace = str(schedule_contract["namespace"])
    run_root = local(Path(formal["run_root"]))
    group_order = keyed_order(namespace, binding, list(state_rows))
    entries = []
    for index, group in enumerate(group_order):
        group_root = run_root / f"{index:05d}_{hashlib.sha256(group.encode()).hexdigest()[:12]}"
        entries.append(
            {
                "execution_index": index,
                "initial_state_group": group,
                "simulator_seed": state_rows[group]["simulator_seed"],
                "state_key": state_rows[group]["state_key"],
                "source_state": state_rows[group]["source_state"],
                "arm_order": keyed_order(namespace, f"{binding}\0{group}", list(ARMS)),
                "expected_started_ticket": portable(
                    group_root / "started.json", repository_root=ROOT
                ),
                "expected_open_report": portable(
                    group_root / "open_source/report.json", repository_root=ROOT
                ),
                "expected_post_open_state": portable(
                    group_root / "open_source/final_state.npz", repository_root=ROOT
                ),
                "expected_arm_reports": {
                    arm: portable(group_root / arm / "report.json", repository_root=ROOT)
                    for arm in ARMS
                },
                "expected_group_receipt": portable(
                    group_root / "receipt.json", repository_root=ROOT
                ),
            }
        )
    result = {
        "schema_version": SCHEDULE_SCHEMA,
        "status": "FROZEN_BEFORE_ORACLE_FORMAL_OUTCOMES",
        "claim_scope": "CAUSAL_ORACLE_DESIGN_ONLY_NO_OUTCOMES",
        "outcomes_loaded": False,
        "selected_style": pilot["confirmation"]["selected_style"],
        "primary_outcome": formal["outcome"],
        "arms": list(ARMS),
        "source_open_candidate": candidate,
        "source_open_subtask": source_subtask,
        "failure_accounting": formal["failure_accounting"],
        "causal_interpretation": formal["causal_interpretation"],
        "run_root": portable(run_root, repository_root=ROOT),
        "randomization": {
            "method": schedule_contract["method"],
            "namespace": namespace,
            "binding_sha256": hashlib.sha256(binding.encode()).hexdigest(),
            "group_order_randomized": True,
            "within_group_arm_order_randomized": True,
        },
        "inputs": {
            "formal_plan": artifact(plan_path, repository_root=ROOT),
            "pilot": artifact(pilot_path, repository_root=ROOT),
            "experiment": artifact(config_path, repository_root=ROOT),
            "formal_execution_protocol": artifact(
                formal_protocol_path, repository_root=ROOT
            ),
            "split_manifest": artifact(split_path, repository_root=ROOT),
            "initial_state_manifest": artifact(state_manifest_path, repository_root=ROOT),
            "scenario_config": artifact(scenario_path, repository_root=ROOT),
            "baseline_registry": artifact(baseline_path, repository_root=ROOT),
            "policy_identity": artifact(identity_path, repository_root=ROOT),
            "open_qualification_certificate": artifact(
                certificate_path, repository_root=ROOT
            ),
            "offline_repro_lock": {
                **artifact(repro_lock_path, repository_root=ROOT),
                "manifest_sha256": sha256(repro_manifest_path),
            },
        },
        "entries": entries,
    }
    publish_validated(result, output)
    print(json.dumps({"output": portable(output, repository_root=ROOT), **result}, indent=2))


if __name__ == "__main__":
    main()
