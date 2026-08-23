#!/usr/bin/env python3
"""Freeze an outcome-independent B0--B8 execution schedule for formal groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

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


def keyed_order(namespace: str, binding: str, names: list[str]) -> list[str]:
    return sorted(
        names,
        key=lambda name: hashlib.sha256(
            f"{namespace}\0{binding}\0{name}".encode()
        ).hexdigest(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-plan", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=ROOT / "configs/experiments/piu_formal_analysis_v1.yaml",
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
    plan_path = resolve(args.formal_plan)
    split_path = resolve(args.split_manifest)
    config_path = resolve(args.analysis_config)
    registry_path = resolve(args.baseline_registry)
    repro_manifest_path = resolve(args.repro_manifest)
    repro_lock_path = resolve(args.repro_lock)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("formal execution schedules are immutable")
    validate_repro_lock(
        repro_lock_path,
        manifest_path=repro_manifest_path,
        repository_root=ROOT,
    )
    plan = json.loads(plan_path.read_text())
    config = yaml.safe_load(config_path.read_text())
    registry = yaml.safe_load(registry_path.read_text())
    split = load_split_manifest(split_path)
    if (
        plan.get("schema_version") != "piu.formal-paired-test-plan.v1"
        or plan.get("status") != "PROSPECTIVE_GROUP_COUNT_FROZEN"
    ):
        raise ValueError("formal scheduling requires a frozen prospective plan")
    if config.get("schema_version") != "piu.formal-analysis-experiment.v1":
        raise ValueError("unsupported formal analysis config")
    if registry.get("schema_version") != "piu.baseline-registry.v1":
        raise ValueError("unsupported baseline registry")
    if plan.get("config", {}).get("sha256") != sha256(config_path):
        raise ValueError("formal plan was created under another analysis config")
    if plan.get("baseline_registry", {}).get("sha256") != sha256(registry_path):
        raise ValueError("formal plan was created under another baseline registry")
    if plan.get("offline_repro_lock", {}).get("sha256") != sha256(repro_lock_path):
        raise ValueError("formal plan was created under another offline release lock")
    prospective = config["prospective_design"]
    primary = config["primary"]
    comparison = plan.get("comparison", {})
    for name in ("treatment", "comparator", "outcome"):
        if comparison.get(name) != prospective[name] or prospective[name] != primary[name]:
            raise ValueError(f"formal schedule comparison differs at {name}")
    methods = [str(row["id"]) for row in registry["methods"]]
    required_methods = [str(item) for item in config["population"]["required_method_ids"]]
    if methods != required_methods or methods != [f"B{index}" for index in range(9)]:
        raise ValueError("formal schedule requires the frozen B0--B8 registry order")
    pilot_groups = set(plan.get("pilot", {}).get("groups", ()))
    all_groups = {row["initial_state_group"] for row in split["assignments"]}
    if pilot_groups & all_groups:
        raise ValueError("formal split reuses a development pilot group")
    sealed_rows = [
        row for row in split["assignments"] if row["split_role"] == "sealed_test"
    ]
    expected_count = int(plan["design"]["prospective_group_count"])
    if len(sealed_rows) != expected_count:
        raise ValueError("sealed-test cohort size differs from prospective power plan")
    identity_path = resolve(Path(registry["shared_contract"]["checkpoint_identity"]))
    identity_digest = sha256(identity_path)
    if plan.get("pilot", {}).get("policy_identity_sha256") != identity_digest:
        raise ValueError("pilot and formal registry use different frozen policy identities")
    schedule_config = config["execution_schedule"]
    if (
        schedule_config.get("method")
        != "sha256_keyed_outcome_independent_permutation"
        or schedule_config.get("outcomes_loaded") is not False
        or schedule_config.get("group_order_randomized") is not True
        or schedule_config.get("within_group_method_order_randomized") is not True
    ):
        raise ValueError("formal execution schedule is not outcome-independent")
    namespace = str(schedule_config["namespace"])
    binding = "\0".join(
        (sha256(plan_path), sha256(split_path), sha256(config_path), sha256(registry_path))
    )
    by_group = {row["initial_state_group"]: row for row in sealed_rows}
    group_order = keyed_order(namespace, binding, list(by_group))
    entries = []
    for group_position, group in enumerate(group_order):
        method_order = keyed_order(
            namespace, f"{binding}\0{group}", list(methods)
        )
        for within_group_order, method in enumerate(method_order):
            entries.append(
                {
                    "execution_index": len(entries),
                    "group_order": group_position,
                    "within_group_order": within_group_order,
                    "initial_state_group": group,
                    "simulator_seed": int(by_group[group]["seed"]),
                    "method_id": method,
                }
            )
    result = {
        "schema_version": "piu.formal-execution-schedule.v1",
        "status": "FROZEN_BEFORE_FORMAL_OUTCOME_COLLECTION",
        "claim_scope": "DESIGN_ONLY_NO_OUTCOMES_LOADED",
        "outcomes_loaded": False,
        "randomization": {
            "method": schedule_config["method"],
            "namespace": namespace,
            "binding_sha256": hashlib.sha256(binding.encode()).hexdigest(),
            "group_order_randomized": True,
            "within_group_method_order_randomized": True,
        },
        "inputs": {
            "formal_plan": {"path": portable(plan_path), "sha256": sha256(plan_path)},
            "split_manifest": {"path": portable(split_path), "sha256": sha256(split_path)},
            "analysis_config": {"path": portable(config_path), "sha256": sha256(config_path)},
            "baseline_registry": {"path": portable(registry_path), "sha256": sha256(registry_path)},
            "policy_identity": {"path": portable(identity_path), "sha256": identity_digest},
            "offline_repro_lock": {
                "path": portable(repro_lock_path),
                "sha256": sha256(repro_lock_path),
                "manifest_sha256": sha256(repro_manifest_path),
            },
        },
        "planned_groups": expected_count,
        "planned_methods": methods,
        "entries": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
