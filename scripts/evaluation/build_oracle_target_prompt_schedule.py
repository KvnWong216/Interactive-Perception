#!/usr/bin/env python3
"""Freeze an outcome-independent order for one oracle target-prompt phase."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.oracle_schedule import (
    artifact,
    load_experiment,
    load_protocol,
    load_schedule,
    phase_specifications,
    portable,
    sha256,
    validate_screen_result,
)
from piu.reproducibility import validate_repro_lock


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/experiments/original_drawer_oracle_target_prompt_pilot_v2.yaml",
    )
    parser.add_argument("--phase", choices=("screen", "confirmation"), required=True)
    parser.add_argument(
        "--schedule-config",
        type=Path,
        default=ROOT
        / "configs/experiments/original_drawer_oracle_target_prompt_schedule_v1.yaml",
    )
    parser.add_argument("--screen-result", type=Path)
    parser.add_argument(
        "--repro-manifest",
        type=Path,
        default=ROOT / "configs/experiments/piu_offline_repro_v3.yaml",
    )
    parser.add_argument(
        "--repro-lock",
        type=Path,
        default=ROOT / "results/diagnostics/piu_offline_repro_preflight_v3.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = resolve(args.config)
    schedule_config_path = resolve(args.schedule_config)
    screen_path = None if args.screen_result is None else resolve(args.screen_result)
    manifest_path = resolve(args.repro_manifest)
    lock_path = resolve(args.repro_lock)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("oracle phase schedules are immutable")
    validate_repro_lock(
        lock_path,
        manifest_path=manifest_path,
        repository_root=ROOT,
    )
    experiment = load_experiment(config_path)
    protocol = load_protocol(
        schedule_config_path,
        repository_root=ROOT,
        experiment_path=config_path,
    )
    selected_style = None
    screen_artifact = None
    if args.phase == "confirmation":
        if screen_path is None or not screen_path.is_file():
            raise ValueError("confirmation scheduling requires the screen result")
        screen = validate_screen_result(
            screen_path,
            repository_root=ROOT,
            experiment_path=config_path,
        )
        selected_style = screen.get("screen", {}).get("selected_style")
        if (
            screen.get("status") != "SCREEN_COMPLETE_AWAITING_CONFIRMATION"
            or not isinstance(selected_style, str)
            or screen.get("experiment", {}).get("sha256") != sha256(config_path)
        ):
            raise ValueError("confirmation requires a unique valid screen selection")
        screen_artifact = artifact(screen_path, repository_root=ROOT)
    elif screen_path is not None:
        raise ValueError("screen scheduling cannot load a screen result")
    specs = phase_specifications(
        experiment, phase=args.phase, selected_style=selected_style
    )
    identity_path = resolve(
        Path(experiment["resource_contract"]["checkpoint_identity"])
    )
    binding = "\0".join(
        (
            sha256(config_path),
            sha256(lock_path),
            args.phase,
            selected_style or "",
            sha256(screen_path) if screen_path is not None else "",
        )
    )
    namespace = str(protocol["namespace"])
    ordered = sorted(
        specs,
        key=lambda item: hashlib.sha256(
            f"{namespace}\0{binding}\0{item[0]}\0{item[1]}".encode()
        ).hexdigest(),
    )
    entries = []
    for index, (style, seed) in enumerate(ordered):
        source = (
            ROOT
            / experiment["source_run_root"]
            / f"seed{seed}"
            / "open_butter/final_state.npz"
        )
        if not source.is_file():
            raise FileNotFoundError(source)
        report = (
            ROOT
            / experiment["run_root"]
            / args.phase
            / style
            / f"seed{seed}/report.json"
        )
        if report.exists():
            raise ValueError("oracle schedule must be frozen before phase outcomes")
        entries.append(
            {
                "execution_index": index,
                "style": style,
                "seed": seed,
                "source_state": artifact(source, repository_root=ROOT),
                "expected_report": portable(report, repository_root=ROOT),
            }
        )
    result = {
        "schema_version": "piu.oracle-target-prompt-execution-schedule.v1",
        "status": "FROZEN_BEFORE_PHASE_OUTCOMES",
        "claim_scope": "EVALUATOR_ONLY_ORACLE_SCHEDULE_NO_PERFORMANCE_RESULT",
        "outcomes_loaded": False,
        "phase": args.phase,
        "selected_style": selected_style,
        "config": artifact(config_path, repository_root=ROOT),
        "schedule_protocol": artifact(
            schedule_config_path, repository_root=ROOT
        ),
        "offline_repro_lock": {
            **artifact(lock_path, repository_root=ROOT),
            "manifest_sha256": sha256(manifest_path),
        },
        "policy_identity": artifact(identity_path, repository_root=ROOT),
        "screen_result": screen_artifact,
        "randomization": {
            "method": protocol["method"],
            "namespace": namespace,
            "binding_sha256": hashlib.sha256(binding.encode()).hexdigest(),
            "within_phase_order_randomized": True,
        },
        "entries": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    load_schedule(output, repository_root=ROOT, config_path=config_path)
    print(
        json.dumps(
            {
                "output": portable(output, repository_root=ROOT),
                "sha256": sha256(output),
                "phase": args.phase,
                "entries": len(entries),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
