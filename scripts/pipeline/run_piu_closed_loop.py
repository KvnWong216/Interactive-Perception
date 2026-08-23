#!/usr/bin/env python3
"""Run a resumable observe-bind-effect-dispatch-reobserve PIU episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def candidate_identity(path: Path, sample_id: str) -> tuple[str, str, tuple[str, ...]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    matches = [row for row in rows if row.get("sample_id") == sample_id]
    if len(matches) != 1:
        raise ValueError("initial sample must select one public candidate set")
    row = matches[0]
    if row.get("schema_version") != "piu.public-candidate-set.v1":
        raise ValueError("unsupported public candidate-set schema")
    group = " ".join(str(row.get("initial_state_group", "")).split())
    split = str(row.get("split", ""))
    if not group or split not in {"train", "development", "calibration", "sealed_test"}:
        raise ValueError("candidate-set group/split is invalid")
    physical = tuple(
        str(candidate["candidate_id"])
        for candidate in row.get("candidates", ())
        if str(candidate.get("primitive", "")).upper()
        not in {"STOP", "REPORT_NOT_FOUND"}
    )
    if not physical or len(set(physical)) != len(physical):
        raise ValueError("candidate set requires unique physical candidate IDs")
    return group, split, physical


def load_qualifications(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    value = json.loads(path.read_text())
    if value.get("schema_version") != "piu.qualified-executor-map.v1":
        raise ValueError("unsupported qualified-executor map")
    result = {}
    for candidate_id, artifact in dict(value.get("candidates", {})).items():
        candidate = " ".join(str(candidate_id).split())
        if not isinstance(artifact, dict) or not candidate:
            raise TypeError("qualified-executor map entries are malformed")
        certificate = resolve(Path(artifact["path"]))
        if not certificate.is_file() or sha256(certificate) != artifact.get("sha256"):
            raise ValueError(f"qualification certificate differs for {candidate}")
        result[candidate] = certificate
    return result


def command_paths(args: argparse.Namespace) -> dict[str, Path]:
    names = (
        "scenario_config",
        "candidate_set",
        "binder_checkpoint",
        "binder_training_report",
        "binder_calibration",
        "effect_checkpoint",
        "effect_training_report",
        "effect_calibration",
        "qualification_map",
        "initial_state",
        "output_dir",
    )
    result = {}
    for name in names:
        value = getattr(args, name)
        if value is not None:
            result[name] = resolve(value)
    return result


def initial_capture_command(
    *, args: argparse.Namespace, paths: dict[str, Path], step_dir: Path
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/data/capture_piu_initial_observation.py"),
        "--scenario-config",
        str(paths["scenario_config"]),
        "--candidate-set",
        str(paths["candidate_set"]),
        "--sample-id",
        args.initial_sample_id,
        "--seed",
        str(args.seed),
        "--output-dir",
        str(step_dir / "capture"),
        "--external-simulator",
    ]
    if "initial_state" in paths:
        command.extend(
            [
                "--initial-state",
                str(paths["initial_state"]),
                "--state-key",
                args.state_key,
            ]
        )
    return command


def inference_commands(
    *,
    args: argparse.Namespace,
    paths: dict[str, Path],
    transition: Path,
    sample_id: str,
    previous_memory: Path | None,
    step_dir: Path,
    split: str,
) -> tuple[list[list[str]], dict[str, Path]]:
    features = step_dir / "spatial_prefix.npz"
    binding = step_dir / "binding_online.npz"
    memory = step_dir / "memory.jsonl"
    states = step_dir / "controller_states.jsonl"
    controller = step_dir / "controller.json"
    commands = [
        [
            sys.executable,
            str(ROOT / "scripts/data/extract_piu_spatial_prefix_features_remote.py"),
            "--public",
            str(transition),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--output",
            str(features),
        ],
        [
            sys.executable,
            str(ROOT / "scripts/pipeline/predict_piu_target_binding_online.py"),
            "--checkpoint",
            str(paths["binder_checkpoint"]),
            "--training-report",
            str(paths["binder_training_report"]),
            "--features",
            str(features),
            "--feature-report",
            str(features.with_suffix(".json")),
            "--expected-split",
            split,
            "--output",
            str(binding),
        ],
    ]
    if args.method_id in {"B3", "B4"}:
        commands.append(
            [
                sys.executable,
                str(
                    ROOT
                    / "scripts/pipeline/run_piu_uncalibrated_ablation_controller.py"
                ),
                "--method-id",
                args.method_id,
                "--checkpoint",
                str(paths["effect_checkpoint"]),
                "--training-report",
                str(paths["effect_training_report"]),
                "--features",
                str(features),
                "--feature-report",
                str(features.with_suffix(".json")),
                "--binding-predictions",
                str(binding),
                "--binding-report",
                str(binding.with_suffix(".json")),
                "--expected-split",
                split,
                "--output",
                str(controller),
            ]
        )
        return commands, {
            "features": features,
            "binding": binding,
            "controller": controller,
        }
    memory_command = [
        sys.executable,
        str(ROOT / "scripts/data/update_piu_temporal_memory.py"),
        "--public-transition",
        str(transition),
        "--sample-id",
        sample_id,
        "--binding-predictions",
        str(binding),
        "--binding-report",
        str(binding.with_suffix(".json")),
        "--binder-calibration",
        str(paths["binder_calibration"]),
        "--output",
        str(memory),
    ]
    if previous_memory is not None:
        memory_command.extend(["--previous-memory", str(previous_memory)])
    commands.extend(
        [
            memory_command,
            [
                sys.executable,
                str(ROOT / "scripts/data/build_piu_controller_public_states.py"),
                "--memory",
                str(memory),
                "--output",
                str(states),
            ],
            [
                sys.executable,
                str(ROOT / "scripts/pipeline/run_piu_calibrated_controller.py"),
                "--checkpoint",
                str(paths["effect_checkpoint"]),
                "--method-id",
                args.method_id,
                "--training-report",
                str(paths["effect_training_report"]),
                "--features",
                str(features),
                "--feature-report",
                str(features.with_suffix(".json")),
                "--binding-predictions",
                str(binding),
                "--binding-report",
                str(binding.with_suffix(".json")),
                "--binder-calibration",
                str(paths["binder_calibration"]),
                "--effect-calibration",
                str(paths["effect_calibration"]),
                "--public-state-sets",
                str(states),
                "--expected-split",
                split,
                "--output",
                str(controller),
            ],
        ]
    )
    return commands, {
        "features": features,
        "binding": binding,
        "memory": memory,
        "states": states,
        "controller": controller,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-id", choices=("B3", "B4", "B5", "B8"), default="B8")
    parser.add_argument("--scenario-config", type=Path, required=True)
    parser.add_argument("--candidate-set", type=Path, required=True)
    parser.add_argument("--initial-sample-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--state-key", default="state")
    parser.add_argument("--binder-checkpoint", type=Path, required=True)
    parser.add_argument("--binder-training-report", type=Path, required=True)
    parser.add_argument("--binder-calibration", type=Path)
    parser.add_argument("--effect-checkpoint", type=Path, required=True)
    parser.add_argument("--effect-training-report", type=Path, required=True)
    parser.add_argument("--effect-calibration", type=Path)
    parser.add_argument("--qualification-map", type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--maximum-decisions", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.maximum_decisions < 1:
        raise ValueError("maximum decisions must be positive")
    paths = command_paths(args)
    for name in (
        "scenario_config",
        "candidate_set",
        "binder_checkpoint",
        "binder_training_report",
        "effect_checkpoint",
        "effect_training_report",
    ):
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])
    if args.method_id in {"B5", "B8"}:
        for name in ("binder_calibration", "effect_calibration"):
            if name not in paths or not paths[name].is_file():
                raise ValueError(f"{args.method_id} requires {name.replace('_', ' ')}")
    group, split, physical_candidate_ids = candidate_identity(
        paths["candidate_set"], args.initial_sample_id
    )
    if split == "calibration":
        raise ValueError(
            "closed-loop method execution cannot consume calibration groups"
        )
    qualifications = load_qualifications(paths.get("qualification_map"))
    output_dir = paths["output_dir"]
    first_step = output_dir / "step000"
    capture = initial_capture_command(args=args, paths=paths, step_dir=first_step)
    dry_commands, _ = inference_commands(
        args=args,
        paths=paths,
        transition=first_step / "capture/public_transition.jsonl",
        sample_id=args.initial_sample_id,
        previous_memory=None,
        step_dir=first_step,
        split=split,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": "piu.closed-loop-plan.v1",
                    "method_id": args.method_id,
                    "initial_state_group": group,
                    "split": split,
                    "maximum_decisions": args.maximum_decisions,
                    "external_pi05": f"{args.host}:{args.port}",
                    "local_pi05_loaded": False,
                    "first_decision_commands": [capture, *dry_commands],
                    "qualified_candidate_ids": sorted(qualifications),
                    "unqualified_physical_candidate_ids": sorted(
                        set(physical_candidate_ids) - set(qualifications)
                    ),
                    "execution_ready": set(physical_candidate_ids)
                    <= set(qualifications),
                    "continuation": (
                        "controller decision selects a hash-verified certificate; "
                        "physical dispatch is reobserved and replanned"
                    ),
                },
                indent=2,
            )
        )
        return
    missing_qualifications = set(physical_candidate_ids) - set(qualifications)
    if missing_qualifications:
        raise ValueError(
            "main-method candidate set contains unqualified physical actions "
            f"{sorted(missing_qualifications)}"
        )
    if output_dir.exists():
        raise FileExistsError("closed-loop run directories are immutable")
    output_dir.mkdir(parents=True)
    run(capture)
    transition = first_step / "capture/public_transition.jsonl"
    source_state = first_step / "capture/initial_state.npz"
    current_state = source_state
    previous_memory = None
    receipts = []
    status = "TIMEOUT"
    for step in range(args.maximum_decisions):
        step_dir = output_dir / f"step{step:03d}"
        if step:
            step_dir.mkdir()
        sample_id = (
            args.initial_sample_id
            if step == 0
            else f"{args.initial_sample_id}-step{step:03d}"
        )
        commands, artifacts = inference_commands(
            args=args,
            paths=paths,
            transition=transition,
            sample_id=sample_id,
            previous_memory=previous_memory,
            step_dir=step_dir,
            split=split,
        )
        run(commands[0])
        if split == "sealed_test":
            binder_authorization = step_dir / "binder_online_authorization.json"
            binder_authorization.write_text(
                json.dumps(
                    {
                        "schema_version": "piu.binder-online-sealed-authorization.v1",
                        "checkpoint_sha256": sha256(paths["binder_checkpoint"]),
                        "feature_sha256": sha256(artifacts["features"]),
                        "single_use_output": portable(artifacts["binding"]),
                    },
                    indent=2,
                )
                + "\n"
            )
            commands[1].extend(["--sealed-authorization", str(binder_authorization)])
        run(commands[1])
        if args.method_id in {"B3", "B4"}:
            if split == "sealed_test":
                controller_authorization = (
                    step_dir / "uncalibrated_controller_authorization.json"
                )
                controller_authorization.write_text(
                    json.dumps(
                        {
                            "schema_version": (
                                "piu.uncalibrated-controller-sealed-authorization.v1"
                            ),
                            "method_id": args.method_id,
                            "checkpoint_sha256": sha256(paths["effect_checkpoint"]),
                            "feature_sha256": sha256(artifacts["features"]),
                            "binding_prediction_sha256": sha256(artifacts["binding"]),
                            "single_use_output": portable(artifacts["controller"]),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                commands[2].extend(
                    ["--sealed-authorization", str(controller_authorization)]
                )
            run(commands[2])
        else:
            run(commands[2])
            run(commands[3])
        if split == "sealed_test" and args.method_id in {"B5", "B8"}:
            controller_authorization = step_dir / "controller_authorization.json"
            controller_authorization.write_text(
                json.dumps(
                    {
                        "schema_version": "piu.controller-sealed-authorization.v1",
                        "checkpoint_sha256": sha256(paths["effect_checkpoint"]),
                        "feature_sha256": sha256(artifacts["features"]),
                        "binding_prediction_sha256": sha256(artifacts["binding"]),
                        "binder_calibration_sha256": sha256(
                            paths["binder_calibration"]
                        ),
                        "effect_calibration_sha256": sha256(
                            paths["effect_calibration"]
                        ),
                        "public_state_sets_sha256": sha256(artifacts["states"]),
                        "method_id": args.method_id,
                        "single_use_output": portable(artifacts["controller"]),
                    },
                    indent=2,
                )
                + "\n"
            )
            commands[4].extend(
                ["--sealed-authorization", str(controller_authorization)]
            )
        if args.method_id in {"B5", "B8"}:
            run(commands[4])
        controller = json.loads(artifacts["controller"].read_text())
        matches = [
            row for row in controller["decisions"] if row.get("sample_id") == sample_id
        ]
        if len(matches) != 1:
            raise ValueError("controller report lacks the current sample")
        decision = matches[0]
        candidate_id = decision.get("selected_candidate_id")
        kind = str(decision.get("decision_kind", ""))
        execution_dir = step_dir / "dispatch"
        dispatch = [
            sys.executable,
            str(ROOT / "scripts/pipeline/execute_piu_controller_decision.py"),
            "--controller-report",
            str(artifacts["controller"]),
            "--sample-id",
            sample_id,
            "--scenario-config",
            str(paths["scenario_config"]),
            "--initial-state",
            str(current_state),
            "--seed",
            str(args.seed),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--run-dir",
            str(execution_dir),
        ]
        if kind in {"EXECUTE", "INTERACT"}:
            if candidate_id not in qualifications:
                raise ValueError(
                    f"selected physical candidate {candidate_id!r} has no frozen qualification"
                )
            dispatch.extend(
                ["--primitive-qualification", str(qualifications[candidate_id])]
            )
        run(dispatch)
        receipt = execution_dir / "dispatch.json"
        receipts.append({"path": portable(receipt), "sha256": sha256(receipt)})
        previous_memory = artifacts.get("memory")
        if kind not in {"EXECUTE", "INTERACT"}:
            status = "COMPLETE" if kind in {"STOP", "REPORT_NOT_FOUND"} else "ABSTAINED"
            break
        current_state = execution_dir / "final_state.npz"
        next_sample = f"{args.initial_sample_id}-step{step + 1:03d}"
        next_transition = execution_dir / "next_public_transition.jsonl"
        run(
            [
                sys.executable,
                str(ROOT / "scripts/data/export_piu_public_transition.py"),
                "--dispatch-receipt",
                str(receipt),
                "--sample-id",
                next_sample,
                "--initial-state-group",
                group,
                "--split",
                split,
                "--output",
                str(next_transition),
            ]
        )
        transition = next_transition
    manifest = {
        "schema_version": "piu.closed-loop-run-manifest.v1",
        "method_id": args.method_id,
        "initial_state_group": group,
        "simulator_seed": args.seed,
        "split": split,
        "rollout_status": status,
        "source_state": {
            "path": portable(source_state),
            "sha256": sha256(source_state),
        },
        "dispatch_receipts": receipts,
        "maximum_decisions": args.maximum_decisions,
        "external_pi05": f"{args.host}:{args.port}",
        "local_pi05_loaded": False,
        "paper_method_claim_allowed": False,
    }
    manifest_path = output_dir / "closed_loop_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
