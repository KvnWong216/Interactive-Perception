#!/usr/bin/env python3
"""Capture a policy-public initial PIU decision state without loading pi0.5."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/infra"), str(ROOT / "src")]

import bootstrap  # noqa: F401

from interactive_perception.policy_client import build_observation
from piu.contracts import PublicTransition, assert_public_policy_value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def candidate_row(path: Path, sample_id: str) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    matches = [row for row in rows if row.get("sample_id") == sample_id]
    if len(matches) != 1:
        raise ValueError("sample ID must select one public candidate set")
    row = matches[0]
    if row.get("schema_version") != "piu.public-candidate-set.v1":
        raise ValueError("unsupported public candidate-set schema")
    if (
        row.get("public_inputs_only") is not True
        or row.get("online_oracle_inputs") != []
    ):
        raise ValueError("initial candidate set must be public-input only")
    assert_public_policy_value(row["candidates"], path="initial.candidates")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-config", type=Path, required=True)
    parser.add_argument("--candidate-set", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--state-key", default="state")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--external-simulator", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    scenario_path = resolve(args.scenario_config)
    candidates_path = resolve(args.candidate_set)
    output_dir = resolve(args.output_dir)
    initial_state_path = (
        resolve(args.initial_state) if args.initial_state is not None else None
    )
    scenario = yaml.safe_load(scenario_path.read_text())
    candidates = candidate_row(candidates_path, args.sample_id)
    plan = {
        "schema_version": "piu.initial-observation-capture-plan.v1",
        "sample_id": args.sample_id,
        "initial_state_group": candidates["initial_state_group"],
        "split": candidates["split"],
        "scenario": portable(scenario_path),
        "seed": args.seed,
        "pi05_loaded": False,
        "external_simulator_required": True,
        "online_oracle_inputs": [],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if not args.external_simulator:
        raise ValueError(
            "initial capture must run in the external simulator environment; "
            "local GPU rendering is prohibited by the release contract"
        )
    if output_dir.exists():
        raise FileExistsError("initial observation captures are immutable")
    from libero.libero.envs import OffScreenRenderEnv

    bddl = resolve(Path(scenario["scene"]["bddl"]))
    prompt = str(scenario["task"]["prompt"])
    output_dir.mkdir(parents=True)
    assets = output_dir / "assets"
    assets.mkdir()
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
    )
    try:
        env.seed(args.seed)
        observation = env.reset()
        if initial_state_path is not None:
            with np.load(initial_state_path) as store:
                state = np.asarray(store[args.state_key], dtype=float)
            observation = env.set_init_state(state)
        else:
            state = np.asarray(env.get_sim_state(), dtype=float)
        state_path = output_dir / "initial_state.npz"
        np.savez_compressed(state_path, state=state)
        packet = build_observation(observation, prompt)
    finally:
        env.close()
    images = {}
    for camera, value in (("agentview", packet.image), ("wrist", packet.wrist_image)):
        path = assets / f"initial_{camera}.png"
        Image.fromarray(value).save(path)
        images[camera] = {
            "path": portable(path),
            "sha256": sha256(path),
            "pixel_sha256": hashlib.sha256(
                np.ascontiguousarray(value, dtype=np.uint8).tobytes()
            ).hexdigest(),
        }
    public_observation = {
        "images": images,
        "public_robot_state": [float(value) for value in packet.state],
    }
    transition = {
        "schema_version": "piu.public-transition.v1",
        "sample_id": args.sample_id,
        "initial_state_group": candidates["initial_state_group"],
        "split": candidates["split"],
        "prompt": prompt,
        "observations": {
            "pre_interaction": public_observation,
            "post_interaction": public_observation,
        },
        "public_action_history": {
            "initial_observation": True,
            "last_executed_candidate": None,
            "events": [],
        },
        "candidate_actions": candidates["candidates"],
        "online_oracle_inputs": [],
    }
    PublicTransition.from_mapping(transition)
    transition_path = output_dir / "public_transition.jsonl"
    transition_path.write_text(json.dumps(transition, sort_keys=True) + "\n")
    report = {
        "schema_version": "piu.initial-observation-capture.v1",
        **plan,
        "source_state": {
            "path": portable(state_path),
            "sha256": sha256(state_path),
            "state_key": "state",
        },
        "inputs": {
            "scenario": {
                "path": portable(scenario_path),
                "sha256": sha256(scenario_path),
            },
            "candidate_set": {
                "path": portable(candidates_path),
                "sha256": sha256(candidates_path),
            },
        },
        "public_transition": {
            "path": portable(transition_path),
            "sha256": sha256(transition_path),
        },
        "evaluator_fields_copied": [],
        "local_pi05_loaded": False,
    }
    report_path = output_dir / "capture.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
