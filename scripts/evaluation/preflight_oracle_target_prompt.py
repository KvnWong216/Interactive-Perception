#!/usr/bin/env python3
"""Preflight oracle prompts on retained post-OPEN states without a policy server."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/infra"), str(ROOT / "src")]
import bootstrap  # noqa: F401

from interactive_perception.oracle_visual_prompt import build_oracle_target_packet

SCHEMAS = frozenset(
    {
        "calibrated-interaction.oracle-target-prompt-gate.v1",
        "calibrated-interaction.oracle-target-prompt-pilot.v2",
    }
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if config.get("schema_version") not in SCHEMAS:
        raise ValueError(f"unsupported oracle gate schema in {path}")
    return config


def preflight(config_path: Path) -> dict[str, Any]:
    from libero.libero.envs import SegmentationRenderEnv

    config = load_config(config_path)
    scenario = yaml.safe_load((ROOT / config["scenario_config"]).read_text())
    bddl = ROOT / scenario["scene"]["bddl"]
    execution = config["execution"]
    threshold = int(
        execution.get(
            "target_presence_minimum_pixels",
            execution.get("target_visible_pixels_minimum"),
        )
    )
    env = SegmentationRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
    )
    rows = []
    try:
        for seed_value in config["preflight"]["source_seeds"]:
            seed = int(seed_value)
            env.seed(seed)
            env.reset()
            state_path = (
                ROOT
                / config["source_run_root"]
                / f"seed{seed}"
                / "open_butter/final_state.npz"
            )
            with np.load(state_path) as store:
                state = np.asarray(store["state"], dtype=float)
            observation = env.set_init_state(state)
            target_id = int(env.instance_to_id[execution["target_object"]])
            prompted = build_oracle_target_packet(
                observation,
                execution["prompt"],
                target_instance_id=target_id,
                style="box",
            )
            diagnostics = prompted.diagnostics
            maximum_visibility = max(diagnostics.visible_pixels.values())
            rows.append(
                {
                    "seed": seed,
                    "source_initial_state": {
                        "path": portable(state_path),
                        "sha256": sha256(state_path),
                        "state_key": "state",
                    },
                    "target_instance_id": target_id,
                    "visible_pixels": dict(diagnostics.visible_pixels),
                    "boxes_xyxy_half_open": {
                        camera: box.as_list() if box is not None else None
                        for camera, box in diagnostics.boxes.items()
                    },
                    "packet": {
                        "serialized_keys": sorted(prompted.packet.to_openpi()),
                        "agentview_shape": list(prompted.packet.image.shape),
                        "wrist_shape": list(prompted.packet.wrist_image.shape),
                        "state_shape": list(prompted.packet.state.shape),
                    },
                    "eligible": maximum_visibility >= threshold,
                }
            )
    finally:
        env.close()
    eligible = [row["seed"] for row in rows if row["eligible"]]
    expected = [int(seed) for seed in config["preflight"]["expected_eligible_seeds"]]
    if eligible != expected:
        raise ValueError(
            f"eligible seeds {eligible} differ from preregistered {expected}"
        )
    return {
        "schema_version": "calibrated-interaction.oracle-prompt-preflight.v2",
        "status": "PASS",
        "policy_server_contacted": False,
        "policy_actions_sampled": False,
        "claim_scope": "EVALUATOR_ONLY_INPUT_AND_RENDERING_PREFLIGHT",
        "experiment": {"path": portable(config_path), "sha256": sha256(config_path)},
        "target_presence_minimum_pixels": threshold,
        "eligible_seeds": eligible,
        "excluded_seeds": [row["seed"] for row in rows if not row["eligible"]],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/experiments/original_drawer_oracle_target_prompt_pilot_v2.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"preflight result is immutable: {output}")
    result = preflight(config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": portable(output),
                "status": result["status"],
                "eligible_seeds": result["eligible_seeds"],
                "excluded_seeds": result["excluded_seeds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
