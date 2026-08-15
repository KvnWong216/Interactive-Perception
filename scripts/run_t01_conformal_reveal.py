#!/usr/bin/env python3
"""Run G4 routing followed by the G5-authorized T01 reveal executor."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from collect_libero_intent_calibration import chunk_features  # noqa: E402
from interactive_perception.policy_client import OpenPiWebsocketPolicy, build_observation  # noqa: E402
from interactive_perception.semantic_conformal import MondrianSemanticConformalCalibrator  # noqa: E402
from run_pure_pi05_scenario_sr import capture_split_frame  # noqa: E402

DUMMY_ACTION = [0.0] * 6 + [-1.0]
FINAL_PROMPT = "Place the butter in the basket"
OPEN_PROMPT = "Open the middle layer of the drawer"
MIDDLE_JOINT = "wooden_cabinet_1_middle_level"
OPEN_LIMIT = -0.14  # Exact LIBERO WoodenCabinet.is_open boundary.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(90, 120)))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument(
        "--executor-steps",
        type=int,
        default=300,
        help="Fixed option horizon; controller never reads the evaluator joint.",
    )
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "results/calibration/semantic_intent_g4_t01_binary_audit_v5.json",
    )
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-seeds", type=int, nargs="+", default=[90])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from libero.libero.envs import OffScreenRenderEnv

    bddl = ROOT / "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl"
    artifact = json.loads(args.artifact.read_text())
    labels = tuple(artifact["labels"])
    calibrator = MondrianSemanticConformalCalibrator(
        alpha=float(artifact["alpha"]),
        thresholds={key: float(value) for key, value in artifact["thresholds"].items()},
        labels=labels,
        calibration_size_per_class={
            key: int(value) for key, value in artifact["calibration_size_per_class"].items()
        },
        policy_id=str(artifact["policy_id"]),
        split_id=str(artifact["split_id"]),
    )
    prototypes = {
        label: np.asarray(value, dtype=np.float64)
        for label, value in artifact["prototypes"].items()
    }
    scale = float(artifact["prototype_scale"])
    source = yaml.safe_load((ROOT / "benchmarks/interactive_manipulation_v0/benchmark.yaml").read_text())
    demo = {
        "demo_camera": source["backend"]["demo_camera"],
        "demo_camera_pose": source["backend"]["demo_camera_pose"],
        "demo_labels": ["FIRST-PERSON / WRIST", "THIRD-PERSON / BEV"],
    }
    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port)
    rows = []
    for seed in args.seeds:
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        writer = None
        try:
            env.seed(seed)
            obs = env.reset()
            video_path = None
            if args.video_dir is not None and seed in args.video_seeds:
                video_path = args.video_dir / f"t01_conformal_reveal_seed{seed:03d}.mp4"
                video_path.parent.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(video_path, fps=20)
            for _ in range(10):
                obs, _, _, _ = env.step(DUMMY_ACTION)

            proposal_chunks = policy.sample_chunks(build_observation(obs, FINAL_PROMPT), args.samples)
            value = np.asarray([chunk_features(chunk) for chunk in proposal_chunks]).mean(axis=0)
            distances = {
                label: float(np.linalg.norm(value - center) / scale)
                for label, center in prototypes.items()
            }
            weights = {label: float(np.exp(-distance)) for label, distance in distances.items()}
            total = sum(weights.values())
            probabilities = {label: weight / total for label, weight in weights.items()}
            prediction = calibrator.predict(weights)
            routed = prediction == ("REMOVE_OCCLUDER",)

            minimum_joint = float(env.env.sim.data.get_joint_qpos(MIDDLE_JOINT))
            if routed:
                plan: collections.deque[np.ndarray] = collections.deque()
                for step in range(args.executor_steps):
                    if writer is not None and step % 2 == 0:
                        writer.append_data(capture_split_frame(env, obs, demo))
                    if not plan:
                        chunk = policy.sample_chunks(build_observation(obs, OPEN_PROMPT), 1)[0]
                        plan.extend(chunk[: args.replan_steps])
                    obs, _, _, _ = env.step(plan.popleft().tolist())
                    minimum_joint = min(
                        minimum_joint,
                        float(env.env.sim.data.get_joint_qpos(MIDDLE_JOINT)),
                    )
            reveal_success = minimum_joint < OPEN_LIMIT
            row = {
                "seed": seed,
                "prediction_set": list(prediction),
                "probabilities": probabilities,
                "routed": routed,
                "executor_steps": args.executor_steps if routed else 0,
                "middle_joint_minimum": minimum_joint,
                "reveal_success": reveal_success,
                "video": str(video_path) if video_path else None,
            }
            rows.append(row)
            print(
                f"seed={seed} set={list(prediction)} routed={routed} "
                f"reveal={reveal_success} joint_min={minimum_joint:.6f}",
                flush=True,
            )
        finally:
            if writer is not None:
                writer.close()
            env.close()

    report = {
        "schema_version": "interactive-perception.t01-conformal-reveal.v1",
        "policy": "pi05_libero",
        "scene": str(bddl.relative_to(ROOT)),
        "final_prompt": FINAL_PROMPT,
        "executor_prompt": OPEN_PROMPT,
        "artifact": str(args.artifact),
        "test_seeds": args.seeds,
        "calibration_test_disjoint": min(args.seeds) >= 90,
        "policy_inputs": ["stock agentview RGB", "stock wrist RGB", "robot state", "prompt"],
        "evaluator_only": [
            "drawer joint logged during execution without affecting control and used for reveal scoring"
        ],
        "controller_joint_reads": 0,
        "episodes": len(rows),
        "routed_remove": sum(row["routed"] for row in rows),
        "reveal_successes": sum(row["reveal_success"] for row in rows),
        "reveal_success_rate": float(np.mean([row["reveal_success"] for row in rows])),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
