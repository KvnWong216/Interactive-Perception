#!/usr/bin/env python3
"""Probe a final-goal-only hidden-target prompt with the frozen T01 G4 artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from collect_libero_intent_calibration import chunk_features  # noqa: E402
from interactive_perception.policy_client import OpenPiWebsocketPolicy, build_observation  # noqa: E402
from interactive_perception.semantic_conformal import MondrianSemanticConformalCalibrator  # noqa: E402

DUMMY_ACTION = [0.0] * 6 + [-1.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(60, 90)))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "results/calibration/semantic_intent_g4_t01_binary_audit_v5.json",
    )
    parser.add_argument(
        "--bddl",
        type=Path,
        default=ROOT / "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl",
    )
    parser.add_argument("--prompt", default="Place the butter in the basket")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from libero.libero.envs import OffScreenRenderEnv

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
    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port)
    rows = []
    for seed in args.seeds:
        env = OffScreenRenderEnv(
            bddl_file_name=str(args.bddl), camera_heights=256, camera_widths=256
        )
        try:
            env.seed(seed)
            obs = env.reset()
            for _ in range(args.wait_steps):
                obs, _, _, _ = env.step(DUMMY_ACTION)
            chunks = policy.sample_chunks(build_observation(obs, args.prompt), args.samples)
            value = np.asarray([chunk_features(chunk) for chunk in chunks]).mean(axis=0)
            distances = {
                label: float(np.linalg.norm(value - center) / scale)
                for label, center in prototypes.items()
            }
            weights = {label: float(np.exp(-distance)) for label, distance in distances.items()}
            total = sum(weights.values())
            probabilities = {label: weight / total for label, weight in weights.items()}
            prediction = calibrator.predict(weights)
            if prediction == ("REMOVE_OCCLUDER",):
                decision = "REMOVE_OCCLUDER"
            elif prediction == ("ACT",):
                decision = "UNSAFE_ACT"
            else:
                decision = "ABSTAIN"
            row = {
                "seed": seed,
                "true_required_route": "REMOVE_OCCLUDER",
                "probabilities": probabilities,
                "prediction_set": list(prediction),
                "decision": decision,
                "covered": "REMOVE_OCCLUDER" in prediction,
            }
            rows.append(row)
            print(
                f"seed={seed} set={list(prediction)} "
                f"p_remove={probabilities['REMOVE_OCCLUDER']:.6f}",
                flush=True,
            )
        finally:
            env.close()

    report = {
        "schema_version": "interactive-perception.t01-route-probe.v1",
        "artifact": str(args.artifact),
        "scene": str(args.bddl),
        "prompt": args.prompt,
        "seeds": args.seeds,
        "samples_per_observation": args.samples,
        "oracle_policy_inputs": [],
        "required_route": "REMOVE_OCCLUDER",
        "coverage": float(np.mean([row["covered"] for row in rows])),
        "decision_counts": {
            decision: sum(row["decision"] == decision for row in rows)
            for decision in ("REMOVE_OCCLUDER", "UNSAFE_ACT", "ABSTAIN")
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
