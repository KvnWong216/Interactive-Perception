#!/usr/bin/env python3
"""Audit whether the frozen T01 router changes with prompt and public scene state.

The policy receives only its stock LIBERO observations and the requested final
goal.  Labels come from the benchmark construction and are used only after
inference for evaluation.  In particular, no simulator joint or object pose is
read by the router.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from collect_libero_intent_calibration import chunk_features  # noqa: E402
from interactive_perception.policy_client import (  # noqa: E402
    OpenPiWebsocketPolicy,
    build_observation,
)
from interactive_perception.semantic_conformal import (  # noqa: E402
    MondrianSemanticConformalCalibrator,
)

DUMMY_ACTION = [0.0] * 6 + [-1.0]


def load_router(path: Path):
    artifact = json.loads(path.read_text())
    labels = tuple(artifact["labels"])
    calibrator = MondrianSemanticConformalCalibrator(
        alpha=float(artifact["alpha"]),
        thresholds={key: float(value) for key, value in artifact["thresholds"].items()},
        labels=labels,
        calibration_size_per_class={
            key: int(value)
            for key, value in artifact["calibration_size_per_class"].items()
        },
        policy_id=str(artifact["policy_id"]),
        split_id=str(artifact["split_id"]),
    )
    prototypes = {
        label: np.asarray(value, dtype=np.float64)
        for label, value in artifact["prototypes"].items()
    }
    return artifact, calibrator, prototypes, float(artifact["prototype_scale"])


def score(chunks, *, prototypes, scale, calibrator):
    value = np.asarray([chunk_features(chunk) for chunk in chunks]).mean(axis=0)
    distances = {
        label: float(np.linalg.norm(value - center) / scale)
        for label, center in prototypes.items()
    }
    evidence = {label: float(np.exp(-distance)) for label, distance in distances.items()}
    total = sum(evidence.values())
    probabilities = {label: weight / total for label, weight in evidence.items()}
    prediction = calibrator.predict(evidence)
    return distances, probabilities, prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(190, 220)))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT
        / "results/calibration/semantic_intent_g4_t01_binary_audit_v5.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/calibration/t01_prompt_state_counterfactual_30seed_v1.json",
    )
    args = parser.parse_args()

    from libero.libero.envs import OffScreenRenderEnv

    artifact, calibrator, prototypes, scale = load_router(args.artifact)
    conditions = (
        {
            "id": "closed_hidden_butter",
            "bddl": ROOT
            / "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl",
            "prompt": "Place the butter in the basket",
            "required_route": "REMOVE_OCCLUDER",
            "contrast": "same prompt, closed drawer",
        },
        {
            "id": "closed_visible_cream_cheese",
            "bddl": ROOT
            / "scenarios/t01_stock_ladder_v1/T01D_hidden_butter_retrieval.bddl",
            "prompt": "Place the cream cheese in the basket",
            "required_route": "ACT",
            "contrast": "same closed scene, visible target named by prompt",
        },
        {
            "id": "open_visible_butter",
            "bddl": ROOT
            / "scenarios/t01_stock_ladder_v1/T01E_open_drawer_retrieval.bddl",
            "prompt": "Place the butter in the basket",
            "required_route": "ACT",
            "contrast": "same prompt, open drawer",
        },
    )
    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port)
    rows = []
    for condition in conditions:
        env = OffScreenRenderEnv(
            bddl_file_name=str(condition["bddl"]),
            camera_heights=256,
            camera_widths=256,
        )
        try:
            for seed in args.seeds:
                env.seed(seed)
                obs = env.reset()
                for _ in range(args.wait_steps):
                    obs, _, _, _ = env.step(DUMMY_ACTION)
                chunks = policy.sample_chunks(
                    build_observation(obs, str(condition["prompt"])), args.samples
                )
                distances, probabilities, prediction = score(
                    chunks,
                    prototypes=prototypes,
                    scale=scale,
                    calibrator=calibrator,
                )
                required = str(condition["required_route"])
                singleton_correct = prediction == (required,)
                rows.append(
                    {
                        "condition": condition["id"],
                        "seed": seed,
                        "prompt": condition["prompt"],
                        "required_route": required,
                        "distances": distances,
                        "probabilities": probabilities,
                        "prediction_set": list(prediction),
                        "covered": required in prediction,
                        "singleton_correct": singleton_correct,
                    }
                )
                print(
                    f"{condition['id']} seed={seed} set={list(prediction)} "
                    f"correct={singleton_correct}",
                    flush=True,
                )
        finally:
            env.close()

    summaries = {}
    for condition in conditions:
        subset = [row for row in rows if row["condition"] == condition["id"]]
        summaries[str(condition["id"])] = {
            "contrast": condition["contrast"],
            "prompt": condition["prompt"],
            "required_route": condition["required_route"],
            "trials": len(subset),
            "coverage": float(np.mean([row["covered"] for row in subset])),
            "singleton_accuracy": float(
                np.mean([row["singleton_correct"] for row in subset])
            ),
            "mean_prediction_set_size": float(
                np.mean([len(row["prediction_set"]) for row in subset])
            ),
        }
    report = {
        "schema_version": "interactive-perception.prompt-state-counterfactual.v1",
        "claim": "router response to prompt target and public drawer state",
        "non_claim": "calibrated target belief or downstream task success",
        "artifact": str(args.artifact.relative_to(ROOT)),
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "seeds": args.seeds,
        "samples_per_observation": args.samples,
        "policy_inputs": ["agentview RGB", "wrist RGB", "robot state", "prompt"],
        "controller_oracle_inputs": [],
        "conditions": summaries,
        "overall_singleton_accuracy": float(
            np.mean([row["singleton_correct"] for row in rows])
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"conditions": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
