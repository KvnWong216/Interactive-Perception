#!/usr/bin/env python3
"""Fit a frame-level RGB target-evidence head with temporal-OR calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interactive_perception.action_outcome import (  # noqa: E402
    BalancedRidgeMulticlass,
    FeatureStandardizer,
    PI05_PATCH_EQUIVALENT_TARGET_PIXELS,
    TemporalTargetEvidencePredictor,
    temporal_target_score_summary,
)
from interactive_perception.prompt_state import BalancedRidgeBinary  # noqa: E402
from interactive_perception.semantic_conformal import (  # noqa: E402
    MondrianSemanticConformalCalibrator,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_truth(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [
            max(point["target_pixels"].values())
            >= PI05_PATCH_EQUIVALENT_TARGET_PIXELS
            for row in rows
            for point in row["evaluator_only"]["visibility_history"]
        ],
        dtype=bool,
    ).reshape(len(rows), 6)


def episode_values(
    frame_probe: BalancedRidgeBinary,
    frame_standardizer: FeatureStandardizer,
    selected_history: np.ndarray,
) -> np.ndarray:
    flat = selected_history.reshape(-1, selected_history.shape[-1])
    scores = frame_probe.score(frame_standardizer.transform(flat)).reshape(-1, 6)
    return np.stack([temporal_target_score_summary(row) for row in scores])


def metrics(
    truth: np.ndarray, predictions: list[tuple[str, ...]]
) -> dict:
    result = {}
    for label in ("REVEALED", "NOT_REVEALED"):
        selected = [
            prediction
            for prediction, observed in zip(predictions, truth, strict=True)
            if observed == label
        ]
        result[label] = {
            "trials": len(selected),
            "coverage": float(np.mean([label in value for value in selected])),
            "singleton_accuracy": float(
                np.mean([value == (label,) for value in selected])
            ),
            "mean_prediction_set_size": float(
                np.mean([len(value) for value in selected])
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/calibration/t01_open_and_observe_effect_v3.jsonl",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT
        / "outputs/t01_open_and_observe_effect_v3/pi05_temporal_embeddings_v5.npz",
    )
    parser.add_argument("--block", choices=("global", "spatial", "cognitive", "spatial_cognitive", "all"), default="cognitive")
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/calibration/t01_temporal_target_evidence_v1_candidate.json",
    )
    args = parser.parse_args()
    for name in ("dataset", "embeddings", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output.exists():
        raise FileExistsError(f"candidate artifact exists: {args.output}")

    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line]
    data = np.load(args.embeddings)
    history = np.asarray(data["history_features"], dtype=np.float64)
    seeds = np.asarray(data["seed"], dtype=np.int64)
    if history.shape != (len(rows), 6, 21504):
        raise ValueError("expected aligned [N,6,21504] frozen history")
    slices = {
        "global": (0, 8192),
        "spatial": (8192, 17408),
        "cognitive": (17408, 21504),
        "spatial_cognitive": (8192, 21504),
        "all": (0, 21504),
    }
    start, end = slices[args.block]
    selected = history[:, :, start:end]
    visible = frame_truth(rows)
    episode_truth = np.where(np.any(visible, axis=1), "REVEALED", "NOT_REVEALED")
    train = (seeds >= 600) & (seeds <= 619)
    calibration = (seeds >= 620) & (seeds <= 652)
    development = (seeds >= 653) & (seeds <= 659)

    train_frames = selected[train].reshape(-1, end - start)
    train_frame_labels = np.where(
        visible[train].reshape(-1), "VISIBLE", "NOT_VISIBLE"
    )
    frame_standardizer = FeatureStandardizer.fit(train_frames)
    frame_probe = BalancedRidgeBinary.fit(
        frame_standardizer.transform(train_frames),
        train_frame_labels,
        negative_label="NOT_VISIBLE",
        positive_label="VISIBLE",
        regularization=args.regularization,
    )
    summaries = episode_values(frame_probe, frame_standardizer, selected)
    episode_standardizer = FeatureStandardizer.fit(summaries[train])
    scaled = episode_standardizer.transform(summaries)
    episode_critic = BalancedRidgeMulticlass.fit(
        scaled[train], episode_truth[train], regularization=args.regularization
    )
    episode_conformal = MondrianSemanticConformalCalibrator.fit(
        list(
            zip(
                episode_critic.evidence(scaled[calibration]),
                episode_truth[calibration],
                strict=True,
            )
        ),
        alpha=args.alpha,
        policy_id="pi05_libero_temporal_target_evidence_v1",
        split_id="t01_target_evidence_seed620_652",
    )
    predictor = TemporalTargetEvidencePredictor(
        input_dimension=history.shape[2],
        feature_start=start,
        feature_end=end,
        frame_standardizer=frame_standardizer,
        frame_probe=frame_probe,
        episode_standardizer=episode_standardizer,
        episode_critic=episode_critic,
        episode_conformal=episode_conformal,
    )
    development_predictions = [
        predictor.predict_history(history[index]).prediction_set
        for index in np.flatnonzero(development)
    ]
    artifact = {
        "schema_version": "interactive-perception.temporal-target-evidence.v1-candidate",
        "claim": "prompt-resolvable target evidence at any of six public RGB history points",
        "dataset": str(args.dataset.relative_to(ROOT)),
        "dataset_sha256": digest(args.dataset),
        "embeddings": str(args.embeddings.relative_to(ROOT)),
        "embeddings_sha256": digest(args.embeddings),
        "input_feature_dimension": int(history.shape[2]),
        "frame_feature_block": args.block,
        "frame_feature_slice": {"start": start, "end": end},
        "frame_standardizer": frame_standardizer.to_dict(),
        "frame_probe": frame_probe.to_dict(),
        "episode_score_summary": [
            "six ordered frame scores",
            "maximum",
            "second maximum",
            "mean",
            "standard deviation",
            "range",
        ],
        "episode_standardizer": episode_standardizer.to_dict(),
        "episode_critic": episode_critic.to_dict(),
        "episode_conformal": episode_conformal.to_dict(),
        "minimum_resolvable_target_pixels": PI05_PATCH_EQUIVALENT_TARGET_PIXELS,
        "offline_label_inputs": ["evaluator-only target segmentation pixel counts"],
        "online_inputs": [
            "six stock agentview RGB frames",
            "six stock wrist RGB frames",
            "query prompt",
        ],
        "online_oracle_inputs": [],
        "development_diagnostic": {
            "metrics": metrics(
                episode_truth[development], development_predictions
            ),
            "rows": [
                {
                    "regime": rows[index]["regime"],
                    "seed": int(rows[index]["seed"]),
                    "truth": str(episode_truth[index]),
                    "prediction_set": list(prediction),
                }
                for index, prediction in zip(
                    np.flatnonzero(development),
                    development_predictions,
                    strict=True,
                )
            ],
            "claim_eligible": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"artifact": str(args.output), **artifact["development_diagnostic"]}, indent=2))


if __name__ == "__main__":
    main()
