#!/usr/bin/env python3
"""Evaluate frozen temporal/camera/coverage ablations for the T01 v11 critic."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from evaluate_t01_outcome_v10_clean import class_metrics, v10_truth  # noqa: E402
from evaluate_t01_outcome_v11_clean import load_head  # noqa: E402
from train_t01_rgb_target_evidence import (  # noqa: E402
    HISTORY_NAMES,
    binary_evidence,
    history_by_name,
)
from interactive_perception.rgb_outcome_critic import (  # noqa: E402
    resolve_v11_cascade,
    resolve_v12_cascade,
    resolve_v12b_cascade,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependency(composite: dict, name: str) -> Path:
    reference = composite["dependencies"][name]
    path = ROOT / reference["path"]
    if digest(path) != reference["sha256"]:
        raise ValueError(f"frozen v11 dependency changed: {path}")
    return path


def logits_for(row: dict, scores: dict, camera: str) -> tuple[float, ...]:
    public = history_by_name(row)
    return tuple(
        float(scores[(str(ROOT / public[name]["image_paths"][camera]), camera)])
        for name in HISTORY_NAMES
    )


def head_set(calibrator, logits, positive: str, negative: str, indices) -> tuple[str, ...]:
    value = max(logits[index] for index in indices)
    return tuple(calibrator.predict(binary_evidence(value, positive, negative)))


def no_wrist_cascade(agent_set, coverage_set) -> tuple[str, ...]:
    if agent_set == ("REVEALED",):
        return ("REVEALED",)
    if agent_set != ("NOT_REVEALED",):
        return ("FAILED", "REVEALED", "EMPTY")
    values = []
    if "FAILED" in coverage_set:
        values.append("FAILED")
    if "COMPLETED" in coverage_set:
        values.append("EMPTY")
    return tuple(values)


def no_coverage_cascade(agent_set, wrist_set) -> tuple[str, ...]:
    target, _ = resolve_v11_cascade(agent_set, wrist_set, ("FAILED", "COMPLETED"))
    return target


def summarize(truth, predictions, confidence: float) -> dict:
    per_class = class_metrics(truth, predictions, confidence=confidence)
    singleton_correct = sum(
        prediction == (observed,)
        for prediction, observed in zip(predictions, truth, strict=True)
    )
    return {
        "samples": len(truth),
        "singleton_correct": singleton_correct,
        "singleton_accuracy": singleton_correct / len(truth),
        "correct_label_retention": sum(
            observed in prediction
            for prediction, observed in zip(predictions, truth, strict=True)
        )
        / len(truth),
        "mean_prediction_set_size": sum(map(len, predictions)) / len(predictions),
        "ambiguous_safe_stop_rate": sum(len(value) != 1 for value in predictions)
        / len(predictions),
        "false_singleton_EMPTY": sum(
            prediction == ("EMPTY",) and observed != "EMPTY"
            for prediction, observed in zip(predictions, truth, strict=True)
        ),
        "false_singleton_REVEALED": sum(
            prediction == ("REVEALED",) and observed != "REVEALED"
            for prediction, observed in zip(predictions, truth, strict=True)
        ),
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/calibration/t01_open_and_observe_effect_v11_clean.jsonl",
    )
    parser.add_argument(
        "--composite",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_v11_composite_candidate.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/calibration/t01_open_and_observe_v11_ablations.json",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    for name in ("dataset", "composite", "output"):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, ROOT / path)
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch
    from PIL import Image

    composite = json.loads(args.composite.read_text())
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line]
    if len(rows) != 120 or sorted({int(row["seed"]) for row in rows}) != list(
        range(1440, 1480)
    ):
        raise ValueError("v11 ablations require the complete 1440-1479 clean block")
    composite_sha = digest(args.composite)
    if any(row["frozen_critic_sha256"] != composite_sha for row in rows):
        raise ValueError("dataset was not collected under this frozen v11 composite")
    if any(row["online_oracle_inputs"] for row in rows):
        raise ValueError("dataset declares privileged online inputs")

    agent_cal, agent_scores = load_head(
        dependency(composite, "agentview_target_artifact"),
        dependency(composite, "agentview_target_checkpoint"),
        rows,
        "agentview",
        "target",
        torch,
        Image,
    )
    wrist_cal, wrist_scores = load_head(
        dependency(composite, "wrist_target_artifact"),
        dependency(composite, "wrist_target_checkpoint"),
        rows,
        "wrist",
        "target",
        torch,
        Image,
    )
    coverage_cal, coverage_scores = load_head(
        dependency(composite, "coverage_artifact"),
        dependency(composite, "coverage_checkpoint"),
        rows,
        "agentview",
        "coverage",
        torch,
        Image,
    )

    arms = {
        "full_v11_six_frame": [],
        "conflict_safe_v12": [],
        "singleton_conflict_v12b": [],
        "no_wrist_rescue": [],
        "endpoint_only_before_after": [],
        "final_frame_only": [],
        "no_coverage_head": [],
        "trivial_all_labels": [],
    }
    truth = []
    records = []
    for row in rows:
        agent_logits = logits_for(row, agent_scores, "agentview")
        wrist_logits = logits_for(row, wrist_scores, "wrist")
        coverage_logits = logits_for(row, coverage_scores, "agentview")

        def sets(indices):
            return (
                head_set(
                    agent_cal,
                    agent_logits,
                    "REVEALED",
                    "NOT_REVEALED",
                    indices,
                ),
                head_set(
                    wrist_cal,
                    wrist_logits,
                    "REVEALED",
                    "NOT_REVEALED",
                    indices,
                ),
                head_set(
                    coverage_cal,
                    coverage_logits,
                    "COMPLETED",
                    "FAILED",
                    indices,
                ),
            )

        six = sets(range(6))
        endpoint = sets((0, 5))
        final = sets((5,))
        full, _ = resolve_v11_cascade(*six)
        arms["full_v11_six_frame"].append(full)
        arms["conflict_safe_v12"].append(resolve_v12_cascade(*six)[0])
        arms["singleton_conflict_v12b"].append(resolve_v12b_cascade(*six)[0])
        arms["no_wrist_rescue"].append(no_wrist_cascade(six[0], six[2]))
        arms["endpoint_only_before_after"].append(resolve_v11_cascade(*endpoint)[0])
        arms["final_frame_only"].append(resolve_v11_cascade(*final)[0])
        arms["no_coverage_head"].append(no_coverage_cascade(six[0], six[1]))
        arms["trivial_all_labels"].append(("FAILED", "REVEALED", "EMPTY"))
        observed = v10_truth(row)
        truth.append(observed)
        records.append(
            {
                "regime": row["regime"],
                "seed": int(row["seed"]),
                "truth": observed,
                "predictions": {name: list(values[-1]) for name, values in arms.items()},
            }
        )

    result = {
        "schema_version": "interactive-perception.t01-v11-ablations.v1",
        "claim_scope": "fresh T01 clean development; not sealed and not scene-disjoint",
        "dataset": {"path": str(args.dataset.relative_to(ROOT)), "sha256": digest(args.dataset)},
        "frozen_composite": {"path": str(args.composite.relative_to(ROOT)), "sha256": composite_sha},
        "online_oracle_inputs": [],
        "arms": {
            name: summarize(truth, predictions, args.confidence)
            for name, predictions in arms.items()
        },
        "records": records,
        "interpretation_rule": (
            "Correct-label retention alone is insufficient; singleton accuracy, "
            "prediction-set size, ambiguity, and false EMPTY/REVEALED are reported."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["arms"], indent=2))


if __name__ == "__main__":
    main()
