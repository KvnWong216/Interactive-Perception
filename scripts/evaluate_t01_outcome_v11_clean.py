#!/usr/bin/env python3
"""Evaluate the frozen all-public-RGB v11 cascade on fresh T01 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from evaluate_t01_outcome_v10_clean import (  # noqa: E402
    class_metrics,
    physical_branch,
    v10_truth,
)
from train_t01_rgb_target_evidence import (  # noqa: E402
    binary_evidence,
    build_model,
    episode_score,
    examples,
    predict_examples,
)
from interactive_perception.action_outcome import (  # noqa: E402
    PI05_PATCH_EQUIVALENT_TARGET_PIXELS,
)
from interactive_perception.semantic_conformal import (  # noqa: E402
    MondrianSemanticConformalCalibrator,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(value: dict) -> Path:
    path = ROOT / value["path"]
    if digest(path) != value["sha256"]:
        raise ValueError(f"frozen v11 dependency changed: {path}")
    return path


def load_head(artifact_path, checkpoint_path, rows, camera, evidence_kind, torch, Image):
    artifact = json.loads(artifact_path.read_text())
    calibrator = MondrianSemanticConformalCalibrator.from_dict(
        artifact["conformal"]
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(torch)
    model.load_state_dict(checkpoint["state_dict"])
    scores = predict_examples(
        model,
        examples(rows, (camera,), evidence_kind),
        device=torch.device("cpu"),
        size=int(artifact["model"]["image_size"]),
        batch_size=128,
        torch=torch,
        Image=Image,
    )
    return calibrator, scores


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
        default=ROOT
        / "results/calibration/t01_open_and_observe_clean_development_v11.json",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    for name in ("dataset", "composite", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch
    from PIL import Image

    composite = json.loads(args.composite.read_text())
    composite_sha = digest(args.composite)
    dependencies = {
        name: resolve(value)
        for name, value in composite["dependencies"].items()
    }
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line]
    if len(rows) != 120 or sorted(set(int(row["seed"]) for row in rows)) != list(
        range(1440, 1480)
    ):
        raise ValueError("v11 clean data must contain seeds 1440-1479 x 3 regimes")
    for row in rows:
        if row["frozen_critic_sha256"] != composite_sha:
            raise ValueError("row was not collected under the frozen v11 composite")
        if row["online_oracle_inputs"]:
            raise ValueError("row declares a privileged controller input")

    agent_cal, agent_scores = load_head(
        dependencies["agentview_target_artifact"],
        dependencies["agentview_target_checkpoint"],
        rows,
        "agentview",
        "target",
        torch,
        Image,
    )
    wrist_cal, wrist_scores = load_head(
        dependencies["wrist_target_artifact"],
        dependencies["wrist_target_checkpoint"],
        rows,
        "wrist",
        "target",
        torch,
        Image,
    )
    coverage_cal, coverage_scores = load_head(
        dependencies["coverage_artifact"],
        dependencies["coverage_checkpoint"],
        rows,
        "agentview",
        "coverage",
        torch,
        Image,
    )

    truth = []
    predictions = []
    records = []
    wrist_rescues = 0
    for row in rows:
        agent_logit = episode_score(row, agent_scores, ("agentview",))
        wrist_logit = episode_score(row, wrist_scores, ("wrist",))
        coverage_logit = episode_score(row, coverage_scores, ("agentview",))
        agent_set = agent_cal.predict(binary_evidence(agent_logit))
        wrist_set = wrist_cal.predict(binary_evidence(wrist_logit))
        coverage_set = coverage_cal.predict(
            binary_evidence(coverage_logit, "COMPLETED", "FAILED")
        )
        if agent_set == ("REVEALED",):
            target_set = ("REVEALED",)
            target_source = "agentview"
        elif wrist_set == ("REVEALED",):
            target_set = ("REVEALED",)
            target_source = "wrist_positive_rescue"
            wrist_rescues += 1
        elif agent_set == ("NOT_REVEALED",):
            target_set = ("NOT_REVEALED",)
            target_source = "agentview_negative"
        else:
            target_set = ("REVEALED", "NOT_REVEALED")
            target_source = "ambiguous"
        if target_set == ("REVEALED",):
            outcome = ("REVEALED",)
        elif target_set == ("NOT_REVEALED",):
            values = []
            if "FAILED" in coverage_set:
                values.append("FAILED")
            if "COMPLETED" in coverage_set:
                values.append("EMPTY")
            outcome = tuple(values)
        else:
            outcome = ("FAILED", "REVEALED", "EMPTY")
        observed = v10_truth(row)
        truth.append(observed)
        predictions.append(outcome)
        records.append(
            {
                "regime": row["regime"],
                "seed": int(row["seed"]),
                "truth": observed,
                "agentview_target_set": list(agent_set),
                "wrist_target_set": list(wrist_set),
                "target_source": target_source,
                "coverage_set": list(coverage_set),
                "outcome_prediction_set": list(outcome),
            }
        )

    per_class = class_metrics(truth, predictions, confidence=args.confidence)
    false_empty = sum(
        prediction == ("EMPTY",) and observed != "EMPTY"
        for prediction, observed in zip(predictions, truth, strict=True)
    )
    false_revealed = sum(
        prediction == ("REVEALED",) and observed != "REVEALED"
        for prediction, observed in zip(predictions, truth, strict=True)
    )
    physical = {
        "REVEALED": physical_branch(
            rows, truth, "revealed_full", "REVEALED", args.confidence
        ),
        "EMPTY": physical_branch(
            rows, truth, "empty_full", "EMPTY", args.confidence
        ),
    }
    revealed_rows = [row for row in rows if row["regime"] == "revealed_full"]
    six_frame = sum(v10_truth(row) == "REVEALED" for row in revealed_rows)
    final_frame = sum(
        max(row["evaluator_only"]["visibility_history"][-1]["target_pixels"].values())
        >= PI05_PATCH_EQUIVALENT_TARGET_PIXELS
        for row in revealed_rows
    )
    gates = {
        "correct_label_retention": all(value["coverage"] >= 0.90 for value in per_class.values()),
        "singleton_lower_0.80": all(
            value["singleton_one_sided_95_lower"] >= 0.80
            for value in per_class.values()
        ),
        "zero_false_singleton_EMPTY": false_empty == 0,
        "zero_false_singleton_REVEALED": false_revealed == 0,
        "physical_information_lower_0.80": all(value["passes_0.80"] for value in physical.values()),
        "privileged_online_read_count_zero": True,
    }
    passed = all(gates.values())
    result = {
        "schema_version": "interactive-perception.t01-v11-clean-development.v1",
        "decision": "GO" if passed else "NOT_GO",
        "sealed_audit_authorized": passed,
        "dataset": {"path": str(args.dataset.relative_to(ROOT)), "sha256": digest(args.dataset)},
        "frozen_composite": {"path": str(args.composite.relative_to(ROOT)), "sha256": composite_sha},
        "samples": len(rows),
        "seeds": "1440-1479",
        "per_class": per_class,
        "false_singleton_EMPTY": false_empty,
        "false_singleton_REVEALED": false_revealed,
        "physical_information_acquisition": physical,
        "history_ablation": {
            "six_frame_resolvable_reveals": six_frame,
            "final_frame_only_resolvable_reveals": final_frame,
            "trials": len(revealed_rows),
        },
        "wrist_positive_rescues": wrist_rescues,
        "privileged_input_audit": {
            "controller_online_oracle_read_count": 0,
            "critic_online_oracle_inputs": [],
            "evaluator_timing": "separate replay after controller termination",
        },
        "gates": gates,
        "rows": records,
        "interpretation": {
            "proves": "fresh T01 public-RGB OPEN_AND_OBSERVE outcome recognition for the target-observability endpoint",
            "does_not_prove": [
                "final butter placement success",
                "global NOT_FOUND",
                "cross-scene or cross-object generalization",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("decision", "per_class", "physical_information_acquisition", "history_ablation", "wrist_positive_rescues", "gates")}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
