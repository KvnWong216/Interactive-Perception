#!/usr/bin/env python3
"""Freeze temporal OPEN_AND_OBSERVE effects without final-frame label loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interactive_perception.action_effect import EffectRegistry, EffectTrial  # noqa: E402
from interactive_perception.action_outcome import (  # noqa: E402
    label_temporal_information_outcome,
)
from interactive_perception.active_risk import EffectOutcome  # noqa: E402
from interactive_perception.capability_gate import exact_binomial_lower_bound  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def temporal_outcome(row: dict) -> EffectOutcome:
    evaluator = row["evaluator_only"]
    return label_temporal_information_outcome(
        full_executor=bool(row["full_executor"]),
        opened=bool(evaluator["drawer_opened"]),
        return_complete=row["return_status"]["phase"] == "COMPLETE",
        target_pixel_history=tuple(
            tuple(point["target_pixels"].values())
            for point in evaluator["visibility_history"]
        ),
        minimum_target_pixels=int(evaluator["minimum_revealed_target_pixels"]),
        empty_coverage_certified=bool(
            evaluator["empty_counterfactual_reveal_certified"]
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/calibration/t01_open_and_observe_effect_v3.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "data/calibration/t01_open_and_observe_effect_v3.manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/calibration/t01_open_and_observe_effect_v3.json",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--minimum-action-reliability", type=float, default=0.8)
    args = parser.parse_args()
    for name in ("dataset", "manifest", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema_version") != "interactive-perception.open-and-observe-manifest.v3":
        raise ValueError("temporal effect freeze requires a v3 manifest")
    if manifest["dataset_sha256"] != digest(args.dataset):
        raise ValueError("development dataset hash differs from its manifest")
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line]
    public_keys = {
        "name",
        "option_phase",
        "step",
        "image_paths",
        "image_sha256",
        "robot_state",
        "action_role",
    }
    for row in rows:
        if len(row.get("public_history", ())) != 6 or any(
            set(point) != public_keys for point in row["public_history"]
        ):
            raise ValueError("public history violates the frozen oracle-free schema")
        if len(row.get("evaluator_only", {}).get("visibility_history", ())) != 6:
            raise ValueError("evaluator visibility history must align to six points")
    full = [row for row in rows if row["full_executor"]]
    registry = EffectRegistry.fit(
        [
            EffectTrial(
                "t01_stock_middle_drawer_search",
                "OPEN_AND_OBSERVE",
                temporal_outcome(row),
                f"{row['regime']}-seed-{row['seed']}",
            )
            for row in full
        ],
        confidence=args.confidence,
        required_reliability=args.minimum_action_reliability,
    )
    branches = {}
    for intended in (EffectOutcome.REVEALED, EffectOutcome.EMPTY):
        selected = [row for row in full if row["intended_outcome"] == intended.value]
        successes = sum(temporal_outcome(row) is intended for row in selected)
        lower = exact_binomial_lower_bound(successes, len(selected), args.confidence)
        branches[intended.value] = {
            "successes": successes,
            "trials": len(selected),
            "one_sided_lower_bound": lower,
            "passes_0.80": lower >= args.minimum_action_reliability,
            "passes_original_0.90": lower >= 0.9,
        }
    entry = registry.get("t01_stock_middle_drawer_search", "OPEN_AND_OBSERVE")
    artifact = registry.to_dict()
    artifact.update(
        {
            "schema_version": "interactive-perception.temporal-effect-registry.v3",
            "source": str(args.dataset.relative_to(ROOT)),
            "source_sha256": digest(args.dataset),
            "manifest_sha256": digest(args.manifest),
            "label_contract": {
                "REVEALED": "target visible at any of six public history points",
                "EMPTY": "never visible; opened; return complete; visibility coverage certified",
                "FAILED": "neither prompt-relevant information result was certified",
                "final_frame_only": False,
                "hidden_initial_target_location_online": False,
            },
            "task_conditional_branch_reliability": branches,
            "information_completion_lower_bound": (
                entry.information_completion_lower_bound()
            ),
            "physical_effect_gate_passed": all(
                row["passes_0.80"] for row in branches.values()
            ),
            "online_oracle_inputs": [],
            "calibration_limitation": (
                "EMPTY coverage uses same-camera-pose counterfactual rendering "
                "of a seed-matched target trajectory; paper-scale work should "
                "also report depth-based visible-volume coverage"
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"frozen v3 effect artifact exists: {args.output}")
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"branches": branches, "output": str(args.output)}, indent=2))
    if not artifact["physical_effect_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
