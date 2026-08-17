#!/usr/bin/env python3
"""Freeze observable OPEN_AND_OBSERVE branches and task-conditional effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interactive_perception.action_effect import EffectRegistry, EffectTrial  # noqa: E402
from interactive_perception.active_risk import EffectOutcome  # noqa: E402
from interactive_perception.action_outcome import (  # noqa: E402
    label_observable_information_outcome,
)
from interactive_perception.capability_gate import exact_binomial_lower_bound  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observable_outcome(row: dict) -> EffectOutcome:
    return label_observable_information_outcome(
        full_executor=bool(row["full_executor"]),
        opened=bool(row["evaluator_only"]["drawer_opened"]),
        return_complete=row["return_status"]["phase"] == "COMPLETE",
        target_pixels=row["evaluator_only"]["after_target_pixels"].values(),
        minimum_target_pixels=int(
            row["evaluator_only"]["minimum_revealed_target_pixels"]
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/calibration/t01_open_and_observe_effect_v1.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/calibration/t01_open_and_observe_effect_v1.manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/calibration/t01_open_and_observe_effect_v2.json",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--minimum-action-reliability", type=float, default=0.8)
    args = parser.parse_args()
    for name in ("dataset", "manifest", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    manifest = json.loads(args.manifest.read_text())
    if manifest["dataset_sha256"] != digest(args.dataset):
        raise ValueError("development dataset hash differs from its manifest")
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line]
    full = [row for row in rows if row["full_executor"]]
    registry = EffectRegistry.fit(
        [
            EffectTrial(
                "t01_stock_middle_drawer_search",
                "OPEN_AND_OBSERVE",
                observable_outcome(row),
                f"{row['regime']}-seed-{row['seed']}",
            )
            for row in full
        ],
        confidence=args.confidence,
        required_reliability=args.minimum_action_reliability,
    )
    branch = {}
    for intended in (EffectOutcome.REVEALED, EffectOutcome.EMPTY):
        selected = [row for row in full if row["intended_outcome"] == intended.value]
        successes = sum(observable_outcome(row) is intended for row in selected)
        lower = exact_binomial_lower_bound(successes, len(selected), args.confidence)
        branch[intended.value] = {
            "successes": successes,
            "trials": len(selected),
            "one_sided_lower_bound": lower,
            "passes_0.80": lower >= args.minimum_action_reliability,
            "passes_original_0.90": lower >= 0.9,
        }
    artifact = registry.to_dict()
    entry = registry.get("t01_stock_middle_drawer_search", "OPEN_AND_OBSERVE")
    artifact.update(
        {
            "schema_version": "interactive-perception.observed-effect-registry.v2",
            "source": str(args.dataset.relative_to(ROOT)),
            "source_sha256": digest(args.dataset),
            "manifest_sha256": digest(args.manifest),
            "label_contract": {
                "FAILED": "no certified observation of the searched region",
                "REVEALED": "target visibly acquired",
                "EMPTY": "searched region observed after opening with no target evidence",
                "hidden_initial_target_location_used": False,
            },
            "task_conditional_branch_reliability": branch,
            "information_completion_lower_bound": entry.information_completion_lower_bound(),
            "physical_effect_gate_passed": all(
                row["passes_0.80"] for row in branch.values()
            ),
            "online_oracle_inputs": [],
            "development_limitation": "v1 rows saved minimum rather than final drawer joint; v7 corrections were visually inspected and audit rows save both",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"frozen v2 effect artifact exists: {args.output}")
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"artifact": str(args.output), "branches": branch, "completion_lower": artifact["information_completion_lower_bound"]}, indent=2))
    if not artifact["physical_effect_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
