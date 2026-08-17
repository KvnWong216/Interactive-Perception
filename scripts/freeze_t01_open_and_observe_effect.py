#!/usr/bin/env python3
"""Freeze physical OPEN_AND_OBSERVE effects from the development split."""

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
from interactive_perception.capability_gate import exact_binomial_lower_bound  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        default=ROOT / "results/calibration/t01_open_and_observe_effect_v1.json",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--required-reliability", type=float, default=0.8)
    args = parser.parse_args()
    for name in ("dataset", "manifest", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)

    manifest = json.loads(args.manifest.read_text())
    if manifest["dataset_sha256"] != digest(args.dataset):
        raise ValueError("dataset hash differs from its immutable manifest")
    if manifest.get("phase") != "development" or manifest.get("seeds") != list(
        range(600, 660)
    ):
        raise ValueError("effect freeze requires the exact development seed contract")
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line]
    expected = 3 * 60
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, got {len(rows)}")
    if any(len(row.get("public_history", ())) != 6 for row in rows):
        raise ValueError("every row must contain the frozen six-point public history")

    full = [row for row in rows if row["full_executor"]]
    trials = [
        EffectTrial(
            context="t01_stock_middle_drawer_search",
            action="OPEN_AND_OBSERVE",
            outcome=EffectOutcome(row["outcome"]),
            source_id=f"{row['regime']}-seed-{row['seed']}",
        )
        for row in full
    ]
    registry = EffectRegistry.fit(
        trials,
        confidence=args.confidence,
        required_reliability=args.required_reliability,
    )

    branch_results = {}
    for intended in (EffectOutcome.REVEALED, EffectOutcome.EMPTY):
        selected = [row for row in full if row["intended_outcome"] == intended.value]
        successes = sum(row["outcome"] == intended.value for row in selected)
        lower = exact_binomial_lower_bound(
            successes, len(selected), args.confidence
        )
        branch_results[intended.value] = {
            "successes": successes,
            "trials": len(selected),
            "empirical_rate": successes / len(selected),
            "one_sided_lower_bound": lower,
            "passes_required_reliability": lower >= args.required_reliability,
            "passes_original_0.90_reliability": lower >= 0.9,
        }

    artifact = registry.to_dict()
    entry = registry.get("t01_stock_middle_drawer_search", "OPEN_AND_OBSERVE")
    artifact.update(
        {
            "source": str(args.dataset.relative_to(ROOT)),
            "source_sha256": digest(args.dataset),
            "manifest": str(args.manifest.relative_to(ROOT)),
            "manifest_sha256": digest(args.manifest),
            "branch_reliability": branch_results,
            "information_completion_lower_bound": (
                entry.information_completion_lower_bound()
            ),
            "physical_effect_gate_passed": all(
                item["passes_required_reliability"]
                for item in branch_results.values()
            ),
            "online_oracle_inputs": [],
            "label_only_inputs": [
                "drawer joint",
                "target instance pixels",
                "frozen target location",
            ],
            "non_claim": "outcome-critic accuracy or held-out audit performance",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"frozen effect artifact already exists: {args.output}")
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "physical_effect_gate_passed": artifact[
                    "physical_effect_gate_passed"
                ],
                "branch_reliability": branch_results,
            },
            indent=2,
        )
    )
    if not artifact["physical_effect_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
