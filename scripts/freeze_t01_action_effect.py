#!/usr/bin/env python3
"""Freeze the measured T01 REMOVE_OCCLUDER effect without claiming a critic.

The 100-trial rollout contains physical evaluator labels but no saved after-RGB
for most episodes.  It can calibrate a context-specific effect probability; it
cannot pass the separate oracle-free visual outcome-critic gate.
"""

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


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "results/capability/t01_conformal_reveal_100seed_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/calibration/t01_action_effect_v1.json",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--required-reliability", type=float, default=0.9)
    args = parser.parse_args()
    for name in ("source", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)

    source = json.loads(args.source.read_text())
    trials = [
        EffectTrial(
            context="t01_stock_middle_drawer_hidden_butter",
            action="REMOVE_OCCLUDER",
            outcome=(
                EffectOutcome.REVEALED
                if bool(row["reveal_success"])
                else EffectOutcome.FAILED
            ),
            source_id=f"seed-{int(row['seed'])}",
        )
        for row in source["rows"]
    ]
    registry = EffectRegistry.fit(
        trials,
        confidence=args.confidence,
        required_reliability=args.required_reliability,
    )
    entry = registry.get(
        "t01_stock_middle_drawer_hidden_butter", "REMOVE_OCCLUDER"
    )
    artifact = {
        **registry.to_dict(),
        "source": str(args.source.relative_to(ROOT)),
        "source_sha256": digest(args.source),
        "label_source": "evaluator-only drawer joint during physical rollout",
        "online_oracle_inputs": [],
        "physical_reveal_reliability_gate_passed": entry.passes(
            EffectOutcome.REVEALED
        ),
        "visual_outcome_critic": {
            "implemented": False,
            "reason": (
                "the source did not save paired before/after policy RGB for all trials"
            ),
        },
        "fp3_action_effect_model_passed": False,
        "non_claim": (
            "no context generalization, EMPTY prediction, or online RGB outcome recognition"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
