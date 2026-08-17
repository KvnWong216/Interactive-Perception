#!/usr/bin/env python3
"""Freeze five-pixel physical effect rates from the audited RGB transitions."""

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
        "--dataset",
        type=Path,
        default=ROOT / "data/calibration/t01_action_effect_v1_audit.jsonl",
    )
    parser.add_argument(
        "--critic-audit",
        type=Path,
        default=ROOT / "results/calibration/t01_action_outcome_critic_audit_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/calibration/t01_action_effect_v2.json",
    )
    args = parser.parse_args()
    for name in ("dataset", "critic_audit", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output.exists():
        raise FileExistsError(f"effect artifact is immutable: {args.output}")
    critic_audit = json.loads(args.critic_audit.read_text())
    if not critic_audit["fp3_passed"]:
        raise ValueError("critic and full-executor audit must pass before v2 freeze")
    rows = [
        json.loads(line) for line in args.dataset.read_text().splitlines() if line
    ]
    trials = []
    context_by_regime = {
        "revealed_full": "t01_stock_middle_drawer_hidden_butter",
        "empty_full": "t01_stock_middle_drawer_target_elsewhere",
    }
    for row in rows:
        if not row["full_executor"]:
            continue
        regime = str(row["regime"])
        trials.append(
            EffectTrial(
                context=context_by_regime[regime],
                action="REMOVE_OCCLUDER",
                outcome=EffectOutcome(str(row["outcome"])),
                source_id=f"{regime}-seed-{int(row['seed'])}",
            )
        )
    registry = EffectRegistry.fit(
        trials, confidence=0.95, required_reliability=0.9
    )
    revealed = registry.get(
        "t01_stock_middle_drawer_hidden_butter", "REMOVE_OCCLUDER"
    )
    empty = registry.get(
        "t01_stock_middle_drawer_target_elsewhere", "REMOVE_OCCLUDER"
    )
    artifact = {
        **registry.to_dict(),
        "label": "drawer opened and at least five target pixels entered one policy view",
        "dataset": str(args.dataset.relative_to(ROOT)),
        "dataset_sha256": digest(args.dataset),
        "critic_audit": str(args.critic_audit.relative_to(ROOT)),
        "critic_audit_sha256": digest(args.critic_audit),
        "desired_effect_gates": {
            "hidden_target_REVEALED": revealed.to_dict()["outcomes"]["REVEALED"],
            "target_elsewhere_EMPTY": empty.to_dict()["outcomes"]["EMPTY"],
        },
        "physical_effect_gate_passed": (
            revealed.passes(EffectOutcome.REVEALED)
            and empty.passes(EffectOutcome.EMPTY)
        ),
        "online_oracle_inputs": [],
        "truncated_controls_in_effect_rates": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
