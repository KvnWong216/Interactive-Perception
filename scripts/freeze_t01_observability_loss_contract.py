#!/usr/bin/env python3
"""Freeze the preregistered T01 observability losses and cost sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "results/t01_expected_risk_router_v1.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/calibration/t01_observability_loss_contract_v1.json",
    )
    args = parser.parse_args()
    for name in ("source", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.output.exists():
        raise FileExistsError(f"loss contract is immutable: {args.output}")
    source = json.loads(args.source.read_text())
    if source["objective"] != "target observability; OBSERVED is a terminal success state":
        raise ValueError("source is not the frozen target-observability sweep")
    artifact = {
        "schema_version": "interactive-perception.observability-loss-contract.v1",
        "objective": source["objective"],
        "losses": {
            "false_commit": float(source["loss_contract"]["false_commit"]),
            "false_absent": float(source["loss_contract"]["false_absent"]),
            "act_execution_failure": float(
                source["loss_contract"]["act_execution_failure"]
            ),
        },
        "information_cost": float(source["declared_information_cost"]),
        "stable_passing_costs_on_preregistered_grid": source[
            "stable_passing_costs_on_grid"
        ],
        "selection_rule": (
            "use the declared unit-normalized cost; report the full frozen grid, "
            "never retune from closed-loop success"
        ),
        "non_claim": "final-task utility or physical cost measurement",
        "source": str(args.source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
