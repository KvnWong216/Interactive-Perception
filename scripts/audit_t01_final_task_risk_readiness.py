#!/usr/bin/env python3
"""Audit whether T01 has the context-specific inputs needed for final-task risk."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interactive_perception.capability_gate import CapabilityGate  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--open-drawer-act",
        type=Path,
        default=ROOT / "results/capability/t01_open_drawer_retrieval_screen_5seed.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/t01_final_task_risk_readiness_v1.json",
    )
    args = parser.parse_args()
    for name in ("open_drawer_act", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)

    source = json.loads(args.open_drawer_act.read_text())
    task = source["tasks"]["T01E_open_drawer_retrieval"]
    butter_gate = CapabilityGate(
        successes=int(task["successes"]),
        trials=int(task["episodes"]),
        confidence=0.95,
        required_reliability=0.9,
    ).to_dict()
    contexts = {
        "open_visible_butter_to_basket": {
            "status": "MEASURED_NOT_GO",
            "gate": butter_gate,
            "source": str(args.open_drawer_act.relative_to(ROOT)),
        },
        "closed_visible_cream_cheese_to_basket": {
            "status": "MISSING",
            "gate": None,
            "reason": "the prompt/state audit did not execute ACT",
        },
    }
    report = {
        "schema_version": "interactive-perception.final-task-risk-readiness.v1",
        "claim": "strict availability audit for context-specific terminal ACT effects",
        "contexts": contexts,
        "context_fallback_allowed": False,
        "stock_act_100_of_100_transfer_allowed": False,
        "reason": (
            "stock LIBERO object success cannot authorize retrieval from an open drawer; "
            "the measured open-drawer context is 0/5"
        ),
        "final_task_risk_ready": False,
        "source_sha256": {str(args.open_drawer_act.relative_to(ROOT)): digest(args.open_drawer_act)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
