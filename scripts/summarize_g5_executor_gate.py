#!/usr/bin/env python3
"""Combine frozen capability results into the G5 executor-authorization gate."""

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
    parser.add_argument("--required-reliability", type=float, default=0.9)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--remove-result",
        type=Path,
        default=ROOT / "results/capability/t01_conformal_reveal_100seed_v1.json",
    )
    parser.add_argument("--remove-task", default="T01C_hidden_butter_basket")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    act_path = ROOT / "results/repro_libero_object.json"
    remove_path = args.remove_result
    unit_path = ROOT / "results/capability/stock_aligned_unit_actions_30episode.json"
    act = json.loads(act_path.read_text())
    remove_report = json.loads(remove_path.read_text())
    if "reveal_gate" in remove_report:
        remove_counts = (
            int(remove_report["reveal_gate"]["successes"]),
            int(remove_report["reveal_gate"]["trials"]),
        )
    else:
        remove = remove_report["tasks"][args.remove_task]
        remove_counts = (int(remove["successes"]), int(remove["episodes"]))
    units = json.loads(unit_path.read_text())["actions"]
    counts = {
        "ACT": (round(float(act["success_rate"]) * int(act["episodes"])), int(act["episodes"])),
        "REMOVE_OCCLUDER": remove_counts,
        "MOVE_CLOSER": (int(units["MOVE_CLOSER"]["successes"]), int(units["MOVE_CLOSER"]["trials"])),
        "ROTATE": (int(units["ROTATE"]["successes"]), int(units["ROTATE"]["trials"])),
    }
    gates = {
        label: CapabilityGate(
            successes=successes,
            trials=trials,
            confidence=args.confidence,
            required_reliability=args.required_reliability,
        ).to_dict()
        for label, (successes, trials) in counts.items()
    }
    allowed = [label for label, gate in gates.items() if gate["passed"]]
    report = {
        "schema_version": "interactive-perception.g5-executor-gate.v1",
        "required_reliability": args.required_reliability,
        "confidence": args.confidence,
        "gates": gates,
        "authorized_primitives": allowed,
        "blocked_primitives": [label for label in gates if label not in allowed],
        "g5_full_passed": all(gate["passed"] for gate in gates.values()),
        "router_rule": "A conformal singleton is executable only if its matching G5 gate passes.",
        "scope_note": (
            "REMOVE_OCCLUDER is authorized only for the stock-aligned T01 drawer context; "
            "the full retrieval chain is a separate gate."
        ),
        "source_sha256": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (act_path, remove_path, unit_path)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
