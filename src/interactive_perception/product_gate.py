"""Validation and summarization for the final-product gate registry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

VALID_STATUSES = frozenset({"GO", "PARTIAL", "NOT_GO"})


def summarize_product_gates(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return a strict release decision; PARTIAL never counts as passing."""

    gates = list(spec.get("gates", []))
    if not gates:
        raise ValueError("final-product gate spec must contain gates")
    ids = [str(gate.get("id", "")) for gate in gates]
    if any(not gate_id for gate_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("gate ids must be non-empty and unique")
    invalid = {
        str(gate.get("status"))
        for gate in gates
        if str(gate.get("status")) not in VALID_STATUSES
    }
    if invalid:
        raise ValueError(f"invalid gate statuses: {sorted(invalid)}")
    if any("pass_condition" not in gate or "current" not in gate for gate in gates):
        raise ValueError("every gate requires pass_condition and current")

    blocking = [gate for gate in gates if bool(gate.get("blocking", True))]
    blockers = [str(gate["id"]) for gate in blocking if gate["status"] != "GO"]
    counts = {
        status: sum(gate["status"] == status for gate in gates)
        for status in ("GO", "PARTIAL", "NOT_GO")
    }
    milestones = dict(spec.get("milestones", {}))
    return {
        "schema_version": "interactive-perception.final-product-gate-summary.v1",
        "name": str(spec.get("name", "")),
        "gate_count": len(gates),
        "status_counts": counts,
        "blocking_gate_ids": [str(gate["id"]) for gate in blocking],
        "blocking_failures": blockers,
        "final_product_go": not blockers,
        "milestones": {
            name: str(value.get("status")) for name, value in milestones.items()
        },
        "gates": gates,
    }
