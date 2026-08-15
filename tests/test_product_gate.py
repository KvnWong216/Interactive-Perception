from pathlib import Path

import pytest
import yaml

from interactive_perception.product_gate import summarize_product_gates


ROOT = Path(__file__).parents[1]


def test_actual_final_product_registry_is_strict_not_go() -> None:
    spec = yaml.safe_load(
        (ROOT / "benchmarks/final_product_v1/gates.yaml").read_text()
    )
    summary = summarize_product_gates(spec)
    assert summary["milestones"]["t01_reveal_prototype"] == "GO"
    assert summary["final_product_go"] is False
    assert {"FP2", "FP3", "FP4", "FP8"} <= set(summary["blocking_failures"])


def test_partial_is_a_blocking_failure() -> None:
    spec = {
        "gates": [
            {
                "id": "A",
                "status": "PARTIAL",
                "blocking": True,
                "pass_condition": "complete it",
                "current": "partial",
            }
        ]
    }
    assert summarize_product_gates(spec)["final_product_go"] is False


def test_invalid_or_duplicate_gates_are_rejected() -> None:
    base = {
        "id": "A",
        "status": "GO",
        "pass_condition": "pass",
        "current": "done",
    }
    with pytest.raises(ValueError, match="unique"):
        summarize_product_gates({"gates": [base, base]})
    with pytest.raises(ValueError, match="invalid"):
        summarize_product_gates({"gates": [{**base, "status": "MAYBE"}]})
