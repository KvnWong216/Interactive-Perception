import pytest

from interactive_perception.capability_gate import (
    CapabilityGate,
    exact_binomial_lower_bound,
)


def test_zero_success_has_zero_lower_bound() -> None:
    assert exact_binomial_lower_bound(0, 5, 0.95) == 0.0


def test_debug_scale_result_cannot_certify_high_reliability() -> None:
    gate = CapabilityGate(3, 5, confidence=0.95, required_reliability=0.8)
    assert gate.empirical_rate == pytest.approx(0.6)
    assert gate.lower_bound < 0.8
    assert gate.passed is False


def test_large_reliable_sample_can_pass() -> None:
    gate = CapabilityGate(99, 100, confidence=0.95, required_reliability=0.9)
    assert gate.passed is True
