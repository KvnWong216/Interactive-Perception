import pytest

from interactive_perception.action_effect import EffectRegistry, EffectTrial
from interactive_perception.active_risk import EffectOutcome


def trial(index: int, outcome: EffectOutcome, context: str = "drawer-a") -> EffectTrial:
    return EffectTrial(context, "OPEN_MIDDLE", outcome, f"seed-{index}")


def test_effect_registry_uses_conservative_reliability() -> None:
    rows = [trial(index, EffectOutcome.REVEALED) for index in range(97)]
    rows += [trial(97 + index, EffectOutcome.FAILED) for index in range(3)]
    entry = EffectRegistry.fit(
        rows, confidence=0.95, required_reliability=0.9
    ).get("drawer-a", "OPEN_MIDDLE")
    assert entry.empirical_probability(EffectOutcome.REVEALED) == pytest.approx(0.97)
    assert entry.lower_bound(EffectOutcome.REVEALED) == pytest.approx(
        0.92428920625017
    )
    assert entry.passes(EffectOutcome.REVEALED)
    effect = entry.planner_effect(resolves=("middle",), cost=0.1)
    assert effect.reliability == pytest.approx(0.92428920625017)
    restored = EffectRegistry.from_dict(
        EffectRegistry.fit(
            rows, confidence=0.95, required_reliability=0.9
        ).to_dict()
    )
    assert restored.get("drawer-a", "OPEN_MIDDLE").to_dict() == entry.to_dict()


def test_effect_registry_refuses_context_fallback() -> None:
    registry = EffectRegistry.fit(
        [trial(0, EffectOutcome.REVEALED)],
        confidence=0.95,
        required_reliability=0.9,
    )
    with pytest.raises(KeyError, match="no exact calibrated effect"):
        registry.get("drawer-b", "OPEN_MIDDLE")


def test_effect_registry_requires_unique_trial_ids() -> None:
    duplicate = trial(0, EffectOutcome.REVEALED)
    with pytest.raises(ValueError, match="source_ids must be unique"):
        EffectRegistry.fit(
            [duplicate, duplicate], confidence=0.95, required_reliability=0.9
        )
