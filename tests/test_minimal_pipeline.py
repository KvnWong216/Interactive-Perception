from interactive_perception.active_risk import (
    ACT,
    SAFE_STOP,
    ActionEffect,
    DecisionLosses,
    ExpectedRiskPlanner,
    TargetBelief,
    TargetHypothesis,
    TargetState,
)
from interactive_perception.minimal_pipeline import (
    INFORMATION_ACQUIRED,
    MinimalPromptAlignedPipeline,
)


def _pipeline(*, attempts=1):
    belief = TargetBelief(
        (
            TargetHypothesis("OBSERVED", TargetState.OBSERVED),
            TargetHypothesis(
                "MANIPULATION_ONLY",
                TargetState.MANIPULATION_ONLY,
                "OPEN_AND_OBSERVE",
            ),
        ),
        (0.0, 1.0),
    )
    planner = ExpectedRiskPlanner(
        losses=DecisionLosses(1.0, 1.0, 1.0, safe_stop=0.8),
        act_reliability=1.0,
        horizon=1,
        temporal_violation_loss=0.0,
    )
    return MinimalPromptAlignedPipeline(
        planner=planner,
        belief=belief,
        conformal_labels=("MANIPULATION_ONLY",),
        effects=(
            ActionEffect(
                "OPEN_AND_OBSERVE",
                ("MANIPULATION_ONLY",),
                0.95,
                0.05,
                "development",
            ),
        ),
        maximum_attempts_per_action=attempts,
    )


def _observed_pipeline():
    pipeline = _pipeline()
    pipeline.belief = TargetBelief(
        (
            TargetHypothesis("OBSERVED", TargetState.OBSERVED),
            TargetHypothesis(
                "MANIPULATION_ONLY",
                TargetState.MANIPULATION_ONLY,
                "OPEN_AND_OBSERVE",
            ),
        ),
        (1.0, 0.0),
    )
    pipeline.conformal_labels = ("OBSERVED",)
    return pipeline


def test_revealed_updates_belief_then_replans_to_information_acquired() -> None:
    pipeline = _pipeline()
    first = pipeline.plan()
    assert first.selected_action == "OPEN_AND_OBSERVE"
    assert first.resolvable_uncertainty["OPEN_AND_OBSERVE"] > 0
    pipeline.begin(first)
    update = pipeline.observe_information_outcome(("REVEALED",))
    assert not update.must_safe_stop
    second = pipeline.plan()
    assert second.selected_action == ACT
    pipeline.begin(second)
    assert pipeline.terminal == INFORMATION_ACQUIRED
    assert not pipeline.trace()["probabilistic_temporal_progress_used"]


def test_initially_observed_goes_directly_to_act_without_opening() -> None:
    pipeline = _observed_pipeline()
    decision = pipeline.plan()
    assert decision.selected_action == ACT
    assert decision.resolvable_uncertainty == {}
    pipeline.begin(decision)
    assert pipeline.terminal == ACT
    assert pipeline.memory.attempt_counts["OPEN_AND_OBSERVE"] == 0


def test_empty_is_local_search_memory_and_never_global_not_found() -> None:
    pipeline = _pipeline()
    decision = pipeline.plan()
    pipeline.begin(decision)
    update = pipeline.observe_information_outcome(("EMPTY",))
    assert update.must_safe_stop
    assert pipeline.terminal == SAFE_STOP
    assert update.memory["searched_set"] == ["MANIPULATION_ONLY"]


def test_failed_preserves_belief_and_safe_stops_when_retry_exhausted() -> None:
    pipeline = _pipeline(attempts=1)
    before = pipeline.belief
    decision = pipeline.plan()
    pipeline.begin(decision)
    update = pipeline.observe_information_outcome(("FAILED",))
    assert update.must_safe_stop
    assert pipeline.belief == before
    assert pipeline.terminal == SAFE_STOP


def test_ambiguous_outcome_safe_stops_without_argmax() -> None:
    pipeline = _pipeline()
    decision = pipeline.plan()
    pipeline.begin(decision)
    update = pipeline.observe_information_outcome(("FAILED", "REVEALED"))
    assert update.must_safe_stop
    assert pipeline.terminal == SAFE_STOP
