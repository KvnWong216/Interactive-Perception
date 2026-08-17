from __future__ import annotations

import pytest

from interactive_perception.active_risk import (
    ACT,
    NOT_FOUND,
    SAFE_STOP,
    ActionEffect,
    DecisionLosses,
    ExpectedRiskPlanner,
    TargetBelief,
    TargetHypothesis,
    TargetState,
)
from interactive_perception.pipeline import PromptAlignedPipeline
from interactive_perception.temporal_belief import TemporalBelief, TemporalPhase


def _pipeline(*, probabilities=(0.0, 0.9, 0.1), attempts=1):
    belief = TargetBelief(
        hypotheses=(
            TargetHypothesis("visible", TargetState.OBSERVED),
            TargetHypothesis(
                "middle_layer", TargetState.MANIPULATION_ONLY, "OPEN_AND_OBSERVE"
            ),
            TargetHypothesis("absent", TargetState.ABSENT),
        ),
        probabilities=probabilities,
    )
    planner = ExpectedRiskPlanner(
        losses=DecisionLosses(1.0, 1.0, 1.0, safe_stop=0.8),
        act_reliability={"visible": 1.0, "middle_layer": 1.0, "absent": 1.0},
        horizon=1,
        temporal_violation_loss=2.0,
    )
    return PromptAlignedPipeline(
        planner=planner,
        belief=belief,
        conformal_labels=tuple(
            item.label
            for item, probability in zip(
                belief.hypotheses, belief.probabilities, strict=True
            )
            if probability > 0.0
        ),
        temporal_belief=TemporalBelief.point(TemporalPhase.NEEDS_EVIDENCE),
        effects=(
            ActionEffect(
                "OPEN_AND_OBSERVE",
                ("middle_layer",),
                0.95,
                0.05,
                "frozen development lower bound",
            ),
        ),
        maximum_attempts_per_action=attempts,
    )


def test_revealed_outcome_advances_to_act() -> None:
    pipeline = _pipeline(probabilities=(0.0, 1.0, 0.0))
    first = pipeline.plan()
    assert first.selected_action == "OPEN_AND_OBSERVE"
    pipeline.begin(first)
    update = pipeline.observe_information_outcome(("REVEALED",))
    assert not update.must_safe_stop
    assert update.temporal_belief.mass(TemporalPhase.READY_TO_COMMIT) == pytest.approx(1.0)
    second = pipeline.plan()
    assert second.selected_action == ACT


def test_empty_outcome_authorizes_not_found_only_after_exhaustion() -> None:
    pipeline = _pipeline(probabilities=(0.0, 0.8, 0.2))
    first = pipeline.plan()
    pipeline.begin(first)
    update = pipeline.observe_information_outcome(("EMPTY",))
    assert update.exhausted_search
    assert update.temporal_belief.mass(TemporalPhase.SEARCH_EXHAUSTED) == pytest.approx(1.0)
    assert pipeline.plan().selected_action == NOT_FOUND


def test_ambiguous_outcome_never_gets_silently_argmaxed() -> None:
    pipeline = _pipeline(probabilities=(0.0, 1.0, 0.0))
    decision = pipeline.plan()
    pipeline.begin(decision)
    update = pipeline.observe_information_outcome(("FAILED", "REVEALED"))
    assert update.must_safe_stop
    assert pipeline.terminal == SAFE_STOP
    assert pipeline.trace()["events"][-1]["kind"] == "AMBIGUOUS_OUTCOME_SAFE_STOP"


def test_failed_action_is_not_retried_without_a_registered_retry_budget() -> None:
    pipeline = _pipeline(probabilities=(0.0, 1.0, 0.0), attempts=1)
    decision = pipeline.plan()
    pipeline.begin(decision)
    pipeline.observe_information_outcome(("FAILED",))
    next_decision = pipeline.plan()
    assert next_decision.selected_action == SAFE_STOP


def test_trace_contains_no_oracle_channel() -> None:
    pipeline = _pipeline(probabilities=(0.0, 1.0, 0.0))
    decision = pipeline.plan()
    pipeline.begin(decision)
    trace = pipeline.trace()
    assert trace["online_oracle_inputs"] == []
    assert trace["pending_action"] == "OPEN_AND_OBSERVE"


def test_impossible_empty_observation_safe_stops_instead_of_inventing_absence() -> None:
    pipeline = _pipeline(probabilities=(0.0, 1.0, 0.0))
    # Remove the zero-mass ABSENT hypothesis entirely, matching the current
    # binary T01 belief artifact.
    pipeline.belief = TargetBelief(
        (
            TargetHypothesis(
                "middle_layer", TargetState.MANIPULATION_ONLY, "OPEN_AND_OBSERVE"
            ),
        ),
        (1.0,),
    )
    pipeline.conformal_labels = ("middle_layer",)
    decision = pipeline.plan()
    pipeline.begin(decision)
    update = pipeline.observe_information_outcome(("EMPTY",))
    assert update.must_safe_stop
    assert pipeline.terminal == SAFE_STOP
    assert pipeline.trace()["events"][-1]["kind"] == "BELIEF_CONTRADICTION_SAFE_STOP"
