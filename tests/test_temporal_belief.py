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
from interactive_perception.temporal_belief import (
    TemporalActionRole,
    TemporalBelief,
    TemporalPhase,
)


def test_generic_automaton_has_no_scene_specific_action_names() -> None:
    names = {item.value for item in TemporalActionRole}
    assert names == {"INFORMATION", "COMMIT", "NOT_FOUND", "OBSERVE"}


def test_information_effect_advances_or_reopens_search() -> None:
    needs = TemporalBelief.point(TemporalPhase.NEEDS_EVIDENCE)
    revealed = needs.resolve_information("REVEALED", search_exhausted=False)
    failed = needs.resolve_information("FAILED", search_exhausted=False)
    empty = needs.resolve_information("EMPTY", search_exhausted=True)
    assert revealed.mass(TemporalPhase.READY_TO_COMMIT) == pytest.approx(1.0)
    assert failed.mass(TemporalPhase.NEEDS_EVIDENCE) == pytest.approx(1.0)
    assert empty.mass(TemporalPhase.SEARCH_EXHAUSTED) == pytest.approx(1.0)


def test_uncertain_progress_produces_a_probability_of_violation() -> None:
    progress = TemporalBelief.from_mapping(
        {
            TemporalPhase.NEEDS_EVIDENCE: 0.7,
            TemporalPhase.READY_TO_COMMIT: 0.3,
        }
    )
    assert progress.violation_probability(TemporalActionRole.COMMIT) == pytest.approx(
        0.7
    )
    assert progress.violation_probability(
        TemporalActionRole.INFORMATION
    ) == pytest.approx(0.3)


def _planner() -> ExpectedRiskPlanner:
    return ExpectedRiskPlanner(
        losses=DecisionLosses(1.0, 1.0, 1.0),
        act_reliability=1.0,
        horizon=1,
        temporal_violation_loss=2.0,
    )


def _planner_with_stop() -> ExpectedRiskPlanner:
    return ExpectedRiskPlanner(
        losses=DecisionLosses(1.0, 1.0, 1.0, safe_stop=0.2),
        act_reliability=1.0,
        horizon=1,
        temporal_violation_loss=2.0,
    )


def test_product_belief_blocks_premature_commit() -> None:
    # The world head alone is overconfident that the target is observed.  The
    # independent progress belief still says evidence acquisition is required.
    belief = TargetBelief(
        (TargetHypothesis("visible", TargetState.OBSERVED),),
        (1.0,),
    )
    effect = ActionEffect("INSPECT", ("visible",), 1.0, 0.1, "test")
    decision = _planner_with_stop().plan(
        belief,
        (effect,),
        temporal_belief=TemporalBelief.point(TemporalPhase.NEEDS_EVIDENCE),
    )
    assert decision.selected_action == "INSPECT"
    by_action = {value.action: value for value in decision.ranking}
    assert by_action[ACT].temporal_violation_probability == pytest.approx(1.0)


def test_not_found_requires_search_exhaustion() -> None:
    belief = TargetBelief(
        (TargetHypothesis("absent", TargetState.ABSENT),),
        (1.0,),
    )
    premature = _planner_with_stop().plan(
        belief,
        (),
        temporal_belief=TemporalBelief.point(TemporalPhase.NEEDS_EVIDENCE),
    )
    justified = _planner_with_stop().plan(
        belief,
        (),
        temporal_belief=TemporalBelief.point(TemporalPhase.SEARCH_EXHAUSTED),
    )
    assert premature.selected_action == SAFE_STOP
    assert justified.selected_action == NOT_FOUND


def test_positive_temporal_loss_requires_progress_belief() -> None:
    belief = TargetBelief(
        (TargetHypothesis("visible", TargetState.OBSERVED),),
        (1.0,),
    )
    with pytest.raises(ValueError, match="temporal_belief"):
        _planner().plan(belief, ())
