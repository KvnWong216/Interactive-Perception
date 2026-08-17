import pytest

from interactive_perception.active_risk import (
    ACT,
    NOT_FOUND,
    ActionEffect,
    DecisionLosses,
    EffectOutcome,
    ExpectedRiskPlanner,
    TargetBelief,
    TargetHypothesis,
    TargetState,
    update_from_effect,
)


def drawer_belief(probabilities=(0.1, 0.8, 0.1)):
    return TargetBelief(
        hypotheses=(
            TargetHypothesis("visible", TargetState.OBSERVED),
            TargetHypothesis(
                "middle_drawer",
                TargetState.MANIPULATION_ONLY,
                "REMOVE_OCCLUDER",
            ),
            TargetHypothesis("absent", TargetState.ABSENT),
        ),
        probabilities=probabilities,
    )


def drawer_effect(cost=0.05):
    return ActionEffect(
        action="REMOVE_OCCLUDER",
        resolves=("middle_drawer",),
        reliability=0.924,
        cost=cost,
        source="97/100 one-sided lower bound",
    )


def planner(horizon=1):
    return ExpectedRiskPlanner(
        losses=DecisionLosses(
            false_commit=1.0,
            false_absent=1.0,
            act_execution_failure=1.0,
        ),
        act_reliability=1.0,
        horizon=horizon,
    )


def test_information_action_wins_for_hidden_and_act_wins_for_visible():
    effect = (drawer_effect(),)
    hidden = drawer_belief((0.0, 1.0, 0.0))
    visible = drawer_belief((1.0, 0.0, 0.0))
    hidden_decision = planner().plan(hidden, effect)
    visible_decision = planner().plan(visible, effect)
    assert hidden_decision.selected_action == "REMOVE_OCCLUDER"
    assert hidden_decision.resolvable_uncertainty["REMOVE_OCCLUDER"] > 0.0
    assert visible_decision.selected_action == ACT
    assert visible_decision.resolvable_uncertainty["REMOVE_OCCLUDER"] < 0.0


def test_not_found_is_a_decision_after_empty_outcome():
    prior = drawer_belief((0.0, 0.8, 0.2))
    posterior = update_from_effect(prior, drawer_effect(), EffectOutcome.EMPTY)
    assert posterior.mass(TargetState.ABSENT) == pytest.approx(1.0)
    assert len(posterior.hypotheses) == 1
    assert planner().plan(posterior, ()).selected_action == NOT_FOUND


def test_conformal_ambiguity_prefers_information_to_unsafe_commit():
    belief = drawer_belief((0.5, 0.5, 0.0))
    decision = planner().robust_plan(
        belief,
        (drawer_effect(),),
        conformal_labels=("visible", "middle_drawer"),
    )
    assert decision.selected_action == "REMOVE_OCCLUDER"
    assert decision.resolvable_uncertainty["REMOVE_OCCLUDER"] > 0.0
    assert decision.margin > 0.0


def test_two_step_planner_branches_between_locations():
    belief = TargetBelief(
        hypotheses=(
            TargetHypothesis("left", TargetState.MANIPULATION_ONLY, "OPEN_LEFT"),
            TargetHypothesis("right", TargetState.MANIPULATION_ONLY, "OPEN_RIGHT"),
            TargetHypothesis("absent", TargetState.ABSENT),
        ),
        probabilities=(0.6, 0.35, 0.05),
    )
    effects = (
        ActionEffect("OPEN_LEFT", ("left",), 1.0, 0.05, "test"),
        ActionEffect("OPEN_RIGHT", ("right",), 1.0, 0.05, "test"),
    )
    decision = planner(horizon=2).plan(belief, effects)
    assert decision.selected_action == "OPEN_LEFT"


def test_planner_rejects_unmeasured_effect_labels():
    with pytest.raises(ValueError, match="outside"):
        planner().plan(
            drawer_belief(),
            (ActionEffect("OPEN_OTHER", ("other",), 1.0, 0.0, "test"),),
        )


def test_context_specific_post_reveal_act_capability_changes_information_value():
    hidden = drawer_belief((0.0, 1.0, 0.0))
    capable = ExpectedRiskPlanner(
        losses=DecisionLosses(1.0, 1.0, 1.0),
        act_reliability={"visible": 1.0, "middle_drawer": 1.0, "absent": 1.0},
        horizon=1,
    )
    incapable = ExpectedRiskPlanner(
        losses=DecisionLosses(1.0, 1.0, 1.0),
        act_reliability={"visible": 1.0, "middle_drawer": 0.0, "absent": 1.0},
        horizon=1,
    )
    assert capable.plan(hidden, (drawer_effect(),)).selected_action == "REMOVE_OCCLUDER"
    assert incapable.plan(hidden, (drawer_effect(),)).selected_action != "REMOVE_OCCLUDER"


def test_context_specific_act_reliability_must_be_declared() -> None:
    visible = drawer_belief((1.0, 0.0, 0.0))
    strict = ExpectedRiskPlanner(
        losses=DecisionLosses(1.0, 1.0, 1.0),
        act_reliability={"middle_drawer": 1.0},
        horizon=0,
    )
    with pytest.raises(KeyError, match="visible"):
        strict.plan(visible, ())
