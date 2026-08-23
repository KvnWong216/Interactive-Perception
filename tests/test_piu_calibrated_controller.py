from __future__ import annotations

import numpy as np

from piu.action_effect import EFFECT_FACTORS
from piu.calibrated_controller import (
    CalibratedCandidate,
    ControllerBeliefSets,
    DecisionKind,
    decide,
    decide_calibrated_sample,
    frechet_joint_lower_bound,
)


def _candidate(
    primitive: str,
    *,
    execution_set=(True,),
    progress_set=(True,),
    change_set=(True,),
) -> CalibratedCandidate:
    probabilities = {name: 0.8 for name in EFFECT_FACTORS}
    sets = {name: frozenset({False, True}) for name in EFFECT_FACTORS}
    sets["execution_succeeded"] = frozenset(execution_set)
    sets["task_progress_succeeded"] = frozenset(progress_set)
    sets["task_relevant_change"] = frozenset(change_set)
    return CalibratedCandidate.from_mapping(
        {
            "candidate_id": primitive.lower(),
            "primitive": primitive,
            "factor_probabilities": probabilities,
            "factor_prediction_sets": sets,
        }
    )


def _belief(
    *,
    sufficient=(False,),
    present=(True,),
    spatial=(True,),
    holding=(False,),
    searched=(False,),
    complete=(False,),
) -> ControllerBeliefSets:
    return ControllerBeliefSets.from_mapping(
        {
            "task_sufficiency": sufficient,
            "target_presence": present,
            "spatial_reference_available": spatial,
            "holding_requested_target": holding,
            "search_coverage_sufficient": searched,
            "task_complete": complete,
        }
    )


def test_information_and_task_execution_have_distinct_gates() -> None:
    opened = _candidate("OPEN")
    decision = decide(
        candidates=[opened],
        route_prediction_set=["open"],
        belief_sets=_belief(sufficient=(False,)),
    )
    assert decision.kind is DecisionKind.INTERACT
    picked = _candidate("PICK")
    decision = decide(
        candidates=[picked],
        route_prediction_set=["pick"],
        belief_sets=_belief(sufficient=(True,)),
    )
    assert decision.kind is DecisionKind.EXECUTE


def test_calibrated_no_spatial_bridge_ablation_is_explicit() -> None:
    picked = _candidate("PICK")
    blocked = decide(
        candidates=[picked],
        route_prediction_set=["pick"],
        belief_sets=_belief(sufficient=(True,), spatial=(False,)),
    )
    assert blocked.kind is DecisionKind.ABSTAIN
    b5 = decide(
        candidates=[picked],
        route_prediction_set=["pick"],
        belief_sets=_belief(sufficient=(True,), spatial=(False,)),
        require_spatial_reference=False,
    )
    assert b5.kind is DecisionKind.EXECUTE


def test_pick_and_place_require_compatible_public_holding_sets() -> None:
    pick = _candidate("PICK")
    ambiguous = decide(
        candidates=[pick],
        route_prediction_set=["pick"],
        belief_sets=_belief(sufficient=(True,), holding=(False, True)),
    )
    assert ambiguous.kind is DecisionKind.ABSTAIN
    place = _candidate("PLACE")
    absent = decide(
        candidates=[place],
        route_prediction_set=["place"],
        belief_sets=_belief(sufficient=(True,), holding=(False,)),
    )
    assert absent.kind is DecisionKind.ABSTAIN
    allowed = decide(
        candidates=[place],
        route_prediction_set=["place"],
        belief_sets=_belief(sufficient=(True,), holding=(True,)),
    )
    assert allowed.kind is DecisionKind.EXECUTE


def test_route_or_effect_ambiguity_abstains() -> None:
    opened = _candidate("OPEN", change_set=(False, True))
    other = _candidate("REMOVE")
    ambiguous_route = decide(
        candidates=[opened, other],
        route_prediction_set=["open", "remove"],
        belief_sets=_belief(),
    )
    assert ambiguous_route.kind is DecisionKind.ABSTAIN
    ambiguous_effect = decide(
        candidates=[opened],
        route_prediction_set=["open"],
        belief_sets=_belief(),
    )
    assert ambiguous_effect.kind is DecisionKind.ABSTAIN


def test_not_found_requires_absence_and_search_coverage() -> None:
    report = _candidate("REPORT_NOT_FOUND")
    blocked = decide(
        candidates=[report],
        route_prediction_set=["report_not_found"],
        belief_sets=_belief(present=(False,), searched=(False, True)),
    )
    assert blocked.kind is DecisionKind.ABSTAIN
    allowed = decide(
        candidates=[report],
        route_prediction_set=["report_not_found"],
        belief_sets=_belief(present=(False,), searched=(True,)),
    )
    assert allowed.kind is DecisionKind.REPORT_NOT_FOUND


def test_stop_is_not_abstain_or_not_found() -> None:
    stop = _candidate("STOP")
    blocked = decide(
        candidates=[stop],
        route_prediction_set=["stop"],
        belief_sets=_belief(complete=(False,)),
    )
    assert blocked.kind is DecisionKind.ABSTAIN
    stopped = decide(
        candidates=[stop],
        route_prediction_set=["stop"],
        belief_sets=_belief(complete=(True,)),
    )
    assert stopped.kind is DecisionKind.STOP


def test_diagnostic_value_is_a_frechet_bound_not_independence_product() -> None:
    assert frechet_joint_lower_bound(0.8, 0.7) == 0.5


def test_calibrated_array_adapter_preserves_empty_unsupported_factor_sets() -> None:
    factor_probability = np.full((1, 1, len(EFFECT_FACTORS)), 0.8)
    factor_sets = np.zeros((1, 1, len(EFFECT_FACTORS), 2), dtype=bool)
    execution = EFFECT_FACTORS.index("execution_succeeded")
    change = EFFECT_FACTORS.index("task_relevant_change")
    factor_sets[0, 0, execution, 1] = True
    factor_sets[0, 0, change, 1] = True
    decision = decide_calibrated_sample(
        candidate_id=np.asarray([["open"]]),
        candidate_primitive=np.asarray([["OPEN"]]),
        candidate_valid_mask=np.asarray([[True]]),
        calibrated={
            "route_prediction_set": np.asarray([[True]]),
            "factor_probability": factor_probability,
            "factor_prediction_sets": factor_sets,
        },
        sample_index=0,
        belief_sets=_belief(sufficient=(False,)),
    )
    assert decision.kind is DecisionKind.INTERACT
