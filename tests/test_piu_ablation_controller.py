from __future__ import annotations

import numpy as np

from piu.ablation_controller import decide_unique_argmax
from piu.calibrated_controller import DecisionKind


def test_uncalibrated_ablation_uses_unique_argmax() -> None:
    decision = decide_unique_argmax(
        route_logits=np.asarray([0.1, 0.8, -np.inf]),
        candidate_valid_mask=np.asarray([True, True, False]),
        candidate_id=np.asarray(["pick", "open", ""]),
        candidate_primitive=np.asarray(["PICK", "OPEN", ""]),
    )
    assert decision.kind is DecisionKind.INTERACT
    assert decision.selected_candidate_id == "open"


def test_uncalibrated_exact_tie_abstains_instead_of_using_array_order() -> None:
    decision = decide_unique_argmax(
        route_logits=np.asarray([0.5, 0.5]),
        candidate_valid_mask=np.asarray([True, True]),
        candidate_id=np.asarray(["pick", "open"]),
        candidate_primitive=np.asarray(["PICK", "OPEN"]),
    )
    assert decision.kind is DecisionKind.ABSTAIN
    assert decision.selected_candidate_id is None
