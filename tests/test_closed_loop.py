from __future__ import annotations

import numpy as np

from scripts.serve_public_perception import evidence_prompt, parse_model_json

from interactive_perception.closed_loop import (
    ClosedLoopPromptRouter,
    PublicSceneEvidence,
)


class SequencePerception:
    def __init__(self, values):
        self.values = iter(values)

    def infer(self, *, image, prompt, task_id):
        assert image.ndim == 3
        assert prompt
        assert task_id
        return next(self.values)


def _obs():
    return {"agentview_image": np.zeros((16, 16, 3), dtype=np.uint8)}


def test_multi_drawer_router_updates_after_empty_search() -> None:
    task = {
        "id": "T01_multi_drawer_search",
        "search_locations": [
            {"label": "top", "prior": 0.6, "action_prompt": "Open top."},
            {"label": "middle", "prior": 0.3, "action_prompt": "Open middle."},
            {"label": "bottom", "prior": 0.1, "action_prompt": "Open bottom."},
        ],
    }
    router = ClosedLoopPromptRouter(
        task,
        SequencePerception(
            [
                PublicSceneEvidence(False, False, {}, {}, 0.9),
                PublicSceneEvidence(False, False, {"top": "searched_empty"}, {}, 0.9),
                PublicSceneEvidence(True, True, {"top": "searched_empty"}, {}, 0.9),
            ]
        ),
    )
    first = router.decide(obs=_obs(), step=0, final_prompt="Fetch butter.", camera="agentview")
    second = router.decide(obs=_obs(), step=5, final_prompt="Fetch butter.", camera="agentview")
    third = router.decide(obs=_obs(), step=10, final_prompt="Fetch butter.", camera="agentview")
    assert first.target == "top"
    assert "REMOVE_OCCLUDER" in first.risks
    assert second.target == "middle"
    assert third.primitive == "ACT"
    assert third.prompt == "Fetch butter."


def test_router_declines_only_after_exhaustive_public_search() -> None:
    task = {
        "id": "absent",
        "search_locations": [
            {"label": "top", "prior": 0.5, "action_prompt": "Open top."},
            {"label": "bottom", "prior": 0.5, "action_prompt": "Open bottom."},
        ],
    }
    router = ClosedLoopPromptRouter(
        task,
        SequencePerception(
            [
                PublicSceneEvidence(
                    False,
                    False,
                    {"top": "searched_empty", "bottom": "searched_empty"},
                    {},
                    0.95,
                )
            ]
        ),
    )
    decision = router.decide(obs=_obs(), step=20, final_prompt="Fetch soup.", camera="agentview")
    assert decision.terminal == "NOT_FOUND"


def test_low_confidence_causes_safe_stop() -> None:
    router = ClosedLoopPromptRouter(
        {"id": "task"},
        SequencePerception([PublicSceneEvidence(False, False, {}, {}, 0.2)]),
        minimum_confidence=0.5,
    )
    decision = router.decide(obs=_obs(), step=0, final_prompt="Do task.", camera="agentview")
    assert decision.terminal == "PERCEPTION_UNCERTAIN"


def test_vlm_prompt_names_only_public_search_locations() -> None:
    prompt = evidence_prompt(
        {
            "prompt": "Fetch butter.",
            "search_locations": [{"label": "top_drawer"}],
            "occluder_actions": [],
        }
    )
    assert "top_drawer" in prompt
    assert "hidden simulator state" in prompt
    assert parse_model_json('```json\n{"confidence": 0.8}\n```') == {"confidence": 0.8}
