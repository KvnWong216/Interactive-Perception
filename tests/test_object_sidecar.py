from __future__ import annotations

import torch

from interaction_uncertainty.object_sidecar import (
    ACTION_LABELS_V1,
    LOCATION_LABELS_V1,
    SEMANTIC_EFFECT_LABELS_V1,
    build_object_torch_model,
    semantic_effect_teacher,
)


def test_semantic_effect_teacher_separates_information_and_task_progress():
    assert semantic_effect_teacher("closed_container", "OPEN_TO_INSPECT") == (
        "TARGET_REVEALED",
        "visible_workspace",
    )
    assert semantic_effect_teacher("visible_workspace", "DIRECT_ACT") == (
        "TASK_PROGRESS",
        "visible_workspace",
    )
    assert semantic_effect_teacher("closed_container", "DIRECT_ACT") == (
        "NO_RELEVANT_CHANGE",
        "closed_container",
    )


def test_object_model_shapes_and_masking():
    model = build_object_torch_model(
        {
            "prefix_projected_dimension": 16,
            "node_input_dimension": 12,
            "hidden_dimension": 8,
        }
    )
    prefix = torch.randn(2, 16)
    nodes = torch.randn(2, 5, 12)
    mask = torch.tensor(
        [[True, True, False, False, False], [True, True, True, True, False]]
    )
    location, action, relevance, attention, fused = model.state_logits(
        prefix, nodes, mask
    )
    assert location.shape == (2, len(LOCATION_LABELS_V1))
    assert action.shape == (2, len(ACTION_LABELS_V1))
    assert relevance.shape == attention.shape == (2, 5)
    assert torch.allclose(attention[~mask], torch.zeros_like(attention[~mask]))
    effect, future = model.effect_logits(
        fused, torch.nn.functional.one_hot(torch.tensor([0, 1]), 2).float()
    )
    assert effect.shape == (2, len(SEMANTIC_EFFECT_LABELS_V1))
    assert future.shape == (2, len(LOCATION_LABELS_V1))
