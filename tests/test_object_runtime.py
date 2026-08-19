from __future__ import annotations

import numpy as np

from interaction_uncertainty.object_runtime import build_public_node_inputs


def test_public_node_schema_uses_prompt_target_match() -> None:
    scene = {
        "objects": [
            {
                "object_id": "agentview_butter_00",
                "feature_row": 0,
                "bbox_xyxy": [0.0, 0.0, 112.0, 224.0],
                "label_candidates": {"butter package": 1.0},
                "grounding_score": 0.8,
                "mask_score": 0.9,
                "visible_area": 256,
                "view": "agentview",
            },
            {
                "object_id": "wrist_drawer_00",
                "feature_row": 1,
                "bbox_xyxy": [56.0, 56.0, 168.0, 168.0],
                "label_candidates": {"drawer": 1.0},
                "grounding_score": 0.7,
                "mask_score": 0.85,
                "visible_area": 512,
                "view": "wrist",
            },
        ]
    }
    nodes, identifiers = build_public_node_inputs(
        scene=scene,
        object_features=np.zeros((2, 384), dtype=np.float32),
        target="butter",
    )
    assert nodes.shape == (2, 394)
    assert identifiers == ["agentview_butter_00", "wrist_drawer_00"]
    assert nodes[0, -1] == 1.0
    assert nodes[1, -1] == 0.0
    assert nodes[0, -3:-1].tolist() == [1.0, 0.0]
    assert nodes[1, -3:-1].tolist() == [0.0, 1.0]
