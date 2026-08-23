from __future__ import annotations

import numpy as np
import pytest

from piu.executor_bridge import (
    current_spatial_references,
    load_public_candidate,
    serialize_pi05_subtask,
)


def test_calibrated_patch_set_becomes_exact_geometry_in_subtask() -> None:
    references = current_spatial_references(
        prediction_set=np.asarray([False, False, True, False]),
        patch_xy=np.asarray(
            [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]
        ),
        camera_id=np.asarray([0, 0, 0, 0]),
        temporal_id=np.asarray([1, 1, 1, 1]),
        camera_names=("agentview",),
    )
    assert references[0].x_interval == (0.0, 0.5)
    assert references[0].y_interval == (0.5, 1.0)
    prompt = serialize_pi05_subtask(
        {"primitive": "PICK", "target": "requested butter"},
        spatial_references=references,
    )
    assert prompt == (
        "Pick up the requested butter at agentview normalized box "
        "x=[0.0000,0.5000], y=[0.5000,1.0000]."
    )
    assert "confidence" not in prompt


def test_executor_bridge_rejects_privileged_candidate_and_keeps_stop_internal() -> None:
    with pytest.raises(ValueError, match="evaluator-only"):
        load_public_candidate(
            '{"candidate_id":"pick","primitive":"PICK","target":"butter",'
            '"target_pose":[0,0,0]}'
        )
    assert (
        serialize_pi05_subtask(
            {"primitive": "STOP", "target": "task"}, spatial_references=()
        )
        is None
    )
