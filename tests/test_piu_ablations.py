from __future__ import annotations

import numpy as np
import pytest

from piu.ablations import BINDING_ABLATIONS, apply_binding_ablation
from piu.binding_data import BindingArrays
from piu.contracts import Split


def _arrays() -> BindingArrays:
    return BindingArrays(
        sample_id=("a", "b"),
        initial_state_group=("ga", "gb"),
        split=(Split.TRAIN, Split.TRAIN),
        image_tokens=np.arange(2 * 8 * 4).reshape(2, 8, 4).astype(np.float32),
        image_valid_mask=np.ones((2, 8), dtype=bool),
        prompt_tokens=np.arange(2 * 6 * 4).reshape(2, 6, 4).astype(np.float32),
        prompt_valid_mask=np.ones((2, 6), dtype=bool),
        patch_xy=np.zeros((2, 8, 2), dtype=np.float32),
        camera_id=np.asarray([[0, 0, 1, 1] * 2] * 2),
        temporal_id=np.asarray([[0] * 4 + [1] * 4] * 2),
        executed_action_id=np.asarray([1, 2]),
        patch_target=np.zeros((2, 8), dtype=np.float32),
        target_present=np.zeros(2, dtype=np.float32),
        task_sufficient=np.zeros(2, dtype=np.float32),
        task_sufficient_mask=np.zeros(2, dtype=bool),
        holding_requested_target=np.zeros(2, dtype=np.float32),
        holding_requested_target_mask=np.zeros(2, dtype=bool),
        region_confirmed_empty=np.zeros(2, dtype=np.float32),
        region_confirmed_empty_mask=np.zeros(2, dtype=bool),
        task_complete=np.zeros(2, dtype=np.float32),
        task_complete_mask=np.zeros(2, dtype=bool),
    )


def test_every_binding_ablation_is_deterministic_and_label_blind() -> None:
    values = _arrays()
    for name in BINDING_ABLATIONS:
        first = apply_binding_ablation(
            values, name=name, seed=7, no_history_action_id=3
        )
        second = apply_binding_ablation(
            values, name=name, seed=7, no_history_action_id=3
        )
        np.testing.assert_array_equal(first.patch_target, values.patch_target)
        np.testing.assert_array_equal(first.image_tokens, second.image_tokens)
        np.testing.assert_array_equal(first.image_valid_mask, second.image_valid_mask)


def test_no_action_history_uses_a_dedicated_declared_token() -> None:
    values = _arrays()
    with pytest.raises(ValueError, match="NO_HISTORY"):
        apply_binding_ablation(values, name="no_action_history", seed=0)
    ablated = apply_binding_ablation(
        values, name="no_action_history", seed=0, no_history_action_id=3
    )
    assert ablated.executed_action_id.tolist() == [3, 3]


def test_last_frame_and_camera_ablations_retain_valid_tokens() -> None:
    values = _arrays()
    last = apply_binding_ablation(values, name="last_frame_only", seed=0)
    assert last.image_valid_mask.sum(axis=1).tolist() == [4, 4]
    agent = apply_binding_ablation(values, name="agent_view_only", seed=0)
    wrist = apply_binding_ablation(values, name="wrist_view_only", seed=0)
    assert agent.image_valid_mask.sum() == wrist.image_valid_mask.sum() == 8
