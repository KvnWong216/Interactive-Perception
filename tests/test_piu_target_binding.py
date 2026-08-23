from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from piu.target_binding import (
    LearnedMultiTaskObjective,
    PromptConditionedTargetBinder,
    binding_objectives,
)


def test_binder_preserves_patch_axis_and_masks_invalid_tokens() -> None:
    torch.manual_seed(11)
    model = PromptConditionedTargetBinder(
        vlm_width=24,
        model_width=16,
        num_heads=4,
        maximum_cameras=2,
        maximum_time_steps=2,
        maximum_action_types=4,
    )
    image = torch.randn(2, 12, 24)
    prompt = torch.randn(2, 5, 24)
    valid = torch.ones(2, 12, dtype=torch.bool)
    valid[:, -2:] = False
    outputs = model(
        image,
        prompt,
        patch_xy=torch.rand(2, 12, 2),
        camera_ids=torch.tensor([[0] * 6 + [1] * 6] * 2),
        temporal_ids=torch.tensor([[0] * 6 + [1] * 6] * 2),
        executed_action_ids=torch.tensor([1, 1]),
        image_valid_mask=valid,
        prompt_valid_mask=torch.ones(2, 5, dtype=torch.bool),
    )
    assert outputs["spatial_logits"].shape == (2, 12)
    assert outputs["target_token"].shape == (2, 16)
    assert torch.isneginf(outputs["spatial_logits"][:, -2:]).all()
    torch.testing.assert_close(outputs["spatial_attention"].sum(dim=-1), torch.ones(2))

    target = torch.zeros(2, 12)
    target[:, 3] = 1.0
    losses = binding_objectives(
        outputs,
        patch_target_distribution=target,
        target_present=torch.ones(2),
        task_sufficient=torch.zeros(2),
        holding_requested_target=torch.zeros(2),
        region_confirmed_empty=torch.zeros(2),
        task_complete=torch.zeros(2),
        image_valid_mask=valid,
    )
    assert set(losses) == {
        "spatial_localization_loss",
        "target_presence_loss",
        "task_sufficiency_loss",
        "holding_requested_target_loss",
        "region_confirmed_empty_loss",
        "task_complete_loss",
    }
    assert all(torch.isfinite(loss) for loss in losses.values())
    learned_objective = LearnedMultiTaskObjective()
    combined = learned_objective(losses)
    combined["loss"].backward()
    assert model.patch_key.weight.grad is not None
    assert learned_objective.log_variances.grad is not None
