from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from calibrated_interaction.model import CandidateInteractionDecoder, interaction_loss


def test_decoder_has_one_candidate_axis_and_two_heads() -> None:
    torch.manual_seed(7)
    model = CandidateInteractionDecoder(
        vlm_width=32,
        model_width=16,
        num_heads=4,
        effect_factors=6,
    )
    outputs = model(torch.randn(2, 9, 32), torch.randn(2, 3, 32))
    assert outputs["interaction_tokens"].shape == (2, 3, 16)
    assert outputs["effect_logits"].shape == (2, 3, 6)
    assert outputs["route_logits"].shape == (2, 3)
    assert not any(
        "belief" in name or "confidence" in name for name, _ in model.named_parameters()
    )


def test_joint_loss_requires_explicit_effect_weight_and_backpropagates() -> None:
    model = CandidateInteractionDecoder(
        vlm_width=12,
        model_width=12,
        num_heads=3,
        effect_factors=6,
    )
    outputs = model(torch.randn(2, 5, 12), torch.randn(2, 3, 12))
    losses = interaction_loss(
        outputs,
        route_labels=torch.tensor([0, 2]),
        effect_labels=torch.zeros(2, 3, 6),
        effect_mask=torch.ones(2, 3, 6),
        effect_weight=0.4,
    )
    losses["loss"].backward()
    assert model.cross_attention.in_proj_weight.grad is not None
    with pytest.raises(TypeError):
        interaction_loss(
            outputs,
            route_labels=torch.tensor([0, 2]),
            effect_labels=torch.zeros(2, 3, 6),
            effect_mask=torch.ones(2, 3, 6),
        )
