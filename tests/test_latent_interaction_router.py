import torch

from latent_interaction.contracts import MultimodalTokenBatch, PrimitiveTokenBatch
from latent_interaction.losses import (
    executed_action_jepa_cosine_loss,
    multi_positive_route_loss,
    set_likelihood_grounding_loss,
)
from latent_interaction.router import LatentInteractionRouter


def _inputs() -> tuple[MultimodalTokenBatch, PrimitiveTokenBatch]:
    context_tokens = torch.randn(2, 9, 12, requires_grad=True)
    primitive_tokens = torch.randn(2, 4, 12, requires_grad=True)
    context = MultimodalTokenBatch(
        tokens=context_tokens,
        valid_mask=torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 0, 0], [1] * 9], dtype=torch.bool
        ),
        patch_mask=torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0]],
            dtype=torch.bool,
        ),
        source_fields=("rgb", "prompt", "public_action_history"),
    )
    primitives = PrimitiveTokenBatch(
        tokens=primitive_tokens,
        valid_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool),
        primitive_text=(
            ("act", "open drawer", "stop", ""),
            ("act", "open drawer", "rotate box", "stop"),
        ),
    )
    return context, primitives


def test_router_shapes_masks_and_frozen_vlm_gradient_boundary() -> None:
    torch.manual_seed(3)
    context, primitives = _inputs()
    router = LatentInteractionRouter(
        vlm_width=12, model_width=16, num_heads=4, num_evidence_queries=3
    )
    output = router(context, primitives)
    assert output.route_logits.shape == (2, 4)
    assert output.grounding_logits.shape == (2, 4, 9)
    assert output.current_evidence.shape == (2, 3, 16)
    assert output.predicted_future_evidence.shape == (2, 4, 3, 16)
    assert torch.isneginf(output.route_logits[0, 3])
    assert torch.isneginf(output.grounding_logits[0, 0, 4:]).all()

    positive = torch.tensor([[0, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.bool)
    route_loss = multi_positive_route_loss(
        output.route_logits, positive, output.primitive_valid_mask
    )
    target_patches = torch.zeros_like(output.grounding_logits, dtype=torch.bool)
    target_patches[0, 1, 1:3] = True
    target_patches[1, 0, 0] = True
    supervised = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    grounding_loss = set_likelihood_grounding_loss(
        output.grounding_logits, target_patches, supervised
    )
    post = torch.randn_like(output.current_evidence)
    future_loss = executed_action_jepa_cosine_loss(
        output.predicted_future_evidence,
        post,
        torch.tensor([1, 0], dtype=torch.long),
    )
    total = route_loss + grounding_loss + future_loss
    total.backward()

    assert context.tokens.grad is None
    assert primitives.tokens.grad is None
    assert router.evidence_queries.grad is not None
    assert router.route_head.weight.grad is not None
    assert router.grounding_query.weight.grad is not None
    assert router.future_predictor[-1].weight.grad is not None


def test_multi_positive_route_loss_rewards_total_positive_mass() -> None:
    valid = torch.ones(1, 3, dtype=torch.bool)
    positives = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    good = multi_positive_route_loss(
        torch.tensor([[2.0, 2.0, -2.0]]), positives, valid
    )
    bad = multi_positive_route_loss(
        torch.tensor([[-2.0, -2.0, 2.0]]), positives, valid
    )
    assert good < bad


def test_target_evidence_is_stop_grad_and_ema_updated() -> None:
    torch.manual_seed(5)
    context, _ = _inputs()
    router = LatentInteractionRouter(
        vlm_width=12, model_width=16, num_heads=4, num_evidence_queries=3
    )
    router.train()
    assert router.target_evidence_encoder.training is False
    assert all(
        parameter.requires_grad is False
        for parameter in router.target_evidence_encoder.parameters()
    )

    target_parameter = next(router.target_evidence_encoder.parameters())
    online_parameter = next(router.evidence_encoder.parameters())
    before = target_parameter.detach().clone()
    with torch.no_grad():
        online_parameter.add_(2.0)
    router.update_target_encoder(momentum=0.75)
    expected = before * 0.75 + online_parameter.detach() * 0.25
    assert torch.allclose(target_parameter, expected)

    post_evidence = router.encode_target_evidence(context)
    assert post_evidence.shape == (2, 3, 16)
    assert post_evidence.requires_grad is False


def test_valid_context_tokens_matter_but_padding_is_ignored() -> None:
    torch.manual_seed(11)
    context, primitives = _inputs()
    router = LatentInteractionRouter(
        vlm_width=12, model_width=16, num_heads=4, num_evidence_queries=3
    ).eval()
    baseline = router(context, primitives).current_evidence.detach()

    changed_valid = context.tokens.detach().clone()
    changed_valid[0, 5] += 5.0
    valid_context = MultimodalTokenBatch(
        changed_valid,
        context.valid_mask,
        context.patch_mask,
        context.source_fields,
    )
    assert not torch.allclose(
        baseline[0], router(valid_context, primitives).current_evidence[0]
    )

    changed_padding = context.tokens.detach().clone()
    changed_padding[0, 8] += 100.0
    padding_context = MultimodalTokenBatch(
        changed_padding,
        context.valid_mask,
        context.patch_mask,
        context.source_fields,
    )
    assert torch.allclose(
        baseline[0], router(padding_context, primitives).current_evidence[0]
    )


def test_primitive_token_conditions_its_own_future_prediction() -> None:
    torch.manual_seed(13)
    context, primitives = _inputs()
    router = LatentInteractionRouter(
        vlm_width=12, model_width=16, num_heads=4, num_evidence_queries=3
    ).eval()
    baseline = router(context, primitives).predicted_future_evidence.detach()
    changed_tokens = primitives.tokens.detach().clone()
    changed_tokens[0, 1] += 4.0
    changed_primitives = PrimitiveTokenBatch(
        changed_tokens,
        primitives.valid_mask,
        primitives.primitive_text,
    )
    changed = router(context, changed_primitives).predicted_future_evidence.detach()
    assert not torch.allclose(baseline[0, 1], changed[0, 1])
    assert torch.allclose(baseline[0, 0], changed[0, 0])
