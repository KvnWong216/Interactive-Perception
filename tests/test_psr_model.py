from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from grounded_interaction.psr.model import (
    FrozenEvidenceNormalizer,
    FrozenOrthogonalProjection,
    PredictiveOutput,
    PredictiveStateModel,
    evidence_mixture_nll,
    failure_bce_loss,
    paired_evidence_separability,
)


def _model() -> PredictiveStateModel:
    torch.manual_seed(31)
    embedding = nn.Embedding(101, 12)
    return PredictiveStateModel(
        word_embeddings=embedding,
        native_embedding_dim=12,
        state_dim=10,
        patch_camera_ids=(0, 0, 1, 1),
        readout_dim=16,
        num_heads=4,
        encoder_layers=1,
        predictor_layers=2,
        num_state_tokens=6,
        max_intent_tokens=8,
        route_count=2,
        mixture_components=3,
        evidence_dim=5,
    )


def _inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(37)
    state = torch.randn(2, 6, 10)
    ids = torch.tensor(
        [
            [[3, 4, 5, 0, 0], [6, 7, 0, 0, 0], [8, 9, 10, 11, 12]],
            [[13, 14, 0, 0, 0], [15, 16, 17, 0, 0], [18, 0, 0, 0, 0]],
        ]
    )
    mask = torch.tensor(
        [
            [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]],
            [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]],
        ],
        dtype=torch.bool,
    )
    route = torch.tensor([[1, 1, 0], [0, 1, 0]])
    valid = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    return state, ids, mask, route, valid


def _prediction(
    *,
    batch: int = 2,
    candidates: int = 3,
    components: int = 4,
    positions: int = 3,
    feature_dim: int = 2,
    requires_grad: bool = False,
) -> PredictiveOutput:
    return PredictiveOutput(
        failure_logits=torch.randn(batch, candidates, requires_grad=requires_grad),
        mixture_logits=torch.randn(
            batch, candidates, components, requires_grad=requires_grad
        ),
        evidence_mean=torch.randn(
            batch,
            candidates,
            components,
            positions,
            feature_dim,
            requires_grad=requires_grad,
        ),
        evidence_logstd=torch.zeros(
            batch,
            candidates,
            components,
            positions,
            feature_dim,
            requires_grad=requires_grad,
        ),
        candidate_valid_mask=torch.ones(batch, candidates, dtype=torch.bool),
    )


def test_candidate_permutation_is_equivariant_and_independent() -> None:
    model = _model().eval()
    state, ids, mask, route, valid = _inputs()
    with torch.no_grad():
        original = model(state, ids, mask, route, valid)
        permutation = torch.tensor([2, 0, 1])
        permuted = model(
            state,
            ids[:, permutation],
            mask[:, permutation],
            route[:, permutation],
            valid[:, permutation],
        )
    assert torch.allclose(
        permuted.failure_logits, original.failure_logits[:, permutation], atol=1e-6
    )
    assert torch.allclose(
        permuted.mixture_logits, original.mixture_logits[:, permutation], atol=1e-6
    )
    assert torch.allclose(
        permuted.evidence_mean, original.evidence_mean[:, permutation], atol=1e-6
    )


def test_intent_encoder_has_no_scene_or_contextual_hidden_input() -> None:
    model = _model().eval()
    _, ids, mask, route, valid = _inputs()
    signature = inspect.signature(model.intent_encoder.forward)
    assert tuple(signature.parameters) == (
        "intent_token_ids",
        "intent_token_mask",
        "route_ids",
        "candidate_valid_mask",
    )
    with torch.no_grad():
        first = model.intent_encoder(ids, mask, route, valid)
        # There is no B/H argument: the same ordinary IDs and route produce the
        # same query regardless of which scene-level state will be read later.
        second = model.intent_encoder(ids.clone(), mask.clone(), route.clone(), valid)
    assert torch.equal(first, second)


def test_intent_padding_is_ignored_but_word_order_is_not() -> None:
    model = _model().eval()
    _, ids, mask, route, valid = _inputs()
    changed_padding = ids.clone()
    changed_padding[~mask] = 99
    reordered = ids.clone()
    reordered[0, 0, :3] = torch.tensor([5, 4, 3])
    with torch.no_grad():
        baseline = model.intent_encoder(ids, mask, route, valid)
        padding_result = model.intent_encoder(changed_padding, mask, route, valid)
        reordered_result = model.intent_encoder(reordered, mask, route, valid)
    assert torch.allclose(baseline, padding_result, atol=1e-7)
    assert not torch.allclose(baseline[0, 0], reordered_result[0, 0])
    assert torch.count_nonzero(baseline[1, 2]).item() == 0


def test_predictive_model_shapes_validity_and_gradients() -> None:
    model = _model().train()
    state, ids, mask, route, valid = _inputs()
    state.requires_grad_(True)
    output = model(state, ids, mask, route, valid)
    assert output.failure_logits.shape == (2, 3)
    assert output.mixture_logits.shape == (2, 3, 3)
    assert output.evidence_mean.shape == (2, 3, 3, 4, 5)
    assert output.evidence_logstd.shape == (2, 3, 3, 4, 5)
    assert output.expected_failure.shape == (2, 3)
    assert torch.count_nonzero(output.evidence_mean[1, 2]).item() == 0
    loss = (
        output.failure_logits[valid].sum()
        + output.mixture_logits[valid].sum()
        + output.evidence_mean[valid].sum()
    )
    loss.backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert state.grad.abs().sum().item() > 0
    assert model.intent_encoder.word_projection.weight.grad is not None
    assert next(model.intent_encoder.word_embeddings.parameters()).grad is None


def test_frozen_projection_and_train_only_normalization() -> None:
    projector = FrozenOrthogonalProjection(input_dim=7, requested_output_dim=4, seed=17)
    identical = FrozenOrthogonalProjection(input_dim=7, requested_output_dim=4, seed=17)
    assert torch.equal(projector.projection, identical.projection)
    identity = projector.projection.T @ projector.projection
    assert torch.allclose(identity, torch.eye(4), atol=1e-5)
    features = torch.tensor([[[1.0, 2.0, 4.0, 8.0], [3.0, 4.0, 4.0, 12.0]]])
    normalizer = FrozenEvidenceNormalizer(4)
    with pytest.raises(RuntimeError, match="not been fitted"):
        normalizer(features)
    normalizer.fit(features, torch.tensor([[True, True]]))
    normalized = normalizer(features)
    assert torch.allclose(normalized.mean(dim=(0, 1)), torch.zeros(4), atol=1e-6)
    assert normalizer.constant_channels.tolist() == [False, False, True, False]
    assert torch.isfinite(normalized).all()
    with pytest.raises(RuntimeError, match="already frozen"):
        normalizer.fit(features, torch.tensor([[True, True]]))


def test_global_mixture_nll_is_finite_at_logstd_boundary_and_backpropagates() -> None:
    prediction = _prediction(requires_grad=True)
    # Very negative log std represents the zero-variance failure mode and must
    # be clamped before exponentiation.
    prediction.evidence_logstd.data.fill_(-1e9)
    targets = torch.randn(2, 3, 3, 2)
    valid = torch.ones(2, 3, 3, dtype=torch.bool)
    result = evidence_mixture_nll(prediction, targets, valid)
    assert result.has_supervision and result.valid_item_count == 6
    loss = result.require_value()
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.mixture_logits.grad is not None
    assert prediction.evidence_mean.grad is not None
    assert torch.isfinite(prediction.evidence_mean.grad).all()


def test_one_global_mixture_component_covers_all_patch_positions() -> None:
    mixture_logits = torch.zeros(1, 1, 2)
    means = torch.tensor([[[[[0.0], [0.0]], [[10.0], [10.0]]]]])
    prediction = PredictiveOutput(
        failure_logits=torch.zeros(1, 1),
        mixture_logits=mixture_logits,
        evidence_mean=means,
        evidence_logstd=torch.zeros_like(means),
        candidate_valid_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    mask = torch.ones(1, 1, 2, dtype=torch.bool)
    one_scene = torch.tensor([[[[0.0], [0.0]]]])
    patchwise_hybrid = torch.tensor([[[[0.0], [10.0]]]])
    scene_loss = evidence_mixture_nll(prediction, one_scene, mask).require_value()
    hybrid_loss = evidence_mixture_nll(
        prediction, patchwise_hybrid, mask
    ).require_value()
    # If components were selected independently per patch, both targets would
    # be equally easy.  A single scene-level component makes the hybrid costly.
    assert hybrid_loss.item() > scene_loss.item() + 10.0


def test_evidence_nll_missing_and_partial_labels_are_explicit() -> None:
    prediction = _prediction()
    targets = torch.full((2, 3, 3, 2), torch.nan)
    missing = torch.zeros(2, 3, 3, dtype=torch.bool)
    empty = evidence_mixture_nll(prediction, targets, missing)
    assert empty.value is None
    assert empty.valid_item_count == 0
    assert torch.isnan(empty.per_item).all()
    with pytest.raises(RuntimeError, match="no valid supervision"):
        empty.require_value()

    targets[0, 1] = torch.randn(3, 2)
    partial = missing.clone()
    partial[0, 1, :2] = True
    supervised = torch.zeros(2, 3, dtype=torch.bool)
    supervised[0, 1] = True
    result = evidence_mixture_nll(
        prediction, targets, partial, supervised_candidates=supervised
    )
    assert result.valid_item_count == 1
    assert result.valid_items.nonzero().tolist() == [[0, 1]]
    assert torch.isfinite(result.require_value())
    assert torch.isnan(result.per_item[~result.valid_items]).all()


def test_failure_loss_keeps_missing_targets_out_of_arithmetic() -> None:
    logits = torch.randn(2, 3, requires_grad=True)
    targets = torch.full((2, 3), torch.nan)
    targets[0, 2] = 1.0
    targets[1, 0] = 0.0
    supervised = torch.tensor([[0, 0, 1], [1, 0, 0]], dtype=torch.bool)
    result = failure_bce_loss(logits, targets, supervised)
    assert result.valid_item_count == 2
    assert torch.isfinite(result.require_value())
    result.require_value().backward()
    assert torch.count_nonzero(logits.grad[~supervised]).item() == 0

    empty = failure_bce_loss(logits.detach(), targets, torch.zeros_like(supervised))
    assert empty.value is None and empty.valid_item_count == 0


def test_target_separability_uses_only_common_finite_positions() -> None:
    first = torch.zeros(2, 3, 2)
    second = first.clone()
    second[0, 0] = 2.0
    second[1, 0] = torch.nan
    mask = torch.tensor([[True, False, False], [True, False, False]])
    report = paired_evidence_separability(first, second, mask, mask)
    assert report.valid_pairs.tolist() == [True, False]
    assert report.pair_rms_distance[0].item() == pytest.approx(2.0)
    assert report.mean_rms_distance == pytest.approx(2.0)
