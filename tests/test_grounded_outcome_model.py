from __future__ import annotations

import dataclasses

import pytest

from grounded_interaction.contracts import Primitive
from grounded_interaction.losses import executed_candidate_bernoulli_nll
from grounded_interaction.model import (
    GroundedCandidateEncoder,
    GroundedOutcomeModel,
    OutcomeScorer,
)
from grounded_interaction.tokens import CandidateTokenField, FrozenTokenField

torch = pytest.importorskip("torch")


def _digest(character: str) -> str:
    return character * 64


def _field(*, requires_grad: bool = False) -> CandidateTokenField:
    context_tokens = torch.randn(1, 6, 8, requires_grad=requires_grad)
    candidate_tokens = torch.randn(1, 3, 8, requires_grad=requires_grad)
    valid = torch.tensor([[True, True, True, True, True, True]])
    current = torch.tensor([[False, True, True, False, False, False]])
    boxes = torch.zeros(1, 6, 4)
    boxes[0, 1] = torch.tensor([0.0, 0.0, 0.5, 1.0])
    boxes[0, 2] = torch.tensor([0.5, 0.0, 1.0, 1.0])
    context = FrozenTokenField(
        tokens=context_tokens,
        valid_mask=valid,
        current_patch_mask=current,
        camera_ids=((None, "wrist", "wrist", "wrist", None, None),),
        frame_ids=((None, "frame-1", "frame-1", "frame-0", None, None),),
        patch_xyxy=boxes,
        context_fingerprints=(_digest("a"),),
        provider_id="test-vlm@checkpoint-a+preprocess-v1",
    )
    support = torch.zeros(1, 3, 6, dtype=torch.bool)
    support[0, 0, 1] = True
    support[0, 1, 2] = True
    return CandidateTokenField(
        tokens=candidate_tokens,
        valid_mask=torch.tensor([[True, True, True]]),
        candidate_ids=(("open-left", "open-right", "stop"),),
        candidate_fingerprints=((_digest("b"), _digest("c"), _digest("d")),),
        primitives=((Primitive.OPEN, Primitive.OPEN, Primitive.STOP),),
        grounding_support=support,
        public_context=context,
    )


def test_pre_grounded_candidate_is_fused_before_outcome_scoring() -> None:
    torch.manual_seed(7)
    field = _field()
    model = GroundedOutcomeModel(
        context_dim=8,
        candidate_dim=8,
        hidden_dim=8,
        num_heads=2,
    ).eval()
    assert isinstance(model.candidate_encoder, GroundedCandidateEncoder)
    grounded = model.candidate_encoder(field)
    prediction = model.outcomes(grounded)
    assert prediction.task_success_logits.shape == (1, 3)
    assert not hasattr(prediction, "branch_logits")
    assert torch.isfinite(prediction.task_success_logits).all()
    assert grounded.grounding_attention[0, 0, 1].detach().item() == pytest.approx(1.0)
    assert grounded.grounding_attention[0, 1, 2].detach().item() == pytest.approx(1.0)
    assert grounded.grounding_attention[0, 2].sum().detach().item() == pytest.approx(
        0.0
    )

    with pytest.raises(TypeError, match="GroundedCandidateBatch"):
        OutcomeScorer(hidden_dim=8, num_heads=2)(field)  # type: ignore[arg-type]


def test_frozen_vlm_inputs_are_detached() -> None:
    torch.manual_seed(9)
    original_context = torch.randn(1, 4, 8, requires_grad=True)
    original_candidates = torch.randn(1, 2, 8, requires_grad=True)
    boxes = torch.zeros(1, 4, 4)
    boxes[0, 1] = torch.tensor([0.0, 0.0, 1.0, 1.0])
    context = FrozenTokenField(
        tokens=original_context,
        valid_mask=torch.ones(1, 4, dtype=torch.bool),
        current_patch_mask=torch.tensor([[False, True, False, False]]),
        camera_ids=((None, "wrist", None, None),),
        frame_ids=((None, "frame-1", None, None),),
        patch_xyxy=boxes,
        context_fingerprints=(_digest("a"),),
        provider_id="test-vlm@checkpoint-a+preprocess-v1",
    )
    support = torch.zeros(1, 2, 4, dtype=torch.bool)
    support[0, 0, 1] = True
    field = CandidateTokenField(
        tokens=original_candidates,
        valid_mask=torch.tensor([[True, True]]),
        candidate_ids=(("open", "stop"),),
        candidate_fingerprints=((_digest("b"), _digest("c")),),
        primitives=((Primitive.OPEN, Primitive.STOP),),
        grounding_support=support,
        public_context=context,
    )
    model = GroundedOutcomeModel(
        context_dim=8,
        candidate_dim=8,
        hidden_dim=8,
        num_heads=2,
    )
    model(field).task_success_logits.sum().backward()
    assert context.provider_id == "test-vlm@checkpoint-a+preprocess-v1"
    assert original_context.grad is None
    assert original_candidates.grad is None
    assert any(parameter.grad is not None for parameter in model.parameters())
    with pytest.raises(ValueError, match="provider_id"):
        dataclasses.replace(context, provider_id=" ")


def test_optional_public_state_is_detached_but_projection_trains() -> None:
    torch.manual_seed(13)
    raw_state = torch.randn(1, 9, requires_grad=True)
    field = dataclasses.replace(
        _field(),
        public_state_values=raw_state,
        public_state_valid_mask=torch.tensor([True]),
    )
    model = GroundedOutcomeModel(
        context_dim=8,
        candidate_dim=8,
        hidden_dim=8,
        num_heads=2,
        use_public_state=True,
    )
    with pytest.raises(RuntimeError, match="statistics"):
        model(field)
    model.set_public_state_statistics(mean=torch.zeros(9), scale=torch.ones(9))
    model(field).task_success_logits.sum().backward()
    assert raw_state.grad is None
    assert model.outcomes.state_projection is not None
    assert model.outcomes.state_projection.weight.grad is not None
    assert torch.count_nonzero(model.outcomes.state_projection.weight.grad) > 0


def test_missing_public_state_is_masked_explicitly() -> None:
    torch.manual_seed(17)
    base = _field()
    model = GroundedOutcomeModel(
        context_dim=8,
        candidate_dim=8,
        hidden_dim=8,
        num_heads=2,
        use_public_state=True,
    ).eval()
    missing_a = dataclasses.replace(
        base,
        public_state_values=torch.zeros(1, 9),
        public_state_valid_mask=torch.tensor([False]),
    )
    missing_b = dataclasses.replace(
        base,
        public_state_values=torch.full((1, 9), 123.0),
        public_state_valid_mask=torch.tensor([False]),
    )
    prediction_a = model(missing_a).task_success_logits
    prediction_b = model(missing_b).task_success_logits
    assert torch.allclose(prediction_a, prediction_b)


def test_outcome_loss_never_reads_unexecuted_labels() -> None:
    logits = torch.tensor([[0.2, -0.4, 1.0], [0.1, 0.6, -0.2]])
    executed = torch.tensor([[True, False, False], [False, True, False]])
    labels_a = torch.tensor(
        [[1.0, float("nan"), float("nan")], [float("nan"), 0.0, float("nan")]]
    )
    labels_b = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    loss_a = executed_candidate_bernoulli_nll(logits, labels_a, executed)
    loss_b = executed_candidate_bernoulli_nll(logits, labels_b, executed)
    assert loss_a.detach().item() == pytest.approx(loss_b.detach().item())


def test_candidate_fusion_does_not_mix_candidate_rows() -> None:
    torch.manual_seed(11)
    field = _field()
    model = GroundedOutcomeModel(
        context_dim=8,
        candidate_dim=8,
        hidden_dim=8,
        num_heads=2,
    ).eval()
    original = model(field).task_success_logits.detach()
    changed_tokens = field.tokens.clone()
    changed_tokens[0, 0] += 5.0
    changed = model(dataclasses.replace(field, tokens=changed_tokens))
    assert torch.allclose(original[0, 1:], changed.task_success_logits.detach()[0, 1:])


def test_candidate_permutation_only_permutes_predictions() -> None:
    torch.manual_seed(19)
    field = _field()
    model = GroundedOutcomeModel(
        context_dim=8,
        candidate_dim=8,
        hidden_dim=8,
        num_heads=2,
    ).eval()
    permutation = (2, 0, 1)
    original = model(field)
    permuted_field = dataclasses.replace(
        field,
        tokens=field.tokens[:, permutation],
        valid_mask=field.valid_mask[:, permutation],
        candidate_ids=(tuple(field.candidate_ids[0][index] for index in permutation),),
        candidate_fingerprints=(
            tuple(field.candidate_fingerprints[0][index] for index in permutation),
        ),
        primitives=(tuple(field.primitives[0][index] for index in permutation),),
        grounding_support=field.grounding_support[:, permutation],
    )
    permuted = model(permuted_field)
    assert torch.allclose(
        permuted.task_success_logits,
        original.task_success_logits[:, permutation],
    )
    assert permuted.candidate_ids == permuted_field.candidate_ids
    assert permuted.candidate_fingerprints == permuted_field.candidate_fingerprints


def test_physical_candidate_cannot_ground_to_old_or_nonvisual_tokens() -> None:
    field = _field()
    invalid_support = field.grounding_support.clone()
    invalid_support[0, 0] = False
    invalid_support[0, 0, 3] = True
    with pytest.raises(ValueError, match="current patches"):
        CandidateTokenField(
            tokens=field.tokens,
            valid_mask=field.valid_mask,
            candidate_ids=field.candidate_ids,
            candidate_fingerprints=field.candidate_fingerprints,
            primitives=field.primitives,
            grounding_support=invalid_support,
            public_context=field.public_context,
        )
