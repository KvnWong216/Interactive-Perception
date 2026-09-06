import pytest
import torch

from latent_interaction.contracts import (
    MultimodalTokenBatch,
    PrimitiveTokenBatch,
)


def test_public_multimodal_and_primitive_contracts() -> None:
    context = MultimodalTokenBatch(
        tokens=torch.randn(2, 7, 8),
        valid_mask=torch.ones(2, 7, dtype=torch.bool),
        patch_mask=torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool
        ),
        source_fields=("rgb", "prompt", "public_action_history"),
    )
    primitives = PrimitiveTokenBatch(
        tokens=torch.randn(2, 3, 8),
        valid_mask=torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        primitive_text=(("act", "open", "stop"), ("act", "open", "")),
    )
    assert context.tokens.shape == (2, 7, 8)
    assert primitives.valid_mask.sum().item() == 5


@pytest.mark.parametrize(
    "field",
    ["semantic_id", "instance_id", "evaluator_label", "oracle_target_mask"],
)
def test_privileged_source_fields_fail_closed(field: str) -> None:
    with pytest.raises(ValueError, match="non-public"):
        MultimodalTokenBatch(
            tokens=torch.randn(1, 3, 4),
            valid_mask=torch.ones(1, 3, dtype=torch.bool),
            patch_mask=torch.tensor([[1, 0, 0]], dtype=torch.bool),
            source_fields=("rgb", field),
        )


def test_masks_are_strictly_validated() -> None:
    with pytest.raises(TypeError, match="torch.bool"):
        MultimodalTokenBatch(
            tokens=torch.randn(1, 3, 4),
            valid_mask=torch.ones(1, 3),
            patch_mask=torch.tensor([[1, 0, 0]], dtype=torch.bool),
            source_fields=("rgb",),
        )
    with pytest.raises(ValueError, match="subset"):
        MultimodalTokenBatch(
            tokens=torch.randn(1, 3, 4),
            valid_mask=torch.tensor([[1, 0, 1]], dtype=torch.bool),
            patch_mask=torch.tensor([[0, 1, 0]], dtype=torch.bool),
            source_fields=("rgb",),
        )

