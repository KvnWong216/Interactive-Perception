"""Typed tensor contracts for the RSS 2027 latent interaction router.

Only public, policy-visible token sources are accepted here.  Simulator state,
semantic instance identifiers, evaluator labels, and oracle annotations must
remain outside this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch


PUBLIC_TOKEN_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "rgb",
        "wrist_rgb",
        "prompt",
        "public_action_history",
        "proprioception",
    }
)


def _require_tensor(name: str, value: torch.Tensor, *, rank: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got {value.ndim}")


def _require_bool_mask(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    _require_tensor(name, value, rank=len(shape))
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if value.dtype is not torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool")


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def validate_public_fields(source_fields: tuple[str, ...]) -> None:
    """Reject undeclared or privileged policy inputs using a strict whitelist."""

    if not source_fields:
        raise ValueError("source_fields must be non-empty")
    if len(source_fields) != len(set(source_fields)):
        raise ValueError("source_fields must be unique")
    unknown = set(source_fields) - PUBLIC_TOKEN_FIELDS
    if unknown:
        raise ValueError(
            "non-public policy field(s) rejected: " + ", ".join(sorted(unknown))
        )


@dataclass(frozen=True)
class MultimodalTokenBatch:
    """Frozen-VLM token field and masks.

    ``tokens`` has shape ``[B, N, D_vlm]``. ``valid_mask`` marks usable tokens;
    ``patch_mask`` is its visual-patch subset and is the only support on which
    grounding is defined. ``source_fields`` records the raw public observations
    from which the external frozen VLM constructed the tokens.
    """

    tokens: torch.Tensor
    valid_mask: torch.Tensor
    patch_mask: torch.Tensor
    source_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_tensor("tokens", self.tokens, rank=3)
        _require_finite("tokens", self.tokens)
        batch, token_count, width = self.tokens.shape
        if min(batch, token_count, width) < 1:
            raise ValueError("tokens dimensions must be positive")
        shape = (batch, token_count)
        _require_bool_mask("valid_mask", self.valid_mask, shape)
        _require_bool_mask("patch_mask", self.patch_mask, shape)
        if bool((self.patch_mask & ~self.valid_mask).any()):
            raise ValueError("patch_mask must be a subset of valid_mask")
        if bool((~self.valid_mask.any(dim=1)).any()):
            raise ValueError("every example needs at least one valid context token")
        if bool((~self.patch_mask.any(dim=1)).any()):
            raise ValueError("every example needs at least one current visual patch")
        validate_public_fields(self.source_fields)


@dataclass(frozen=True)
class PrimitiveTokenBatch:
    """Open-vocabulary candidate primitive/referent tokens.

    ``tokens`` has shape ``[B, A, D_vlm]`` and is produced by the same frozen
    VLM/tokenizer as the multimodal context. Invalid padded candidates are
    excluded by ``valid_mask``. Candidate strings are provenance only; they are
    never simulator identifiers.
    """

    tokens: torch.Tensor
    valid_mask: torch.Tensor
    primitive_text: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        _require_tensor("tokens", self.tokens, rank=3)
        _require_finite("tokens", self.tokens)
        batch, action_count, width = self.tokens.shape
        if min(batch, action_count, width) < 1:
            raise ValueError("candidate token dimensions must be positive")
        _require_bool_mask(
            "valid_mask", self.valid_mask, (batch, action_count)
        )
        if bool((~self.valid_mask.any(dim=1)).any()):
            raise ValueError("every example needs at least one valid primitive")
        if len(self.primitive_text) != batch:
            raise ValueError("primitive_text must contain one row per batch item")
        for row_index, row in enumerate(self.primitive_text):
            if len(row) != action_count:
                raise ValueError("primitive_text rows must match candidate dimension")
            for action_index, text in enumerate(row):
                if self.valid_mask[row_index, action_index] and not text.strip():
                    raise ValueError("valid primitives require non-empty public text")


@dataclass(frozen=True)
class RouterOutput:
    """Latent route prediction and current-frame grounding distribution."""

    route_logits: torch.Tensor
    grounding_logits: torch.Tensor
    current_evidence: torch.Tensor
    predicted_future_evidence: torch.Tensor
    decision_tokens: torch.Tensor
    primitive_valid_mask: torch.Tensor
    patch_mask: torch.Tensor

    def __post_init__(self) -> None:
        _require_tensor("route_logits", self.route_logits, rank=2)
        _require_tensor("grounding_logits", self.grounding_logits, rank=3)
        _require_tensor("current_evidence", self.current_evidence, rank=3)
        _require_tensor(
            "predicted_future_evidence", self.predicted_future_evidence, rank=4
        )
        _require_tensor("decision_tokens", self.decision_tokens, rank=3)
        batch, actions = self.route_logits.shape
        if self.grounding_logits.shape[:2] != (batch, actions):
            raise ValueError("grounding logits must start with [B,A]")
        if self.current_evidence.shape[0] != batch:
            raise ValueError("current evidence batch mismatch")
        expected_future = (
            batch,
            actions,
            self.current_evidence.shape[1],
            self.current_evidence.shape[2],
        )
        if tuple(self.predicted_future_evidence.shape) != expected_future:
            raise ValueError(
                "predicted future evidence must have shape [B,A,Q,D_model]"
            )
        if tuple(self.decision_tokens.shape[:2]) != (batch, actions):
            raise ValueError("decision tokens must start with [B,A]")
        _require_bool_mask(
            "primitive_valid_mask", self.primitive_valid_mask, (batch, actions)
        )
        _require_bool_mask(
            "patch_mask", self.patch_mask, (batch, self.grounding_logits.shape[2])
        )

