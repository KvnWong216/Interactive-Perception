"""Frozen Stage-1 VLM boundary.

The core method depends on token tensors, not on a particular Hugging Face
class or checkpoint.  Concrete Qwen/Molmo adapters live outside this module and
must expose only public observations through this keyword-only protocol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import torch

from .contracts import MultimodalTokenBatch, PrimitiveTokenBatch


@runtime_checkable
class FrozenVLMTokenProvider(Protocol):
    """Interface for extracting full hidden-token fields without fine-tuning."""

    @property
    def frozen(self) -> bool: ...

    @property
    def provider_id(self) -> str: ...

    @property
    def output_width(self) -> int: ...

    @torch.no_grad()
    def encode_context(
        self,
        *,
        prompts: Sequence[str],
        rgb_history: Mapping[str, torch.Tensor],
        public_action_history: Sequence[Sequence[str]],
        proprioception: torch.Tensor | None = None,
    ) -> MultimodalTokenBatch:
        """Encode prompt, cameras, public history, and optional public state."""

    @torch.no_grad()
    def encode_primitives(
        self,
        *,
        primitive_text: Sequence[Sequence[str]],
    ) -> PrimitiveTokenBatch:
        """Encode the registered, public candidate descriptions."""


def validate_frozen_token_provider(provider: FrozenVLMTokenProvider) -> None:
    """Fail closed before a trainable router consumes external VLM tokens."""

    if not isinstance(provider, FrozenVLMTokenProvider):
        raise TypeError("provider does not implement FrozenVLMTokenProvider")
    if provider.frozen is not True:
        raise ValueError("stage-one VLM token provider must be frozen")
    if not provider.provider_id.strip():
        raise ValueError("provider_id must be non-empty")
    if provider.output_width < 1:
        raise ValueError("output_width must be positive")
