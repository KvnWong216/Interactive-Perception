"""Prompt-conditioned latent interaction router.

The router consumes full token fields from an external frozen VLM. Learned
evidence queries summarize what is currently known; each candidate primitive
predicts a counterfactual successor evidence state. A shared decision token
drives both routing and current-patch grounding. No hand-authored effect
ontology or scalar utility is used.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from .contracts import MultimodalTokenBatch, PrimitiveTokenBatch, RouterOutput


class _ResidualAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * width, width),
        )

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        *,
        context_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(queries),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=~context_valid_mask,
            need_weights=False,
        )
        hidden = queries + attended
        return hidden + self.feed_forward(self.output_norm(hidden))


class _EvidenceEncoder(nn.Module):
    """Learned queries over a frozen multimodal token field."""

    def __init__(
        self,
        *,
        vlm_width: int,
        model_width: int,
        num_heads: int,
        num_queries: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(vlm_width, model_width)
        self.queries = nn.Parameter(torch.empty(num_queries, model_width))
        nn.init.normal_(self.queries, mean=0.0, std=model_width**-0.5)
        self.decoder = _ResidualAttentionBlock(model_width, num_heads, dropout)

    def forward(
        self,
        frozen_tokens: torch.Tensor,
        *,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        context = self.projection(frozen_tokens)
        return self.decode_projected(context, valid_mask=valid_mask)

    def decode_projected(
        self,
        context: torch.Tensor,
        *,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        queries = self.queries.unsqueeze(0).expand(context.shape[0], -1, -1)
        return self.decoder(queries, context, context_valid_mask=valid_mask)


class LatentInteractionRouter(nn.Module):
    """Small trainable adapter over detached frozen-VLM token fields."""

    def __init__(
        self,
        *,
        vlm_width: int,
        model_width: int,
        num_heads: int,
        num_evidence_queries: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(vlm_width, model_width, num_heads, num_evidence_queries) < 1:
            raise ValueError("all router dimensions must be positive")
        if model_width % num_heads:
            raise ValueError("model_width must be divisible by num_heads")

        self.vlm_width = vlm_width
        self.model_width = model_width
        self.num_evidence_queries = num_evidence_queries
        self.primitive_projection = nn.Linear(vlm_width, model_width)
        self.evidence_encoder = _EvidenceEncoder(
            vlm_width=vlm_width,
            model_width=model_width,
            num_heads=num_heads,
            num_queries=num_evidence_queries,
            dropout=dropout,
        )
        # The post-action target branch is an exponential-moving-average copy.
        # It supplies training targets only and never receives gradients.
        self.target_evidence_encoder = copy.deepcopy(self.evidence_encoder)
        self.target_evidence_encoder.requires_grad_(False)
        self.target_evidence_encoder.eval()
        self.future_predictor = nn.Sequential(
            nn.LayerNorm(2 * model_width),
            nn.Linear(2 * model_width, 4 * model_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * model_width, model_width),
        )
        self.decision_decoder = nn.Sequential(
            nn.LayerNorm(3 * model_width),
            nn.Linear(3 * model_width, 2 * model_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * model_width, model_width),
        )
        self.route_head = nn.Linear(model_width, 1)
        self.patch_projection = nn.Linear(model_width, model_width, bias=False)
        self.grounding_query = nn.Linear(model_width, model_width, bias=False)

    @property
    def evidence_queries(self) -> nn.Parameter:
        """Expose the online queries for diagnostics and stable checkpoint tests."""

        return self.evidence_encoder.queries

    def train(self, mode: bool = True) -> "LatentInteractionRouter":
        super().train(mode)
        # EMA targets must not inherit dropout or training-mode behavior.
        self.target_evidence_encoder.eval()
        return self

    @torch.no_grad()
    def update_target_encoder(self, *, momentum: float) -> None:
        """EMA-update the future-evidence target after an optimizer step."""

        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must lie in [0,1)")
        for target, online in zip(
            self.target_evidence_encoder.parameters(),
            self.evidence_encoder.parameters(),
            strict=True,
        ):
            target.mul_(momentum).add_(online, alpha=1.0 - momentum)

    @torch.no_grad()
    def encode_target_evidence(
        self, post_context: MultimodalTokenBatch
    ) -> torch.Tensor:
        """Encode a real post-action observation with the frozen EMA branch."""

        if post_context.tokens.shape[2] != self.vlm_width:
            raise ValueError("post-action token width differs from configured vlm_width")
        return self.target_evidence_encoder(
            post_context.tokens.detach(), valid_mask=post_context.valid_mask
        )

    def forward(
        self,
        context: MultimodalTokenBatch,
        primitives: PrimitiveTokenBatch,
    ) -> RouterOutput:
        if context.tokens.shape[0] != primitives.tokens.shape[0]:
            raise ValueError("context and primitive batch dimensions must match")
        if context.tokens.shape[2] != self.vlm_width:
            raise ValueError("context token width differs from configured vlm_width")
        if primitives.tokens.shape[2] != self.vlm_width:
            raise ValueError("primitive token width differs from configured vlm_width")
        if context.tokens.device != primitives.tokens.device:
            raise ValueError("context and primitive tokens must share a device")

        # The VLM boundary is intentionally frozen even if a caller accidentally
        # supplies tensors connected to an upstream gradient graph.
        frozen_context = context.tokens.detach()
        frozen_primitives = primitives.tokens.detach()
        context_tokens = self.evidence_encoder.projection(frozen_context)
        primitive_tokens = self.primitive_projection(frozen_primitives)

        batch, actions, _ = primitive_tokens.shape
        current_evidence = self.evidence_encoder.decode_projected(
            context_tokens, valid_mask=context.valid_mask
        )

        query_count = current_evidence.shape[1]
        current_by_action = current_evidence[:, None].expand(
            -1, actions, -1, -1
        )
        primitive_by_query = primitive_tokens[:, :, None].expand(
            -1, -1, query_count, -1
        )
        future_delta = self.future_predictor(
            torch.cat((current_by_action, primitive_by_query), dim=-1)
        )
        predicted_future = current_by_action + future_delta

        current_summary = current_evidence.mean(dim=1)[:, None].expand(-1, actions, -1)
        future_summary = predicted_future.mean(dim=2)
        decision_tokens = self.decision_decoder(
            torch.cat((primitive_tokens, current_summary, future_summary), dim=-1)
        )
        route_logits = self.route_head(decision_tokens).squeeze(-1)
        route_logits = route_logits.masked_fill(
            ~primitives.valid_mask, float("-inf")
        )

        grounding_queries = self.grounding_query(decision_tokens)
        patch_keys = self.patch_projection(context_tokens)
        grounding_logits = torch.einsum(
            "bad,bnd->ban", grounding_queries, patch_keys
        ) / math.sqrt(self.model_width)
        grounding_support = (
            primitives.valid_mask[:, :, None] & context.patch_mask[:, None, :]
        )
        grounding_logits = grounding_logits.masked_fill(
            ~grounding_support, float("-inf")
        )

        return RouterOutput(
            route_logits=route_logits,
            grounding_logits=grounding_logits,
            current_evidence=current_evidence,
            predicted_future_evidence=predicted_future,
            decision_tokens=decision_tokens,
            primitive_valid_mask=primitives.valid_mask,
            patch_mask=context.patch_mask,
        )
