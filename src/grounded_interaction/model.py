"""Minimal learned model for grounded, candidate-conditioned outcomes.

The architecture has exactly two learned steps.  First, a candidate query can
attend only to the current-image patch support fixed by its intervention
contract.  Second, an outcome scorer lets that grounded candidate attend to the
remaining public prompt/observation/history context and predicts whether the
bounded task will succeed under the frozen execution contract.

The output is an uncalibrated predictive logit.  It is not a hand-designed
uncertainty score, a route label, or evidence that the system works.  Empirical
claims require trained checkpoints and held-out reset-controlled evaluation.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from .tokens import (
    CandidateTokenField,
    GroundedCandidateBatch,
)

try:  # Core contracts stay importable without the learned dependency.
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - exercised in torch-free installations.
    torch = None
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


_Module = nn.Module if nn is not None else object


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError(
            "the grounded outcome model requires the optional PyTorch dependency"
        )


def _masked_support_attention(
    scores: Tensor,
    support: Tensor,
) -> Tensor:
    """Softmax over a fixed support without NaNs for support-free STOP rows."""

    expanded_support = support.unsqueeze(2)
    has_support = expanded_support.any(dim=-1, keepdim=True)
    masked_scores = scores.masked_fill(~expanded_support, -torch.inf)
    safe_scores = torch.where(has_support, masked_scores, torch.zeros_like(scores))
    weights = torch.softmax(safe_scores, dim=-1)
    return torch.where(expanded_support, weights, torch.zeros_like(weights))


class GroundedCandidateEncoder(_Module):
    """Fuse each candidate with its pre-declared current-patch support.

    This module does not discover or supervise a referent.  Candidate proposal
    and grounding happen upstream; this encoder preserves that binding while it
    builds the representation consumed by the outcome scorer.
    """

    def __init__(
        self,
        *,
        context_dim: int,
        candidate_dim: int,
        hidden_dim: int,
        num_heads: int,
    ) -> None:
        _require_torch()
        super().__init__()
        if min(context_dim, candidate_dim, hidden_dim, num_heads) < 1:
            raise ValueError("model dimensions and head count must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.candidate_projection = nn.Linear(candidate_dim, hidden_dim)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.grounding_norm = nn.LayerNorm(hidden_dim)

    def forward(self, field: CandidateTokenField) -> GroundedCandidateBatch:
        if not isinstance(field, CandidateTokenField):
            raise TypeError("GroundedCandidateEncoder accepts CandidateTokenField only")
        context = field.public_context
        if context.channel_count != self.context_projection.in_features:
            raise ValueError("context token dimension does not match the model")
        if field.tokens.shape[2] != self.candidate_projection.in_features:
            raise ValueError("candidate token dimension does not match the model")

        public_tokens = self.context_projection(context.tokens)
        public_tokens = public_tokens * context.valid_mask.unsqueeze(-1)
        candidate_tokens = self.candidate_projection(field.tokens)

        batch_size, candidate_count, _ = candidate_tokens.shape
        token_count = public_tokens.shape[1]
        query = self.query_projection(candidate_tokens).reshape(
            batch_size, candidate_count, self.num_heads, self.head_dim
        )
        key = self.key_projection(public_tokens).reshape(
            batch_size, token_count, self.num_heads, self.head_dim
        )
        value = self.value_projection(public_tokens).reshape(
            batch_size, token_count, self.num_heads, self.head_dim
        )
        scores = torch.einsum("bahd,bnhd->bahn", query, key)
        scores = scores / math.sqrt(self.head_dim)
        head_weights = _masked_support_attention(scores, field.grounding_support)
        attended = torch.einsum("bahn,bnhd->bahd", head_weights, value)
        attended = attended.reshape(batch_size, candidate_count, self.hidden_dim)
        grounded = self.grounding_norm(
            candidate_tokens + self.output_projection(attended)
        )
        grounded = grounded * field.valid_mask.unsqueeze(-1)
        attention = head_weights.mean(dim=2) * field.valid_mask.unsqueeze(-1)

        return GroundedCandidateBatch(
            grounded_tokens=grounded,
            public_context_tokens=public_tokens,
            public_context_valid_mask=context.valid_mask,
            candidate_valid_mask=field.valid_mask,
            grounding_attention=attention,
            candidate_ids=field.candidate_ids,
            candidate_fingerprints=field.candidate_fingerprints,
            primitives=field.primitives,
            context_fingerprints=field.context_fingerprints,
        )


@dataclasses.dataclass(frozen=True)
class OutcomePrediction:
    """Per-candidate predictions tied to immutable intervention fingerprints."""

    task_success_logits: Tensor
    candidate_valid_mask: Tensor
    candidate_ids: tuple[tuple[str | None, ...], ...]
    candidate_fingerprints: tuple[tuple[str | None, ...], ...]
    context_fingerprints: tuple[str, ...]


class OutcomeScorer(_Module):
    """Predict bounded task outcome from already-grounded candidates.

    The type boundary is deliberate: raw context tokens or ungrounded candidate
    tokens cannot be passed to this scorer.  The single task-success logit is
    the direct quantity used for candidate comparison.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int | None = None,
    ) -> None:
        _require_torch()
        super().__init__()
        if hidden_dim < 1 or num_heads < 1 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if feedforward_dim is None:
            feedforward_dim = 4 * hidden_dim
        if feedforward_dim < 1:
            raise ValueError("feedforward_dim must be positive")
        self.context_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.success_head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: GroundedCandidateBatch) -> OutcomePrediction:
        if not isinstance(batch, GroundedCandidateBatch):
            raise TypeError("OutcomeScorer accepts GroundedCandidateBatch only")
        attended, _ = self.context_attention(
            query=batch.grounded_tokens,
            key=batch.public_context_tokens,
            value=batch.public_context_tokens,
            key_padding_mask=~batch.public_context_valid_mask,
            need_weights=False,
        )
        state = self.context_norm(batch.grounded_tokens + attended)
        state = self.output_norm(state + self.feedforward(state))
        task_success_logits = self.success_head(state).squeeze(-1)
        task_success_logits = task_success_logits.masked_fill(
            ~batch.candidate_valid_mask, 0.0
        )
        return OutcomePrediction(
            task_success_logits=task_success_logits,
            candidate_valid_mask=batch.candidate_valid_mask,
            candidate_ids=batch.candidate_ids,
            candidate_fingerprints=batch.candidate_fingerprints,
            context_fingerprints=batch.context_fingerprints,
        )


class GroundedOutcomeModel(_Module):
    """Composition of grounded-candidate fusion and outcome scoring."""

    def __init__(
        self,
        *,
        context_dim: int,
        candidate_dim: int,
        hidden_dim: int,
        num_heads: int,
    ) -> None:
        _require_torch()
        super().__init__()
        self.candidate_encoder = GroundedCandidateEncoder(
            context_dim=context_dim,
            candidate_dim=candidate_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )
        self.outcomes = OutcomeScorer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )

    def forward(self, field: CandidateTokenField) -> OutcomePrediction:
        return self.outcomes(self.candidate_encoder(field))
