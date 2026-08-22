"""One cross-attention decoder with effect and route heads.

The shared VLM is deliberately external and frozen.  It supplies context tokens
``H_t`` and one open-vocabulary embedding per structured candidate.  This file
does not introduce a belief token, confidence head, manual utility, or policy
trajectory decoder.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
    from torch.nn import functional
except ModuleNotFoundError:  # pragma: no cover - exercised on minimal runtime installs
    torch = None
    nn = None
    functional = None


if nn is not None:

    class CandidateInteractionDecoder(nn.Module):
        """Candidate queries cross-attend once to shared frozen-VLM tokens."""

        def __init__(
            self,
            *,
            vlm_width: int,
            model_width: int,
            num_heads: int,
            effect_factors: int,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            if min(vlm_width, model_width, num_heads, effect_factors) < 1:
                raise ValueError("all decoder dimensions must be positive")
            if model_width % num_heads:
                raise ValueError("model_width must be divisible by num_heads")
            self.context_projection = nn.Linear(vlm_width, model_width)
            self.candidate_projection = nn.Linear(vlm_width, model_width)
            self.cross_attention = nn.MultiheadAttention(
                model_width, num_heads, dropout=dropout, batch_first=True
            )
            self.attention_norm = nn.LayerNorm(model_width)
            self.feed_forward = nn.Sequential(
                nn.Linear(model_width, 4 * model_width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * model_width, model_width),
            )
            self.output_norm = nn.LayerNorm(model_width)
            self.effect_head = nn.Linear(model_width, effect_factors)
            self.route_head = nn.Sequential(
                nn.Linear(model_width + effect_factors, model_width),
                nn.GELU(),
                nn.Linear(model_width, 1),
            )

        def forward(
            self,
            context_tokens: torch.Tensor,
            candidate_tokens: torch.Tensor,
            *,
            context_padding_mask: torch.Tensor | None = None,
            candidate_mask: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            """Return ``c_tj [B,A,D]``, effects ``[B,A,F]``, routes ``[B,A]``."""

            if context_tokens.ndim != 3 or candidate_tokens.ndim != 3:
                raise ValueError(
                    "context_tokens and candidate_tokens must be rank three"
                )
            if context_tokens.shape[0] != candidate_tokens.shape[0]:
                raise ValueError("context and candidate batches must match")
            context = self.context_projection(context_tokens)
            queries = self.candidate_projection(candidate_tokens)
            attended, _ = self.cross_attention(
                queries,
                context,
                context,
                key_padding_mask=context_padding_mask,
                need_weights=False,
            )
            interaction = self.attention_norm(queries + attended)
            interaction = self.output_norm(interaction + self.feed_forward(interaction))
            effect_logits = self.effect_head(interaction)
            route_features = torch.cat((interaction, effect_logits.sigmoid()), dim=-1)
            route_logits = self.route_head(route_features).squeeze(-1)
            if candidate_mask is not None:
                if candidate_mask.shape != route_logits.shape:
                    raise ValueError("candidate_mask must have shape [B,A]")
                route_logits = route_logits.masked_fill(
                    ~candidate_mask.bool(), float("-inf")
                )
            return {
                "interaction_tokens": interaction,
                "effect_logits": effect_logits,
                "route_logits": route_logits,
            }

    def interaction_loss(
        outputs: dict[str, torch.Tensor],
        *,
        route_labels: torch.Tensor,
        effect_labels: torch.Tensor,
        effect_mask: torch.Tensor,
        effect_weight: float,
    ) -> dict[str, torch.Tensor]:
        """Joint objective; callers must explicitly choose and report lambda."""

        if effect_weight < 0.0:
            raise ValueError("effect_weight must be non-negative")
        route_loss = functional.cross_entropy(outputs["route_logits"], route_labels)
        elementwise = functional.binary_cross_entropy_with_logits(
            outputs["effect_logits"], effect_labels.float(), reduction="none"
        )
        mask = effect_mask.to(dtype=elementwise.dtype)
        denominator = mask.sum().clamp_min(1.0)
        effect_loss = (elementwise * mask).sum() / denominator
        return {
            "loss": route_loss + effect_weight * effect_loss,
            "route_loss": route_loss,
            "effect_loss": effect_loss,
        }

else:

    class CandidateInteractionDecoder:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "CandidateInteractionDecoder requires the optional 'learned' dependencies"
            )

    def interaction_loss(*_: Any, **__: Any) -> Any:
        raise RuntimeError(
            "interaction_loss requires the optional 'learned' dependencies"
        )
