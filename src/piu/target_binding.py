"""Lightweight prompt-conditioned spatial binder over frozen prefix tokens."""

from __future__ import annotations

import math
from typing import Any

try:
    import torch
    from torch import nn
    from torch.nn import functional
except ModuleNotFoundError:  # pragma: no cover - optional learned dependency
    torch = None
    nn = None
    functional = None


if nn is not None:

    class PromptConditionedTargetBinder(nn.Module):
        """Learn a prompt query and a calibrated-ready spatial target map.

        There is no online decision threshold here. The model retains each valid
        image token, camera identity, temporal identity, and 2-D patch position.
        """

        def __init__(
            self,
            *,
            vlm_width: int,
            model_width: int,
            num_heads: int,
            maximum_cameras: int,
            maximum_time_steps: int,
            maximum_action_types: int,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            if min(
                vlm_width,
                model_width,
                num_heads,
                maximum_cameras,
                maximum_time_steps,
                maximum_action_types,
            ) < 1:
                raise ValueError("all binder dimensions must be positive")
            if model_width % num_heads:
                raise ValueError("model_width must be divisible by num_heads")
            self.input_norm = nn.LayerNorm(vlm_width, elementwise_affine=False)
            self.image_projection = nn.Linear(vlm_width, model_width)
            self.prompt_projection = nn.Linear(vlm_width, model_width)
            self.coordinate_projection = nn.Sequential(
                nn.Linear(2, model_width),
                nn.GELU(),
                nn.Linear(model_width, model_width),
            )
            self.camera_embedding = nn.Embedding(maximum_cameras, model_width)
            self.time_embedding = nn.Embedding(maximum_time_steps, model_width)
            self.action_embedding = nn.Embedding(maximum_action_types, model_width)
            self.prompt_seed = nn.Parameter(torch.empty(1, 1, model_width))
            nn.init.normal_(self.prompt_seed, std=model_width**-0.5)
            self.prompt_readout = nn.MultiheadAttention(
                model_width, num_heads, dropout=dropout, batch_first=True
            )
            self.image_readout = nn.MultiheadAttention(
                model_width, num_heads, dropout=dropout, batch_first=True
            )
            self.query_norm = nn.LayerNorm(model_width)
            self.patch_key = nn.Linear(model_width, model_width)
            self.target_value = nn.Linear(model_width, model_width)
            self.present_head = nn.Linear(2 * model_width, 1)
            self.sufficiency_head = nn.Linear(2 * model_width, 1)

        def forward(
            self,
            image_tokens: torch.Tensor,
            prompt_tokens: torch.Tensor,
            *,
            patch_xy: torch.Tensor,
            camera_ids: torch.Tensor,
            temporal_ids: torch.Tensor,
            executed_action_ids: torch.Tensor,
            image_valid_mask: torch.Tensor,
            prompt_valid_mask: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            if image_tokens.ndim != 3 or prompt_tokens.ndim != 3:
                raise ValueError("image and prompt tokens must be rank three")
            batch, token_count, _ = image_tokens.shape
            if prompt_tokens.shape[0] != batch:
                raise ValueError("image/prompt batches must match")
            if patch_xy.shape != (batch, token_count, 2):
                raise ValueError("patch_xy must have shape [B,S,2]")
            for name, value in (
                ("camera_ids", camera_ids),
                ("temporal_ids", temporal_ids),
                ("image_valid_mask", image_valid_mask),
            ):
                if value.shape != (batch, token_count):
                    raise ValueError(f"{name} must have shape [B,S]")
            if prompt_valid_mask.shape != prompt_tokens.shape[:2]:
                raise ValueError("prompt_valid_mask shape mismatch")
            if executed_action_ids.shape != (batch,):
                raise ValueError("executed_action_ids must have shape [B]")
            if not image_valid_mask.bool().any(dim=1).all():
                raise ValueError("every row requires a valid image token")
            if not prompt_valid_mask.bool().any(dim=1).all():
                raise ValueError("every row requires a valid prompt token")

            image = self.image_projection(self.input_norm(image_tokens))
            image = (
                image
                + self.coordinate_projection(patch_xy)
                + self.camera_embedding(camera_ids)
                + self.time_embedding(temporal_ids)
            )
            prompt = self.prompt_projection(self.input_norm(prompt_tokens))
            seed = self.prompt_seed.expand(batch, -1, -1) + self.action_embedding(
                executed_action_ids
            )[:, None, :]
            prompt_query, _ = self.prompt_readout(
                seed,
                prompt,
                prompt,
                key_padding_mask=~prompt_valid_mask.bool(),
                need_weights=False,
            )
            grounded_query, _ = self.image_readout(
                prompt_query,
                image,
                image,
                key_padding_mask=~image_valid_mask.bool(),
                need_weights=False,
            )
            query = self.query_norm(prompt_query + grounded_query).squeeze(1)
            keys = self.patch_key(image)
            spatial_logits = torch.einsum("bd,bsd->bs", query, keys) / math.sqrt(
                keys.shape[-1]
            )
            spatial_logits = spatial_logits.masked_fill(
                ~image_valid_mask.bool(), float("-inf")
            )
            spatial_attention = spatial_logits.softmax(dim=-1)
            target_token = torch.einsum(
                "bs,bsd->bd", spatial_attention, self.target_value(image)
            )
            joint = torch.cat((query, target_token), dim=-1)
            return {
                "spatial_logits": spatial_logits,
                "spatial_attention": spatial_attention,
                "target_token": target_token,
                "target_present_logit": self.present_head(joint).squeeze(-1),
                "task_sufficiency_logit": self.sufficiency_head(joint).squeeze(-1),
            }


    def binding_objectives(
        outputs: dict[str, torch.Tensor],
        *,
        patch_target_distribution: torch.Tensor,
        target_present: torch.Tensor,
        task_sufficient: torch.Tensor,
        image_valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return separate objectives; no hand-weighted total is hidden here."""

        logits = outputs["spatial_logits"]
        if patch_target_distribution.shape != logits.shape:
            raise ValueError("patch_target_distribution shape mismatch")
        target = patch_target_distribution.masked_fill(~image_valid_mask.bool(), 0.0)
        denominator = target.sum(dim=-1, keepdim=True)
        localized = denominator.squeeze(-1) > 0
        normalized = target / denominator.clamp_min(1.0)
        log_probabilities = functional.log_softmax(logits, dim=-1)
        log_probabilities = log_probabilities.masked_fill(
            ~image_valid_mask.bool(), 0.0
        )
        per_row = -(normalized * log_probabilities).sum(dim=-1)
        spatial_loss = (
            per_row[localized].mean()
            if localized.any()
            else logits.sum() * 0.0
        )
        return {
            "spatial_localization_loss": spatial_loss,
            "target_presence_loss": functional.binary_cross_entropy_with_logits(
                outputs["target_present_logit"], target_present.float()
            ),
            "task_sufficiency_loss": functional.binary_cross_entropy_with_logits(
                outputs["task_sufficiency_logit"], task_sufficient.float()
            ),
        }

else:

    class PromptConditionedTargetBinder:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "PromptConditionedTargetBinder requires the optional learned dependencies"
            )

    def binding_objectives(*_: Any, **__: Any) -> Any:
        raise RuntimeError("binding_objectives requires the learned dependencies")
