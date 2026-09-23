"""Training-only action-conditioned patch prediction from shared VLM states."""

import torch
from torch import nn
from torch.nn import functional as F


class PatchReadout(nn.Module):
    """One residual cross-attention/FFN block, with no future-query self-attention."""

    def __init__(self, width, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.output_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width)
        )

    def forward(self, queries, memory, valid):
        normalized = self.memory_norm(memory)
        read, _ = self.attention(
            self.query_norm(queries),
            normalized,
            normalized,
            key_padding_mask=~valid,
            need_weights=False,
        )
        queries = queries + read
        return queries + self.ffn(self.output_norm(queries))


class ActionConditionedPredictor(nn.Module):
    """Future queries read shared context and the real action prefix.

    Queries identify time/view/patch positions, never future image content.
    This decoder is absent from the deployed action/answer path.
    """

    def __init__(self, native_width, *, width=256, heads=8, layers=2, time_scale=300):
        super().__init__()
        if min(native_width, width, heads, layers, time_scale) < 1 or width % heads:
            raise ValueError("invalid predictor dimensions or time scale")
        self.time_scale = time_scale
        self.context = nn.Linear(native_width, width)
        self.action = nn.Linear(8, width)  # actual control + its temporal position
        self.query = nn.Linear(5, width)  # u, v, view, crop level, prediction horizon
        self.readouts = nn.ModuleList(PatchReadout(width, heads) for _ in range(layers))
        self.output = nn.Linear(width, native_width)

    def forward(self, shared, context_valid, actions, action_valid, positions, horizon):
        # shared: [B,S,D]; actions [B,T,7]; positions: [B,J,4] = u,v,view,crop level
        if shared.ndim != 3 or context_valid.shape != shared.shape[:2]:
            raise ValueError("shared context mask is misaligned")
        if (
            actions.ndim != 3
            or actions.shape[-1] != 7
            or action_valid.shape != actions.shape[:2]
        ):
            raise ValueError("actions must be [B,T,7] with an aligned mask")
        if (
            positions.ndim != 3
            or positions.shape[-1] != 4
            or positions.shape[0] != shared.shape[0]
        ):
            raise ValueError("future position queries must be [B,J,4]")
        if actions.shape[0] != shared.shape[0]:
            raise ValueError("action and context batch sizes differ")
        if any(mask.dtype != torch.bool for mask in (context_valid, action_valid)):
            raise ValueError("valid masks must be boolean")
        lengths = action_valid.sum(-1)
        h = torch.as_tensor(horizon, device=actions.device).reshape(-1)
        if h.shape != lengths.shape or not torch.equal(h, lengths) or (h < 1).any():
            raise ValueError(
                "future horizon must equal the actual action prefix length"
            )
        times = torch.arange(actions.shape[1], device=actions.device)[None]
        if not torch.equal(action_valid, times < lengths[:, None]):
            raise ValueError("action padding must follow a contiguous real prefix")
        if not context_valid.any(-1).all():
            raise ValueError("shared context cannot be entirely masked")
        # Sanitize padding before projection: 0 * NaN is still NaN.
        clean_actions = torch.where(action_valid[..., None], actions, 0.0)
        clean_shared = torch.where(context_valid[..., None], shared, 0.0)
        if not all(
            torch.isfinite(x).all() for x in (clean_actions, clean_shared, positions)
        ):
            raise ValueError("valid predictor inputs must be finite")
        dtype = self.context.weight.dtype
        action_tokens = self.action(
            torch.cat((clean_actions, ((times + 1) / h[:, None]).unsqueeze(-1)), -1).to(
                dtype
            )
        )
        memory = torch.cat((self.context(clean_shared.to(dtype)), action_tokens), 1)
        valid = torch.cat((context_valid, action_valid), 1)
        query_input = torch.cat(
            (
                positions,
                h[:, None, None].expand(-1, positions.shape[1], 1) / self.time_scale,
            ),
            -1,
        )
        predicted = self.query(query_input.to(dtype))
        for readout in self.readouts:
            predicted = readout(predicted, memory, valid)
        return self.output(predicted)


def patch_prediction_loss(predicted, target, valid):
    """Equal weight per valid native patch, no global pooling or target gradients."""
    if (
        predicted.shape != target.shape
        or valid.shape != target.shape[:-1]
        or valid.dtype != torch.bool
    ):
        raise ValueError("future patch tensors/masks are misaligned")
    if not valid.any():
        raise ValueError("future supervision contains no valid patches")
    prediction = predicted[valid].float()
    truth = target.detach()[valid].float()
    if not torch.isfinite(prediction).all() or not torch.isfinite(truth).all():
        raise ValueError("valid predicted/target patches must be finite")
    # The frozen, per-frame target prevents teacher collapse; normalization
    # prevents arbitrary feature magnitudes from dominating the action loss.
    truth = F.layer_norm(truth, (truth.shape[-1],))
    return F.smooth_l1_loss(prediction, truth)
