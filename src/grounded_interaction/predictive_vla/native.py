"""Small repository-owned adapters around the pretrained model's native API."""

import math
from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class NativeCapabilities:
    hidden_dim: int
    num_vlm_layers: int
    max_action_horizon: int
    max_action_dim: int
    action_trigger_id: int


def discover_capabilities(model):
    backbone = model.model
    text = backbone.transformer.config
    required = (
        "merge_visual_inputs",
        "build_input_embeddings",
        "_extract_kv_states",
        "_get_encoder_attention_mask",
        "generate_actions_from_inputs",
    )
    missing = [name for name in required if not callable(getattr(backbone, name, None))]
    if missing:
        raise ValueError(f"native MolmoAct2 API is missing {missing}")
    values = [
        int(text.hidden_size),
        int(text.num_hidden_layers),
        int(model.config.max_action_horizon),
        int(model.config.max_action_dim),
    ]
    if min(values) < 1 or values[-1] < 7:
        raise ValueError("invalid native model/action dimensions")
    if (
        len(backbone.transformer.blocks) != values[1]
        or len(backbone.action_expert.blocks) != values[1]
    ):
        raise ValueError(
            "the native Action Expert requires one KV context per VLM layer"
        )
    trigger = getattr(model.config, "action_output_token_id", None)
    if type(trigger) is not int or trigger < 0:
        raise ValueError("native action trigger is missing")
    return NativeCapabilities(*values, trigger)


class ResidualLinear(nn.Module):
    """Low-rank trainable update with an exact switchable native bypass."""

    def __init__(self, base, *, rank, alpha, enabled):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("native adaptation target must be Linear")
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("rank and alpha must be positive")
        self.base = base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False).to(base.weight)
        self.up = nn.Linear(rank, base.out_features, bias=False).to(base.weight)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale, self.enabled = alpha / rank, enabled

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, value):
        result = self.base(value)
        if self.enabled():
            result = result + self.up(self.down(value)) * self.scale
        return result


def postprocess_actions(*, outer, actions, stats, tag, action_dim, n_action_steps):
    """Preserve checkpoint action slicing and normalization, including obs offset."""
    actions = outer._slice_action_dim(actions, action_dim)
    actions = outer._slice_action_chunk(
        actions, int(outer.config.n_obs_steps), n_action_steps
    )
    return stats.unnormalize_action(actions, tag)
