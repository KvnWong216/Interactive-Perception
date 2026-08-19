"""Object-level PIU sidecar over frozen prefix and public region features."""

from __future__ import annotations

from typing import Any, Mapping


LOCATION_LABELS_V1 = ("visible_workspace", "closed_container")
ACTION_LABELS_V1 = ("DIRECT_ACT", "OPEN_TO_INSPECT")
SEMANTIC_EFFECT_LABELS_V1 = (
    "NO_RELEVANT_CHANGE",
    "TARGET_REVEALED",
    "TASK_PROGRESS",
)


def build_object_torch_model(config: Mapping[str, Any]):
    import math

    import torch
    from torch import nn

    prefix_dimension = int(config["prefix_projected_dimension"])
    node_dimension = int(config["node_input_dimension"])
    hidden = int(config["hidden_dimension"])
    action_dimension = len(ACTION_LABELS_V1)

    class ObjectPIUModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.prefix_encoder = nn.Sequential(
                nn.LayerNorm(prefix_dimension),
                nn.Linear(prefix_dimension, hidden),
                nn.GELU(),
            )
            self.node_encoder = nn.Sequential(
                nn.LayerNorm(node_dimension),
                nn.Linear(node_dimension, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
            self.query = nn.Linear(hidden, hidden, bias=False)
            self.key = nn.Linear(hidden, hidden, bias=False)
            self.node_relevance = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            self.fuse = nn.Sequential(
                nn.LayerNorm(hidden * 2),
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
            )
            self.location = nn.Linear(hidden, len(LOCATION_LABELS_V1))
            self.action = nn.Linear(hidden, len(ACTION_LABELS_V1))
            self.effect = nn.Sequential(
                nn.LayerNorm(hidden + action_dimension),
                nn.Linear(hidden + action_dimension, hidden),
                nn.GELU(),
                nn.Linear(hidden, len(SEMANTIC_EFFECT_LABELS_V1)),
            )
            self.future = nn.Sequential(
                nn.LayerNorm(hidden + action_dimension),
                nn.Linear(hidden + action_dimension, hidden),
                nn.GELU(),
                nn.Linear(hidden, len(LOCATION_LABELS_V1)),
            )

        def encode_state(self, prefix, nodes, node_mask):
            global_value = self.prefix_encoder(prefix)
            node_values = self.node_encoder(nodes)
            attention = torch.einsum(
                "bd,bnd->bn", self.query(global_value), self.key(node_values)
            ) / math.sqrt(float(hidden))
            attention = attention.masked_fill(~node_mask, -1e9)
            weights = torch.softmax(attention, dim=-1)
            weights = weights * node_mask.to(weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            pooled = torch.einsum("bn,bnd->bd", weights, node_values)
            fused = self.fuse(torch.cat((global_value, pooled), dim=-1))
            expanded_global = global_value[:, None, :].expand_as(node_values)
            relevance = self.node_relevance(
                torch.cat((node_values, expanded_global), dim=-1)
            ).squeeze(-1)
            relevance = relevance.masked_fill(~node_mask, -20.0)
            return fused, relevance, weights

        def state_logits(self, prefix, nodes, node_mask):
            fused, relevance, attention = self.encode_state(prefix, nodes, node_mask)
            return self.location(fused), self.action(fused), relevance, attention, fused

        def effect_logits(self, fused, action_one_hot):
            values = torch.cat((fused, action_one_hot), dim=-1)
            return self.effect(values), self.future(values)

    return ObjectPIUModel()


def semantic_effect_teacher(location: str, action: str) -> tuple[str, str]:
    """Counterfactual semantic effect, separated from physical execution risk."""

    if location not in LOCATION_LABELS_V1 or action not in ACTION_LABELS_V1:
        raise ValueError("unknown V1 location/action")
    if location == "visible_workspace" and action == "DIRECT_ACT":
        return "TASK_PROGRESS", "visible_workspace"
    if location == "closed_container" and action == "OPEN_TO_INSPECT":
        return "TARGET_REVEALED", "visible_workspace"
    return "NO_RELEVANT_CHANGE", location
