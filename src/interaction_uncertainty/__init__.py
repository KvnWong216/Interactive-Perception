"""Prompt-conditioned Interaction Uncertainty (PIU).

The package is intentionally small: public scene/task contracts, structured
beliefs, typed action effects, explicit utility optimization, and a lightweight
sidecar that can be trained on frozen VLM features.  Simulator-private labels
belong to offline data builders and never enter these runtime interfaces.
"""

from .contracts import (
    ActionEffectForecast,
    CandidateAction,
    EffectOutcome,
    FactDistribution,
    ObjectNode,
    Primitive,
    ScenePacket,
    TaskBelief,
    TaskSpec,
    UnknownRegion,
)

__all__ = [
    "ActionEffectForecast",
    "CandidateAction",
    "EffectOutcome",
    "FactDistribution",
    "ObjectNode",
    "Primitive",
    "ScenePacket",
    "TaskBelief",
    "TaskSpec",
    "UnknownRegion",
]
