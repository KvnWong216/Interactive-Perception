"""Prompt-conditioned Interaction Uncertainty (PIU).

The package is intentionally small: public scene/task contracts, structured
beliefs, typed action effects, explicit utility optimization, and a lightweight
sidecar that can be trained on frozen VLM features.  Simulator-private labels
belong to offline data builders and never enter these runtime interfaces.
"""

# Transitional namespace extension: three legacy Interactive-Perception modules
# still import the sibling research package.  The PIU runner does not.  This is
# removed together with those legacy modules after the replacement runner has
# passed its embodied smoke.
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

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
