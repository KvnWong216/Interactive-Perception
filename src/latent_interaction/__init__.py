"""RSS 2027 latent prompt-conditioned interaction routing scaffold."""

from .conformal import PrimitiveDecision, SplitConformalPrimitiveSet
from .contracts import MultimodalTokenBatch, PrimitiveTokenBatch, RouterOutput
from .executor import (
    FrozenMolmoAct2TextExecutor,
    FrozenVLAActionChunk,
    FrozenVLAObservation,
    validate_frozen_executor,
)
from .losses import (
    executed_action_jepa_cosine_loss,
    multi_positive_route_loss,
    set_likelihood_grounding_loss,
)
from .router import LatentInteractionRouter
from .token_provider import FrozenVLMTokenProvider, validate_frozen_token_provider

__all__ = [
    "FrozenMolmoAct2TextExecutor",
    "FrozenVLMTokenProvider",
    "FrozenVLAActionChunk",
    "FrozenVLAObservation",
    "LatentInteractionRouter",
    "MultimodalTokenBatch",
    "PrimitiveDecision",
    "PrimitiveTokenBatch",
    "RouterOutput",
    "SplitConformalPrimitiveSet",
    "executed_action_jepa_cosine_loss",
    "multi_positive_route_loss",
    "set_likelihood_grounding_loss",
    "validate_frozen_executor",
    "validate_frozen_token_provider",
]
