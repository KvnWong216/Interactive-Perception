"""Narrow adapters that connect the learned Stage-1 model to the runtime.

The only model-specific boundary is a frozen token provider.  A concrete VLM
integration must implement :class:`FrozenTokenProvider` and declare its own
identity; the Protocol itself is not evidence that such an integration exists.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from .contracts import GroundedIntervention, PolicyContext
from .selection import ValuePrediction
from .tokens import CandidateTokenField

try:
    import torch
except ImportError:  # pragma: no cover - exercised by core-only installs.
    torch = None


@runtime_checkable
class FrozenTokenProvider(Protocol):
    """Encode public context and complete candidates with one frozen VLM."""

    @property
    def provider_id(self) -> str: ...

    def encode(
        self,
        context: PolicyContext,
        candidates: Sequence[GroundedIntervention],
    ) -> CandidateTokenField: ...


class TorchOutcomeScorerAdapter:
    """Expose a PyTorch grounded-outcome model through the runtime interface."""

    def __init__(
        self,
        *,
        model: object,
        token_provider: FrozenTokenProvider,
        is_feasible: Callable[[GroundedIntervention], bool],
    ) -> None:
        if torch is None:
            raise RuntimeError("TorchOutcomeScorerAdapter requires PyTorch")
        if not isinstance(token_provider, FrozenTokenProvider):
            raise TypeError("token_provider must implement FrozenTokenProvider")
        if not callable(is_feasible):
            raise TypeError("is_feasible must be callable")
        self.model = model
        self.token_provider = token_provider
        self.is_feasible = is_feasible

    def score(
        self,
        context: PolicyContext,
        candidates: Sequence[GroundedIntervention],
    ) -> tuple[ValuePrediction, ...]:
        options = tuple(candidates)
        field = self.token_provider.encode(context, options)
        if field.batch_size != 1:
            raise ValueError("online scorer expects one policy context per call")
        if field.provider_id != self.token_provider.provider_id:
            raise ValueError("token field provider_id does not match called provider")
        expected_ids = tuple(candidate.candidate_id for candidate in options)
        expected_fingerprints = tuple(candidate.fingerprint() for candidate in options)
        if field.candidate_ids[0] != expected_ids:
            raise ValueError("token provider changed candidate IDs or order")
        if field.candidate_fingerprints[0] != expected_fingerprints:
            raise ValueError("token provider changed candidate fingerprints or order")
        if field.context_fingerprints != (context.fingerprint(),):
            raise ValueError("token provider changed the public context identity")

        if not callable(self.model):
            raise TypeError("model must be callable")
        training = getattr(self.model, "training", None)
        if hasattr(self.model, "eval"):
            self.model.eval()
        with torch.no_grad():
            output = self.model(field)
            probabilities = torch.sigmoid(output.task_success_logits[0]).detach().cpu()
        if training and hasattr(self.model, "train"):
            self.model.train()

        if output.candidate_ids[0] != expected_ids:
            raise ValueError("outcome model changed candidate IDs or order")
        if output.candidate_fingerprints[0] != expected_fingerprints:
            raise ValueError("outcome model changed candidate fingerprints or order")
        if output.context_fingerprints != (context.fingerprint(),):
            raise ValueError("outcome model changed the public context identity")
        if tuple(output.candidate_valid_mask.shape) != (1, len(options)):
            raise ValueError("outcome model returned an invalid candidate mask")

        return tuple(
            ValuePrediction(
                candidate_id=candidate.candidate_id,
                candidate_fingerprint=candidate.fingerprint(),
                success_probability=float(probabilities[index]),
                feasible=(
                    bool(output.candidate_valid_mask[0, index])
                    and bool(self.is_feasible(candidate))
                ),
            )
            for index, candidate in enumerate(options)
        )
