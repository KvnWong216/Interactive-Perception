"""Learning objectives for latent interaction routing."""

from __future__ import annotations

import torch
from torch.nn import functional


def multi_positive_route_loss(
    route_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Negative log probability assigned to the set of admissible primitives."""

    if route_logits.ndim != 2:
        raise ValueError("route_logits must have shape [B,A]")
    if positive_mask.shape != route_logits.shape or valid_mask.shape != route_logits.shape:
        raise ValueError("positive_mask and valid_mask must match route_logits")
    if positive_mask.dtype is not torch.bool or valid_mask.dtype is not torch.bool:
        raise TypeError("route masks must have dtype torch.bool")
    if bool((positive_mask & ~valid_mask).any()):
        raise ValueError("positive primitives must be valid candidates")
    if bool((~positive_mask.any(dim=1)).any()):
        raise ValueError("every example requires at least one positive primitive")
    if bool((~valid_mask.any(dim=1)).any()):
        raise ValueError("every example requires at least one valid primitive")

    valid_logits = route_logits.masked_fill(~valid_mask, float("-inf"))
    positive_logits = route_logits.masked_fill(~positive_mask, float("-inf"))
    return (
        torch.logsumexp(valid_logits, dim=1)
        - torch.logsumexp(positive_logits, dim=1)
    ).mean()


def executed_action_jepa_cosine_loss(
    predicted_future_evidence: torch.Tensor,
    post_action_evidence: torch.Tensor,
    executed_action_indices: torch.Tensor,
    *,
    evidence_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Align an executed action's predicted successor with a stop-grad target."""

    if predicted_future_evidence.ndim != 4:
        raise ValueError("predicted_future_evidence must have shape [B,A,Q,D]")
    batch, actions, queries, width = predicted_future_evidence.shape
    if post_action_evidence.shape != (batch, queries, width):
        raise ValueError("post_action_evidence must have shape [B,Q,D]")
    if executed_action_indices.shape != (batch,):
        raise ValueError("executed_action_indices must have shape [B]")
    if executed_action_indices.dtype != torch.long:
        raise TypeError("executed_action_indices must have dtype torch.long")
    if bool(
        ((executed_action_indices < 0) | (executed_action_indices >= actions)).any()
    ):
        raise ValueError("executed action index is out of range")

    batch_indices = torch.arange(batch, device=predicted_future_evidence.device)
    predicted = predicted_future_evidence[batch_indices, executed_action_indices]
    target = post_action_evidence.detach()
    cosine_distance = 1.0 - functional.cosine_similarity(predicted, target, dim=-1)
    if evidence_mask is None:
        return cosine_distance.mean()
    if evidence_mask.shape != (batch, queries) or evidence_mask.dtype is not torch.bool:
        raise ValueError("evidence_mask must be boolean with shape [B,Q]")
    if not bool(evidence_mask.any()):
        raise ValueError("evidence_mask must supervise at least one query")
    return cosine_distance[evidence_mask].mean()


def set_likelihood_grounding_loss(
    grounding_logits: torch.Tensor,
    target_patch_mask: torch.Tensor,
    supervised_candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Maximize probability mass on any valid target patch, not one chosen pixel."""

    if grounding_logits.ndim != 3:
        raise ValueError("grounding_logits must have shape [B,A,N]")
    if target_patch_mask.shape != grounding_logits.shape:
        raise ValueError("target_patch_mask must match grounding_logits")
    if supervised_candidate_mask.shape != grounding_logits.shape[:2]:
        raise ValueError("supervised_candidate_mask must have shape [B,A]")
    if (
        target_patch_mask.dtype is not torch.bool
        or supervised_candidate_mask.dtype is not torch.bool
    ):
        raise TypeError("grounding masks must have dtype torch.bool")
    if not bool(supervised_candidate_mask.any()):
        raise ValueError("at least one candidate must have grounding supervision")
    missing_targets = supervised_candidate_mask & ~target_patch_mask.any(dim=-1)
    if bool(missing_targets.any()):
        raise ValueError("every supervised candidate needs at least one target patch")

    selected_logits = grounding_logits[supervised_candidate_mask]
    selected_targets = target_patch_mask[supervised_candidate_mask]
    if bool((~torch.isfinite(selected_logits).any(dim=-1)).any()):
        raise ValueError("supervised grounding logits must have finite support")
    log_normalizer = torch.logsumexp(selected_logits, dim=-1)
    target_logits = selected_logits.masked_fill(~selected_targets, float("-inf"))
    log_target_mass = torch.logsumexp(target_logits, dim=-1)
    losses = log_normalizer - log_target_mass
    if not bool(torch.isfinite(losses).all()):
        raise ValueError("supervised target patches must have finite logits")
    return losses.mean()
