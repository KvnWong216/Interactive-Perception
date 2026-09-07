"""Observed-outcome loss for the formal grounded-interaction model.

The only supervised quantity in formal v1 is task success within the frozen
executor and continuation contract.  The loss reads exactly one receipt-backed
executed candidate per row; labels on unexecuted alternatives are ignored.

This is a software training contract, not evidence of model performance.
"""

from __future__ import annotations

from typing import Any, Literal

try:  # Keep the core package importable without PyTorch.
    import torch
    import torch.nn.functional as F
    from torch import Tensor
except ImportError:  # pragma: no cover - exercised in torch-free installations.
    torch = None
    F = None
    Tensor = Any  # type: ignore[misc,assignment]


Reduction = Literal["none", "mean", "sum"]


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("training losses require the optional PyTorch dependency")


def executed_candidate_bernoulli_nll(
    task_success_logits: Tensor,
    observed_outcomes: Tensor,
    executed_mask: Tensor,
    *,
    reduction: Reduction = "mean",
) -> Tensor:
    """Return Bernoulli NLL for executed candidates only.

    ``observed_outcomes`` may contain any placeholder, including NaN, outside
    ``executed_mask``.  Those values are removed before validation and
    arithmetic, so changing an unexecuted label cannot change this loss.
    """

    _require_torch()
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if (
        not isinstance(task_success_logits, torch.Tensor)
        or task_success_logits.ndim != 2
    ):
        raise ValueError("task_success_logits must have shape [batch, candidate]")
    if (
        not isinstance(observed_outcomes, torch.Tensor)
        or observed_outcomes.shape != task_success_logits.shape
    ):
        raise ValueError("observed_outcomes must match task_success_logits")
    if (
        not isinstance(executed_mask, torch.Tensor)
        or executed_mask.dtype is not torch.bool
        or executed_mask.shape != task_success_logits.shape
    ):
        raise ValueError("executed_mask must be a matching bool tensor")
    if not (
        task_success_logits.device == observed_outcomes.device == executed_mask.device
    ):
        raise ValueError("all loss tensors must share one device")
    if bool((executed_mask.sum(dim=1) != 1).any()):
        raise ValueError(
            "each training row must identify exactly one executed candidate"
        )

    selected_logits = task_success_logits[executed_mask]
    selected_targets = observed_outcomes[executed_mask].to(dtype=selected_logits.dtype)
    if not bool(torch.isfinite(selected_targets).all()) or bool(
        ((selected_targets != 0) & (selected_targets != 1)).any()
    ):
        raise ValueError("executed outcomes must be finite binary labels")
    return F.binary_cross_entropy_with_logits(
        selected_logits,
        selected_targets,
        reduction=reduction,
    )
