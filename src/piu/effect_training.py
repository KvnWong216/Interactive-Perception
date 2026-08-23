"""CPU training and proper development scores for PIU action effects."""

from __future__ import annotations

import copy
import dataclasses
import math
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from .action_effect import EFFECT_FACTORS, EffectArrays

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional learned dependency
    torch = None


@dataclasses.dataclass(frozen=True)
class EffectHyperparameters:
    model_width: int
    num_heads: int
    dropout: float
    learning_rate: float
    epochs: int
    batch_size: int
    seed: int

    def __post_init__(self) -> None:
        if min(self.model_width, self.num_heads, self.epochs, self.batch_size) < 1:
            raise ValueError("effect hyperparameters must be positive")
        if self.model_width % self.num_heads:
            raise ValueError("effect width must be divisible by heads")
        if not 0.0 <= self.dropout < 1.0 or self.learning_rate <= 0.0:
            raise ValueError("invalid effect dropout or learning rate")


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("effect training requires optional learned dependencies")


def batch_indices(
    count: int, *, batch_size: int, generator: torch.Generator
) -> Iterator[torch.Tensor]:
    _require_torch()
    order = torch.randperm(count, generator=generator)
    for start in range(0, count, batch_size):
        yield order[start : start + batch_size]


def tensor_batch(values: EffectArrays, index: Any | None = None) -> dict[str, Any]:
    _require_torch()
    selected = slice(None) if index is None else np.asarray(index, dtype=np.int64)
    return {
        "belief_token": torch.as_tensor(values.belief_token[selected]),
        "candidate_prompt_tokens": torch.as_tensor(
            values.candidate_prompt_tokens[selected]
        ),
        "candidate_prompt_valid_mask": torch.as_tensor(
            values.candidate_prompt_valid_mask[selected]
        ),
        "candidate_valid_mask": torch.as_tensor(
            values.candidate_valid_mask[selected]
        ),
        "candidate_action_id": torch.as_tensor(values.candidate_action_id[selected]),
        "route_target": torch.as_tensor(values.route_target[selected]),
        "effect_target": torch.as_tensor(values.effect_target[selected]),
        "effect_support_mask": torch.as_tensor(
            values.effect_support_mask[selected]
        ),
    }


def _forward(
    model: Any,
    batch: Mapping[str, Any],
    *,
    effect_backprop_to_shared: bool,
    route_use_predicted_effects: bool,
) -> dict[str, Any]:
    return model(
        batch["belief_token"],
        batch["candidate_prompt_tokens"],
        candidate_prompt_valid_mask=batch["candidate_prompt_valid_mask"],
        candidate_valid_mask=batch["candidate_valid_mask"],
        candidate_action_ids=batch["candidate_action_id"],
        effect_backprop_to_shared=effect_backprop_to_shared,
        route_use_predicted_effects=route_use_predicted_effects,
    )


def effect_probability_metrics(
    outputs: Mapping[str, Any], batch: Mapping[str, Any]
) -> dict[str, Any]:
    _require_torch()
    valid = batch["candidate_valid_mask"].bool()
    route_probability = outputs["route_logits"].softmax(dim=1)
    route_target = batch["route_target"]
    route_nll = -(
        route_target
        * outputs["route_logits"].log_softmax(dim=1).masked_fill(~valid, 0.0)
    ).sum(dim=1)
    route_truth = route_target.argmax(dim=1)
    factor_probability = torch.sigmoid(outputs["factor_logits"])
    factor_metrics = {}
    brier_values = []
    for factor_index, factor_name in enumerate(EFFECT_FACTORS):
        support = batch["effect_support_mask"][:, :, factor_index].bool() & valid
        if support.any():
            probability = factor_probability[:, :, factor_index][support]
            truth = batch["effect_target"][:, :, factor_index][support]
            brier = float(((probability - truth) ** 2).mean().detach())
            brier_values.append(brier)
            factor_metrics[factor_name] = {
                "samples": int(support.sum()),
                "brier": brier,
                "nll": float(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        outputs["factor_logits"][:, :, factor_index][support],
                        truth,
                    ).detach()
                ),
            }
        else:
            factor_metrics[factor_name] = {"samples": 0, "unsupported": True}
    return {
        "samples": len(route_target),
        "route_nll": float(route_nll.mean().detach()),
        "route_top1_accuracy": float(
            (route_probability.argmax(dim=1) == route_truth).float().mean().detach()
        ),
        "macro_supported_factor_brier": (
            float(np.mean(brier_values)) if brier_values else None
        ),
        "factors": factor_metrics,
        "raw": {
            "route_logits": outputs["route_logits"].detach().cpu().numpy(),
            "factor_logits": outputs["factor_logits"].detach().cpu().numpy(),
        },
    }


def train_effect_predictor(
    *,
    model: Any,
    objective: Any,
    train: EffectArrays,
    development: EffectArrays,
    hyperparameters: EffectHyperparameters,
    variant: str,
) -> dict[str, Any]:
    _require_torch()
    from .action_effect import effect_objectives

    variants = {"route_only", "stop_gradient_effect", "joint_effect"}
    if variant not in variants:
        raise ValueError(f"unknown effect-training variant {variant!r}")
    if len(train.sample_id) < 1 or len(development.sample_id) < 1:
        raise ValueError("effect train/development data must be non-empty")
    torch.manual_seed(hyperparameters.seed)
    generator = torch.Generator(device="cpu").manual_seed(hyperparameters.seed)
    model.cpu()
    objective.cpu()
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *objective.parameters()],
        lr=hyperparameters.learning_rate,
    )
    development_batch = tensor_batch(development)
    history = []
    best_rank = None
    best_state = None
    effect_enabled = variant != "route_only"
    shared_effect_gradient = variant == "joint_effect"
    for epoch in range(hyperparameters.epochs):
        model.train()
        objective.train()
        train_losses = []
        for index in batch_indices(
            len(train.sample_id),
            batch_size=hyperparameters.batch_size,
            generator=generator,
        ):
            batch = tensor_batch(train, index.numpy())
            outputs = _forward(
                model,
                batch,
                effect_backprop_to_shared=shared_effect_gradient,
                route_use_predicted_effects=effect_enabled,
            )
            separate = effect_objectives(
                outputs,
                route_target_distribution=batch["route_target"],
                factor_target=batch["effect_target"],
                factor_support_mask=batch["effect_support_mask"],
                candidate_valid_mask=batch["candidate_valid_mask"],
            )
            support = {"route_selection": True}
            support.update(
                {
                    factor: effect_enabled
                    and bool(
                        batch["effect_support_mask"][:, :, factor_index].any()
                    )
                    for factor_index, factor in enumerate(EFFECT_FACTORS)
                }
            )
            combined = objective(separate, supported=support)
            optimizer.zero_grad(set_to_none=True)
            combined["loss"].backward()
            optimizer.step()
            train_losses.append(float(combined["loss"].detach()))
        model.eval()
        objective.eval()
        with torch.no_grad():
            metrics = effect_probability_metrics(
                _forward(
                    model,
                    development_batch,
                    effect_backprop_to_shared=shared_effect_gradient,
                    route_use_predicted_effects=effect_enabled,
                ),
                development_batch,
            )
        factor_brier = metrics["macro_supported_factor_brier"]
        rank = (
            float(metrics["route_nll"]),
            (
                float("inf")
                if factor_brier is None or not effect_enabled
                else float(factor_brier)
            ),
            epoch,
        )
        if not math.isfinite(rank[0]):
            raise RuntimeError("development route NLL is non-finite")
        history.append(
            {
                "epoch": epoch + 1,
                "mean_train_loss": float(np.mean(train_losses)),
                "development": {key: value for key, value in metrics.items() if key != "raw"},
            }
        )
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "objective": copy.deepcopy(objective.state_dict()),
                "epoch": epoch + 1,
            }
    if best_state is None:
        raise RuntimeError("effect training did not produce a checkpoint")
    model.load_state_dict(best_state["model"])
    objective.load_state_dict(best_state["objective"])
    model.eval()
    with torch.no_grad():
        metrics = effect_probability_metrics(
            _forward(
                model,
                development_batch,
                effect_backprop_to_shared=shared_effect_gradient,
                route_use_predicted_effects=effect_enabled,
            ),
            development_batch,
        )
    return {
        "variant": variant,
        "history": history,
        "best_epoch": best_state["epoch"],
        "development_metrics": {
            key: value for key, value in metrics.items() if key != "raw"
        },
        "raw_development_predictions": metrics["raw"],
        "model_state": best_state["model"],
        "objective_state": best_state["objective"],
    }
