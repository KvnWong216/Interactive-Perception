"""CPU training and threshold-free development metrics for the PIU binder."""

from __future__ import annotations

import copy
import dataclasses
import math
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from .binding_data import BindingArrays

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional learned dependency
    torch = None


@dataclasses.dataclass(frozen=True)
class BinderHyperparameters:
    model_width: int
    num_heads: int
    dropout: float
    learning_rate: float
    epochs: int
    batch_size: int
    seed: int

    def __post_init__(self) -> None:
        if (
            min(
                self.model_width,
                self.num_heads,
                self.epochs,
                self.batch_size,
            )
            < 1
        ):
            raise ValueError("binder hyperparameters must be positive")
        if self.model_width % self.num_heads:
            raise ValueError("model_width must be divisible by num_heads")
        if not (0.0 <= self.dropout < 1.0) or self.learning_rate <= 0.0:
            raise ValueError("invalid dropout or learning rate")


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("binding training requires the learned dependencies")


def batch_indices(
    count: int, *, batch_size: int, generator: torch.Generator
) -> Iterator[torch.Tensor]:
    _require_torch()
    order = torch.randperm(count, generator=generator)
    for start in range(0, count, batch_size):
        yield order[start : start + batch_size]


def tensor_batch(values: BindingArrays, index: Any | None = None) -> dict[str, Any]:
    """Materialize only the selected frozen-feature rows on CPU."""

    _require_torch()
    selected = slice(None) if index is None else np.asarray(index, dtype=np.int64)
    return {
        "image_tokens": torch.as_tensor(values.image_tokens[selected]),
        "prompt_tokens": torch.as_tensor(values.prompt_tokens[selected]),
        "patch_xy": torch.as_tensor(values.patch_xy[selected]),
        "camera_ids": torch.as_tensor(values.camera_id[selected]),
        "temporal_ids": torch.as_tensor(values.temporal_id[selected]),
        "executed_action_ids": torch.as_tensor(values.executed_action_id[selected]),
        "image_valid_mask": torch.as_tensor(values.image_valid_mask[selected]),
        "prompt_valid_mask": torch.as_tensor(values.prompt_valid_mask[selected]),
        "patch_target_distribution": torch.as_tensor(values.patch_target[selected]),
        "target_present": torch.as_tensor(values.target_present[selected]),
        "task_sufficient": torch.as_tensor(values.task_sufficient[selected]),
        "task_sufficient_mask": torch.as_tensor(values.task_sufficient_mask[selected]),
        "holding_requested_target": torch.as_tensor(
            values.holding_requested_target[selected]
        ),
        "holding_requested_target_mask": torch.as_tensor(
            values.holding_requested_target_mask[selected]
        ),
        "region_confirmed_empty": torch.as_tensor(
            values.region_confirmed_empty[selected]
        ),
        "region_confirmed_empty_mask": torch.as_tensor(
            values.region_confirmed_empty_mask[selected]
        ),
        "task_complete": torch.as_tensor(values.task_complete[selected]),
        "task_complete_mask": torch.as_tensor(values.task_complete_mask[selected]),
    }


def _forward(model: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    return model(
        batch["image_tokens"],
        batch["prompt_tokens"],
        patch_xy=batch["patch_xy"],
        camera_ids=batch["camera_ids"],
        temporal_ids=batch["temporal_ids"],
        executed_action_ids=batch["executed_action_ids"],
        image_valid_mask=batch["image_valid_mask"],
        prompt_valid_mask=batch["prompt_valid_mask"],
    )


def probability_metrics(
    outputs: Mapping[str, Any], batch: Mapping[str, Any]
) -> dict[str, Any]:
    """Report proper scores and exact patch hits without a fitted threshold."""

    _require_torch()
    spatial = outputs["spatial_attention"].detach().cpu().numpy()
    patch_target = batch["patch_target_distribution"].detach().cpu().numpy()
    localized = patch_target.sum(axis=1) > 0
    if localized.any():
        normalized = patch_target[localized] / patch_target[localized].sum(
            axis=1, keepdims=True
        )
        probabilities = np.clip(spatial[localized], 1e-12, 1.0)
        spatial_nll = float(-(normalized * np.log(probabilities)).sum(axis=1).mean())
        predicted_patch = spatial[localized].argmax(axis=1)
        point_hit = float(
            np.mean(
                patch_target[localized][
                    np.arange(int(localized.sum())), predicted_patch
                ]
                > 0
            )
        )
        target_mass = float(
            np.mean((spatial[localized] * (patch_target[localized] > 0)).sum(axis=1))
        )
    else:
        spatial_nll = None
        point_hit = None
        target_mass = None
    presence_probability = (
        torch.sigmoid(outputs["target_present_logit"]).detach().cpu().numpy()
    )
    presence_truth = batch["target_present"].detach().cpu().numpy()
    presence_brier = float(np.mean((presence_probability - presence_truth) ** 2))
    presence_nll = float(
        -np.mean(
            presence_truth * np.log(np.clip(presence_probability, 1e-12, 1.0))
            + (1.0 - presence_truth)
            * np.log(np.clip(1.0 - presence_probability, 1e-12, 1.0))
        )
    )
    sufficiency_mask = batch["task_sufficient_mask"].detach().cpu().numpy().astype(bool)
    if sufficiency_mask.any():
        sufficiency_probability = (
            torch.sigmoid(outputs["task_sufficiency_logit"]).detach().cpu().numpy()
        )
        sufficiency_truth = batch["task_sufficient"].detach().cpu().numpy()
        sufficiency_brier = float(
            np.mean(
                (
                    sufficiency_probability[sufficiency_mask]
                    - sufficiency_truth[sufficiency_mask]
                )
                ** 2
            )
        )
    else:
        sufficiency_brier = None
    holding_mask = (
        batch["holding_requested_target_mask"].detach().cpu().numpy().astype(bool)
    )
    if holding_mask.any():
        holding_probability = (
            torch.sigmoid(outputs["holding_requested_target_logit"])
            .detach()
            .cpu()
            .numpy()
        )
        holding_truth = batch["holding_requested_target"].detach().cpu().numpy()
        holding_brier = float(
            np.mean(
                (holding_probability[holding_mask] - holding_truth[holding_mask]) ** 2
            )
        )
    else:
        holding_brier = None
    verifier_brier = {}
    for name in ("region_confirmed_empty", "task_complete"):
        support = batch[f"{name}_mask"].detach().cpu().numpy().astype(bool)
        if support.any():
            probability = torch.sigmoid(outputs[f"{name}_logit"]).detach().cpu().numpy()
            truth = batch[name].detach().cpu().numpy()
            verifier_brier[f"{name}_brier"] = float(
                np.mean((probability[support] - truth[support]) ** 2)
            )
        else:
            verifier_brier[f"{name}_brier"] = None
    return {
        "samples": len(presence_truth),
        "localized_samples": int(localized.sum()),
        "spatial_nll": spatial_nll,
        "point_hit": point_hit,
        "target_probability_mass": target_mass,
        "presence_brier": presence_brier,
        "presence_nll": presence_nll,
        "sufficiency_brier": sufficiency_brier,
        "holding_brier": holding_brier,
        **verifier_brier,
        "raw": {
            "spatial_logits": outputs["spatial_logits"].detach().cpu().numpy(),
            "spatial_attention": spatial,
            "target_token": outputs["target_token"].detach().cpu().numpy(),
            "target_present_logit": outputs["target_present_logit"]
            .detach()
            .cpu()
            .numpy(),
            "task_sufficiency_logit": outputs["task_sufficiency_logit"]
            .detach()
            .cpu()
            .numpy(),
            "holding_requested_target_logit": outputs["holding_requested_target_logit"]
            .detach()
            .cpu()
            .numpy(),
            "region_confirmed_empty_logit": outputs["region_confirmed_empty_logit"]
            .detach()
            .cpu()
            .numpy(),
            "task_complete_logit": outputs["task_complete_logit"]
            .detach()
            .cpu()
            .numpy(),
        },
    }


def train_binder(
    *,
    model: Any,
    objective: Any,
    train: BindingArrays,
    development: BindingArrays,
    hyperparameters: BinderHyperparameters,
) -> dict[str, Any]:
    """Train for a declared epoch budget and retain the best dev spatial NLL."""

    _require_torch()
    from .target_binding import binding_objectives

    if len(train.sample_id) < 1 or len(development.sample_id) < 1:
        raise ValueError("train and development sets must be non-empty")
    if not np.any(train.patch_target.sum(axis=1) > 0):
        raise ValueError("training split has no localizable target example")
    if not np.any(development.patch_target.sum(axis=1) > 0):
        raise ValueError("development split has no localizable target example")
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
    best: tuple[float, float, int] | None = None
    best_state = None
    for epoch in range(hyperparameters.epochs):
        model.train()
        objective.train()
        epoch_losses = []
        for indices in batch_indices(
            len(train.sample_id),
            batch_size=hyperparameters.batch_size,
            generator=generator,
        ):
            batch = tensor_batch(train, indices.numpy())
            outputs = _forward(model, batch)
            separate = binding_objectives(
                outputs,
                patch_target_distribution=batch["patch_target_distribution"],
                target_present=batch["target_present"],
                task_sufficient=batch["task_sufficient"],
                image_valid_mask=batch["image_valid_mask"],
                task_sufficient_mask=batch["task_sufficient_mask"],
                holding_requested_target=batch["holding_requested_target"],
                holding_requested_target_mask=batch["holding_requested_target_mask"],
                region_confirmed_empty=batch["region_confirmed_empty"],
                region_confirmed_empty_mask=batch["region_confirmed_empty_mask"],
                task_complete=batch["task_complete"],
                task_complete_mask=batch["task_complete_mask"],
            )
            support = {
                "spatial_localization_loss": bool(
                    (batch["patch_target_distribution"].sum(dim=1) > 0).any()
                ),
                "target_presence_loss": True,
                "task_sufficiency_loss": bool(batch["task_sufficient_mask"].any()),
                "holding_requested_target_loss": bool(
                    batch["holding_requested_target_mask"].any()
                ),
                "region_confirmed_empty_loss": bool(
                    batch["region_confirmed_empty_mask"].any()
                ),
                "task_complete_loss": bool(batch["task_complete_mask"].any()),
            }
            combined = objective(separate, supported=support)
            optimizer.zero_grad(set_to_none=True)
            combined["loss"].backward()
            optimizer.step()
            epoch_losses.append(float(combined["loss"].detach()))
        model.eval()
        objective.eval()
        with torch.no_grad():
            metrics = probability_metrics(
                _forward(model, development_batch), development_batch
            )
        spatial_nll = metrics["spatial_nll"]
        if spatial_nll is None or not math.isfinite(spatial_nll):
            raise RuntimeError("development spatial NLL is unavailable")
        rank = (float(spatial_nll), float(metrics["presence_brier"]), epoch)
        history.append(
            {
                "epoch": epoch + 1,
                "mean_train_loss": float(np.mean(epoch_losses)),
                "development": {
                    key: value for key, value in metrics.items() if key != "raw"
                },
            }
        )
        if best is None or rank < best:
            best = rank
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "objective": copy.deepcopy(objective.state_dict()),
                "epoch": epoch + 1,
            }
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state["model"])
    objective.load_state_dict(best_state["objective"])
    model.eval()
    with torch.no_grad():
        final_metrics = probability_metrics(
            _forward(model, development_batch), development_batch
        )
    return {
        "history": history,
        "best_epoch": best_state["epoch"],
        "development_metrics": {
            key: value for key, value in final_metrics.items() if key != "raw"
        },
        "raw_development_predictions": final_metrics["raw"],
        "model_state": best_state["model"],
        "objective_state": best_state["objective"],
    }
