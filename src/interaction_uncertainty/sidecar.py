"""Lightweight trainable PIU heads over frozen prompt-conditioned VLM features."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import ActionEffectForecast, FactDistribution, TaskBelief

LOCATION_LABELS = (
    "visible_workspace",
    "middle_drawer",
    "other_unsearched_region",
    "absent",
)
OUTCOME_LABELS = ("FAILED", "REVEALED", "EMPTY")
ACTION_LABELS = ("DIRECT_ACT", "OPEN_TO_INSPECT", "STOP_NOT_FOUND", "COMPLETE", "ABSTAIN")


def fixed_project(values: np.ndarray, *, output_dimension: int, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim < 1:
        raise ValueError("fixed projection expects [..., feature] input")
    rng = np.random.default_rng(seed)
    projection = rng.choice(
        (-1.0, 1.0), size=(values.shape[-1], output_dimension)
    ).astype(np.float32) / np.sqrt(float(output_dimension))
    return values @ projection


def class_conditional_set(
    probabilities: np.ndarray,
    *,
    labels: tuple[str, ...],
    thresholds: Mapping[str, float],
) -> tuple[str, ...]:
    eligible = tuple(label for label in labels if float(thresholds[label]) >= 0.0)
    result = tuple(
        label
        for index, label in enumerate(labels)
        if label in eligible
        and 1.0 - float(probabilities[index]) <= float(thresholds[label])
    )
    return result or eligible


def build_torch_model(config: Mapping[str, Any]):
    import torch
    from torch import nn

    projected = int(config["projected_dimension"])
    hidden = int(config["hidden_dimension"])
    action_context_dimension = int(
        config.get("action_context_dimension", len(ACTION_LABELS))
    )

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.belief = nn.Sequential(nn.LayerNorm(projected), nn.Linear(projected, hidden), nn.GELU(), nn.Linear(hidden, len(LOCATION_LABELS)))
            self.rank = nn.Sequential(nn.LayerNorm(projected), nn.Linear(projected, hidden), nn.GELU(), nn.Linear(hidden, len(ACTION_LABELS)))
            self.forecast_feature_norm = nn.LayerNorm(projected)
            self.forecast = nn.Sequential(nn.Linear(projected + action_context_dimension, hidden), nn.GELU(), nn.Linear(hidden, len(OUTCOME_LABELS) + 1))
            self.temporal = nn.GRU(projected + 8, hidden, batch_first=True)
            self.outcome = nn.Linear(hidden, len(OUTCOME_LABELS))
            self.future_feature_norm = nn.LayerNorm(projected)
            self.future = nn.Sequential(nn.Linear(projected + action_context_dimension + len(OUTCOME_LABELS), hidden), nn.GELU(), nn.Linear(hidden, len(LOCATION_LABELS)))

        def belief_logits(self, features):
            return self.belief(features)

        def rank_logits(self, features):
            return self.rank(features)

        def forecast_values(self, features, action):
            return self.forecast(
                torch.cat((self.forecast_feature_norm(features), action), dim=-1)
            )

        def outcome_logits(self, history, robot):
            encoded, _ = self.temporal(torch.cat((history, robot), dim=-1))
            return self.outcome(encoded[:, -1])

        def future_logits(self, features, action, outcome):
            return self.future(
                torch.cat(
                    (self.future_feature_norm(features), action, outcome), dim=-1
                )
            )

    return Model()


class PIUSidecarPredictor:
    def __init__(self, artifact: Path) -> None:
        import torch

        payload = torch.load(artifact, map_location="cpu", weights_only=False)
        self.config = dict(payload["config"])
        self.metadata = dict(payload["metadata"])
        self.model = build_torch_model(self.config)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.artifact = artifact
        self.model_stamp = f"piu-sidecar-v0:{hashlib.sha256(artifact.read_bytes()).hexdigest()[:16]}"

    def _project_global(self, features: np.ndarray) -> np.ndarray:
        return fixed_project(
            np.asarray(features, dtype=np.float32)[None, :],
            output_dimension=int(self.config["projected_dimension"]),
            seed=int(self.config["global_projection_seed"]),
        )[0]

    def _project_spatial(self, features: np.ndarray) -> np.ndarray:
        return fixed_project(
            np.asarray(features, dtype=np.float32),
            output_dimension=int(self.config["projected_dimension"]),
            seed=int(self.config["spatial_projection_seed"]),
        )

    def initial_belief(self, *, prompt: str, global_features: np.ndarray) -> TaskBelief:
        import torch

        projected = torch.from_numpy(self._project_global(global_features)[None, :])
        with torch.no_grad():
            probabilities = torch.softmax(self.model.belief_logits(projected), dim=-1)[0].numpy()
        distribution = dict(zip(LOCATION_LABELS, probabilities.tolist(), strict=True))
        prediction_set = class_conditional_set(
            probabilities,
            labels=LOCATION_LABELS,
            thresholds=self.metadata["conformal"]["belief"],
        )
        return TaskBelief(
            prompt=prompt,
            facts=(FactDistribution("target_location", distribution, 1.0),),
            node_uncertainty={"drawer_middle_interior": float(distribution["middle_drawer"])},
            model_stamp=self.model_stamp,
            conformal_sets={"target_location": prediction_set},
        )

    def action_context(self, action: str, *, execution_budget: float = 1.0) -> np.ndarray:
        dimension = int(self.config.get("action_context_dimension", len(ACTION_LABELS)))
        result = np.zeros(dimension, dtype=np.float32)
        result[ACTION_LABELS.index(action)] = 1.0
        if dimension > len(ACTION_LABELS):
            result[len(ACTION_LABELS)] = float(np.clip(execution_budget, 0.0, 1.0))
        return result

    def forecast(
        self,
        *,
        candidate_id: str,
        primitive: str,
        prompt: str,
        spatial_features: np.ndarray,
    ) -> ActionEffectForecast:
        import torch

        projected_np = self._project_spatial(spatial_features)
        if projected_np.ndim == 2:
            projected_np = projected_np[0]
        projected = torch.from_numpy(projected_np[None, :])
        action_np = self.action_context(primitive, execution_budget=1.0)
        action = torch.from_numpy(action_np[None, :])
        with torch.no_grad():
            raw = self.model.forecast_values(projected, action)[0]
            outcome_probabilities = torch.softmax(raw[: len(OUTCOME_LABELS)], dim=-1).numpy()
            progress = float(torch.sigmoid(raw[-1]).item())
            futures: dict[str, TaskBelief] = {}
            for index, outcome in enumerate(OUTCOME_LABELS):
                outcome_one_hot = np.zeros(len(OUTCOME_LABELS), dtype=np.float32)
                outcome_one_hot[index] = 1.0
                logits = self.model.future_logits(
                    projected,
                    action,
                    torch.from_numpy(outcome_one_hot[None, :]),
                )[0]
                values = torch.softmax(logits, dim=-1).numpy()
                futures[outcome] = TaskBelief(
                    prompt=prompt,
                    facts=(
                        FactDistribution(
                            "target_location",
                            dict(zip(LOCATION_LABELS, values.tolist(), strict=True)),
                            1.0,
                        ),
                    ),
                    node_uncertainty={"drawer_middle_interior": float(values[1])},
                    model_stamp=self.model_stamp,
                )
        outcomes = dict(zip(OUTCOME_LABELS, outcome_probabilities.tolist(), strict=True))
        return ActionEffectForecast(
            candidate_id=candidate_id,
            outcome_probabilities=outcomes,
            future_beliefs=futures,
            execution_success_probability=1.0 - outcomes["FAILED"],
            expected_task_progress=progress,
            model_stamp=self.model_stamp,
        )

    def outcome_set(self, *, history_features: np.ndarray, robot_history: np.ndarray) -> tuple[str, ...]:
        import torch

        history = torch.from_numpy(self._project_spatial(history_features)[None, :, :])
        robot = torch.from_numpy(np.asarray(robot_history, dtype=np.float32)[None, :, :])
        with torch.no_grad():
            probabilities = torch.softmax(self.model.outcome_logits(history, robot), dim=-1)[0].numpy()
        return class_conditional_set(
            probabilities,
            labels=OUTCOME_LABELS,
            thresholds=self.metadata["conformal"]["outcome"],
        )

    def learned_progress(self, *, global_features: np.ndarray) -> dict[str, float]:
        import torch

        projected = torch.from_numpy(self._project_global(global_features)[None, :])
        with torch.no_grad():
            values = torch.softmax(self.model.rank_logits(projected), dim=-1)[0].numpy()
        return {
            "direct_act:prompt_target": float(values[ACTION_LABELS.index("DIRECT_ACT")]),
            "open_to_inspect:drawer_middle": float(values[ACTION_LABELS.index("OPEN_TO_INSPECT")]),
            "stop_not_found": float(values[ACTION_LABELS.index("STOP_NOT_FOUND")]),
            "complete": float(values[ACTION_LABELS.index("COMPLETE")]),
            "abstain": float(values[ACTION_LABELS.index("ABSTAIN")]),
        }


def artifact_metadata(path: Path) -> dict[str, Any]:
    import torch

    return dict(torch.load(path, map_location="cpu", weights_only=False)["metadata"])
