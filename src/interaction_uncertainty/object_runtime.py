"""Online, oracle-free runtime adapter for the frozen object-level PIU sidecar."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .object_sidecar import (
    ACTION_LABELS_V1,
    LOCATION_LABELS_V1,
    SEMANTIC_EFFECT_LABELS_V1,
    build_object_torch_model,
)
from .sidecar import class_conditional_set, fixed_project


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in value.lower().replace("_", " ").split()
        if len(token) > 2 and token not in {"the", "with", "into", "layer"}
    }


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-9, 1.0)
    return -(values * np.log(values)).sum(axis=-1) / math.log(values.shape[-1])


def build_public_node_inputs(
    *,
    scene: Mapping[str, Any],
    object_features: np.ndarray,
    target: str,
) -> tuple[np.ndarray, list[str]]:
    """Reproduce the frozen policy-side node schema without teacher labels."""

    features = np.asarray(object_features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("object feature store must be a rank-two matrix")
    target_terms = _tokens(target)
    rows: list[np.ndarray] = []
    identifiers: list[str] = []
    for node in scene.get("objects", ()):  # all fields are public frontend output
        feature_row = int(node["feature_row"])
        if not 0 <= feature_row < len(features):
            raise IndexError(f"invalid object feature row {feature_row}")
        box = np.asarray(node["bbox_xyxy"], dtype=np.float32) / 224.0
        label_terms = set().union(
            *(_tokens(str(value)) for value in node["label_candidates"])
        )
        metadata = np.asarray(
            [
                float(node["grounding_score"]),
                float(node["mask_score"]),
                min(1.0, float(node["visible_area"]) / (224.0 * 224.0)),
                *box.tolist(),
                float(node["view"] == "agentview"),
                float(node["view"] == "wrist"),
                float(bool(target_terms & label_terms)),
            ],
            dtype=np.float32,
        )
        rows.append(np.concatenate((features[feature_row], metadata)))
        identifiers.append(str(node["object_id"]))
    if not rows:
        raise ValueError("public scene packet contains no object proposals")
    return np.stack(rows), identifiers


class ObjectPIURuntime:
    """Frozen state/effect/utility inference over one live public observation."""

    def __init__(self, artifact: Path) -> None:
        import torch

        artifact = Path(artifact)
        payload = torch.load(artifact, map_location="cpu", weights_only=False)
        self.config = dict(payload["config"])
        self.metadata = dict(payload["metadata"])
        if self.metadata.get("online_oracle_inputs"):
            raise ValueError("frozen object sidecar declares online oracle inputs")
        self.model = build_object_torch_model(self.config)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.artifact = artifact
        self.model_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()

    def infer(
        self,
        *,
        prefix_features: np.ndarray,
        scene: Mapping[str, Any],
        object_features: np.ndarray,
        target: str,
    ) -> dict[str, Any]:
        import torch

        if scene.get("online_oracle_inputs"):
            raise ValueError("scene packet contains online oracle inputs")
        prefix_values = np.asarray(prefix_features, dtype=np.float32)
        if prefix_values.shape != (8192,):
            raise ValueError(f"expected one 8192-D frozen prefix, got {prefix_values.shape}")
        projected = fixed_project(
            prefix_values[None, :],
            output_dimension=int(self.config["prefix_projected_dimension"]),
            seed=int(self.config["prefix_projection_seed"]),
        ).astype(np.float32)
        node_values, object_ids = build_public_node_inputs(
            scene=scene,
            object_features=object_features,
            target=target,
        )
        if node_values.shape[1] != int(self.config["node_input_dimension"]):
            raise ValueError(f"unexpected public node dimension {node_values.shape[1]}")
        with torch.no_grad():
            prefix = torch.from_numpy(projected)
            nodes = torch.from_numpy(node_values[None, :, :])
            node_mask = torch.ones((1, len(node_values)), dtype=torch.bool)
            location_logits, action_logits, relevance, attention, fused = (
                self.model.state_logits(prefix, nodes, node_mask)
            )
            location_probability = torch.softmax(location_logits, dim=-1)[0].numpy()
            action_probability = torch.softmax(action_logits, dim=-1)[0].numpy()
            action_indices = torch.arange(len(ACTION_LABELS_V1))
            one_hot = torch.nn.functional.one_hot(
                action_indices, len(ACTION_LABELS_V1)
            ).float()
            effect_logits, future_logits = self.model.effect_logits(
                fused.repeat(len(ACTION_LABELS_V1), 1), one_hot
            )
            effect_probability = torch.softmax(effect_logits, dim=-1).numpy()
            future_probability = torch.softmax(future_logits, dim=-1).numpy()
            relevance_probability = torch.sigmoid(relevance)[0].numpy()
            attention_probability = attention[0].numpy()

        location_set = class_conditional_set(
            location_probability,
            labels=LOCATION_LABELS_V1,
            thresholds=self.metadata["conformal"]["location"],
        )
        action_set = class_conditional_set(
            action_probability,
            labels=ACTION_LABELS_V1,
            thresholds=self.metadata["conformal"]["action"],
        )
        current_uncertainty = float(_entropy(location_probability[None, :])[0])
        future_uncertainty = _entropy(future_probability)
        information_gain = current_uncertainty - future_uncertainty
        progress = (
            effect_probability[:, SEMANTIC_EFFECT_LABELS_V1.index("TASK_PROGRESS")]
            + effect_probability[:, SEMANTIC_EFFECT_LABELS_V1.index("TARGET_REVEALED")]
        )
        reliability = np.asarray(
            [
                self.config["utility"]["execution_lower_bound"][action]
                for action in ACTION_LABELS_V1
            ],
            dtype=np.float64,
        )
        cost = np.asarray(
            [self.config["utility"]["cost"][action] for action in ACTION_LABELS_V1],
            dtype=np.float64,
        )
        utility = reliability * (
            float(self.config["utility"]["eta_information"]) * information_gain
            + float(self.config["utility"]["eta_task"]) * progress
        ) - cost
        optimizer_action = ACTION_LABELS_V1[int(utility.argmax())]
        location_to_action = {
            "visible_workspace": "DIRECT_ACT",
            "closed_container": "OPEN_TO_INSPECT",
        }
        if (
            len(location_set) == 1
            and len(action_set) == 1
            and location_to_action[location_set[0]] == action_set[0]
            and action_set[0] == optimizer_action
        ):
            emitted_action = optimizer_action
            emission_reason = "singleton belief/action sets agree with explicit utility"
        else:
            emitted_action = "SAFE_STOP"
            emission_reason = "conformal or utility disagreement"

        node_rows = sorted(
            (
                {
                    "object_id": object_id,
                    "relevance_probability": float(relevance_probability[index]),
                    "attention_mass": float(attention_probability[index]),
                }
                for index, object_id in enumerate(object_ids)
            ),
            key=lambda row: (-row["relevance_probability"], -row["attention_mass"]),
        )
        effects = {}
        for index, action in enumerate(ACTION_LABELS_V1):
            effects[action] = {
                "outcome_probabilities": dict(
                    zip(
                        SEMANTIC_EFFECT_LABELS_V1,
                        effect_probability[index].astype(float).tolist(),
                        strict=True,
                    )
                ),
                "future_location_belief": dict(
                    zip(
                        LOCATION_LABELS_V1,
                        future_probability[index].astype(float).tolist(),
                        strict=True,
                    )
                ),
                "expected_future_uncertainty": float(future_uncertainty[index]),
                "expected_information_gain": float(information_gain[index]),
                "expected_task_progress": float(progress[index]),
                "execution_reliability_lower_bound": float(reliability[index]),
                "cost": float(cost[index]),
                "utility": float(utility[index]),
            }
        return {
            "schema_version": "interaction-uncertainty.piu-object-runtime.v1",
            "model_sha256": self.model_sha256,
            "location_probabilities": dict(
                zip(LOCATION_LABELS_V1, location_probability.astype(float).tolist(), strict=True)
            ),
            "location_prediction_set": list(location_set),
            "action_probabilities": dict(
                zip(ACTION_LABELS_V1, action_probability.astype(float).tolist(), strict=True)
            ),
            "action_prediction_set": list(action_set),
            "task_uncertainty": current_uncertainty,
            "node_uncertainty": node_rows,
            "action_effects": effects,
            "optimizer_selected": optimizer_action,
            "emitted_action": emitted_action,
            "emission_reason": emission_reason,
            "online_oracle_inputs": [],
        }
