"""Deterministic, label-blind ablations for the frozen PIU mainline."""

from __future__ import annotations

import dataclasses

import numpy as np

from .binding_data import BindingArrays

BINDING_ABLATIONS = (
    "full",
    "no_prompt",
    "prompt_swap",
    "last_frame_only",
    "mean_image_tokens",
    "no_action_history",
    "agent_view_only",
    "wrist_view_only",
    "shuffled_spatial_positions",
    "shuffled_temporal_order",
)


def apply_binding_ablation(
    values: BindingArrays,
    *,
    name: str,
    seed: int,
    no_history_action_id: int | None = None,
) -> BindingArrays:
    """Transform public inputs only; evaluator targets are never inspected."""

    if name not in BINDING_ABLATIONS:
        raise ValueError(f"unknown binding ablation {name!r}")
    if name == "full":
        return values
    rng = np.random.default_rng(seed)
    replacements = {}
    if name == "no_prompt":
        replacements["prompt_tokens"] = np.zeros_like(values.prompt_tokens)
    elif name == "prompt_swap":
        if len(values.sample_id) < 2:
            raise ValueError("prompt swap requires at least two samples")
        permutation = np.roll(rng.permutation(len(values.sample_id)), 1)
        if np.any(permutation == np.arange(len(permutation))):
            permutation = np.roll(np.arange(len(permutation)), 1)
        replacements["prompt_tokens"] = values.prompt_tokens[permutation].copy()
        replacements["prompt_valid_mask"] = values.prompt_valid_mask[
            permutation
        ].copy()
    elif name == "last_frame_only":
        latest = values.temporal_id.max(axis=1, keepdims=True)
        replacements["image_valid_mask"] = values.image_valid_mask & (
            values.temporal_id == latest
        )
        time_count = int(values.temporal_id.max()) + 1
        if values.prompt_tokens.shape[1] % time_count:
            raise ValueError("prompt tokens cannot be partitioned by time")
        prompt_per_time = values.prompt_tokens.shape[1] // time_count
        prompt_time = np.repeat(np.arange(time_count), prompt_per_time)
        replacements["prompt_valid_mask"] = values.prompt_valid_mask & (
            prompt_time[None] == time_count - 1
        )
    elif name == "mean_image_tokens":
        image = values.image_tokens.copy()
        for row in range(len(values.sample_id)):
            group_ids = np.stack(
                (values.temporal_id[row], values.camera_id[row]), axis=1
            )
            for group in np.unique(group_ids, axis=0):
                selected = (
                    (group_ids == group).all(axis=1)
                    & values.image_valid_mask[row]
                )
                if selected.any():
                    image[row, selected] = image[row, selected].mean(axis=0)
        replacements["image_tokens"] = image
    elif name == "no_action_history":
        if no_history_action_id is None or no_history_action_id < 0:
            raise ValueError(
                "no_action_history requires the explicit non-negative ID of the "
                "dedicated NO_HISTORY vocabulary token"
            )
        replacements["executed_action_id"] = np.full_like(
            values.executed_action_id, no_history_action_id
        )
    elif name in {"agent_view_only", "wrist_view_only"}:
        camera = 0 if name == "agent_view_only" else 1
        replacements["image_valid_mask"] = values.image_valid_mask & (
            values.camera_id == camera
        )
    elif name == "shuffled_spatial_positions":
        patch_xy = values.patch_xy.copy()
        for row in range(len(values.sample_id)):
            patch_xy[row] = patch_xy[row, rng.permutation(patch_xy.shape[1])]
        replacements["patch_xy"] = patch_xy
    elif name == "shuffled_temporal_order":
        temporal_id = values.temporal_id.copy()
        for row in range(len(values.sample_id)):
            temporal_id[row] = temporal_id[row, rng.permutation(temporal_id.shape[1])]
        replacements["temporal_id"] = temporal_id
    return dataclasses.replace(values, **replacements)
