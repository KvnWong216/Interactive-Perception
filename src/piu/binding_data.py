"""Target-binding labels and exact mask-to-prefix-token alignment."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import Split
from .spatial_prefix import PrefixLayout, validate_feature_arrays

TIME_STEPS = ("pre_interaction", "post_interaction")


def rle_encode(mask: np.ndarray) -> dict[str, Any]:
    """Encode a two-dimensional boolean mask using alternating flat runs."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("binding masks must be two-dimensional")
    flat = values.astype(np.uint8).reshape(-1)
    counts: list[int] = []
    current = 0
    length = 0
    for value in flat:
        item = int(value)
        if item == current:
            length += 1
        else:
            counts.append(length)
            current = item
            length = 1
    counts.append(length)
    return {"size": list(values.shape), "counts": counts, "starts_with": 0}


def rle_decode(value: Mapping[str, Any]) -> np.ndarray:
    """Decode the evaluator-side RLE and reject malformed run lengths."""

    size = tuple(int(item) for item in value.get("size", ()))
    if len(size) != 2 or min(size) < 1:
        raise ValueError("RLE size must contain two positive dimensions")
    current = int(value.get("starts_with", 0))
    if current not in (0, 1):
        raise ValueError("RLE starts_with must be zero or one")
    parts = []
    for length_value in value.get("counts", ()):
        length = int(length_value)
        if length < 0:
            raise ValueError("RLE run lengths must be non-negative")
        parts.append(np.full(length, current, dtype=bool))
        current = 1 - current
    flat = np.concatenate(parts) if parts else np.empty(0, dtype=bool)
    if flat.size != math.prod(size):
        raise ValueError("RLE runs do not match declared mask size")
    return flat.reshape(size)


def mask_to_patch_coverage(mask: np.ndarray, *, grid_side: int) -> np.ndarray:
    """Return the exact target-pixel fraction in every row-major patch."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or grid_side < 1:
        raise ValueError("mask must be 2-D and grid_side positive")
    height, width = values.shape
    if height % grid_side or width % grid_side:
        raise ValueError(
            f"mask shape {values.shape} is not divisible by a {grid_side}x{grid_side} grid"
        )
    patch_height = height // grid_side
    patch_width = width // grid_side
    blocks = values.reshape(grid_side, patch_height, grid_side, patch_width).transpose(
        0, 2, 1, 3
    )
    return blocks.mean(axis=(2, 3), dtype=np.float32).reshape(-1)


@dataclasses.dataclass(frozen=True)
class BindingLabel:
    """Evaluator-only spatial supervision, never a policy observation."""

    sample_id: str
    initial_state_group: str
    split: Split
    target_masks: Mapping[str, Mapping[str, Mapping[str, Any]]]
    target_present_post: bool
    task_sufficient_post: bool | None
    holding_requested_target_post: bool | None
    region_confirmed_empty_post: bool | None
    task_complete_post: bool | None
    executed_action: str
    simulator_teacher_only: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BindingLabel:
        if value.get("schema_version") != "piu.binding-label.v1":
            raise ValueError("unsupported binding-label schema")
        masks = value.get("target_mask_policy_resolution_rle", {})
        if set(masks) != set(TIME_STEPS):
            raise ValueError("binding labels require pre/post target masks")
        decoded: dict[str, dict[str, Mapping[str, Any]]] = {}
        for time_step in TIME_STEPS:
            camera_masks = masks[time_step]
            if not isinstance(camera_masks, Mapping) or not camera_masks:
                raise ValueError(f"{time_step} requires camera masks")
            decoded[time_step] = {}
            for camera, encoded in camera_masks.items():
                rle_decode(encoded)
                decoded[time_step][str(camera)] = encoded
        present = value.get("target_present_post")
        sufficient = value.get("task_sufficient_post")
        required_nullable = {
            "holding_requested_target_post",
            "region_confirmed_empty_post",
            "task_complete_post",
        }
        missing_nullable = required_nullable - set(value)
        if missing_nullable:
            raise ValueError(
                f"binding label lacks post-observation verifier fields "
                f"{sorted(missing_nullable)}"
            )
        holding = value.get("holding_requested_target_post")
        region_empty = value.get("region_confirmed_empty_post")
        task_complete = value.get("task_complete_post")
        teacher = value.get("simulator_teacher_only")
        if not isinstance(present, bool) or not isinstance(teacher, bool):
            raise TypeError("presence and teacher flags must be explicit booleans")
        if sufficient is not None and not isinstance(sufficient, bool):
            raise TypeError("task_sufficient_post must be boolean or null")
        if holding is not None and not isinstance(holding, bool):
            raise TypeError("holding_requested_target_post must be boolean or null")
        if region_empty is not None and not isinstance(region_empty, bool):
            raise TypeError("region_confirmed_empty_post must be boolean or null")
        if task_complete is not None and not isinstance(task_complete, bool):
            raise TypeError("task_complete_post must be boolean or null")
        observed_present = any(
            rle_decode(encoded).any()
            for encoded in decoded["post_interaction"].values()
        )
        if bool(present) != observed_present:
            raise ValueError("target_present_post disagrees with post masks")
        if teacher is not True:
            raise ValueError("binding masks must be declared simulator_teacher_only")
        sample_id = " ".join(str(value.get("sample_id", "")).split())
        group = " ".join(str(value.get("initial_state_group", "")).split())
        action = " ".join(str(value.get("executed_action", "")).split()).upper()
        if not sample_id or not group or not action:
            raise ValueError("sample, group, and executed action are required")
        return cls(
            sample_id=sample_id,
            initial_state_group=group,
            split=Split(str(value.get("split", ""))),
            target_masks=decoded,
            target_present_post=bool(present),
            task_sufficient_post=sufficient,
            holding_requested_target_post=holding,
            region_confirmed_empty_post=region_empty,
            task_complete_post=task_complete,
            executed_action=action,
            simulator_teacher_only=True,
        )


def load_binding_labels(path: Path) -> list[BindingLabel]:
    rows = [
        BindingLabel.from_mapping(json.loads(line))
        for line in path.read_text().splitlines()
        if line
    ]
    if not rows:
        raise ValueError(f"empty binding labels: {path}")
    identifiers = [row.sample_id for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate binding-label sample_id")
    groups: dict[str, Split] = {}
    for row in rows:
        previous = groups.setdefault(row.initial_state_group, row.split)
        if previous is not row.split:
            raise ValueError("binding-label initial-state group leakage")
    return rows


@dataclasses.dataclass(frozen=True)
class BindingInputs:
    """CPU-ready public inputs for label-free online binder inference."""

    sample_id: tuple[str, ...]
    initial_state_group: tuple[str, ...]
    split: tuple[Split, ...]
    image_tokens: np.ndarray
    image_valid_mask: np.ndarray
    prompt_tokens: np.ndarray
    prompt_valid_mask: np.ndarray
    patch_xy: np.ndarray
    camera_id: np.ndarray
    temporal_id: np.ndarray
    executed_action_id: np.ndarray


@dataclasses.dataclass(frozen=True)
class BindingArrays(BindingInputs):
    """Public inputs after a separate evaluator-label join."""

    patch_target: np.ndarray
    target_present: np.ndarray
    task_sufficient: np.ndarray
    task_sufficient_mask: np.ndarray
    holding_requested_target: np.ndarray
    holding_requested_target_mask: np.ndarray
    region_confirmed_empty: np.ndarray
    region_confirmed_empty_mask: np.ndarray
    task_complete: np.ndarray
    task_complete_mask: np.ndarray


def _camera_patch_targets(
    *,
    label: BindingLabel,
    layout: PrefixLayout,
    camera_to_label_view: Mapping[str, str | None],
) -> np.ndarray:
    values = []
    for time_step in TIME_STEPS:
        if time_step != "post_interaction":
            values.extend(
                np.zeros(count, dtype=np.float32) for count in layout.tokens_per_camera
            )
            continue
        time_masks = label.target_masks[time_step]
        for camera, count in zip(
            layout.camera_names, layout.tokens_per_camera, strict=True
        ):
            label_view = camera_to_label_view.get(camera)
            if label_view is None:
                values.append(np.zeros(count, dtype=np.float32))
                continue
            if label_view not in time_masks:
                raise ValueError(f"binding label lacks camera {label_view!r}")
            side = math.isqrt(count)
            if side * side != count:
                raise ValueError("binding supervision requires square patch grids")
            values.append(
                mask_to_patch_coverage(
                    rle_decode(time_masks[label_view]), grid_side=side
                )
            )
    return np.concatenate(values)


def build_binding_inputs(
    *,
    feature_arrays: Mapping[str, Any],
    feature_report: Mapping[str, Any],
    action_vocabulary: Sequence[str],
) -> BindingInputs:
    """Build deployment tensors using only the public frozen feature cache."""

    forbidden = {
        "patch_target",
        "target_present",
        "task_sufficient",
        "task_sufficient_mask",
        "holding_requested_target",
        "holding_requested_target_mask",
        "region_confirmed_empty",
        "region_confirmed_empty_mask",
        "task_complete",
        "task_complete_mask",
        "route_target",
        "effect_target",
        "effect_support_mask",
    }
    leaked = forbidden & set(feature_arrays)
    if leaked:
        raise ValueError(
            f"online binding inputs contain evaluator targets {sorted(leaked)}"
        )
    validate_feature_arrays(feature_arrays)
    if feature_report.get("schema_version") != "piu.spatial-prefix-features.v1":
        raise ValueError("unsupported spatial-prefix report")
    layout_value = feature_report["layout"]
    layout = PrefixLayout(
        tuple(layout_value["camera_names"]),
        tuple(int(value) for value in layout_value["tokens_per_camera"]),
    )
    image = np.asarray(feature_arrays["image_tokens"], dtype=np.float32)
    image_mask = np.asarray(feature_arrays["image_valid_mask"], dtype=bool)
    prompt = np.asarray(feature_arrays["prompt_tokens"], dtype=np.float32)
    prompt_mask = np.asarray(feature_arrays["prompt_valid_mask"], dtype=bool)
    if image.shape[2] != layout.total_image_tokens:
        raise ValueError("feature image-token count differs from the reported layout")
    expected_camera, expected_xy = layout.patch_metadata()
    if not np.array_equal(
        np.asarray(feature_arrays["camera_id"], dtype=np.int16), expected_camera
    ) or not np.allclose(
        np.asarray(feature_arrays["patch_xy"], dtype=np.float32), expected_xy
    ):
        raise ValueError("feature patch metadata differs from the reported layout")
    sample_ids = np.asarray(feature_arrays["sample_id"]).astype(str).tolist()
    groups = np.asarray(feature_arrays["initial_state_group"]).astype(str).tolist()
    splits = tuple(
        Split(value) for value in np.asarray(feature_arrays["split"]).astype(str)
    )
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("duplicate spatial-prefix sample_id")
    vocabulary = tuple(str(value).upper() for value in action_vocabulary)
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("action vocabulary contains duplicates")
    action_to_id = {action: index for index, action in enumerate(vocabulary)}
    executed = np.char.upper(
        np.asarray(feature_arrays["executed_action"]).astype(str)
    ).tolist()
    if "NO_HISTORY" in executed:
        raise ValueError(
            "NO_HISTORY is an ablation-only token, not a public executed action"
        )
    unknown = set(executed) - set(action_to_id)
    if unknown:
        raise ValueError(f"unknown public executed actions {sorted(unknown)}")
    count, time_count, image_count, width = image.shape
    prompt_count = prompt.shape[2]
    camera_id = np.tile(expected_camera.astype(np.int64), time_count)
    patch_xy = np.tile(expected_xy.astype(np.float32), (time_count, 1))
    temporal_id = np.repeat(np.arange(time_count, dtype=np.int64), image_count)
    return BindingInputs(
        sample_id=tuple(sample_ids),
        initial_state_group=tuple(groups),
        split=splits,
        image_tokens=image.reshape(count, time_count * image_count, width),
        image_valid_mask=image_mask.reshape(count, time_count * image_count),
        prompt_tokens=prompt.reshape(count, time_count * prompt_count, width),
        prompt_valid_mask=prompt_mask.reshape(count, time_count * prompt_count),
        patch_xy=np.broadcast_to(
            patch_xy[None], (count, time_count * image_count, 2)
        ).copy(),
        camera_id=np.broadcast_to(
            camera_id[None], (count, time_count * image_count)
        ).copy(),
        temporal_id=np.broadcast_to(
            temporal_id[None], (count, time_count * image_count)
        ).copy(),
        executed_action_id=np.asarray(
            [action_to_id[value] for value in executed], dtype=np.int64
        ),
    )


def join_binding_features(
    *,
    feature_arrays: Mapping[str, Any],
    feature_report: Mapping[str, Any],
    labels: Sequence[BindingLabel],
    action_vocabulary: Sequence[str],
) -> BindingArrays:
    """Join by sample ID and flatten only the explicit time/token axes."""

    inputs = build_binding_inputs(
        feature_arrays=feature_arrays,
        feature_report=feature_report,
        action_vocabulary=action_vocabulary,
    )
    layout_value = feature_report["layout"]
    layout = PrefixLayout(
        tuple(layout_value["camera_names"]),
        tuple(int(value) for value in layout_value["tokens_per_camera"]),
    )
    camera_mapping = dict(layout_value["camera_to_label_view"])
    by_id = {label.sample_id: label for label in labels}
    if set(inputs.sample_id) != set(by_id):
        raise ValueError("spatial features and binding-label sample IDs differ")
    vocabulary = tuple(str(value).upper() for value in action_vocabulary)
    id_to_action = {index: action for index, action in enumerate(vocabulary)}
    targets = []
    presence = []
    sufficiency = []
    sufficiency_mask = []
    holding = []
    holding_mask = []
    region_empty = []
    region_empty_mask = []
    task_complete = []
    task_complete_mask = []
    for index, (sample_id, group, split) in enumerate(
        zip(
            inputs.sample_id,
            inputs.initial_state_group,
            inputs.split,
            strict=True,
        )
    ):
        label = by_id[sample_id]
        if label.initial_state_group != group or label.split is not split:
            raise ValueError("feature/binding-label group or split mismatch")
        if label.executed_action != id_to_action[inputs.executed_action_id[index]]:
            raise ValueError("public executed action and evaluator label differ")
        targets.append(
            _camera_patch_targets(
                label=label,
                layout=layout,
                camera_to_label_view=camera_mapping,
            )
        )
        presence.append(label.target_present_post)
        sufficiency.append(bool(label.task_sufficient_post))
        sufficiency_mask.append(label.task_sufficient_post is not None)
        holding.append(bool(label.holding_requested_target_post))
        holding_mask.append(label.holding_requested_target_post is not None)
        region_empty.append(bool(label.region_confirmed_empty_post))
        region_empty_mask.append(label.region_confirmed_empty_post is not None)
        task_complete.append(bool(label.task_complete_post))
        task_complete_mask.append(label.task_complete_post is not None)
    patch_target = np.stack(targets).astype(np.float32)
    if np.any((patch_target > 0) & ~inputs.image_valid_mask):
        raise ValueError("binding target occupies an invalid frozen-prefix token")
    return BindingArrays(
        **{
            field.name: getattr(inputs, field.name)
            for field in dataclasses.fields(BindingInputs)
        },
        patch_target=patch_target,
        target_present=np.asarray(presence, dtype=np.float32),
        task_sufficient=np.asarray(sufficiency, dtype=np.float32),
        task_sufficient_mask=np.asarray(sufficiency_mask, dtype=bool),
        holding_requested_target=np.asarray(holding, dtype=np.float32),
        holding_requested_target_mask=np.asarray(holding_mask, dtype=bool),
        region_confirmed_empty=np.asarray(region_empty, dtype=np.float32),
        region_confirmed_empty_mask=np.asarray(region_empty_mask, dtype=bool),
        task_complete=np.asarray(task_complete, dtype=np.float32),
        task_complete_mask=np.asarray(task_complete_mask, dtype=bool),
    )
