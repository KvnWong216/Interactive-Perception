"""Deterministic, confidence-free PIU subtask bridge for frozen pi0.5."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .contracts import assert_public_policy_value


@dataclasses.dataclass(frozen=True)
class SpatialReference:
    """Exact enclosure of calibrated current-frame patches in one camera."""

    camera: str
    selected_patch_indices: tuple[int, ...]
    x_interval: tuple[float, float]
    y_interval: tuple[float, float]


def current_spatial_references(
    *,
    prediction_set: np.ndarray,
    patch_xy: np.ndarray,
    camera_id: np.ndarray,
    temporal_id: np.ndarray,
    camera_names: Sequence[str],
) -> tuple[SpatialReference, ...]:
    """Summarize every selected latest-frame patch with no score threshold."""

    selected = np.asarray(prediction_set, dtype=bool)
    xy = np.asarray(patch_xy, dtype=np.float64)
    cameras = np.asarray(camera_id, dtype=np.int64)
    times = np.asarray(temporal_id, dtype=np.int64)
    if selected.ndim != 1 or xy.shape != (len(selected), 2):
        raise ValueError("spatial reference patch arrays differ")
    if cameras.shape != selected.shape or times.shape != selected.shape:
        raise ValueError("spatial reference metadata shapes differ")
    if not len(camera_names) or np.any(cameras < 0) or np.any(cameras >= len(camera_names)):
        raise ValueError("spatial reference camera IDs are invalid")
    current = times == times.max()
    references = []
    for identifier, name in enumerate(camera_names):
        indices = np.flatnonzero(selected & current & (cameras == identifier))
        if not len(indices):
            continue
        centers = xy[indices]
        unique_x = np.unique(xy[current & (cameras == identifier), 0])
        unique_y = np.unique(xy[current & (cameras == identifier), 1])
        if not len(unique_x) or not len(unique_y):
            raise ValueError("camera has no current-frame patch coordinates")
        half_width = 0.5 / len(unique_x)
        half_height = 0.5 / len(unique_y)
        references.append(
            SpatialReference(
                camera=str(name),
                selected_patch_indices=tuple(int(index) for index in indices),
                x_interval=(
                    max(0.0, float(centers[:, 0].min() - half_width)),
                    min(1.0, float(centers[:, 0].max() + half_width)),
                ),
                y_interval=(
                    max(0.0, float(centers[:, 1].min() - half_height)),
                    min(1.0, float(centers[:, 1].max() + half_height)),
                ),
            )
        )
    return tuple(references)


def load_public_candidate(payload: str) -> dict[str, Any]:
    value = json.loads(payload)
    if not isinstance(value, Mapping):
        raise TypeError("candidate payload must be a mapping")
    result = dict(value)
    assert_public_policy_value(result, path="executor_bridge.candidate")
    for name in ("candidate_id", "primitive", "target"):
        if not " ".join(str(result.get(name, "")).split()):
            raise ValueError(f"candidate payload lacks {name}")
    return result


def _reference_text(references: Sequence[SpatialReference]) -> str:
    return "; ".join(
        f"{item.camera} normalized box x=[{item.x_interval[0]:.4f},{item.x_interval[1]:.4f}], "
        f"y=[{item.y_interval[0]:.4f},{item.y_interval[1]:.4f}]"
        for item in references
    )


def _referent(value: str) -> str:
    if value.lower().startswith(("the ", "a ", "an ", "this ", "that ")):
        return value
    return f"the {value}"


def serialize_pi05_subtask(
    candidate: Mapping[str, Any], *, spatial_references: Sequence[SpatialReference]
) -> str | None:
    """Serialize only action, referent, and learned spatial geometry."""

    assert_public_policy_value(candidate, path="executor_bridge.candidate")
    primitive = " ".join(str(candidate.get("primitive", "")).split()).upper()
    target = " ".join(str(candidate.get("target", "")).split())
    reference = " ".join(str(candidate.get("reference", "") or "").split())
    if not primitive or not target:
        raise ValueError("executor candidate primitive and target are required")
    if primitive in {"STOP", "REPORT_NOT_FOUND"}:
        return None
    spatial = _reference_text(spatial_references)
    target_phrase = _referent(target)
    if primitive in {"PICK", "DIRECT"}:
        location = f" at {spatial}" if spatial else ""
        if primitive == "DIRECT" and reference:
            return f"Place {target_phrase}{location} in {_referent(reference)}."
        return f"Pick up {target_phrase}{location}."
    if primitive == "PLACE":
        if not reference:
            raise ValueError("PLACE requires a public destination reference")
        return f"Place {target_phrase} in {_referent(reference)}."
    if primitive == "OPEN":
        return f"Open {target_phrase}."
    if primitive == "REMOVE":
        return f"Remove {target_phrase}."
    if primitive == "ROTATE":
        return f"Rotate {target_phrase} to inspect it."
    if primitive == "MOVE_CLOSER":
        return f"Move {target_phrase} closer for inspection."
    if primitive == "PICK_TO_INSPECT":
        return f"Pick up {target_phrase} for inspection."
    raise ValueError(f"no frozen pi0.5 serializer for primitive {primitive!r}")
