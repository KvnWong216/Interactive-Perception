"""Array/layout contract for full frozen PaliGemma prefix features."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class PrefixLayout:
    """Exact token spans discovered from a particular OpenPI checkpoint."""

    camera_names: tuple[str, ...]
    tokens_per_camera: tuple[int, ...]

    @classmethod
    def from_counts(cls, counts: Mapping[str, int]) -> PrefixLayout:
        if not counts:
            raise ValueError("at least one camera token span is required")
        names = tuple(str(name) for name in counts)
        values = tuple(int(counts[name]) for name in counts)
        if len(set(names)) != len(names) or any(count < 1 for count in values):
            raise ValueError("camera names must be unique and token counts positive")
        return cls(names, values)

    @property
    def total_image_tokens(self) -> int:
        return sum(self.tokens_per_camera)

    def spans(self) -> dict[str, tuple[int, int]]:
        result: dict[str, tuple[int, int]] = {}
        start = 0
        for name, count in zip(
            self.camera_names, self.tokens_per_camera, strict=True
        ):
            result[name] = (start, start + count)
            start += count
        return result

    def patch_metadata(self) -> tuple[np.ndarray, np.ndarray]:
        """Return camera IDs and normalized row-major patch centers.

        OpenPI's frozen SigLIP encoder flattens a square patch grid in row-major
        order. A non-square count is rejected instead of guessing coordinates.
        """

        camera_ids: list[int] = []
        coordinates: list[tuple[float, float]] = []
        for camera_id, count in enumerate(self.tokens_per_camera):
            side = math.isqrt(count)
            if side * side != count:
                raise ValueError(
                    f"camera {self.camera_names[camera_id]!r} has non-square "
                    f"token count {count}"
                )
            camera_ids.extend([camera_id] * count)
            coordinates.extend(
                ((column + 0.5) / side, (row + 0.5) / side)
                for row in range(side)
                for column in range(side)
            )
        return np.asarray(camera_ids, dtype=np.int16), np.asarray(
            coordinates, dtype=np.float32
        )


def validate_feature_arrays(arrays: Mapping[str, Any]) -> None:
    """Validate a cached temporal prefix tensor without loading a model."""

    required = {
        "image_tokens",
        "image_valid_mask",
        "prompt_tokens",
        "prompt_valid_mask",
        "patch_xy",
        "camera_id",
        "sample_id",
        "initial_state_group",
        "split",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"spatial prefix cache missing {sorted(missing)}")
    image = np.asarray(arrays["image_tokens"])
    image_mask = np.asarray(arrays["image_valid_mask"])
    prompt = np.asarray(arrays["prompt_tokens"])
    prompt_mask = np.asarray(arrays["prompt_valid_mask"])
    patch_xy = np.asarray(arrays["patch_xy"])
    camera_id = np.asarray(arrays["camera_id"])
    if image.ndim != 4 or prompt.ndim != 4:
        raise ValueError("image and prompt tokens must have shape [N,time,tokens,width]")
    if image.shape[:3] != image_mask.shape:
        raise ValueError("image_valid_mask shape mismatch")
    if prompt.shape[:3] != prompt_mask.shape:
        raise ValueError("prompt_valid_mask shape mismatch")
    if image.shape[:2] != prompt.shape[:2] or image.shape[-1] != prompt.shape[-1]:
        raise ValueError("image/prompt batch, time, and width must agree")
    if patch_xy.shape != (image.shape[2], 2):
        raise ValueError("patch_xy must have shape [image_tokens,2]")
    if camera_id.shape != (image.shape[2],):
        raise ValueError("camera_id must have shape [image_tokens]")
    if not np.isfinite(image).all() or not np.isfinite(prompt).all():
        raise ValueError("prefix tensors contain non-finite values")
    if not np.isfinite(patch_xy).all() or np.any((patch_xy < 0) | (patch_xy > 1)):
        raise ValueError("patch coordinates must be finite and normalized")
    count = image.shape[0]
    for name in ("sample_id", "initial_state_group", "split"):
        if np.asarray(arrays[name]).shape != (count,):
            raise ValueError(f"{name} must have shape [N]")


def libero_camera_to_label_view(camera_names: Sequence[str]) -> dict[str, str | None]:
    """Map the official pi05 LIBERO image keys to evaluator camera names."""

    contract = {
        "base_0_rgb": "agentview",
        "left_wrist_0_rgb": "robot0_eye_in_hand",
        "right_wrist_0_rgb": None,
    }
    unknown = set(camera_names) - set(contract)
    if unknown:
        raise ValueError(f"unknown pi05 LIBERO camera keys: {sorted(unknown)}")
    return {name: contract[name] for name in camera_names}
