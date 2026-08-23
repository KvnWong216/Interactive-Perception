"""Array/layout contract for full frozen PaliGemma prefix features."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .contracts import assert_public_policy_value


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
        "executed_action",
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
    for name in ("sample_id", "initial_state_group", "split", "executed_action"):
        if np.asarray(arrays[name]).shape != (count,):
            raise ValueError(f"{name} must have shape [N]")
    executed = np.char.upper(np.asarray(arrays["executed_action"]).astype(str))
    if any(not " ".join(value.split()) for value in executed.tolist()):
        raise ValueError("executed_action must contain public semantic primitives")
    candidate_names = {
        "candidate_prompt_tokens",
        "candidate_prompt_valid_mask",
        "candidate_valid_mask",
        "candidate_id",
        "candidate_primitive",
        "candidate_payload",
    }
    present_candidate_names = candidate_names & set(arrays)
    if present_candidate_names and present_candidate_names != candidate_names:
        raise ValueError(
            "candidate-prefix arrays must be all present or all absent"
        )
    if present_candidate_names:
        candidate = np.asarray(arrays["candidate_prompt_tokens"])
        candidate_mask = np.asarray(arrays["candidate_prompt_valid_mask"])
        candidate_valid = np.asarray(arrays["candidate_valid_mask"])
        if candidate.ndim != 5:
            raise ValueError(
                "candidate_prompt_tokens must have shape [N,candidate,time,tokens,width]"
            )
        if candidate.shape[:1] != (count,) or candidate.shape[2] != image.shape[1]:
            raise ValueError("candidate prefix batch/time shape mismatch")
        if candidate.shape[-1] != image.shape[-1]:
            raise ValueError("candidate/task prefix widths differ")
        if candidate_mask.shape != candidate.shape[:-1]:
            raise ValueError("candidate_prompt_valid_mask shape mismatch")
        if candidate_valid.shape != candidate.shape[:2]:
            raise ValueError("candidate_valid_mask shape mismatch")
        for name in ("candidate_id", "candidate_primitive", "candidate_payload"):
            if np.asarray(arrays[name]).shape != candidate.shape[:2]:
                raise ValueError(f"{name} shape mismatch")
        padded = ~candidate_valid.astype(bool)
        if candidate_mask[padded].any():
            raise ValueError("padded candidates contain valid prompt tokens")
        if not np.isfinite(candidate).all():
            raise ValueError("candidate prefix tensor contains non-finite values")


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


def candidate_conditioned_prompt(
    task_prompt: str, candidate: Mapping[str, Any]
) -> str:
    """Serialize only public candidate fields into the frozen text interface."""

    assert_public_policy_value(candidate, path="spatial_prefix.candidate")
    task = " ".join(str(task_prompt).split())
    candidate_id = " ".join(str(candidate.get("candidate_id", "")).split())
    primitive = " ".join(str(candidate.get("primitive", "")).split()).upper()
    target = " ".join(str(candidate.get("target", "")).split())
    if not task or not candidate_id or not primitive or not target:
        raise ValueError("task and candidate identity/primitive/target are required")
    lines = [
        f"Task: {task}",
        f"Candidate ID: {candidate_id}",
        f"Candidate primitive: {primitive}",
        f"Candidate target: {target}",
    ]
    for key, label in (("reference", "Reference"), ("purpose", "Purpose")):
        value = " ".join(str(candidate.get(key, "") or "").split())
        if value:
            lines.append(f"{label}: {value}")
    handled = {"candidate_id", "primitive", "target", "reference", "purpose"}
    for key in sorted(set(candidate) - handled):
        value = json.dumps(
            candidate[key], sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        lines.append(f"Candidate public field {key}: {value}")
    lines.append("Predict the task-relevant physical effect of executing this candidate.")
    return "\n".join(lines)
