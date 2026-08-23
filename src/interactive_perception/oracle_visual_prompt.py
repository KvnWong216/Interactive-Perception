"""Evaluator-only target prompts for testing a frozen policy's binding ceiling.

This module is deliberately named ``oracle``.  It reads simulator instance
segmentation and therefore cannot be used by the public-input method.  Its only
purpose is a qualification experiment: if a visible target box still cannot
make the frozen policy grasp the requested object, learning a public RGB box
predictor cannot repair the executor and the project must change primitives.

Only the rendered RGB marker reaches :class:`ObservationPacket`; the integer
mask and target instance id are never serialized to the policy server.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from .policy_client import ObservationPacket, build_observation

__all__ = [
    "OraclePromptDiagnostics",
    "OraclePromptResult",
    "PromptBox",
    "VisualPromptStyle",
    "build_oracle_target_packet",
    "find_instance_segmentation",
    "render_visual_prompt",
    "target_box",
]

VisualPromptStyle = Literal["box", "point", "spotlight"]
_VALID_STYLES = frozenset(("box", "point", "spotlight"))


@dataclasses.dataclass(frozen=True)
class PromptBox:
    """Half-open target rectangle in raw camera pixel coordinates."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def center(self) -> tuple[int, int]:
        return ((self.left + self.right - 1) // 2, (self.top + self.bottom - 1) // 2)

    def as_list(self) -> list[int]:
        return [self.left, self.top, self.right, self.bottom]


@dataclasses.dataclass(frozen=True)
class OraclePromptDiagnostics:
    """Audit data retained by the evaluator, never sent to the policy."""

    style: VisualPromptStyle
    target_instance_id: int
    visible_pixels: Mapping[str, int]
    boxes: Mapping[str, PromptBox | None]
    changed_pixels: Mapping[str, int]

    def to_json(self) -> dict[str, Any]:
        return {
            "style": self.style,
            "target_instance_id": self.target_instance_id,
            "visible_pixels": dict(self.visible_pixels),
            "boxes_xyxy_half_open": {
                camera: box.as_list() if box is not None else None
                for camera, box in self.boxes.items()
            },
            "changed_rgb_pixels": dict(self.changed_pixels),
        }


@dataclasses.dataclass(frozen=True)
class OraclePromptResult:
    packet: ObservationPacket
    diagnostics: OraclePromptDiagnostics


def find_instance_segmentation(
    observation: Mapping[str, Any], camera: str
) -> np.ndarray:
    """Return the sole instance-segmentation image for ``camera``."""

    keys = [
        key for key in observation if key.startswith(camera) and "segmentation" in key
    ]
    if len(keys) != 1:
        raise KeyError(f"expected one {camera} segmentation image, got {keys}")
    segmentation = np.asarray(observation[keys[0]]).squeeze()
    if segmentation.ndim != 2:
        raise ValueError(
            f"{keys[0]} must reduce to a two-dimensional image, got "
            f"{segmentation.shape}"
        )
    return segmentation


def target_box(mask: np.ndarray, *, padding: int = 4) -> PromptBox | None:
    """Compute a padded, clipped half-open rectangle around a binary mask."""

    if mask.ndim != 2:
        raise ValueError(f"mask must be two-dimensional, got {mask.shape}")
    if padding < 0:
        raise ValueError("padding must be nonnegative")
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return None
    height, width = mask.shape
    return PromptBox(
        left=max(0, int(columns.min()) - padding),
        top=max(0, int(rows.min()) - padding),
        right=min(width, int(columns.max()) + 1 + padding),
        bottom=min(height, int(rows.max()) + 1 + padding),
    )


def _marker_color(image: np.ndarray) -> np.ndarray:
    if np.issubdtype(image.dtype, np.floating):
        return np.asarray((1.0, 0.0, 1.0), dtype=image.dtype)
    return np.asarray((255, 0, 255), dtype=image.dtype)


def _draw_box(
    image: np.ndarray, box: PromptBox, *, color: np.ndarray, line_width: int
) -> None:
    top = box.top
    bottom = box.bottom
    left = box.left
    right = box.right
    image[top : min(bottom, top + line_width), left:right] = color
    image[max(top, bottom - line_width) : bottom, left:right] = color
    image[top:bottom, left : min(right, left + line_width)] = color
    image[top:bottom, max(left, right - line_width) : right] = color


def render_visual_prompt(
    image: np.ndarray,
    box: PromptBox | None,
    *,
    style: VisualPromptStyle,
    line_width: int = 4,
) -> np.ndarray:
    """Render one visual prompt without changing image shape or dtype."""

    if style not in _VALID_STYLES:
        raise ValueError(f"unknown visual prompt style {style!r}")
    if line_width < 1:
        raise ValueError("line_width must be positive")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")
    result = np.asarray(image).copy()
    if box is None:
        return result
    height, width = result.shape[:2]
    if not (0 <= box.left < box.right <= width and 0 <= box.top < box.bottom <= height):
        raise ValueError(f"box {box} lies outside image shape {result.shape}")
    color = _marker_color(result)
    if style == "spotlight":
        outside = np.ones((height, width), dtype=bool)
        outside[box.top : box.bottom, box.left : box.right] = False
        if np.issubdtype(result.dtype, np.integer):
            result[outside] = np.rint(result[outside].astype(float) * 0.25).astype(
                result.dtype
            )
        else:
            result[outside] *= 0.25
        _draw_box(result, box, color=color, line_width=line_width)
    elif style == "box":
        _draw_box(result, box, color=color, line_width=line_width)
    else:
        center_x, center_y = box.center
        radius = max(6, 2 * line_width)
        result[
            max(0, center_y - line_width // 2) : min(
                height, center_y + (line_width + 1) // 2
            ),
            max(0, center_x - radius) : min(width, center_x + radius + 1),
        ] = color
        result[
            max(0, center_y - radius) : min(height, center_y + radius + 1),
            max(0, center_x - line_width // 2) : min(
                width, center_x + (line_width + 1) // 2
            ),
        ] = color
    return result


def build_oracle_target_packet(
    observation: Mapping[str, Any],
    prompt: str,
    *,
    target_instance_id: int,
    style: VisualPromptStyle,
    primary_camera: str = "agentview",
    wrist_camera: str = "robot0_eye_in_hand",
) -> OraclePromptResult:
    """Render evaluator masks into RGB and build the stock policy packet."""

    if style not in _VALID_STYLES:
        raise ValueError(f"unknown visual prompt style {style!r}")
    prompted_images: dict[str, np.ndarray] = {}
    visible_pixels: dict[str, int] = {}
    boxes: dict[str, PromptBox | None] = {}
    changed_pixels: dict[str, int] = {}
    for camera in (primary_camera, wrist_camera):
        image_key = f"{camera}_image"
        image = np.asarray(observation[image_key])
        segmentation = find_instance_segmentation(observation, camera)
        if segmentation.shape != image.shape[:2]:
            raise ValueError(
                f"{camera} image/segmentation mismatch: {image.shape[:2]} vs "
                f"{segmentation.shape}"
            )
        mask = segmentation == target_instance_id
        box = target_box(mask)
        visible_pixels[camera] = int(np.count_nonzero(mask))
        boxes[camera] = box
        prompted = render_visual_prompt(image, box, style=style)
        prompted_images[image_key] = prompted
        changed_pixels[camera] = int(
            np.count_nonzero(np.any(prompted != image, axis=-1))
        )

    # Deliberately reconstruct only the stock public observation fields.  This
    # is the hard boundary that prevents masks or object ids from entering the
    # websocket payload after their evaluator-only RGB rendering role.
    public_prompted_observation = {
        **prompted_images,
        "robot0_eef_pos": observation["robot0_eef_pos"],
        "robot0_eef_quat": observation["robot0_eef_quat"],
        "robot0_gripper_qpos": observation["robot0_gripper_qpos"],
    }
    return OraclePromptResult(
        packet=build_observation(
            public_prompted_observation,
            prompt,
            primary_camera=primary_camera,
            wrist_camera=wrist_camera,
        ),
        diagnostics=OraclePromptDiagnostics(
            style=style,
            target_instance_id=int(target_instance_id),
            visible_pixels=visible_pixels,
            boxes=boxes,
            changed_pixels=changed_pixels,
        ),
    )
