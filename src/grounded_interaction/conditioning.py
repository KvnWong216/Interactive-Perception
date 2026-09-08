"""Concrete public-image conditioning for the Stage-2 interface experiment."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from .contracts import GroundedIntervention, PolicyContext, canonical_sha256
from .rgb import RGBFrameStore, canonical_rgb_array, canonical_rgb_sha256
from .serialization import SerializedSubtask, SpatialConditioningMode


def _latest_frames_by_camera(context: PolicyContext) -> dict[str, object]:
    latest: dict[str, object] = {}
    for frame in context.frames:
        current = latest.get(frame.camera)
        if current is None or frame.frame_index > current.frame_index:  # type: ignore[attr-defined]
            latest[frame.camera] = frame
    return latest


def draw_public_grounding_marker(
    rgb: Any,
    *,
    box_xyxy: tuple[float, float, float, float],
    point_xy: tuple[float, float],
) -> Any:
    """Draw a fixed magenta box/cross from a public manual annotation."""

    array = canonical_rgb_array(rgb)
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError as error:  # pragma: no cover - optional extra.
        raise RuntimeError(
            "visual-marker conditioning requires Pillow and numpy"
        ) from error

    height, width = array.shape[:2]
    x0, y0, x1, y1 = box_xyxy
    x, y = point_xy
    pixels = (
        round(x0 * (width - 1)),
        round(y0 * (height - 1)),
        round(x1 * (width - 1)),
        round(y1 * (height - 1)),
    )
    center = (round(x * (width - 1)), round(y * (height - 1)))
    stroke = max(3, min(height, width) // 64)
    radius = max(7, min(height, width) // 24)
    image = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(image)
    color = (255, 0, 255)
    draw.rectangle(pixels, outline=color, width=stroke)
    draw.line(
        (center[0] - radius, center[1], center[0] + radius, center[1]),
        fill=color,
        width=stroke,
    )
    draw.line(
        (center[0], center[1] - radius, center[0], center[1] + radius),
        fill=color,
        width=stroke,
    )
    return canonical_rgb_array(np.asarray(image, dtype=np.uint8))


@dataclasses.dataclass(frozen=True)
class ConditionedStage2Input:
    """Exactly the public RGB/state/text payload sent to MolmoAct2."""

    instruction: str
    agentview_rgb: Any
    wrist_rgb: Any
    proprioception: tuple[float, ...]
    conditioning_mode: SpatialConditioningMode
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        instruction = " ".join(str(self.instruction or "").split())
        if not instruction:
            raise ValueError("instruction must be non-empty")
        object.__setattr__(self, "instruction", instruction)
        object.__setattr__(
            self, "agentview_rgb", canonical_rgb_array(self.agentview_rgb)
        )
        object.__setattr__(self, "wrist_rgb", canonical_rgb_array(self.wrist_rgb))
        values = tuple(float(item) for item in self.proprioception)
        if len(values) != 8:
            raise ValueError("MolmoAct2-LIBERO proprioception must contain 8 values")
        object.__setattr__(self, "proprioception", values)
        if not isinstance(self.conditioning_mode, SpatialConditioningMode):
            raise TypeError("conditioning_mode must be a SpatialConditioningMode")
        detached = dict(self.provenance)
        canonical_sha256(detached)
        object.__setattr__(self, "provenance", MappingProxyType(detached))

    def public_identity(self) -> dict[str, object]:
        return {
            "instruction": self.instruction,
            "agentview_rgb_sha256": canonical_rgb_sha256(self.agentview_rgb),
            "wrist_rgb_sha256": canonical_rgb_sha256(self.wrist_rgb),
            "proprioception": list(self.proprioception),
            "conditioning_mode": self.conditioning_mode.value,
            "provenance": dict(self.provenance),
        }


class ReferentConditioner:
    """Resolve real RGB and apply only the preregistered E1 conditioning mode."""

    def __init__(self, frame_store: RGBFrameStore) -> None:
        if not isinstance(frame_store, RGBFrameStore):
            raise TypeError("frame_store must be an RGBFrameStore")
        self.frame_store = frame_store

    def prepare(
        self,
        *,
        context: PolicyContext,
        intervention: GroundedIntervention,
        serialized: SerializedSubtask,
        mode: SpatialConditioningMode,
    ) -> ConditionedStage2Input:
        if not isinstance(context, PolicyContext):
            raise TypeError("context must be a PolicyContext")
        if not isinstance(intervention, GroundedIntervention):
            raise TypeError("intervention must be a GroundedIntervention")
        if not isinstance(serialized, SerializedSubtask):
            raise TypeError("serialized must be a SerializedSubtask")
        if not isinstance(mode, SpatialConditioningMode):
            raise TypeError("mode must be a SpatialConditioningMode")
        intervention.validate_against(context)
        if serialized.candidate_fingerprint != intervention.fingerprint():
            raise ValueError("serialized subtask does not match intervention")
        if serialized.context_fingerprint != context.fingerprint():
            raise ValueError("serialized subtask does not match context")
        if (
            serialized.spatial_audit_payload.get("spatial_conditioning_mode")
            != mode.value
        ):
            raise ValueError("serializer and image conditioner modes disagree")
        if context.proprioception is None:
            raise ValueError("MolmoAct2-LIBERO requires public proprioception")

        latest = _latest_frames_by_camera(context)
        try:
            agentview_frame = latest["agentview"]
            wrist_frame = latest["wrist"]
        except KeyError as error:
            raise ValueError(
                "MolmoAct2-LIBERO requires agentview and wrist frames"
            ) from error
        agentview = self.frame_store.resolve(agentview_frame)  # type: ignore[arg-type]
        wrist = self.frame_store.resolve(wrist_frame)  # type: ignore[arg-type]
        source_hashes = {
            "agentview": canonical_rgb_sha256(agentview),
            "wrist": canonical_rgb_sha256(wrist),
        }

        marker_applied = False
        grounding = intervention.grounding
        if mode is SpatialConditioningMode.VISUAL_MARKER:
            if grounding is None:
                raise ValueError("visual marker requires physical grounding")
            if grounding.camera == "agentview":
                agentview = draw_public_grounding_marker(
                    agentview,
                    box_xyxy=grounding.box_xyxy,
                    point_xy=grounding.point_xy,
                )
            elif grounding.camera == "wrist":
                wrist = draw_public_grounding_marker(
                    wrist,
                    box_xyxy=grounding.box_xyxy,
                    point_xy=grounding.point_xy,
                )
            else:
                raise ValueError("visual marker camera must be agentview or wrist")
            marker_applied = True

        provenance = {
            "source_frame_sha256": source_hashes,
            "executed_frame_sha256": {
                "agentview": canonical_rgb_sha256(agentview),
                "wrist": canonical_rgb_sha256(wrist),
            },
            "manual_public_rgb_grounding": grounding is not None,
            "visual_marker_applied": marker_applied,
            "native_vla_spatial_api_used": False,
        }
        return ConditionedStage2Input(
            instruction=serialized.subtask_text,
            agentview_rgb=agentview,
            wrist_rgb=wrist,
            proprioception=context.proprioception,
            conditioning_mode=mode,
            provenance=provenance,
        )
