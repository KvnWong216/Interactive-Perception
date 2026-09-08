"""Concrete frozen Qwen2.5-VL proposal and token provider for Method V1.

The provider owns two uses of one immutable checkpoint:

* greedy, schema-constrained candidate proposal from current public RGB; and
* detached contextual token extraction for the existing grounded outcome model.

The language model is loaded lazily and never shares a gradient graph with the
trainable scorer.  Static-image token locations are reconstructed from
``image_grid_thw`` and ``spatial_merge_size`` exactly; no square-grid or
``sqrt(token_count)`` assumption is used.  Image-token hidden states are
contextual, so their boxes are *spatial anchors*, not claims of local-only
receptive fields.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import (
    GroundedIntervention,
    GroundingReference,
    PolicyContext,
    Primitive,
    PublicFrame,
    canonical_json_bytes,
    canonical_sha256,
)
from .proposals import (
    METHOD_V1_ENABLED_PRIMITIVES,
    METHOD_V1_MAX_CANDIDATES,
    METHOD_V1_MAX_PER_PRIMITIVE,
    QWEN_COORDINATE_CONTRACT,
    QWEN_DIRECT_ONLY_SYSTEM_PROMPT,
    QWEN_PROPOSAL_SYSTEM_PROMPT,
    proposal_request_text,
    qwen_proposal_prompt_contract,
)
from .tokens import CandidateTokenField, FrozenTokenField

QWEN25VL_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN25VL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
QWEN25VL_TRANSFORMERS_VERSION = "4.57.6"
QWEN25VL_TARGET_IMAGE_TOKENS = 256
QWEN_CONTEXT_TEMPLATE_VERSION = "method-v1-public-history-v1"
QWEN_CANDIDATE_TEMPLATE_VERSION = "method-v1-candidate-last-content-v1"
QWEN_PATCH_MAPPING_VERSION = "qwen2.5-vl-grid-thw-row-major-merged-v1"
QWEN_TORCH_DTYPE = "bfloat16"
QWEN_QUANTIZATION_MODE = "none"
QWEN_LOAD_MODE = "transformers-from-pretrained-low-cpu-mem-v1"
QWEN_CONTEXT_SYSTEM_PROMPT = (
    "Encode only the supplied public task, action history, and RGB "
    "observations for grounded robot outcome prediction."
)
QWEN_CANDIDATE_PREFIX = "Candidate action instruction:\n"
METHOD_V1_PRIMITIVE_ORDER = (Primitive.DIRECT, Primitive.OPEN)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional dependency.
        raise RuntimeError(
            "Qwen token extraction requires PyTorch; use the isolated Method-V1 "
            "Qwen environment"
        ) from error
    return torch


def _installed_transformers_version() -> str:
    try:
        return importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


@dataclasses.dataclass(frozen=True)
class QwenProviderIdentity:
    """Complete identity of the frozen feature/proposal boundary."""

    model_id: str = QWEN25VL_MODEL_ID
    revision: str = QWEN25VL_REVISION
    transformers_version: str = QWEN25VL_TRANSFORMERS_VERSION
    target_image_tokens: int = QWEN25VL_TARGET_IMAGE_TOKENS
    processor_use_fast: bool = False
    attention_implementation: str = "sdpa"
    torch_dtype: str = QWEN_TORCH_DTYPE
    quantization_mode: str = QWEN_QUANTIZATION_MODE
    load_mode: str = QWEN_LOAD_MODE
    coordinate_contract: str = QWEN_COORDINATE_CONTRACT
    context_template: str = QWEN_CONTEXT_TEMPLATE_VERSION
    candidate_template: str = QWEN_CANDIDATE_TEMPLATE_VERSION
    patch_mapping: str = QWEN_PATCH_MAPPING_VERSION
    proposal_prompt_sha256: str = dataclasses.field(
        default_factory=lambda: _text_sha256(QWEN_PROPOSAL_SYSTEM_PROMPT)
    )
    direct_only_proposal_prompt_sha256: str = dataclasses.field(
        default_factory=lambda: _text_sha256(QWEN_DIRECT_ONLY_SYSTEM_PROMPT)
    )
    context_prompt_sha256: str = dataclasses.field(
        default_factory=lambda: _text_sha256(QWEN_CONTEXT_SYSTEM_PROMPT)
    )
    candidate_prefix_sha256: str = dataclasses.field(
        default_factory=lambda: _text_sha256(QWEN_CANDIDATE_PREFIX)
    )
    primitive_order: tuple[str, ...] = tuple(
        primitive.value for primitive in METHOD_V1_PRIMITIVE_ORDER
    )

    def __post_init__(self) -> None:
        if self.model_id != QWEN25VL_MODEL_ID:
            raise ValueError("Method V1 requires Qwen/Qwen2.5-VL-3B-Instruct")
        if self.revision != QWEN25VL_REVISION:
            raise ValueError("Method V1 requires the frozen Qwen checkpoint revision")
        if self.transformers_version != QWEN25VL_TRANSFORMERS_VERSION:
            raise ValueError("Method V1 requires the frozen Transformers version")
        if self.target_image_tokens < 1:
            raise ValueError("target_image_tokens must be positive")
        if self.processor_use_fast is not False:
            raise ValueError("Method V1 freezes the audited slow Qwen processor")
        if self.attention_implementation != "sdpa":
            raise ValueError("Method V1 freezes Qwen attention to SDPA")
        if self.torch_dtype != QWEN_TORCH_DTYPE:
            raise ValueError("Method V1 freezes Qwen weights to torch.bfloat16")
        if self.quantization_mode != QWEN_QUANTIZATION_MODE:
            raise ValueError("Method V1 does not permit quantized Qwen weights")
        if self.load_mode != QWEN_LOAD_MODE:
            raise ValueError("Method V1 Qwen load mode changed")
        if self.coordinate_contract != QWEN_COORDINATE_CONTRACT:
            raise ValueError("Qwen proposal coordinate contract changed")
        expected_text_hashes = {
            "proposal_prompt_sha256": _text_sha256(QWEN_PROPOSAL_SYSTEM_PROMPT),
            "direct_only_proposal_prompt_sha256": _text_sha256(
                QWEN_DIRECT_ONLY_SYSTEM_PROMPT
            ),
            "context_prompt_sha256": _text_sha256(QWEN_CONTEXT_SYSTEM_PROMPT),
            "candidate_prefix_sha256": _text_sha256(QWEN_CANDIDATE_PREFIX),
        }
        for name, expected in expected_text_hashes.items():
            if getattr(self, name) != expected:
                raise ValueError(f"Qwen frozen text changed: {name}")
        if self.primitive_order != tuple(
            primitive.value for primitive in METHOD_V1_PRIMITIVE_ORDER
        ):
            raise ValueError(
                "primitive order differs from the persisted Method-V1 order"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "transformers_version": self.transformers_version,
            "target_image_tokens": self.target_image_tokens,
            "processor_use_fast": self.processor_use_fast,
            "attention_implementation": self.attention_implementation,
            "torch_dtype": self.torch_dtype,
            "quantization_mode": self.quantization_mode,
            "load_mode": self.load_mode,
            "coordinate_contract": self.coordinate_contract,
            "context_template": self.context_template,
            "candidate_template": self.candidate_template,
            "patch_mapping": self.patch_mapping,
            "proposal_prompt_sha256": self.proposal_prompt_sha256,
            "direct_only_proposal_prompt_sha256": (
                self.direct_only_proposal_prompt_sha256
            ),
            "context_prompt_sha256": self.context_prompt_sha256,
            "candidate_prefix_sha256": self.candidate_prefix_sha256,
            "primitive_order": list(self.primitive_order),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def provider_id(self) -> str:
        return (
            f"{self.model_id}@{self.revision}+transformers="
            f"{self.transformers_version}+method-v1={self.fingerprint[:16]}"
        )


@dataclasses.dataclass(frozen=True)
class SelectedPublicFrame:
    frame: PublicFrame
    pixels: Any
    is_current: bool
    temporal_label: str


@dataclasses.dataclass(frozen=True)
class QwenContextEncoding:
    hidden_states: Any
    input_ids: Any
    attention_mask: Any
    image_grid_thw: tuple[tuple[int, int, int], ...]
    image_token_id: int
    spatial_merge_size: int
    patch_size: int
    selected_frames: tuple[SelectedPublicFrame, ...]


@dataclasses.dataclass(frozen=True)
class QwenProposalGeneration:
    raw_json: str
    processed_width: int
    processed_height: int
    prompt_contract_sha256: str
    rendered_user_prompt_sha256: str


def processed_image_size_from_grid(
    image_grid_thw: Sequence[int], *, patch_size: int
) -> tuple[int, int]:
    """Return the exact processed ``(width, height)`` represented by a grid."""

    grid = tuple(int(item) for item in image_grid_thw)
    if len(grid) != 3 or any(item < 1 for item in grid):
        raise ValueError("image_grid_thw must contain three positive integers")
    if (
        not isinstance(patch_size, int)
        or isinstance(patch_size, bool)
        or patch_size < 1
    ):
        raise ValueError("patch_size must be a positive integer")
    _, grid_height, grid_width = grid
    return grid_width * patch_size, grid_height * patch_size


def merged_patch_boxes(
    image_grid_thw: Sequence[int], *, spatial_merge_size: int
) -> tuple[tuple[float, float, float, float], ...]:
    """Map one static Qwen image grid to row-major normalized merged boxes.

    Qwen's processor expands one language ``image_token_id`` for each merged
    spatial cell.  In the pinned implementation, the vision tower restores
    windowed features to original merged-grid order before replacing those
    language positions.  Consequently the corresponding language positions
    follow row-major ``(row, column)`` order on this grid.
    """

    grid = tuple(int(item) for item in image_grid_thw)
    if len(grid) != 3 or any(item < 1 for item in grid):
        raise ValueError("image_grid_thw must contain three positive integers")
    if (
        not isinstance(spatial_merge_size, int)
        or isinstance(spatial_merge_size, bool)
        or spatial_merge_size < 1
    ):
        raise ValueError("spatial_merge_size must be a positive integer")
    temporal, grid_height, grid_width = grid
    if temporal != 1:
        raise ValueError("Method V1 accepts static images only (grid temporal size 1)")
    if grid_height % spatial_merge_size or grid_width % spatial_merge_size:
        raise ValueError("image grid is not divisible by spatial_merge_size")
    merged_height = grid_height // spatial_merge_size
    merged_width = grid_width // spatial_merge_size
    return tuple(
        (
            column / merged_width,
            row / merged_height,
            (column + 1) / merged_width,
            (row + 1) / merged_height,
        )
        for row in range(merged_height)
        for column in range(merged_width)
    )


def image_token_position_runs(
    input_ids: Sequence[int],
    *,
    image_token_id: int,
    image_grid_thw: Sequence[Sequence[int]],
    spatial_merge_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Match contiguous language image-token runs to processor image grids."""

    values = tuple(int(item) for item in input_ids)
    positions = [index for index, value in enumerate(values) if value == image_token_id]
    runs: list[tuple[int, ...]] = []
    current: list[int] = []
    for position in positions:
        if current and position != current[-1] + 1:
            runs.append(tuple(current))
            current = []
        current.append(position)
    if current:
        runs.append(tuple(current))
    grids = tuple(tuple(int(item) for item in grid) for grid in image_grid_thw)
    if len(runs) != len(grids):
        raise ValueError(
            "the number of image-token runs differs from image_grid_thw rows"
        )
    expected_lengths = []
    for grid in grids:
        if len(grid) != 3 or any(item < 1 for item in grid):
            raise ValueError("image_grid_thw contains an invalid row")
        numerator = grid[0] * grid[1] * grid[2]
        denominator = spatial_merge_size**2
        if numerator % denominator:
            raise ValueError("image token count is not divisible by merge area")
        expected_lengths.append(numerator // denominator)
    for image_index, (run, expected) in enumerate(zip(runs, expected_lengths)):
        if len(run) != expected:
            raise ValueError(
                f"image-token run {image_index} has {len(run)} positions; "
                f"processor grid requires {expected}"
            )
    return tuple(runs)


def select_public_context_frames(
    context: PolicyContext,
    *,
    frame_store: Any,
    camera_order: Sequence[str] = ("agentview", "wrist"),
    max_previous_boundaries: int = 2,
) -> tuple[SelectedPublicFrame, ...]:
    """Resolve current RGB plus at most two prior high-level boundaries."""

    if not isinstance(context, PolicyContext):
        raise TypeError("context must be a PolicyContext")
    if not hasattr(frame_store, "resolve"):
        raise TypeError("frame_store must expose resolve(PublicFrame)")
    if max_previous_boundaries < 0:
        raise ValueError("max_previous_boundaries must be non-negative")
    order = {camera: index for index, camera in enumerate(camera_order)}
    boundary_indices = sorted({frame.frame_index for frame in context.frames})
    selected_indices = set(boundary_indices[-(max_previous_boundaries + 1) :])
    selected = sorted(
        (frame for frame in context.frames if frame.frame_index in selected_indices),
        key=lambda frame: (
            frame.frame_index,
            order.get(frame.camera, len(order)),
            frame.camera,
            frame.frame_id,
        ),
    )
    latest_by_camera = {
        camera: context.latest_frame_index(camera)
        for camera in {frame.camera for frame in context.frames}
    }
    result: list[SelectedPublicFrame] = []
    for frame in selected:
        is_current = frame.frame_index == latest_by_camera[frame.camera]
        phase = "current" if is_current else "history"
        result.append(
            SelectedPublicFrame(
                frame=frame,
                pixels=frame_store.resolve(frame),
                is_current=is_current,
                temporal_label=(
                    f"{phase} public RGB; frame_index={frame.frame_index}; "
                    f"camera={frame.camera}; frame_id={frame.frame_id}"
                ),
            )
        )
    if not result or not any(item.is_current for item in result):
        raise ValueError("public context selection lost all current RGB frames")
    return tuple(result)


def public_context_content_blocks(
    context: PolicyContext,
    selected_frames: Sequence[SelectedPublicFrame],
) -> tuple[dict[str, str], ...]:
    """Build chronologically delimited text/image blocks for Qwen's template."""

    if not isinstance(context, PolicyContext):
        raise TypeError("context must be a PolicyContext")
    selected = tuple(selected_frames)
    if not selected or any(
        not isinstance(item, SelectedPublicFrame) for item in selected
    ):
        raise ValueError("selected_frames must contain public RGB frames")
    actions_by_next_frame: dict[int, list[str]] = defaultdict(list)
    for event in context.public_history:
        actions_by_next_frame[event.step_index + 1].append(
            f"{event.primitive.value}: {event.subtask_text}; "
            f"status={event.execution_status.value}"
        )
    # Qwen's official chat template concatenates adjacent text blocks
    # verbatim. Every block owns its trailing newline so task, action, camera,
    # and temporal labels cannot collapse together.
    content: list[dict[str, str]] = [
        {"type": "text", "text": f"Task instruction: {context.prompt}\n"}
    ]
    previous_boundary: int | None = None
    for selected_frame in selected:
        boundary = selected_frame.frame.frame_index
        if boundary != previous_boundary:
            for action_text in actions_by_next_frame.get(boundary, []):
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"Completed action before this observation: {action_text}\n"
                        ),
                    }
                )
            previous_boundary = boundary
        content.extend(
            [
                {
                    "type": "text",
                    "text": f"{selected_frame.temporal_label}\n",
                },
                {"type": "image"},
            ]
        )
    return tuple(content)


def canonical_candidate_instruction(candidate: GroundedIntervention) -> str:
    """Return the immutable instruction bound into a proposal fingerprint."""

    parameters = dict(candidate.parameters)
    if "instruction" in parameters:
        return parameters["instruction"]
    details = "; ".join(f"{name}={value}" for name, value in candidate.parameters)
    suffix = f"; {details}" if details else ""
    if candidate.referent is None:
        return f"{candidate.primitive.value}{suffix}"
    return f"{candidate.primitive.value} {candidate.referent}{suffix}"


def candidate_feature_cache_key(
    *,
    provider_id: str,
    context: PolicyContext,
    candidates: Sequence[GroundedIntervention],
) -> str:
    """Bind a feature cache entry to all decision-time public identities."""

    return canonical_sha256(
        {
            "provider_id": provider_id,
            "context_fingerprint": context.fingerprint(),
            "candidate_fingerprints": [item.fingerprint() for item in candidates],
            "candidate_ids": [item.candidate_id for item in candidates],
        }
    )


def grounding_support_mask(
    public_context: FrozenTokenField,
    grounding: GroundingReference,
) -> Any:
    """Select bbox-center patches, falling back to largest positive overlap."""

    torch = _require_torch()
    if public_context.batch_size != 1:
        raise ValueError("grounding_support_mask expects one public context")
    eligible: list[int] = []
    for token_index in range(public_context.token_count):
        if not bool(public_context.current_patch_mask[0, token_index]):
            continue
        if public_context.camera_ids[0][token_index] != grounding.camera:
            continue
        if public_context.frame_ids[0][token_index] != grounding.frame_id:
            continue
        eligible.append(token_index)
    if not eligible:
        raise ValueError("grounding frame has no current image-token anchors")

    target = grounding.box_xyxy
    centered: list[int] = []
    overlaps: list[tuple[float, int]] = []
    for token_index in eligible:
        patch = tuple(float(item) for item in public_context.patch_xyxy[0, token_index])
        center_x = (patch[0] + patch[2]) / 2.0
        center_y = (patch[1] + patch[3]) / 2.0
        if target[0] <= center_x <= target[2] and target[1] <= center_y <= target[3]:
            centered.append(token_index)
        intersection_width = max(
            0.0, min(target[2], patch[2]) - max(target[0], patch[0])
        )
        intersection_height = max(
            0.0, min(target[3], patch[3]) - max(target[1], patch[1])
        )
        overlap = intersection_width * intersection_height
        if overlap > 0.0:
            overlaps.append((overlap, token_index))

    support = torch.zeros(
        public_context.token_count,
        dtype=torch.bool,
        device=public_context.tokens.device,
    )
    if centered:
        support[centered] = True
        return support
    if overlaps:
        # Stable tie-break: greatest intersection, then lowest token index.
        best_index = max(overlaps, key=lambda item: (item[0], -item[1]))[1]
        support[best_index] = True
        return support
    raise ValueError("candidate bbox has no positive overlap with its image-token grid")


def save_grounding_support_overlay(
    image: Any,
    *,
    patch_boxes: Sequence[Sequence[float]],
    support_mask: Sequence[bool] | Any,
    candidate_box: Sequence[float],
    output_path: str | Path,
) -> Path:
    """Write a human-auditable original-RGB overlay for patch-map tests."""

    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError as error:  # pragma: no cover - optional integration extra.
        raise RuntimeError("overlay rendering requires numpy and Pillow") from error
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("overlay image must have HWC RGB shape")
    if array.dtype != np.uint8:
        raise TypeError("overlay image must use uint8 pixels")
    height, width = array.shape[:2]
    rendered = Image.fromarray(array.copy(), mode="RGB")
    draw = ImageDraw.Draw(rendered, "RGBA")
    if hasattr(support_mask, "detach"):
        support_values = support_mask.detach().cpu().tolist()
    else:
        support_values = list(support_mask)
    boxes = tuple(tuple(float(value) for value in box) for box in patch_boxes)
    if len(boxes) != len(support_values):
        raise ValueError("support_mask length differs from patch_boxes")
    for box, selected in zip(boxes, support_values):
        if not selected:
            continue
        xy = (box[0] * width, box[1] * height, box[2] * width, box[3] * height)
        draw.rectangle(
            xy, fill=(50, 180, 220, 55), outline=(20, 120, 180, 230), width=2
        )
    target = tuple(float(value) for value in candidate_box)
    if len(target) != 4:
        raise ValueError("candidate_box must contain four values")
    draw.rectangle(
        (
            target[0] * width,
            target[1] * height,
            target[2] * width,
            target[3] * height,
        ),
        outline=(230, 55, 70, 255),
        width=3,
    )
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(path, format="PNG")
    return path


class Qwen25VLRuntime:
    """Lazy, frozen Hugging Face runtime pinned to the audited checkpoint."""

    def __init__(
        self,
        *,
        identity: QwenProviderIdentity | None = None,
        device_map: str | Mapping[str, Any] | None = "auto",
        dtype: str | None = None,
        local_files_only: bool = False,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.identity = identity or QwenProviderIdentity()
        if (processor is None) != (model is None):
            raise ValueError(
                "processor and model test injections must be supplied together"
            )
        self.device_map = device_map
        self.dtype = self.identity.torch_dtype if dtype is None else str(dtype)
        if self.dtype != self.identity.torch_dtype:
            raise ValueError(
                "runtime torch dtype differs from frozen provider identity"
            )
        self.local_files_only = bool(local_files_only)
        self._processor = processor
        self._model = model
        self._loaded = processor is not None
        if self._loaded:
            self._freeze_model()

    @property
    def provider_id(self) -> str:
        return self.identity.provider_id

    @property
    def processor(self) -> Any:
        self._ensure_loaded()
        return self._processor

    @property
    def model(self) -> Any:
        self._ensure_loaded()
        return self._model

    @property
    def is_loaded(self) -> bool:
        """Whether the pinned processor and checkpoint are resident and frozen."""

        return bool(self._loaded)

    def ensure_ready(self) -> None:
        """Load and freeze the exact provider before an execution slot is claimed.

        Online collection calls this through the isolated service's ``/ready``
        endpoint.  A missing checkpoint, dependency error, or OOM is therefore
        diagnosed as preflight infrastructure rather than consuming a physical
        branch whose post-OPEN continuation could never run.
        """

        self._ensure_loaded()
        if self._processor is None or self._model is None or not self._loaded:
            raise RuntimeError("Qwen provider did not finish loading")

    def _freeze_model(self) -> None:
        if hasattr(self._model, "requires_grad_"):
            self._model.requires_grad_(False)
        if hasattr(self._model, "eval"):
            self._model.eval()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        installed = _installed_transformers_version()
        if installed != self.identity.transformers_version:
            raise RuntimeError(
                "transformers version changed after provider identity construction: "
                f"identity={self.identity.transformers_version}, installed={installed}"
            )
        if installed != QWEN25VL_TRANSFORMERS_VERSION:
            raise RuntimeError(
                "Method V1 patch mapping is audited against transformers=="
                f"{QWEN25VL_TRANSFORMERS_VERSION}; found {installed}. Use the isolated "
                "pinned environment rather than silently changing preprocessing."
            )
        torch = _require_torch()
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as error:  # pragma: no cover - environment dependent.
            raise RuntimeError(
                "Qwen2.5-VL classes are unavailable in the installed transformers"
            ) from error
        self._processor = AutoProcessor.from_pretrained(
            self.identity.model_id,
            revision=self.identity.revision,
            local_files_only=self.local_files_only,
            trust_remote_code=False,
            use_fast=self.identity.processor_use_fast,
        )
        image_processor = self._processor.image_processor
        patch_size = int(image_processor.patch_size)
        merge_size = int(image_processor.merge_size)
        factor = patch_size * merge_size
        pixel_budget = self.identity.target_image_tokens * factor * factor
        image_processor.min_pixels = pixel_budget
        image_processor.max_pixels = pixel_budget
        dtype_value = getattr(torch, self.dtype, None)
        if dtype_value is None:
            raise ValueError(f"unknown torch dtype {self.dtype!r}")
        load_kwargs: dict[str, Any] = {
            "revision": self.identity.revision,
            "local_files_only": self.local_files_only,
            "trust_remote_code": False,
            "dtype": dtype_value,
            "low_cpu_mem_usage": True,
            "attn_implementation": self.identity.attention_implementation,
        }
        if self.device_map is not None:
            load_kwargs["device_map"] = self.device_map
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.identity.model_id,
            **load_kwargs,
        )
        self._loaded = True
        self._freeze_model()

    def _input_device(self) -> Any:
        torch = _require_torch()
        for parameter in self.model.parameters():
            if parameter.device.type != "meta":
                return parameter.device
        return torch.device("cpu")

    def _move_inputs(self, inputs: Any) -> dict[str, Any]:
        torch = _require_torch()
        device = self._input_device()
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in dict(inputs).items()
        }

    def _multimodal_text(
        self,
        *,
        system_text: str,
        content: list[dict[str, str]],
        add_generation_prompt: bool,
    ) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": content},
        ]
        return str(
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        )

    def _processed_grid_for_image(self, image: Any) -> tuple[int, int, int]:
        result = self.processor.image_processor(images=[image], return_tensors="pt")
        grid = result["image_grid_thw"]
        values = tuple(int(item) for item in grid[0].tolist())
        if len(values) != 3:
            raise ValueError("processor returned malformed image_grid_thw")
        return values  # type: ignore[return-value]

    def generate_proposal_json(
        self,
        *,
        task_prompt: str,
        public_history_text: str,
        image: Any,
        camera_label: str,
        frame_id: str,
        enabled_primitives: Sequence[Primitive] = METHOD_V1_ENABLED_PRIMITIVES,
        max_candidates: int = METHOD_V1_MAX_CANDIDATES,
        max_per_primitive: int = METHOD_V1_MAX_PER_PRIMITIVE,
    ) -> QwenProposalGeneration:
        """Greedily propose JSON from one current public image (temperature 0)."""

        torch = _require_torch()
        grid = self._processed_grid_for_image(image)
        patch_size = int(self.processor.image_processor.patch_size)
        processed_width, processed_height = processed_image_size_from_grid(
            grid, patch_size=patch_size
        )
        contract = qwen_proposal_prompt_contract(
            enabled_primitives=enabled_primitives,
            max_candidates=max_candidates,
            max_per_primitive=max_per_primitive,
        )
        user_text = proposal_request_text(
            task_prompt=task_prompt,
            public_history_text=public_history_text,
            camera_label=camera_label,
            frame_id=frame_id,
            processed_width=processed_width,
            processed_height=processed_height,
            enabled_primitives=enabled_primitives,
            max_candidates=max_candidates,
            max_per_primitive=max_per_primitive,
        )
        content = [
            {"type": "text", "text": user_text},
            {"type": "image"},
        ]
        text = self._multimodal_text(
            system_text=contract.system_prompt,
            content=content,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        final_grid = tuple(int(item) for item in inputs["image_grid_thw"][0].tolist())
        if final_grid != grid:
            raise ValueError("processor produced inconsistent proposal image resize")
        model_inputs = self._move_inputs(inputs)
        self._freeze_model()
        with torch.no_grad():
            generated = self.model.generate(
                **model_inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=512,
            )
        prompt_length = int(model_inputs["input_ids"].shape[1])
        continuation_ids = generated[:, prompt_length:]
        decoded = self.processor.batch_decode(
            continuation_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(decoded) != 1:
            raise ValueError("Qwen proposal generation returned a non-singleton batch")
        return QwenProposalGeneration(
            raw_json=str(decoded[0]).strip(),
            processed_width=processed_width,
            processed_height=processed_height,
            prompt_contract_sha256=contract.fingerprint,
            rendered_user_prompt_sha256=_text_sha256(user_text),
        )

    def encode_public_context(
        self,
        *,
        context: PolicyContext,
        frame_store: Any,
    ) -> QwenContextEncoding:
        """Encode ordered public history and RGB in one multimodal sequence."""

        torch = _require_torch()
        selected = select_public_context_frames(context, frame_store=frame_store)
        content = list(public_context_content_blocks(context, selected))
        text = self._multimodal_text(
            system_text=QWEN_CONTEXT_SYSTEM_PROMPT,
            content=content,
            add_generation_prompt=False,
        )
        inputs = self.processor(
            text=[text],
            images=[item.pixels for item in selected],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        input_ids_cpu = inputs["input_ids"].detach().cpu()
        attention_cpu = inputs["attention_mask"].detach().cpu().to(dtype=torch.bool)
        grids = tuple(
            tuple(int(item) for item in row.tolist())
            for row in inputs["image_grid_thw"].detach().cpu()
        )
        if len(grids) != len(selected):
            raise ValueError(
                "processor image grids do not align with selected public frames"
            )
        model_inputs = self._move_inputs(inputs)
        self._freeze_model()
        with torch.no_grad():
            outputs = self.model(
                **model_inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise ValueError("Qwen did not return language hidden states")
        final_hidden = hidden_states[-1].detach().to(dtype=torch.float32).cpu()
        if final_hidden.shape[:2] != input_ids_cpu.shape:
            raise ValueError("Qwen hidden states do not align with processor input IDs")
        image_processor = self.processor.image_processor
        return QwenContextEncoding(
            hidden_states=final_hidden,
            input_ids=input_ids_cpu,
            attention_mask=attention_cpu,
            image_grid_thw=grids,
            image_token_id=int(self.processor.image_token_id),
            spatial_merge_size=int(image_processor.merge_size),
            patch_size=int(image_processor.patch_size),
            selected_frames=selected,
        )

    def encode_candidate_instructions(self, instructions: Sequence[str]) -> Any:
        """Encode canonical instructions and pool their last content token.

        The fixed prefix states the role of the text.  There is deliberately no
        artificial suffix: pooling an ``[END]`` marker would make the feature
        describe our formatting token instead of the candidate instruction.
        With right padding, the last valid sequence position is therefore the
        final token produced by the canonical instruction itself.
        """

        torch = _require_torch()
        values = tuple(" ".join(str(item).split()) for item in instructions)
        if not values or any(not item for item in values):
            raise ValueError("candidate instructions must be non-empty")
        templates = [f"{QWEN_CANDIDATE_PREFIX}{instruction}" for instruction in values]
        inputs = self.processor.tokenizer(
            templates,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        model_inputs = self._move_inputs(inputs)
        self._freeze_model()
        with torch.no_grad():
            outputs = self.model(
                **model_inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise ValueError("Qwen did not return candidate text hidden states")
        final_hidden = hidden_states[-1]
        attention = model_inputs["attention_mask"].to(dtype=torch.bool)
        pooled = []
        for row in range(len(values)):
            positions = attention[row].nonzero(as_tuple=False).flatten()
            if positions.numel() == 0:
                raise ValueError("candidate template tokenized to an empty sequence")
            pooled.append(final_hidden[row, int(positions[-1])])
        return torch.stack(pooled).detach().to(dtype=torch.float32).cpu()


def _frozen_context_from_encoding(
    encoding: QwenContextEncoding,
    *,
    context: PolicyContext,
    provider_id: str,
) -> FrozenTokenField:
    torch = _require_torch()
    if encoding.hidden_states.ndim != 3 or encoding.hidden_states.shape[0] != 1:
        raise ValueError("Qwen context encoding must have shape [1, token, channel]")
    input_ids = encoding.input_ids
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Qwen input IDs must have shape [1, token]")
    runs = image_token_position_runs(
        input_ids[0].tolist(),
        image_token_id=encoding.image_token_id,
        image_grid_thw=encoding.image_grid_thw,
        spatial_merge_size=encoding.spatial_merge_size,
    )
    if len(runs) != len(encoding.selected_frames):
        raise ValueError("image token runs do not align with public frame metadata")
    token_count = int(input_ids.shape[1])
    current_patch_mask = torch.zeros((1, token_count), dtype=torch.bool)
    patch_xyxy = torch.zeros((1, token_count, 4), dtype=torch.float32)
    camera_ids: list[str | None] = [None] * token_count
    frame_ids: list[str | None] = [None] * token_count
    for selected, grid, positions in zip(
        encoding.selected_frames, encoding.image_grid_thw, runs
    ):
        boxes = merged_patch_boxes(grid, spatial_merge_size=encoding.spatial_merge_size)
        if len(boxes) != len(positions):
            raise ValueError("patch boxes do not align with image token positions")
        for token_index, box in zip(positions, boxes):
            camera_ids[token_index] = selected.frame.camera
            frame_ids[token_index] = selected.frame.frame_id
            patch_xyxy[0, token_index] = torch.tensor(box, dtype=torch.float32)
            if selected.is_current:
                current_patch_mask[0, token_index] = True
    return FrozenTokenField(
        tokens=encoding.hidden_states,
        valid_mask=encoding.attention_mask,
        current_patch_mask=current_patch_mask,
        camera_ids=(tuple(camera_ids),),
        frame_ids=(tuple(frame_ids),),
        patch_xyxy=patch_xyxy,
        context_fingerprints=(context.fingerprint(),),
        provider_id=provider_id,
    )


@dataclasses.dataclass(frozen=True)
class QwenCacheArtifactIdentity:
    """Physical cache identity, distinct from the logical decision-time key."""

    logical_key: str
    artifact_sha256: str
    tensor_artifact_sha256: str
    sidecar_artifact_sha256: str

    def to_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)


def qwen_cache_artifact_sha256(
    *,
    logical_key: str,
    tensor_artifact_sha256: str,
    sidecar_artifact_sha256: str,
) -> str:
    """Bind one logical decision input to its two serialized cache files."""

    return canonical_sha256(
        {
            "schema": QwenFeatureCache.ARTIFACT_SCHEMA,
            "logical_key": logical_key,
            "tensor_artifact_sha256": tensor_artifact_sha256,
            "sidecar_artifact_sha256": sidecar_artifact_sha256,
        }
    )


class QwenFeatureCache:
    """Detached tensors with separate logical and physical identities."""

    SIDECAR_SCHEMA = "qwen-method-v1-feature-cache-v2"
    ARTIFACT_SCHEMA = "qwen-method-v1-feature-cache-artifact-v1"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def _paths(self, key: str) -> tuple[Path, Path]:
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("feature cache key must be lowercase SHA-256")
        directory = self.root / "sha256" / key[:2]
        return directory / f"{key}.pt", directory / f"{key}.json"

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _tensor_sha256(value: Any) -> str:
        torch = _require_torch()
        tensor = value.detach().cpu().contiguous()
        header = canonical_json_bytes(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        )
        payload = tensor.view(torch.uint8).numpy().tobytes(order="C")
        return hashlib.sha256(header + b"\n" + payload).hexdigest()

    @staticmethod
    def _artifact_sha256(
        *,
        logical_key: str,
        tensor_artifact_sha256: str,
        sidecar_artifact_sha256: str,
    ) -> str:
        return qwen_cache_artifact_sha256(
            logical_key=logical_key,
            tensor_artifact_sha256=tensor_artifact_sha256,
            sidecar_artifact_sha256=sidecar_artifact_sha256,
        )

    @staticmethod
    def _tensor_names() -> set[str]:
        return {
            "context_tokens",
            "context_valid_mask",
            "current_patch_mask",
            "patch_xyxy",
            "candidate_tokens",
            "candidate_valid_mask",
            "grounding_support",
            "public_state_values",
            "public_state_valid_mask",
        }

    def _load_with_identity(
        self,
        key: str,
        *,
        context: PolicyContext,
        candidates: Sequence[GroundedIntervention],
        provider_id: str,
        allow_logical_key: bool = False,
    ) -> tuple[CandidateTokenField, QwenCacheArtifactIdentity] | None:
        torch = _require_torch()
        tensor_path, manifest_path = self._paths(key)
        if not tensor_path.exists() and not manifest_path.exists():
            return None
        if not tensor_path.is_file() or not manifest_path.is_file():
            raise ValueError("feature cache entry is incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_keys = {
            "schema",
            "logical_key",
            "provider_id",
            "context_fingerprint",
            "candidate_ids",
            "candidate_fingerprints",
            "primitives",
            "camera_ids",
            "frame_ids",
            "tensor_sha256",
            "tensor_artifact_sha256",
        }
        if not isinstance(manifest, dict) or set(manifest) != manifest_keys:
            raise ValueError("feature cache manifest schema changed")
        logical_key = candidate_feature_cache_key(
            provider_id=provider_id,
            context=context,
            candidates=candidates,
        )
        expected = {
            "schema": self.SIDECAR_SCHEMA,
            "logical_key": logical_key,
            "provider_id": provider_id,
            "context_fingerprint": context.fingerprint(),
            "candidate_ids": [item.candidate_id for item in candidates],
            "candidate_fingerprints": [item.fingerprint() for item in candidates],
            "primitives": [item.primitive.value for item in candidates],
        }
        for name, value in expected.items():
            if manifest.get(name) != value:
                raise ValueError(f"feature cache manifest changed field {name}")
        tensor_artifact_sha256 = self._file_sha256(tensor_path)
        if manifest["tensor_artifact_sha256"] != tensor_artifact_sha256:
            raise ValueError("feature cache serialized tensor artifact digest mismatch")
        sidecar_artifact_sha256 = self._file_sha256(manifest_path)
        artifact_sha256 = self._artifact_sha256(
            logical_key=logical_key,
            tensor_artifact_sha256=tensor_artifact_sha256,
            sidecar_artifact_sha256=sidecar_artifact_sha256,
        )
        if key not in {logical_key, artifact_sha256}:
            raise ValueError("feature cache artifact identity mismatch")
        if (
            key == logical_key
            and provider_id.startswith(f"{QWEN25VL_MODEL_ID}@")
            and not allow_logical_key
        ):
            raise ValueError(
                "formal Method-V1 cache loads require the physical artifact identity"
            )
        tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if not isinstance(tensors, Mapping) or set(tensors) != self._tensor_names():
            raise ValueError("feature cache tensor fields changed")
        observed_digests = {
            name: self._tensor_sha256(value) for name, value in tensors.items()
        }
        if manifest["tensor_sha256"] != observed_digests:
            raise ValueError("feature cache tensor digest mismatch")
        context_field = FrozenTokenField(
            tokens=tensors["context_tokens"],
            valid_mask=tensors["context_valid_mask"],
            current_patch_mask=tensors["current_patch_mask"],
            camera_ids=tuple(tuple(row) for row in manifest["camera_ids"]),
            frame_ids=tuple(tuple(row) for row in manifest["frame_ids"]),
            patch_xyxy=tensors["patch_xyxy"],
            context_fingerprints=(context.fingerprint(),),
            provider_id=provider_id,
        )
        field = CandidateTokenField(
            tokens=tensors["candidate_tokens"],
            valid_mask=tensors["candidate_valid_mask"],
            candidate_ids=(tuple(item.candidate_id for item in candidates),),
            candidate_fingerprints=(tuple(item.fingerprint() for item in candidates),),
            primitives=(tuple(item.primitive for item in candidates),),
            grounding_support=tensors["grounding_support"],
            public_context=context_field,
            public_state_values=tensors["public_state_values"],
            public_state_valid_mask=tensors["public_state_valid_mask"],
        )
        identity = QwenCacheArtifactIdentity(
            logical_key=logical_key,
            artifact_sha256=artifact_sha256,
            tensor_artifact_sha256=tensor_artifact_sha256,
            sidecar_artifact_sha256=sidecar_artifact_sha256,
        )
        return field, identity

    def load(
        self,
        key: str,
        *,
        context: PolicyContext,
        candidates: Sequence[GroundedIntervention],
        provider_id: str,
        allow_logical_key: bool = False,
    ) -> CandidateTokenField | None:
        loaded = self._load_with_identity(
            key,
            context=context,
            candidates=candidates,
            provider_id=provider_id,
            allow_logical_key=allow_logical_key,
        )
        return None if loaded is None else loaded[0]

    def artifact_identity(
        self,
        key: str,
        *,
        context: PolicyContext,
        candidates: Sequence[GroundedIntervention],
        provider_id: str,
    ) -> QwenCacheArtifactIdentity:
        loaded = self._load_with_identity(
            key,
            context=context,
            candidates=candidates,
            provider_id=provider_id,
            allow_logical_key=True,
        )
        if loaded is None:
            raise FileNotFoundError(f"missing Qwen feature cache {key}")
        return loaded[1]

    def put(self, key: str, field: CandidateTokenField) -> QwenCacheArtifactIdentity:
        torch = _require_torch()
        logical_tensor_path, logical_manifest_path = self._paths(key)
        if logical_tensor_path.exists() or logical_manifest_path.exists():
            raise FileExistsError("feature cache entries are immutable")
        tensors = {
            "context_tokens": field.public_context.tokens.detach().cpu(),
            "context_valid_mask": field.public_context.valid_mask.detach().cpu(),
            "current_patch_mask": field.public_context.current_patch_mask.detach().cpu(),
            "patch_xyxy": field.public_context.patch_xyxy.detach().cpu(),
            "candidate_tokens": field.tokens.detach().cpu(),
            "candidate_valid_mask": field.valid_mask.detach().cpu(),
            "grounding_support": field.grounding_support.detach().cpu(),
            "public_state_values": field.public_state_values.detach().cpu(),
            "public_state_valid_mask": field.public_state_valid_mask.detach().cpu(),
        }
        manifest: dict[str, Any] = {
            "schema": self.SIDECAR_SCHEMA,
            "logical_key": key,
            "provider_id": field.public_context.provider_id,
            "context_fingerprint": field.context_fingerprints[0],
            "candidate_ids": list(field.candidate_ids[0]),
            "candidate_fingerprints": list(field.candidate_fingerprints[0]),
            "primitives": [item.value for item in field.primitives[0]],
            "camera_ids": [list(row) for row in field.public_context.camera_ids],
            "frame_ids": [list(row) for row in field.public_context.frame_ids],
            "tensor_sha256": {
                name: self._tensor_sha256(value) for name, value in tensors.items()
            },
        }
        temporary_dir = self.root / ".tmp"
        temporary_dir.mkdir(parents=True, exist_ok=True)
        tensor_tmp = temporary_dir / f"{key}.pt.tmp-{os.getpid()}"
        manifest_tmp = temporary_dir / f"{key}.json.tmp-{os.getpid()}"
        artifact_tensor_path: Path | None = None
        artifact_manifest_path: Path | None = None
        try:
            torch.save(tensors, tensor_tmp)
            tensor_artifact_sha256 = self._file_sha256(tensor_tmp)
            manifest["tensor_artifact_sha256"] = tensor_artifact_sha256
            manifest_tmp.write_bytes(canonical_json_bytes(manifest) + b"\n")
            sidecar_artifact_sha256 = self._file_sha256(manifest_tmp)
            artifact_sha256 = self._artifact_sha256(
                logical_key=key,
                tensor_artifact_sha256=tensor_artifact_sha256,
                sidecar_artifact_sha256=sidecar_artifact_sha256,
            )
            identity = QwenCacheArtifactIdentity(
                logical_key=key,
                artifact_sha256=artifact_sha256,
                tensor_artifact_sha256=tensor_artifact_sha256,
                sidecar_artifact_sha256=sidecar_artifact_sha256,
            )
            artifact_tensor_path, artifact_manifest_path = self._paths(artifact_sha256)
            if artifact_tensor_path.exists() or artifact_manifest_path.exists():
                raise FileExistsError("feature cache artifact identity already exists")
            artifact_tensor_path.parent.mkdir(parents=True, exist_ok=True)
            tensor_tmp.replace(artifact_tensor_path)
            manifest_tmp.replace(artifact_manifest_path)
            logical_tensor_path.parent.mkdir(parents=True, exist_ok=True)
            os.link(artifact_tensor_path, logical_tensor_path)
            os.link(artifact_manifest_path, logical_manifest_path)
            return identity
        finally:
            tensor_tmp.unlink(missing_ok=True)
            manifest_tmp.unlink(missing_ok=True)
            if logical_tensor_path.exists() != logical_manifest_path.exists():
                logical_tensor_path.unlink(missing_ok=True)
                logical_manifest_path.unlink(missing_ok=True)
            if (
                artifact_tensor_path is not None
                and artifact_manifest_path is not None
                and artifact_tensor_path.exists() != artifact_manifest_path.exists()
            ):
                artifact_tensor_path.unlink(missing_ok=True)
                artifact_manifest_path.unlink(missing_ok=True)


class Qwen25VLTokenProvider:
    """Implement ``FrozenTokenProvider`` for one public decision context."""

    def __init__(
        self,
        *,
        runtime: Qwen25VLRuntime,
        frame_store: Any,
        cache: QwenFeatureCache | None = None,
        remaining_budget_fraction: float | None = 1.0,
    ) -> None:
        if not isinstance(runtime, Qwen25VLRuntime):
            raise TypeError("runtime must be a Qwen25VLRuntime")
        if not hasattr(frame_store, "resolve"):
            raise TypeError("frame_store must expose resolve(PublicFrame)")
        self.runtime = runtime
        self.frame_store = frame_store
        self.cache = cache
        if remaining_budget_fraction is not None:
            remaining_budget_fraction = float(remaining_budget_fraction)
            if not 0.0 <= remaining_budget_fraction <= 1.0:
                raise ValueError("remaining_budget_fraction must be in [0, 1]")
        self.remaining_budget_fraction = remaining_budget_fraction
        self._memory_cache: dict[str, CandidateTokenField] = {}

    @property
    def provider_id(self) -> str:
        budget = (
            "missing"
            if self.remaining_budget_fraction is None
            else format(self.remaining_budget_fraction, ".17g")
        )
        return f"{self.runtime.provider_id}+public-state=proprio8+budget={budget}"

    def encode(
        self,
        context: PolicyContext,
        candidates: Sequence[GroundedIntervention],
    ) -> CandidateTokenField:
        torch = _require_torch()
        if not isinstance(context, PolicyContext):
            raise TypeError("context must be a PolicyContext")
        values = tuple(candidates)
        if not values:
            raise ValueError("at least one candidate is required")
        if any(not isinstance(candidate, GroundedIntervention) for candidate in values):
            raise TypeError("candidates must contain GroundedIntervention values")
        if any(candidate.primitive is Primitive.STOP for candidate in values):
            raise ValueError("Method V1 does not learn STOP as an outcome candidate")
        for candidate in values:
            candidate.validate_against(context)
        key = candidate_feature_cache_key(
            provider_id=self.provider_id,
            context=context,
            candidates=values,
        )
        if key in self._memory_cache:
            return self._memory_cache[key]
        if self.cache is not None:
            cached = self.cache.load(
                key,
                context=context,
                candidates=values,
                provider_id=self.provider_id,
                allow_logical_key=True,
            )
            if cached is not None:
                self._memory_cache[key] = cached
                return cached

        encoding = self.runtime.encode_public_context(
            context=context,
            frame_store=self.frame_store,
        )
        public_context = _frozen_context_from_encoding(
            encoding,
            context=context,
            provider_id=self.provider_id,
        )
        instructions = tuple(canonical_candidate_instruction(item) for item in values)
        text_embeddings = self.runtime.encode_candidate_instructions(instructions)
        if text_embeddings.ndim != 2 or text_embeddings.shape[0] != len(values):
            raise ValueError("candidate embeddings have the wrong shape")
        if text_embeddings.shape[1] != public_context.channel_count:
            raise ValueError("context and candidate Qwen hidden dimensions differ")

        candidate_rows = []
        supports = []
        primitive_count = len(METHOD_V1_PRIMITIVE_ORDER)
        for candidate_index, candidate in enumerate(values):
            if candidate.grounding is None:
                raise ValueError("physical Method-V1 candidate has no grounding")
            one_hot = torch.zeros(primitive_count, dtype=torch.float32)
            one_hot[METHOD_V1_PRIMITIVE_ORDER.index(candidate.primitive)] = 1.0
            bbox = torch.tensor(candidate.grounding.box_xyxy, dtype=torch.float32)
            candidate_rows.append(
                torch.cat((text_embeddings[candidate_index], bbox, one_hot), dim=0)
            )
            supports.append(grounding_support_mask(public_context, candidate.grounding))
        public_state_values = torch.zeros((1, 9), dtype=torch.float32)
        public_state_valid_mask = torch.zeros((1,), dtype=torch.bool)
        if context.proprioception is not None:
            if len(context.proprioception) != 8:
                raise ValueError("Method V1 public proprioception must have 8 values")
            if self.remaining_budget_fraction is None:
                raise ValueError(
                    "public proprioception cannot be used without a public budget fraction"
                )
            public_state_values[0, :8] = torch.tensor(
                context.proprioception, dtype=torch.float32
            )
            public_state_values[0, 8] = self.remaining_budget_fraction
            public_state_valid_mask[0] = True
        field = CandidateTokenField(
            tokens=torch.stack(candidate_rows).unsqueeze(0),
            valid_mask=torch.ones((1, len(values)), dtype=torch.bool),
            candidate_ids=(tuple(candidate.candidate_id for candidate in values),),
            candidate_fingerprints=(
                tuple(candidate.fingerprint() for candidate in values),
            ),
            primitives=(tuple(candidate.primitive for candidate in values),),
            grounding_support=torch.stack(supports).unsqueeze(0),
            public_context=public_context,
            public_state_values=public_state_values,
            public_state_valid_mask=public_state_valid_mask,
        )
        if self.cache is not None:
            self.cache.put(key, field)
        self._memory_cache[key] = field
        return field
