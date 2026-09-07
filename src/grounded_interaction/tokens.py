"""Auditable token contracts for grounded intervention scoring.

The classes in this module are *software contracts*, not empirical evidence.
They make two boundaries explicit:

1. frozen VLM features are detached from the trainable planner; and
2. a physical candidate is bound to current public image patches before any
   learned outcome model may score it.

No class here assigns semantic truth, task success, or calibrated uncertainty.
Those properties require separately collected data and evaluation.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from typing import Any

from .contracts import Primitive

try:  # The public contracts remain importable in a CPU/core-only install.
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - exercised in torch-free installations.
    torch = None
    Tensor = Any  # type: ignore[misc,assignment]


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "token tensors require the optional PyTorch dependency; "
            "install the learned extra before constructing this object"
        )


def _tensor(value: Any, *, name: str) -> Tensor:
    _require_torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value.detach()


def _bool_tensor(value: Any, *, name: str) -> Tensor:
    result = _tensor(value, name=name)
    if result.dtype is not torch.bool:
        raise TypeError(f"{name} must have bool dtype")
    return result


def _fingerprint(value: Any, *, name: str) -> str:
    result = str(value)
    if not _SHA256_RE.fullmatch(result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _metadata_grid(
    value: Sequence[Sequence[str | None]],
    *,
    batch_size: int,
    width: int,
    name: str,
) -> tuple[tuple[str | None, ...], ...]:
    rows = tuple(tuple(item for item in row) for row in value)
    if len(rows) != batch_size or any(len(row) != width for row in rows):
        raise ValueError(f"{name} must have shape [{batch_size}, {width}]")
    normalized: list[tuple[str | None, ...]] = []
    for batch_index, row in enumerate(rows):
        items: list[str | None] = []
        for item_index, item in enumerate(row):
            if item is None:
                items.append(None)
                continue
            text = " ".join(str(item).split())
            if not text:
                raise ValueError(f"{name}[{batch_index}][{item_index}] is empty")
            items.append(text)
        normalized.append(tuple(items))
    return tuple(normalized)


@dataclasses.dataclass(frozen=True)
class FrozenTokenField:
    """Detached public tokens emitted by a frozen multimodal encoder.

    ``current_patch_mask`` identifies image-patch tokens from the current
    observation.  Prompt, history, proprioceptive, latent, or stale visual
    tokens may remain in ``tokens`` and ``valid_mask``, but cannot be used as a
    physical grounding support.  Patch boxes are normalized to ``[0, 1]`` and
    are meaningful only where ``current_patch_mask`` is true.

    ``provider_id`` is supplied by the concrete token adapter and uniquely
    names its encoder checkpoint plus preprocessing version.

    Construction validates tensor and provenance consistency only; it does not
    establish that the encoder is accurate or useful.
    """

    tokens: Tensor
    valid_mask: Tensor
    current_patch_mask: Tensor
    camera_ids: tuple[tuple[str | None, ...], ...]
    frame_ids: tuple[tuple[str | None, ...], ...]
    patch_xyxy: Tensor
    context_fingerprints: tuple[str, ...]
    provider_id: str

    def __post_init__(self) -> None:
        tokens = _tensor(self.tokens, name="tokens")
        valid_mask = _bool_tensor(self.valid_mask, name="valid_mask")
        current_patch_mask = _bool_tensor(
            self.current_patch_mask, name="current_patch_mask"
        )
        patch_xyxy = _tensor(self.patch_xyxy, name="patch_xyxy")
        if not isinstance(self.provider_id, str):
            raise TypeError("provider_id must be a string")
        provider_id = " ".join(self.provider_id.split())
        if not provider_id:
            raise ValueError("provider_id must identify checkpoint and preprocessing")

        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch, token, channel]")
        batch_size, token_count, channel_count = tokens.shape
        if batch_size < 1 or token_count < 1 or channel_count < 1:
            raise ValueError("tokens must have non-empty dimensions")
        if not tokens.is_floating_point():
            raise TypeError("tokens must use a floating dtype")
        if not bool(torch.isfinite(tokens).all()):
            raise ValueError("tokens must be finite")
        if valid_mask.shape != (batch_size, token_count):
            raise ValueError("valid_mask must have shape [batch, token]")
        if current_patch_mask.shape != valid_mask.shape:
            raise ValueError("current_patch_mask must match valid_mask")
        if bool((current_patch_mask & ~valid_mask).any()):
            raise ValueError("current image patches must also be valid tokens")
        if bool((valid_mask.sum(dim=1) == 0).any()):
            raise ValueError("every context must contain at least one valid token")
        if patch_xyxy.shape != (batch_size, token_count, 4):
            raise ValueError("patch_xyxy must have shape [batch, token, 4]")
        if not patch_xyxy.is_floating_point():
            raise TypeError("patch_xyxy must use a floating dtype")
        if not bool(torch.isfinite(patch_xyxy).all()):
            raise ValueError("patch_xyxy must be finite")
        if not (
            tokens.device
            == valid_mask.device
            == current_patch_mask.device
            == patch_xyxy.device
        ):
            raise ValueError("all token-field tensors must share one device")

        cameras = _metadata_grid(
            self.camera_ids,
            batch_size=batch_size,
            width=token_count,
            name="camera_ids",
        )
        frames = _metadata_grid(
            self.frame_ids,
            batch_size=batch_size,
            width=token_count,
            name="frame_ids",
        )
        fingerprints = tuple(
            _fingerprint(value, name=f"context_fingerprints[{index}]")
            for index, value in enumerate(self.context_fingerprints)
        )
        if len(fingerprints) != batch_size:
            raise ValueError("context_fingerprints must contain one digest per batch")

        for batch_index, token_index in current_patch_mask.nonzero(
            as_tuple=False
        ).tolist():
            if cameras[batch_index][token_index] is None:
                raise ValueError("each current patch requires a camera_id")
            if frames[batch_index][token_index] is None:
                raise ValueError("each current patch requires a frame_id")
            x0, y0, x1, y1 = patch_xyxy[batch_index, token_index].tolist()
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                raise ValueError(
                    "current patch boxes must be non-empty normalized xyxy boxes"
                )

        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "valid_mask", valid_mask)
        object.__setattr__(self, "current_patch_mask", current_patch_mask)
        object.__setattr__(self, "patch_xyxy", patch_xyxy)
        object.__setattr__(self, "camera_ids", cameras)
        object.__setattr__(self, "frame_ids", frames)
        object.__setattr__(self, "context_fingerprints", fingerprints)
        object.__setattr__(self, "provider_id", provider_id)

    @property
    def batch_size(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def token_count(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def channel_count(self) -> int:
        return int(self.tokens.shape[2])


@dataclasses.dataclass(frozen=True)
class CandidateTokenField:
    """Detached candidate queries irreversibly bound to public patch support.

    A valid physical candidate has a complete immutable identifier, contract
    fingerprint, primitive, and a non-empty support set.  Every supported token
    must be a valid *current* image patch, and the support may not cross cameras
    or frames.  ``STOP`` is the only valid candidate allowed to have no visual
    support.  This prevents a learned scorer from predicting a generic action
    first and attaching an unrelated execution target afterwards.

    The contract proves identity and tensor alignment, not grounding quality.
    """

    tokens: Tensor
    valid_mask: Tensor
    candidate_ids: tuple[tuple[str | None, ...], ...]
    candidate_fingerprints: tuple[tuple[str | None, ...], ...]
    primitives: tuple[tuple[Primitive | str | None, ...], ...]
    grounding_support: Tensor
    public_context: FrozenTokenField

    def __post_init__(self) -> None:
        if not isinstance(self.public_context, FrozenTokenField):
            raise TypeError("public_context must be a FrozenTokenField")
        tokens = _tensor(self.tokens, name="candidate tokens")
        valid_mask = _bool_tensor(self.valid_mask, name="candidate valid_mask")
        support = _bool_tensor(self.grounding_support, name="grounding_support")
        if tokens.ndim != 3:
            raise ValueError(
                "candidate tokens must have shape [batch, candidate, channel]"
            )
        batch_size, candidate_count, channel_count = tokens.shape
        if batch_size < 1 or candidate_count < 1 or channel_count < 1:
            raise ValueError("candidate tokens must have non-empty dimensions")
        if not tokens.is_floating_point():
            raise TypeError("candidate tokens must use a floating dtype")
        if not bool(torch.isfinite(tokens).all()):
            raise ValueError("candidate tokens must be finite")
        if valid_mask.shape != (batch_size, candidate_count):
            raise ValueError("candidate valid_mask must have shape [batch, candidate]")
        if bool((valid_mask.sum(dim=1) == 0).any()):
            raise ValueError("every example must contain at least one valid candidate")
        if batch_size != self.public_context.batch_size:
            raise ValueError("candidate and context batch sizes differ")
        expected_support_shape = (
            batch_size,
            candidate_count,
            self.public_context.token_count,
        )
        if support.shape != expected_support_shape:
            raise ValueError(
                "grounding_support must have shape [batch, candidate, context-token]"
            )
        if not (tokens.device == valid_mask.device == support.device):
            raise ValueError("candidate and context tensors must share one device")

        ids = _metadata_grid(
            self.candidate_ids,
            batch_size=batch_size,
            width=candidate_count,
            name="candidate_ids",
        )
        raw_fingerprints = _metadata_grid(
            self.candidate_fingerprints,
            batch_size=batch_size,
            width=candidate_count,
            name="candidate_fingerprints",
        )
        raw_primitives = tuple(tuple(row) for row in self.primitives)
        if len(raw_primitives) != batch_size or any(
            len(row) != candidate_count for row in raw_primitives
        ):
            raise ValueError(
                f"primitives must have shape [{batch_size}, {candidate_count}]"
            )

        fingerprints: list[tuple[str | None, ...]] = []
        primitives: list[tuple[Primitive | None, ...]] = []
        allowed_support = (
            self.public_context.valid_mask & self.public_context.current_patch_mask
        )

        for batch_index in range(batch_size):
            batch_fingerprints: list[str | None] = []
            batch_primitives: list[Primitive | None] = []
            seen_ids: set[str] = set()
            seen_fingerprints: set[str] = set()
            for candidate_index in range(candidate_count):
                is_valid = bool(valid_mask[batch_index, candidate_index])
                candidate_support = support[batch_index, candidate_index]
                candidate_id = ids[batch_index][candidate_index]
                raw_fingerprint = raw_fingerprints[batch_index][candidate_index]
                raw_primitive = raw_primitives[batch_index][candidate_index]

                if not is_valid:
                    if (
                        candidate_id is not None
                        or raw_fingerprint is not None
                        or raw_primitive is not None
                        or bool(candidate_support.any())
                    ):
                        raise ValueError(
                            "padded candidates must not carry identity or grounding"
                        )
                    batch_fingerprints.append(None)
                    batch_primitives.append(None)
                    continue

                if candidate_id is None:
                    raise ValueError("every valid candidate requires candidate_id")
                if candidate_id in seen_ids:
                    raise ValueError("candidate_id values must be unique per context")
                seen_ids.add(candidate_id)
                if raw_fingerprint is None:
                    raise ValueError(
                        "every valid candidate requires candidate_fingerprint"
                    )
                fingerprint = _fingerprint(
                    raw_fingerprint,
                    name=(f"candidate_fingerprints[{batch_index}][{candidate_index}]"),
                )
                if fingerprint in seen_fingerprints:
                    raise ValueError(
                        "candidate fingerprints must be unique per context"
                    )
                seen_fingerprints.add(fingerprint)
                try:
                    primitive = (
                        raw_primitive
                        if isinstance(raw_primitive, Primitive)
                        else Primitive(str(raw_primitive))
                    )
                except ValueError as error:
                    raise ValueError(
                        f"invalid primitive for candidate {candidate_id!r}"
                    ) from error

                if bool((candidate_support & ~allowed_support[batch_index]).any()):
                    raise ValueError(
                        "grounding support may contain only valid current patches"
                    )
                support_indices = (
                    candidate_support.nonzero(as_tuple=False).flatten().tolist()
                )
                if primitive is Primitive.STOP:
                    if support_indices:
                        raise ValueError("STOP must not claim physical patch support")
                else:
                    if not support_indices:
                        raise ValueError(
                            "every physical candidate requires non-empty current-patch support"
                        )
                    support_cameras = {
                        self.public_context.camera_ids[batch_index][token_index]
                        for token_index in support_indices
                    }
                    support_frames = {
                        self.public_context.frame_ids[batch_index][token_index]
                        for token_index in support_indices
                    }
                    if len(support_cameras) != 1 or len(support_frames) != 1:
                        raise ValueError(
                            "one candidate grounding may not cross cameras or frames"
                        )

                batch_fingerprints.append(fingerprint)
                batch_primitives.append(primitive)
            fingerprints.append(tuple(batch_fingerprints))
            primitives.append(tuple(batch_primitives))

        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "valid_mask", valid_mask)
        object.__setattr__(self, "candidate_ids", ids)
        object.__setattr__(self, "candidate_fingerprints", tuple(fingerprints))
        object.__setattr__(self, "primitives", tuple(primitives))
        object.__setattr__(self, "grounding_support", support)

    @property
    def context_fingerprints(self) -> tuple[str, ...]:
        return self.public_context.context_fingerprints

    @property
    def batch_size(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.tokens.shape[1])


@dataclasses.dataclass(frozen=True)
class GroundedCandidateBatch:
    """Trainable candidate states produced after support-constrained attention.

    Instances are emitted by the grounded-candidate encoder and are the sole
    input type accepted by :class:`grounded_interaction.model.OutcomeScorer`.
    They retain candidate and context fingerprints so the exact scored
    intervention can be propagated to an executor receipt.  This is an
    interface invariant, not a performance claim.
    """

    grounded_tokens: Tensor
    public_context_tokens: Tensor
    public_context_valid_mask: Tensor
    candidate_valid_mask: Tensor
    grounding_attention: Tensor
    candidate_ids: tuple[tuple[str | None, ...], ...]
    candidate_fingerprints: tuple[tuple[str | None, ...], ...]
    primitives: tuple[tuple[Primitive | None, ...], ...]
    context_fingerprints: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_torch()
        tensor_fields = {
            "grounded_tokens": self.grounded_tokens,
            "public_context_tokens": self.public_context_tokens,
            "public_context_valid_mask": self.public_context_valid_mask,
            "candidate_valid_mask": self.candidate_valid_mask,
            "grounding_attention": self.grounding_attention,
        }
        if any(not isinstance(value, torch.Tensor) for value in tensor_fields.values()):
            raise TypeError(
                "every GroundedCandidateBatch tensor field must be a tensor"
            )
        if self.grounded_tokens.ndim != 3:
            raise ValueError(
                "grounded_tokens must have shape [batch, candidate, hidden]"
            )
        batch_size, candidate_count, hidden_size = self.grounded_tokens.shape
        if self.public_context_tokens.ndim != 3:
            raise ValueError(
                "public_context_tokens must have shape [batch, token, hidden]"
            )
        if self.public_context_tokens.shape[0] != batch_size:
            raise ValueError("grounded and public-context batch sizes differ")
        if self.public_context_tokens.shape[2] != hidden_size:
            raise ValueError("grounded and public-context hidden sizes differ")
        token_count = self.public_context_tokens.shape[1]
        if self.public_context_valid_mask.shape != (batch_size, token_count):
            raise ValueError("public_context_valid_mask has the wrong shape")
        if self.candidate_valid_mask.shape != (batch_size, candidate_count):
            raise ValueError("candidate_valid_mask has the wrong shape")
        if self.grounding_attention.shape != (
            batch_size,
            candidate_count,
            token_count,
        ):
            raise ValueError("grounding_attention has the wrong shape")
        if self.public_context_valid_mask.dtype is not torch.bool:
            raise TypeError("public_context_valid_mask must have bool dtype")
        if self.candidate_valid_mask.dtype is not torch.bool:
            raise TypeError("candidate_valid_mask must have bool dtype")
        if bool((self.public_context_valid_mask.sum(dim=1) == 0).any()):
            raise ValueError("every grounded batch row requires public context")
        devices = {value.device for value in tensor_fields.values()}
        if len(devices) != 1:
            raise ValueError("all grounded-batch tensors must share one device")
        metadata = (
            self.candidate_ids,
            self.candidate_fingerprints,
            self.primitives,
        )
        if any(
            len(rows) != batch_size or any(len(row) != candidate_count for row in rows)
            for rows in metadata
        ):
            raise ValueError("candidate metadata must match [batch, candidate]")
        if len(self.context_fingerprints) != batch_size:
            raise ValueError("context fingerprint count differs from batch size")
