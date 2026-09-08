"""Public-RGB-only candidate proposals for Method V1.

This module is deliberately strict at the VLM boundary.  Qwen may *propose*
grounded actions, but its text is not trusted until every field has passed a
closed schema, coordinate, primitive, duplication, and cardinality check.
No simulator identifier or evaluator value is accepted by this adapter.

Qwen2.5-VL reports absolute coordinates in the image dimensions seen by its
processor.  ``absolute_processed_bbox_to_normalized_original`` makes the
resize transform explicit before constructing the repository's normalized
``GroundingReference``.  The final normalization algebraically cancels the
original dimensions; retaining both steps documents and tests the coordinate
contract instead of guessing a legacy ``[0, 1000]`` convention.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from .contracts import (
    GroundedIntervention,
    GroundingReference,
    PolicyContext,
    Primitive,
    canonical_json_bytes,
    canonical_sha256,
)

METHOD_V1_ENABLED_PRIMITIVES = (Primitive.DIRECT, Primitive.OPEN)
METHOD_V1_MAX_CANDIDATES = 6
METHOD_V1_MAX_PER_PRIMITIVE = 3
METHOD_V1_DIRECT_ONLY_PRIMITIVES = (Primitive.DIRECT,)
METHOD_V1_DIRECT_ONLY_MAX_CANDIDATES = 3
QWEN_COORDINATE_CONTRACT = "qwen25-smart-resized-absolute-xyxy-v1"
QWEN_PROPOSAL_HISTORY_VERSION = "method-v1-public-action-history-v1"
QWEN_PROPOSAL_PROMPT_CONTRACT_SCHEMA = "method-v1-qwen-prompt-contract-v1"
QWEN_PROPOSAL_RANKING_SEMANTICS = (
    "preserve_frozen_vlm_array_order_most_likely_final_task_success_first_v1"
)
_PROPOSAL_KEYS = {"visible_label", "bbox", "primitive", "instruction"}
_TOP_LEVEL_KEYS = {"candidates"}


QWEN_PROPOSAL_SYSTEM_PROMPT = """\
You propose public-image-grounded robot candidates. Return exactly one JSON
object with exactly the key "candidates". Its value is a JSON array. Every
array item must contain exactly these keys: "visible_label", "bbox",
"primitive", and "instruction".

Rules:
- bbox is [x0,y0,x1,y1] in absolute pixels of the processed image dimensions
  stated by the user; do not use normalized or 0-1000 coordinates.
- primitive is exactly "DIRECT" or "OPEN".
- DIRECT is a complete attempt to perform the user's final task on a visible
  candidate object. OPEN opens one visible container or articulated part so a
  later policy can continue the final task.
- Refer only to visible evidence. Never claim what is hidden behind or inside
  an occluder. Never invent simulator names, instance IDs, poses, or labels.
- instruction is a complete executable high-level instruction and must name
  the visible referent unambiguously.
- Order the array from most to least likely to complete the user's final task
  under the fixed Method-V1 horizon. DIRECT receives up to 300 control steps.
  OPEN receives 100 steps, then a new public image is used to choose one
  DIRECT continuation for the remaining 200 steps. This order is a frozen-VLM
  baseline and continuation tie-break only; do not emit confidence numbers.
- Propose at most 6 candidates and at most 3 of either primitive.
- Emit JSON only: no markdown, prose, comments, NaN, or trailing text.
"""

QWEN_DIRECT_ONLY_SYSTEM_PROMPT = """\
You propose the single post-information continuation from public RGB. Return
exactly one JSON object with exactly the key "candidates". Its value is a JSON
array. Every array item must contain exactly these keys: "visible_label",
"bbox", "primitive", and "instruction".

Rules:
- bbox is [x0,y0,x1,y1] in absolute pixels of the processed image dimensions
  stated by the user; do not use normalized or 0-1000 coordinates.
- primitive is exactly "DIRECT". Do not output OPEN or another information
  action: the preceding information action has already consumed its budget.
- DIRECT is a complete attempt to perform the user's original final task on a
  currently visible candidate object.
- Refer only to visible evidence. Never claim what is hidden behind or inside
  an occluder. Never invent simulator names, instance IDs, poses, or labels.
- instruction is a complete executable high-level instruction and must name
  the visible referent unambiguously.
- Order the array from most to least likely to complete the user's original
  final task within the remaining 200 control steps. The first valid DIRECT is
  executed as the frozen-VLM continuation; do not emit confidence numbers.
- Propose at most 3 DIRECT candidates.
- Emit JSON only: no markdown, prose, comments, NaN, or trailing text.
"""

_INITIAL_PROPOSAL_USER_TEMPLATE = """\
Final task: {task_prompt}
{public_history_text}
Current public RGB: camera={camera_label}; frame={frame_id}.
Processed image size: width={processed_width}, height={processed_height}.
Return at most 6 ranked grounded candidates under the initial DIRECT-or-OPEN contract, with at most 3 per primitive.
"""

_DIRECT_ONLY_PROPOSAL_USER_TEMPLATE = """\
Original final task: {task_prompt}
{public_history_text}
Current post-information public RGB: camera={camera_label}; frame={frame_id}.
Processed image size: width={processed_width}, height={processed_height}.
Return at most 3 ranked grounded DIRECT candidates for the remaining 200 control steps.
"""


@dataclasses.dataclass(frozen=True)
class QwenProposalPromptContract:
    """One of the two frozen proposal interfaces used by Method V1.

    The system prompt and fixed user-message template are content-addressed
    together. Runtime task text, public history, image identity, and processed
    dimensions are bound separately by the rendered-request digest.
    """

    mode: str
    enabled_primitives: tuple[Primitive, ...]
    max_candidates: int
    max_per_primitive: int
    system_prompt: str
    user_template: str
    schema_version: str = QWEN_PROPOSAL_PROMPT_CONTRACT_SCHEMA
    ranking_semantics: str = QWEN_PROPOSAL_RANKING_SEMANTICS

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "enabled_primitives": [item.value for item in self.enabled_primitives],
            "max_candidates": self.max_candidates,
            "max_per_primitive": self.max_per_primitive,
            "system_prompt": self.system_prompt,
            "user_template": self.user_template,
            "ranking_semantics": self.ranking_semantics,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())


_INITIAL_PROPOSAL_CONTRACT = QwenProposalPromptContract(
    mode="INITIAL_DIRECT_OR_OPEN",
    enabled_primitives=METHOD_V1_ENABLED_PRIMITIVES,
    max_candidates=METHOD_V1_MAX_CANDIDATES,
    max_per_primitive=METHOD_V1_MAX_PER_PRIMITIVE,
    system_prompt=QWEN_PROPOSAL_SYSTEM_PROMPT,
    user_template=_INITIAL_PROPOSAL_USER_TEMPLATE,
)
_DIRECT_ONLY_PROPOSAL_CONTRACT = QwenProposalPromptContract(
    mode="POST_INFORMATION_DIRECT_ONLY",
    enabled_primitives=METHOD_V1_DIRECT_ONLY_PRIMITIVES,
    max_candidates=METHOD_V1_DIRECT_ONLY_MAX_CANDIDATES,
    max_per_primitive=METHOD_V1_MAX_PER_PRIMITIVE,
    system_prompt=QWEN_DIRECT_ONLY_SYSTEM_PROMPT,
    user_template=_DIRECT_ONLY_PROPOSAL_USER_TEMPLATE,
)


def qwen_proposal_prompt_contract(
    *,
    enabled_primitives: Sequence[Primitive],
    max_candidates: int,
    max_per_primitive: int,
) -> QwenProposalPromptContract:
    """Select a frozen prompt contract; arbitrary runtime prompt modes fail."""

    key = (
        tuple(enabled_primitives),
        max_candidates,
        max_per_primitive,
    )
    contracts = (_INITIAL_PROPOSAL_CONTRACT, _DIRECT_ONLY_PROPOSAL_CONTRACT)
    for contract in contracts:
        expected = (
            contract.enabled_primitives,
            contract.max_candidates,
            contract.max_per_primitive,
        )
        if key == expected:
            return contract
    raise ValueError(
        "Method V1 supports only the frozen initial DIRECT+OPEN (6/3) and "
        "post-information DIRECT-only (3/3) proposal contracts"
    )


def qwen_proposer_id(
    *,
    provider_id: str,
    proposal_camera: str,
    enabled_primitives: Sequence[Primitive],
    max_candidates: int,
    max_per_primitive: int,
) -> str:
    """Bind every proposal restriction that can change the candidate policy."""

    contract = qwen_proposal_prompt_contract(
        enabled_primitives=enabled_primitives,
        max_candidates=max_candidates,
        max_per_primitive=max_per_primitive,
    )
    primitive_values = tuple(
        primitive.value for primitive in contract.enabled_primitives
    )
    payload = {
        "schema_version": "method-v1-qwen-proposer-identity-v2",
        "provider_id": _clean_text(provider_id, name="provider_id"),
        "prompt_contract_sha256": contract.fingerprint,
        "proposal_schema_sha256": proposal_schema_fingerprint(
            enabled_primitives=contract.enabled_primitives,
            max_candidates=contract.max_candidates,
            max_per_primitive=contract.max_per_primitive,
        ),
        "proposal_camera": _clean_text(proposal_camera, name="proposal_camera"),
        "enabled_primitives": list(primitive_values),
        "max_candidates": contract.max_candidates,
        "max_per_primitive": contract.max_per_primitive,
        "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": 512},
        "public_history_template": QWEN_PROPOSAL_HISTORY_VERSION,
        "ranking_semantics": contract.ranking_semantics,
    }
    return f"method-v1-qwen-proposer@{canonical_sha256(payload)}"


def _clean_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = " ".join(value.split())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if "<|" in result or "|>" in result:
        raise ValueError(f"{name} contains a reserved model control-token pattern")
    return result


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


@dataclasses.dataclass(frozen=True)
class ProposalRejection:
    """One deterministic reason why a raw proposal did not enter the set."""

    index: int | None
    code: str
    detail: str

    def __post_init__(self) -> None:
        if self.index is not None and (
            not isinstance(self.index, int)
            or isinstance(self.index, bool)
            or self.index < 0
        ):
            raise ValueError("proposal rejection index must be non-negative")
        object.__setattr__(self, "code", _clean_text(self.code, name="code"))
        object.__setattr__(self, "detail", _clean_text(self.detail, name="detail"))

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "code": self.code, "detail": self.detail}


class ProposalFailure(ValueError):
    """Raised when no policy-eligible proposal survives validation."""

    def __init__(
        self,
        message: str,
        *,
        rejections: Sequence[ProposalRejection] = (),
        raw_response: str | None = None,
        processed_image_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.rejections = tuple(rejections)
        self.raw_response = raw_response
        self.processed_image_size = processed_image_size

    def with_generation(
        self,
        *,
        raw_response: str,
        processed_width: int,
        processed_height: int,
    ) -> ProposalFailure:
        return ProposalFailure(
            str(self),
            rejections=self.rejections,
            raw_response=raw_response,
            processed_image_size=(processed_width, processed_height),
        )

    def audit_dict(self) -> dict[str, Any]:
        """Preserve failed generation evidence without inventing candidates."""

        raw_digest = (
            None
            if self.raw_response is None
            else hashlib.sha256(self.raw_response.encode("utf-8")).hexdigest()
        )
        return {
            "message": str(self),
            "raw_response": self.raw_response,
            "raw_response_sha256": raw_digest,
            "processed_image_size": (
                None
                if self.processed_image_size is None
                else list(self.processed_image_size)
            ),
            "rejections": [item.to_dict() for item in self.rejections],
        }


@dataclasses.dataclass(frozen=True)
class ProposalParseResult:
    """Validated candidates plus a complete audit of discarded raw items."""

    candidates: tuple[GroundedIntervention, ...]
    rejections: tuple[ProposalRejection, ...]
    raw_response: str
    raw_response_sha256: str
    processed_image_size: tuple[int, int]
    prompt_contract_sha256: str
    enabled_primitives: tuple[Primitive, ...]
    max_candidates: int
    max_per_primitive: int
    coordinate_contract: str = QWEN_COORDINATE_CONTRACT
    ranking_semantics: str = QWEN_PROPOSAL_RANKING_SEMANTICS

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("proposal result requires at least one candidate")
        contract = qwen_proposal_prompt_contract(
            enabled_primitives=self.enabled_primitives,
            max_candidates=self.max_candidates,
            max_per_primitive=self.max_per_primitive,
        )
        if self.prompt_contract_sha256 != contract.fingerprint:
            raise ValueError("proposal result changed the frozen prompt contract")
        if len(self.candidates) > self.max_candidates:
            raise ValueError("proposal result exceeds its frozen total limit")
        counts = Counter(candidate.primitive for candidate in self.candidates)
        if any(count > self.max_per_primitive for count in counts.values()):
            raise ValueError("proposal result exceeds its per-primitive limit")
        if any(
            candidate.primitive not in contract.enabled_primitives
            for candidate in self.candidates
        ):
            raise ValueError("proposal result contains a disabled primitive")
        if self.ranking_semantics != contract.ranking_semantics:
            raise ValueError("proposal result changed frozen ranking semantics")
        if len(self.raw_response_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.raw_response_sha256
        ):
            raise ValueError("raw_response_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.raw_response, str) or not self.raw_response:
            raise ValueError("raw_response must preserve the non-empty Qwen output")
        if hashlib.sha256(self.raw_response.encode("utf-8")).hexdigest() != (
            self.raw_response_sha256
        ):
            raise ValueError("raw Qwen response and its digest differ")
        if len(self.processed_image_size) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in self.processed_image_size
        ):
            raise ValueError("processed_image_size must contain positive integers")

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "candidates": [item.policy_payload() for item in self.candidates],
                "rejections": [item.to_dict() for item in self.rejections],
                "raw_response": self.raw_response,
                "raw_response_sha256": self.raw_response_sha256,
                "processed_image_size": list(self.processed_image_size),
                "coordinate_contract": self.coordinate_contract,
                "prompt_contract_sha256": self.prompt_contract_sha256,
                "enabled_primitives": [item.value for item in self.enabled_primitives],
                "max_candidates": self.max_candidates,
                "max_per_primitive": self.max_per_primitive,
                "ranking_semantics": self.ranking_semantics,
            }
        )

    def audit_dict(self) -> dict[str, Any]:
        """Return the complete public proposer output and deterministic parse audit."""

        return {
            "raw_response": self.raw_response,
            "raw_response_sha256": self.raw_response_sha256,
            "processed_image_size": list(self.processed_image_size),
            "coordinate_contract": self.coordinate_contract,
            "prompt_contract_sha256": self.prompt_contract_sha256,
            "enabled_primitives": [item.value for item in self.enabled_primitives],
            "max_candidates": self.max_candidates,
            "max_per_primitive": self.max_per_primitive,
            "ranking_semantics": self.ranking_semantics,
            "accepted_candidate_ids": [item.candidate_id for item in self.candidates],
            "accepted_candidate_fingerprints": [
                item.fingerprint() for item in self.candidates
            ],
            "rejections": [item.to_dict() for item in self.rejections],
            "proposal_result_sha256": self.fingerprint,
        }


def absolute_processed_bbox_to_normalized_original(
    bbox: Sequence[float],
    *,
    processed_width: int,
    processed_height: int,
    original_width: int,
    original_height: int,
) -> tuple[float, float, float, float]:
    """Convert Qwen processed-image pixels to normalized original-image xyxy.

    The function intentionally does not clamp.  A coordinate outside the
    declared processed image is invalid evidence and must be rejected rather
    than silently moved onto a visible object.
    """

    dimensions = {
        "processed_width": processed_width,
        "processed_height": processed_height,
        "original_width": original_width,
        "original_height": original_height,
    }
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in dimensions.values()
    ):
        raise ValueError("processed and original image dimensions must be positive")
    values = tuple(bbox)
    if len(values) != 4:
        raise ValueError("bbox must contain exactly four coordinates")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("bbox coordinates must be finite JSON numbers")
    x0, y0, x1, y1 = (float(value) for value in values)
    if not (
        0.0 <= x0 < x1 <= float(processed_width)
        and 0.0 <= y0 < y1 <= float(processed_height)
    ):
        raise ValueError("bbox is empty or outside the processed image")

    # Explicit processed -> original -> normalized-original transform.
    original_box = (
        x0 * original_width / processed_width,
        y0 * original_height / processed_height,
        x1 * original_width / processed_width,
        y1 * original_height / processed_height,
    )
    normalized = (
        original_box[0] / original_width,
        original_box[1] / original_height,
        original_box[2] / original_width,
        original_box[3] / original_height,
    )
    if not (
        0.0 <= normalized[0] < normalized[2] <= 1.0
        and 0.0 <= normalized[1] < normalized[3] <= 1.0
    ):
        raise AssertionError("validated coordinate transform left normalized bounds")
    return normalized


def _candidate_from_raw(
    raw: Any,
    *,
    index: int,
    context: PolicyContext,
    frame_id: str,
    processed_width: int,
    processed_height: int,
    enabled_primitives: tuple[Primitive, ...],
) -> GroundedIntervention:
    if not isinstance(raw, Mapping):
        raise TypeError("candidate must be a JSON object")
    observed_keys = set(raw)
    if observed_keys != _PROPOSAL_KEYS:
        missing = sorted(_PROPOSAL_KEYS - observed_keys)
        extra = sorted(observed_keys - _PROPOSAL_KEYS)
        raise ValueError(f"candidate keys mismatch; missing={missing}, extra={extra}")
    visible_label = _clean_text(raw["visible_label"], name="visible_label")
    instruction = _clean_text(raw["instruction"], name="instruction")
    if not isinstance(raw["bbox"], list):
        raise TypeError("bbox must be a JSON array")
    try:
        primitive = Primitive(_clean_text(raw["primitive"], name="primitive"))
    except ValueError as error:
        raise ValueError(
            "primitive is unknown or not in canonical uppercase form"
        ) from error
    if primitive not in enabled_primitives:
        raise ValueError(f"primitive {primitive.value} is disabled for Method V1")

    frame = context.frame_by_id(frame_id)
    if frame.frame_index != context.latest_frame_index(frame.camera):
        raise ValueError("proposal frame is not the current image for its camera")
    normalized = absolute_processed_bbox_to_normalized_original(
        raw["bbox"],
        processed_width=processed_width,
        processed_height=processed_height,
        original_width=frame.width,
        original_height=frame.height,
    )
    point = (
        (normalized[0] + normalized[2]) / 2.0,
        (normalized[1] + normalized[3]) / 2.0,
    )
    grounding = GroundingReference(
        camera=frame.camera,
        frame_id=frame.frame_id,
        frame_index=frame.frame_index,
        image_sha256=frame.image_sha256,
        box_xyxy=normalized,
        point_xy=point,
    )
    public_payload = {
        "frame": frame.to_dict(),
        "primitive": primitive.value,
        "visible_label": visible_label,
        "instruction": instruction,
        "box_xyxy": list(normalized),
    }
    candidate_id = f"qwen-{canonical_sha256(public_payload)[:20]}"
    candidate = GroundedIntervention(
        candidate_id=candidate_id,
        primitive=primitive,
        referent=visible_label,
        parameters=(("instruction", instruction),),
        grounding=grounding,
    )
    candidate.validate_against(context)
    return candidate


def parse_qwen_proposal_json(
    response_text: str,
    *,
    context: PolicyContext,
    frame_id: str,
    processed_width: int,
    processed_height: int,
    enabled_primitives: Sequence[Primitive] = METHOD_V1_ENABLED_PRIMITIVES,
    max_candidates: int = METHOD_V1_MAX_CANDIDATES,
    max_per_primitive: int = METHOD_V1_MAX_PER_PRIMITIVE,
) -> ProposalParseResult:
    """Parse one Qwen response without repair, clamping, or oracle fallback."""

    if not isinstance(response_text, str) or not response_text.strip():
        rejection = ProposalRejection(None, "invalid_json", "response is empty")
        raise ProposalFailure(
            "Qwen proposal response is empty", rejections=(rejection,)
        )
    if len(response_text.encode("utf-8")) > 65_536:
        rejection = ProposalRejection(None, "invalid_json", "response exceeds 64 KiB")
        raise ProposalFailure(
            "Qwen proposal response is too large", rejections=(rejection,)
        )
    if not isinstance(context, PolicyContext):
        raise TypeError("context must be a PolicyContext")
    enabled = tuple(enabled_primitives)
    if not enabled or any(not isinstance(item, Primitive) for item in enabled):
        raise ValueError("enabled_primitives must contain Primitive values")
    if len(set(enabled)) != len(enabled):
        raise ValueError("enabled_primitives contains duplicates")
    if (
        not isinstance(max_candidates, int)
        or isinstance(max_candidates, bool)
        or not 1 <= max_candidates <= METHOD_V1_MAX_CANDIDATES
    ):
        raise ValueError("max_candidates must be in [1, 6]")
    if (
        not isinstance(max_per_primitive, int)
        or isinstance(max_per_primitive, bool)
        or not 1 <= max_per_primitive <= METHOD_V1_MAX_PER_PRIMITIVE
    ):
        raise ValueError("max_per_primitive must be in [1, 3]")
    contract = qwen_proposal_prompt_contract(
        enabled_primitives=enabled,
        max_candidates=max_candidates,
        max_per_primitive=max_per_primitive,
    )

    try:
        payload = json.loads(
            response_text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        rejection = ProposalRejection(None, "invalid_json", str(error))
        raise ProposalFailure(
            "Qwen proposal response is not strict JSON", rejections=(rejection,)
        ) from error
    if not isinstance(payload, Mapping) or set(payload) != _TOP_LEVEL_KEYS:
        rejection = ProposalRejection(
            None,
            "invalid_schema",
            'top-level JSON must contain exactly the key "candidates"',
        )
        raise ProposalFailure(
            "Qwen proposal schema is invalid", rejections=(rejection,)
        )
    raw_candidates = payload["candidates"]
    if not isinstance(raw_candidates, list):
        rejection = ProposalRejection(
            None, "invalid_schema", "candidates must be a JSON array"
        )
        raise ProposalFailure(
            "Qwen proposal schema is invalid", rejections=(rejection,)
        )

    accepted: list[GroundedIntervention] = []
    rejections: list[ProposalRejection] = []
    primitive_counts: Counter[Primitive] = Counter()
    seen_instruction_regions: set[tuple[str, str, tuple[float, ...]]] = set()
    for index, raw in enumerate(raw_candidates):
        try:
            candidate = _candidate_from_raw(
                raw,
                index=index,
                context=context,
                frame_id=frame_id,
                processed_width=processed_width,
                processed_height=processed_height,
                enabled_primitives=enabled,
            )
        except (TypeError, ValueError) as error:
            rejections.append(ProposalRejection(index, "invalid_candidate", str(error)))
            continue

        assert candidate.grounding is not None
        instruction = dict(candidate.parameters)["instruction"]
        duplicate_key = (
            instruction.casefold(),
            candidate.grounding.frame_id,
            tuple(round(value, 12) for value in candidate.grounding.box_xyxy),
        )
        if duplicate_key in seen_instruction_regions:
            rejections.append(
                ProposalRejection(
                    index,
                    "duplicate_instruction_region",
                    "the same instruction and exact public region was already accepted",
                )
            )
            continue
        if primitive_counts[candidate.primitive] >= max_per_primitive:
            rejections.append(
                ProposalRejection(
                    index,
                    "per_primitive_limit",
                    f"{candidate.primitive.value} already has {max_per_primitive} candidates",
                )
            )
            continue
        if len(accepted) >= max_candidates:
            rejections.append(
                ProposalRejection(
                    index,
                    "total_limit",
                    f"candidate set already contains {max_candidates} candidates",
                )
            )
            continue
        seen_instruction_regions.add(duplicate_key)
        primitive_counts[candidate.primitive] += 1
        accepted.append(candidate)

    if not accepted:
        if not rejections:
            rejections.append(
                ProposalRejection(
                    None, "empty_candidate_set", "candidate array is empty"
                )
            )
        raise ProposalFailure(
            "no valid public candidate survived proposal validation",
            rejections=tuple(rejections),
        )

    return ProposalParseResult(
        candidates=tuple(accepted),
        rejections=tuple(rejections),
        raw_response=response_text,
        raw_response_sha256=hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
        processed_image_size=(processed_width, processed_height),
        enabled_primitives=enabled,
        max_candidates=max_candidates,
        max_per_primitive=max_per_primitive,
        prompt_contract_sha256=contract.fingerprint,
    )


class ProposalRuntime(Protocol):
    """Narrow generation surface supplied by ``Qwen25VLRuntime``."""

    @property
    def provider_id(self) -> str: ...

    def generate_proposal_json(
        self,
        *,
        task_prompt: str,
        public_history_text: str,
        image: Any,
        camera_label: str,
        frame_id: str,
        enabled_primitives: Sequence[Primitive],
        max_candidates: int,
        max_per_primitive: int,
    ) -> Any: ...


def proposal_public_history_text(context: PolicyContext) -> str:
    """Serialize only policy-visible completed actions for proposal context."""

    if not isinstance(context, PolicyContext):
        raise TypeError("context must be a PolicyContext")
    if not context.public_history:
        return "Completed public action history: none."
    rows = ["Completed public action history:"]
    for event in context.public_history:
        rows.append(
            f"{event.step_index}. {event.primitive.value}: {event.subtask_text}; "
            f"status={event.execution_status.value}"
        )
    return "\n".join(rows)


class FrameResolver(Protocol):
    def resolve(self, frame: Any) -> Any: ...


class Qwen25VLProposer:
    """Generate exactly once per public context and cache the validated set."""

    def __init__(
        self,
        *,
        runtime: ProposalRuntime,
        frame_store: FrameResolver,
        proposal_camera: str = "agentview",
        enabled_primitives: Sequence[Primitive] = METHOD_V1_ENABLED_PRIMITIVES,
        max_candidates: int = METHOD_V1_MAX_CANDIDATES,
        max_per_primitive: int = METHOD_V1_MAX_PER_PRIMITIVE,
    ) -> None:
        self.runtime = runtime
        self.frame_store = frame_store
        self.proposal_camera = _clean_text(proposal_camera, name="proposal_camera")
        self.enabled_primitives = tuple(enabled_primitives)
        self.max_candidates = max_candidates
        self.max_per_primitive = max_per_primitive
        self.prompt_contract = qwen_proposal_prompt_contract(
            enabled_primitives=self.enabled_primitives,
            max_candidates=self.max_candidates,
            max_per_primitive=self.max_per_primitive,
        )
        self._cache: dict[str, ProposalParseResult] = {}

    @property
    def proposer_id(self) -> str:
        return qwen_proposer_id(
            provider_id=self.runtime.provider_id,
            proposal_camera=self.proposal_camera,
            enabled_primitives=self.enabled_primitives,
            max_candidates=self.max_candidates,
            max_per_primitive=self.max_per_primitive,
        )

    def propose(self, context: PolicyContext) -> ProposalParseResult:
        if not isinstance(context, PolicyContext):
            raise TypeError("context must be a PolicyContext")
        cache_key = canonical_sha256(
            {
                "context_fingerprint": context.fingerprint(),
                "proposer_id": self.proposer_id,
                "enabled_primitives": [item.value for item in self.enabled_primitives],
                "max_candidates": self.max_candidates,
                "max_per_primitive": self.max_per_primitive,
            }
        )
        if cache_key in self._cache:
            return self._cache[cache_key]
        current_index = context.latest_frame_index(self.proposal_camera)
        frames = [
            frame
            for frame in context.frames
            if frame.camera == self.proposal_camera
            and frame.frame_index == current_index
        ]
        if len(frames) != 1:
            raise ProposalFailure("proposal camera has no unique current public frame")
        frame = frames[0]
        generation = self.runtime.generate_proposal_json(
            task_prompt=context.prompt,
            public_history_text=proposal_public_history_text(context),
            image=self.frame_store.resolve(frame),
            camera_label=frame.camera,
            frame_id=frame.frame_id,
            enabled_primitives=self.enabled_primitives,
            max_candidates=self.max_candidates,
            max_per_primitive=self.max_per_primitive,
        )
        try:
            raw_json = str(generation.raw_json)
            processed_width = int(generation.processed_width)
            processed_height = int(generation.processed_height)
            prompt_contract_sha256 = str(generation.prompt_contract_sha256)
            rendered_user_prompt_sha256 = str(generation.rendered_user_prompt_sha256)
        except (AttributeError, TypeError, ValueError) as error:
            raise ProposalFailure(
                "proposal runtime returned an invalid result"
            ) from error
        expected_user_text = proposal_request_text(
            task_prompt=context.prompt,
            public_history_text=proposal_public_history_text(context),
            camera_label=frame.camera,
            frame_id=frame.frame_id,
            processed_width=processed_width,
            processed_height=processed_height,
            enabled_primitives=self.enabled_primitives,
            max_candidates=self.max_candidates,
            max_per_primitive=self.max_per_primitive,
        )
        if prompt_contract_sha256 != self.prompt_contract.fingerprint:
            raise ProposalFailure("proposal runtime changed the frozen prompt contract")
        if (
            rendered_user_prompt_sha256
            != hashlib.sha256(expected_user_text.encode("utf-8")).hexdigest()
        ):
            raise ProposalFailure("proposal runtime changed the rendered user prompt")
        try:
            result = parse_qwen_proposal_json(
                raw_json,
                context=context,
                frame_id=frame.frame_id,
                processed_width=processed_width,
                processed_height=processed_height,
                enabled_primitives=self.enabled_primitives,
                max_candidates=self.max_candidates,
                max_per_primitive=self.max_per_primitive,
            )
        except ProposalFailure as error:
            raise error.with_generation(
                raw_response=raw_json,
                processed_width=processed_width,
                processed_height=processed_height,
            ) from error
        self._cache[cache_key] = result
        return result


def proposal_request_text(
    *,
    task_prompt: str,
    public_history_text: str,
    camera_label: str,
    frame_id: str,
    processed_width: int,
    processed_height: int,
    enabled_primitives: Sequence[Primitive] = METHOD_V1_ENABLED_PRIMITIVES,
    max_candidates: int = METHOD_V1_MAX_CANDIDATES,
    max_per_primitive: int = METHOD_V1_MAX_PER_PRIMITIVE,
) -> str:
    """Render one of two fixed public proposal request templates."""

    contract = qwen_proposal_prompt_contract(
        enabled_primitives=enabled_primitives,
        max_candidates=max_candidates,
        max_per_primitive=max_per_primitive,
    )
    values = {
        "task_prompt": _clean_text(task_prompt, name="task_prompt"),
        "public_history_text": _clean_text(
            public_history_text, name="public_history_text"
        ),
        "camera_label": _clean_text(camera_label, name="camera_label"),
        "frame_id": _clean_text(frame_id, name="frame_id"),
        "processed_width": processed_width,
        "processed_height": processed_height,
    }
    if processed_width < 1 or processed_height < 1:
        raise ValueError("processed image dimensions must be positive")
    return contract.user_template.format(**values)


def proposal_schema_fingerprint(
    *,
    enabled_primitives: Sequence[Primitive] = METHOD_V1_ENABLED_PRIMITIVES,
    max_candidates: int = METHOD_V1_MAX_CANDIDATES,
    max_per_primitive: int = METHOD_V1_MAX_PER_PRIMITIVE,
) -> str:
    contract = qwen_proposal_prompt_contract(
        enabled_primitives=enabled_primitives,
        max_candidates=max_candidates,
        max_per_primitive=max_per_primitive,
    )
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "prompt_contract": contract.to_dict(),
                "top_level_keys": sorted(_TOP_LEVEL_KEYS),
                "candidate_keys": sorted(_PROPOSAL_KEYS),
                "coordinate_contract": QWEN_COORDINATE_CONTRACT,
            }
        )
    ).hexdigest()
