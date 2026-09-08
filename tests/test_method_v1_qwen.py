from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grounded_interaction.contracts import (
    ExecutionStatus,
    PolicyContext,
    Primitive,
    PublicActionEvent,
)
from grounded_interaction.proposals import (
    QWEN_DIRECT_ONLY_SYSTEM_PROMPT,
    QWEN_PROPOSAL_SYSTEM_PROMPT,
    ProposalFailure,
    Qwen25VLProposer,
    absolute_processed_bbox_to_normalized_original,
    parse_qwen_proposal_json,
    proposal_request_text,
    qwen_proposal_prompt_contract,
)
from grounded_interaction.qwen_provider import (
    METHOD_V1_PRIMITIVE_ORDER,
    QWEN25VL_MODEL_ID,
    QWEN25VL_REVISION,
    Qwen25VLRuntime,
    Qwen25VLTokenProvider,
    QwenContextEncoding,
    QwenFeatureCache,
    QwenProposalGeneration,
    QwenProviderIdentity,
    image_token_position_runs,
    merged_patch_boxes,
    processed_image_size_from_grid,
    public_context_content_blocks,
    save_grounding_support_overlay,
    select_public_context_frames,
)
from grounded_interaction.rgb import RGBFrameStore

torch = pytest.importorskip("torch")


def _context(
    tmp_path: Path, *, history: bool = False
) -> tuple[PolicyContext, RGBFrameStore]:
    store = RGBFrameStore(tmp_path / "frames")
    agent = np.zeros((40, 80, 3), dtype=np.uint8)
    agent[:, :, 0] = 90
    wrist = np.zeros((60, 30, 3), dtype=np.uint8)
    wrist[:, :, 1] = 140
    frames = [
        store.put(agent, frame_id="agent-0", camera="agentview", frame_index=0),
        store.put(wrist, frame_id="wrist-0", camera="wrist", frame_index=0),
    ]
    if history:
        frames.extend(
            [
                store.put(
                    agent + 1,
                    frame_id="agent-1",
                    camera="agentview",
                    frame_index=1,
                ),
                store.put(
                    wrist + 1,
                    frame_id="wrist-1",
                    camera="wrist",
                    frame_index=1,
                ),
            ]
        )
    return PolicyContext(
        prompt="Put the butter in the basket.", frames=tuple(frames)
    ), store


def _response(items: list[dict[str, object]]) -> str:
    return json.dumps({"candidates": items}, separators=(",", ":"))


def _raw_candidate(
    *,
    label: str,
    bbox: list[float],
    primitive: str,
    instruction: str,
) -> dict[str, object]:
    return {
        "visible_label": label,
        "bbox": bbox,
        "primitive": primitive,
        "instruction": instruction,
    }


def test_frozen_qwen_candidate_order_has_explicit_baseline_semantics() -> None:
    prompt = " ".join(QWEN_PROPOSAL_SYSTEM_PROMPT.split())
    assert "most to least likely to complete" in prompt
    assert "DIRECT receives up to 300 control steps" in prompt
    assert "OPEN receives 100 steps" in prompt
    assert "remaining 200 steps" in prompt
    assert "do not emit confidence numbers" in prompt


def test_post_information_contract_is_ranked_direct_only_and_rejects_open(
    tmp_path: Path,
) -> None:
    prompt = " ".join(QWEN_DIRECT_ONLY_SYSTEM_PROMPT.split())
    assert 'primitive is exactly "DIRECT"' in prompt
    assert "at most 3 DIRECT candidates" in prompt
    assert "remaining 200 control steps" in prompt
    assert "most to least likely to complete" in prompt
    assert 'primitive is exactly "DIRECT" or "OPEN"' not in prompt

    context, _ = _context(tmp_path)
    result = parse_qwen_proposal_json(
        _response(
            [
                _raw_candidate(
                    label="drawer handle",
                    bbox=[0, 0, 20, 20],
                    primitive="OPEN",
                    instruction="Open the visible drawer.",
                ),
                _raw_candidate(
                    label="butter",
                    bbox=[20, 0, 40, 20],
                    primitive="DIRECT",
                    instruction="Put the visible butter in the basket.",
                ),
            ]
        ),
        context=context,
        frame_id="agent-0",
        processed_width=80,
        processed_height=40,
        enabled_primitives=(Primitive.DIRECT,),
        max_candidates=3,
        max_per_primitive=3,
    )
    assert [item.primitive for item in result.candidates] == [Primitive.DIRECT]
    assert "disabled" in result.rejections[0].detail


def test_absolute_processed_bbox_conversion_is_explicit_and_non_square() -> None:
    assert absolute_processed_bbox_to_normalized_original(
        [12, 8, 60, 32],
        processed_width=120,
        processed_height=40,
        original_width=1920,
        original_height=640,
    ) == pytest.approx((0.1, 0.2, 0.5, 0.8))
    with pytest.raises(ValueError, match="outside"):
        absolute_processed_bbox_to_normalized_original(
            [-1, 0, 30, 20],
            processed_width=120,
            processed_height=40,
            original_width=1920,
            original_height=640,
        )
    with pytest.raises(ValueError, match="finite"):
        absolute_processed_bbox_to_normalized_original(
            [0, 0, float("nan"), 20],
            processed_width=120,
            processed_height=40,
            original_width=1920,
            original_height=640,
        )


def test_strict_proposal_parser_keeps_only_public_direct_and_open(
    tmp_path: Path,
) -> None:
    context, _ = _context(tmp_path)
    response = _response(
        [
            _raw_candidate(
                label="red box",
                bbox=[8, 4, 40, 20],
                primitive="DIRECT",
                instruction="Pick up the visible red box and place it in the basket.",
            ),
            _raw_candidate(
                label="middle drawer handle",
                bbox=[48, 20, 72, 36],
                primitive="OPEN",
                instruction="Open the visible middle drawer by its handle.",
            ),
            _raw_candidate(
                label="unknown",
                bbox=[0, 0, 5, 5],
                primitive="REMOVE",
                instruction="Remove it.",
            ),
        ]
    )
    result = parse_qwen_proposal_json(
        response,
        context=context,
        frame_id="agent-0",
        processed_width=80,
        processed_height=40,
    )
    assert [candidate.primitive for candidate in result.candidates] == [
        Primitive.DIRECT,
        Primitive.OPEN,
    ]
    assert result.candidates[0].grounding is not None
    assert result.candidates[0].grounding.box_xyxy == pytest.approx(
        (0.1, 0.1, 0.5, 0.5)
    )
    assert dict(result.candidates[0].parameters)["instruction"].startswith("Pick up")
    assert result.rejections[0].code == "invalid_candidate"
    assert "disabled" in result.rejections[0].detail
    assert len(result.raw_response_sha256) == 64


def test_proposal_parser_rejects_repairs_duplicates_and_excess(tmp_path: Path) -> None:
    context, _ = _context(tmp_path)
    items = [
        _raw_candidate(
            label=f"object {index}",
            bbox=[index * 5, 0, index * 5 + 4, 4],
            primitive="DIRECT",
            instruction=f"Move visible object {index} to the basket.",
        )
        for index in range(4)
    ]
    items.append(dict(items[0]))
    items.extend(
        _raw_candidate(
            label=f"drawer {index}",
            bbox=[index * 5, 10, index * 5 + 4, 14],
            primitive="OPEN",
            instruction=f"Open visible drawer {index}.",
        )
        for index in range(4)
    )
    result = parse_qwen_proposal_json(
        _response(items),
        context=context,
        frame_id="agent-0",
        processed_width=80,
        processed_height=40,
    )
    assert len(result.candidates) == 6
    assert [candidate.primitive for candidate in result.candidates].count(
        Primitive.DIRECT
    ) == 3
    assert [candidate.primitive for candidate in result.candidates].count(
        Primitive.OPEN
    ) == 3
    codes = [item.code for item in result.rejections]
    assert "per_primitive_limit" in codes
    # The repeated item occurs after the first three DIRECT proposals; duplicate
    # detection precedes cardinality handling and remains auditable.
    assert "duplicate_instruction_region" in codes

    for malformed in (
        '```json\n{"candidates":[]}\n```',
        '{"candidates":[],"extra":1}',
        (
            '{"candidates":[{"visible_label":"x","bbox":[0,0,NaN,1],'
            '"primitive":"DIRECT","instruction":"do"}]}'
        ),
        '{"candidates":[],"candidates":[]}',
    ):
        with pytest.raises(ProposalFailure):
            parse_qwen_proposal_json(
                malformed,
                context=context,
                frame_id="agent-0",
                processed_width=80,
                processed_height=40,
            )


def test_all_invalid_is_a_proposal_failure_with_rejection_log(tmp_path: Path) -> None:
    context, _ = _context(tmp_path)
    with pytest.raises(ProposalFailure) as caught:
        parse_qwen_proposal_json(
            _response(
                [
                    _raw_candidate(
                        label="bad",
                        bbox=[10, 10, 10, 11],
                        primitive="DIRECT",
                        instruction="Try the bad zero-width box.",
                    )
                ]
            ),
            context=context,
            frame_id="agent-0",
            processed_width=80,
            processed_height=40,
        )
    assert caught.value.rejections
    assert caught.value.rejections[0].index == 0


def test_candidate_identity_does_not_depend_on_qwen_array_order(tmp_path: Path) -> None:
    context, _ = _context(tmp_path)
    direct = _raw_candidate(
        label="red box",
        bbox=[8, 4, 40, 20],
        primitive="DIRECT",
        instruction="Put the visible red box in the basket.",
    )
    opening = _raw_candidate(
        label="drawer handle",
        bbox=[48, 20, 72, 36],
        primitive="OPEN",
        instruction="Open the visible middle drawer by its handle.",
    )
    first = parse_qwen_proposal_json(
        _response([direct, opening]),
        context=context,
        frame_id="agent-0",
        processed_width=80,
        processed_height=40,
    )
    second = parse_qwen_proposal_json(
        _response([opening, direct]),
        context=context,
        frame_id="agent-0",
        processed_width=80,
        processed_height=40,
    )
    first_ids = {item.primitive: item.candidate_id for item in first.candidates}
    second_ids = {item.primitive: item.candidate_id for item in second.candidates}
    assert first_ids == second_ids


def test_grid_mapping_uses_processor_shape_not_square_root(tmp_path: Path) -> None:
    grid = (1, 8, 12)
    boxes = merged_patch_boxes(grid, spatial_merge_size=2)
    assert len(boxes) == 24
    assert boxes[0] == pytest.approx((0.0, 0.0, 1 / 6, 1 / 4))
    assert boxes[5] == pytest.approx((5 / 6, 0.0, 1.0, 1 / 4))
    assert boxes[-1] == pytest.approx((5 / 6, 3 / 4, 1.0, 1.0))
    assert processed_image_size_from_grid(grid, patch_size=14) == (168, 112)

    # Two non-square image runs are separated by an ordinary language token.
    ids = [1, *([99] * 24), 2, *([99] * 8), 3]
    runs = image_token_position_runs(
        ids,
        image_token_id=99,
        image_grid_thw=(grid, (1, 4, 8)),
        spatial_merge_size=2,
    )
    assert (len(runs[0]), len(runs[1])) == (24, 8)

    image = np.full((112, 168, 3), 245, dtype=np.uint8)
    support = [index in {7, 8, 13, 14} for index in range(len(boxes))]
    output = save_grounding_support_overlay(
        image,
        patch_boxes=boxes,
        support_mask=support,
        candidate_box=(0.2, 0.2, 0.5, 0.7),
        output_path=tmp_path / "mapping-overlay.png",
    )
    assert output.is_file() and output.stat().st_size > 100


def test_context_selection_keeps_current_plus_only_two_prior_boundaries(
    tmp_path: Path,
) -> None:
    store = RGBFrameStore(tmp_path / "history-frames")
    frames = []
    for frame_index in range(4):
        for camera, width in (("agentview", 80), ("wrist", 30)):
            pixels = np.full((40, width, 3), frame_index * 20, dtype=np.uint8)
            frames.append(
                store.put(
                    pixels,
                    frame_id=f"{camera}-{frame_index}",
                    camera=camera,
                    frame_index=frame_index,
                )
            )
    context = PolicyContext(prompt="Find the requested package.", frames=tuple(frames))
    selected = select_public_context_frames(context, frame_store=store)
    assert {item.frame.frame_index for item in selected} == {1, 2, 3}
    assert len(selected) == 6
    assert {item.frame.camera for item in selected if item.is_current} == {
        "agentview",
        "wrist",
    }


def test_public_context_text_blocks_have_explicit_temporal_delimiters(
    tmp_path: Path,
) -> None:
    base, store = _context(tmp_path, history=True)
    context = PolicyContext(
        prompt=base.prompt,
        frames=base.frames,
        public_history=(
            PublicActionEvent(
                step_index=0,
                primitive=Primitive.OPEN,
                subtask_text="Open the visible middle drawer.",
                execution_status=ExecutionStatus.COMPLETED,
            ),
        ),
    )
    selected = select_public_context_frames(context, frame_store=store)
    blocks = public_context_content_blocks(context, selected)
    text_blocks = [block["text"] for block in blocks if block["type"] == "text"]
    assert all(text.endswith("\n") for text in text_blocks)
    rendered = "".join(text_blocks)
    assert "basket.\nhistory public RGB" in rendered
    assert "status=COMPLETED\ncurrent public RGB" in rendered


class _FakeRuntime(Qwen25VLRuntime):
    def __init__(self) -> None:
        self.identity = QwenProviderIdentity()
        self.proposal_calls = 0
        self.proposal_histories: list[str] = []
        self.proposal_modes: list[str] = []
        self.context_calls = 0
        self.candidate_calls = 0

    def generate_proposal_json(
        self,
        *,
        task_prompt: str,
        public_history_text: str,
        image: object,
        camera_label: str,
        frame_id: str,
        enabled_primitives: tuple[Primitive, ...],
        max_candidates: int,
        max_per_primitive: int,
    ) -> QwenProposalGeneration:
        del image
        self.proposal_calls += 1
        self.proposal_histories.append(public_history_text)
        contract = qwen_proposal_prompt_contract(
            enabled_primitives=enabled_primitives,
            max_candidates=max_candidates,
            max_per_primitive=max_per_primitive,
        )
        self.proposal_modes.append(contract.mode)
        user_text = proposal_request_text(
            task_prompt=task_prompt,
            public_history_text=public_history_text,
            camera_label=camera_label,
            frame_id=frame_id,
            processed_width=40,
            processed_height=20,
            enabled_primitives=enabled_primitives,
            max_candidates=max_candidates,
            max_per_primitive=max_per_primitive,
        )
        return QwenProposalGeneration(
            raw_json=_response(
                [
                    _raw_candidate(
                        label="red box",
                        bbox=[1, 1, 20, 20],
                        primitive="DIRECT",
                        instruction="Put the visible red box in the basket.",
                    ),
                    _raw_candidate(
                        label="drawer handle",
                        bbox=[21, 1, 39, 20],
                        primitive="OPEN",
                        instruction="Open the visible drawer by its handle.",
                    ),
                ]
            ),
            processed_width=40,
            processed_height=20,
            prompt_contract_sha256=contract.fingerprint,
            rendered_user_prompt_sha256=hashlib.sha256(
                user_text.encode("utf-8")
            ).hexdigest(),
        )

    def encode_public_context(
        self, *, context: PolicyContext, frame_store: object
    ) -> QwenContextEncoding:
        self.context_calls += 1
        selected = select_public_context_frames(context, frame_store=frame_store)
        assert len(selected) == 2
        # Each [1,4,4] grid becomes four merged image tokens.
        input_ids = torch.tensor([[7, 99, 99, 99, 99, 8, 99, 99, 99, 99, 9]])
        hidden = torch.arange(11 * 8, dtype=torch.float32).reshape(1, 11, 8)
        hidden.requires_grad_(True)
        return QwenContextEncoding(
            hidden_states=hidden,
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
            image_grid_thw=((1, 4, 4), (1, 4, 4)),
            image_token_id=99,
            spatial_merge_size=2,
            patch_size=14,
            selected_frames=selected,
        )

    def encode_candidate_instructions(self, instructions: tuple[str, ...]) -> object:
        self.candidate_calls += 1
        values = torch.arange(len(instructions) * 8, dtype=torch.float32).reshape(
            len(instructions), 8
        )
        return values.requires_grad_(True)


def test_proposer_runs_once_per_context_and_caches(tmp_path: Path) -> None:
    context, store = _context(tmp_path)
    runtime = _FakeRuntime()
    proposer = Qwen25VLProposer(runtime=runtime, frame_store=store)
    first = proposer.propose(context)
    second = proposer.propose(context)
    assert first is second
    assert runtime.proposal_calls == 1
    assert [item.primitive for item in first.candidates] == [
        Primitive.DIRECT,
        Primitive.OPEN,
    ]
    assert runtime.proposal_histories == ["Completed public action history: none."]


def test_post_open_proposal_receives_only_public_completed_history(
    tmp_path: Path,
) -> None:
    context, store = _context(tmp_path)
    context = PolicyContext(
        prompt=context.prompt,
        frames=context.frames,
        public_history=(
            PublicActionEvent(
                step_index=0,
                primitive=Primitive.OPEN,
                subtask_text="Open the visible middle drawer.",
                execution_status=ExecutionStatus.COMPLETED,
            ),
        ),
    )
    runtime = _FakeRuntime()
    Qwen25VLProposer(runtime=runtime, frame_store=store).propose(context)
    assert runtime.proposal_histories == [
        (
            "Completed public action history:\n"
            "0. OPEN: Open the visible middle drawer.; status=COMPLETED"
        )
    ]


def test_post_open_proposer_uses_direct_only_generation_contract(
    tmp_path: Path,
) -> None:
    context, store = _context(tmp_path)
    runtime = _FakeRuntime()
    result = Qwen25VLProposer(
        runtime=runtime,
        frame_store=store,
        enabled_primitives=(Primitive.DIRECT,),
        max_candidates=3,
        max_per_primitive=3,
    ).propose(context)
    assert runtime.proposal_modes == ["POST_INFORMATION_DIRECT_ONLY"]
    assert [item.primitive for item in result.candidates] == [Primitive.DIRECT]
    assert any("OPEN" in item.detail for item in result.rejections)


def test_provider_builds_detached_candidate_features_and_exact_support(
    tmp_path: Path,
) -> None:
    context, store = _context(tmp_path)
    runtime = _FakeRuntime()
    candidates = (
        Qwen25VLProposer(runtime=runtime, frame_store=store).propose(context).candidates
    )
    cache = QwenFeatureCache(tmp_path / "features")
    provider = Qwen25VLTokenProvider(runtime=runtime, frame_store=store, cache=cache)
    field = provider.encode(context, candidates)
    assert field.tokens.shape == (
        1,
        2,
        8 + 4 + len(METHOD_V1_PRIMITIVE_ORDER),
    )
    assert field.public_context.tokens.shape == (1, 11, 8)
    assert int(field.public_context.current_patch_mask.sum()) == 8
    assert field.grounding_support[0, 0].any()
    assert field.grounding_support[0, 1].any()
    assert not field.tokens.requires_grad
    assert not field.public_context.tokens.requires_grad
    assert runtime.context_calls == 1
    assert runtime.candidate_calls == 1

    # Memory cache is used first; a new provider proves the disk cache can be
    # reconstructed without invoking the frozen model again.
    second_runtime = _FakeRuntime()
    second_provider = Qwen25VLTokenProvider(
        runtime=second_runtime, frame_store=store, cache=cache
    )
    restored = second_provider.encode(context, candidates)
    assert torch.equal(restored.tokens, field.tokens)
    assert torch.equal(restored.grounding_support, field.grounding_support)
    assert second_runtime.context_calls == 0
    assert second_runtime.candidate_calls == 0


def test_provider_adds_only_public_raw_state_with_explicit_validity(
    tmp_path: Path,
) -> None:
    context, store = _context(tmp_path)
    context = PolicyContext(
        prompt=context.prompt,
        frames=context.frames,
        proprioception=tuple(float(index) for index in range(8)),
    )
    runtime = _FakeRuntime()
    candidates = (
        Qwen25VLProposer(runtime=runtime, frame_store=store).propose(context).candidates
    )
    field = Qwen25VLTokenProvider(runtime=runtime, frame_store=store).encode(
        context, candidates
    )
    assert bool(field.public_state_valid_mask[0])
    assert field.public_state_values[0].tolist() == pytest.approx(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0]
    )
    assert not field.public_state_values.requires_grad


def test_provider_identity_pins_real_checkpoint_and_processor_version() -> None:
    identity = QwenProviderIdentity()
    assert identity.model_id == QWEN25VL_MODEL_ID
    assert identity.revision == QWEN25VL_REVISION
    assert identity.transformers_version == "4.57.6"
    assert identity.primitive_order == tuple(
        primitive.value for primitive in METHOD_V1_PRIMITIVE_ORDER
    )
    assert QWEN25VL_REVISION in identity.provider_id
    assert len(identity.proposal_prompt_sha256) == 64
    assert len(identity.context_prompt_sha256) == 64
    assert len(identity.candidate_prefix_sha256) == 64


def test_candidate_embedding_pools_last_instruction_content_token() -> None:
    class Tokenizer:
        def __init__(self) -> None:
            self.texts: list[str] = []

        def __call__(self, texts, **kwargs):
            del kwargs
            self.texts = list(texts)
            lengths = [len(text) for text in self.texts]
            width = max(lengths)
            ids = torch.zeros(len(lengths), width, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for row, length in enumerate(lengths):
                ids[row, :length] = torch.arange(1, length + 1)
                mask[row, :length] = 1
            return {"input_ids": ids, "attention_mask": mask}

    class Processor:
        def __init__(self) -> None:
            self.tokenizer = Tokenizer()

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, input_ids, attention_mask, **kwargs):
            del attention_mask, kwargs
            hidden = input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 1, 4)
            return SimpleNamespace(hidden_states=(hidden,))

    processor = Processor()
    runtime = Qwen25VLRuntime(processor=processor, model=Model(), device_map=None)
    values = ("Open the visible drawer.", "Place the visible butter in the basket.")
    pooled = runtime.encode_candidate_instructions(values)
    expected_texts = [f"Candidate action instruction:\n{value}" for value in values]
    assert processor.tokenizer.texts == expected_texts
    assert all("END OF CANDIDATE" not in text for text in processor.tokenizer.texts)
    assert pooled[:, 0].tolist() == pytest.approx(
        [float(len(text)) for text in expected_texts]
    )


@pytest.mark.skipif(
    os.environ.get("IP_RUN_QWEN_PROCESSOR_INTEGRATION") != "1",
    reason="set IP_RUN_QWEN_PROCESSOR_INTEGRATION=1 to use the real pinned processor",
)
def test_real_qwen_processor_non_square_patch_overlay(tmp_path: Path) -> None:
    """Opt-in test of the exact HF processor; it never downloads model weights."""

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        QWEN25VL_MODEL_ID,
        revision=QWEN25VL_REVISION,
        trust_remote_code=False,
        use_fast=False,
    )
    factor = int(processor.image_processor.patch_size) * int(
        processor.image_processor.merge_size
    )
    budget = 256 * factor * factor
    processor.image_processor.min_pixels = budget
    processor.image_processor.max_pixels = budget
    observed_grids = []
    images = (
        np.full((180, 420, 3), 210, dtype=np.uint8),
        np.full((310, 170, 3), 190, dtype=np.uint8),
    )
    for image_index, image in enumerate(images):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Task: inspect the visible object."},
                    {"type": "image"},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        grid = tuple(int(item) for item in inputs["image_grid_thw"][0].tolist())
        observed_grids.append(grid)
        runs = image_token_position_runs(
            inputs["input_ids"][0].tolist(),
            image_token_id=int(processor.image_token_id),
            image_grid_thw=(grid,),
            spatial_merge_size=int(processor.image_processor.merge_size),
        )
        boxes = merged_patch_boxes(
            grid, spatial_merge_size=int(processor.image_processor.merge_size)
        )
        assert len(runs[0]) == len(boxes)
        support = [0.3 <= (box[0] + box[2]) / 2 <= 0.6 for box in boxes]
        save_grounding_support_overlay(
            image,
            patch_boxes=boxes,
            support_mask=support,
            candidate_box=(0.3, 0.15, 0.6, 0.85),
            output_path=tmp_path / f"real-qwen-processor-overlay-{image_index}.png",
        )
    assert observed_grids[0][1:] != observed_grids[1][1:]


@pytest.mark.skipif(
    os.environ.get("IP_RUN_QWEN_MODEL_INTEGRATION") != "1",
    reason="set IP_RUN_QWEN_MODEL_INTEGRATION=1 to load the pinned 3B checkpoint",
)
def test_real_qwen_checkpoint_multiframe_hidden_state_alignment(
    tmp_path: Path,
) -> None:
    """Opt-in checkpoint canary for the exact multi-image token boundary."""

    base, store = _context(tmp_path, history=True)
    context = PolicyContext(
        prompt=base.prompt,
        frames=base.frames,
        public_history=(
            PublicActionEvent(
                step_index=0,
                primitive=Primitive.OPEN,
                subtask_text="Open the visible middle drawer by its handle.",
                execution_status=ExecutionStatus.COMPLETED,
            ),
        ),
    )
    runtime = Qwen25VLRuntime(device_map="auto")
    encoding = runtime.encode_public_context(context=context, frame_store=store)
    assert encoding.hidden_states.shape[:2] == encoding.input_ids.shape
    assert encoding.hidden_states.device.type == "cpu"
    assert encoding.hidden_states.dtype is torch.float32
    assert not encoding.hidden_states.requires_grad
    runs = image_token_position_runs(
        encoding.input_ids[0].tolist(),
        image_token_id=encoding.image_token_id,
        image_grid_thw=encoding.image_grid_thw,
        spatial_merge_size=encoding.spatial_merge_size,
    )
    assert len(runs) == len(encoding.selected_frames)
    for run, grid in zip(runs, encoding.image_grid_thw):
        assert len(run) == int(np.prod(grid)) // encoding.spatial_merge_size**2
    assert any(
        item.frame.width > item.frame.height for item in encoding.selected_frames
    )
    assert any(
        item.frame.height > item.frame.width for item in encoding.selected_frames
    )
    assert all(not parameter.requires_grad for parameter in runtime.model.parameters())
    assert all(parameter.grad is None for parameter in runtime.model.parameters())
