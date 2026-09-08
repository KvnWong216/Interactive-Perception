from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from grounded_interaction.contracts import (
    GroundedIntervention,
    GroundingReference,
    PolicyContext,
    Primitive,
    PublicFrame,
)
from grounded_interaction.qwen_provider import (
    QWEN25VL_TRANSFORMERS_VERSION,
    Qwen25VLRuntime,
    QwenFeatureCache,
    QwenProviderIdentity,
    candidate_feature_cache_key,
)


def _public_inputs() -> tuple[PolicyContext, tuple[GroundedIntervention, ...]]:
    image_sha256 = "a" * 64
    frame = PublicFrame(
        frame_id="agentview-0",
        camera="agentview",
        frame_index=0,
        image_sha256=image_sha256,
        width=32,
        height=32,
    )
    context = PolicyContext(prompt="Put the object in the basket.", frames=(frame,))
    candidate = GroundedIntervention(
        candidate_id="direct-object",
        primitive=Primitive.DIRECT,
        referent="visible object",
        parameters=(("instruction", "Put this object in the basket."),),
        grounding=GroundingReference(
            camera="agentview",
            frame_id=frame.frame_id,
            frame_index=frame.frame_index,
            image_sha256=image_sha256,
            box_xyxy=(0.0, 0.0, 1.0, 1.0),
            point_xy=(0.5, 0.5),
        ),
    )
    return context, (candidate,)


def _field(
    *,
    torch: Any,
    context: PolicyContext,
    candidates: tuple[GroundedIntervention, ...],
    provider_id: str,
    offset: float = 0.0,
) -> Any:
    from grounded_interaction.tokens import CandidateTokenField, FrozenTokenField

    context_tokens = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8) + offset
    public_context = FrozenTokenField(
        tokens=context_tokens,
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
        current_patch_mask=torch.ones((1, 1), dtype=torch.bool),
        camera_ids=(("agentview",),),
        frame_ids=(("agentview-0",),),
        patch_xyxy=torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]),
        context_fingerprints=(context.fingerprint(),),
        provider_id=provider_id,
    )
    return CandidateTokenField(
        tokens=torch.arange(14, dtype=torch.float32).reshape(1, 1, 14) + offset,
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
        candidate_ids=((candidates[0].candidate_id,),),
        candidate_fingerprints=((candidates[0].fingerprint(),),),
        primitives=((Primitive.DIRECT,),),
        grounding_support=torch.ones((1, 1, 1), dtype=torch.bool),
        public_context=public_context,
    )


def test_provider_identity_binds_numeric_and_load_semantics() -> None:
    identity = QwenProviderIdentity(transformers_version=QWEN25VL_TRANSFORMERS_VERSION)
    assert identity.to_dict()["torch_dtype"] == "bfloat16"
    assert identity.to_dict()["quantization_mode"] == "none"
    assert identity.to_dict()["load_mode"] == (
        "transformers-from-pretrained-low-cpu-mem-v1"
    )
    with pytest.raises(ValueError, match="quantized"):
        QwenProviderIdentity(
            transformers_version=QWEN25VL_TRANSFORMERS_VERSION,
            quantization_mode="int8",
        )
    with pytest.raises(ValueError, match="torch.bfloat16"):
        QwenProviderIdentity(
            transformers_version=QWEN25VL_TRANSFORMERS_VERSION,
            torch_dtype="float16",
        )
    with pytest.raises(ValueError, match="load mode"):
        QwenProviderIdentity(
            transformers_version=QWEN25VL_TRANSFORMERS_VERSION,
            load_mode="quantized-device-map",
        )
    with pytest.raises(ValueError, match="dtype"):
        Qwen25VLRuntime(identity=identity, dtype="float32")


def test_physical_cache_identity_differs_from_logical_key_and_loads(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    context, candidates = _public_inputs()
    provider_id = QwenProviderIdentity(
        transformers_version=QWEN25VL_TRANSFORMERS_VERSION
    ).provider_id
    logical_key = candidate_feature_cache_key(
        provider_id=provider_id, context=context, candidates=candidates
    )
    cache = QwenFeatureCache(tmp_path)
    identity = cache.put(
        logical_key,
        _field(
            torch=torch,
            context=context,
            candidates=candidates,
            provider_id=provider_id,
        ),
    )
    assert identity.logical_key == logical_key
    assert identity.artifact_sha256 != logical_key
    with pytest.raises(ValueError, match="physical artifact identity"):
        cache.load(
            logical_key,
            context=context,
            candidates=candidates,
            provider_id=provider_id,
        )
    assert (
        cache.load(
            identity.artifact_sha256,
            context=context,
            candidates=candidates,
            provider_id=provider_id,
        )
        is not None
    )


def test_replacing_tensor_and_sidecar_cannot_preserve_frozen_artifact_identity(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    context, candidates = _public_inputs()
    provider_id = QwenProviderIdentity(
        transformers_version=QWEN25VL_TRANSFORMERS_VERSION
    ).provider_id
    logical_key = candidate_feature_cache_key(
        provider_id=provider_id, context=context, candidates=candidates
    )
    original = QwenFeatureCache(tmp_path / "original")
    replacement = QwenFeatureCache(tmp_path / "replacement")
    original_identity = original.put(
        logical_key,
        _field(
            torch=torch,
            context=context,
            candidates=candidates,
            provider_id=provider_id,
        ),
    )
    replacement_identity = replacement.put(
        logical_key,
        _field(
            torch=torch,
            context=context,
            candidates=candidates,
            provider_id=provider_id,
            offset=100.0,
        ),
    )
    assert replacement_identity.artifact_sha256 != original_identity.artifact_sha256

    original_paths = original._paths(original_identity.artifact_sha256)
    replacement_paths = replacement._paths(replacement_identity.artifact_sha256)
    for source, destination in zip(replacement_paths, original_paths, strict=True):
        shutil.copyfile(source, destination)

    with pytest.raises(ValueError, match="artifact identity mismatch"):
        original.load(
            original_identity.artifact_sha256,
            context=context,
            candidates=candidates,
            provider_id=provider_id,
        )
