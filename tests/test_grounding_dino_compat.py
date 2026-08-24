from __future__ import annotations

import inspect

import pytest

from interaction_uncertainty.grounding_dino_compat import (
    COMPATIBILITY_WRAPPER_VERSION,
    GroundingDinoPostProcessCompatibilityError,
    grounding_dino_post_process_identity,
    grounding_dino_threshold_branch,
    post_process_grounded_object_detection_compat,
)


def test_new_grounding_dino_api_receives_threshold() -> None:
    received = {}

    def post_process(outputs, input_ids=None, threshold=0.25, text_threshold=0.25, target_sizes=None):
        received.update(
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )
        return [outputs, input_ids]

    result = post_process_grounded_object_detection_compat(
        post_process,
        "outputs",
        "input_ids",
        box_threshold=0.31,
        text_threshold=0.19,
        target_sizes="sizes",
    )
    assert result == ["outputs", "input_ids"]
    assert received == {
        "threshold": 0.31,
        "text_threshold": 0.19,
        "target_sizes": "sizes",
    }


def test_old_grounding_dino_api_receives_box_threshold() -> None:
    received = {}

    def post_process(outputs, input_ids=None, box_threshold=0.25, text_threshold=0.25, target_sizes=None):
        received.update(
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )
        return [outputs, input_ids]

    result = post_process_grounded_object_detection_compat(
        post_process,
        "outputs",
        "input_ids",
        box_threshold=0.31,
        text_threshold=0.19,
        target_sizes="sizes",
    )
    assert result == ["outputs", "input_ids"]
    assert received == {
        "box_threshold": 0.31,
        "text_threshold": 0.19,
        "target_sizes": "sizes",
    }


def test_unsupported_grounding_dino_api_fails_closed() -> None:
    def post_process(outputs, input_ids=None, text_threshold=0.25, target_sizes=None, **kwargs):
        return outputs

    with pytest.raises(
        GroundingDinoPostProcessCompatibilityError,
        match="neither 'threshold' nor 'box_threshold'",
    ):
        grounding_dino_threshold_branch(post_process)


def test_current_transformers_grounding_dino_api_selects_threshold() -> None:
    transformers = pytest.importorskip("transformers")
    processor_type = transformers.GroundingDinoProcessor
    branch = grounding_dino_threshold_branch(
        processor_type.post_process_grounded_object_detection
    )
    identity = grounding_dino_post_process_identity(
        processor_type.post_process_grounded_object_detection
    )
    assert branch == "threshold"
    assert identity["score_threshold_keyword"] == "threshold"
    assert identity["compatibility_wrapper_version"] == COMPATIBILITY_WRAPPER_VERSION
    assert "threshold" in inspect.signature(
        processor_type.post_process_grounded_object_detection
    ).parameters
