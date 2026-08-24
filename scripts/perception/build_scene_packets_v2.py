"""S03-v2 Grounding DINO adapter over the byte-frozen public frontend v1."""

from __future__ import annotations

from typing import Any

from build_scene_packets import PublicObjectFrontend as PublicObjectFrontendV1
from interaction_uncertainty.grounding_dino_compat import (
    grounding_dino_post_process_identity,
    post_process_grounded_object_detection_compat,
)


class _GroundingDinoProcessorCompatibilityProxy:
    """Preserve the v1 caller while adapting only the processor API boundary."""

    def __init__(self, processor: Any) -> None:
        self._processor = processor
        self.api_identity = grounding_dino_post_process_identity(
            processor.post_process_grounded_object_detection
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._processor(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._processor, name)

    def post_process_grounded_object_detection(
        self,
        outputs: Any,
        input_ids: Any = None,
        *,
        box_threshold: float,
        text_threshold: float,
        target_sizes: Any,
    ) -> Any:
        return post_process_grounded_object_detection_compat(
            self._processor.post_process_grounded_object_detection,
            outputs,
            input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )


class PublicObjectFrontend(PublicObjectFrontendV1):
    """V1 frontend with a provenance-bearing, fail-closed API adapter."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        proxy = _GroundingDinoProcessorCompatibilityProxy(self.grounding_processor)
        self.grounding_processor = proxy
        self.grounding_dino_post_process = proxy.api_identity
