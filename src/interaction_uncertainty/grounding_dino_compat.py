"""Fail-closed Grounding DINO post-processing API compatibility.

Transformers renamed the detection score keyword from ``box_threshold`` to
``threshold``.  The public perception backend must preserve the frozen
numeric threshold across that API change, rather than dropping it through a
catch-all ``**kwargs`` path.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any, Literal


COMPATIBILITY_WRAPPER_VERSION = "grounding_dino_post_process_signature_v1"
GroundingDinoThresholdBranch = Literal["threshold", "box_threshold"]


class GroundingDinoPostProcessCompatibilityError(RuntimeError):
    """The installed processor cannot receive the frozen score threshold."""


def grounding_dino_threshold_branch(
    post_process: Callable[..., Any],
) -> GroundingDinoThresholdBranch:
    """Return the explicit score-threshold keyword supported by ``post_process``.

    ``threshold`` is preferred when both names are explicitly present because
    it is the current Transformers API.  A generic ``**kwargs`` parameter is
    intentionally insufficient: accepting it would not prove that the score
    threshold is consumed.
    """

    try:
        parameters: Mapping[str, inspect.Parameter] = inspect.signature(
            post_process
        ).parameters
    except (TypeError, ValueError) as exc:
        raise GroundingDinoPostProcessCompatibilityError(
            "cannot inspect Grounding DINO post-processing signature; "
            "refusing to run without an explicit score-threshold parameter"
        ) from exc
    if "threshold" in parameters:
        return "threshold"
    if "box_threshold" in parameters:
        return "box_threshold"
    raise GroundingDinoPostProcessCompatibilityError(
        "Grounding DINO post-processing supports neither 'threshold' nor "
        "'box_threshold'; refusing to silently ignore the frozen threshold"
    )


def grounding_dino_post_process_identity(
    post_process: Callable[..., Any],
) -> dict[str, str]:
    """Describe the inspected API branch for provenance artifacts."""

    return {
        "compatibility_wrapper_version": COMPATIBILITY_WRAPPER_VERSION,
        "score_threshold_keyword": grounding_dino_threshold_branch(post_process),
        "callable_signature": str(inspect.signature(post_process)),
    }


def post_process_grounded_object_detection_compat(
    post_process: Callable[..., Any],
    outputs: Any,
    input_ids: Any,
    *,
    box_threshold: float,
    text_threshold: float,
    target_sizes: Any,
) -> Any:
    """Call either supported API while forwarding every frozen threshold."""

    branch = grounding_dino_threshold_branch(post_process)
    score_threshold = float(box_threshold)
    keyword = {branch: score_threshold}
    return post_process(
        outputs,
        input_ids,
        **keyword,
        text_threshold=float(text_threshold),
        target_sizes=target_sizes,
    )
