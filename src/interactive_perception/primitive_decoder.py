"""Decode continuous action chunks into the coarse action space ``A``.

A monolithic VLA emits joint deltas, not primitives.  To compare it against a
method that selects from ``A = {ACT, NOT_FOUND, ROTATE, MOVE_CLOSER,
REMOVE_OCCLUDER}``, its motion has to be read as an intent.  This module does
that with explicit geometric rules over the sampled chunks and the roles of the
scene anchors.

The decoder is a **diagnostic instrument, not a claim about the policy**.  It
says where the motion is going and what that motion would accomplish; it does
not assert that the policy internally represents primitives.  Its thresholds
are exposed rather than tuned in secret, because a reader has to be able to
check that a reported behavioural distribution is not an artefact of them.

``NOT_FOUND`` is deliberately undecodable.  There is no continuous action that
means "abstain": a policy without an abstention channel cannot express one, so
its evidence for that member is structurally zero.  Reporting it as zero rather
than omitting the member is the point -- it is the measurement behind the claim
that baseline VLAs cannot give up.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from interaction_uncertainty.beliefs import DirichletBelief
from interaction_uncertainty.v2.primitives import COARSE_ACTION_SPACE, CoarsePrimitive
from interaction_uncertainty.v2.vla_bridge import (
    ActionChunkSample,
    AttributionConfig,
    HypothesisAnchor,
    attribute_samples,
)

from .anchors import AnchorRole, ResolvedAnchor

__all__ = [
    "PrimitiveDecoderConfig",
    "decode_primitive_belief",
    "decode_primitive_evidence",
]


_ROLE_TO_PRIMITIVE: dict[AnchorRole, CoarsePrimitive] = {
    AnchorRole.TASK_TARGET: CoarsePrimitive.ACT,
    AnchorRole.PLACEMENT: CoarsePrimitive.ACT,
    AnchorRole.OCCLUDER: CoarsePrimitive.REMOVE_OCCLUDER,
    AnchorRole.LABEL_SURFACE: CoarsePrimitive.ROTATE,
    AnchorRole.DISTRACTOR: CoarsePrimitive.ACT,
}


@dataclasses.dataclass(frozen=True)
class PrimitiveDecoderConfig:
    """Thresholds separating one motion intent from another.

    ``rotation_dominance`` is the ratio of summed rotation command to summed
    translation command above which a grasped chunk reads as reorientation
    rather than transport.  ``retract_speed`` is the backward end-effector
    motion, along the axis from the fixed camera into the scene, above which a
    grasped chunk reads as bringing the object back toward the viewpoint.
    """

    attribution: AttributionConfig = dataclasses.field(default_factory=AttributionConfig)
    rotation_dominance: float = 1.5
    retract_speed: float = 0.01
    grasp_threshold: float = 0.0
    camera_forward: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.rotation_dominance <= 0.0:
            raise ValueError("rotation_dominance must be > 0")
        if self.retract_speed <= 0.0:
            raise ValueError("retract_speed must be > 0")
        forward = np.asarray(self.camera_forward, dtype=np.float64)
        if forward.shape != (3,) or not np.isfinite(forward).all():
            raise ValueError("camera_forward must be a finite length-3 vector")
        norm = float(np.linalg.norm(forward))
        if norm <= 1e-9:
            raise ValueError("camera_forward must be non-zero")
        object.__setattr__(
            self, "camera_forward", tuple((forward / norm).tolist())
        )


def decode_primitive_evidence(
    samples: Sequence[ActionChunkSample],
    *,
    eef_position: Sequence[float],
    anchors: Sequence[ResolvedAnchor],
    config: PrimitiveDecoderConfig | None = None,
) -> dict[CoarsePrimitive, float]:
    """Accumulate evidence over ``A`` from sampled chunks.

    Each sample contributes evidence equal to its decisiveness, routed to a
    member of ``A`` by three checks applied in order: a grasped chunk turning
    far more than it travels is ``ROTATE``; a grasped chunk retracting toward
    the viewpoint is ``MOVE_CLOSER``; otherwise the destination anchor's role
    decides.  Samples that commit to nothing contribute nothing, so the total
    evidence -- and hence the vacuity of the resulting belief -- reflects how
    decisive the policy was, not how many times it was queried.
    """

    config = config or PrimitiveDecoderConfig()
    if len(anchors) < 2:
        raise ValueError("primitive decoding needs at least two role-tagged anchors")

    hypothesis_anchors = [
        HypothesisAnchor(anchor.label, anchor.position) for anchor in anchors
    ]
    attributions = attribute_samples(
        samples,
        eef_position=eef_position,
        anchors=hypothesis_anchors,
        config=config.attribution,
    )

    forward = np.asarray(config.camera_forward, dtype=np.float64)
    evidence = {primitive: 0.0 for primitive in COARSE_ACTION_SPACE}

    for sample, attribution in zip(samples, attributions, strict=True):
        if attribution.decisiveness <= 0.0:
            continue
        weight = attribution.decisiveness
        grasped = sample.gripper_command > config.grasp_threshold
        translation = np.asarray(sample.translation_delta, dtype=np.float64)
        translation_norm = float(np.linalg.norm(translation))

        if grasped and translation_norm > 1e-9:
            if sample.rotation_norm > config.rotation_dominance * translation_norm:
                evidence[CoarsePrimitive.ROTATE] += weight
                continue
            if float(translation @ forward) < -config.retract_speed:
                evidence[CoarsePrimitive.MOVE_CLOSER] += weight
                continue

        # Otherwise the motion is transport, and the destination's role says
        # what that transport is for.
        weights = np.asarray(attribution.weights, dtype=np.float64)
        for anchor, share in zip(anchors, weights.tolist(), strict=True):
            if share <= 0.0:
                continue
            evidence[_ROLE_TO_PRIMITIVE[anchor.role]] += weight * share

    # NOT_FOUND stays at zero by construction: see the module docstring.
    return evidence


def decode_primitive_belief(
    samples: Sequence[ActionChunkSample],
    *,
    eef_position: Sequence[float],
    anchors: Sequence[ResolvedAnchor],
    config: PrimitiveDecoderConfig | None = None,
) -> DirichletBelief:
    """Same decoding, returned as a Dirichlet belief over ``A``.

    Because the label set is fixed to ``A``, vacuity and dissonance computed on
    this belief are comparable across scenarios and across methods.
    """

    evidence = decode_primitive_evidence(
        samples, eef_position=eef_position, anchors=anchors, config=config
    )
    labels = tuple(primitive.value for primitive in COARSE_ACTION_SPACE)
    return DirichletBelief.from_evidence(
        labels, tuple(evidence[primitive] for primitive in COARSE_ACTION_SPACE)
    )
