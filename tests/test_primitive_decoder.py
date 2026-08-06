"""Decoder and metric behaviour that does not require a simulator."""

from __future__ import annotations

import numpy as np
import pytest

from interaction_uncertainty.v2.primitives import CoarsePrimitive
from interaction_uncertainty.v2.vla_bridge import ActionChunkSample
from interactive_perception.anchors import AnchorRole, AnchorSpec, ResolvedAnchor
from interactive_perception.metrics import EpisodeOutcome, aggregate, paired_difference
from interactive_perception.primitive_decoder import (
    PrimitiveDecoderConfig,
    decode_primitive_belief,
    decode_primitive_evidence,
)

EEF = (0.0, 0.0, 0.0)
ANCHORS = (
    ResolvedAnchor("butter", AnchorRole.TASK_TARGET, (0.0, 0.5, 0.0)),
    ResolvedAnchor("basket", AnchorRole.PLACEMENT, (0.0, -0.5, 0.0)),
    ResolvedAnchor("drawer_front", AnchorRole.OCCLUDER, (0.5, 0.0, 0.0)),
)


def _chunk(translation, *, rotation=(0.0, 0.0, 0.0), gripper=-1.0, horizon=10):
    chunk = np.zeros((horizon, 7), dtype=np.float64)
    chunk[:, 0:3] = np.asarray(translation, dtype=np.float64) / horizon
    chunk[:, 3:6] = np.asarray(rotation, dtype=np.float64) / horizon
    chunk[:, 6] = gripper
    return ActionChunkSample.from_chunk(chunk)


def test_motion_toward_occluder_decodes_as_remove_occluder() -> None:
    samples = [_chunk((0.2, 0.0, 0.0)) for _ in range(8)]
    evidence = decode_primitive_evidence(samples, eef_position=EEF, anchors=ANCHORS)
    assert evidence[CoarsePrimitive.REMOVE_OCCLUDER] > evidence[CoarsePrimitive.ACT]


def test_motion_toward_target_decodes_as_act() -> None:
    samples = [_chunk((0.0, 0.2, 0.0)) for _ in range(8)]
    evidence = decode_primitive_evidence(samples, eef_position=EEF, anchors=ANCHORS)
    assert evidence[CoarsePrimitive.ACT] > evidence[CoarsePrimitive.REMOVE_OCCLUDER]


def test_grasped_rotation_dominates_transport() -> None:
    samples = [_chunk((0.0, 0.02, 0.0), rotation=(0.0, 0.0, 1.2), gripper=1.0) for _ in range(8)]
    evidence = decode_primitive_evidence(samples, eef_position=EEF, anchors=ANCHORS)
    assert evidence[CoarsePrimitive.ROTATE] > 0.0
    assert evidence[CoarsePrimitive.ACT] == pytest.approx(0.0)


def test_grasped_retraction_decodes_as_move_closer() -> None:
    config = PrimitiveDecoderConfig(camera_forward=(1.0, 0.0, 0.0))
    samples = [_chunk((-0.2, 0.0, 0.0), gripper=1.0) for _ in range(8)]
    evidence = decode_primitive_evidence(
        samples, eef_position=EEF, anchors=ANCHORS, config=config
    )
    assert evidence[CoarsePrimitive.MOVE_CLOSER] > 0.0


def test_not_found_evidence_is_structurally_zero() -> None:
    """The measurement behind the claim that a baseline VLA cannot abstain."""

    for motion in [(0.2, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.0)]:
        evidence = decode_primitive_evidence(
            [_chunk(motion) for _ in range(8)], eef_position=EEF, anchors=ANCHORS
        )
        assert evidence[CoarsePrimitive.NOT_FOUND] == pytest.approx(0.0)


def test_primitive_belief_uses_the_full_coarse_label_set() -> None:
    belief = decode_primitive_belief(
        [_chunk((0.2, 0.0, 0.0)) for _ in range(8)], eef_position=EEF, anchors=ANCHORS
    )
    assert set(belief.labels) == {p.value for p in CoarsePrimitive}
    assert belief.probability(CoarsePrimitive.NOT_FOUND.value) > 0.0  # prior mass only


def test_decoder_requires_two_anchors() -> None:
    with pytest.raises(ValueError):
        decode_primitive_evidence(
            [_chunk((0.2, 0.0, 0.0))], eef_position=EEF, anchors=ANCHORS[:1]
        )


def test_anchor_spec_rejects_unknown_kinds() -> None:
    with pytest.raises(ValueError):
        AnchorSpec.from_dict({"label": "x", "role": "task_target", "kind": "camera", "ref": "y"})
    spec = AnchorSpec.from_dict({"label": "x", "role": "occluder", "ref": "y"})
    assert spec.kind == "object"
    assert spec.role is AnchorRole.OCCLUDER


def _outcome(**overrides):
    base = dict(
        task_id="T01_drawer_retrieval",
        prompt_variant="implicit",
        seed=0,
        steps=200,
        task_success=False,
        information_endpoint_reached=False,
        max_target_visible_pixels=0,
        steps_to_endpoint=None,
        first_committed_anchor="basket",
        first_committed_step=30,
        committed_before_endpoint=True,
        terminal_decision="ACT_NO_ABSTENTION",
        expected_terminal="TASK_SUCCESS",
        mean_vacuity=0.1,
        min_vacuity=0.05,
        mean_dissonance=0.2,
        mean_predictive_entropy=0.5,
        not_found_evidence=0.0,
    )
    base.update(overrides)
    return EpisodeOutcome(**base)


def test_premature_commit_requires_commitment_without_evidence() -> None:
    assert _outcome().premature_commit is True
    assert _outcome(information_endpoint_reached=True).premature_commit is False
    assert _outcome(committed_before_endpoint=False).premature_commit is False


def test_abstention_scenario_scores_terminal_decision_as_wrong() -> None:
    outcome = _outcome(expected_terminal="NOT_FOUND")
    assert outcome.correct_terminal_decision is False
    assert outcome.false_not_found is False


def test_aggregate_reports_endpoint_rate_beside_success_rate() -> None:
    outcomes = [_outcome(seed=seed) for seed in range(4)]
    report = aggregate(outcomes, condition="implicit")
    assert report.episodes == 4
    assert report.success_rate == pytest.approx(0.0)
    assert report.endpoint_rate == pytest.approx(0.0)
    assert report.premature_commit_rate == pytest.approx(1.0)


def test_aggregate_rejects_all_errored_conditions() -> None:
    with pytest.raises(ValueError):
        aggregate([_outcome(error="boom")], condition="implicit")


def test_paired_difference_matches_on_task_and_seed() -> None:
    control = [_outcome(seed=seed) for seed in range(3)]
    treatment = [
        _outcome(seed=seed, prompt_variant="capability", information_endpoint_reached=True)
        for seed in range(3)
    ]
    result = paired_difference(
        treatment, control, attribute="information_endpoint_reached"
    )
    assert result["pairs"] == 3
    assert result["mean_difference"] == pytest.approx(1.0)
