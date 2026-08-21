import pytest

from interactive_perception.rgb_outcome_critic import resolve_outcome


def test_strict_fusion_preserves_singleton_camera_conflict() -> None:
    outcome, source = resolve_outcome(
        ("NOT_REVEALED",),
        ("REVEALED",),
        ("COMPLETED",),
        fusion="strict",
    )
    assert outcome == ("REVEALED", "EMPTY")
    assert source == "singleton_camera_conflict"


def test_strict_fusion_treats_multilabel_wrist_as_abstention() -> None:
    outcome, source = resolve_outcome(
        ("NOT_REVEALED",),
        ("REVEALED", "NOT_REVEALED"),
        ("COMPLETED",),
        fusion="strict",
    )
    assert outcome == ("EMPTY",)
    assert source == "agentview_negative_no_singleton_wrist_counterevidence"


def test_complementary_fusion_accepts_either_singleton_positive() -> None:
    outcome, source = resolve_outcome(
        ("NOT_REVEALED",),
        ("REVEALED",),
        ("COMPLETED",),
        fusion="complementary",
    )
    assert outcome == ("REVEALED",)
    assert source == "wrist_positive_evidence"


def test_local_empty_requires_target_negative_and_completed_coverage() -> None:
    empty, _ = resolve_outcome(
        ("NOT_REVEALED",),
        ("NOT_REVEALED",),
        ("COMPLETED",),
    )
    failed, _ = resolve_outcome(
        ("NOT_REVEALED",),
        ("NOT_REVEALED",),
        ("FAILED",),
    )
    assert empty == ("EMPTY",)
    assert failed == ("FAILED",)


def test_unknown_fusion_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown camera fusion"):
        resolve_outcome(("REVEALED",), (), (), fusion="legacy")
