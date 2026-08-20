from interactive_perception.rgb_outcome_critic import (
    resolve_v11_cascade,
    resolve_v12_cascade,
    resolve_v12b_cascade,
    resolve_v13_complementary_cascade,
)


def test_v11_cascade_uses_wrist_only_as_positive_rescue() -> None:
    outcome, source = resolve_v11_cascade(
        ("NOT_REVEALED",), ("REVEALED",), ("FAILED",)
    )
    assert outcome == ("REVEALED",)
    assert source == "wrist_positive_rescue"


def test_v11_cascade_separates_local_empty_from_failed() -> None:
    empty, _ = resolve_v11_cascade(
        ("NOT_REVEALED",), ("NOT_REVEALED",), ("COMPLETED",)
    )
    failed, _ = resolve_v11_cascade(
        ("NOT_REVEALED",), ("NOT_REVEALED",), ("FAILED",)
    )
    assert empty == ("EMPTY",)
    assert failed == ("FAILED",)


def test_v11_cascade_never_resolves_ambiguous_target_for_convenience() -> None:
    outcome, source = resolve_v11_cascade(
        ("REVEALED", "NOT_REVEALED"),
        ("REVEALED", "NOT_REVEALED"),
        ("COMPLETED",),
    )
    assert outcome == ("FAILED", "REVEALED", "EMPTY")
    assert source == "ambiguous"


def test_v12_cascade_preserves_cross_camera_conflict_as_safe_stop() -> None:
    outcome, source = resolve_v12_cascade(
        ("NOT_REVEALED",), ("REVEALED",), ("COMPLETED",)
    )
    assert outcome == ("REVEALED", "EMPTY")
    assert source == "camera_conflict"


def test_v12_cascade_maps_agreed_negative_through_coverage() -> None:
    empty, empty_source = resolve_v12_cascade(
        ("NOT_REVEALED",), ("NOT_REVEALED",), ("COMPLETED",)
    )
    failed, failed_source = resolve_v12_cascade(
        ("NOT_REVEALED",), ("NOT_REVEALED",), ("FAILED",)
    )
    assert empty == ("EMPTY",)
    assert failed == ("FAILED",)
    assert empty_source == failed_source == "camera_agreement_negative"


def test_v12_cascade_retains_coverage_ambiguity() -> None:
    outcome, _ = resolve_v12_cascade(
        ("NOT_REVEALED",),
        ("NOT_REVEALED",),
        ("FAILED", "COMPLETED"),
    )
    assert outcome == ("FAILED", "EMPTY")


def test_v12b_ignores_non_singleton_wrist_abstention_against_agent_negative() -> None:
    outcome, source = resolve_v12b_cascade(
        ("NOT_REVEALED",),
        ("REVEALED", "NOT_REVEALED"),
        ("COMPLETED",),
    )
    assert outcome == ("EMPTY",)
    assert source == "agentview_negative_no_singleton_wrist_counterevidence"


def test_v12b_preserves_singleton_wrist_conflict() -> None:
    outcome, source = resolve_v12b_cascade(
        ("NOT_REVEALED",), ("REVEALED",), ("COMPLETED",)
    )
    assert outcome == ("REVEALED", "EMPTY")
    assert source == "singleton_camera_conflict"


def test_v13_treats_wrist_reveal_as_complementary_positive_evidence() -> None:
    outcome, source = resolve_v13_complementary_cascade(
        ("NOT_REVEALED",), ("REVEALED",), ("COMPLETED",)
    )
    assert outcome == ("REVEALED",)
    assert source == "wrist_positive_evidence"


def test_v13_requires_both_camera_negatives_before_empty() -> None:
    ambiguous, ambiguous_source = resolve_v13_complementary_cascade(
        ("NOT_REVEALED",),
        ("REVEALED", "NOT_REVEALED"),
        ("COMPLETED",),
    )
    empty, empty_source = resolve_v13_complementary_cascade(
        ("NOT_REVEALED",), ("NOT_REVEALED",), ("COMPLETED",)
    )
    assert ambiguous == ("REVEALED", "EMPTY")
    assert ambiguous_source == "complementary_target_ambiguous"
    assert empty == ("EMPTY",)
    assert empty_source == "complementary_camera_agreement_negative"
