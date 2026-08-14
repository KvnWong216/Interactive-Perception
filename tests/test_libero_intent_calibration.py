from pathlib import Path


def test_collector_is_oracle_free_and_split_is_frozen() -> None:
    source = (Path(__file__).parents[1] / "scripts/collect_libero_intent_calibration.py").read_text()
    assert "OffScreenRenderEnv" in source
    assert "SegmentationRenderEnv" not in source
    assert "resolve_anchors" not in source
    assert "get_joint_qpos" not in source
    assert 'range(20)' in source
    assert 'range(20, 40)' in source
    assert 'range(40, 50)' in source
