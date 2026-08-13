from pathlib import Path


def test_pure_pi05_runner_has_no_oracle_wrapper() -> None:
    source = (Path(__file__).parents[1] / "scripts/run_pure_pi05_scenario_sr.py").read_text()
    assert "OffScreenRenderEnv" in source
    assert "SegmentationRenderEnv" not in source
    assert "build_observation" in source
    assert "LIBERO_DUMMY_ACTION" in source
    assert "interactive_perception.rollout" not in source
    assert "get_joint_qpos" in source
    assert "agentview_image" in source
