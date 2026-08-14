from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def test_route_calibration_is_separate_and_oracle_free() -> None:
    spec = yaml.safe_load(
        (ROOT / "benchmarks/calibration/route_intents_v2.yaml").read_text()
    )
    assert spec["calibration_only"] is True
    assert {item["label"] for item in spec["conditions"]} == {
        "ROTATE",
        "MOVE_CLOSER",
    }
    source = (ROOT / "scripts/collect_libero_route_calibration.py").read_text()
    assert "camera_segmentations" not in source
    assert "task_target" not in source
    assert '"oracle_inputs": []' in source
