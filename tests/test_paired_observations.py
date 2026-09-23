"""Renderer repair must preserve identical inputs without concealing scene drift."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "paired_observations", Path(__file__).resolve().parents[1] / "scripts/paired_observations.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path, *, delta=1, changed=3, state_drift=False, depth_drift=False):
    paths = []
    for index, name in enumerate(("native", "best")):
        directory = tmp_path / name
        directory.mkdir()
        rgb = np.full((256, 256, 3), 128, dtype=np.uint8)
        rgb.ravel()[:changed] += index * delta
        depth = np.ones((256, 256), dtype=np.float32)
        if depth_drift and index:
            depth[0, 0] += 0.001
        path = directory / "observation_0000.npz"
        np.savez_compressed(path, wrist_rgb=rgb, wrist_depth=depth)
        (directory / "scene_audit.json").write_text(json.dumps(
            {"case": {"manual_seed": 17}, "physical_state": [0, index if state_drift else 0]}
        ))
        paths.append(path)
    return paths


def test_sparse_roundoff_is_archived_and_policy_input_is_exact(tmp_path):
    reference, current = fixture(tmp_path)
    differences = module.align_initial_observation(reference, current)
    assert differences["wrist_rgb"]["changed_components"] == 3
    with np.load(reference) as a, np.load(current) as b:
        assert all(np.array_equal(a[k], b[k]) for k in a.files)
    assert (current.parent / "observation_0000_raw_render.npz").exists()


@pytest.mark.parametrize("kwargs", [
    {"delta": 2}, {"changed": 20}, {"state_drift": True}, {"depth_drift": True},
])
def test_real_mismatch_is_rejected_without_rewriting(tmp_path, kwargs):
    reference, current = fixture(tmp_path, **kwargs)
    before = current.read_bytes()
    with pytest.raises(ValueError):
        module.align_initial_observation(reference, current)
    assert current.read_bytes() == before
