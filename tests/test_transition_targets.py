"""Check geometric oracle convention and nonuniform feature resampling."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_transition_targets import camera_map, layout, sample


def test_native_nonuniform_grid_identity_and_translation():
    axis = torch.tensor([0.1, 0.45, 0.9])
    v, u = torch.meshgrid(axis, axis, indexing="ij")
    uv = torch.stack((u.flatten(), v.flatten()), -1)
    p = torch.cat(
        [
            torch.cat((uv, torch.full((9, 1), float(view)), torch.zeros(9, 1)), 1)
            for view in [0, 1]
        ]
    )
    f = p[:, :2][None].clone()
    grid = layout(p)
    torch.testing.assert_close(sample(f, grid, p[None, :, :2]), f)
    shifted = p[None, :, :2].clone()
    shifted[:, :9, 0] = 0.45
    prediction = sample(f, grid, shifted)
    torch.testing.assert_close(prediction[0, :9, 0], torch.full((9,), 0.45))
    torch.testing.assert_close(prediction[0, 9:], f[0, 9:])


def test_camera_oracle_signed_intrinsics_and_occlusion():
    depth = np.full((11, 11), 2.0)
    k = np.array([[-10.0, 0, 5], [0, 10, 5], [0, 0, 1]])
    current = {"wrist_depth": depth, "wrist_K": k, "wrist_T_world": np.eye(4)}
    uv = np.array([[0.5, 0.5], [0.6, 0.6]])
    identity, visible = camera_map(current, current, uv, "wrist")
    np.testing.assert_allclose(identity, uv)
    assert visible.all()
    future = {**current, "wrist_T_world": np.eye(4)}
    future["wrist_T_world"][0, 3] = 0.2
    moved, visible = camera_map(current, future, uv, "wrist")
    np.testing.assert_allclose(moved[:, 0], uv[:, 0] - 0.1)
    assert visible.all()
    hidden = {**future, "wrist_depth": np.ones((11, 11))}
    _, visible = camera_map(current, hidden, uv, "wrist")
    assert not visible.any()


def test_predictor_loader_excludes_confirmation_and_future_geometry(tmp_path):
    import json

    from diagnose_transition_targets import load

    train = {"case_id": 0, "split": "train"}
    confirm = {"case_id": 1, "split": "confirmation"}
    (tmp_path / "plan.json").write_text(json.dumps({"cases": [train, confirm]}))
    features = tmp_path / "features"
    features.mkdir()
    folder = tmp_path / "case_000"
    folder.mkdir()
    np.savez(folder / "history_2.npz", states=np.zeros(8, dtype=np.float32))
    torch.manual_seed(17)
    target = torch.randn(1, 8, 12)
    records = [
        {"endpoints": [{"horizon": 10, "file": "must_not_open_future.npz"}]}
        for _ in range(6)
    ]
    records += [{"endpoints": [{"repeat_max_abs": {"rgb": 0}}]} for _ in range(2)]
    torch.save(
        {
            "case": train,
            "source_report": {
                "complete": True,
                "initial_arrays_identical": True,
                "records": records,
            },
            "positions": torch.zeros(1, 8, 4),
            "current": target,
            "valid": torch.ones(1, 8, dtype=torch.bool),
            "shared": torch.randn(1, 3, 12),
            "branches": [
                {
                    "endpoints": [
                        {
                            "horizon": 10,
                            "target": target,
                            "actions": torch.zeros(1, 10, 7),
                        }
                    ]
                }
                for _ in range(6)
            ],
        },
        features / "case_000.pt",
    )
    # Neither the confirmation cache nor future geometry files exist: ordinary
    # prediction must succeed without opening either privileged information path.
    cases = load("cpu", tmp_path)
    assert len(cases) == 1 and cases[0]["case"] == train
    assert {"oracle_uv", "delta", "visible"}.isdisjoint(cases[0])
    assert cases[0]["action_sequences"].shape == (6, 10, 7)
