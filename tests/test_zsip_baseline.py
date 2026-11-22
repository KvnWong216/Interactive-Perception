"""Unit tests for ZS-IP (`arXiv:2602.18374`) reproduction on ActiveArena Astribot S1."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from eval.activearena.asa_astribot_embodiment import pose7_to_matrix
from eval.activearena.run_eval import check_readiness
from eval.activearena.zsip_eo_and_memory import (
    OBJECT_RAY_Z_M,
    TABLE_Z_M,
    CameraPinhole,
    ZSIPMemory,
    build_enhanced_observation,
    extract_proprio_reveal_crop,
)
from eval.activearena.zsip_server import parse_zsip_vlm_output
from ipvla.config import load_paths_config


def _make_test_camera_and_robot() -> tuple[np.ndarray, np.ndarray]:
    T_world_robot0 = pose7_to_matrix(np.array([0.0, -0.45, -0.15, 0.70710678, 0.0, 0.0, 0.70710678]))
    # Head camera at [0.0, -0.355, 1.266] pitched down by 0.96 rad
    R_local = Rotation.from_euler("ZYX", [0.0, 0.96, 0.0]).as_matrix()
    T_world_cam = np.eye(4, dtype=np.float64)
    T_world_cam[:3, :3] = T_world_robot0[:3, :3] @ R_local
    T_world_cam[:3, 3] = np.array([0.0, -0.355, 1.266], dtype=np.float64)
    return T_world_cam, T_world_robot0


def test_pinhole_projection_raycast_roundtrip() -> None:
    T_world_cam, _ = _make_test_camera_and_robot()
    for w, h in ((512, 384), (224, 224)):
        cam = CameraPinhole(width=w, height=h)
        target = np.array([-0.14, -0.05, TABLE_Z_M], dtype=np.float64)
        uv = cam.project_world_to_pixel(target, T_world_cam)
        assert uv is not None
        recovered = cam.raycast_pixel_to_z_plane(uv[0], uv[1], T_world_cam, z_plane=TABLE_Z_M)
        assert recovered is not None
        assert np.allclose(recovered, target, atol=1e-5)


def test_enhanced_observation_and_proprio_mask() -> None:
    T_world_cam, T_world_robot0 = _make_test_camera_and_robot()
    cam = CameraPinhole(width=512, height=384)
    rgb = np.full((384, 512, 3), 245, dtype=np.uint8)

    # Paint a red near pad, a blue near pad, a gray mid-zone block, and a yellow outer clutter item
    red_uv = cam.project_world_to_pixel(np.array([-0.15, -0.10, TABLE_Z_M]), T_world_cam)
    blue_uv = cam.project_world_to_pixel(np.array([+0.15, -0.10, TABLE_Z_M]), T_world_cam)
    gray_uv = cam.project_world_to_pixel(np.array([-0.08, +0.02, OBJECT_RAY_Z_M]), T_world_cam)
    clutter_uv = cam.project_world_to_pixel(np.array([+0.18, +0.16, OBJECT_RAY_Z_M]), T_world_cam)
    held_uv = cam.project_world_to_pixel(np.array([0.0, -0.04, 0.88]), T_world_cam)
    assert all(x is not None for x in (red_uv, blue_uv, gray_uv, clutter_uv, held_uv))

    def _paint(uv: tuple[float, float], col: tuple[int, int, int], half: int = 16) -> None:
        u, v = int(round(uv[0])), int(round(uv[1]))
        rgb[max(0, v - half) : min(384, v + half), max(0, u - half) : min(512, u + half)] = col

    _paint(red_uv, (225, 35, 35), half=18)      # type: ignore[arg-type]
    _paint(blue_uv, (30, 65, 230), half=18)     # type: ignore[arg-type]
    _paint(gray_uv, (145, 145, 145), half=15)   # type: ignore[arg-type]
    _paint(clutter_uv, (235, 210, 30), half=15) # type: ignore[arg-type]
    _paint(held_uv, (40, 40, 40), half=14)      # type: ignore[arg-type]

    eo = build_enhanced_observation(
        rgb=rgb,
        T_world_cam=T_world_cam,
        T_world_robot0=T_world_robot0,
        exclude_uvs=[held_uv],  # type: ignore[list-item]
    )
    assert set(eo.grid_anchors.keys()) == {"G1", "G2", "G3", "G4", "G5", "G6"}
    colors = {kp.color_tag for kp in eo.keypoints.values()}
    assert {"red", "blue", "gray", "yellow"}.issubset(colors)
    assert "dark/black" not in colors  # Masked out by proprioceptive `exclude_uvs`!
    assert len(eo.pushlines) >= 1
    assert eo.annotated_rgb.shape == (384, 512, 3)


def test_memory_and_reveal_crop() -> None:
    T_world_cam, _ = _make_test_camera_and_robot()
    rgb = np.zeros((384, 512, 3), dtype=np.uint8)
    rgb[:, :, 1] = 180
    crop = extract_proprio_reveal_crop(rgb, np.array([0.0, -0.05, 0.90]), T_world_cam, crop_size=256)
    assert crop.shape == (256, 256, 3)

    mem = ZSIPMemory(max_reveal_frames=2)
    mem.holding = True
    mem.holding_arm = "left"
    mem.origin_xyz = [-0.08, 0.02, 0.766]
    mem.record_action(1, "GRASP", "K2", "Grasped gray block")
    mem.pin_reveal_frame(crop)
    mem.pin_reveal_frame(crop + 1)
    mem.pin_reveal_frame(crop + 2)
    assert len(mem.reveal_frames) == 2
    ctx = mem.format_context_lookup()
    assert "HOLDING object in left gripper" in ctx
    assert "GRASP(K2)" in ctx


def test_parse_zsip_vlm_output() -> None:
    out1 = parse_zsip_vlm_output(
        '```json\n{"thought": "Pick gray block K3", "primitive": "grasp", "target": "K3"}\n```'
    )
    assert out1["primitive"] == "GRASP"
    assert out1["target"] == "K3"

    out2 = parse_zsip_vlm_output("Next I should INSPECT_HELD(self) to see the back label.")
    assert out2["primitive"] == "INSPECT_HELD"
    assert out2["target"] == "self"


def test_zsip_readiness_preflight() -> None:
    paths = load_paths_config()
    rep = check_readiness(
        paths,
        mode="zsip_baseline",
        tasks=["check_block_color", "check_cola_date", "check_cola_color"],
        setting_name="info_gathering_demo",
    )
    assert rep["ready"] is True
    assert rep["checks"]["zsip_training_steps"] == 0
