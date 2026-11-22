from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from eval.activearena.asa_astribot_embodiment import (
    ASAAstribotAlignmentConfig,
    ASAAstribotEmbodimentAdapter,
    ASANormBundle,
    R_EE_TO_ASA,
    matrix_to_pose7,
    matrix_to_rot6d_cols,
    pose7_to_matrix,
    rot6d_cols_to_matrix,
)


def test_rot6d_column_roundtrip():
    R = Rotation.from_euler("ZYX", [0.42, -0.31, 0.19]).as_matrix()
    r6 = matrix_to_rot6d_cols(R)
    assert r6.shape == (6,)
    assert np.allclose(r6[:3], R[:, 0])
    assert np.allclose(r6[3:], R[:, 1])
    R_rec = rot6d_cols_to_matrix(r6)
    assert np.allclose(R, R_rec, atol=1e-7)
    assert np.isclose(np.linalg.det(R_rec), 1.0, atol=1e-7)


def test_ee_frd_to_asa_flu_roundtrip():
    assert np.isclose(np.linalg.det(R_EE_TO_ASA), 1.0)
    assert np.allclose(R_EE_TO_ASA @ R_EE_TO_ASA, np.eye(3))

    # Simulate Astribot homestate left TCP in Sapien world (+Y_w forward, -X_w left, +Z_w up)
    R_ee = np.column_stack(
        [
            [0.0, 1.0, 0.0],   # +X_ee = +Y_w (forward along fingers)
            [1.0, 0.0, 0.0],   # +Y_ee = +X_w (rightward in FRD)
            [0.0, 0.0, -1.0],  # +Z_ee = -Z_w (downward in FRD)
        ]
    )
    T_tcp = np.eye(4)
    T_tcp[:3, :3] = R_ee
    T_tcp[:3, 3] = [-0.3081, 0.1185, 0.9386]
    tcp7 = matrix_to_pose7(T_tcp)

    T_asa = ASAAstribotEmbodimentAdapter.tcp_world_to_asa_world(tcp7)
    # In ASA FLU: col0 = +Y_w (forward), col1 = -X_w (left), col2 = +Z_w (up)!
    assert np.allclose(T_asa[:3, 0], [0.0, 1.0, 0.0])
    assert np.allclose(T_asa[:3, 1], [-1.0, 0.0, 0.0])
    assert np.allclose(T_asa[:3, 2], [0.0, 0.0, 1.0])

    # Roundtrip back to ActiveArena `get_left_ee_pose` (12 cm behind TCP along +X_ee)
    ee7 = ASAAstribotEmbodimentAdapter.asa_world_to_ee_pose7(T_asa)
    T_ee = pose7_to_matrix(ee7)
    assert np.allclose(T_ee[:3, :3], R_ee, atol=1e-7)
    assert np.allclose(T_ee[:3, 3], [-0.3081, 0.1185 - 0.12, 0.9386], atol=1e-7)


def test_encode_decode_proprio29_identity_consistency():
    adapter = ASAAstribotEmbodimentAdapter(
        norm_bundle=ASANormBundle.astribot_human_aligned(),
        cfg=ASAAstribotAlignmentConfig(
            base_frame_mode="canonical_pitch_047",
            max_ee_step_pos_m=1.0,
            max_head_step_rad=2.0,
            max_torso_step_rad=2.0,
        ),
    )
    # Robot base facing +Y_world (`quat_wxyz = [cos(pi/4), 0, 0, sin(pi/4)]`)
    T_robot0 = pose7_to_matrix(np.array([0.0, -0.45, -0.15, np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]))
    q_torso = 0.10
    q_head = np.array([0.15, 0.85])
    R_cam = T_robot0[:3, :3] @ Rotation.from_euler("ZYX", [q_torso + q_head[0], q_head[1], 0.0]).as_matrix()
    T_cam = np.eye(4)
    T_cam[:3, :3] = R_cam
    T_cam[:3, 3] = [0.0, -0.2376, 1.2656]

    left_tcp7 = np.array([-0.3081, 0.1185, 0.9386, 1.0, 0.0, 0.0, 0.0])
    right_tcp7 = np.array([0.3081, 0.1185, 0.9386, 1.0, 0.0, 0.0, 0.0])

    p29 = adapter.encode_proprio29(
        T_world_cam=T_cam,
        left_tcp_pose7=left_tcp7,
        right_tcp_pose7=right_tcp7,
        left_gripper=1.0,
        right_gripper=0.25,
        T_world_robot0=T_robot0,
        q_torso=q_torso,
        q_head1=q_head[0],
    )
    assert p29.shape == (29,)
    assert np.isfinite(p29).all()

    # Decode exact encoded proprio back into head/torso & EE targets -> must recover exact pose!
    q_h_rec, q_t_rec = adapter.decode_view_target_to_head_torso(p29[:9], q_head_now=q_head, q_torso_now=q_torso)
    assert np.isclose(q_t_rec + q_h_rec[0], q_torso + q_head[0], atol=1e-5)
    assert np.isclose(q_h_rec[1], q_head[1], atol=1e-5)

    mani_2x10 = np.stack(
        [
            np.concatenate([p29[9:18], p29[27:28]]),
            np.concatenate([p29[18:27], p29[28:29]]),
        ],
        axis=0,
    )
    l_ee7, lg, r_ee7, rg = adapter.decode_mani_target_to_ee_poses(mani_2x10)
    # TCP = p_ee + R_ee @ [0.12, 0, 0]
    T_l_ee = pose7_to_matrix(l_ee7)
    p_l_tcp_rec = T_l_ee[:3, 3] + T_l_ee[:3, :3] @ np.array([0.12, 0.0, 0.0])
    assert np.allclose(p_l_tcp_rec, left_tcp7[:3], atol=1e-5)
    assert np.isclose(lg, 1.0, atol=1e-5)
    assert np.isclose(rg, 0.25, atol=1e-5)
