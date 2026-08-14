"""Camera setup helpers that never move the robot or alter task state."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _mat_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        )
    else:
        index = int(np.argmax(np.diag(m)))
        if index == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            quat = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif index == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            quat = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            quat = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return quat / np.linalg.norm(quat)


def initialize_attached_camera_look_at(
    env: Any, *, camera: str, target: Sequence[float]
) -> dict[str, list[float]]:
    """Aim an attached camera at a world point without changing robot qpos."""

    sim = env.env.sim
    camera_id = sim.model.camera_name2id(camera)
    eye = np.asarray(sim.data.cam_xpos[camera_id], dtype=float).copy()
    target_value = np.asarray(target, dtype=float)
    z_axis = eye - target_value
    z_axis /= np.linalg.norm(z_axis)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(z_axis @ up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    desired_world = np.stack([x_axis, y_axis, z_axis], axis=1)
    body_id = int(sim.model.cam_bodyid[camera_id])
    body_world = np.asarray(sim.data.xmat[body_id], dtype=float).reshape(3, 3)
    local_rotation = body_world.T @ desired_world
    sim.model.cam_quat[camera_id] = _mat_to_quat_wxyz(local_rotation)
    sim.forward()
    rotation = np.asarray(sim.data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
    forward = -rotation[:, 2]
    return {
        "position": np.asarray(sim.data.cam_xpos[camera_id], dtype=float).tolist(),
        "forward": forward.tolist(),
        "target": target_value.tolist(),
    }
