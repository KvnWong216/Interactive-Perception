"""Policy inputs contain only real, time-aligned public observations/actions."""

from dataclasses import dataclass

import numpy as np


def finite_array(value, shape, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    depth_m: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    camera_to_world: np.ndarray | None = None

    def __post_init__(self):
        rgb = np.asarray(self.rgb)
        if (
            rgb.ndim != 3
            or rgb.shape[2] != 3
            or rgb.dtype != np.uint8
            or min(rgb.shape) < 1
        ):
            raise ValueError("RGB must be nonempty H x W x 3 uint8")
        object.__setattr__(self, "rgb", np.ascontiguousarray(rgb))
        values = (self.depth_m, self.intrinsics, self.camera_to_world)
        if any(v is not None for v in values) and not all(
            v is not None for v in values
        ):
            raise ValueError(
                "metric depth, K and camera_to_world must be supplied together"
            )
        if self.depth_m is not None:
            depth = np.asarray(self.depth_m, dtype=np.float32)
            if depth.shape != rgb.shape[:2]:
                raise ValueError("depth and RGB must have the same pixel coordinates")
            k = finite_array(self.intrinsics, (3, 3), "intrinsics")
            t = finite_array(self.camera_to_world, (4, 4), "camera_to_world")
            # Signed focal lengths support an explicitly rotated raster.
            if abs(np.linalg.det(k)) < 1e-8 or not np.allclose(k[2], [0, 0, 1]):
                raise ValueError("intrinsics must be invertible pinhole calibration")
            r = t[:3, :3]
            if not (
                np.allclose(t[3], [0, 0, 0, 1])
                and np.allclose(r.T @ r, np.eye(3), atol=1e-4)
                and np.isclose(np.linalg.det(r), 1, atol=1e-4)
            ):
                raise ValueError(
                    "camera_to_world must be a rigid right-handed transform"
                )
            object.__setattr__(self, "depth_m", depth)
            object.__setattr__(self, "intrinsics", k)
            object.__setattr__(self, "camera_to_world", t)


@dataclass(frozen=True)
class Observation:
    step: int
    cameras: tuple[CameraFrame, CameraFrame]
    state: np.ndarray

    def __post_init__(self):
        if type(self.step) is not int or self.step < 0:
            raise ValueError("observation step must be a nonnegative integer")
        if len(self.cameras) != 2 or not all(
            isinstance(c, CameraFrame) for c in self.cameras
        ):
            raise ValueError("ordered agentview and wrist CameraFrames are required")
        object.__setattr__(self, "state", finite_array(self.state, (8,), "robot state"))


@dataclass(frozen=True)
class AppliedAction:
    step: int  # action maps observation at step to observation at step + 1
    value: np.ndarray

    def __post_init__(self):
        if type(self.step) is not int or self.step < 0:
            raise ValueError("applied action step must be nonnegative")
        object.__setattr__(
            self, "value", finite_array(self.value, (7,), "applied action")
        )


@dataclass(frozen=True)
class PolicyContext:
    task: str
    observations: tuple[Observation, ...]
    applied_actions: tuple[AppliedAction, ...]
    remaining_steps: int

    def __post_init__(self):
        if (
            not isinstance(self.task, str)
            or not self.task.strip()
            or not self.observations
        ):
            raise ValueError("nonempty task and observed history are required")
        steps = [o.step for o in self.observations]
        if steps != sorted(set(steps)):
            raise ValueError("observations must be strictly chronological")
        if type(self.remaining_steps) is not int or self.remaining_steps < 0:
            raise ValueError("remaining_steps must be nonnegative")
        action_steps = [a.step for a in self.applied_actions]
        if action_steps != list(range(steps[0], steps[-1])):
            raise ValueError(
                "history must contain exactly the actually applied actions between observations"
            )

    @property
    def current(self):
        return self.observations[-1]

    def advance(self, observation, actions, *, max_frames):
        if max_frames < 1:
            raise ValueError("history capacity must be positive")
        rows = tuple(actions)
        if [a.step for a in rows] != list(
            range(self.current.step, observation.step)
        ) or not rows:
            raise ValueError(
                "feedback must correspond to a contiguous nonempty applied prefix"
            )
        if len(rows) > self.remaining_steps:
            raise ValueError("executed prefix exceeds the episode budget")
        obs = (*self.observations, observation)[-max_frames:]
        actual = tuple(
            a for a in (*self.applied_actions, *rows) if a.step >= obs[0].step
        )
        return PolicyContext(self.task, obs, actual, self.remaining_steps - len(rows))
