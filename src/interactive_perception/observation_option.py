"""Policy-visible controller for returning to a calibrated observation pose.

``OPEN_CONTAINER`` is not an information action unless the robot also obtains
a useful post-action image.  This module implements the second half of the
composite ``OPEN_AND_OBSERVE`` option.  It deliberately consumes only robot
proprioception already present in the frozen pi0.5 observation contract.  It
does not read simulator joints, object poses, segmentation, or target labels.

The pose below is context scoped to the stock-aligned T01 workspace.  It was
measured at reset and then checked, evaluator-side, to expose the opened middle
drawer in 60/60 development seeds.  That static check is only a coverage
diagnostic; the controller still needs a physical rollout gate before it can
be registered as a reliable action effect.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "ObservationPose",
    "ObservationReturnConfig",
    "ObservationReturnController",
    "ObservationReturnPhase",
    "ObservationReturnStatus",
    "T01_STOCK_OBSERVATION_POSE",
    "relative_axis_angle_xyzw",
]


@dataclasses.dataclass(frozen=True)
class ObservationPose:
    """A robot-frame end-effector pose, independent of hidden scene state."""

    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64)
        quaternion = np.asarray(self.quaternion_xyzw, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position must contain three finite values")
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("quaternion_xyzw must contain four finite values")
        if not math.isclose(float(np.linalg.norm(quaternion)), 1.0, abs_tol=1e-5):
            raise ValueError("quaternion_xyzw must be normalized")


T01_STOCK_OBSERVATION_POSE = ObservationPose(
    position=(-0.2084646605658236, 0.0, 1.1732794757296403),
    quaternion_xyzw=(
        0.9995966048795359,
        0.00024621283188763776,
        -0.028400120485872867,
        -6.995295959032546e-06,
    ),
)
"""Reset pose shared exactly by all 60 T01 development seeds."""


@dataclasses.dataclass(frozen=True)
class ObservationReturnConfig:
    """Controller settings, kept separate from uncertainty-routing losses.

    ``position_tolerance`` and ``orientation_tolerance_radians`` are actuator
    completion tolerances.  They never decide whether evidence is sufficient,
    whether an information action should run, or whether to abstain.
    """

    pose: ObservationPose = T01_STOCK_OBSERVATION_POSE
    release_steps: int = 8
    maximum_return_steps: int = 160
    settled_steps: int = 5
    position_tolerance: float = 0.012
    orientation_tolerance_radians: float = 0.08
    position_output_scale: float = 0.05
    orientation_output_scale: float = 0.5
    maximum_normalized_translation: float = 0.6
    maximum_normalized_rotation: float = 0.5
    gripper_release_command: float = -1.0

    def __post_init__(self) -> None:
        if self.release_steps < 0:
            raise ValueError("release_steps must be nonnegative")
        if self.maximum_return_steps < 1:
            raise ValueError("maximum_return_steps must be positive")
        if self.settled_steps < 1:
            raise ValueError("settled_steps must be positive")
        for name in (
            "position_tolerance",
            "orientation_tolerance_radians",
            "position_output_scale",
            "orientation_output_scale",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "maximum_normalized_translation",
            "maximum_normalized_rotation",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must lie in (0, 1]")
        if not -1 <= self.gripper_release_command <= 1:
            raise ValueError("gripper_release_command must lie in [-1, 1]")


class ObservationReturnPhase(str, enum.Enum):
    RELEASE = "RELEASE"
    RETURN = "RETURN"
    COMPLETE = "COMPLETE"
    TIMED_OUT = "TIMED_OUT"


@dataclasses.dataclass(frozen=True)
class ObservationReturnStatus:
    phase: ObservationReturnPhase
    commands_issued: int
    position_error_metres: float
    orientation_error_radians: float
    consecutive_settled_steps: int

    @property
    def terminal(self) -> bool:
        return self.phase in {
            ObservationReturnPhase.COMPLETE,
            ObservationReturnPhase.TIMED_OUT,
        }

    @property
    def succeeded(self) -> bool:
        return self.phase is ObservationReturnPhase.COMPLETE


def _normalized_quaternion_xyzw(values: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must have shape (4,) with finite xyzw values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be nonzero")
    return quaternion / norm


def _quaternion_matrix_xyzw(values: Sequence[float]) -> np.ndarray:
    x, y, z, w = _normalized_quaternion_xyzw(values)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_axis_angle(matrix: np.ndarray) -> np.ndarray:
    """Return the shortest axis-angle vector for a proper rotation matrix."""

    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    if math.pi - angle <= 1e-6:
        # The usual skew formula becomes singular at pi.  Recover the axis
        # from the diagonal and choose signs from the symmetric entries.
        axis = np.sqrt(np.maximum((np.diag(matrix) + 1.0) / 2.0, 0.0))
        largest = int(np.argmax(axis))
        if axis[largest] <= 1e-9:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            for index in range(3):
                if index != largest:
                    axis[index] = (
                        matrix[index, largest] + matrix[largest, index]
                    ) / (4.0 * axis[largest])
            axis /= np.linalg.norm(axis)
        return axis * angle
    axis = np.asarray(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    ) / (2.0 * math.sin(angle))
    return axis * angle


def relative_axis_angle_xyzw(
    current_quaternion_xyzw: Sequence[float],
    desired_quaternion_xyzw: Sequence[float],
) -> np.ndarray:
    """World-frame delta expected by LIBERO's base-frame ``OSC_POSE``.

    robosuite 1.4 scales the last three controller inputs into an axis-angle
    delta and left-multiplies that delta onto the current end-effector rotation.
    Therefore ``R_delta = R_desired R_current.T``.
    """

    current = _quaternion_matrix_xyzw(current_quaternion_xyzw)
    desired = _quaternion_matrix_xyzw(desired_quaternion_xyzw)
    return _matrix_axis_angle(desired @ current.T)


class ObservationReturnController:
    """Closed-loop proprioceptive controller for ``RETURN_TO_OBSERVE``."""

    _ALLOWED_OBSERVATION_KEYS = frozenset(
        {"robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"}
    )

    def __init__(self, config: ObservationReturnConfig | None = None) -> None:
        self.config = config or ObservationReturnConfig()
        self.reset()

    def reset(self) -> None:
        self._commands_issued = 0
        self._return_commands = 0
        self._settled = 0
        self._phase = (
            ObservationReturnPhase.RELEASE
            if self.config.release_steps
            else ObservationReturnPhase.RETURN
        )
        self._last_position_error = math.inf
        self._last_orientation_error = math.inf

    @property
    def allowed_observation_keys(self) -> frozenset[str]:
        return self._ALLOWED_OBSERVATION_KEYS

    def status(self) -> ObservationReturnStatus:
        return ObservationReturnStatus(
            phase=self._phase,
            commands_issued=self._commands_issued,
            position_error_metres=self._last_position_error,
            orientation_error_radians=self._last_orientation_error,
            consecutive_settled_steps=self._settled,
        )

    def _release_action(self) -> np.ndarray:
        action = np.zeros(7, dtype=np.float64)
        action[6] = self.config.gripper_release_command
        return action

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Produce one normalized LIBERO action from policy-visible state."""

        if self.status().terminal:
            raise RuntimeError(f"controller is already terminal: {self._phase.value}")
        for key in self._ALLOWED_OBSERVATION_KEYS:
            if key not in observation:
                raise KeyError(f"missing policy-visible observation key: {key}")

        position = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
        quaternion = np.asarray(observation["robot0_eef_quat"], dtype=np.float64)
        gripper = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("robot0_eef_pos must contain three finite values")
        _normalized_quaternion_xyzw(quaternion)
        if gripper.ndim != 1 or not len(gripper) or not np.all(np.isfinite(gripper)):
            raise ValueError("robot0_gripper_qpos must be a finite nonempty vector")

        target_position = np.asarray(self.config.pose.position, dtype=np.float64)
        position_error = target_position - position
        orientation_error = relative_axis_angle_xyzw(
            quaternion, self.config.pose.quaternion_xyzw
        )
        self._last_position_error = float(np.linalg.norm(position_error))
        self._last_orientation_error = float(np.linalg.norm(orientation_error))

        if self._phase is ObservationReturnPhase.RELEASE:
            action = self._release_action()
            self._commands_issued += 1
            if self._commands_issued >= self.config.release_steps:
                self._phase = ObservationReturnPhase.RETURN
            return action

        reached = (
            self._last_position_error <= self.config.position_tolerance
            and self._last_orientation_error
            <= self.config.orientation_tolerance_radians
        )
        self._settled = self._settled + 1 if reached else 0

        action = self._release_action()
        if not reached:
            action[:3] = np.clip(
                position_error / self.config.position_output_scale,
                -self.config.maximum_normalized_translation,
                self.config.maximum_normalized_translation,
            )
            action[3:6] = np.clip(
                orientation_error / self.config.orientation_output_scale,
                -self.config.maximum_normalized_rotation,
                self.config.maximum_normalized_rotation,
            )

        self._commands_issued += 1
        self._return_commands += 1
        if self._settled >= self.config.settled_steps:
            self._phase = ObservationReturnPhase.COMPLETE
        elif self._return_commands >= self.config.maximum_return_steps:
            self._phase = ObservationReturnPhase.TIMED_OUT
        return action
