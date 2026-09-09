"""Narrow public-observation LIBERO runtime used by PSR-VLA."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import sys
from pathlib import Path
from typing import Any

from .contracts import canonical_sha256
from .rgb import canonical_rgb_array


def _integration_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from libero.libero.envs import OffScreenRenderEnv
        from robosuite.utils import transform_utils as transform
    except ImportError as error:  # pragma: no cover - requires LIBERO environment.
        raise RuntimeError(
            "LIBERO integration requires the external LIBERO/robosuite environment"
        ) from error
    return np, torch, OffScreenRenderEnv, transform


def canonical_array_sha256(value: Any) -> str:
    """Hash the exact reset array including dtype and shape."""

    np, _, _, _ = _integration_dependencies()
    array = np.ascontiguousarray(np.asarray(value))
    header = (
        f"ndarray-v1\n{array.dtype.str}\n"
        + ",".join(str(int(item)) for item in array.shape)
        + "\n"
    ).encode("ascii")
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def canonical_libero_state_sha256(value: Any) -> str:
    """Hash the exact finite 8-D public state consumed by MolmoAct2."""

    try:
        state = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError("LIBERO public state must be numeric") from error
    if len(state) != 8 or any(not math.isfinite(item) for item in state):
        raise ValueError("LIBERO public state must contain 8 finite values")
    return canonical_sha256(
        {"schema_version": "libero-public-state-v1", "values": list(state)}
    )


def libero_runtime_identity() -> dict[str, str]:
    """Return simulator library versions that can alter a rollout."""

    np, torch, _, _ = _integration_dependencies()
    import mujoco
    import robosuite

    return {
        "python": sys.version.split()[0],
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "robosuite": str(robosuite.__version__),
        "mujoco": str(mujoco.__version__),
    }


@dataclasses.dataclass(frozen=True)
class LiberoPublicObservation:
    """Only the public fields passed to a policy or public frame store."""

    agentview_rgb: Any
    wrist_rgb: Any
    state: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "agentview_rgb", canonical_rgb_array(self.agentview_rgb)
        )
        object.__setattr__(self, "wrist_rgb", canonical_rgb_array(self.wrist_rgb))
        state = tuple(float(item) for item in self.state)
        if len(state) != 8 or any(not math.isfinite(item) for item in state):
            raise ValueError("LIBERO public state must contain 8 finite values")
        object.__setattr__(self, "state", state)


def libero_public_observation(
    raw_observation: dict[str, Any],
) -> LiberoPublicObservation:
    """Apply the official LIBERO camera and state convention.

    Raw robosuite images are rotated 180 degrees.  State is end-effector xyz,
    robosuite axis-angle, and two gripper positions.  No simulator object state
    or semantic identity is returned.
    """

    np, _, _, transform = _integration_dependencies()
    required = (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    missing = [key for key in required if key not in raw_observation]
    if missing:
        raise ValueError(f"LIBERO observation is missing public fields: {missing}")
    agentview = np.ascontiguousarray(raw_observation["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(
        raw_observation["robot0_eye_in_hand_image"][::-1, ::-1]
    )
    state = np.concatenate(
        (
            np.asarray(raw_observation["robot0_eef_pos"], dtype=np.float32),
            np.asarray(
                transform.quat2axisangle(raw_observation["robot0_eef_quat"]),
                dtype=np.float32,
            ),
            np.asarray(raw_observation["robot0_gripper_qpos"], dtype=np.float32),
        )
    )
    return LiberoPublicObservation(agentview, wrist, tuple(state.tolist()))


def normalize_libero_action(value: Any) -> Any:
    """Validate a raw MolmoAct2 action and apply official gripper binarization."""

    np, _, _, _ = _integration_dependencies()
    action = np.asarray(value, dtype=np.float32).reshape(-1)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("LIBERO action must contain 7 finite values")
    action = action.copy()
    action[-1] = -1.0 if action[-1] < 0 else 1.0
    return action


class LiberoEnvironment:
    """One exact-reset LIBERO environment with an evaluator-only side channel."""

    def __init__(
        self,
        *,
        bddl_file: str | Path,
        init_states_file: str | Path,
        init_state_index: int,
        env_seed: int,
        expected_reset_state_sha256: str,
        image_size: int = 256,
        settle_steps: int = 10,
        control_mode: str = "relative",
    ) -> None:
        np, torch, OffScreenRenderEnv, _ = _integration_dependencies()
        self._np = np
        self.bddl_file = Path(bddl_file).expanduser().resolve()
        self.init_states_file = Path(init_states_file).expanduser().resolve()
        if not self.bddl_file.is_file() or not self.init_states_file.is_file():
            raise FileNotFoundError("LIBERO BDDL and initial-state files must exist")
        if init_state_index < 0 or settle_steps < 0:
            raise ValueError("init_state_index and settle_steps must be non-negative")
        if control_mode != "relative":
            raise ValueError("MolmoAct2-LIBERO requires relative control mode")
        self.control_mode = control_mode
        try:
            states = torch.load(self.init_states_file, weights_only=False)
        except TypeError:  # PyTorch versions before weights_only.
            states = torch.load(self.init_states_file)
        if init_state_index >= len(states):
            raise IndexError("init_state_index is outside the LIBERO state file")
        self.reset_state = np.asarray(states[init_state_index]).copy()
        observed_digest = canonical_array_sha256(self.reset_state)
        if observed_digest != expected_reset_state_sha256:
            raise ValueError(
                "LIBERO reset-state digest mismatch: "
                f"expected {expected_reset_state_sha256}, observed {observed_digest}"
            )
        self.reset_state_sha256 = observed_digest
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(self.bddl_file),
            camera_heights=image_size,
            camera_widths=image_size,
        )
        self.env.seed(int(env_seed))
        raw = self.env.reset()
        raw = self.env.set_init_state(self.reset_state)
        dummy = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
        for _ in range(settle_steps):
            raw, _, _, _ = self.env.step(dummy)
        # Match the official MolmoAct2/LeRobot LIBERO wrapper explicitly rather
        # than inheriting a robosuite default that may change across versions.
        for robot in self.env.env.robots:
            robot.controller.use_delta = True
        if not all(bool(robot.controller.use_delta) for robot in self.env.env.robots):
            raise RuntimeError("failed to enable relative LIBERO control")
        self.raw_observation = raw
        self.public_observation = libero_public_observation(raw)
        self.step_count = 0

    def step(self, action: Any) -> tuple[LiberoPublicObservation, tuple[float, ...]]:
        """Apply one canonical float32 action and return what robosuite received."""

        processed = normalize_libero_action(action)
        raw, _, _, _ = self.env.step(processed)
        self.raw_observation = raw
        self.public_observation = libero_public_observation(raw)
        self.step_count += 1
        applied = tuple(float(item) for item in processed.tolist())
        return self.public_observation, applied

    def contacts(self, object_names: tuple[str, ...]) -> tuple[str, ...]:
        """Evaluator-only contact query; names never enter policy input."""

        found: list[str] = []
        for name in object_names:
            if name not in self.env.env.objects_dict:
                raise ValueError(f"unknown evaluator object name {name!r}")
            if self.env.env.check_contact(
                self.env.env.robots[0].gripper,
                self.env.env.objects_dict[name],
            ):
                found.append(name)
        return tuple(found)

    def is_grasping(self, object_name: str) -> bool:
        """Evaluator-only grasp query."""

        if object_name not in self.env.env.objects_dict:
            raise ValueError(f"unknown evaluator object name {object_name!r}")
        return bool(
            self.env.env._check_grasp(
                self.env.env.robots[0].gripper,
                self.env.env.objects_dict[object_name],
            )
        )

    def predicate(self, predicate: tuple[str, ...]) -> bool:
        """Evaluator-only task predicate with the same rule for either referent."""

        if len(predicate) not in {2, 3} or any(
            not str(item).strip() for item in predicate
        ):
            raise ValueError(
                "evaluator predicate must have two or three non-empty items"
            )
        return bool(self.env.env._eval_predicate(list(predicate)))

    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> LiberoEnvironment:  # noqa: PYI034 - Python 3.10 runtime
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
