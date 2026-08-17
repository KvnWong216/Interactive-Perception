"""Policy backends and the repeated-sampling probe used to read uncertainty.

The observation contract here is a deliberate copy of ``examples/libero/main.py``
in openpi: 180-degree image rotation, pad-preserving resize to 224, and an
eight-dimensional state of end-effector position, axis-angle orientation, and
gripper joints.  Any drift from that contract silently degrades the policy and
would be indistinguishable from the failures this benchmark is trying to
measure, so it is reproduced rather than reimplemented.

The probe rests on one property of openpi's server: :meth:`Policy.infer` splits
its JAX PRNG key on every call, so querying the same frozen observation ``n``
times draws ``n`` independent flow-matching samples.  No server-side change is
needed to obtain an action distribution.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

__all__ = [
    "LIBERO_ENV_RESOLUTION",
    "ObservationPacket",
    "OpenPiWebsocketPolicy",
    "PolicyBackend",
    "ScriptedStubPolicy",
    "build_observation",
    "quat2axisangle",
]

LIBERO_ENV_RESOLUTION = 256
"""Render resolution LIBERO training data used; kept for parity with openpi."""

DEFAULT_RESIZE = 224
DEFAULT_ACTION_DIM = 7


def quat2axisangle(quat: Sequence[float]) -> np.ndarray:
    """Convert a robosuite ``xyzw`` quaternion to an axis-angle vector.

    Ported from robosuite via openpi's LIBERO example so that the state vector
    fed to the policy matches its training distribution exactly.
    """

    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat.shape != (4,):
        raise ValueError("quaternion must have shape (4,) in xyzw order, now with {0} (shape: {1})".format(quat, quat.shape))
    quat[3] = float(np.clip(quat[3], -1.0, 1.0))
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


@dataclasses.dataclass(frozen=True)
class ObservationPacket:
    """A policy-visible observation plus the prompt it is conditioned on.

    Nothing evaluator-private may appear here.  Segmentation, object poses, and
    scenario metadata stay on the evaluator side; see ``anchors.py``.
    """

    image: np.ndarray
    wrist_image: np.ndarray
    state: np.ndarray
    prompt: str

    def to_openpi(self) -> dict[str, Any]:
        return {
            "observation/image": self.image,
            "observation/wrist_image": self.wrist_image,
            "observation/state": self.state,
            "prompt": self.prompt,
        }


def build_observation(
    obs: dict[str, Any],
    prompt: str,
    *,
    resize_size: int = DEFAULT_RESIZE,
    primary_camera: str = "agentview",
    wrist_camera: str = "robot0_eye_in_hand",
) -> ObservationPacket:
    """Convert a raw LIBERO observation into the policy's input format.

    The 180-degree rotation is not cosmetic: LIBERO renders images flipped
    relative to the frames the checkpoint was trained on, and omitting it costs
    most of the policy's success rate.
    """

    from openpi_client import image_tools  # imported lazily; CPU-only paths skip it

    image = np.ascontiguousarray(obs[f"{primary_camera}_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs[f"{wrist_camera}_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(image, resize_size, resize_size)
    )
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist, resize_size, resize_size)
    )
    state = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
            quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64),
        )
    )
    return ObservationPacket(image=image, wrist_image=wrist, state=state, prompt=prompt)


class PolicyBackend(Protocol):
    """Anything that can be asked for action chunks on a frozen observation."""

    def sample_chunks(self, packet: ObservationPacket, count: int) -> list[np.ndarray]:
        """Return ``count`` independently sampled ``(horizon, 7)`` chunks."""


@dataclasses.dataclass
class OpenPiWebsocketPolicy:
    """A ``pi05_libero`` policy served by ``openpi/scripts/serve_policy.py``.

    Set ``host``/``port`` to the machine holding the GPU.  The client itself is
    pure CPU, so the scenario process and the model process can live on
    different hosts.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None
    _client: Any = dataclasses.field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        from openpi_client import websocket_client_policy

        self._client = websocket_client_policy.WebsocketClientPolicy(
            self.host, self.port, api_key=self.api_key
        )

    @property
    def server_metadata(self) -> dict[str, Any]:
        return dict(self._client.get_server_metadata())

    def sample_chunks(self, packet: ObservationPacket, count: int) -> list[np.ndarray]:
        if count < 1:
            raise ValueError("count must be >= 1")
        payload = packet.to_openpi()
        chunks: list[np.ndarray] = []
        for _ in range(count):
            # Each call advances the server's PRNG, so repeated queries on this
            # identical payload return independent flow-matching samples.
            chunk = np.asarray(self._client.infer(payload)["actions"], dtype=np.float64)
            if chunk.ndim != 2 or chunk.shape[1] < DEFAULT_ACTION_DIM:
                raise ValueError(f"unexpected action chunk shape {chunk.shape}")
            chunks.append(chunk)
        return chunks

    def encode_prefix(self, packet: ObservationPacket) -> np.ndarray:
        """Read frozen multimodal prefix features from the extended server.

        Prefix requests do not sample an action and therefore must not advance
        the server-side flow-matching PRNG.  The stock OpenPI server does not
        implement this request; launch ``serve_pi05_with_prefix.py``.
        """

        payload = {**packet.to_openpi(), "__request_type": "prefix"}
        response = self._client.infer(payload)
        if "prefix_features" not in response:
            raise RuntimeError(
                "policy server does not expose frozen prefix features; use the "
                "versioned extended server"
            )
        features = np.asarray(response["prefix_features"], dtype=np.float32)
        if features.shape != (8192,) or not np.all(np.isfinite(features)):
            raise ValueError(f"unexpected prefix feature shape/value: {features.shape}")
        return features


@dataclasses.dataclass
class ScriptedStubPolicy:
    """A CPU stand-in that exercises the pipeline without a model server.

    This is NOT a baseline and must NEVER appear in reported results.  It
    exists so the rollout loop, trace schema, metrics, and plots can be
    VALIDATED on a machine with no GPU.  It drives toward a supplied world
    position with Gaussian noise, which is enough to produce traces with
    realistic structure.
    """

    goal_position: tuple[float, float, float] = (0.0, 0.0, 1.0)
    horizon: int = 10
    # Net chunk translation, in the controller's normalized action units rather
    # than metres. Matched to measured pi05_libero output on these scenes (1.1
    # to 9.3, median 6.0) so that a stub trace exercises the same decisiveness
    # regime as a real one; a metre-scale stub would sit far below
    # AttributionConfig.motion_scale and never register as committed.
    step_gain: float = 6.0
    noise_scale: float = 0.3
    grip_distance: float = 0.05
    seed: int = 0
    _rng: np.random.Generator = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def sample_chunks(self, packet: ObservationPacket, count: int) -> list[np.ndarray]:
        if count < 1:
            raise ValueError("count must be >= 1")
        eef = np.asarray(packet.state[:3], dtype=np.float64)
        direction = np.asarray(self.goal_position, dtype=np.float64) - eef
        distance = float(np.linalg.norm(direction))
        if distance > 1e-9:
            direction = direction / distance
        gripper = -1.0 if distance > self.grip_distance else 1.0

        chunks: list[np.ndarray] = []
        for _ in range(count):
            chunk = np.zeros((self.horizon, DEFAULT_ACTION_DIM), dtype=np.float64)
            per_step = direction * self.step_gain / self.horizon
            chunk[:, 0:3] = per_step[None, :] + self._rng.normal(
                0.0, self.noise_scale, size=(self.horizon, 3)
            )
            chunk[:, 6] = gripper
            chunks.append(chunk)
        return chunks
