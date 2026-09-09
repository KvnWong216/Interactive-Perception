"""Public-only adapter between the existing LIBERO runtime and PSR V1."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from grounded_interaction.libero_runtime import (
    LiberoEnvironment,
    LiberoPublicObservation,
)
from grounded_interaction.rgb import RGBFrameStore

from .types import PublicHistory, PublicObservation, RGBReference


class PSRFrameStore:
    """Translate the existing content-addressed store to PSR references."""

    def __init__(self, root: str | Path) -> None:
        self.store = RGBFrameStore(root)

    def put(
        self,
        value: Any,
        *,
        frame_id: str,
        camera: str,
        frame_index: int,
    ) -> RGBReference:
        frame = self.store.put(
            value,
            frame_id=frame_id,
            camera=camera,
            frame_index=frame_index,
        )
        return RGBReference(
            frame_id=frame.frame_id,
            image_sha256=frame.image_sha256,
            width=frame.width,
            height=frame.height,
        )

    def resolve(self, reference: RGBReference) -> Any:
        if not isinstance(reference, RGBReference):
            raise TypeError("reference must be a PSR RGBReference")
        return self.store.resolve(reference)


class PSRLiberoEnvironment:
    """Expose real dual RGB/state and applied continuous actions, nothing private."""

    def __init__(
        self,
        *,
        environment: LiberoEnvironment,
        frame_store: PSRFrameStore,
        episode_namespace: str,
    ) -> None:
        if not isinstance(environment, LiberoEnvironment):
            raise TypeError("environment must be LiberoEnvironment")
        if not isinstance(frame_store, PSRFrameStore):
            raise TypeError("frame_store must be PSRFrameStore")
        namespace = " ".join(str(episode_namespace).split())
        if not namespace:
            raise ValueError("episode_namespace must be non-empty")
        self.environment = environment
        self.frame_store = frame_store
        self.episode_namespace = namespace
        self._terminal = False
        self._current = self._convert(environment.public_observation, control_step=0)

    def _convert(
        self, observation: LiberoPublicObservation, *, control_step: int
    ) -> PublicObservation:
        agent = self.frame_store.put(
            observation.agentview_rgb,
            frame_id=f"{self.episode_namespace}:{control_step}:agentview",
            camera="agentview",
            frame_index=control_step,
        )
        wrist = self.frame_store.put(
            observation.wrist_rgb,
            frame_id=f"{self.episode_namespace}:{control_step}:wrist",
            camera="wrist",
            frame_index=control_step,
        )
        return PublicObservation(
            agentview_rgb=agent,
            wrist_rgb=wrist,
            robot_state=observation.state,
            control_step=control_step,
        )

    @property
    def current(self) -> PublicObservation:
        return self._current

    def initial_history(self, *, task: str) -> PublicHistory:
        return PublicHistory(
            task=task,
            current=self._current,
            previous=(),
            executed=(),
            remaining_control_steps=300,
        )

    def step_public(self, action: Sequence[float]) -> PublicObservation:
        observation, _applied = self.environment.step(action)
        self._current = self._convert(
            observation, control_step=self._current.control_step + 1
        )
        return self._current

    def public_terminal(self) -> bool:
        # Current LIBERO exposes no deployment-visible task terminal bit.
        return self._terminal

    def mark_public_execution_failure(self) -> None:
        self._terminal = True


__all__ = ["PSRFrameStore", "PSRLiberoEnvironment"]
