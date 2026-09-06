"""Native text-only boundary to a separately hosted frozen VLA executor.

No model is imported, downloaded, or mutated in this module. In particular,
the latent router never injects soft tokens into MolmoAct2: a calibrated
singleton primitive is rendered as public subtask text at this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class FrozenVLAObservation:
    rgb: np.ndarray
    proprioception: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.rgb.ndim not in (3, 4):
            raise ValueError("rgb must be [H,W,C] or [V,H,W,C]")
        if self.rgb.shape[-1] != 3:
            raise ValueError("rgb must have three channels")
        if not np.isfinite(self.rgb).all():
            raise ValueError("rgb must be finite")
        if self.proprioception is not None and not np.isfinite(
            self.proprioception
        ).all():
            raise ValueError("proprioception must be finite")


@dataclass(frozen=True)
class FrozenVLAActionChunk:
    actions: np.ndarray
    executor_id: str

    def __post_init__(self) -> None:
        if self.actions.ndim != 2 or min(self.actions.shape) < 1:
            raise ValueError("actions must have shape [horizon, action_dim]")
        if not np.isfinite(self.actions).all():
            raise ValueError("actions must be finite")
        if not self.executor_id:
            raise ValueError("executor_id is required")


@runtime_checkable
class FrozenMolmoAct2TextExecutor(Protocol):
    """Structural interface implemented by an external frozen MolmoAct2 host."""

    @property
    def frozen(self) -> bool: ...

    @property
    def executor_id(self) -> str: ...

    def act(
        self,
        observation: FrozenVLAObservation,
        *,
        subtask_text: str,
    ) -> FrozenVLAActionChunk:
        """Return a flow-matching action chunk for non-empty public text."""


def validate_frozen_executor(executor: FrozenMolmoAct2TextExecutor) -> None:
    if not isinstance(executor, FrozenMolmoAct2TextExecutor):
        raise TypeError("executor does not implement FrozenMolmoAct2TextExecutor")
    if executor.frozen is not True:
        raise ValueError("stage-two executor must be frozen")
    if not executor.executor_id.strip():
        raise ValueError("executor_id must be non-empty")

