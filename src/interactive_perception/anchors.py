"""Evaluator-private scene readout.

Everything in this module reads simulator ground truth: object poses, drawer
joint angles, instance segmentation.  None of it may reach a policy
observation.  It exists so the evaluator can answer two questions the policy is
never told the answer to -- *where are the task-relevant places in this scene*
and *did the information the task needed ever become visible*.

Anchors are declared per task in ``benchmark.yaml`` rather than hardcoded here,
so adding a scenario does not require touching this file.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from enum import Enum
from typing import Any

import numpy as np

__all__ = [
    "AnchorRole",
    "AnchorSpec",
    "ResolvedAnchor",
    "drawer_joint_value",
    "eef_position",
    "object_position",
    "resolve_anchors",
    "visible_pixels",
]


class AnchorRole(str, Enum):
    """What a location means for the task.

    The role decides which coarse primitive an action toward that location
    counts as: reaching for the target is ``ACT``, reaching for the drawer
    front that hides it is ``REMOVE_OCCLUDER``.  Without roles, both look like
    identical purposeful motion.
    """

    TASK_TARGET = "task_target"
    PLACEMENT = "placement"
    OCCLUDER = "occluder"
    LABEL_SURFACE = "label_surface"
    DISTRACTOR = "distractor"


@dataclasses.dataclass(frozen=True)
class AnchorSpec:
    """A declarative reference to a scene location."""

    label: str
    role: AnchorRole
    kind: str
    ref: str

    def __post_init__(self) -> None:
        if self.kind not in {"object", "site", "body"}:
            raise ValueError(f"unsupported anchor kind: {self.kind}")
        if not self.label.strip() or not self.ref.strip():
            raise ValueError("anchor label and ref must be non-empty")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AnchorSpec:
        return cls(
            label=str(payload["label"]),
            role=AnchorRole(str(payload["role"])),
            kind=str(payload.get("kind", "object")),
            ref=str(payload["ref"]),
        )


@dataclasses.dataclass(frozen=True)
class ResolvedAnchor:
    label: str
    role: AnchorRole
    position: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "role": self.role.value,
            "position": list(self.position),
        }


def object_position(env: Any, instance: str) -> np.ndarray:
    obj = env.env.objects_dict[instance]
    if len(obj.joints) != 1:
        raise ValueError(f"{instance} is not a single-free-joint object: {obj.joints}")
    qpos = np.asarray(
        env.env.sim.data.get_joint_qpos(obj.joints[0]), dtype=np.float64
    ).ravel()
    if qpos.size < 7:
        # Fixtures carry slide and hinge joints whose qpos is a scalar, not a
        # pose. Silently slicing one would yield a meaningless anchor position.
        raise ValueError(
            f"{instance}.{obj.joints[0]} has qpos of size {qpos.size}, so it is not "
            f'a free joint; declare this anchor with kind: "body" or "site" instead'
        )
    return qpos[:3].copy()


def _site_position(env: Any, site: str) -> np.ndarray:
    site_id = env.env.sim.model.site_name2id(site)
    return np.asarray(env.env.sim.data.site_xpos[site_id], dtype=np.float64).copy()


def _body_position(env: Any, body: str) -> np.ndarray:
    body_id = env.env.sim.model.body_name2id(body)
    return np.asarray(env.env.sim.data.body_xpos[body_id], dtype=np.float64).copy()


def resolve_anchors(env: Any, specs: Sequence[AnchorSpec]) -> list[ResolvedAnchor]:
    """Look up current world positions for every declared anchor."""

    resolvers = {
        "object": object_position,
        "site": _site_position,
        "body": _body_position,
    }
    resolved: list[ResolvedAnchor] = []
    for spec in specs:
        position = resolvers[spec.kind](env, spec.ref)
        resolved.append(
            ResolvedAnchor(
                label=spec.label,
                role=spec.role,
                position=tuple(float(value) for value in position),  # type: ignore[arg-type]
            )
        )
    return resolved


def eef_position(obs: dict[str, Any]) -> np.ndarray:
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()


def _segmentation_key(obs: dict[str, Any], camera: str) -> str:
    keys = [key for key in obs if key.startswith(camera) and "segmentation" in key]
    if len(keys) != 1:
        raise KeyError(
            f"expected exactly one {camera} segmentation key, got {keys}; "
            f"available={sorted(obs)}"
        )
    return keys[0]


def visible_pixels(env: Any, obs: dict[str, Any], *, camera: str, instance: str) -> int:
    """Count segmentation pixels for one instance in one camera.

    This is the endpoint measurement the graded metrics depend on: whether the
    information the task needed ever entered the policy's view, independent of
    whether the policy went on to succeed.
    """

    instance_id = env.instance_to_id[instance]
    segmentation = np.asarray(obs[_segmentation_key(obs, camera)]).squeeze()
    return int(np.count_nonzero(segmentation == instance_id))


def drawer_joint_value(env: Any, joint: str) -> float:
    """Read an articulated joint, e.g. how far a drawer has been pulled out."""

    return float(np.asarray(env.env.sim.data.get_joint_qpos(joint), dtype=np.float64).ravel()[0])
