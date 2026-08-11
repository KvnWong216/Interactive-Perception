"""Evaluator-private, task-specific information endpoint scoring.

Final task success remains LIBERO's native BDDL ``done`` signal.  Information
endpoints answer a different question: did the policy causally perform the
scene interaction that makes the missing information available?  Visibility
alone is insufficient for articulated fixtures, partial-visibility controls,
and absent-target search, so endpoint predicates are declared in benchmark
metadata and combine simulator state with policy-camera evidence.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from .anchors import drawer_joint_value, object_position, visible_pixels

__all__ = ["EndpointEvaluator", "EndpointObservation"]


@dataclasses.dataclass(frozen=True)
class EndpointObservation:
    reached: bool
    target_visible_pixels: int | None
    visibility_gain: int | None
    visibility_ratio: float | None
    evidence: dict[str, Any]


class EndpointEvaluator:
    """Evaluate one declarative endpoint without exposing state to the policy."""

    def __init__(self, env: Any, obs: dict[str, Any], task: dict[str, Any], *, camera: str):
        spec = task.get("information_endpoint")
        if not isinstance(spec, dict) or not spec.get("type"):
            raise ValueError(f'{task.get("id", "task")} has no information_endpoint spec')
        self.env = env
        self.task = task
        self.spec = spec
        self.camera = camera
        self.target = task.get("target")
        self.initial_pixels = self._pixels(obs)
        object_ref = spec.get("object")
        self.initial_object_position = (
            object_position(env, str(object_ref)) if object_ref else None
        )

    @property
    def endpoint_type(self) -> str:
        return str(self.spec["type"])

    def _pixels(self, obs: dict[str, Any]) -> int | None:
        if self.target is None or self.target not in self.env.instance_to_id:
            return None
        return visible_pixels(self.env, obs, camera=self.camera, instance=str(self.target))

    def _joint_passes(self, item: dict[str, Any]) -> tuple[bool, float]:
        value = drawer_joint_value(self.env, str(item["name"]))
        threshold = float(item["threshold"])
        operator = str(item["operator"])
        if operator == "lt":
            return value < threshold, value
        if operator == "gt":
            return value > threshold, value
        raise ValueError(f"unsupported joint operator: {operator}")

    def observe(self, obs: dict[str, Any]) -> EndpointObservation:
        kind = self.endpoint_type
        pixels = self._pixels(obs)
        gain = None if pixels is None or self.initial_pixels is None else pixels - self.initial_pixels
        ratio = None
        if pixels is not None and self.initial_pixels is not None:
            ratio = pixels / max(self.initial_pixels, 1)

        evidence: dict[str, Any] = {
            "initial_target_visible_pixels": self.initial_pixels,
            "target_visible_pixels": pixels,
        }
        min_pixels = int(self.spec.get("min_target_visible_pixels", 5))
        visibility_ok = pixels is not None and pixels >= min_pixels

        if kind == "visible_target":
            reached = visibility_ok
        elif kind in {"articulated_reveal", "all_joints_open"}:
            joint_results = [self._joint_passes(item) for item in self.spec.get("joints", [])]
            if not joint_results:
                raise ValueError(f"{kind} endpoint requires joints")
            evidence["joint_values"] = {
                str(item["name"]): value
                for item, (_, value) in zip(self.spec["joints"], joint_results)
            }
            joints_ok = all(passed for passed, _ in joint_results)
            reached = joints_ok and (visibility_ok if kind == "articulated_reveal" else True)
        elif kind in {"object_reveal", "object_visibility_gain"}:
            if self.initial_object_position is None:
                raise ValueError(f"{kind} endpoint requires object")
            current = object_position(self.env, str(self.spec["object"]))
            displacement = float(np.linalg.norm(current - self.initial_object_position))
            evidence["object_displacement"] = displacement
            moved = displacement >= float(self.spec["min_object_displacement"])
            if kind == "object_reveal":
                reached = moved and visibility_ok
            else:
                min_gain = int(self.spec.get("min_visibility_gain_pixels", 1))
                min_ratio = float(self.spec.get("min_visibility_ratio", 1.0))
                reached = (
                    moved
                    and visibility_ok
                    and gain is not None
                    and gain >= min_gain
                    and ratio is not None
                    and ratio >= min_ratio
                )
        else:
            raise ValueError(f"unsupported endpoint type: {kind}")

        evidence["visibility_gain"] = gain
        evidence["visibility_ratio"] = ratio
        return EndpointObservation(bool(reached), pixels, gain, ratio, evidence)
