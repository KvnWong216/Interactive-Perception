"""Public history and open-vocabulary intent contracts for PSR V1."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
import unicodedata
from collections.abc import Mapping
from itertools import pairwise
from typing import Any, Literal

from grounded_interaction.contracts import (
    assert_public_policy_value,
    canonical_json_bytes,
    canonical_sha256,
)

ExecutionRoute = Literal["native", "conditioned"]
EXECUTION_ROUTES = frozenset({"native", "conditioned"})
MAX_PREVIOUS_OBSERVATIONS = 2
ROBOT_STATE_DIM = 8
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _text(value: Any, name: str) -> str:
    result = " ".join(str(value or "").split())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _digest(value: Any, name: str) -> str:
    result = str(value)
    if not _SHA256.fullmatch(result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _exact(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    observed = set(value)
    if observed != keys:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(keys - observed)}, "
            f"extra={sorted(observed - keys)}"
        )
    return value


def normalize_intent_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", _text(value, "intent text")).split())


@dataclasses.dataclass(frozen=True)
class RGBReference:
    """Content-addressed image reference; a filesystem path is never a feature."""

    frame_id: str
    image_sha256: str
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _text(self.frame_id, "frame_id"))
        object.__setattr__(
            self, "image_sha256", _digest(self.image_sha256, "image_sha256")
        )
        for name in ("width", "height"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_mapping(cls, value: Any) -> RGBReference:
        item = _exact(
            value, {"frame_id", "image_sha256", "width", "height"}, "RGB reference"
        )
        return cls(
            frame_id=item["frame_id"],
            image_sha256=item["image_sha256"],
            width=item["width"],
            height=item["height"],
        )


@dataclasses.dataclass(frozen=True)
class PublicObservation:
    """Exactly the dual RGB observations and public 8D robot state."""

    agentview_rgb: RGBReference
    wrist_rgb: RGBReference
    robot_state: tuple[float, ...]
    control_step: int

    def __post_init__(self) -> None:
        if not isinstance(self.agentview_rgb, RGBReference):
            raise TypeError("agentview_rgb must be an RGBReference")
        if not isinstance(self.wrist_rgb, RGBReference):
            raise TypeError("wrist_rgb must be an RGBReference")
        state = tuple(self.robot_state)
        if len(state) != ROBOT_STATE_DIM:
            raise ValueError("robot_state must contain exactly 8 values")
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(float(v))
            for v in state
        ):
            raise ValueError("robot_state values must be finite numbers")
        object.__setattr__(self, "robot_state", tuple(float(v) for v in state))
        if (
            not isinstance(self.control_step, int)
            or isinstance(self.control_step, bool)
            or self.control_step < 0
        ):
            raise ValueError("control_step must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "agentview_rgb": self.agentview_rgb.to_dict(),
            "wrist_rgb": self.wrist_rgb.to_dict(),
            "robot_state": list(self.robot_state),
            "control_step": self.control_step,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> PublicObservation:
        item = _exact(
            value,
            {"agentview_rgb", "wrist_rgb", "robot_state", "control_step"},
            "public observation",
        )
        return cls(
            agentview_rgb=RGBReference.from_mapping(item["agentview_rgb"]),
            wrist_rgb=RGBReference.from_mapping(item["wrist_rgb"]),
            robot_state=tuple(item["robot_state"]),
            control_step=item["control_step"],
        )


@dataclasses.dataclass(frozen=True)
class ExecutedIntent:
    """Public record of continuous action execution, not a primitive label."""

    text: str
    execution_route: ExecutionRoute
    start_step: int
    end_step: int
    actions_ref: str
    public_status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", normalize_intent_text(self.text))
        if self.execution_route not in EXECUTION_ROUTES:
            raise ValueError("execution_route must be native or conditioned")
        if not isinstance(self.start_step, int) or isinstance(self.start_step, bool):
            raise TypeError("start_step must be an integer")
        if not isinstance(self.end_step, int) or isinstance(self.end_step, bool):
            raise TypeError("end_step must be an integer")
        if self.start_step < 0 or self.end_step <= self.start_step:
            raise ValueError("executed intent requires 0 <= start_step < end_step")
        object.__setattr__(self, "actions_ref", _text(self.actions_ref, "actions_ref"))
        status = _text(self.public_status, "public_status")
        if any(
            word in status.casefold()
            for word in ("oracle", "reward", "hidden", "success", "target")
        ):
            raise ValueError("public_status contains evaluator-only semantics")
        object.__setattr__(self, "public_status", status)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_mapping(cls, value: Any) -> ExecutedIntent:
        item = _exact(
            value,
            {
                "text",
                "execution_route",
                "start_step",
                "end_step",
                "actions_ref",
                "public_status",
            },
            "executed intent",
        )
        return cls(
            text=item["text"],
            execution_route=item["execution_route"],
            start_step=item["start_step"],
            end_step=item["end_step"],
            actions_ref=item["actions_ref"],
            public_status=item["public_status"],
        )


@dataclasses.dataclass(frozen=True)
class PublicHistory:
    """Finite, real-observation history H available at one decision point."""

    task: str
    current: PublicObservation
    previous: tuple[PublicObservation, ...]
    executed: tuple[ExecutedIntent, ...]
    remaining_control_steps: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", _text(self.task, "task"))
        if not isinstance(self.current, PublicObservation):
            raise TypeError("current must be a PublicObservation")
        previous = tuple(self.previous)
        if len(previous) > MAX_PREVIOUS_OBSERVATIONS:
            raise ValueError("at most two previous high-level observations are allowed")
        if any(not isinstance(item, PublicObservation) for item in previous):
            raise TypeError("previous entries must be PublicObservation values")
        steps = [item.control_step for item in previous]
        if steps != sorted(set(steps)) or any(
            step >= self.current.control_step for step in steps
        ):
            raise ValueError(
                "previous observations must be unique, ordered, and older than current"
            )
        current_ids = (
            self.current.agentview_rgb.frame_id,
            self.current.wrist_rgb.frame_id,
        )
        if any(
            (item.agentview_rgb.frame_id, item.wrist_rgb.frame_id) == current_ids
            for item in previous
        ):
            raise ValueError(
                "previous observations may not duplicate the current snapshot"
            )
        object.__setattr__(self, "previous", previous)
        executed = tuple(self.executed)
        if any(not isinstance(item, ExecutedIntent) for item in executed):
            raise TypeError("executed entries must be ExecutedIntent values")
        if any(item.end_step > self.current.control_step for item in executed):
            raise ValueError(
                "executed history cannot extend beyond the current observation"
            )
        if any(left.end_step > right.start_step for left, right in pairwise(executed)):
            raise ValueError(
                "executed intent intervals must be chronological and non-overlapping"
            )
        object.__setattr__(self, "executed", executed)
        if not isinstance(self.remaining_control_steps, int) or isinstance(
            self.remaining_control_steps, bool
        ):
            raise TypeError("remaining_control_steps must be an integer")
        if self.remaining_control_steps < 0:
            raise ValueError("remaining_control_steps must be non-negative")
        assert_public_policy_value(self.to_dict(), path="psr.public_history")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "current": self.current.to_dict(),
            "previous": [item.to_dict() for item in self.previous],
            "executed": [item.to_dict() for item in self.executed],
            "remaining_control_steps": self.remaining_control_steps,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())

    def history_text(self) -> str:
        """Fixed public text format; image references remain separate tensors."""

        lines = [
            f"Task: {self.task}",
            f"Remaining control steps: {self.remaining_control_steps}",
        ]
        if not self.executed:
            lines.append("Executed intents: none")
        else:
            lines.append("Executed intents:")
            for event in self.executed:
                lines.append(
                    f"- steps {event.start_step}-{event.end_step}; "
                    f"route {event.execution_route}; status {event.public_status}; "
                    f"intent {event.text}"
                )
        return "\n".join(lines)

    @classmethod
    def from_mapping(cls, value: Any) -> PublicHistory:
        item = _exact(
            value,
            {
                "task",
                "current",
                "previous",
                "executed",
                "remaining_control_steps",
            },
            "public history",
        )
        return cls(
            task=item["task"],
            current=PublicObservation.from_mapping(item["current"]),
            previous=tuple(PublicObservation.from_mapping(v) for v in item["previous"]),
            executed=tuple(ExecutedIntent.from_mapping(v) for v in item["executed"]),
            remaining_control_steps=item["remaining_control_steps"],
        )


@dataclasses.dataclass(frozen=True)
class IntentCandidate:
    """One open-vocabulary U candidate and its actual execution route."""

    text: str
    token_ids: tuple[int, ...]
    execution_route: ExecutionRoute

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", normalize_intent_text(self.text))
        ids = tuple(self.token_ids)
        if not ids or any(
            not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in ids
        ):
            raise ValueError("token_ids must contain non-negative integers")
        object.__setattr__(self, "token_ids", ids)
        if self.execution_route not in EXECUTION_ROUTES:
            raise ValueError("execution_route must be native or conditioned")

    @property
    def candidate_id(self) -> str:
        normalized = self.text.casefold()
        payload = {
            "schema": "psr-intent-candidate-v1",
            "route": self.execution_route,
            "text": normalized,
        }
        return "psr-u-" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def predictor_input(self) -> dict[str, Any]:
        """The complete Q input: ordinary U token IDs plus route only."""

        return {
            "token_ids": list(self.token_ids),
            "execution_route": self.execution_route,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "text": self.text,
            "token_ids": list(self.token_ids),
            "execution_route": self.execution_route,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> IntentCandidate:
        item = _exact(
            value,
            {
                "candidate_id",
                "text",
                "token_ids",
                "execution_route",
            },
            "intent candidate",
        )
        candidate = cls(
            text=item["text"],
            token_ids=tuple(item["token_ids"]),
            execution_route=item["execution_route"],
        )
        if item["candidate_id"] != candidate.candidate_id:
            raise ValueError("candidate_id does not match normalized text and route")
        return candidate


def advance_public_history(
    history: PublicHistory,
    observation: PublicObservation,
    *,
    remaining_control_steps: int,
    completed_intent: ExecutedIntent | None = None,
    high_level_boundary: bool,
    boundary_observation: PublicObservation | None = None,
) -> PublicHistory:
    """Advance from a real observation with a fixed two-boundary truncation."""

    if observation.control_step <= history.current.control_step:
        raise ValueError("a new observation must advance control_step")
    previous = history.previous
    if high_level_boundary:
        if boundary_observation is None:
            raise ValueError(
                "a completed high-level boundary requires its start observation"
            )
        if boundary_observation.control_step > history.current.control_step:
            raise ValueError("boundary observation cannot come from the future")
        previous = (*previous, boundary_observation)[-MAX_PREVIOUS_OBSERVATIONS:]
    elif boundary_observation is not None:
        raise ValueError("boundary_observation is only valid at a high-level boundary")
    executed = history.executed
    if completed_intent is not None:
        if completed_intent.end_step > observation.control_step:
            raise ValueError("completed intent ends after the new observation")
        executed = (*executed, completed_intent)
    return PublicHistory(
        task=history.task,
        current=observation,
        previous=tuple(previous),
        executed=tuple(executed),
        remaining_control_steps=remaining_control_steps,
    )


def candidate_seed(
    base_seed: int, episode_seed: int, control_step: int, candidate: IntentCandidate
) -> int:
    """Content-stable seed; list order never enters the derivation."""

    values = (base_seed, episode_seed, control_step)
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values):
        raise ValueError("seed components must be non-negative integers")
    payload = {
        "schema": "psr-candidate-seed-v1",
        "base_seed": base_seed,
        "episode_seed": episode_seed,
        "control_step": control_step,
        "candidate_id": candidate.candidate_id,
    }
    return int.from_bytes(
        hashlib.sha256(canonical_json_bytes(payload)).digest()[:8], "big"
    )
