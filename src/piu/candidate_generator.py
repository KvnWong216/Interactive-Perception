"""Public-affordance candidate generation without target-location shortcuts."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import assert_public_policy_value

INFORMATION_CAPABILITIES = {
    "OPEN": "inspect inside",
    "REMOVE": "inspect the newly uncovered region",
    "ROTATE": "inspect hidden identity evidence",
    "MOVE_CLOSER": "inspect at higher visual resolution",
    "PICK_TO_INSPECT": "inspect from a closer manipulation pose",
}


def _text(value: Any, *, name: str) -> str:
    result = " ".join(str(value or "").split())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not result:
        raise ValueError("candidate identifier has no alphanumeric content")
    return result


@dataclasses.dataclass(frozen=True)
class PublicAffordanceEntity:
    entity_id: str
    description: str
    capabilities: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PublicAffordanceEntity:
        assert_public_policy_value(value, path="candidate_generator.entity")
        entity_id = _text(value.get("entity_id"), name="entity_id")
        description = _text(value.get("description"), name="description")
        capabilities = tuple(
            dict.fromkeys(str(item).upper() for item in value.get("capabilities", ()))
        )
        unknown = set(capabilities) - set(INFORMATION_CAPABILITIES)
        if unknown:
            raise ValueError(
                f"unknown public information capabilities {sorted(unknown)}"
            )
        return cls(entity_id, description, capabilities)


def generate_candidates(
    *,
    task_target_description: str,
    destination_description: str,
    entities: Sequence[PublicAffordanceEntity],
) -> list[dict[str, Any]]:
    """Instantiate a public action superset, never a hidden correct route.

    PICK and PLACE are both retained at every decision boundary.  The calibrated
    post-observation holding belief gates them downstream.  Keeping that belief
    out of candidate construction avoids a circular dependency (the prefix is
    needed to infer the belief) and prevents a context file from injecting task
    progress into the online method.
    """

    target = _text(task_target_description, name="task_target_description")
    destination = _text(destination_description, name="destination_description")
    if len({entity.entity_id for entity in entities}) != len(entities):
        raise ValueError("duplicate public affordance entity ID")
    candidates = [
        {
            "candidate_id": f"pick_{_slug(target)}",
            "primitive": "PICK",
            "target": target,
            "reference": destination,
            "purpose": "advance the requested task",
            "required_capability": "PICK",
        },
        {
            "candidate_id": f"place_{_slug(target)}_in_{_slug(destination)}",
            "primitive": "PLACE",
            "target": target,
            "reference": destination,
            "purpose": "advance the requested task",
            "required_capability": "PLACE",
        },
    ]
    for entity in entities:
        for primitive in entity.capabilities:
            candidates.append(
                {
                    "candidate_id": f"{primitive.lower()}_{_slug(entity.entity_id)}",
                    "primitive": primitive,
                    "target": entity.description,
                    "purpose": INFORMATION_CAPABILITIES[primitive],
                    "required_capability": primitive,
                }
            )
    candidates.extend(
        [
            {
                "candidate_id": "report_not_found",
                "primitive": "REPORT_NOT_FOUND",
                "target": target,
                "purpose": "report evidence-backed search exhaustion",
                "required_capability": "REPORT_NOT_FOUND",
            },
            {
                "candidate_id": "stop",
                "primitive": "STOP",
                "target": "current task",
                "purpose": "terminate a certified complete task",
                "required_capability": "STOP",
            },
        ]
    )
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise ValueError("generated candidate IDs collide")
    assert_public_policy_value(candidates, path="candidate_generator.output")
    return candidates
