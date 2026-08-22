"""Capability grounding and deterministic frozen-VLA subtask serialization."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .contracts import CandidateAction, Primitive


@dataclasses.dataclass(frozen=True)
class Capability:
    primitive: Primitive
    executor: str | None
    instruction_template: str | None

    @property
    def executable(self) -> bool:
        return self.executor is not None


class CapabilityRegistry:
    """The sole authority for whether a VLM-proposed primitive is executable."""

    def __init__(
        self, capabilities: Mapping[Primitive, Capability], *, registry_id: str
    ):
        self._capabilities = dict(capabilities)
        self.registry_id = str(registry_id)
        if set(self._capabilities) != set(Primitive):
            missing = sorted(
                primitive.value
                for primitive in set(Primitive) - set(self._capabilities)
            )
            extra = sorted(
                str(primitive) for primitive in set(self._capabilities) - set(Primitive)
            )
            raise ValueError(
                f"registry must define every primitive; missing={missing}, extra={extra}"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CapabilityRegistry:
        if value.get("schema_version") != "calibrated-interaction.capabilities.v1":
            raise ValueError("unsupported capability registry schema")
        rows = value.get("primitives")
        if not isinstance(rows, Mapping):
            raise TypeError("primitives must be a mapping")
        capabilities: dict[Primitive, Capability] = {}
        for name, row in rows.items():
            primitive = Primitive(str(name))
            if not isinstance(row, Mapping):
                raise TypeError(f"invalid capability row for {name}")
            capabilities[primitive] = Capability(
                primitive=primitive,
                executor=(
                    str(row["executor"]) if row.get("executor") is not None else None
                ),
                instruction_template=(
                    str(row["instruction_template"])
                    if row.get("instruction_template") is not None
                    else None
                ),
            )
        return cls(capabilities, registry_id=str(value.get("id", "")))

    @classmethod
    def load(cls, path: str | Path) -> CapabilityRegistry:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
        if not isinstance(value, Mapping):
            raise TypeError("capability registry root must be a mapping")
        return cls.from_mapping(value)

    def validate(self, candidate: CandidateAction) -> None:
        if candidate.primitive not in self._capabilities:
            raise ValueError(f"primitive {candidate.primitive.value} is not registered")
        capability = self._capabilities[candidate.primitive]
        if candidate.primitive is Primitive.STOP:
            if capability.executable:
                raise ValueError("STOP must not invoke the low-level executor")
        elif not capability.executable or not capability.instruction_template:
            raise ValueError(
                f"primitive {candidate.primitive.value} has no executable adapter"
            )

    def serialize(self, candidate: CandidateAction, *, task_prompt: str) -> str | None:
        """Produce a short action instruction; uncertainty/effects never enter the prompt."""

        self.validate(candidate)
        capability = self._capabilities[candidate.primitive]
        if not capability.executable:
            return None
        instruction = capability.instruction_template.format(
            target=candidate.target or "",
            reference=candidate.reference or "",
            task_prompt=" ".join(task_prompt.split()),
        )
        instruction = " ".join(instruction.split()).strip()
        if not instruction:
            raise ValueError("serialized executor instruction is empty")
        return instruction
