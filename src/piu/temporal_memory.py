"""Finite-trace memory built only from calibrated post-observation verifiers."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any


def _set(values: Sequence[bool]) -> frozenset[bool]:
    if any(not isinstance(value, bool) for value in values):
        raise TypeError("temporal verifier-set values must be booleans")
    return frozenset(values)


def _digest(value: Any) -> str:
    result = str(value)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError("post-observation digest must be lowercase SHA-256")
    return result


@dataclasses.dataclass(frozen=True)
class PublicObservationEvent:
    """One executed candidate followed by a public-RGB verifier prediction."""

    step: int
    candidate_id: str
    primitive: str
    information_source_id: str | None
    region_confirmed_empty_set: frozenset[bool]
    task_complete_set: frozenset[bool]
    post_observation_sha256: str

    @classmethod
    def create(
        cls,
        *,
        step: int,
        candidate_id: str,
        primitive: str,
        information_source_id: str | None,
        region_confirmed_empty_set: Sequence[bool],
        task_complete_set: Sequence[bool],
        post_observation_sha256: str,
    ) -> PublicObservationEvent:
        identifier = " ".join(str(candidate_id).split())
        action = " ".join(str(primitive).split()).upper()
        source = (
            None
            if information_source_id is None
            else " ".join(str(information_source_id).split())
        )
        if step < 0 or not identifier or not action or source == "":
            raise ValueError("invalid public observation-event identity")
        region_set = _set(region_confirmed_empty_set)
        task_set = _set(task_complete_set)
        if source is None and region_set:
            raise ValueError(
                "a region-empty verifier set requires an information source"
            )
        return cls(
            step=step,
            candidate_id=identifier,
            primitive=action,
            information_source_id=source,
            region_confirmed_empty_set=region_set,
            task_complete_set=task_set,
            post_observation_sha256=_digest(post_observation_sha256),
        )


@dataclasses.dataclass(frozen=True)
class PublicTemporalMemory:
    registered_information_sources: tuple[str, ...]
    current_observation_sha256: str
    events: tuple[PublicObservationEvent, ...] = ()

    def __post_init__(self) -> None:
        if not self.registered_information_sources:
            raise ValueError("search memory requires registered information sources")
        _digest(self.current_observation_sha256)
        if len(set(self.registered_information_sources)) != len(
            self.registered_information_sources
        ):
            raise ValueError("duplicate registered information source")
        if any(not source.strip() for source in self.registered_information_sources):
            raise ValueError("registered information source must be non-empty")
        if any(
            left.step >= right.step
            for left, right in zip(self.events, self.events[1:], strict=False)
        ):
            raise ValueError("temporal observation-event steps must be increasing")
        unknown = {
            event.information_source_id
            for event in self.events
            if event.information_source_id is not None
        } - set(self.registered_information_sources)
        if unknown:
            raise ValueError(
                f"observation events reference unknown sources {sorted(unknown)}"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PublicTemporalMemory:
        if value.get("schema_version") != "piu.public-temporal-memory.v2":
            raise ValueError("unsupported public temporal-memory schema")
        sources = tuple(
            " ".join(str(item).split())
            for item in value.get("registered_information_sources", ())
        )
        current_digest = _digest(value.get("current_observation_sha256"))
        events = tuple(
            PublicObservationEvent.create(
                step=int(row["step"]),
                candidate_id=str(row["candidate_id"]),
                primitive=str(row["primitive"]),
                information_source_id=row.get("information_source_id"),
                region_confirmed_empty_set=row.get("region_confirmed_empty_set", ()),
                task_complete_set=row.get("task_complete_set", ()),
                post_observation_sha256=str(row.get("post_observation_sha256", "")),
            )
            for row in value.get("events", ())
        )
        result = cls(sources, current_digest, events)
        if events and events[-1].post_observation_sha256 != current_digest:
            raise ValueError("memory head digest differs from its latest event")
        declared = {
            "confirmed_empty_sources": sorted(result.confirmed_empty_sources),
            "search_coverage_sufficient": result.search_coverage_sufficient,
            "task_complete": result.task_complete,
        }
        for name, expected in declared.items():
            if name in value and value[name] != expected:
                raise ValueError(
                    f"public temporal-memory derived field differs: {name}"
                )
        return result

    def append(self, event: PublicObservationEvent) -> PublicTemporalMemory:
        if self.events and event.step <= self.events[-1].step:
            raise ValueError("new public observation event must advance the trace")
        return PublicTemporalMemory(
            self.registered_information_sources,
            event.post_observation_sha256,
            (*self.events, event),
        )

    @property
    def confirmed_empty_sources(self) -> frozenset[str]:
        return frozenset(
            event.information_source_id
            for event in self.events
            if event.information_source_id is not None
            and event.region_confirmed_empty_set == frozenset({True})
        )

    @property
    def search_coverage_sufficient(self) -> bool:
        return self.confirmed_empty_sources == frozenset(
            self.registered_information_sources
        )

    @property
    def task_complete(self) -> bool:
        return any(
            event.task_complete_set == frozenset({True}) for event in self.events
        )

    def to_public_history(self) -> dict[str, Any]:
        return {
            "schema_version": "piu.public-temporal-memory.v2",
            "registered_information_sources": list(self.registered_information_sources),
            "current_observation_sha256": self.current_observation_sha256,
            "confirmed_empty_sources": sorted(self.confirmed_empty_sources),
            "search_coverage_sufficient": self.search_coverage_sufficient,
            "task_complete": self.task_complete,
            "events": [
                {
                    **{
                        key: value
                        for key, value in dataclasses.asdict(event).items()
                        if not key.endswith("_set")
                    },
                    "region_confirmed_empty_set": sorted(
                        event.region_confirmed_empty_set
                    ),
                    "task_complete_set": sorted(event.task_complete_set),
                }
                for event in self.events
            ],
        }
