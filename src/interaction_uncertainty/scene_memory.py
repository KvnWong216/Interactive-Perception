"""Deterministic public control memory for PIU."""

from __future__ import annotations

import dataclasses
from collections import Counter
from typing import Any

from .contracts import EffectOutcome, ScenePacket, TaskBelief


@dataclasses.dataclass
class SceneMemory:
    packets: list[ScenePacket] = dataclasses.field(default_factory=list)
    beliefs: list[TaskBelief] = dataclasses.field(default_factory=list)
    searched_regions: set[str] = dataclasses.field(default_factory=set)
    attempt_counts: Counter[str] = dataclasses.field(default_factory=Counter)
    observed_outcomes: list[tuple[str, tuple[str, ...]]] = dataclasses.field(default_factory=list)

    def append_observation(self, packet: ScenePacket, belief: TaskBelief) -> None:
        self.packets.append(packet)
        self.beliefs.append(belief)

    def begin(self, candidate_id: str) -> None:
        self.attempt_counts[candidate_id] += 1

    def accept_outcome(self, candidate_id: str, prediction_set: tuple[str, ...], *, searched_region: str | None) -> None:
        self.observed_outcomes.append((candidate_id, prediction_set))
        if prediction_set == (EffectOutcome.EMPTY.value,) and searched_region:
            self.searched_regions.add(searched_region)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_count": len(self.packets),
            "belief_count": len(self.beliefs),
            "searched_regions": sorted(self.searched_regions),
            "attempt_counts": dict(self.attempt_counts),
            "observed_outcomes": [
                {"candidate_id": candidate, "prediction_set": list(labels)}
                for candidate, labels in self.observed_outcomes
            ],
        }
