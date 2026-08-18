"""Strict loader for the single RSS seed/data provenance registry."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

__all__ = ["SeedBlock", "load_seed_registry", "seeds_for_blocks"]


@dataclasses.dataclass(frozen=True)
class SeedBlock:
    identifier: str
    seeds: tuple[int, ...]
    role: str
    status: str


def _expand_seeds(value: Any) -> tuple[int, ...]:
    if isinstance(value, list):
        seeds = tuple(int(seed) for seed in value)
    elif isinstance(value, str) and "-" in value:
        lower_text, upper_text = value.split("-", maxsplit=1)
        lower, upper = int(lower_text), int(upper_text)
        if upper < lower:
            raise ValueError(f"invalid descending seed range: {value}")
        seeds = tuple(range(lower, upper + 1))
    else:
        raise ValueError(f"seed block must be a list or inclusive range: {value!r}")
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError(f"invalid seed block: {value!r}")
    return seeds


def load_seed_registry(path: str | Path) -> Mapping[str, SeedBlock]:
    raw = yaml.safe_load(Path(path).read_text())
    if raw.get("schema_version") != "interactive-perception.seed-registry.v1":
        raise ValueError("unsupported seed registry schema")
    blocks: dict[str, SeedBlock] = {}
    owner: dict[int, str] = {}
    for item in raw.get("blocks", ()):  # type: ignore[union-attr]
        identifier = str(item["id"])
        if identifier in blocks:
            raise ValueError(f"duplicate seed block id: {identifier}")
        seeds = _expand_seeds(item["seeds"])
        for seed in seeds:
            if seed in owner:
                raise ValueError(
                    f"seed {seed} overlaps blocks {owner[seed]} and {identifier}"
                )
            owner[seed] = identifier
        blocks[identifier] = SeedBlock(
            identifier=identifier,
            seeds=seeds,
            role=str(item["role"]),
            status=str(item["status"]),
        )
    if not blocks:
        raise ValueError("seed registry contains no blocks")
    return blocks


def seeds_for_blocks(
    registry: Mapping[str, SeedBlock], identifiers: Iterable[str]
) -> list[int]:
    selected: list[int] = []
    for identifier in identifiers:
        try:
            selected.extend(registry[identifier].seeds)
        except KeyError as error:
            raise KeyError(f"unknown seed block: {identifier}") from error
    return selected
