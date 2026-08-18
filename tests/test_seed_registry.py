from pathlib import Path

from interactive_perception.seed_registry import load_seed_registry, seeds_for_blocks


def test_authoritative_seed_registry_has_disjoint_frozen_blocks() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = load_seed_registry(root / "benchmarks/rss_v1/seed_registry.yaml")
    assert seeds_for_blocks(registry, ["t01_open_observe_clean_extension"]) == list(
        range(660, 700)
    )
    assert seeds_for_blocks(
        registry, ["t01_open_observe_v10_clean_development"]
    ) == list(range(1400, 1440))
    assert seeds_for_blocks(registry, ["t01_open_observe_sealed_audit"]) == list(
        range(900, 1000)
    )
    assert seeds_for_blocks(registry, ["t01_open_observe_smoke"]) == [1399]


def test_historical_development_composes_without_touching_clean_extension() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = load_seed_registry(root / "benchmarks/rss_v1/seed_registry.yaml")
    seeds = seeds_for_blocks(
        registry,
        [
            "t01_open_observe_prototype_train",
            "t01_open_observe_conformal_calibration",
            "t01_open_observe_v8_v9_diagnostic",
        ],
    )
    assert seeds == list(range(600, 660))
