#!/usr/bin/env python3
"""Size the B8-vs-B0 formal paired test from independent development episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.formal_design import prospective_paired_design, validate_development_episode
from piu.reproducibility import validate_repro_lock


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def verified_artifact(
    episode: Mapping[str, Any], name: str
) -> tuple[Path, str]:
    value = episode.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"paired pilot episode lacks {name} provenance")
    path = resolve(Path(str(value.get("path", ""))))
    digest = sha256(path)
    if digest != value.get("sha256"):
        raise ValueError(f"paired pilot {name} differs from its content hash")
    return path, digest


def load_arm(
    paths: list[Path], *, method_id: str, outcome: str
) -> dict[str, dict[str, Any]]:
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("paired pilot arm paths must be nonempty and unique")
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        episode = json.loads(path.read_text())
        row = validate_development_episode(
            episode, method_id=method_id, outcome=outcome
        )
        group = row["initial_state_group"]
        if group in rows:
            raise ValueError("paired pilot arm duplicates an initial-state group")
        source_path, source_digest = verified_artifact(episode, "source_state")
        identity_path, identity_digest = verified_artifact(episode, "policy_identity")
        history_path, history_digest = verified_artifact(
            episode, "public_action_history"
        )
        rows[group] = {
            **row,
            "episode": {"path": portable(path), "sha256": sha256(path)},
            "source_state": {
                "path": portable(source_path),
                "sha256": source_digest,
            },
            "policy_identity": {
                "path": portable(identity_path),
                "sha256": identity_digest,
            },
            "public_action_history": {
                "path": portable(history_path),
                "sha256": history_digest,
            },
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiments/piu_formal_analysis_v1.yaml",
    )
    parser.add_argument(
        "--baseline-registry",
        type=Path,
        default=ROOT / "configs/experiments/piu_baselines_v1.yaml",
    )
    parser.add_argument(
        "--repro-manifest",
        type=Path,
        default=ROOT / "configs/experiments/piu_offline_repro_v3.yaml",
    )
    parser.add_argument(
        "--repro-lock",
        type=Path,
        default=ROOT / "results/diagnostics/piu_offline_repro_preflight_v3.json",
    )
    parser.add_argument("--treatment-episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--comparator-episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = resolve(args.config)
    registry_path = resolve(args.baseline_registry)
    repro_manifest_path = resolve(args.repro_manifest)
    repro_lock_path = resolve(args.repro_lock)
    treatment_paths = [resolve(path) for path in args.treatment_episodes]
    comparator_paths = [resolve(path) for path in args.comparator_episodes]
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("formal paired-test plans are immutable")
    validate_repro_lock(
        repro_lock_path,
        manifest_path=repro_manifest_path,
        repository_root=ROOT,
    )
    config = yaml.safe_load(config_path.read_text())
    registry = yaml.safe_load(registry_path.read_text())
    if config.get("schema_version") != "piu.formal-analysis-experiment.v1":
        raise ValueError("unsupported formal-analysis experiment")
    if registry.get("schema_version") != "piu.baseline-registry.v1":
        raise ValueError("unsupported baseline registry")
    specification = config["prospective_design"]
    primary = config["primary"]
    if (
        specification.get("pilot_split") != "development"
        or specification.get("pilot_significance_is_a_gate") is not False
        or specification.get("pilot_groups_excluded_from_formal") is not True
    ):
        raise ValueError("paired pilot isolation contract changed")
    for name in ("treatment", "comparator", "outcome"):
        if specification[name] != primary[name]:
            raise ValueError(f"prospective design differs from primary at {name}")
    treatment_id = str(specification["treatment"])
    comparator_id = str(specification["comparator"])
    outcome = str(specification["outcome"])
    treatment = load_arm(
        treatment_paths, method_id=treatment_id, outcome=outcome
    )
    comparator = load_arm(
        comparator_paths, method_id=comparator_id, outcome=outcome
    )
    if set(treatment) != set(comparator):
        raise ValueError("pilot arms are not paired by initial-state group")
    groups = sorted(treatment)
    if len({treatment[group]["source_state"]["sha256"] for group in groups}) != len(
        groups
    ):
        raise ValueError("pilot groups reuse an opaque source state")
    identities = {
        row["policy_identity"]["sha256"]
        for arm in (treatment, comparator)
        for row in arm.values()
    }
    if len(identities) != 1:
        raise ValueError("paired pilot arms use different frozen policy identities")
    identity_path = resolve(
        Path(registry["shared_contract"]["checkpoint_identity"])
    )
    if identities != {sha256(identity_path)}:
        raise ValueError("paired pilot does not use the registered frozen policy")
    pair_records = []
    for group in groups:
        left = treatment[group]
        right = comparator[group]
        if (
            left["source_state"]["sha256"] != right["source_state"]["sha256"]
            or left["simulator_seed"] != right["simulator_seed"]
        ):
            raise ValueError("pilot pair differs in source state or simulator seed")
        pair_records.append(
            {
                "initial_state_group": group,
                "simulator_seed": left["simulator_seed"],
                "source_state_sha256": left["source_state"]["sha256"],
                "treatment_outcome": left["outcome"],
                "comparator_outcome": right["outcome"],
                "treatment_episode": left["episode"],
                "comparator_episode": right["episode"],
            }
        )
    planned = prospective_paired_design(
        [bool(treatment[group]["outcome"]) for group in groups],
        [bool(comparator[group]["outcome"]) for group in groups],
        alpha=float(primary["alpha"]),
        target_power=float(specification["target_power"]),
        report_confidence=float(specification["report_confidence"]),
        design_joint_confidence=float(
            specification["conservative_joint_confidence"]
        ),
        search_limit=int(specification["numerical_search_limit"]),
    )
    result = {
        "schema_version": "piu.formal-paired-test-plan.v1",
        "status": planned["status"],
        "claim_scope": "DESIGN_ONLY_INDEPENDENT_PILOT_NOT_FORMAL_EVIDENCE",
        "config": {"path": portable(config_path), "sha256": sha256(config_path)},
        "baseline_registry": {
            "path": portable(registry_path),
            "sha256": sha256(registry_path),
        },
        "offline_repro_lock": {
            "path": portable(repro_lock_path),
            "sha256": sha256(repro_lock_path),
            "manifest_sha256": sha256(repro_manifest_path),
        },
        "comparison": {
            "treatment": treatment_id,
            "comparator": comparator_id,
            "outcome": outcome,
            "test": primary["test"],
            "alpha": float(primary["alpha"]),
            "target_power": float(specification["target_power"]),
        },
        "pilot": {
            "split": "development",
            "groups": groups,
            "pairs": pair_records,
            "policy_identity_sha256": next(iter(identities)),
            "excluded_from_formal_analysis": True,
            "significance_used_as_gate": False,
        },
        **planned,
        "numerical_search_limit": int(specification["numerical_search_limit"]),
        "numerical_search_limit_role": "resource bound, never a success threshold",
        "warning": (
            "The formal cohort must be group-disjoint from every pilot pair. "
            "A blocked plan is retained and cannot be replaced by a hand-picked N."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
