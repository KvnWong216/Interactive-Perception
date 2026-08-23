#!/usr/bin/env python3
"""Assemble one complete, group-disjoint development baseline arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.formal_design import validate_development_episode
from piu.splits import load_split_manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def verified_reference(value: object, *, name: str) -> Path:
    if not isinstance(value, dict):
        raise TypeError(f"development episode lacks {name} provenance")
    path = resolve(Path(str(value.get("path", ""))))
    if not path.is_file() or sha256(path) != value.get("sha256"):
        raise ValueError(f"development episode {name} differs from its hash")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, nargs="+", required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--baseline-registry",
        type=Path,
        default=ROOT / "configs/experiments/piu_baselines_v1.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episode_paths = [resolve(path) for path in args.episode]
    split_path = resolve(args.split_manifest)
    registry_path = resolve(args.baseline_registry)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("development episode-arm outputs are immutable")
    if not episode_paths or len({path.resolve() for path in episode_paths}) != len(
        episode_paths
    ):
        raise ValueError("development episode paths must be nonempty and unique")
    registry = yaml.safe_load(registry_path.read_text())
    if registry.get("schema_version") != "piu.baseline-registry.v1":
        raise ValueError("unsupported baseline registry")
    registered_methods = {str(row["id"]) for row in registry["methods"]}
    if args.method_id not in registered_methods:
        raise ValueError("development method is absent from the frozen registry")
    identity_path = resolve(
        Path(registry["shared_contract"]["checkpoint_identity"])
    )
    identity_digest = sha256(identity_path)
    split = load_split_manifest(split_path)
    assignments = {
        str(row["initial_state_group"]): int(row["seed"])
        for row in split["assignments"]
        if row["split_role"] == "development"
    }
    if not assignments:
        raise ValueError("split manifest has no development allocation")
    by_group: dict[str, dict] = {}
    state_hashes: set[str] = set()
    for path in episode_paths:
        value = json.loads(path.read_text())
        summary = validate_development_episode(
            value, method_id=args.method_id, outcome="task_success"
        )
        group = str(summary["initial_state_group"])
        if group in by_group:
            raise ValueError("development arm duplicates an initial-state group")
        source = verified_reference(value.get("source_state"), name="source state")
        identity = verified_reference(
            value.get("policy_identity"), name="policy identity"
        )
        verified_reference(
            value.get("public_action_history"), name="public action history"
        )
        source_digest = sha256(source)
        if (
            group not in assignments
            or summary["simulator_seed"] != assignments[group]
            or sha256(identity) != identity_digest
            or source_digest in state_hashes
        ):
            raise ValueError(
                "development episode differs from its group, seed, policy, or state"
            )
        state_hashes.add(source_digest)
        by_group[group] = value
    if set(by_group) != set(assignments):
        raise ValueError("development arm does not cover the complete frozen allocation")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(by_group[group], sort_keys=True, allow_nan=False) + "\n"
            for group in sorted(by_group)
        )
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256(output),
                "method_id": args.method_id,
                "groups": len(by_group),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
