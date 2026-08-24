#!/usr/bin/env python3
"""Freeze a purpose-specific qualification split without reading outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.primitive_registry import load_primitive_qualification_plan
from piu.splits import load_split_manifest

_SEED_PATH_TOKEN = re.compile(r"(?i)(?:^|[^a-z0-9])seed[_-]?(\d+)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def reference(path: Path) -> dict[str, str]:
    return {"path": portable(path), "sha256": sha256(path)}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return None


def _collect_seed_fields(value: Any, *, path: str, output: dict[int, set[str]]) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            key_path = f"{path}:{raw_key}"
            is_seed_field = (
                key in {"seed", "seeds", "simulator_seed", "simulator_seeds"}
                or key.endswith("_seed")
                or key.endswith("_seeds")
                or key == "seed_start"
            )
            if is_seed_field:
                scalar = _integer(child)
                if scalar is not None:
                    output[scalar].add(key_path)
                elif isinstance(child, Sequence) and not isinstance(
                    child, (str, bytes, bytearray)
                ):
                    for item in child:
                        seed = _integer(item)
                        if seed is not None:
                            output[seed].add(key_path)
            _collect_seed_fields(child, path=path, output=output)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for child in value:
            _collect_seed_fields(child, path=path, output=output)


def _documents(relative: str, text: str) -> list[Any]:
    suffix = Path(relative).suffix.lower()
    if suffix == ".json":
        return [json.loads(text)]
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix in {".yaml", ".yml"}:
        return [item for item in yaml.safe_load_all(text) if item is not None]
    return []


def _seed_inventory(commit: str, roots: Sequence[str]) -> tuple[list[dict], list[str]]:
    tracked = [
        line
        for line in _git("ls-tree", "-r", "--name-only", commit, "--", *roots).splitlines()
        if line
    ]
    observed: dict[int, set[str]] = defaultdict(set)
    parse_errors = []
    for relative in tracked:
        for token in _SEED_PATH_TOKEN.findall(relative):
            observed[int(token)].add(f"{relative}:path_token")
        if Path(relative).suffix.lower() not in {".json", ".jsonl", ".yaml", ".yml"}:
            continue
        text = _git("show", f"{commit}:{relative}")
        try:
            documents = _documents(relative, text)
        except (json.JSONDecodeError, yaml.YAMLError) as exc:
            parse_errors.append(f"{relative}: {exc}")
            continue
        for document in documents:
            _collect_seed_fields(document, path=relative, output=observed)
    if parse_errors:
        raise ValueError("seed inventory could not parse tracked metadata: " + "; ".join(parse_errors))
    return [
        {"seed": seed, "sources": sorted(sources)}
        for seed, sources in sorted(observed.items())
    ], tracked


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--seed-inventory-output", type=Path, required=True)
    parser.add_argument("--split-output", type=Path, required=True)
    parser.add_argument("--contexts-output", type=Path, required=True)
    args = parser.parse_args()
    config_path = resolve(args.config)
    plan_path = resolve(args.plan)
    inventory_path = resolve(args.seed_inventory_output)
    split_path = resolve(args.split_output)
    contexts_path = resolve(args.contexts_output)
    for output in (inventory_path, split_path, contexts_path):
        if output.exists():
            raise FileExistsError(f"qualification allocation artifact is immutable: {output}")

    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, Mapping) or config.get("schema_version") != (
        "piu.primitive-qualification-input-allocation.v1"
    ):
        raise ValueError("unsupported primitive qualification input allocation")
    if config.get("status") != "FROZEN_BEFORE_PRIMITIVE_QUALIFICATION_INPUT_CAPTURE":
        raise ValueError("qualification input allocation was not frozen before capture")
    claims = config.get("claim_contract", {})
    if claims != {
        "rollout_executed": False,
        "outcomes_loaded": False,
        "pre_outcome_only": True,
        "pi05_loaded_during_allocation": False,
        "simulator_privileged_semantic_ids_allowed_as_controller_input": False,
    }:
        raise ValueError("qualification allocation claim firewall differs")
    plan = load_primitive_qualification_plan(plan_path, repository_root=ROOT)
    for name in ("primitive", "context", "candidate_id"):
        if str(config.get(name)) != str(plan.get(name)):
            raise ValueError(f"qualification allocation differs from plan at {name}")
    design = plan.get("design")
    if not isinstance(design, Mapping):
        raise ValueError("qualification allocation requires a feasible frozen design")
    trials = int(design["trials"])

    snapshot = config.get("repository_seed_snapshot", {})
    commit = _git("rev-parse", f"{snapshot.get('commit')}^{{commit}}").strip()
    if commit != snapshot.get("commit"):
        raise ValueError("seed snapshot must use a full immutable commit hash")
    roots = snapshot.get("roots")
    if (
        not isinstance(roots, Sequence)
        or isinstance(roots, (str, bytes))
        or list(roots) != ["configs", "results", "runs"]
        or snapshot.get("extraction")
        != "explicit_seed_fields_and_seed_tokens_in_tracked_paths"
    ):
        raise ValueError("qualification seed snapshot scope differs")
    rule = config.get("allocation_rule", {})
    if (
        rule.get("method")
        != "contiguous_block_immediately_above_maximum_observed_seed"
        or rule.get("count_source") != "frozen_qualification_plan"
        or rule.get("outcome_access") != "prohibited"
        or rule.get("replacement_after_capture") != "prohibited"
        or rule.get("replacement_after_rollout") != "prohibited"
        or not " ".join(str(rule.get("rationale", "")).split())
    ):
        raise ValueError("qualification seed allocation rule differs")
    inventory_rows, tracked = _seed_inventory(commit, list(roots))
    if not inventory_rows:
        raise ValueError("repository seed inventory is unexpectedly empty")
    maximum = max(row["seed"] for row in inventory_rows)
    seed_start = maximum + 1
    seed_end = seed_start + trials - 1
    if seed_end >= 2**32:
        raise ValueError("qualification seed block exceeds NumPy's seed domain")
    inventory = {
        "schema_version": "piu.repository-seed-usage-inventory.v1",
        "status": "FROZEN_BEFORE_PRIMITIVE_QUALIFICATION_INPUT_CAPTURE",
        "snapshot_commit": commit,
        "scanned_roots": list(roots),
        "tracked_path_count": len(tracked),
        "extraction": snapshot["extraction"],
        "observed_seed_count": len(inventory_rows),
        "maximum_observed_seed": maximum,
        "observed": inventory_rows,
        "rollout_executed": False,
        "outcomes_loaded": False,
        "pre_outcome_only": True,
    }
    _write_new(inventory_path, json.dumps(inventory, indent=2, allow_nan=False) + "\n")

    identifiers = config.get("identifiers", {})
    prefix = " ".join(str(identifiers.get("group_prefix", "")).split())
    suffix = " ".join(str(identifiers.get("sample_suffix", "")).split())
    if not prefix or not suffix:
        raise ValueError("qualification allocation identifiers are missing")
    assignments = [
        {
            "initial_state_group": f"{prefix}_{index:03d}",
            "seed": seed_start + index,
            "split_role": "primitive_qualification",
        }
        for index in range(trials)
    ]
    split = {
        "schema_version": "piu.group-split-manifest.v1",
        "status": "FROZEN_BEFORE_COLLECTION",
        "allocation_method": "prospective_without_outcome_access",
        "scenario": "original_cluttered_drawer",
        "required_roles": ["primitive_qualification"],
        "purpose": "OPEN_executor_qualification_only",
        "allocation": {
            "rule": rule["method"],
            "rationale": " ".join(str(rule["rationale"]).split()),
            "seed_start": seed_start,
            "seed_end": seed_end,
            "count": trials,
            "repository_seed_inventory": reference(inventory_path),
            "frozen_plan": reference(plan_path),
            "replacement_after_capture": "prohibited",
            "replacement_after_rollout": "prohibited",
            "rollout_executed": False,
            "outcomes_loaded": False,
            "pre_outcome_only": True,
        },
        "assignments": assignments,
    }
    _write_new(split_path, json.dumps(split, indent=2, allow_nan=False) + "\n")
    load_split_manifest(split_path)

    public = config.get("public_candidate_context", {})
    entity = public.get("affordance_entity")
    if not isinstance(entity, Mapping):
        raise TypeError("qualification allocation lacks a public affordance entity")
    contexts = []
    for row in assignments:
        group = row["initial_state_group"]
        contexts.append(
            {
                "schema_version": "piu.public-candidate-context.v1",
                "sample_id": f"{group}::{suffix}",
                "initial_state_group": group,
                "split": "primitive_qualification",
                "split_role": "primitive_qualification",
                "public_inputs_only": True,
                "online_oracle_inputs": [],
                "task": {
                    "target_description": public["task_target_description"],
                    "destination_description": public["destination_description"],
                },
                "public_affordance_entities": [dict(entity)],
            }
        )
    _write_new(
        contexts_path,
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in contexts),
    )
    print(
        json.dumps(
            {
                "seed_inventory": reference(inventory_path),
                "split_manifest": reference(split_path),
                "candidate_contexts": reference(contexts_path),
                "groups": trials,
                "seed_start": seed_start,
                "seed_end": seed_end,
                "rollout_executed": False,
                "outcomes_loaded": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
