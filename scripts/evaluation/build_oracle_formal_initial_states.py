#!/usr/bin/env python3
"""Freeze exact pre-OPEN states for the disjoint formal oracle cohort."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.formal_states import validate_state_archive
from piu.oracle_formal import (
    INITIAL_STATE_SCHEMA,
    artifact,
    load_oracle_formal_initial_states,
    portable,
    sha256,
)
from piu.reproducibility import validate_repro_lock
from piu.splits import load_split_manifest


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def publish_validated(value: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pending: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".pending",
            delete=False,
        ) as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            pending = Path(handle.name)
        load_oracle_formal_initial_states(pending, repository_root=ROOT)
        os.link(pending, output)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--state",
        action="append",
        nargs=2,
        metavar=("INITIAL_STATE_GROUP", "NPZ_PATH"),
        required=True,
    )
    parser.add_argument("--state-key", default="state")
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split_path = resolve(args.split_manifest)
    repro_manifest_path = resolve(args.repro_manifest)
    repro_lock_path = resolve(args.repro_lock)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("oracle formal initial-state manifests are immutable")
    state_key = " ".join(args.state_key.split())
    if not state_key:
        raise ValueError("oracle formal state key must be nonempty")
    validate_repro_lock(
        repro_lock_path,
        manifest_path=repro_manifest_path,
        repository_root=ROOT,
    )
    split = load_split_manifest(split_path)
    assignments = {
        str(row["initial_state_group"]): int(row["seed"])
        for row in split["assignments"]
        if row["split_role"] == "oracle_formal"
    }
    if not assignments:
        raise ValueError("split manifest has no oracle-formal groups")
    supplied: dict[str, Path] = {}
    for raw_group, raw_path in args.state:
        group = " ".join(raw_group.split())
        if not group or group in supplied:
            raise ValueError("oracle formal source groups must be nonempty and unique")
        supplied[group] = resolve(Path(raw_path))
    if set(supplied) != set(assignments):
        raise ValueError("supplied states differ from the oracle-formal cohort")
    rows = []
    digests: set[str] = set()
    for group in sorted(assignments):
        path = supplied[group]
        shape, dtype = validate_state_archive(path, state_key=state_key)
        digest = sha256(path)
        if digest in digests:
            raise ValueError("oracle formal groups reuse an opaque source state")
        digests.add(digest)
        rows.append(
            {
                "initial_state_group": group,
                "simulator_seed": assignments[group],
                "state_key": state_key,
                "source_state": {
                    **artifact(path, repository_root=ROOT),
                    "shape": shape,
                    "dtype": dtype,
                },
            }
        )
    result = {
        "schema_version": INITIAL_STATE_SCHEMA,
        "status": "FROZEN_BEFORE_ORACLE_FORMAL_OUTCOMES",
        "claim_scope": "OPAQUE_TRANSPORT_ONLY_NOT_POLICY_INPUT",
        "outcomes_loaded": False,
        "split_manifest": artifact(split_path, repository_root=ROOT),
        "offline_repro_lock": {
            **artifact(repro_lock_path, repository_root=ROOT),
            "manifest_sha256": sha256(repro_manifest_path),
        },
        "states": rows,
    }
    publish_validated(result, output)
    print(json.dumps({"output": portable(output, repository_root=ROOT), **result}, indent=2))


if __name__ == "__main__":
    main()
