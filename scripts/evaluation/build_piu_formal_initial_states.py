#!/usr/bin/env python3
"""Freeze the exact opaque simulator states for a prospective formal cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.formal_states import validate_state_archive
from piu.reproducibility import validate_repro_lock
from piu.splits import load_split_manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--state",
        action="append",
        nargs=2,
        metavar=("INITIAL_STATE_GROUP", "NPZ_PATH"),
        required=True,
        help="repeat once for every sealed-test group",
    )
    parser.add_argument("--state-key", default="state")
    parser.add_argument(
        "--repro-manifest",
        type=Path,
        default=ROOT / "configs/experiments/piu_offline_repro_v4.yaml",
    )
    parser.add_argument(
        "--repro-lock",
        type=Path,
        default=ROOT / "results/diagnostics/piu_offline_repro_preflight_v4.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split_path = resolve(args.split_manifest)
    repro_manifest_path = resolve(args.repro_manifest)
    repro_lock_path = resolve(args.repro_lock)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("formal initial-state manifests are immutable")
    state_key = " ".join(args.state_key.split())
    if not state_key:
        raise ValueError("formal source state key must be nonempty")
    validate_repro_lock(
        repro_lock_path,
        manifest_path=repro_manifest_path,
        repository_root=ROOT,
    )
    split = load_split_manifest(split_path)
    sealed = {
        row["initial_state_group"]: int(row["seed"])
        for row in split["assignments"]
        if row["split_role"] == "sealed_test"
    }
    supplied: dict[str, Path] = {}
    for raw_group, raw_path in args.state:
        group = " ".join(raw_group.split())
        if not group or group in supplied:
            raise ValueError("formal source-state groups must be nonempty and unique")
        supplied[group] = resolve(Path(raw_path))
    if set(supplied) != set(sealed):
        raise ValueError("formal source states differ from the sealed-test cohort")
    rows = []
    digests: set[str] = set()
    for group in sorted(sealed):
        path = supplied[group]
        shape, dtype = validate_state_archive(path, state_key=state_key)
        digest = sha256(path)
        if digest in digests:
            raise ValueError("formal groups reuse an identical opaque source state")
        digests.add(digest)
        rows.append(
            {
                "initial_state_group": group,
                "simulator_seed": sealed[group],
                "state_key": state_key,
                "source_state": {
                    "path": portable(path),
                    "sha256": digest,
                    "shape": shape,
                    "dtype": dtype,
                },
            }
        )
    result = {
        "schema_version": "piu.formal-initial-state-manifest.v1",
        "status": "FROZEN_BEFORE_FORMAL_OUTCOME_COLLECTION",
        "claim_scope": "OPAQUE_TRANSPORT_ONLY_NOT_POLICY_INPUT",
        "outcomes_loaded": False,
        "split_manifest": {"path": portable(split_path), "sha256": sha256(split_path)},
        "offline_repro_lock": {
            "path": portable(repro_lock_path),
            "sha256": sha256(repro_lock_path),
            "manifest_sha256": sha256(repro_manifest_path),
        },
        "states": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
