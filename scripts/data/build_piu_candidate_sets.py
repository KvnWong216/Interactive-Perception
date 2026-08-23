#!/usr/bin/env python3
"""Build dynamic PIU candidate supersets from public task affordances."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.candidate_generator import PublicAffordanceEntity, generate_candidates
from piu.contracts import assert_public_policy_value
from piu.splits import assignment_for, load_split_manifest, role_to_split


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
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contexts = resolve(args.contexts)
    split_manifest_path = resolve(args.split_manifest)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("public candidate-set artifacts are immutable")
    rows = [json.loads(line) for line in contexts.read_text().splitlines() if line]
    if not rows:
        raise ValueError("public candidate-context file is empty")
    result = []
    split_manifest = load_split_manifest(split_manifest_path)
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("schema_version") != "piu.public-candidate-context.v1":
            raise ValueError("unsupported public candidate-context schema")
        if (
            row.get("public_inputs_only") is not True
            or row.get("online_oracle_inputs") != []
        ):
            raise ValueError("candidate context must be public-input only")
        assert_public_policy_value(
            {
                "task": row.get("task", {}),
                "public_affordance_entities": row.get("public_affordance_entities", ()),
            },
            path="candidate_context",
        )
        if "holding_requested_target_set" in row:
            raise ValueError(
                "candidate context cannot inject holding belief; the calibrated "
                "post-observation binder gates the public PICK/PLACE superset"
            )
        sample_id = " ".join(str(row.get("sample_id", "")).split())
        group = " ".join(str(row.get("initial_state_group", "")).split())
        if not sample_id or not group or sample_id in seen:
            raise ValueError("candidate-context identities must be nonempty and unique")
        seen.add(sample_id)
        assignment = assignment_for(split_manifest, group)
        split = role_to_split(assignment["split_role"])
        if (
            row.get("split") != split.value
            or row.get("split_role") != assignment["split_role"]
        ):
            raise ValueError("candidate context differs from frozen split assignment")
        task = row.get("task", {})
        entities = [
            PublicAffordanceEntity.from_mapping(value)
            for value in row.get("public_affordance_entities", ())
        ]
        candidates = generate_candidates(
            task_target_description=str(task.get("target_description", "")),
            destination_description=str(task.get("destination_description", "")),
            entities=entities,
        )
        result.append(
            {
                "schema_version": "piu.public-candidate-set.v1",
                "sample_id": sample_id,
                "initial_state_group": group,
                "split": split.value,
                "public_inputs_only": True,
                "online_oracle_inputs": [],
                "candidates": candidates,
                "derivation": {
                    "source_path": portable(contexts),
                    "source_sha256": sha256(contexts),
                    "source_row_index": index,
                    "task_action_semantics": (
                        "pick_place_public_superset_gated_by_calibrated_binder"
                    ),
                    "split_manifest_path": portable(split_manifest_path),
                    "split_manifest_sha256": sha256(split_manifest_path),
                    "split_role": assignment["split_role"],
                },
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in result))
    print(json.dumps({"output": portable(output), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
