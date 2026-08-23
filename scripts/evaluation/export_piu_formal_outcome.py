#!/usr/bin/env python3
"""Project one authorized sealed episode into the immutable formal-row schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.statistics import load_formal_outcomes


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
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--sealed-authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episode_path = resolve(args.episode)
    authorization_path = resolve(args.sealed_authorization)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("formal outcome rows are immutable")
    episode = json.loads(episode_path.read_text())
    if episode.get("schema_version") != "piu.closed-loop-episode.v1":
        raise ValueError("unsupported closed-loop episode schema")
    if episode.get("split") != "sealed_test":
        raise ValueError("formal rows require a sealed-test episode")
    method = str(episode.get("method_id", ""))
    simulator_seed = episode.get("simulator_seed")
    if not isinstance(simulator_seed, int) or isinstance(simulator_seed, bool):
        raise TypeError("formal episode requires an integer simulator seed")
    oracle = method in {"B6", "B7"}
    if episode.get("evidence_class") != (
        "oracle_upper_bound" if oracle else "public_method"
    ):
        raise ValueError("episode method/evidence class differs")
    online_oracle = episode.get("online_oracle_inputs")
    if (oracle and not online_oracle) or (not oracle and online_oracle != []):
        raise ValueError("episode oracle-input declaration differs from method class")
    source = episode.get("source_state", {})
    history = episode.get("public_action_history", {})
    policy_identity = episode.get("policy_identity", {})
    source_path = resolve(Path(source["path"]))
    history_path = resolve(Path(history["path"]))
    identity_path = resolve(Path(policy_identity["path"]))
    if sha256(source_path) != source.get("sha256") or sha256(
        history_path
    ) != history.get("sha256"):
        raise ValueError("episode state/action provenance differs from content")
    if sha256(identity_path) != policy_identity.get("sha256"):
        raise ValueError("episode policy identity differs from content")
    authorization = json.loads(authorization_path.read_text())
    if authorization.get("schema_version") != "piu.formal-row-sealed-authorization.v1":
        raise ValueError("unsupported formal-row sealed authorization")
    expected = {
        "episode_sha256": sha256(episode_path),
        "source_state_sha256": sha256(source_path),
        "action_history_sha256": sha256(history_path),
        "policy_identity_sha256": sha256(identity_path),
        "method_id": method,
        "single_use_output": portable(output),
    }
    for name, value in expected.items():
        if authorization.get(name) != value:
            raise ValueError(f"formal-row authorization differs at {name}")
    row = {
        "schema_version": "piu.formal-outcome.v1",
        "initial_state_group": episode["initial_state_group"],
        "simulator_seed": simulator_seed,
        "method_id": method,
        "split": "sealed_test",
        "evidence_class": episode["evidence_class"],
        "rollout_status": episode["rollout_status"],
        "source_state_sha256": sha256(source_path),
        "action_history_sha256": sha256(history_path),
        "policy_identity_sha256": sha256(identity_path),
        "outcomes": episode["outcomes"],
        "episode": {"path": portable(episode_path), "sha256": sha256(episode_path)},
        "sealed_authorization": {
            "path": portable(authorization_path),
            "sha256": sha256(authorization_path),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, sort_keys=True) + "\n")
    load_formal_outcomes(output)
    print(json.dumps({"output": portable(output), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
