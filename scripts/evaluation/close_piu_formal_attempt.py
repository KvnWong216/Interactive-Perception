#!/usr/bin/env python3
"""Close one ordered sealed attempt only after its bound episode exists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.formal_attempt import sha256, validate_attempt_ticket


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticket", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    args = parser.parse_args()
    ticket_path = resolve(args.ticket)
    episode_path = resolve(args.episode)
    ticket = json.loads(ticket_path.read_text())
    entry = ticket.get("entry", {})
    output_dir = Path(str(ticket.get("single_use_output_dir", "")))
    source_state = resolve(Path(str(entry.get("source_state", {}).get("path", ""))))
    validate_attempt_ticket(
        ticket_path,
        repository_root=ROOT,
        method_id=str(entry.get("method_id", "")),
        initial_state_group=str(entry.get("initial_state_group", "")),
        simulator_seed=int(entry["simulator_seed"]),
        source_state=source_state,
        output_dir=output_dir,
    )
    if episode_path.resolve() != Path(ticket["expected_episode_path"]).resolve():
        raise ValueError("formal close uses another episode path")
    episode = json.loads(episode_path.read_text())
    ticket_reference = episode.get("formal_attempt_ticket", {})
    if (
        episode.get("schema_version") != "piu.closed-loop-episode.v1"
        or episode.get("split") != "sealed_test"
        or episode.get("method_id") != entry["method_id"]
        or episode.get("initial_state_group") != entry["initial_state_group"]
        or episode.get("simulator_seed") != entry["simulator_seed"]
        or episode.get("source_state", {}).get("sha256")
        != entry["source_state"]["sha256"]
        or ticket_reference.get("sha256") != sha256(ticket_path)
    ):
        raise ValueError("formal episode differs from its attempt ticket")
    close_path = Path(str(ticket["expected_close_path"]))
    close = {
        "schema_version": "piu.formal-attempt-close.v1",
        "status": "CLOSED_WITH_EPISODE",
        "execution_index": ticket["execution_index"],
        "schedule_sha256": ticket["schedule"]["sha256"],
        "ticket_sha256": sha256(ticket_path),
        "episode_sha256": sha256(episode_path),
        "rollout_status": episode["rollout_status"],
    }
    with close_path.open("x") as handle:
        handle.write(json.dumps(close, indent=2) + "\n")
    print(json.dumps({"close": str(close_path.resolve()), **close}, indent=2))


if __name__ == "__main__":
    main()
