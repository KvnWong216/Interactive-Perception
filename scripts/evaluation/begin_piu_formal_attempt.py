#!/usr/bin/env python3
"""Issue only the next immutable ticket in a sealed execution schedule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.formal_attempt import artifact, load_formal_schedule, sha256


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def validate_previous_close(
    *,
    ticket: Path,
    close: Path,
    schedule_sha256: str,
    index: int,
    expected_previous_close_sha256: str | None,
) -> None:
    ticket_value = json.loads(ticket.read_text())
    close_value = json.loads(close.read_text())
    episode_path = Path(str(ticket_value.get("expected_episode_path", "")))
    if (
        close_value.get("schema_version") != "piu.formal-attempt-close.v1"
        or close_value.get("status") != "CLOSED_WITH_EPISODE"
        or close_value.get("execution_index") != index
        or close_value.get("schedule_sha256") != schedule_sha256
        or close_value.get("ticket_sha256") != sha256(ticket)
        or ticket_value.get("previous_close_sha256")
        != expected_previous_close_sha256
        or not episode_path.is_file()
        or close_value.get("episode_sha256") != sha256(episode_path)
    ):
        raise ValueError(f"formal attempt {index} has an invalid close receipt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--run-output-dir", type=Path, required=True)
    args = parser.parse_args()
    schedule_path = resolve(args.schedule)
    ledger_dir = resolve(args.ledger_dir)
    run_output_dir = resolve(args.run_output_dir)
    schedule = load_formal_schedule(schedule_path, repository_root=ROOT)
    schedule_digest = sha256(schedule_path)
    if run_output_dir.exists():
        raise FileExistsError("formal run output directory already exists")
    ledger_dir.mkdir(parents=True, exist_ok=True)
    next_index = None
    previous_close_sha256 = None
    for index in range(len(schedule["entries"])):
        ticket_path = ledger_dir / f"{index:05d}.started.json"
        close_path = ledger_dir / f"{index:05d}.closed.json"
        if not ticket_path.exists():
            if close_path.exists():
                raise ValueError("formal attempt ledger has a close without a ticket")
            next_index = index
            break
        if not close_path.exists():
            raise ValueError(f"formal attempt {index} is still open")
        validate_previous_close(
            ticket=ticket_path,
            close=close_path,
            schedule_sha256=schedule_digest,
            index=index,
            expected_previous_close_sha256=previous_close_sha256,
        )
        previous_close_sha256 = sha256(close_path)
    if next_index is None:
        raise ValueError("formal execution schedule is already complete")
    for path in ledger_dir.glob("*.json"):
        try:
            observed_index = int(path.name.split(".", 1)[0])
        except ValueError as exc:
            raise ValueError(
                f"unexpected file in formal attempt ledger: {path}"
            ) from exc
        if observed_index > next_index:
            raise ValueError("formal attempt ledger contains entries after a gap")
    ticket_path = ledger_dir / f"{next_index:05d}.started.json"
    close_path = ledger_dir / f"{next_index:05d}.closed.json"
    ticket = {
        "schema_version": "piu.formal-attempt-ticket.v1",
        "status": "STARTED",
        "claim_scope": "SEALED_SINGLE_ATTEMPT_NO_OUTCOME_LOADED",
        "outcomes_loaded": False,
        "split": "sealed_test",
        "execution_index": next_index,
        "entry": schedule["entries"][next_index],
        "schedule": artifact(schedule_path, repository_root=ROOT),
        "ledger_dir": str(ledger_dir.resolve()),
        "single_use_output_dir": str(run_output_dir.resolve()),
        "expected_episode_path": str((run_output_dir / "episode.json").resolve()),
        "expected_close_path": str(close_path.resolve()),
        "previous_close_sha256": previous_close_sha256,
    }
    with ticket_path.open("x") as handle:
        handle.write(json.dumps(ticket, indent=2) + "\n")
    print(
        json.dumps(
            {"ticket": artifact(ticket_path, repository_root=ROOT), **ticket},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
