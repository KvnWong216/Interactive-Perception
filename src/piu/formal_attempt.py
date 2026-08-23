"""Runtime enforcement for one-shot, ordered sealed execution attempts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .reproducibility import validate_repro_lock


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_artifact_path(value: str, *, repository_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def artifact(path: Path, *, repository_root: Path) -> dict[str, str]:
    try:
        portable = str(path.resolve().relative_to(repository_root))
    except ValueError:
        portable = str(path.resolve())
    return {"path": portable, "sha256": sha256(path)}


def load_formal_schedule(
    path: Path, *, repository_root: Path
) -> dict[str, Any]:
    """Load a schedule and revalidate its code, policy, and source artifacts."""

    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != "piu.formal-execution-schedule.v1"
        or value.get("status") != "FROZEN_BEFORE_FORMAL_OUTCOME_COLLECTION"
        or value.get("outcomes_loaded") is not False
    ):
        raise ValueError("unsupported formal execution schedule")
    inputs = value.get("inputs")
    if not isinstance(inputs, Mapping):
        raise TypeError("formal schedule lacks input provenance")
    lock_value = inputs.get("offline_repro_lock")
    if not isinstance(lock_value, Mapping):
        raise TypeError("formal schedule lacks the offline release lock")
    lock_path = resolve_artifact_path(
        str(lock_value.get("path", "")), repository_root=repository_root
    )
    if not lock_path.is_file() or sha256(lock_path) != lock_value.get("sha256"):
        raise ValueError("formal schedule offline release lock differs")
    manifest_path = repository_root / "configs/experiments/piu_offline_repro_v3.yaml"
    if lock_value.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("formal schedule reproduction manifest differs")
    validate_repro_lock(
        lock_path,
        manifest_path=manifest_path,
        repository_root=repository_root,
    )
    identity_value = inputs.get("policy_identity")
    if not isinstance(identity_value, Mapping):
        raise TypeError("formal schedule lacks policy identity")
    identity_path = resolve_artifact_path(
        str(identity_value.get("path", "")), repository_root=repository_root
    )
    if not identity_path.is_file() or sha256(identity_path) != identity_value.get(
        "sha256"
    ):
        raise ValueError("formal schedule policy identity differs")
    frozen_paths: dict[str, Path] = {}
    for name in ("baseline_registry", "scenario_config"):
        artifact_value = inputs.get(name)
        if not isinstance(artifact_value, Mapping):
            raise TypeError(f"formal schedule lacks {name} provenance")
        artifact_path = resolve_artifact_path(
            str(artifact_value.get("path", "")), repository_root=repository_root
        )
        if not artifact_path.is_file() or sha256(artifact_path) != artifact_value.get(
            "sha256"
        ):
            raise ValueError(f"formal schedule {name} differs")
        frozen_paths[name] = artifact_path
    registry = yaml.safe_load(frozen_paths["baseline_registry"].read_text())
    scenario = yaml.safe_load(frozen_paths["scenario_config"].read_text())
    if registry.get("schema_version") != "piu.baseline-registry.v1":
        raise ValueError("formal schedule baseline registry is unsupported")
    if scenario.get("schema_version") != "piu.scenario.v1":
        raise ValueError("formal schedule scenario config is unsupported")
    registered_bddl = resolve_artifact_path(
        str(registry.get("scenario", "")), repository_root=repository_root
    ).resolve()
    configured_bddl = resolve_artifact_path(
        str(scenario.get("scene", {}).get("bddl", "")),
        repository_root=repository_root,
    ).resolve()
    if registered_bddl != configured_bddl:
        raise ValueError("formal schedule scenario differs from baseline registry")
    maximum_decisions = registry.get("shared_contract", {}).get(
        "maximum_controller_decisions"
    )
    if (
        not isinstance(maximum_decisions, int)
        or isinstance(maximum_decisions, bool)
        or maximum_decisions <= 0
    ):
        raise ValueError("formal schedule has an invalid shared decision cap")
    execution_contract = value.get("shared_execution_contract")
    if not isinstance(execution_contract, Mapping) or execution_contract != {
        "maximum_controller_decisions": maximum_decisions,
        "interpretation": "resource_cap_not_learned_decision_threshold",
    }:
        raise ValueError("formal schedule differs from the shared execution contract")
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("formal schedule has no execution entries")
    keys: set[tuple[str, str]] = set()
    for index, row in enumerate(entries):
        if not isinstance(row, Mapping) or row.get("execution_index") != index:
            raise ValueError("formal schedule execution indices are not contiguous")
        key = (
            " ".join(str(row.get("initial_state_group", "")).split()),
            str(row.get("method_id", "")),
        )
        if not all(key) or key in keys:
            raise ValueError("formal schedule group/method entries are invalid")
        keys.add(key)
        seed = row.get("simulator_seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("formal schedule simulator seed must be an integer")
        source = row.get("source_state")
        if not isinstance(source, Mapping):
            raise TypeError("formal schedule entry lacks a source state")
        source_path = resolve_artifact_path(
            str(source.get("path", "")), repository_root=repository_root
        )
        if not source_path.is_file() or sha256(source_path) != source.get("sha256"):
            raise ValueError(f"formal schedule source state differs at index {index}")
    return value


def validate_attempt_ticket(
    ticket_path: Path,
    *,
    repository_root: Path,
    method_id: str,
    initial_state_group: str,
    simulator_seed: int,
    source_state: Path,
    output_dir: Path,
    baseline_registry: Path | None = None,
    scenario_config: Path | None = None,
    allow_closed: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one ticket against its frozen schedule entry and run target."""

    ticket = json.loads(ticket_path.read_text())
    if (
        ticket.get("schema_version") != "piu.formal-attempt-ticket.v1"
        or ticket.get("status") != "STARTED"
        or ticket.get("split") != "sealed_test"
        or ticket.get("outcomes_loaded") is not False
    ):
        raise ValueError("unsupported formal attempt ticket")
    schedule_value = ticket.get("schedule")
    if not isinstance(schedule_value, Mapping):
        raise TypeError("formal attempt ticket lacks schedule provenance")
    schedule_path = resolve_artifact_path(
        str(schedule_value.get("path", "")), repository_root=repository_root
    )
    if not schedule_path.is_file() or sha256(schedule_path) != schedule_value.get(
        "sha256"
    ):
        raise ValueError("formal attempt schedule differs from ticket")
    schedule = load_formal_schedule(schedule_path, repository_root=repository_root)
    for name, provided in (
        ("baseline_registry", baseline_registry),
        ("scenario_config", scenario_config),
    ):
        if provided is not None and sha256(provided) != schedule["inputs"][name].get(
            "sha256"
        ):
            raise ValueError(f"formal attempt uses another {name}")
    index = ticket.get("execution_index")
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError("formal attempt execution index must be an integer")
    entries = schedule["entries"]
    if not 0 <= index < len(entries) or ticket.get("entry") != entries[index]:
        raise ValueError("formal attempt ticket differs from scheduled entry")
    entry = entries[index]
    expected = {
        "method_id": method_id,
        "initial_state_group": " ".join(initial_state_group.split()),
        "simulator_seed": simulator_seed,
    }
    for name, required in expected.items():
        if entry.get(name) != required:
            raise ValueError(f"formal attempt differs from schedule at {name}")
    if sha256(source_state) != entry.get("source_state", {}).get("sha256"):
        raise ValueError("formal attempt uses another opaque source state")
    expected_output_dir = str(output_dir.resolve())
    if ticket.get("single_use_output_dir") != expected_output_dir:
        raise ValueError("formal attempt uses another single-use output directory")
    if ticket.get("expected_episode_path") != str(
        (output_dir / "episode.json").resolve()
    ):
        raise ValueError("formal attempt ticket expects another episode path")
    ledger_dir = Path(str(ticket.get("ledger_dir", ""))).resolve()
    expected_ticket = ledger_dir / f"{index:05d}.started.json"
    if ticket_path.resolve() != expected_ticket.resolve():
        raise ValueError("formal attempt ticket is outside its ordered ledger")
    expected_close = ledger_dir / f"{index:05d}.closed.json"
    if ticket.get("expected_close_path") != str(expected_close.resolve()):
        raise ValueError("formal attempt ticket expects another close receipt")
    if expected_close.exists() and not allow_closed:
        raise ValueError("formal attempt ticket is already closed and cannot rerun")
    previous_close_sha256 = None
    schedule_digest = sha256(schedule_path)
    for prior_index in range(index):
        prior_ticket = ledger_dir / f"{prior_index:05d}.started.json"
        prior_close = ledger_dir / f"{prior_index:05d}.closed.json"
        if not prior_ticket.is_file() or not prior_close.is_file():
            raise ValueError("formal attempt ledger has an incomplete prior entry")
        prior_ticket_value = json.loads(prior_ticket.read_text())
        prior_close_value = json.loads(prior_close.read_text())
        prior_episode = Path(
            str(prior_ticket_value.get("expected_episode_path", ""))
        )
        if (
            prior_ticket_value.get("previous_close_sha256")
            != previous_close_sha256
            or prior_close_value.get("schema_version")
            != "piu.formal-attempt-close.v1"
            or prior_close_value.get("status") != "CLOSED_WITH_EPISODE"
            or prior_close_value.get("execution_index") != prior_index
            or prior_close_value.get("schedule_sha256") != schedule_digest
            or prior_close_value.get("ticket_sha256") != sha256(prior_ticket)
            or not prior_episode.is_file()
            or prior_close_value.get("episode_sha256") != sha256(prior_episode)
        ):
            raise ValueError("formal attempt ledger has an invalid prior close chain")
        previous_close_sha256 = sha256(prior_close)
    if ticket.get("previous_close_sha256") != previous_close_sha256:
        raise ValueError("formal attempt ticket is outside the prior close chain")
    for ledger_path in ledger_dir.glob("*.json"):
        try:
            ledger_index = int(ledger_path.name.split(".", 1)[0])
        except ValueError as exc:
            raise ValueError(
                f"unexpected file in formal attempt ledger: {ledger_path}"
            ) from exc
        if ledger_index > index and not allow_closed:
            raise ValueError("formal attempt ledger contains a later entry")
    return ticket, schedule


def validate_attempt_close(
    close_path: Path,
    *,
    ticket_path: Path,
    episode_path: Path,
) -> dict[str, Any]:
    value = json.loads(close_path.read_text())
    if value.get("schema_version") != "piu.formal-attempt-close.v1":
        raise ValueError("unsupported formal attempt close receipt")
    if value.get("status") != "CLOSED_WITH_EPISODE":
        raise ValueError("formal attempt is not closed with an episode")
    if value.get("ticket_sha256") != sha256(ticket_path):
        raise ValueError("formal close receipt identifies another ticket")
    if value.get("episode_sha256") != sha256(episode_path):
        raise ValueError("formal close receipt identifies another episode")
    return value
