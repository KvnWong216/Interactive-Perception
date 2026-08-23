"""Pre-outcome scheduling contract for the oracle target-prompt phases."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .reproducibility import validate_repro_lock

CONFIG_SCHEMA = "calibrated-interaction.oracle-target-prompt-pilot.v2"
PROTOCOL_SCHEMA = "piu.oracle-target-prompt-schedule-protocol.v1"
SCHEDULE_SCHEMA = "piu.oracle-target-prompt-execution-schedule.v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path, *, repository_root: Path) -> Path:
    return path if path.is_absolute() else repository_root / path


def portable(path: Path, *, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root))
    except ValueError:
        return str(path.resolve())


def artifact(path: Path, *, repository_root: Path) -> dict[str, str]:
    return {
        "path": portable(path, repository_root=repository_root),
        "sha256": sha256(path),
    }


def validate_screen_result(
    path: Path,
    *,
    repository_root: Path,
    experiment_path: Path,
) -> dict[str, Any]:
    validator_path = (
        repository_root
        / "scripts/evaluation/summarize_oracle_target_prompt_gate.py"
    )
    specification = importlib.util.spec_from_file_location(
        "piu_oracle_prompt_screen_validator", validator_path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the oracle screen validator")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    observed = json.loads(path.read_text())
    expected = module.summarize(experiment_path, "screen", None)
    if observed != expected:
        raise ValueError("oracle screen result differs from validated phase reports")
    if (
        observed.get("status") != "SCREEN_COMPLETE_AWAITING_CONFIRMATION"
        or not isinstance(observed.get("screen", {}).get("selected_style"), str)
    ):
        raise ValueError("oracle screen did not select one unique style")
    return observed


def load_experiment(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, Mapping) or value.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported oracle target-prompt experiment")
    return dict(value)


def load_protocol(
    path: Path,
    *,
    repository_root: Path,
    experiment_path: Path,
) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != PROTOCOL_SCHEMA
        or value.get("method")
        != "sha256_keyed_outcome_independent_permutation"
        or value.get("within_phase_order_randomized") is not True
        or value.get("outcomes_loaded") is not False
        or not " ".join(str(value.get("namespace", "")).split())
    ):
        raise ValueError("oracle target-prompt execution schedule is not frozen")
    target = resolve(
        Path(str(value.get("target_protocol", ""))),
        repository_root=repository_root,
    )
    if target.resolve() != experiment_path.resolve():
        raise ValueError("oracle schedule protocol targets another experiment")
    return dict(value)


def phase_specifications(
    protocol: Mapping[str, Any], *, phase: str, selected_style: str | None
) -> list[tuple[str, int]]:
    if phase == "screen":
        if selected_style is not None:
            raise ValueError("screen scheduling cannot preselect a style")
        return [
            (str(style), int(seed))
            for style in protocol["screen"]["styles"]
            for seed in protocol["screen"]["seeds"]
        ]
    if phase != "confirmation":
        raise ValueError("oracle target-prompt phase is unsupported")
    allowed = {str(style) for style in protocol["screen"]["styles"]}
    if selected_style not in allowed:
        raise ValueError("confirmation requires the uniquely selected screen style")
    return [
        (str(selected_style), int(seed))
        for seed in protocol["confirmation"]["seeds"]
    ]


def load_schedule(
    path: Path,
    *,
    repository_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != SCHEDULE_SCHEMA
        or value.get("status") != "FROZEN_BEFORE_PHASE_OUTCOMES"
        or value.get("outcomes_loaded") is not False
    ):
        raise ValueError("unsupported oracle target-prompt execution schedule")
    experiment = load_experiment(config_path)
    config_value = value.get("config", {})
    if (
        resolve(
            Path(str(config_value.get("path", ""))),
            repository_root=repository_root,
        )
        .resolve()
        != config_path.resolve()
        or config_value.get("sha256") != sha256(config_path)
    ):
        raise ValueError("oracle schedule uses another protocol config")
    lock_value = value.get("offline_repro_lock", {})
    lock_path = resolve(
        Path(str(lock_value.get("path", ""))), repository_root=repository_root
    )
    manifest_path = repository_root / "configs/experiments/piu_offline_repro_v1.yaml"
    if (
        not lock_path.is_file()
        or lock_value.get("sha256") != sha256(lock_path)
        or lock_value.get("manifest_sha256") != sha256(manifest_path)
    ):
        raise ValueError("oracle schedule offline release lock differs")
    validate_repro_lock(
        lock_path,
        manifest_path=manifest_path,
        repository_root=repository_root,
    )
    protocol_value = value.get("schedule_protocol", {})
    protocol_path = resolve(
        Path(str(protocol_value.get("path", ""))), repository_root=repository_root
    )
    if not protocol_path.is_file() or protocol_value.get("sha256") != sha256(
        protocol_path
    ):
        raise ValueError("oracle execution schedule protocol differs")
    protocol = load_protocol(
        protocol_path,
        repository_root=repository_root,
        experiment_path=config_path,
    )
    identity_path = resolve(
        Path(experiment["resource_contract"]["checkpoint_identity"]),
        repository_root=repository_root,
    )
    identity_value = value.get("policy_identity", {})
    if (
        resolve(
            Path(str(identity_value.get("path", ""))),
            repository_root=repository_root,
        ).resolve()
        != identity_path.resolve()
        or identity_value.get("sha256") != sha256(identity_path)
    ):
        raise ValueError("oracle schedule policy identity differs")
    phase = str(value.get("phase", ""))
    selected_style = value.get("selected_style")
    if selected_style is not None:
        selected_style = str(selected_style)
    screen_digest = ""
    if phase == "confirmation":
        screen_value = value.get("screen_result", {})
        screen_path = resolve(
            Path(str(screen_value.get("path", ""))),
            repository_root=repository_root,
        )
        if not screen_path.is_file() or screen_value.get("sha256") != sha256(
            screen_path
        ):
            raise ValueError("confirmation schedule screen result differs")
        screen = validate_screen_result(
            screen_path,
            repository_root=repository_root,
            experiment_path=config_path,
        )
        if (
            screen.get("status") != "SCREEN_COMPLETE_AWAITING_CONFIRMATION"
            or screen.get("screen", {}).get("selected_style") != selected_style
            or screen.get("experiment", {}).get("sha256") != sha256(config_path)
        ):
            raise ValueError("confirmation schedule lacks a valid screen selection")
        screen_digest = sha256(screen_path)
    elif value.get("screen_result") is not None:
        raise ValueError("screen schedule cannot load a prior screen outcome")
    expected_specs_orderless = phase_specifications(
        experiment, phase=phase, selected_style=selected_style
    )
    expected_specs = set(expected_specs_orderless)
    binding = "\0".join(
        (
            sha256(config_path),
            sha256(lock_path),
            phase,
            selected_style or "",
            screen_digest,
        )
    )
    namespace = str(protocol["namespace"])
    expected_order = sorted(
        expected_specs_orderless,
        key=lambda item: hashlib.sha256(
            f"{namespace}\0{binding}\0{item[0]}\0{item[1]}".encode()
        ).hexdigest(),
    )
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) != len(expected_specs):
        raise ValueError("oracle schedule has an incomplete phase matrix")
    observed_specs: set[tuple[str, int]] = set()
    for index, row in enumerate(entries):
        if not isinstance(row, Mapping) or row.get("execution_index") != index:
            raise ValueError("oracle schedule indices are not contiguous")
        style = str(row.get("style", ""))
        seed = row.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("oracle schedule seed must be an integer")
        spec = (style, seed)
        if spec not in expected_specs or spec in observed_specs:
            raise ValueError("oracle schedule contains an unexpected phase cell")
        observed_specs.add(spec)
        source_value = row.get("source_state", {})
        source_path = resolve(
            Path(str(source_value.get("path", ""))),
            repository_root=repository_root,
        )
        expected_source = (
            repository_root
            / experiment["source_run_root"]
            / f"seed{seed}"
            / "open_butter/final_state.npz"
        )
        if (
            source_path.resolve() != expected_source.resolve()
            or not source_path.is_file()
            or source_value.get("sha256") != sha256(source_path)
        ):
            raise ValueError("oracle schedule source state differs")
        expected_report = (
            repository_root
            / experiment["run_root"]
            / phase
            / style
            / f"seed{seed}/report.json"
        )
        if (
            resolve(
                Path(str(row.get("expected_report", ""))),
                repository_root=repository_root,
            ).resolve()
            != expected_report.resolve()
        ):
            raise ValueError("oracle schedule expects another report path")
    if observed_specs != expected_specs:
        raise ValueError("oracle schedule phase cells differ from the protocol")
    randomization = value.get("randomization", {})
    if (
        randomization.get("method") != protocol["method"]
        or randomization.get("namespace") != namespace
        or randomization.get("binding_sha256")
        != hashlib.sha256(binding.encode()).hexdigest()
        or randomization.get("within_phase_order_randomized") is not True
    ):
        raise ValueError("oracle schedule randomization contract differs")
    observed_order = [(str(row["style"]), int(row["seed"])) for row in entries]
    if observed_order != expected_order:
        raise ValueError("oracle schedule order differs from its hash permutation")
    return value
