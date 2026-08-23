#!/usr/bin/env python3
"""Run the external-server-only oracle target-prompt qualification matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.oracle_schedule import load_schedule
from piu.policy_identity import load_checkpoint_identity, validate_server_metadata
from piu.compute_provenance import validate_external_pi05_endpoint_artifact

SCHEMAS = frozenset(
    {
        "calibrated-interaction.oracle-target-prompt-gate.v1",
        "calibrated-interaction.oracle-target-prompt-pilot.v2",
    }
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def validate_endpoint_check(
    path: Path,
    *,
    host: str,
    port: int,
    identity_path: Path,
    require_session: bool,
) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema_version") == "piu.external-pi05-check.v2":
        value = validate_external_pi05_endpoint_artifact(
            value,
            checkpoint_identity_path=identity_path,
            compute_contract_path=(
                ROOT / "configs/experiments/piu_empirical_compute_contract_v1.yaml"
            ),
            repository_root=ROOT,
        )
    if (
        value.get("schema_version")
        not in {"piu.external-pi05-check.v1", "piu.external-pi05-check.v2"}
        or value.get("status") != "PASS"
        or value.get("endpoint") != {"host": host, "port": port}
    ):
        raise ValueError("external pi0.5 endpoint check differs from this run")
    validate_server_metadata(
        dict(value.get("identity", {})), load_checkpoint_identity(identity_path)
    )
    if require_session and not isinstance(
        value.get("identity", {}).get("server_session_id"), str
    ):
        raise ValueError("v2 oracle endpoint check lacks a server session ID")
    probe = value.get("action_probe")
    if require_session and not isinstance(probe, dict):
        raise ValueError("v2 oracle endpoint check lacks its finite action probe")
    if probe is not None and (
        not isinstance(probe, dict) or probe.get("finite") is not True
    ):
        raise ValueError("external pi0.5 action probe is invalid")
    return value


def validate_existing_report(
    path: Path,
    *,
    config: dict[str, Any],
    seed: int,
    style: str,
) -> None:
    validator_path = (
        ROOT / "scripts/evaluation/summarize_oracle_target_prompt_gate.py"
    )
    specification = importlib.util.spec_from_file_location(
        "piu_oracle_prompt_report_validator", validator_path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load the oracle report validator")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.validate_oracle_report(path, config=config, seed=seed, style=style)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if config.get("schema_version") not in SCHEMAS:
        raise ValueError(f"unsupported oracle gate schema in {path}")
    contract = config.get("resource_contract", {})
    if contract.get("policy_server") != "external_only":
        raise ValueError("oracle gate policy server must be external_only")
    if contract.get("local_policy_server_allowed") is not False:
        raise ValueError("oracle gate must explicitly prohibit a local policy server")
    return config


def run_specs(
    config: dict[str, Any], *, phase: str, selected_style: str | None
) -> list[tuple[str, int]]:
    if phase == "screen":
        return [
            (str(style), int(seed))
            for style in config["screen"]["styles"]
            for seed in config["screen"]["seeds"]
        ]
    if selected_style is None:
        raise ValueError("--style is required for the confirmation phase")
    allowed = set(config["screen"]["styles"])
    if selected_style not in allowed:
        raise ValueError(f"style must be one of {sorted(allowed)}")
    return [(selected_style, int(seed)) for seed in config["confirmation"]["seeds"]]


def command_for(
    *,
    config: dict[str, Any],
    phase: str,
    style: str,
    seed: int,
    host: str,
    port: int,
    server_timeout: float,
) -> tuple[list[str], Path]:
    execution = config["execution"]
    run_directory = ROOT / config["run_root"] / phase / style / f"seed{seed}"
    initial_state = (
        ROOT / config["source_run_root"] / f"seed{seed}" / "open_butter/final_state.npz"
    )
    if not initial_state.is_file():
        raise FileNotFoundError(initial_state)
    command = [
        sys.executable,
        str(ROOT / "scripts/pipeline/execute.py"),
        "--scenario-config",
        str(ROOT / config["scenario_config"]),
        "--role",
        str(execution["role"]),
        "--prompt",
        str(execution["prompt"]),
        "--seed",
        str(seed),
        "--initial-state",
        str(initial_state),
        "--state-key",
        "state",
        "--steps",
        str(execution["steps"]),
        "--replan-steps",
        str(execution["replan_steps"]),
        "--report-schema",
        (
            "v2"
            if config["schema_version"]
            == "calibrated-interaction.oracle-target-prompt-pilot.v2"
            else "v1"
        ),
        "--target-object",
        str(execution["target_object"]),
        "--external-server",
        "--expected-policy-identity",
        str(ROOT / config["resource_contract"]["checkpoint_identity"]),
        "--host",
        host,
        "--port",
        str(port),
        "--server-timeout",
        str(server_timeout),
        "--oracle-target-visual-prompt",
        style,
        "--oracle-minimum-visible-pixels",
        str(
            execution.get(
                "target_presence_minimum_pixels",
                execution.get("target_visible_pixels_minimum"),
            )
        ),
        "--assets",
        str(run_directory / "assets"),
        "--work",
        str(run_directory / "work"),
        "--output",
        str(run_directory / "report.json"),
    ]
    return command, run_directory / "report.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/experiments/original_drawer_oracle_target_prompt_pilot_v2.yaml",
    )
    parser.add_argument("--phase", choices=("screen", "confirmation"), required=True)
    parser.add_argument("--style", choices=("box", "point", "spotlight"))
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--server-timeout", type=float, default=30.0)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--endpoint-check", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    config_path = resolve(args.config)
    config = load_config(config_path)
    schedule_path = None if args.schedule is None else resolve(args.schedule)
    endpoint_check_path = (
        None if args.endpoint_check is None else resolve(args.endpoint_check)
    )
    schedule = None
    scheduled_rows = None
    if (
        config["schema_version"]
        == "calibrated-interaction.oracle-target-prompt-pilot.v2"
    ):
        if schedule_path is None:
            raise ValueError("v2 oracle execution requires a frozen phase schedule")
        schedule = load_schedule(
            schedule_path,
            repository_root=ROOT,
            config_path=config_path,
        )
        if schedule.get("phase") != args.phase:
            raise ValueError("oracle schedule phase differs from the requested phase")
        if schedule.get("selected_style") != args.style:
            raise ValueError("oracle schedule style differs from the requested style")
        scheduled_rows = list(schedule["entries"])
        specs = [(str(row["style"]), int(row["seed"])) for row in scheduled_rows]
        observed_gap = False
        for row in scheduled_rows:
            exists = resolve(Path(str(row["expected_report"]))).is_file()
            if exists and observed_gap:
                raise ValueError(
                    "oracle reports exist outside the frozen execution prefix"
                )
            if not exists:
                observed_gap = True
    else:
        specs = run_specs(config, phase=args.phase, selected_style=args.style)
    if not args.dry_run:
        identity = ROOT / config["resource_contract"]["checkpoint_identity"]
        if endpoint_check_path is None:
            raise ValueError(
                "live oracle execution requires an endpoint check artifact"
            )
        endpoint_check = validate_endpoint_check(
            endpoint_check_path,
            host=args.host,
            port=args.port,
            identity_path=identity,
            require_session=(
                config["schema_version"]
                == "calibrated-interaction.oracle-target-prompt-pilot.v2"
            ),
        )
        live_check = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/infra/check_external_pi05.py"),
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--timeout",
                str(args.server_timeout),
                "--identity",
                str(identity),
                "--deployment-mode",
                str(
                    endpoint_check.get("compute_provenance", {}).get(
                        "deployment_mode", "remote_identified_server"
                    )
                ),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        live_identity = json.loads(live_check.stdout)["identity"]
        expected_session = endpoint_check["identity"].get("server_session_id")
        if (
            expected_session is not None
            and live_identity.get("server_session_id") != expected_session
        ):
            raise ValueError(
                "external pi0.5 server session differs from the endpoint check"
            )
    emitted = []
    for execution_index, (style, seed) in enumerate(specs):
        command, report = command_for(
            config=config,
            phase=args.phase,
            style=style,
            seed=seed,
            host=args.host,
            port=args.port,
            server_timeout=args.server_timeout,
        )
        if scheduled_rows is not None and report.resolve() != resolve(
            Path(scheduled_rows[execution_index]["expected_report"])
        ).resolve():
            raise ValueError("oracle runner report path differs from its schedule")
        if report.exists():
            if args.skip_existing:
                validate_existing_report(
                    report,
                    config=config,
                    seed=seed,
                    style=style,
                )
                emitted.append(
                    {
                        "execution_index": execution_index,
                        "seed": seed,
                        "style": style,
                        "status": "VALIDATED_EXISTING",
                        "report_sha256": sha256(report),
                    }
                )
                continue
            raise FileExistsError(f"immutable report already exists: {report}")
        if args.dry_run:
            emitted.append(
                {
                    "execution_index": execution_index,
                    "seed": seed,
                    "style": style,
                    "status": "DRY_RUN",
                    "command": shlex.join(command),
                    "schedule_sha256": (
                        sha256(schedule_path) if schedule_path is not None else None
                    ),
                }
            )
            continue
        subprocess.run(command, cwd=ROOT, check=True)
        emitted.append(
            {
                "execution_index": execution_index,
                "seed": seed,
                "style": style,
                "status": "COMPLETE",
                "report": str(report.relative_to(ROOT)),
                "report_sha256": sha256(report),
            }
        )
    print(json.dumps(emitted, indent=2))


if __name__ == "__main__":
    main()
