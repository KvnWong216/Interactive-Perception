#!/usr/bin/env python3
"""Run the external-server-only oracle target-prompt qualification matrix."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = frozenset(
    {
        "calibrated-interaction.oracle-target-prompt-gate.v1",
        "calibrated-interaction.oracle-target-prompt-pilot.v2",
    }
)


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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    specs = run_specs(config, phase=args.phase, selected_style=args.style)
    if not args.dry_run:
        identity = ROOT / config["resource_contract"]["checkpoint_identity"]
        subprocess.run(
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
            ],
            cwd=ROOT,
            check=True,
        )
    emitted = []
    for style, seed in specs:
        command, report = command_for(
            config=config,
            phase=args.phase,
            style=style,
            seed=seed,
            host=args.host,
            port=args.port,
            server_timeout=args.server_timeout,
        )
        if report.exists():
            if args.skip_existing:
                emitted.append({"seed": seed, "style": style, "status": "SKIPPED"})
                continue
            raise FileExistsError(f"immutable report already exists: {report}")
        if args.dry_run:
            emitted.append(
                {
                    "seed": seed,
                    "style": style,
                    "status": "DRY_RUN",
                    "command": shlex.join(command),
                }
            )
            continue
        subprocess.run(command, cwd=ROOT, check=True)
        emitted.append(
            {
                "seed": seed,
                "style": style,
                "status": "COMPLETE",
                "report": str(report.relative_to(ROOT)),
            }
        )
    print(json.dumps(emitted, indent=2))


if __name__ == "__main__":
    main()
