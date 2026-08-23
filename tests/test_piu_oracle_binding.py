from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORACLE_CONFIG = (
    ROOT / "configs/experiments/original_drawer_oracle_target_prompt_pilot_v2.yaml"
)


def _selection(path: Path, *, style: str | None = "point") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "calibrated-interaction.oracle-target-prompt-result.v1"
                ),
                "experiment": {
                    "sha256": hashlib.sha256(ORACLE_CONFIG.read_bytes()).hexdigest()
                },
                "status": "SCREEN_COMPLETE_AWAITING_CONFIRMATION",
                "screen": {"selected_style": style},
                "online_oracle_input_count": 2,
                "formal_method_claim": False,
            }
        )
    )


def _command(tmp_path: Path, selection: Path) -> list[str]:
    state = tmp_path / "state.npz"
    state.write_bytes(b"opaque paired simulator state")
    return [
        sys.executable,
        str(ROOT / "scripts/pipeline/run_piu_oracle_binding_full_loop.py"),
        "--scenario-config",
        str(ROOT / "configs/scenarios/original_drawer.yaml"),
        "--style-selection",
        str(selection),
        "--initial-state",
        str(state),
        "--initial-state-group",
        "group",
        "--split",
        "sealed_test",
        "--seed",
        "5",
        "--host",
        "pi05.example.internal",
        "--output-dir",
        str(tmp_path / "b7"),
        "--dry-run",
    ]


def test_b7_full_loop_dry_run_uses_selected_style_and_paired_direct_budget(
    tmp_path: Path,
) -> None:
    selection = tmp_path / "selection.json"
    _selection(selection)
    completed = subprocess.run(
        _command(tmp_path, selection),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)
    assert plan["method_id"] == "B7"
    assert plan["selected_style"] == "point"
    assert plan["local_pi05_loaded"] is False
    command = plan["command"]
    assert "--oracle-target-allow-absent-until-visible" in command
    assert command[command.index("--steps") + 1] == "400"
    assert command[command.index("--replan-steps") + 1] == "5"
    assert command[command.index("--initial-state") + 1].endswith("state.npz")


def test_b7_rejects_a_tied_or_missing_development_style(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    _selection(selection, style=None)
    completed = subprocess.run(
        _command(tmp_path, selection),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "uniquely selected development style" in completed.stderr
