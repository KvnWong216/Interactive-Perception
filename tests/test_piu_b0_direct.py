from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_b0_direct_dry_run_uses_frozen_shared_budget(tmp_path: Path) -> None:
    state = tmp_path / "state.npz"
    state.write_bytes(b"opaque")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/pipeline/run_piu_b0_direct.py"),
            "--scenario-config",
            str(ROOT / "configs/scenarios/original_drawer.yaml"),
            "--initial-state",
            str(state),
            "--initial-state-group",
            "group",
            "--split",
            "development",
            "--seed",
            "3",
            "--host",
            "pi05.example.internal",
            "--output-dir",
            str(tmp_path / "output"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)
    assert plan["method_id"] == "B0"
    assert plan["local_pi05_loaded"] is False
    steps = plan["command"].index("--steps")
    assert plan["command"][steps + 1] == "400"
