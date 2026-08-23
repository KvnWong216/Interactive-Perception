from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from piu.paper_figures import load_figure_config, render_figures

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/piu_paper_figures_v1.yaml"


def test_figures_separate_observed_negative_evidence_from_pending_claims() -> None:
    rendered = render_figures(CONFIG, repository_root=ROOT)
    assert "9/10" in rendered["evidence_boundary"]
    assert "8/10" in rendered["evidence_boundary"]
    assert "0/8 given evidence" in rendered["evidence_boundary"]
    assert "PENDING" in rendered["evidence_boundary"]
    assert "automatic method success = null" in rendered["evidence_boundary"]
    assert "external delta" in rendered["method"]
    assert "1 - delta/8" in rendered["method"]
    assert "never policy inputs" in rendered["method"]
    assert "<script" not in rendered["method"]
    assert "<image" not in rendered["method"]


def test_figure_config_rejects_pending_as_zero(tmp_path: Path) -> None:
    value = yaml.safe_load(CONFIG.read_text())
    value["claim_contract"]["pending_may_be_rendered_as_zero"] = True
    path = tmp_path / "unsafe.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    with pytest.raises(ValueError, match="claim firewall"):
        load_figure_config(path, repository_root=ROOT)


def test_paper_figure_cli_is_immutable_and_verifiable(tmp_path: Path) -> None:
    method = tmp_path / "method.svg"
    evidence = tmp_path / "evidence.svg"
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluation/build_piu_paper_figures.py"),
        "--method-output",
        str(method),
        "--evidence-output",
        str(evidence),
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    subprocess.run(
        [*command, "--verify"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    duplicate = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True
    )
    assert duplicate.returncode != 0
    assert "immutable" in duplicate.stderr
