"""Deterministic, evidence-bound SVG figures for the PIU manuscript."""

from __future__ import annotations

import hashlib
import html
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(value: str, *, repository_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def load_figure_config(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, Mapping) or value.get("schema_version") != (
        "piu.paper-figures.v1"
    ):
        raise ValueError("unsupported PIU paper-figure config")
    if value.get("status") != "frozen_automatic_rendering":
        raise ValueError("PIU paper-figure protocol is not frozen")
    scope = value.get("scope")
    claims = value.get("claim_contract")
    inputs = value.get("inputs")
    outputs = value.get("outputs")
    if not all(isinstance(item, Mapping) for item in (scope, claims, inputs, outputs)):
        raise TypeError("PIU paper-figure sections must be mappings")
    if (
        scope.get("method_interpretation")
        != "implemented_public_input_successor_not_empirically_validated"
        or scope.get("evidence_interpretation")
        != "retained_negative_result_plus_explicit_pending_successor"
    ):
        raise ValueError("PIU paper-figure scope overclaims the evidence")
    if (
        claims.get("missing_marker") != "PENDING"
        or claims.get("pending_may_be_rendered_as_zero") is not False
        or claims.get("software_verification_is_performance_evidence") is not False
        or claims.get("oracle_is_public_method") is not False
        or claims.get("automatic_success_threshold") is not None
        or claims.get("broad_ood_claim") is not False
        or claims.get("local_gpu_actions_performed") is not False
    ):
        raise ValueError("PIU paper-figure claim firewall was weakened")
    evidence_path = _resolve(str(inputs.get("evidence_tables", "")), repository_root=repository_root)
    evidence = json.loads(evidence_path.read_text())
    if (
        evidence.get("schema_version") != "piu.paper-evidence-tables.v1"
        or evidence.get("missing_values_encoded_as_zero") is not False
        or evidence.get("automatic_method_success") is not None
        or evidence.get("paper_method_claim_allowed") is not False
    ):
        raise ValueError("paper figures require the claim-safe evidence table")
    baseline_path = _resolve(str(inputs.get("baseline_registry", "")), repository_root=repository_root)
    risk_path = _resolve(str(inputs.get("risk_allocation", "")), repository_root=repository_root)
    baseline = yaml.safe_load(baseline_path.read_text())
    risk = yaml.safe_load(risk_path.read_text())
    if (
        baseline.get("schema_version") != "piu.baseline-registry.v1"
        or risk.get("schema_version") != "piu.executor-risk-allocation.v1"
        or baseline.get("scenario") != scope.get("scenario")
        or risk.get("scope", {}).get("scenario") != scope.get("scenario")
    ):
        raise ValueError("paper figures use inconsistent scenario contracts")
    if outputs.get("immutable") is not True:
        raise ValueError("paper figure outputs must be immutable")
    output_paths = [Path(str(outputs.get(name, ""))) for name in ("method", "evidence_boundary")]
    if (
        any(path.is_absolute() or path.suffix != ".svg" for path in output_paths)
        or len(set(output_paths)) != len(output_paths)
    ):
        raise ValueError("paper figure output paths are malformed")
    return {
        **dict(value),
        "_evidence": evidence,
        "_source_sha256": _sha256(evidence_path),
        "_maximum_decisions": baseline["shared_contract"][
            "maximum_controller_decisions"
        ],
    }


def _text(
    x: int,
    y: int,
    value: str,
    *,
    size: int = 22,
    weight: int = 500,
    fill: str = "#16324F",
    anchor: str = "middle",
) -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Inter,Arial,sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{html.escape(value)}</text>'
    )


def _box(
    x: int,
    y: int,
    width: int,
    height: int,
    lines: tuple[str, ...],
    *,
    fill: str,
    stroke: str = "#16324F",
    dashed: bool = False,
) -> str:
    dash = ' stroke-dasharray="9 7"' if dashed else ""
    result = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="16" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2.5"{dash}/>'
    ]
    baseline = y + height // 2 - (len(lines) - 1) * 15 + 7
    for index, line in enumerate(lines):
        result.append(_text(x + width // 2, baseline + index * 30, line, size=19))
    return "".join(result)


def _arrow(x1: int, y1: int, x2: int, y2: int, *, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="8 7"' if dashed else ""
    return (
        f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="#355C7D" '
        f'stroke-width="3" marker-end="url(#arrow)"{dash}/>'
    )


def _svg_start(width: int, height: int, *, title: str, description: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
        f'<title id="title">{html.escape(title)}</title>'
        f'<desc id="desc">{html.escape(description)}</desc>'
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" '
        'refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" '
        'fill="#355C7D"/></marker></defs>'
        '<rect width="100%" height="100%" fill="#FFFFFF"/>'
    )


def render_method_pipeline(config: Mapping[str, Any]) -> str:
    maximum = int(config["_maximum_decisions"])
    parts = [
        _svg_start(
            1500,
            900,
            title="PIU public-input method and executor qualification gates",
            description=(
                "Public prompt, RGB, and action history pass through the frozen "
                "pi0.5 prefix, learned binding and effect modules, isolated "
                "calibration, a typed controller, an externally risk-derived "
                "qualification gate, and the frozen external executor."
            ),
        ),
        _text(750, 48, "Prompt-conditioned Interaction Belief (implemented successor)", size=29, weight=700),
        _text(750, 78, "Public online path; real successor training and rollout evidence remain PENDING", size=18, fill="#5D6D7E"),
        _box(35, 145, 190, 120, ("prompt q", "RGB history", "public actions"), fill="#EAF2F8"),
        _box(270, 145, 210, 120, ("frozen pi0.5", "PaliGemma", "full prefix tokens"), fill="#D6EAF8"),
        _box(525, 115, 220, 105, ("prompt-conditioned", "spatial binder"), fill="#FDEBD0"),
        _box(525, 250, 220, 105, ("candidate-conditioned", "route + effect"), fill="#FDEBD0"),
        _box(790, 145, 205, 120, ("disjoint", "temperature +", "conformal roles"), fill="#E8DAEF"),
        _box(1040, 145, 205, 120, ("typed set-valued", "controller", "no top-1 fallback"), fill="#D5F5E3"),
        _arrow(225, 205, 270, 205),
        _arrow(480, 205, 525, 168),
        _arrow(480, 205, 525, 303),
        _arrow(745, 168, 790, 195),
        _arrow(745, 303, 790, 218),
        _arrow(995, 205, 1040, 205),
        _box(1280, 105, 185, 85, ("ABSTAIN /", "NOT_FOUND"), fill="#F4F6F7"),
        _arrow(1245, 175, 1280, 150),
        _box(1025, 360, 235, 105, ("exact candidate +", "current-frame boxes", "deterministic text"), fill="#D6EAF8"),
        _arrow(1142, 265, 1142, 360),
        _box(710, 365, 245, 100, ("external delta", f"1 - delta/{maximum}", "single-use certificate"), fill="#FADBD8", stroke="#A93226"),
        _arrow(1025, 413, 955, 413),
        _box(395, 365, 245, 100, ("identified external", "frozen pi0.5", "physical option"), fill="#D6EAF8"),
        _arrow(710, 415, 640, 415),
        _box(80, 365, 245, 100, ("new public RGB", "+ robot state", "+ action history"), fill="#EAF2F8"),
        _arrow(395, 415, 325, 415),
        '<path d="M202,365 C202,310 125,310 125,265" fill="none" stroke="#355C7D" stroke-width="3" marker-end="url(#arrow)"/>',
        _text(205, 325, "reobserve + update", size=17, fill="#355C7D"),
        '<line x1="35" y1="555" x2="1465" y2="555" stroke="#D1495B" stroke-width="2.5" stroke-dasharray="10 8"/>',
        _text(210, 588, "PRIVILEGED OFFLINE / EVALUATOR-ONLY", size=17, weight=700, fill="#A93226", anchor="start"),
        _box(90, 630, 280, 110, ("segmentation + contacts", "joints + task predicates", "never policy inputs"), fill="#FDEDEC", stroke="#A93226", dashed=True),
        _box(475, 630, 255, 110, ("training labels", "executed effects", "patch coverage"), fill="#FDEDEC", stroke="#A93226", dashed=True),
        _box(835, 630, 255, 110, ("registered success", "predicate replay", "formal outcomes"), fill="#FDEDEC", stroke="#A93226", dashed=True),
        _box(1195, 630, 235, 110, ("oracle B6 / B7", "upper bounds only", "separate columns"), fill="#FDEDEC", stroke="#A93226", dashed=True),
        _arrow(370, 685, 475, 685, dashed=True),
        _arrow(730, 685, 835, 685, dashed=True),
        _text(750, 825, "No learned confidence sum, visibility cutoff, manual utility, or unqualified physical dispatch", size=20, weight=650),
        "</svg>\n",
    ]
    return "".join(parts)


def _evidence_row(evidence: Mapping[str, Any], condition: str, metric: str) -> Mapping[str, Any]:
    matches = [
        row
        for row in evidence["retained_negative_evidence"]
        if row.get("condition") == condition and row.get("metric") == metric
    ]
    if len(matches) != 1:
        raise ValueError(f"paper evidence lacks {condition}/{metric}")
    return matches[0]


def render_evidence_boundary(config: Mapping[str, Any]) -> str:
    evidence = config["_evidence"]
    opened = _evidence_row(evidence, "open_closed_drawer", "drawer_open")
    acquired = _evidence_row(evidence, "open_closed_drawer", "information_acquired")
    utilized = _evidence_row(evidence, "direct_after_actual_open", "target_pick")
    artifacts = evidence["artifacts"]
    if any(
        artifacts[name]["status"] != "PENDING"
        for name in ("binding_development", "effect_development", "closed_loop_sealed")
    ):
        raise ValueError("v1 evidence-boundary figure must not silently change status")
    source = str(config["_source_sha256"])
    parts = [
        _svg_start(
            1400,
            720,
            title="Observed acquisition-to-utilization gap and successor claim boundary",
            description=(
                "Retained data show drawer opening and information acquisition but "
                "zero post-open target contact. Successor model and closed-loop "
                "performance evidence are pending, not zero."
            ),
        ),
        f'<metadata>evidence-table-sha256:{source}</metadata>',
        _text(700, 48, "What is observed, and what is still untested", size=30, weight=700),
        _text(330, 95, "Retained same-scenario physical evidence", size=22, weight=700),
        _box(55, 145, 190, 115, ("OPEN", f"{opened['successes']}/{opened['trials']}", "drawer predicate"), fill="#D5F5E3"),
        _box(280, 145, 190, 115, ("new butter evidence", f"{acquired['successes']}/{acquired['trials']}", "nonempty mask"), fill="#FCF3CF"),
        _box(505, 145, 190, 115, ("post-OPEN contact", f"{utilized['successes']}/{utilized['trials']}", f"0/{acquired['successes']} given evidence"), fill="#FADBD8", stroke="#A93226"),
        _arrow(245, 202, 280, 202),
        _arrow(470, 202, 505, 202),
        _text(375, 315, "Information acquisition does not imply information utilization", size=21, weight=700, fill="#A93226"),
        _text(1045, 95, "Successor evidence boundary", size=22, weight=700),
        _box(795, 140, 240, 95, ("pipeline contracts", "VERIFIED", "CPU regression suite"), fill="#D6EAF8"),
        _box(1070, 140, 260, 95, ("binder + effect", "PENDING", "PENDING is not zero"), fill="#F4F6F7", dashed=True),
        _box(795, 275, 240, 95, ("calibration claims", "PENDING", "real isolated groups"), fill="#F4F6F7", dashed=True),
        _box(1070, 275, 260, 95, ("sealed B0--B8", "PENDING", "new paired groups"), fill="#F4F6F7", dashed=True),
        _box(875, 420, 375, 100, ("automatic method success = null", "paper-method claim = blocked", "missing evidence is never imputed"), fill="#FADBD8", stroke="#A93226"),
        '<line x1="735" y1="100" x2="735" y2="560" stroke="#AAB7B8" stroke-width="2"/>',
        _text(700, 610, "Allowed conclusion: a reproducible acquisition-to-utilization failure decomposition", size=22, weight=700),
        _text(700, 646, "Disallowed conclusion: PIU-VLA improves target contact or task success before real external experiments", size=19, fill="#A93226"),
        _text(700, 687, f"Source: paper evidence table SHA-256 {source[:16]}...", size=15, fill="#5D6D7E"),
        "</svg>\n",
    ]
    return "".join(parts)


def render_figures(config_path: Path, *, repository_root: Path) -> dict[str, str]:
    config = load_figure_config(config_path, repository_root=repository_root)
    return {
        "method": render_method_pipeline(config),
        "evidence_boundary": render_evidence_boundary(config),
    }
