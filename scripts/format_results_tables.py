#!/usr/bin/env python3
"""Render the sweep's measured numbers as Markdown tables.

The README quotes results, and numbers that are retyped by hand drift from the
run that produced them. This script reads ``rollout_summary.json`` for the
pooled per-condition report and the per-episode trace headers for the per-task
breakdown, then prints Markdown that can be pasted -- or diffed -- directly.

The per-task breakdown is not redundant with the pooled table. Pooling hides
the distinction the benchmark exists to draw: a condition can show a healthy
average endpoint rate purely because the two scenes that need no interaction
carry it, while every scene that does need interaction sits at zero.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", default=str(root / "outputs" / "challenge_rollout"))
    parser.add_argument(
        "--summary",
        default=str(root / "outputs" / "challenge_rollout" / "rollout_summary.json"),
    )
    parser.add_argument("--output", default=None, help="write here instead of stdout")
    return parser.parse_args()


def _pct(value: float | None) -> str:
    if value is None or value != value:  # NaN
        return "n/a"
    return f"{100.0 * value:.0f}%"


def _ci(bounds: Any) -> str:
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return ""
    low, high = bounds
    if low != low or high != high:
        return ""
    return f" [{100.0 * low:.0f}, {100.0 * high:.0f}]"


def load_headers(traces: Path) -> list[dict[str, Any]]:
    headers: list[dict[str, Any]] = []
    for trace in sorted(traces.rglob("trace.jsonl")):
        with trace.open("r", encoding="utf-8") as file:
            first = file.readline()
        if not first.strip():
            continue
        record = json.loads(first)
        if record.get("kind") == "episode":
            headers.append(record)
    return headers


def condition_table(summary: dict[str, Any], order: list[str]) -> list[str]:
    lines = [
        "| Prompt rung | Episodes | Task success | Information endpoint | "
        "Correct terminal | Premature commit | False NOT_FOUND |",
        "|---|---|---|---|---|---|---|",
    ]
    conditions = summary.get("conditions", {})
    for variant in order:
        report = conditions.get(variant)
        if not report:
            continue
        lines.append(
            f'| `{variant}` | {report["episodes"]} '
            f'| {_pct(report["success_rate"])}{_ci(report.get("success_ci95"))} '
            f'| {_pct(report["endpoint_rate"])}{_ci(report.get("endpoint_ci95"))} '
            f'| {_pct(report["correct_terminal_rate"])} '
            f'| {_pct(report["premature_commit_rate"])} '
            f'| {_pct(report["false_not_found_rate"])} |'
        )
    return lines


def per_task_table(
    headers: list[dict[str, Any]], order: list[str], attribute: str, title: str
) -> list[str]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for header in headers:
        key = (header["task_id"], header["prompt_variant"])
        grouped[key].append(float(header.get(attribute, 0.0)))

    tasks = sorted({task for task, _ in grouped})
    present = [variant for variant in order if any((t, variant) in grouped for t in tasks)]
    lines = [
        f"**{title}**",
        "",
        "| Task | " + " | ".join(f"`{variant}`" for variant in present) + " | n |",
        "|---" * (len(present) + 2) + "|",
    ]
    for task in tasks:
        cells = []
        count = 0
        for variant in present:
            values = grouped.get((task, variant), [])
            count = max(count, len(values))
            cells.append(_pct(sum(values) / len(values)) if values else "n/a")
        lines.append(f"| {task} | " + " | ".join(cells) + f" | {count} |")
    return lines


def validity_block(summary: dict[str, Any], order: list[str]) -> list[str]:
    lines = [
        "| Prompt rung | Mean vacuity | Mean dissonance | Saturated fraction | "
        "Uninformative episodes | Errors |",
        "|---|---|---|---|---|---|",
    ]
    for variant in order:
        report = summary.get("conditions", {}).get(variant)
        if not report:
            continue
        lines.append(
            f'| `{variant}` | {report["mean_vacuity"]:.3f} '
            f'| {report["mean_dissonance"]:.3f} '
            f'| {report["mean_saturated_fraction"]:.2f} '
            f'| {report["uninformative_episodes"]}/{report["episodes"]} '
            f'| {report["errors"]} |'
        )
    return lines


def contrast_block(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Contrast | Paired episodes | Mean difference in endpoint rate | 95% CI |",
        "|---|---|---|---|",
    ]
    labels = {
        "capability_minus_implicit": "`capability` - `implicit`",
        "explicit_minus_implicit": "`explicit` - `implicit`",
    }
    for key, label in labels.items():
        contrast = summary.get(key)
        if not contrast:
            continue
        low, high = contrast.get("ci95", (float("nan"), float("nan")))
        lines.append(
            f'| {label} | {contrast["pairs"]} '
            f'| {contrast["mean_difference"]:+.3f} '
            f"| [{low:+.3f}, {high:+.3f}] |"
        )
    return lines


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary).expanduser().resolve()
    if not summary_path.exists():
        raise SystemExit(f"no summary at {summary_path}; run the sweep first")
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    headers = load_headers(Path(args.traces).expanduser().resolve())
    order = list(summary.get("variants") or ["implicit", "hinted", "explicit", "capability"])

    blocks: list[str] = []
    blocks.append("#### Prompt ladder, pooled over scenes\n")
    blocks.extend(condition_table(summary, order))
    blocks.append("")
    blocks.append("#### Decision failure versus skill gap\n")
    blocks.extend(contrast_block(summary))
    blocks.append("")
    if headers:
        blocks.extend(
            per_task_table(
                headers, order, "information_endpoint_reached", "Information endpoint rate, per scene"
            )
        )
        blocks.append("")
        blocks.extend(
            per_task_table(headers, order, "task_success", "Task success rate, per scene")
        )
        blocks.append("")
    blocks.append("#### Are the uncertainty readings interpretable?\n")
    blocks.extend(validity_block(summary, order))
    blocks.append("")

    text = "\n".join(blocks)
    if args.output:
        path = Path(args.output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            file.write(text + "\n")
        print(f"wrote {path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
