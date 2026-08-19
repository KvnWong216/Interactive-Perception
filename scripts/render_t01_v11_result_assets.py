#!/usr/bin/env python3
"""Render compact paper-facing plots from frozen T01 v11 result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("FAILED", "REVEALED", "EMPTY")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path) -> None:
    plt.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_clean_development_v11.json",
    )
    parser.add_argument(
        "--ablations",
        type=Path,
        default=ROOT / "results/calibration/t01_open_and_observe_v11_ablations.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/assets/t01_open_and_observe_v11_clean",
    )
    args = parser.parse_args()
    for name in ("result", "ablations", "output_dir"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    manifest_path = args.output_dir / "assets_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = json.loads(args.result.read_text())
    ablations = json.loads(args.ablations.read_text())

    produced = []

    per_class_path = args.output_dir / "v11_per_class_reliability.png"
    coverage = [result["per_class"][label]["coverage"] for label in CLASSES]
    lower = [
        result["per_class"][label]["singleton_one_sided_95_lower"]
        for label in CLASSES
    ]
    x = np.arange(len(CLASSES))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    axis.bar(x - width / 2, coverage, width, label="Correct-label retention")
    axis.bar(x + width / 2, lower, width, label="Singleton 95% lower bound")
    axis.axhline(0.90, color="#D62828", linestyle="--", label="Original 0.90")
    axis.axhline(0.80, color="#F4A261", linestyle=":", label="Development 0.80")
    axis.set_xticks(x, CLASSES)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Rate / lower bound")
    axis.set_title(f"T01 v11 clean development · {result['decision']}")
    axis.legend(loc="lower left", fontsize=8)
    save(per_class_path)
    produced.append(("per_class_reliability", per_class_path))

    physical_path = args.output_dir / "v11_physical_information_effect.png"
    physical = result["physical_information_acquisition"]
    labels = ("REVEALED", "EMPTY")
    points = [physical[label]["rate"] for label in labels]
    bounds = [physical[label]["one_sided_95_lower"] for label in labels]
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    bars = axis.bar(labels, points, color=("#2A9D8F", "#F4A261"))
    for index, (point, bound) in enumerate(zip(points, bounds, strict=True)):
        axis.vlines(index, bound, point, color="black", linewidth=2)
        axis.scatter([index], [bound], color="black", marker="_", s=120)
        axis.text(index, min(point + 0.025, 1.02), f"{point:.3f}\nLB {bound:.3f}", ha="center")
    axis.axhline(0.80, color="#D62828", linestyle="--", label="0.80 gate")
    axis.set_ylim(0.0, 1.07)
    axis.set_ylabel("Information-acquisition success")
    axis.set_title("Physical endpoint (policy variability reported separately)")
    axis.legend()
    save(physical_path)
    produced.append(("physical_information_effect", physical_path))

    arms_path = args.output_dir / "v11_ablation_summary.png"
    arm_order = (
        "full_v11_six_frame",
        "no_wrist_rescue",
        "endpoint_only_before_after",
        "final_frame_only",
        "no_coverage_head",
        "trivial_all_labels",
    )
    display = (
        "Full v11",
        "No wrist",
        "Before+after",
        "Final only",
        "No coverage",
        "All labels",
    )
    singletons = [ablations["arms"][arm]["singleton_accuracy"] for arm in arm_order]
    set_sizes = [ablations["arms"][arm]["mean_prediction_set_size"] for arm in arm_order]
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    colors = ["#E63946" if arm == "full_v11_six_frame" else "#457B9D" for arm in arm_order]
    axes[0].bar(display, singletons, color=colors)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Singleton accuracy")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(display, set_sizes, color=colors)
    axes[1].axhline(1.0, color="black", linewidth=0.8)
    axes[1].set_ylim(0.0, 3.1)
    axes[1].set_ylabel("Mean prediction-set size")
    axes[1].tick_params(axis="x", rotation=25)
    figure.suptitle("T01 v11 temporal/camera/coverage ablations")
    save(arms_path)
    produced.append(("ablation_summary", arms_path))

    history_path = args.output_dir / "v11_history_reveal_ablation.png"
    history = result["history_ablation"]
    values = (
        history["six_frame_resolvable_reveals"],
        history["final_frame_only_resolvable_reveals"],
    )
    figure, axis = plt.subplots(figsize=(5.8, 4.1))
    axis.bar(("Six-frame history", "Final frame only"), values, color=("#2A9D8F", "#64748B"))
    axis.set_ylim(0, history["trials"] + 2)
    axis.set_ylabel(f"Resolvable reveals / {history['trials']}")
    axis.set_title("Evaluator-only history retention diagnostic")
    for index, value in enumerate(values):
        axis.text(index, value + 0.35, str(value), ha="center")
    save(history_path)
    produced.append(("history_ablation", history_path))

    manifest = {
        "schema_version": "interactive-perception.t01-v11-result-assets.v1",
        "source_result": {
            "path": str(args.result.relative_to(ROOT)),
            "sha256": digest(args.result),
        },
        "source_ablations": {
            "path": str(args.ablations.relative_to(ROOT)),
            "sha256": digest(args.ablations),
        },
        "claim_scope": result.get("interpretation", {}),
        "assets": [
            {
                "kind": kind,
                "path": str(path.relative_to(ROOT)),
                "sha256": digest(path),
            }
            for kind, path in produced
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"assets": len(produced), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
