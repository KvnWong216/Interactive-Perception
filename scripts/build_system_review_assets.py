"""Export presentation figures from the completed three-system evaluation only."""

import csv
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    output = ROOT / "docs/assets/stage2_assessment"
    evidence = json.loads((output / "evidence.json").read_text())
    if not evidence["complete"] or evidence["balanced_cases"] != 740:
        raise ValueError("presentation requires all 740 matched cases")
    models = ("native", "best", "stage2_best")
    labels = ("MolmoAct2 native", "Stage 1 best", "Stage 2 best")
    colors = ("#7B8794", "#2878B5", "#E07B39")
    conditions = ("normal", "camera", "lighting", "combined")
    rows = [r for r in evidence["scores"] if r["kind"] == "S" and r["split"] == "all"]
    table = []
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for i, (model, label, color) in enumerate(zip(models, labels, colors, strict=True)):
        selected = [next(r for r in rows if r["policy"] == model and r["condition"] == c) for c in conditions]
        values = [100 * r["successes"] / r["episodes"] for r in selected]
        bars = ax.bar(np.arange(4)+(i-1)*.26, values, width=.25, color=color, label=label)
        for bar, row in zip(bars, selected, strict=True):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f"{row['successes']}/{row['episodes']}", ha="center", fontsize=10)
        table.extend(selected)
    ax.set_xticks(range(4), ["Normal", "Camera shift", "Lighting shift", "Camera + lighting"])
    ax.set_ylim(0, 112); ax.set_ylabel("Full-task success (%)")
    ax.set_title("Stage 1 gains under perturbations; Stage 2 adds no overall gain")
    ax.legend(loc="lower left"); ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .01, "40 tasks × 4 paired resets per condition; one trained seed per stage. Exploratory evaluation.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(output / f"three_system_task_success.{ext}", dpi=220)
    plt.close(fig)

    variants = ("copy_current", "actual", "same_task_other_episode", "zero_control")
    variant_labels = ("Copy current", "True actions", "Wrong actions", "Zero controls")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    pred_table = []
    for ax, split in zip(axes, ("validation", "test"), strict=True):
        values = []
        for variant in variants:
            selected = [r for r in evidence["prediction_by_task"]
                        if r["split"] == split and r["region"] == "changed" and r["variant"] == variant]
            value = float(np.mean([r["loss"] for r in selected])); values.append(value)
            pred_table.append({"split": split, "variant": variant, "changed_patch_task_macro_loss": value})
        bars = ax.bar(range(4), values, color=("#7B8794", "#E07B39", "#F0AC82", "#FACDB1"))
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x()+bar.get_width()/2, value+.009, f"{value:.6f}", ha="center", fontsize=9)
        ax.set_xticks(range(4), variant_labels, rotation=20, ha="right")
        ax.set_title(split.capitalize()+" (4 tasks, 96 windows)")
        ax.set_ylim(0, .43); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Future-feature error (lower is better)")
    fig.suptitle("Future prediction fails the persistence baseline and action-utility diagnostic")
    fig.text(.5, .01, "Changed patches: normalized feature RMS > 0.1; trajectory then task averaging. Offline substitutions.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    for ext in ("png", "pdf"):
        fig.savefig(output / f"prediction_mechanism_review.{ext}", dpi=220)
    plt.close(fig)
    for name, values in (("presentation_success.csv", table), ("presentation_prediction.csv", pred_table)):
        with (output / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    print(json.dumps({"output": str(output.relative_to(ROOT)), "complete": True}))


if __name__ == "__main__":
    main()
