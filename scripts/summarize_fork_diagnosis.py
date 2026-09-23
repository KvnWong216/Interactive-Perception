"""Consolidate all diagnostic evidence and render a scientific summary figure."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/stage2_forks_s7227"
ASSETS = ROOT / "docs/assets/stage2_debug"


def compact(metric):
    rows = metric["rows"]
    action = mean(r["action_component_mse"] for r in rows)
    target = mean(r["target_action_mse"] for r in rows)
    return {
        key: metric[key]
        for key in [
            "mean_loss",
            "copy_loss",
            "pair_assignment_accuracy",
            "pair_tie_fraction",
            "wrong_minus_correct",
            "separation_ratio",
        ]
    } | {
        "explained_action_variance": 1 - action / target,
        "mean_component_mse": mean(r["mean_component_mse"] for r in rows),
        "action_component_mse": action,
        "target_action_mse": target,
    }


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    counts = {}
    all_rows = []
    for name, folder, expected in [
        ("query_convergence", ROOT / "runs/stage2_query_convergence_s7227", 36),
        ("fork_fits", OUT / "fits", 30),
        ("effect_fits", OUT / "effect_fits", 15),
        ("single_case_fits", OUT / "single_case_fits", 12),
        ("width_fits", OUT / "width_fits", 6),
    ]:
        reports = [
            json.loads(p.read_text()) for p in sorted((folder / "jobs").glob("*.json"))
        ]
        if len(reports) != expected or not all(r["complete"] for r in reports):
            raise ValueError(f"incomplete {name}")
        counts[name] = len(reports)
        for report in reports:
            job = report["job"]
            row = {"experiment": name, **job}
            if name == "query_convergence":
                row.update(
                    train_loss=report["train"]["mean"]["loss"],
                    development_loss=report["development"]["mean"]["loss"],
                )
            else:
                row.update(
                    train_loss=report["train"]["mean_loss"],
                    development_loss=report["development"]["mean_loss"],
                    train_separation=report["train"]["separation_ratio"],
                    development_separation=report["development"]["separation_ratio"],
                    development_assignment=report["development"][
                        "pair_assignment_accuracy"
                    ],
                )
            all_rows.append(row)
    models = []
    for shard in [0, 1]:
        report = json.loads((OUT / f"confirmation_{shard}.json").read_text())
        assert report["complete"]
        for r in report["results"]:
            models.append(
                {"model": r["model"], "job": r["job"], **compact(r["confirmation"])}
            )
    groups = defaultdict(list)
    for r in models:
        label = (
            "stage2"
            if r["model"] == "original stage2 best"
            else "blind"
            if r["job"]["blind"]
            else "effect"
            if r["job"]["effect_weight"]
            else "absolute"
        )
        groups[label].append(r)
    summary = {}
    for label, rows in groups.items():
        metrics = [k for k in rows[0] if k not in ["model", "job"]]
        summary[label] = {
            "n_seeds": len(rows),
            "mean": {k: mean(r[k] for r in rows) for k in metrics},
            "seed_std": {
                k: stdev(r[k] for r in rows) if len(rows) > 1 else None for k in metrics
            },
        }
    result = {
        "complete": True,
        "additional_fits": sum(counts.values()),
        "counts": counts,
        "initial_states": 48,
        "distinct_action_rollouts": 288,
        "repeated_hold_rollouts": 96,
        "horizon_endpoints": 1152,
        "confirmation": summary,
        "confirmation_models": models,
        "limitations": [
            "Predictor diagnostics, not VLA task success",
            "Confirmation: 2 task/scene groups, 6 source episodes, 12 takeover states; correlated branch pairs are not independent trials",
            "No candidate passed overall future error improvement gate; no production weights changed",
        ],
        "teacher_confirmation_probe": json.loads(
            (OUT / "teacher_confirmation_probe.json").read_text()
        ),
    }
    (OUT / "final_diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    (ASSETS / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = list(dict.fromkeys(k for r in all_rows for k in r))
    with (ASSETS / "all_diagnostic_fits.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    with (ASSETS / "confirmation.csv").open("w") as file:
        rows = [
            {
                "model": r["model"],
                **{k: v for k, v in r.items() if k not in ["model", "job"]},
            }
            for r in models
        ]
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["Stage2", "Residual", "Effect loss", "Action blind"]
    keys = ["stage2", "absolute", "effect", "blind"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    metrics = [
        ("mean_loss", "Future-feature error (lower better)", 1),
        ("pair_assignment_accuracy", "Correct pair assignment (%)", 100),
        ("separation_ratio", "Predicted / actual difference (%)", 100),
    ]
    for ax, (metric, title, scale) in zip(axes, metrics):
        values = [summary[k]["mean"][metric] * scale for k in keys]
        errors = [(summary[k]["seed_std"][metric] or 0) * scale for k in keys]
        ax.bar(
            labels,
            values,
            yerr=errors,
            color=["#7a8793", "#3274a1", "#d48a30", "#b9bcc0"],
            capsize=3,
        )
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        for i, v in enumerate(values):
            ax.text(
                i,
                v + max(values) * 0.05,
                f"{v:.3f}" if scale == 1 else f"{v:.1f}",
                ha="center",
                fontsize=8,
            )
        ax.set_ylim(0, max(values) * 1.3)
    axes[0].axhline(
        summary["stage2"]["mean"]["copy_loss"],
        color="black",
        linestyle="--",
        linewidth=1,
        label="Copy current",
    )
    axes[0].legend(fontsize=8)
    axes[2].set_ylim(0, 110)
    axes[2].axhline(100, color="black", linestyle="--", linewidth=1)
    fig.suptitle(
        "Locked diagnostic confirmation: discrimination improves, future prediction remains inaccurate",
        fontsize=12,
    )
    fig.savefig(ASSETS / "confirmation_diagnosis.png", dpi=180)
    fig.savefig(ASSETS / "confirmation_diagnosis.pdf")
    plt.close(fig)
    # Fixed first confirmation case, selected before looking at model outcomes.
    folder = OUT / "case_036"
    names = ["Current", "+X after 10 steps", "-X after 10 steps"]
    files = ["history_2.npz", "positive_x_10.npz", "negative_x_10.npz"]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), layout="constrained")
    for col, (name, file) in enumerate(zip(names, files)):
        with np.load(folder / file, allow_pickle=False) as frame:
            for row, view in enumerate(["agent", "wrist"]):
                axes[row, col].imshow(frame[f"{view}_rgb"])
                axes[row, col].axis("off")
                axes[row, col].set_title(f"{name} / {view}", fontsize=9)
    fig.suptitle(
        "Same initial observation, different actual controls and futures", fontsize=12
    )
    fig.savefig(ASSETS / "same_state_fork.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"counts": counts, "confirmation": summary}, indent=2))


if __name__ == "__main__":
    main()
