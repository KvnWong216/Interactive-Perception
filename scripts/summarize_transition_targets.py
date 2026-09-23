"""Export measured transition diagnostics; never relabel development as confirmation."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/transition_targets_s8227"
ASSETS = ROOT / "docs/assets/stage2_debug"
ASSETS.mkdir(exist_ok=True)
audit = json.loads((RUN / "audit.json").read_text())
controls = json.loads((RUN / "controls.json").read_text())
rgb = json.loads((RUN / "rgb_oracle.json").read_text())
rows = []
for fit in controls["fits"]:
    values = [r for r in fit["rows"] if r["case"]["split"] == "development"]
    rows.append(
        {
            "model": "small_affine_" + fit["kind"],
            "manual_seed": fit["manual_seed"],
            "loss": np.mean([r["loss"] for r in values]),
            "swapped_loss": np.mean([r["swapped_loss"] for r in values]),
            "action_variance_explained_macro": np.mean(
                [r["action_variance_explained"] for r in values]
            ),
            "action_variance_explained": None,
            "seconds": None,
            "peak_allocated_bytes": None,
        }
    )
for path in sorted((ROOT / "runs/transition_candidate_s8227").glob("*/report.json")):
    report = json.loads(path.read_text())
    m = report["result"]["summary"]["development"]
    rows.append(
        {
            "model": path.parent.name,
            "manual_seed": report["manual_seed"],
            "loss": m["loss"],
            "swapped_loss": m["swapped_loss"],
            "action_variance_explained_macro": None,
            "action_variance_explained": m["action_variance_explained"],
            "seconds": report["seconds"],
            "peak_allocated_bytes": report["peak_allocated_bytes"],
        }
    )
with (ASSETS / "transition_interventions.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
physical = [r for r in audit["physical"] if r["split"] == "development"]
summary = {
    "split": "existing head development, not fresh confirmation",
    "physical": physical,
    "rows": rows,
    "native_affine_grid": "two validated native 14x14 grids, nonuniform edge spacing",
    "production_training_started": False,
}
(ASSETS / "transition_interventions.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
axes[0].bar(
    ["Copy state", "Actions", "State+actions", "VLM+actions"],
    [
        physical[0]["copy_rmse_m"] * 1000,
        physical[0]["rmse_m"] * 1000,
        physical[2]["rmse_m"] * 1000,
        physical[3]["rmse_m"] * 1000,
    ],
    color=["#888", "#729fcf", "#3b82a0", "#3b82a0"],
)
axes[0].set_ylabel("End-effector coordinate RMSE (mm)")
axes[0].set_title("A. Physical transfer is learnable")
axes[0].tick_params(axis="x", rotation=25)
rr = [
    r
    for r in rgb["rows"]
    if r["case"]["split"] == "development" and r["view"] == "wrist"
]
ff = [
    r for r in audit["oracle"] if r["case"]["split"] == "development" and r["view"] == 1
]
gain_rgb = 1 - np.mean([r["oracle_mse"] for r in rr]) / np.mean(
    [r["copy_mse"] for r in rr]
)
gain_latent = 1 - np.mean([r["oracle_loss"] for r in ff]) / np.mean(
    [r["copy_loss"] for r in ff]
)
axes[1].bar(
    ["RGB MSE", "Latent SmoothL1"],
    [gain_rgb * 100, gain_latent * 100],
    color=["#3b82a0", "#d09052"],
)
axes[1].set_ylabel("Reduction vs copy (%)")
axes[1].set_title("B. Privileged geometric transport")
axes[1].text(
    0.03,
    0.92,
    "Visible pixels/patches only\nDifferent losses and sampling",
    transform=axes[1].transAxes,
    fontsize=9,
    va="top",
)
live = [r["loss"] for r in rows if r["model"].startswith("control_residual_s")]
blind = [r["loss"] for r in rows if r["model"].startswith("control_residual_blind_s")]
wrong = [r["swapped_loss"] for r in rows if r["model"].startswith("control_residual_s")]
axes[2].bar(
    ["Copy", "Blind", "Correct", "Wrong"],
    [0.09322385241587956, np.mean(blind), np.mean(live), np.mean(wrong)],
    yerr=[0, np.std(blind, ddof=1), np.std(live, ddof=1), np.std(wrong, ddof=1)],
    capsize=3,
    color=["#888", "#aaa", "#3b82a0", "#d09052"],
)
axes[2].set_ylabel("Future latent SmoothL1")
axes[2].set_title("C. Implemented candidate, 3 seeds")
axes[2].set_ylim(0, 0.12)
fig.suptitle("Development diagnostics: frozen policy; no VLA success-rate claim")
fig.savefig(ASSETS / "transition_interventions.png", dpi=180)
fig.savefig(ASSETS / "transition_interventions.pdf")
plt.close(fig)
print(
    json.dumps(
        {
            "physical": physical,
            "oracle_rgb_gain": gain_rgb,
            "oracle_latent_gain": gain_latent,
            "candidate_loss": float(np.mean(live)),
            "candidate_vs_copy_gain": float(1 - np.mean(live) / 0.09322385241587956),
            "blind_loss": float(np.mean(blind)),
            "swapped_loss": float(np.mean(wrong)),
        }
    )
)
