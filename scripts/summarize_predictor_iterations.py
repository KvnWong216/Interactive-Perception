"""Export every completed iteration, including failed qualification, for review."""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="runs/predictor_iterations_s11227")
    p.add_argument("--output", default="docs/assets/stage2_debug/iterations")
    a = p.parse_args()
    source, out = (ROOT / a.source).resolve(), (ROOT / a.output).resolve()
    if any(not x.is_relative_to(ROOT) for x in [source, out]):
        raise ValueError("paths must stay in repository")
    out.mkdir(parents=True, exist_ok=True)
    fits, families = [], []
    for path in sorted(source.glob("*/*/report.json")):
        d = json.loads(path.read_text())
        if "result" not in d:
            continue
        info = {
            k: d.get(k)
            for k in [
                "model_kind",
                "effect_weight",
                "steps",
                "manual_seed",
                "training_cases",
                "blind",
                "parameters",
                "seconds",
                "peak_allocated_bytes",
            ]
        }
        info["source_report"] = str(path.relative_to(ROOT))
        for split, row in d["result"]["summary"].items():
            fits.append({**info, "split": split, **row})
        groups = {}
        for row in d["result"]["rows"]:
            groups.setdefault((row["case"]["split"], row["case"]["family"]), []).append(
                row
            )
        for (split, family), rows in groups.items():

            def mean(key, rows=rows):
                return sum(r[key] for r in rows) / len(rows)

            families.append(
                {
                    **info,
                    "split": split,
                    "family": family,
                    "episodes": len(rows),
                    "loss": mean("loss"),
                    "copy_gain": 1 - mean("loss") / mean("copy_loss"),
                    "action_variance_explained": 1
                    - mean("action_mse") / mean("target_action_mse"),
                }
            )
    qualification = json.loads((source / "confirmation/qualification.json").read_text())
    confirmations = []
    for r in qualification["seed_results"]:
        live, blind = r["live"], r["blind"]
        confirmations.append(
            {
                "manual_seed": r["manual_seed"],
                "loss": live["loss"],
                "copy_gain": 1 - live["loss"] / live["copy_loss"],
                "blind_gain": 1 - live["loss"] / blind["loss"],
                "wrong_cost": live["swapped_loss"] / live["loss"] - 1,
                "action_variance_explained": live["action_variance_explained"],
                **r["checks"],
            }
        )
    for name, rows in [
        ("fits", fits),
        ("families", families),
        ("confirmation", confirmations),
    ]:
        with (out / f"{name}.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    evidence = {
        "scope": "Frozen predictor mechanisms; no VLA success or active-perception claim.",
        "manual_seeds": [17, 29, 43],
        "qualification_passed": qualification["passed"],
        "fits": fits,
        "families": families,
        "qualification": qualification,
    }
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, title, metric in zip(
        axes,
        ["Copy-current error reduction", "Action variance explained"],
        ["copy_gain", "action_variance_explained"],
        strict=True,
    ):
        ax.bar(
            [str(r["manual_seed"]) for r in confirmations],
            [100 * r[metric] for r in confirmations],
            color="#52789b",
        )
        ax.axhline(10, color="#b44343", linestyle="--", label="Fixed 10% gate")
        ax.set_title(title)
        ax.set_xlabel("Manual seed")
        ax.set_ylabel("Percent")
        ax.set_ylim(0, 14)
        ax.legend()
    fig.suptitle(
        "Local transport: independent confirmation FAILED (36 episodes / 3 families)"
    )
    for extension in ["png", "pdf"]:
        fig.savefig(out / f"local_confirmation.{extension}", dpi=180)
    plt.close(fig)
    (out / "index.html").write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8"><title>Predictor iteration evidence</title><body><h1>Predictor iteration evidence</h1><p>Frozen Molmo features, 250 training initial states, six actual action branches per state. These are prediction diagnostics, not closed-loop VLA scores.</p><p>Local transport failed the fixed independent confirmation gate. The failed result is retained. Later development fits do not override this decision.</p><img width="1000" src="local_confirmation.png" alt="All three seeds below copy and action variance gates"><ul><li><a href="fits.csv">Every completed fit and split</a></li><li><a href="families.csv">Family-level results</a></li><li><a href="confirmation.csv">Independent confirmation by seed</a></li><li><a href="evidence.json">Full numerical evidence and paired cases</a></li></ul></body></html>"""
    )
    print(
        json.dumps(
            {
                "fits": len(fits),
                "output": str(out),
                "qualification_passed": qualification["passed"],
            }
        )
    )


if __name__ == "__main__":
    main()
