"""Aggregate completed diagnostic fits; never pick a production checkpoint."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/stage2_real_debug_s6227"


def main():
    reports = [
        json.loads(p.read_text()) for p in sorted((OUT / "jobs").glob("job_*.json"))
    ]
    reports = [r for r in reports if r["complete"]]
    groups = defaultdict(list)
    rows = []
    for r in reports:
        job = r["job"]
        metrics = r["development"]["mean"]
        row = {
            **job,
            "train_loss": r["train"]["mean"]["loss"],
            **{"development_" + k: v for k, v in metrics.items()},
            "seconds": r["seconds"],
        }
        rows.append(row)
        groups[(job["variant"], job["initialization"], job["n"], job["lr"])].append(row)
    summary = []
    for (variant, init, n, lr), values in groups.items():
        summary.append(
            {
                "variant": variant,
                "initialization": init,
                "n": n,
                "lr": lr,
                "seeds": len(values),
                "train_loss": mean(x["train_loss"] for x in values),
                "development_loss": mean(x["development_loss"] for x in values),
                "development_loss_std": stdev(x["development_loss"] for x in values)
                if len(values) > 1
                else None,
                "changed_loss": mean(x["development_changed_loss"] for x in values),
                "wrong_minus_correct": mean(
                    x["development_wrong_loss"] - x["development_loss"] for x in values
                ),
            }
        )
    result = {
        "completed_jobs": len(rows),
        "expected_jobs": 48,
        "complete": len(rows) == 48,
        "baseline": json.loads((OUT / "baseline.json").read_text())
        if (OUT / "baseline.json").exists()
        else None,
        "groups": summary,
        "claim_limit": "Development only, 3 scene groups seen during original stage2 training. No causal fork or policy rollout. Groups with fewer than 3 seeds incomplete.",
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if rows:
        with (OUT / "scores.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
