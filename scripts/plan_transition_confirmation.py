"""Plan new same-state forks on policy-validation families; do not collect or train.

These are held out from predictor fitting, not an untouched final policy test.
Policy-test episodes remain excluded. Manual seed 9227 controls all selections.
"""

import json
import random
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/transition_confirmation_s9227"
seed = 9227
rng = random.Random(seed)
entries = json.loads((ROOT / "data/prepared/stage1_full/manifest.json").read_text())[
    "episodes"
]
entries = [r for r in entries if r["split"] == "validation"]
old = json.loads((ROOT / "runs/stage2_forks_s7227/plan.json").read_text())
old_families = {c["family"] for c in old["cases"]}
cases = []
for task_index, family in enumerate(sorted({r["reset_family"] for r in entries})):
    assert family not in old_families
    choices = [r for r in entries if r["reset_family"] == family]
    task = rng.choice(sorted({r["task"] for r in choices}))
    choices = [r for r in choices if r["task"] == task]
    rng.shuffle(choices)
    for entry in choices[:4]:
        suite, name, episode = entry["episode_id"].split("/")
        source = ROOT / "data/libero_raw" / suite / (name + ".hdf5")
        with h5py.File(source, "r") as f:
            length = len(f[f"data/{episode}/actions"])
        for fraction in [0.25, 0.65]:
            cases.append(
                {
                    "case_id": len(cases),
                    "task_index": task_index,
                    "family": family,
                    "task": task,
                    "split": "confirmation",
                    "source": str(source.relative_to(ROOT)),
                    "episode": episode,
                    "recorded_step": int(length * fraction),
                    "manual_seed": seed + len(cases),
                    "source_policy_split": "validation",
                }
            )
plan = {
    "data_manual_seed": seed,
    "cases": cases,
    "horizons": [5, 10, 30],
    "branches": old["branches"],
    "reset": old["reset"],
    "collected": False,
    "claim_limit": "New branch outcomes, disjoint predictor task families. Policy-validation tasks were used in earlier checkpoint selection; not a final policy test.",
    "excluded_policy_test": True,
}
OUT.mkdir(parents=True, exist_ok=False)
(OUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
print(
    json.dumps(
        {
            "cases": len(cases),
            "families": len({c["family"] for c in cases}),
            "collected": False,
        }
    )
)
