"""Lock balanced, episode-disjoint data before collecting scaling experiments."""

import json
import random
from collections import Counter
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/transition_scaling_s10227"


def main():
    rng = random.Random(10227)
    entries = json.loads(
        (ROOT / "data/prepared/stage1_full/manifest.json").read_text()
    )["episodes"]
    entries = [e for e in entries if e["split"] == "train"]
    old = json.loads((ROOT / "runs/stage2_forks_s7227/plan.json").read_text())
    used = {(c["source"], c["episode"]) for c in old["cases"]}
    fresh_families = ["scene_11", "scene_14", "scene_17"]
    dev_families = ["scene_13", "scene_21"]
    assert not set(fresh_families) & {c["family"] for c in old["cases"]}
    train_families = sorted(
        {e["reset_family"] for e in entries} - set(fresh_families + dev_families)
    )
    pools = {}
    for family in sorted({e["reset_family"] for e in entries}):
        by_task = {}
        for e in entries:
            if e["reset_family"] != family:
                continue
            suite, name, episode = e["episode_id"].split("/")
            source = f"data/libero_raw/{suite}/{name}.hdf5"
            if (source, episode) not in used:
                by_task.setdefault(e["task"], []).append((e, source, episode))
        for values in by_task.values():
            rng.shuffle(values)
        tasks = sorted(by_task)
        rng.shuffle(tasks)
        pool = []
        while any(by_task.values()):
            for task in tasks:
                if by_task[task]:
                    pool.append(by_task[task].pop())
        pools[family] = pool
    cases = []

    def add(family, split):
        e, source, episode = pools[family].pop(0)
        with h5py.File(ROOT / source, "r") as f:
            length = len(f[f"data/{episode}/actions"])
        # Different manipulation phases; exactly one initial state per source episode.
        fraction = rng.choice([0.15, 0.35, 0.55, 0.75])
        cases.append(
            {
                "case_id": len(cases),
                "task_index": sorted(pools).index(family),
                "family": family,
                "task": e["task"],
                "split": split,
                "source": source,
                "episode": episode,
                "recorded_step": min(length - 1, int(length * fraction)),
                "takeover_fraction": fraction,
                "manual_seed": 10227 + len(cases),
                "source_policy_split": "train",
            }
        )

    for i in range(250):
        add(train_families[i % len(train_families)], "train")
    for i in range(36):
        add(train_families[i % len(train_families)], "id_development")
    for family in dev_families:
        for _ in range(12):
            add(family, "development")
    for family in fresh_families:
        for _ in range(12):
            add(family, "confirmation")
    assert len({(c["source"], c["episode"]) for c in cases}) == len(cases)
    assert all(c["source_policy_split"] == "train" for c in cases)
    plan = {
        "data_manual_seed": 10227,
        "cases": cases,
        "horizons": old["horizons"],
        "branches": old["branches"],
        "reset": old["reset"],
        "split_counts": dict(Counter(c["split"] for c in cases)),
        "training_family_counts": dict(
            Counter(c["family"] for c in cases if c["split"] == "train")
        ),
        "nested_training_case_ids": [
            c["case_id"] for c in cases if c["split"] == "train"
        ][:24],
        "fresh_confirmation_families": fresh_families,
        "held_out_manipulated_categories": ["ketchup", "salad dressing"],
        "object_claim_limit": "Target categories held out from predictor fitting; distractor assets may overlap. Not unseen objects for the pretrained policy.",
        "claim_limit": "Episode-disjoint and predictor-family-disjoint tests. Original policy has seen these task families; no claim of novel pretrained-policy tasks or new procedural geometry.",
        "excluded_policy_test": True,
        "excluded_previous_fork_episodes": True,
        "locked_protocol": {
            "seeds": [17, 29, 43],
            "conditions": [[24, 4000], [250, 400], [250, 4000]],
            "controls": [
                "actual actions",
                "hidden actions",
                "copy current",
                "wrong actions at evaluation",
            ],
            "confirmation_candidate": [250, 4000],
            "prediction_horizon_steps": 10,
            "no_action_expert_updates": True,
            "no_automatic_joint_training": True,
        },
    }
    OUT.mkdir(exist_ok=False)
    (OUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: plan[k]
                for k in [
                    "split_counts",
                    "training_family_counts",
                    "fresh_confirmation_families",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
