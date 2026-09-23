"""Declare the expanded stage-one comparison before examining any new results."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/diagnostic_cases/stage1_expanded_s3217"
GPUS = (0, 4, 5, 6, 7)
TARGETS = (
    ("libero_goal/put_the_bowl_on_the_plate_demo", "akita_black_bowl_1"),
    ("libero_goal/put_the_wine_bottle_on_the_rack_demo", "wine_bottle_1"),
    ("libero_object/pick_up_the_milk_and_place_it_in_the_basket_demo", "milk_1"),
    ("libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo", "orange_juice_1"),
    ("libero_object/pick_up_the_cream_cheese_and_place_it_in_the_basket_demo", "cream_cheese_1"),
    ("libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo", "black_book_1"),
)


def main():
    records = json.loads((ROOT / "data/prepared/stage1_full/manifest.json").read_text())["episodes"]
    splits = {}
    for record in records:
        task = "/".join(record["episode_id"].split("/")[:2])
        splits.setdefault(task, set()).add(record["split"])
    assert len(splits) == 40 and all(len(s) == 1 for s in splits.values())
    tasks = sorted(splits)
    rng = np.random.default_rng(3217)
    grouped = {(gpu, kind): [] for gpu in GPUS for kind in ("H", "P", "S")}
    for task_index, task in enumerate(tasks):
        for reset in (22, 26, 34, 44):
            base = {"source": f"data/libero_raw/{task}.hdf5", "episode": f"demo_{reset}",
                    "manual_seed": 3217+1000*task_index+reset,
                    "our_training_split": next(iter(splits[task])), "task_index": task_index,
                    "max_policy_steps": 500 if task.startswith("libero_10/") else 300}
            sign = int(rng.choice((-1, 1)))
            layout = f"Sx_{task_index:02d}_{reset:02d}"
            for condition in ("normal", "camera", "lighting", "combined"):
                camera = condition in {"camera", "combined"}
                grouped[GPUS[task_index % 5], "S"].append({**base,
                    "kind": "S", "layout": layout, "case_id": f"{layout}_{condition}",
                    "condition": condition, "yaw_degrees": sign*20 if camera else 0,
                    "pitch_degrees": -sign*10 if camera else 0,
                    "light_scale": .55 if condition in {"lighting", "combined"} else 1.})
    for target_index, (task, target) in enumerate(TARGETS):
        task_index = tasks.index(task)
        for reset in (22, 26, 34, 44):
            base = {"source": f"data/libero_raw/{task}.hdf5", "episode": f"demo_{reset}",
                    "manual_seed": 3217+1000*task_index+reset,
                    "our_training_split": next(iter(splits[task])), "task_index": task_index,
                    "target_object": target}
            gpu = GPUS[target_index % 5]
            layout = f"Hx_{target_index:02d}_{reset:02d}"
            for world in (0, 1):
                for condition in ("visible", "hidden"):
                    grouped[gpu, "H"].append({**base, "kind": "H", "layout": layout,
                        "case_id": f"{layout}_target{world}_{condition}", "condition": condition,
                        "target_index": world, "history_setup": "single_object_position",
                        "max_policy_steps": 5})
            layout = f"Px_{target_index:02d}_{reset:02d}"
            for condition, offset in (("normal", 0.), ("minus_y", -.04), ("plus_y", .04)):
                grouped[gpu, "P"].append({**base, "kind": "P", "layout": layout,
                    "case_id": f"{layout}_{condition}", "condition": condition,
                    "target_offset_m": [0., offset, 0.],
                    "max_policy_steps": 500 if task.startswith("libero_10/") else 300})
    OUTPUT.mkdir(parents=True, exist_ok=False)
    shards = []
    for (gpu, kind), cases in grouped.items():
        directory = OUTPUT / f"gpu{gpu}_{kind}"
        directory.mkdir()
        for case in cases:
            assert (ROOT / case["source"]).is_file()
            (directory / f"{case['case_id']}.json").write_text(json.dumps(case, indent=2)+"\n")
        spec = {"manual_seed": 3217, "split": "expanded_exploratory", "modes": ["native", "best"],
                "checkpoint": "runs/stage1_s17_gpu7/best.pt", "physical_gpu": gpu, "kind": kind,
                "cases": cases, "status": "candidate scenes; model-blind audit before eligibility"}
        path = directory / "manifest.json"
        path.write_text(json.dumps(spec, indent=2)+"\n")
        shards.append({"physical_gpu": gpu, "kind": kind, "manifest": str(path.relative_to(ROOT)),
                       "candidate_cases": len(cases), "candidate_policy_episodes": 2*len(cases)})
    protocol = {"schema": "stage1-expanded-generalization-v1", "manual_seed": 3217,
        "analysis_manual_seed": 3317, "gpus": list(GPUS), "models": ["native", "best"],
        "checkpoint_update": 1500, "shards": shards,
        "tasks": [{"task_index": i, "source": s, "our_training_split": next(iter(splits[s]))} for i,s in enumerate(tasks)],
        "targets": [{"source": s, "target_object": o} for s,o in TARGETS],
        "budget": {"scene_candidates": 808, "policy_episodes_before_exclusions": 1616,
                   "gpu_hours_estimate": [15, 22], "five_gpu_wall_hours_estimate": [3, 6],
                   "new_output_limit_decimal_bytes": 100_000_000_000,
                   "repository_limit_decimal_bytes": 1_000_000_000_000},
        "primary_endpoints": {"S": "whole-task success within fixed horizon, split and macro task averages",
                              "P": "whole-task success after a real target translation; retain normal paired control",
                              "H": "target-aligned first-five-step displacement > 2 mm, before vision returns"},
        "eligibility": ["No model results are consulted to retain a scene.",
                        "Exclude entire paired layout if any condition fails calibration, physics isolation or fixture construction.",
                        "Target-specific probes reject target penetration beyond 2 mm; placement uses a local 4 cm envelope with original height preserved, not absolute world-z bounds.",
                        "History rejects target drift > 1 cm and any non-target physical-state difference > 1e-10.",
                        "Report all exclusions by object/task; a missing cohort is not zero success."],
        "interpretation": ["Stage-one weights and thresholds remain fixed; no training or checkpoint selection from test outcomes.",
                           "New resets are new to this evaluation, not necessarily absent from our training corpus.",
                           "Only 4 source tasks are in the original test split; expansion does not create 40 held-out tasks.",
                           "Pretrained MolmoAct2 may know all existing LIBERO meshes and tasks; no novel identity claim.",
                           "Previous 192 outcomes already inspected. This is a declared exploratory extension, not independent confirmatory replication.",
                           "Keep original and expanded rounds separate; additionally label any pooled analysis explicitly.",
                           "Only two systems; no component-specific causal attribution."]}
    assert sum(s["candidate_cases"] for s in shards) == 808
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), "candidates": 808, "policy_episodes": 1616}))


if __name__ == "__main__":
    main()
