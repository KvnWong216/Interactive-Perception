"""Predeclare a native/best scene robustness matrix from local LIBERO assets.

Creates new physical camera/lighting combinations, not new object identities.
All stage-one validation/test tasks plus four explicit trained-task controls.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/diagnostic_cases/stage1_generalization_s1217"
TRAIN_CONTROLS = (
    "libero_goal/open_the_middle_drawer_of_the_cabinet_demo",
    "libero_goal/put_the_wine_bottle_on_the_rack_demo",
    "libero_object/pick_up_the_milk_and_place_it_in_the_basket_demo",
    "libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo",
)


def main():
    records = json.loads((ROOT / "data/prepared/stage1_full/manifest.json").read_text())["episodes"]
    by_source = {}
    for record in records:
        source = "/".join(record["episode_id"].split("/")[:2])
        by_source.setdefault(source, set()).add(record["split"])
    assert all(len(splits) == 1 for splits in by_source.values())
    selected = sorted(s for s, split in by_source.items() if split != {"train"}) + list(TRAIN_CONTROLS)
    assert len(selected) == len(set(selected)) == 12
    rng = np.random.default_rng(1217)
    cases, tasks = [], []
    for task_index, source in enumerate(selected):
        source_split = next(iter(by_source[source]))
        tasks.append({"task_index": task_index, "source": source, "our_training_split": source_split})
        assert (ROOT / "data/libero_raw" / (source + ".hdf5")).is_file()
        for index in (30, 40):
            episode_id = f"{source}/demo_{index}"
            assert any(r["episode_id"] == episode_id for r in records)
            sign = int(rng.choice([-1, 1]))
            layout = f"S_{task_index:02d}_{index:02d}"
            for condition in ("normal", "camera", "lighting", "combined"):
                camera = condition in {"camera", "combined"}
                cases.append({
                    "case_id": f"{layout}_{condition}", "layout": layout, "kind": "S",
                    "condition": condition, "source": f"data/libero_raw/{source}.hdf5",
                    "episode": f"demo_{index}", "manual_seed": 1217+task_index*1000+index,
                    "yaw_degrees": sign*20 if camera else 0,
                    "pitch_degrees": -sign*10 if camera else 0,
                    "light_scale": 0.55 if condition in {"lighting", "combined"} else 1.0,
                    "max_policy_steps": 500 if source.startswith("libero_10/") else 300,
                    "our_training_split": source_split, "task_index": task_index,
                })
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for case in cases:
        (OUTPUT / f"{case['case_id']}.json").write_text(json.dumps(case, indent=2)+"\n")
    manifest = {
        "schema": "stage1-paired-scene-generalization-v1", "manual_seed": 1217,
        "split": "exploratory_generalization", "modes": ["native", "best"],
        "weight": "runs/stage1_s17_gpu7/best.pt", "checkpoint_update": 1500,
        "tasks": tasks, "cases": cases, "independent_reset_layouts": 24,
        "episodes_per_model": 96, "total_policy_episodes": 192,
        "status": "declared_before_policy_results; scene audit required",
        "primary_endpoint": "task success by fixed task-specific horizon",
        "secondary_endpoints": ["paired degradation from normal", "failure-penalized steps", "failure case videos"],
        "analysis_manual_seed": 1317, "bootstrap_unit": "task_then_reset; keep four conditions and two policies paired",
        "limits": [
            "4 tasks withheld from our optimizer and checkpoint selection; 4 validation tasks influenced best selection; 4 trained task controls.",
            "Native pretrained model may have seen every LIBERO task and object; no pretrained-unseen claim.",
            "Objects are existing meshes; novel views/lighting and compositions do not constitute novel object identities.",
            "Train-control initial states were in our training dataset, not unseen-layout evidence.",
            "Only two resets per task in this first breadth pass; no strong per-task statistics.",
            "Camera views are world-frame transformations with recomputed extrinsics, not image warps.",
            "No individual attribution to geometry or history from this two-system comparison.",
        ],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), "tasks": tasks,
                      "cases": len(cases), "policy_episodes": 2*len(cases)}, indent=2))


if __name__ == "__main__":
    main()
