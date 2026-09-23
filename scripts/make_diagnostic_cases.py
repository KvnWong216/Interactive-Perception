"""Explicit development/test cases; randomness depends only on manual_seed."""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOAL_TASKS = ("open_the_middle_drawer_of_the_cabinet", "put_the_bowl_on_the_plate")
HISTORY_TASK = (
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["development", "test"], required=True)
    parser.add_argument("--kinds", nargs="+", choices=["G", "H"], default=["G", "H"])
    parser.add_argument(
        "--history-setup",
        choices=["two_bowls_color", "single_bowl_position"],
        default="two_bowls_color",
    )
    args = parser.parse_args()
    development = args.split == "development"
    seed = 117 if development else 217
    rng = np.random.default_rng(seed)
    label = (
        args.split
        if args.history_setup == "two_bowls_color"
        else f"{args.split}_single_bowl"
    )
    if set(args.kinds) != {"G", "H"}:
        label += "_" + "".join(sorted(set(args.kinds)))
    output = ROOT / "experiments/diagnostic_cases" / label
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    indices = [2] if development else list(range(10, 20))
    for task_index, task in enumerate(GOAL_TASKS):
        for index in indices:
            sign = int(rng.choice([-1, 1]))
            layout = f"G_{task_index}_{index:02d}"
            for condition in ("normal", "shifted"):
                cases.append(
                    {
                        "case_id": f"{layout}_{condition}",
                        "layout": layout,
                        "kind": "G",
                        "condition": condition,
                        "source": f"data/libero_raw/libero_goal/{task}_demo.hdf5",
                        "episode": f"demo_{index}",
                        "manual_seed": seed + task_index * 1000 + index,
                        "yaw_degrees": 0 if condition == "normal" else sign * 10,
                        "pitch_degrees": 0 if condition == "normal" else -sign * 5,
                        "max_policy_steps": 300,
                    }
                )
    indices = [2] if development else list(range(10, 20))
    for index in indices:
        for target in (0, 1):
            for condition in ("visible", "hidden"):
                layout = f"H_{index:02d}"
                cases.append(
                    {
                        "case_id": f"{layout}_target{target}_{condition}",
                        "layout": layout,
                        "kind": "H",
                        "condition": condition,
                        "source": f"data/libero_raw/libero_spatial/{HISTORY_TASK}_demo.hdf5"
                        if args.history_setup == "two_bowls_color"
                        else "data/libero_raw/libero_goal/put_the_bowl_on_the_plate_demo.hdf5",
                        "episode": f"demo_{index}",
                        "manual_seed": seed + 2000 + index,
                        "target_index": target,
                        "history_setup": args.history_setup,
                        # Formal H measures the decision before evidence returns.
                        "max_policy_steps": 300 if development else 5,
                    }
                )
    cases = [c for c in cases if c["kind"] in args.kinds]
    for case in cases:
        if not (ROOT / case["source"]).is_file():
            raise FileNotFoundError(case["source"])
        (output / f"{case['case_id']}.json").write_text(
            json.dumps(case, indent=2) + "\n"
        )
    manifest = {
        "manual_seed": seed,
        "split": args.split,
        "weight": "runs/stage1_s17_gpu7/best.pt",
        "cases": cases,
        "status": "development feasibility pending",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "path": str(output.relative_to(ROOT)),
                "cases": len(cases),
                "manual_seed": seed,
            }
        )
    )


if __name__ == "__main__":
    main()
