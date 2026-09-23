"""Summarize paired diagnostic effects without treating windows as replications."""

import argparse
import json
from collections import defaultdict

import numpy as np
from evaluate_stage1 import ROOT, inside, save_report


def paired_effect(values, layouts, manual_seed):
    values = np.asarray(values, dtype=float)
    if len(values) != len(layouts) or not len(values):
        raise ValueError("paired effects need one value per independent layout")
    # Keep the two geometry task proportions fixed while resampling resets.
    strata = defaultdict(list)
    for index, layout in enumerate(layouts):
        strata[layout.rsplit("_", 1)[0] if layout.startswith("G_") else "H"].append(
            index
        )
    rng = np.random.default_rng(manual_seed)
    draws = np.concatenate(
        [
            rng.choice(indices, (10000, len(indices)), replace=True)
            for indices in strata.values()
        ],
        axis=1,
    )
    low, high = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
    return {
        "paired_mean_difference": float(values.mean()),
        "interval_95": [float(low), float(high)],
        "independent_layouts": len(values),
        "manual_seed": manual_seed,
        "bootstrap_replicates": 10000,
        "scope": "These tasks/layouts; not generalization across all robot tasks.",
    }


def first_actions(directory):
    with (directory / "rollout.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if "requested_actions" in row:
                return np.asarray(row["requested_actions"])
    raise ValueError("rollout contains no policy action")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--manual-seed", type=int, default=317)
    args = parser.parse_args()
    run = inside(args.run)
    report = json.loads((run / "report.json").read_text())
    if not report["complete"]:
        raise ValueError("only a completed test scope can be summarized")
    rows = report["episodes"]
    kinds = {r["kind"] for r in rows}
    result = {
        "run": str(run.relative_to(ROOT)),
        "split": report["split"],
        "checkpoint": report["checkpoint"],
        "manual_seed": args.manual_seed,
        "groups": {},
        "limits": "Input interventions are not retrained architecture ablations; small intervals do not prove equivalence.",
    }
    for kind in sorted(kinds):
        group = [r for r in rows if r["kind"] == kind]
        policies = sorted({r["policy"] for r in group})
        layouts = sorted({r["layout"] for r in group})
        conditions = ("normal", "shifted") if kind == "G" else ("visible", "hidden")
        scores, tables = {}, {}
        expected = 1 if kind == "G" else 2
        for policy in policies:
            tables[policy] = {}
            for condition in conditions:
                subset = [
                    r
                    for r in group
                    if r["policy"] == policy and r["condition"] == condition
                ]
                if len(subset) != len(layouts) * expected:
                    raise ValueError("missing or duplicated diagnostic conditions")
                endpoint = lambda r, kind=kind: (
                    bool(r["success"])
                    if kind == "G"
                    else bool(r["first_choice"]["correct"])
                )
                for layout in layouts:
                    paired = [r for r in subset if r["layout"] == layout]
                    if len(paired) != expected:
                        raise ValueError("unbalanced layout")
                    if kind == "H" and {r["target_index"] for r in paired} != {0, 1}:
                        raise ValueError("history targets must be balanced")
                    scores[policy, condition, layout] = float(
                        np.mean([endpoint(r) for r in paired])
                    )
                tables[policy][condition] = {
                    "correct_or_successful": sum(endpoint(r) for r in subset),
                    "episodes": len(subset),
                    "rate": float(np.mean([endpoint(r) for r in subset])),
                    "mean_control_steps": float(
                        np.mean([r["policy_control_steps"] for r in subset])
                    ),
                }
                if kind == "H":
                    tables[policy][condition]["no_direction_choice"] = sum(
                        r["first_choice"]["chosen_index"] is None for r in subset
                    )
                    if report["split"] == "test" and any(
                        r["policy_calls"] != 1 or r["policy_control_steps"] != 5
                        for r in subset
                    ):
                        raise ValueError(
                            "formal history scores must precede any policy call after vision returns"
                        )
        effects = {}
        difficult, ordinary = conditions[1], conditions[0]
        intervention = "best_no_geometry" if kind == "G" else "best_current"
        for reference in ("native", intervention):
            if reference not in policies or "best" not in policies:
                continue
            delta = [
                scores["best", difficult, layout] - scores[reference, difficult, layout]
                for layout in layouts
            ]
            interaction = [
                delta[i]
                - (
                    scores["best", ordinary, layout]
                    - scores[reference, ordinary, layout]
                )
                for i, layout in enumerate(layouts)
            ]
            effects[f"best_minus_{reference}_difficult"] = paired_effect(
                delta, layouts, args.manual_seed
            )
            effects[f"best_minus_{reference}_difficulty_change"] = paired_effect(
                interaction, layouts, args.manual_seed
            )
        entry = {
            "endpoint": "task success"
            if kind == "G"
            else "correct first-five-step target direction",
            "conditions": tables,
            "paired_effects": effects,
        }
        if kind == "H":
            differences = {}
            for policy in policies:
                maxima = []
                for layout in layouts:
                    pair = sorted(
                        [
                            r
                            for r in group
                            if r["policy"] == policy
                            and r["layout"] == layout
                            and r["condition"] == "hidden"
                        ],
                        key=lambda r: r["target_index"],
                    )
                    actions = [
                        first_actions(run / policy / r["case_id"] / r["episode"])
                        for r in pair
                    ]
                    change = float(np.max(np.abs(actions[0] - actions[1])))
                    if policy in {"native", "best_current"} and change != 0:
                        raise ValueError(
                            "current-only paired hidden actions differ: inspect information/noise leakage"
                        )
                    maxima.append(change)
                differences[policy] = {
                    "mean_max_abs_action_change": float(np.mean(maxima)),
                    "max_abs_action_change": max(maxima),
                }
            entry["hidden_action_dependence"] = differences
        result["groups"][kind] = entry
    save_report(run / "analysis.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
