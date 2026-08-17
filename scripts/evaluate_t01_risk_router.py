#!/usr/bin/env python3
"""Evaluate conformal robust-risk routing on the frozen 100-seed T01 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interactive_perception.active_risk import (  # noqa: E402
    ACT,
    DecisionLosses,
    EffectOutcome,
    ExpectedRiskPlanner,
    TargetBelief,
    TargetHypothesis,
    TargetState,
)
from interactive_perception.action_effect import EffectRegistry  # noqa: E402
from interactive_perception.capability_gate import exact_binomial_lower_bound  # noqa: E402

REMOVE = "REMOVE_OCCLUDER"
COST_GRID = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.925, 0.95, 1.0)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_action(row):
    return REMOVE if row["true_state"] == "MANIPULATION_ONLY" else ACT


def baseline_action(name, row):
    if name == "always_act":
        return ACT
    if name == "always_remove":
        return REMOVE
    if name == "drawer_state_rule":
        return ACT if row["condition"] == "open_visible_butter" else REMOVE
    raise ValueError(name)


def paired_test(treatment, baseline):
    treatment_only = sum(a and not b for a, b in zip(treatment, baseline, strict=True))
    baseline_only = sum(b and not a for a, b in zip(treatment, baseline, strict=True))
    discordant = treatment_only + baseline_only
    pvalue = 1.0 if discordant == 0 else float(
        binomtest(min(treatment_only, baseline_only), discordant, 0.5).pvalue
    )
    return {
        "treatment_only_correct": treatment_only,
        "baseline_only_correct": baseline_only,
        "discordant": discordant,
        "exact_two_sided_pvalue": pvalue,
    }


def route_rows(rows, cost, effect_entry):
    planner = ExpectedRiskPlanner(
        losses=DecisionLosses(
            false_commit=1.0,
            false_absent=1.0,
            act_execution_failure=1.0,
        ),
        act_reliability=1.0,
        horizon=1,
    )
    effect = effect_entry.planner_effect(
        resolves=("MANIPULATION_ONLY",),
        cost=cost,
        desired_outcome=EffectOutcome.REVEALED,
    )
    routed = []
    for row in rows:
        belief = TargetBelief(
            hypotheses=(
                TargetHypothesis("OBSERVED", TargetState.OBSERVED),
                TargetHypothesis(
                    "MANIPULATION_ONLY", TargetState.MANIPULATION_ONLY, REMOVE
                ),
                TargetHypothesis("ABSENT", TargetState.ABSENT),
            ),
            probabilities=(
                float(row["probabilities"]["OBSERVED"]),
                float(row["probabilities"]["MANIPULATION_ONLY"]),
                0.0,
            ),
        )
        decision = planner.robust_plan(
            belief,
            (effect,),
            conformal_labels=tuple(row["prediction_set"]),
        )
        routed.append(
            {
                "condition": row["condition"],
                "seed": row["seed"],
                "expected_action": expected_action(row),
                "selected_action": decision.selected_action,
                "correct": decision.selected_action == expected_action(row),
                "decision": decision.to_dict(),
            }
        )
    return routed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--effect",
        type=Path,
        default=ROOT / "results/calibration/t01_action_effect_v1.json",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT
        / "results/calibration/prompt_state_belief_t01_audit_100seed_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/t01_expected_risk_router_v1.json",
    )
    args = parser.parse_args()
    for name in ("audit", "effect", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    audit = json.loads(args.audit.read_text())
    if not audit["audit_passed"]:
        raise ValueError("prompt-state audit must pass before risk routing")
    rows = audit["rows"]
    effect_artifact = json.loads(args.effect.read_text())
    effect_entry = EffectRegistry.from_dict(effect_artifact).get(
        "t01_stock_middle_drawer_hidden_butter", REMOVE
    )
    reveal_reliability = effect_entry.lower_bound(EffectOutcome.REVEALED)
    sensitivity = []
    routed_by_cost = {}
    for cost in COST_GRID:
        routed = route_rows(rows, cost, effect_entry)
        routed_by_cost[str(cost)] = routed
        sensitivity.append(
            {
                "normalized_information_cost": cost,
                "correct": sum(item["correct"] for item in routed),
                "trials": len(routed),
                "accuracy": float(np.mean([item["correct"] for item in routed])),
                "action_counts": {
                    action: sum(item["selected_action"] == action for item in routed)
                    for action in (ACT, REMOVE, "NOT_FOUND")
                },
            }
        )
    declared_cost = 0.1
    selected_rows = routed_by_cost[str(declared_cost)]
    treatment = [item["correct"] for item in selected_rows]
    per_condition = {}
    for condition in sorted({item["condition"] for item in selected_rows}):
        subset = [item for item in selected_rows if item["condition"] == condition]
        successes = sum(item["correct"] for item in subset)
        lower = exact_binomial_lower_bound(successes, len(subset), 0.95)
        per_condition[condition] = {
            "successes": successes,
            "trials": len(subset),
            "accuracy": successes / len(subset),
            "one_sided_95_lower": lower,
            "required_reliability": 0.9,
            "passed": lower >= 0.9,
        }
    baselines = {}
    for name in ("always_act", "always_remove", "drawer_state_rule"):
        correct = [
            baseline_action(name, row) == expected_action(row) for row in rows
        ]
        baselines[name] = {
            "correct": sum(correct),
            "trials": len(correct),
            "accuracy": float(np.mean(correct)),
            "paired_vs_risk": paired_test(treatment, correct),
        }
    stable_passing_costs = [
        item["normalized_information_cost"]
        for item in sensitivity
        if all(
            exact_binomial_lower_bound(
                sum(
                    row["correct"]
                    for row in routed_by_cost[str(item["normalized_information_cost"])]
                    if row["condition"] == condition
                ),
                100,
                0.95,
            )
            >= 0.9
            for condition in per_condition
        )
    ]
    report = {
        "schema_version": "interactive-perception.expected-risk-router-audit.v1",
        "claim": "prompt/state-aligned ACT versus REMOVE_OCCLUDER selection on T01",
        "objective": "target observability; OBSERVED is a terminal success state",
        "non_claim": "final-task risk, scene-disjoint generalization, learned outcome criticism, or retrieval",
        "belief_audit": str(args.audit.relative_to(ROOT)),
        "belief_audit_sha256": digest(args.audit),
        "effect_artifact": str(args.effect.relative_to(ROOT)),
        "effect_artifact_sha256": digest(args.effect),
        "effect_reliability": {
            "REMOVE_OCCLUDER": reveal_reliability,
            "kind": "context-scoped one-sided 95% lower bound loaded from the frozen effect registry",
        },
        "terminal_act_reliability": {
            "value": 1.0,
            "meaning": "definition of the target-observability endpoint, not physical final-task capability",
        },
        "loss_contract": {
            "false_commit": 1.0,
            "false_absent": 1.0,
            "act_execution_failure": 1.0,
            "interpretation": "unit-normalized error losses; information cost is reported as a full sensitivity sweep",
        },
        "declared_information_cost": declared_cost,
        "per_condition": per_condition,
        "route_gate_passed": all(value["passed"] for value in per_condition.values()),
        "baselines": baselines,
        "cost_sensitivity": sensitivity,
        "stable_passing_costs_on_grid": stable_passing_costs,
        "analytic_switch_point": reveal_reliability,
        "analytic_note": "with unit loss and a singleton hidden belief, REMOVE wins iff cost < conservative reveal reliability",
        "rows_at_declared_cost": selected_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "route_gate_passed": report["route_gate_passed"],
                "per_condition": per_condition,
                "baselines": baselines,
                "stable_passing_costs_on_grid": stable_passing_costs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
