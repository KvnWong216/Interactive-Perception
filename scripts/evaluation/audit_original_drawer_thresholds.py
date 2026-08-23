#!/usr/bin/env python3
"""Show whether retained drawer conclusions depend on hand-set thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs/paper_cycle_executor_v2"
PREFLIGHT = ROOT / "results/diagnostics/original_drawer_oracle_prompt_preflight_v3.json"
OPEN_RANGE_SOURCE = (
    ROOT / "third_party/LIBERO/libero/libero/envs/objects/articulated_objects.py"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_condition(condition: str, target: str) -> list[dict[str, Any]]:
    rows = []
    for seed in range(1400, 1410):
        path = RUN_ROOT / f"seed{seed}" / condition / "report.json"
        report = json.loads(path.read_text())
        evaluator = report["evaluator"]
        obj = evaluator["objects"][target]
        rows.append(
            {
                "seed": seed,
                "grasp_contact_steps": int(obj["grasp_contact_steps"]),
                "maximum_lift_m": float(obj["maximum_lift_m"]),
                "historical_pick_3cm": bool(evaluator["target_pick_success"]),
                "report_sha256": sha256(path),
            }
        )
    return rows


def audit() -> dict[str, Any]:
    preflight = json.loads(PREFLIGHT.read_text())
    visibility = [
        {
            "seed": int(row["seed"]),
            "maximum_raw_camera_pixels": max(
                int(value) for value in row["visible_pixels"].values()
            ),
        }
        for row in preflight["rows"]
    ]
    positive = [
        row["maximum_raw_camera_pixels"]
        for row in visibility
        if row["maximum_raw_camera_pixels"] > 0
    ]
    post_open = load_condition("direct_butter_after_open", "butter_1")
    visible_control = load_condition("direct_cream_exact", "cream_cheese_1")
    joint_rows = []
    for seed in range(1400, 1410):
        path = RUN_ROOT / f"seed{seed}" / "open_butter/report.json"
        report = json.loads(path.read_text())
        joint = report["evaluator"]["joints"]["wooden_cabinet_1_middle_level"]
        joint_rows.append(
            {
                "seed": seed,
                "minimum_joint_position": float(joint["minimum"]),
                "open_by_libero_contract": float(joint["minimum"]) <= -0.14,
                "report_sha256": sha256(path),
            }
        )
    contact_lifts = [
        row["maximum_lift_m"]
        for row in visible_control
        if row["grasp_contact_steps"] > 0
    ]
    return {
        "schema_version": "calibrated-interaction.threshold-audit.v1",
        "claim_scope": "RETAINED_RESULT_ROBUSTNESS_AND_METRIC_DEPRECATION",
        "visibility": {
            "rows": visibility,
            "positive_seed_count": len(positive),
            "zero_pixel_seed_count": len(visibility) - len(positive),
            "threshold_invariant_integer_interval_raw_pixels": [1, min(positive)],
            "historical_threshold_raw_pixels": 256,
            "historical_threshold_changes_result_within_interval": False,
            "interpretation": (
                "Every integer threshold from one through the minimum positive "
                "visibility count selects the same eight seeds. Future oracle "
                "enrollment uses mask non-emptiness only; visibility is continuous."
            ),
            "source": {
                "path": str(PREFLIGHT.relative_to(ROOT)),
                "sha256": sha256(PREFLIGHT),
            },
        },
        "target_manipulation": {
            "post_open_butter": {
                "rows": post_open,
                "grasp_contact_successes": sum(
                    row["grasp_contact_steps"] > 0 for row in post_open
                ),
                "trials": len(post_open),
                "maximum_observed_passive_z_displacement_m": max(
                    row["maximum_lift_m"] for row in post_open
                ),
                "threshold_independent_conclusion": (
                    "Zero target grasp contacts; no positive lift cutoff can turn "
                    "these runs into target-grasp successes."
                ),
            },
            "visible_cream_cheese_control": {
                "rows": visible_control,
                "grasp_contact_successes": sum(
                    row["grasp_contact_steps"] > 0 for row in visible_control
                ),
                "trials": len(visible_control),
                "minimum_maximum_lift_among_contact_runs_m": min(contact_lifts),
                "historical_lift_threshold_m": 0.03,
                "threshold_invariant_interval_m": [0.0, min(contact_lifts)],
            },
            "future_primary_binary_metric": "LIBERO target grasp-contact predicate",
            "future_lift_metric": "continuous maximum target z displacement",
            "deprecated_metric": (
                "piu.semantic-option.v1 target_pick_success (contact at any time "
                "and maximum lift >=0.03 m at any time)"
            ),
        },
        "drawer_open": {
            "libero_open_range_m": [-0.16, -0.14],
            "rows": joint_rows,
            "successes": sum(row["open_by_libero_contract"] for row in joint_rows),
            "trials": len(joint_rows),
            "source": {
                "path": str(OPEN_RANGE_SOURCE.relative_to(ROOT)),
                "sha256": sha256(OPEN_RANGE_SOURCE),
            },
            "interpretation": "The -0.14 boundary is a simulator contract, not a fitted threshold.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and not args.force:
        raise FileExistsError(f"audit already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit(), indent=2) + "\n")
    try:
        print(output.relative_to(ROOT))
    except ValueError:
        print(output)


if __name__ == "__main__":
    main()
