#!/usr/bin/env python3
"""Validate and summarize the fresh original-drawer physical experiment matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def wilson(successes: int, trials: int) -> list[float]:
    """Return a two-sided 95% Wilson score interval."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (rate + z**2 / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def rate(values: list[bool]) -> dict[str, Any]:
    successes = sum(values)
    return {
        "successes": successes,
        "trials": len(values),
        "rate": successes / len(values),
        "wilson_95": wilson(successes, len(values)),
    }


def exact_paired_binomial(left: list[bool], right: list[bool]) -> dict[str, Any]:
    """Exact two-sided McNemar/binomial test over discordant pairs."""

    if len(left) != len(right):
        raise ValueError("paired vectors differ in length")
    left_only = sum(a and not b for a, b in zip(left, right, strict=True))
    right_only = sum(b and not a for a, b in zip(left, right, strict=True))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(left_only, right_only) + 1)
        ) / 2**discordant
        p_value = min(1.0, 2.0 * tail)
    return {
        "left_only": left_only,
        "right_only": right_only,
        "discordant_pairs": discordant,
        "exact_two_sided_p": p_value,
    }


def maximum_visibility(report: dict[str, Any], phase: str) -> int:
    values = report["evaluator"]["target_visibility_pixels"][phase]
    return max(int(value) for value in values.values()) if values else 0


def keyframe_hashes(report: dict[str, Any], name: str) -> dict[str, str]:
    rows = [row for row in report["controller"]["keyframes"] if row["name"] == name]
    if len(rows) != 1:
        raise ValueError(f"expected one {name} keyframe")
    return dict(rows[0]["image_sha256"])


def validate_report(
    path: Path, *, seed: int, condition: dict[str, Any]
) -> dict[str, Any]:
    report = json.loads(path.read_text())
    expected = {
        "schema_version": "piu.semantic-option.v1",
        "seed": seed,
        "role": condition["role"],
        "prompt": condition["prompt"],
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"{path}: {key}={report.get(key)!r}, expected {value!r}")
    evaluator = report["evaluator"]
    if evaluator["target_object"] != condition["target_object"]:
        raise ValueError(f"{path}: target object mismatch")
    if report["controller"]["online_oracle_inputs"]:
        raise ValueError(f"{path}: online oracle input leakage")
    for row in report["controller"]["keyframes"]:
        for view, relative in row["image_paths"].items():
            asset = ROOT / relative
            if sha256(asset) != row["image_sha256"][view]:
                raise ValueError(f"{path}: keyframe hash mismatch for {asset}")
    return report


def load_matrix(config_path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != "calibrated-interaction.paper-cycle.v2":
        raise ValueError("unsupported experiment schema")
    root = ROOT / config["run_root"]
    matrix: dict[int, dict[str, Any]] = {}
    for seed in config["seeds"]:
        matrix[seed] = {}
        for name, condition in config["conditions"].items():
            path = root / f"seed{seed}" / condition["directory"] / "report.json"
            matrix[seed][name] = {
                "path": path,
                "sha256": sha256(path),
                "report": validate_report(path, seed=seed, condition=condition),
            }
        initial_names = (
            "direct_closed_butter",
            "open_closed_drawer",
            "direct_visible_cream_cheese",
        )
        initial_hashes = [
            keyframe_hashes(matrix[seed][name]["report"], "00_before")
            for name in initial_names
        ]
        if any(value != initial_hashes[0] for value in initial_hashes[1:]):
            raise ValueError(f"seed {seed}: closed-state public RGB is not paired")
    return config, matrix


def summarize(config_path: Path, direct_cream_relabel_path: Path) -> dict[str, Any]:
    config, matrix = load_matrix(config_path)
    direct_cream_relabel = json.loads(direct_cream_relabel_path.read_text())
    if (
        direct_cream_relabel.get("source_condition")
        != "direct_visible_cream_cheese"
        or direct_cream_relabel.get("target_object") != "cream_cheese_1"
        or not direct_cream_relabel.get("evaluator_only")
    ):
        raise ValueError("invalid visible-cream task-specific relabel")
    cream_relabels = {
        int(row["seed"]): row for row in direct_cream_relabel["rows"]
    }
    threshold = config["thresholds"]
    visibility_minimum = int(threshold["target_visible_pixels_minimum"])
    drawer_maximum = float(threshold["drawer_open_joint_maximum"])
    per_seed = []
    for seed, conditions in matrix.items():
        closed = conditions["direct_closed_butter"]["report"]
        opened = conditions["open_closed_drawer"]["report"]
        after = conditions["direct_after_actual_open"]["report"]
        visible = conditions["direct_visible_cream_cheese"]["report"]
        cream_relabel = cream_relabels[seed]
        if (
            cream_relabel["source_report"]["sha256"]
            != conditions["direct_visible_cream_cheese"]["sha256"]
        ):
            raise ValueError(f"seed {seed}: cream relabel source mismatch")
        joint_name = config["conditions"]["open_closed_drawer"]["tracked_joint"]

        def record(
            report: dict[str, Any],
            wrong_object: str | None,
            evaluator_override: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            evaluator = evaluator_override or report["evaluator"]
            wrong_contacts = (
                int(evaluator["objects"][wrong_object]["grasp_contact_steps"])
                if wrong_object
                else 0
            )
            return {
                "target_visible_initial_pixels": maximum_visibility(report, "initial"),
                "target_visible_after_subtask_pixels": maximum_visibility(
                    report, "after_subtask"
                ),
                "target_visible_final_pixels": maximum_visibility(report, "final"),
                "target_visible_initial": (
                    maximum_visibility(report, "initial") >= visibility_minimum
                ),
                "target_visible_after_or_final": (
                    max(
                        maximum_visibility(report, "after_subtask"),
                        maximum_visibility(report, "final"),
                    )
                    >= visibility_minimum
                ),
                "target_pick": bool(evaluator["target_pick_success"]),
                "target_destination": bool(evaluator["target_reached_destination"]),
                "target_destination_final": bool(
                    evaluator.get(
                        "target_in_destination_final",
                        evaluator["target_reached_destination"],
                    )
                ),
                "scenario_task_success": bool(evaluator["task_success"]),
                "wrong_object_contact": wrong_contacts > 0,
                "wrong_object_contact_steps": wrong_contacts,
                "return_complete": report["controller"]["return_phase"] == "COMPLETE",
            }

        closed_row = record(closed, "cream_cheese_1")
        closed_row["task_success"] = closed_row["scenario_task_success"]
        open_row = record(opened, None)
        open_row.update(
            {
                "drawer_open": (
                    opened["evaluator"]["joints"][joint_name]["minimum"]
                    <= drawer_maximum
                ),
                "drawer_joint_minimum": opened["evaluator"]["joints"][joint_name][
                    "minimum"
                ],
                "information_acquired": (
                    open_row["target_visible_initial_pixels"] < visibility_minimum
                    and max(
                        open_row["target_visible_after_subtask_pixels"],
                        open_row["target_visible_final_pixels"],
                    )
                    >= visibility_minimum
                ),
            }
        )
        after_row = record(after, "cream_cheese_1")
        after_row["task_success"] = after_row["scenario_task_success"]
        visible_row = record(visible, "butter_1", cream_relabel["evaluator"])
        per_seed.append(
            {
                "seed": seed,
                "direct_closed_butter": closed_row,
                "open_closed_drawer": open_row,
                "direct_after_actual_open": after_row,
                "direct_visible_cream_cheese": visible_row,
                "reports": {
                    name: {
                        "path": portable(row["path"]),
                        "sha256": row["sha256"],
                        "action_history": {
                            "path": row["report"]["controller"]["action_history"],
                            "sha256": sha256(
                                ROOT
                                / row["report"]["controller"]["action_history"]
                            ),
                        },
                    }
                    for name, row in conditions.items()
                },
            }
        )

    def values(condition: str, metric: str) -> list[bool]:
        return [bool(row[condition][metric]) for row in per_seed]

    aggregates = {
        "direct_closed_butter": {
            metric: rate(values("direct_closed_butter", metric))
            for metric in (
                "target_visible_initial",
                "target_visible_after_or_final",
                "target_pick",
                "target_destination",
                "target_destination_final",
                "wrong_object_contact",
                "task_success",
                "return_complete",
            )
        },
        "open_closed_drawer": {
            metric: rate(values("open_closed_drawer", metric))
            for metric in ("drawer_open", "information_acquired", "return_complete")
        },
        "direct_after_actual_open": {
            metric: rate(values("direct_after_actual_open", metric))
            for metric in (
                "target_visible_initial",
                "target_visible_after_or_final",
                "target_pick",
                "target_destination",
                "target_destination_final",
                "wrong_object_contact",
                "task_success",
                "return_complete",
            )
        },
        "direct_visible_cream_cheese": {
            metric: rate(values("direct_visible_cream_cheese", metric))
            for metric in (
                "target_visible_initial",
                "target_visible_after_or_final",
                "target_pick",
                "target_destination",
                "target_destination_final",
                "wrong_object_contact",
                "return_complete",
            )
        },
    }
    closed_visible_after = [
        max(
            row["direct_closed_butter"]["target_visible_after_subtask_pixels"],
            row["direct_closed_butter"]["target_visible_final_pixels"],
        )
        >= visibility_minimum
        for row in per_seed
    ]
    information = values("open_closed_drawer", "information_acquired")
    comparisons = {
        "open_information_vs_direct_closed_visibility": {
            **exact_paired_binomial(information, closed_visible_after),
            "interpretation": (
                "OPEN yields task-target evidence more often than DIRECT from the "
                "same initial public RGB."
            ),
        },
        "wrong_object_contact_closed_vs_after_open": {
            **exact_paired_binomial(
                values("direct_closed_butter", "wrong_object_contact"),
                values("direct_after_actual_open", "wrong_object_contact"),
            ),
            "interpretation": (
                "Opening changes the executor failure mode, but does not produce a "
                "butter grasp."
            ),
        },
        "post_open_visibility_vs_target_pick": {
            **exact_paired_binomial(
                values("direct_after_actual_open", "target_visible_initial"),
                values("direct_after_actual_open", "target_pick"),
            ),
            "interpretation": (
                "The target is visibly available at the post-OPEN DIRECT input, "
                "but the executor does not convert that evidence into a grasp."
            ),
        },
        "visible_object_pick_vs_place": {
            **exact_paired_binomial(
                values("direct_visible_cream_cheese", "target_pick"),
                values("direct_visible_cream_cheese", "target_destination_final"),
            ),
            "interpretation": (
                "The visible-object executor reliably picks but does not reliably "
                "complete placement."
            ),
        },
    }
    information_rows = [
        row for row in per_seed if row["open_closed_drawer"]["information_acquired"]
    ]
    picked_visible_rows = [
        row
        for row in per_seed
        if row["direct_visible_cream_cheese"]["target_pick"]
    ]
    stage_conversion = {
        "post_open_target_pick_given_information": rate(
            [row["direct_after_actual_open"]["target_pick"] for row in information_rows]
        ),
        "post_open_task_success_given_information": rate(
            [row["direct_after_actual_open"]["task_success"] for row in information_rows]
        ),
        "visible_terminal_place_given_pick": rate(
            [
                row["direct_visible_cream_cheese"]["target_destination_final"]
                for row in picked_visible_rows
            ]
        ),
    }
    gates = {
        "visible_target_pick": {
            "required": int(threshold["visible_target_pick_gate_count"]),
            "observed": aggregates["direct_visible_cream_cheese"]["target_pick"][
                "successes"
            ],
        },
        "visible_target_place": {
            "required": int(threshold["visible_target_place_gate_count"]),
            "observed": aggregates["direct_visible_cream_cheese"][
                "target_destination_final"
            ]["successes"],
        },
        "post_open_butter_pick": {
            "required": int(threshold["post_open_butter_pick_gate_count"]),
            "observed": aggregates["direct_after_actual_open"]["target_pick"][
                "successes"
            ],
        },
    }
    for value in gates.values():
        value["passed"] = value["observed"] >= value["required"]
    return {
        "schema_version": "calibrated-interaction.paper-cycle-summary.v2",
        "experiment_id": config["id"],
        "scenario": config["scenario"],
        "claim_scope": config["claim_scope"],
        "config": {"path": portable(config_path), "sha256": sha256(config_path)},
        "task_specific_relabels": {
            "direct_visible_cream_cheese": {
                "path": portable(direct_cream_relabel_path),
                "sha256": sha256(direct_cream_relabel_path),
            }
        },
        "seeds": list(config["seeds"]),
        "thresholds": threshold,
        "online_oracle_input_count": 0,
        "aggregates": aggregates,
        "paired_comparisons": comparisons,
        "stage_conversion": stage_conversion,
        "preregistered_executor_gates": gates,
        "overall_executor_gate_passed": all(value["passed"] for value in gates.values()),
        "paper_claim": (
            "OPEN reliably executes and usually exposes the hidden target, but the "
            "frozen executor fails to exploit the acquired evidence; visible-object "
            "placement is also unreliable."
        ),
        "per_seed": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiments/original_drawer_paper_cycle_v2.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--direct-cream-relabel",
        type=Path,
        default=(
            ROOT / "results/method/original_drawer_direct_cream_final_relabel_v2.json"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    direct_cream_relabel = (
        args.direct_cream_relabel
        if args.direct_cream_relabel.is_absolute()
        else ROOT / args.direct_cream_relabel
    )
    if output.exists() and not args.force:
        raise FileExistsError(output)
    result = summarize(config, direct_cream_relabel)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": portable(output),
                "aggregates": result["aggregates"],
                "paired_comparisons": result["paired_comparisons"],
                "gates": result["preregistered_executor_gates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
