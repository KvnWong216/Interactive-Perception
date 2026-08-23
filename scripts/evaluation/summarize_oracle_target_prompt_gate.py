#!/usr/bin/env python3
"""Validate and summarize the oracle target-prompt screen or confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
V1_SCHEMA = "calibrated-interaction.oracle-target-prompt-gate.v1"
V2_SCHEMA = "calibrated-interaction.oracle-target-prompt-pilot.v2"
SCHEMAS = frozenset((V1_SCHEMA, V2_SCHEMA))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def wilson(successes: int, trials: int) -> list[float]:
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
    if len(left) != len(right):
        raise ValueError("paired vectors differ in length")
    left_only = sum(a and not b for a, b in zip(left, right, strict=True))
    right_only = sum(b and not a for a, b in zip(left, right, strict=True))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = (
            sum(
                math.comb(discordant, index)
                for index in range(min(left_only, right_only) + 1)
            )
            / 2**discordant
        )
        p_value = min(1.0, 2.0 * tail)
    return {
        "left_only": left_only,
        "right_only": right_only,
        "discordant_pairs": discordant,
        "exact_two_sided_p": p_value,
    }


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if config.get("schema_version") not in SCHEMAS:
        raise ValueError(f"unsupported oracle gate schema in {path}")
    return config


def keyframe_hashes(report: dict[str, Any], name: str) -> dict[str, str]:
    rows = [row for row in report["controller"]["keyframes"] if row["name"] == name]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one keyframe {name!r}")
    return dict(rows[0]["image_sha256"])


def validate_oracle_report(
    path: Path,
    *,
    config: dict[str, Any],
    seed: int,
    style: str,
) -> dict[str, Any]:
    report = json.loads(path.read_text())
    execution = config["execution"]
    expected = {
        "schema_version": (
            "piu.semantic-option.v2"
            if config["schema_version"] == V2_SCHEMA
            else "piu.semantic-option.v1"
        ),
        "claim_scope": "EVALUATOR_ONLY_ORACLE_UPPER_BOUND",
        "seed": seed,
        "role": execution["role"],
        "prompt": execution["prompt"],
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"{path}: {key}={report.get(key)!r}, expected {value!r}")
    controller = report["controller"]
    if controller["server_mode"] != "external":
        raise ValueError(f"{path}: oracle run did not use an external server")
    identity_path = ROOT / config["resource_contract"]["checkpoint_identity"]
    identity = json.loads(identity_path.read_text())
    expected_server_metadata = {
        "schema_version": config["resource_contract"]["identified_server_schema"],
        "policy_config": identity["policy_config"],
        "environment": "LIBERO",
        "checkpoint": identity["checkpoint"],
    }
    if controller["server_metadata"] != expected_server_metadata:
        raise ValueError(f"{path}: frozen policy server identity mismatch")
    if len(controller["online_oracle_inputs"]) != 2:
        raise ValueError(f"{path}: incomplete online oracle declaration")
    oracle = controller["oracle_visual_prompt"]
    if oracle["style"] != style:
        raise ValueError(f"{path}: oracle style mismatch")
    source = oracle["source_initial_state"]
    expected_state = (
        ROOT / config["source_run_root"] / f"seed{seed}" / "open_butter/final_state.npz"
    )
    if source["path"] != portable(expected_state) or source["sha256"] != sha256(
        expected_state
    ):
        raise ValueError(f"{path}: post-OPEN state provenance mismatch")
    audits = oracle["policy_call_audit"]
    if len(audits) != controller["policy_calls"] or not audits:
        raise ValueError(f"{path}: policy-call oracle audit is incomplete")
    visible_minimum = int(
        execution.get(
            "target_presence_minimum_pixels",
            execution.get("target_visible_pixels_minimum"),
        )
    )
    if max(audits[0]["visible_pixels"].values()) < visible_minimum:
        raise ValueError(f"{path}: target was not initially visible")
    for collection in (controller["keyframes"], oracle["keyframes"]):
        for row in collection:
            for view, relative in row["image_paths"].items():
                asset = ROOT / relative
                if row["image_sha256"][view] != sha256(asset):
                    raise ValueError(f"{path}: keyframe hash mismatch for {asset}")
    action_path = ROOT / controller["action_history"]
    evaluator = report["evaluator"]
    if evaluator["target_object"] != execution["target_object"]:
        raise ValueError(f"{path}: evaluator target mismatch")
    wrong_object = execution["wrong_object"]
    target_contact = bool(
        evaluator["objects"][execution["target_object"]]["grasp_contact_steps"] > 0
    )
    changed = audits[0].get("changed_rgb_pixels", {})
    return {
        "seed": seed,
        "style": style,
        "target_pick": bool(evaluator.get("target_pick_success", target_contact)),
        "target_grasp_contact": target_contact,
        "target_maximum_lift_m": float(
            evaluator["objects"][execution["target_object"]]["maximum_lift_m"]
        ),
        "target_destination": bool(evaluator["target_reached_destination"]),
        "target_destination_final": bool(evaluator["target_in_destination_final"]),
        "task_success": bool(evaluator["task_success"]),
        "wrong_object_contact": (
            evaluator["objects"][wrong_object]["grasp_contact_steps"] > 0
        ),
        "initial_visible_pixels": max(audits[0]["visible_pixels"].values()),
        "initial_changed_rgb_pixels": sum(int(value) for value in changed.values()),
        "report": {"path": portable(path), "sha256": sha256(path)},
        "action_history": {
            "path": portable(action_path),
            "sha256": sha256(action_path),
        },
        "initial_public_rgb_sha256": keyframe_hashes(report, "00_before"),
    }


def load_phase(
    config: dict[str, Any], *, phase: str, styles: list[str], seeds: list[int]
) -> list[dict[str, Any]]:
    root = ROOT / config["run_root"] / phase
    rows = []
    for style in styles:
        for seed in seeds:
            path = root / style / f"seed{seed}" / "report.json"
            rows.append(
                validate_oracle_report(path, config=config, seed=seed, style=style)
            )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        metric: rate([bool(row[metric]) for row in rows])
        for metric in (
            "target_pick",
            "target_destination",
            "target_destination_final",
            "task_success",
            "wrong_object_contact",
        )
    }


def aggregate_v2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        metric: rate([bool(row[metric]) for row in rows])
        for metric in (
            "target_grasp_contact",
            "target_destination",
            "target_destination_final",
            "task_success",
            "wrong_object_contact",
        )
    }


def select_style(
    config: dict[str, Any], rows: list[dict[str, Any]]
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    styles = [str(style) for style in config["screen"]["styles"]]
    if config["schema_version"] == V2_SCHEMA:
        by_style = {
            style: aggregate_v2([row for row in rows if row["style"] == style])
            for style in styles
        }
        changed = {
            style: sum(
                int(row["initial_changed_rgb_pixels"])
                for row in rows
                if row["style"] == style
            )
            for style in styles
        }
        scores = {
            style: (
                by_style[style]["target_grasp_contact"]["successes"],
                by_style[style]["target_destination_final"]["successes"],
                -by_style[style]["wrong_object_contact"]["successes"],
                -changed[style],
            )
            for style in styles
        }
        best = max(scores.values())
        winners = [style for style in styles if scores[style] == best]
        for style in styles:
            by_style[style]["initial_changed_rgb_pixels_total"] = changed[style]
            by_style[style]["selection_score"] = list(scores[style])
        return (winners[0] if len(winners) == 1 else None), by_style
    by_style = {
        style: aggregate([row for row in rows if row["style"] == style])
        for style in styles
    }
    preferred = [str(style) for style in config["screen"]["prefer_style_order"]]
    if set(preferred) != set(styles) or len(preferred) != len(styles):
        raise ValueError("prefer_style_order must contain each screened style once")
    preference = {style: index for index, style in enumerate(preferred)}
    selected = max(
        styles,
        key=lambda style: (
            by_style[style]["target_pick"]["successes"],
            by_style[style]["target_destination_final"]["successes"],
            -by_style[style]["wrong_object_contact"]["successes"],
            -preference[style],
        ),
    )
    return selected, by_style


def baseline_rows(config: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        path = (
            ROOT
            / config["source_run_root"]
            / f"seed{seed}"
            / config["baseline_condition"]
            / "report.json"
        )
        report = json.loads(path.read_text())
        if report["seed"] != seed or report["controller"]["online_oracle_inputs"]:
            raise ValueError(f"invalid public-input baseline {path}")
        evaluator = report["evaluator"]
        target = evaluator["objects"][config["execution"]["target_object"]]
        rows.append(
            {
                "seed": seed,
                "target_pick": bool(evaluator["target_pick_success"]),
                "target_grasp_contact": bool(target["grasp_contact_steps"] > 0),
                "target_maximum_lift_m": float(target["maximum_lift_m"]),
                "target_destination_final": bool(
                    evaluator["target_in_destination_final"]
                ),
                "initial_public_rgb_sha256": keyframe_hashes(report, "00_before"),
                "report": {"path": portable(path), "sha256": sha256(path)},
            }
        )
    return rows


def summarize(config_path: Path, phase: str, selected_style: str | None) -> dict:
    config = load_config(config_path)
    screen_styles = [str(style) for style in config["screen"]["styles"]]
    screen_seeds = [int(seed) for seed in config["screen"]["seeds"]]
    screen_rows = load_phase(
        config, phase="screen", styles=screen_styles, seeds=screen_seeds
    )
    selected, screen_aggregates = select_style(config, screen_rows)
    if selected is None:
        if phase == "confirmation":
            raise ValueError(
                "screen has an exact tie; collect another development group before "
                "confirmation"
            )
    elif selected_style is not None and selected_style != selected:
        raise ValueError(
            f"requested style {selected_style!r} differs from preregistered "
            f"screen selection {selected!r}"
        )
    common = {
        "schema_version": "calibrated-interaction.oracle-target-prompt-result.v1",
        "claim_scope": config["claim_scope"],
        "experiment": {
            "path": portable(config_path),
            "sha256": sha256(config_path),
        },
        "resource_contract": config["resource_contract"],
        "screen": {
            "rows": screen_rows,
            "aggregates_by_style": screen_aggregates,
            "selected_style": selected,
        },
        "online_oracle_input_count": 2,
        "formal_method_claim": False,
    }
    if phase == "screen":
        if selected is None:
            return {
                **common,
                "status": "SCREEN_TIED_NEEDS_MORE_DEVELOPMENT",
                "decision": "COLLECT_ANOTHER_SCREEN_GROUP",
            }
        return {
            **common,
            "status": "SCREEN_COMPLETE_AWAITING_CONFIRMATION",
            "decision": "RUN_PREREGISTERED_CONFIRMATION",
        }

    confirmation_seeds = [int(seed) for seed in config["confirmation"]["seeds"]]
    rows = load_phase(
        config,
        phase="confirmation",
        styles=[selected],
        seeds=confirmation_seeds,
    )
    baselines = baseline_rows(config, confirmation_seeds)
    for oracle, baseline in zip(rows, baselines, strict=True):
        if oracle["initial_public_rgb_sha256"] != baseline["initial_public_rgb_sha256"]:
            raise ValueError(
                f"seed {oracle['seed']}: oracle/baseline RGB is not paired"
            )
    if config["schema_version"] == V2_SCHEMA:
        aggregates_v2 = aggregate_v2(rows)
        comparison_v2 = exact_paired_binomial(
            [row["target_grasp_contact"] for row in rows],
            [row["target_grasp_contact"] for row in baselines],
        )
        return {
            **common,
            "status": "INDEPENDENT_DEVELOPMENT_PILOT_COMPLETE",
            "confirmation": {
                "selected_style": selected,
                "rows": rows,
                "aggregates": aggregates_v2,
                "public_baselines": baselines,
                "paired_target_grasp_contact_comparison": comparison_v2,
                "primary_estimand": config["confirmation"]["primary_estimand"],
                "evidence_grade": config["confirmation"]["evidence_grade"],
                "passed": None,
            },
            "decision": "DESIGN_PROSPECTIVE_FORMAL_TEST_FROM_PILOT",
            "automatic_method_branch": False,
            "decision_rationale": config["confirmation"]["rationale"],
            "formal_followup": config["formal_followup"],
        }
    aggregates = aggregate(rows)
    gate = config["confirmation"]["development_gate"]
    passed = aggregates["target_pick"]["successes"] >= int(
        gate["minimum_target_pick_successes"]
    ) and aggregates["wrong_object_contact"]["successes"] <= int(
        gate["maximum_wrong_object_contact_successes"]
    )
    comparison = exact_paired_binomial(
        [row["target_pick"] for row in rows],
        [row["target_pick"] for row in baselines],
    )
    return {
        **common,
        "status": "CONFIRMATION_COMPLETE",
        "confirmation": {
            "selected_style": selected,
            "rows": rows,
            "aggregates": aggregates,
            "public_baselines": baselines,
            "paired_target_pick_comparison": comparison,
            "development_gate": gate,
            "passed": passed,
        },
        "decision": (
            "TRAIN_PUBLIC_RGB_TARGET_BINDER"
            if passed
            else "SWITCH_TO_TARGETED_GRASP_AND_PLACE_PRIMITIVE"
        ),
        "decision_rationale": (
            config["confirmation"]["decision_if_passed"]
            if passed
            else config["confirmation"]["decision_if_failed"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/experiments/original_drawer_oracle_target_prompt_pilot_v2.yaml",
    )
    parser.add_argument("--phase", choices=("screen", "confirmation"), required=True)
    parser.add_argument("--style", choices=("box", "point", "spotlight"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"result is immutable: {output}")
    result = summarize(config_path, args.phase, args.style)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": portable(output),
                "status": result["status"],
                "selected_style": result["screen"]["selected_style"],
                "decision": result["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
