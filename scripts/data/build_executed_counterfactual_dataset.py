#!/usr/bin/env python3
"""Build development-only effect supervision from retained physical forks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from calibrated_interaction.contracts import CandidateAction, EffectFactor
from calibrated_interaction.data import CounterfactualSample


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def frames(report: dict[str, Any], keyframe: str) -> list[str]:
    rows = [row for row in report["controller"]["keyframes"] if row["name"] == keyframe]
    if len(rows) != 1:
        raise ValueError(f"expected one {keyframe} keyframe")
    return [rows[0]["image_paths"][view] for view in ("agentview", "wrist")]


def maximum_visibility(evaluator: dict[str, Any], phase: str) -> int:
    values = evaluator["target_visibility_pixels"][phase]
    return max(int(value) for value in values.values()) if values else 0


def labels(
    *,
    candidate_id: str,
    target_location: str,
    evaluator: dict[str, Any] | None,
    drawer_open: bool,
    visibility_minimum: int,
) -> dict[str, bool]:
    initial_visible = bool(
        evaluator and maximum_visibility(evaluator, "initial") >= visibility_minimum
    )
    post_visible = bool(
        evaluator
        and max(
            maximum_visibility(evaluator, "after_subtask"),
            maximum_visibility(evaluator, "final"),
        )
        >= visibility_minimum
    )
    target_pick = bool(evaluator and evaluator["target_pick_success"])
    destination = bool(
        evaluator
        and evaluator.get(
            "target_in_destination_final",
            evaluator["target_reached_destination"],
        )
    )
    if candidate_id == "direct_requested_to_basket":
        execution_succeeded = destination
        task_relevant_change = target_pick or destination
        ambiguity_reduced = not initial_visible and (post_visible or target_pick)
        target_confirmed = post_visible or target_pick
    elif candidate_id == "open_middle_drawer":
        execution_succeeded = drawer_open
        task_relevant_change = target_location == "closed_container" and post_visible
        ambiguity_reduced = (
            target_location == "closed_container"
            and not initial_visible
            and post_visible
        )
        target_confirmed = post_visible
    elif candidate_id == "stop_unsupported":
        execution_succeeded = True
        task_relevant_change = False
        ambiguity_reduced = False
        target_confirmed = initial_visible
    else:
        raise ValueError(candidate_id)
    return {
        EffectFactor.EXECUTION_SUCCEEDED.value: execution_succeeded,
        EffectFactor.TASK_RELEVANT_CHANGE.value: task_relevant_change,
        EffectFactor.AMBIGUITY_REDUCED.value: ambiguity_reduced,
        EffectFactor.TARGET_CONFIRMED.value: target_confirmed,
        EffectFactor.CANDIDATE_REJECTED.value: False,
        EffectFactor.REGION_CONFIRMED_EMPTY.value: False,
    }


def load_report(config: dict[str, Any], seed: int, condition: str) -> tuple[Path, dict[str, Any]]:
    directory = config["conditions"][condition]["directory"]
    path = ROOT / config["run_root"] / f"seed{seed}" / directory / "report.json"
    return path, json.loads(path.read_text())


def build(
    config_path: Path,
    candidates_path: Path,
    relabel_path: Path,
    direct_cream_relabel_path: Path,
) -> list[dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text())
    candidate_config = yaml.safe_load(candidates_path.read_text())
    candidates = [CandidateAction.from_mapping(row) for row in candidate_config["candidates"]]
    candidate_rows = [candidate.to_dict() for candidate in candidates]
    relabel_artifact = json.loads(relabel_path.read_text())
    if not relabel_artifact.get("evaluator_only") or relabel_artifact.get("policy_rerun"):
        raise ValueError("task-specific relabel provenance is invalid")
    cream_open = {
        int(row["seed"]): row["evaluator"] for row in relabel_artifact["rows"]
    }
    direct_cream_artifact = json.loads(direct_cream_relabel_path.read_text())
    if (
        direct_cream_artifact.get("source_condition")
        != "direct_visible_cream_cheese"
        or not direct_cream_artifact.get("evaluator_only")
    ):
        raise ValueError("direct cream relabel provenance is invalid")
    direct_cream = {
        int(row["seed"]): row["evaluator"]
        for row in direct_cream_artifact["rows"]
    }
    visibility_minimum = int(config["thresholds"]["target_visible_pixels_minimum"])
    drawer_maximum = float(config["thresholds"]["drawer_open_joint_maximum"])
    drawer_joint = config["conditions"]["open_closed_drawer"]["tracked_joint"]
    rows: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        closed_path, closed = load_report(config, seed, "direct_closed_butter")
        open_path, opened = load_report(config, seed, "open_closed_drawer")
        cream_path, cream = load_report(config, seed, "direct_visible_cream_cheese")
        before = frames(closed, "00_before")
        # Paths differ by rollout, so equality is checked through retained hashes.
        before_hashes = [
            next(
                row["image_sha256"]
                for row in report["controller"]["keyframes"]
                if row["name"] == "00_before"
            )
            for report in (closed, opened, cream)
        ]
        if not before_hashes[0] == before_hashes[1] == before_hashes[2]:
            raise ValueError(f"seed {seed}: initial public RGB differs")
        drawer_open = (
            opened["evaluator"]["joints"][drawer_joint]["minimum"] <= drawer_maximum
        )
        prompt_specs = (
            {
                "name": "hidden_butter",
                "prompt": config["conditions"]["direct_closed_butter"]["prompt"],
                "location": "closed_container",
                "route": "open_middle_drawer",
                "direct_report": closed,
                "direct_path": closed_path,
                "direct_evaluator": closed["evaluator"],
                "open_evaluator": opened["evaluator"],
            },
            {
                "name": "visible_cream_cheese",
                "prompt": config["conditions"]["direct_visible_cream_cheese"]["prompt"],
                "location": "visible_workspace",
                "route": "direct_requested_to_basket",
                "direct_report": cream,
                "direct_path": cream_path,
                "direct_evaluator": direct_cream[int(seed)],
                "open_evaluator": cream_open[int(seed)],
            },
        )
        for prompt_spec in prompt_specs:
            for candidate_id in (
                "direct_requested_to_basket",
                "open_middle_drawer",
                "stop_unsupported",
            ):
                if candidate_id == "direct_requested_to_basket":
                    source_report = prompt_spec["direct_report"]
                    source_path = prompt_spec["direct_path"]
                    evaluator = prompt_spec["direct_evaluator"]
                    post = [
                        *frames(source_report, "04_subtask_end"),
                        *frames(source_report, "05_returned_home"),
                    ]
                    provenance = (
                        "executed_policy_rollout_final_state_relabel"
                        if prompt_spec["name"] == "visible_cream_cheese"
                        else "executed_policy_rollout"
                    )
                elif candidate_id == "open_middle_drawer":
                    source_report = opened
                    source_path = open_path
                    evaluator = prompt_spec["open_evaluator"]
                    post = [
                        *frames(opened, "04_subtask_end"),
                        *frames(opened, "05_returned_home"),
                    ]
                    provenance = "executed_policy_rollout_task_specific_relabel"
                else:
                    source_report = None
                    source_path = None
                    evaluator = prompt_spec["direct_report"]["evaluator"]
                    post = list(before)
                    provenance = "deterministic_null_option"
                effect_labels = labels(
                    candidate_id=candidate_id,
                    target_location=prompt_spec["location"],
                    evaluator=evaluator,
                    drawer_open=drawer_open,
                    visibility_minimum=visibility_minimum,
                )
                task_success = bool(
                    candidate_id == "direct_requested_to_basket"
                    and evaluator.get(
                        "target_in_destination_final",
                        evaluator["target_reached_destination"],
                    )
                )
                row = {
                    "schema_version": "calibrated-interaction.counterfactual-sample.v1",
                    "split": "development",
                    "episode_id": f"t01d-seed{seed}-{prompt_spec['name']}-{candidate_id}",
                    "initial_state_id": f"t01d-seed{seed}-closed",
                    "prompt": prompt_spec["prompt"],
                    "observation_frames": list(before),
                    "history": [],
                    "candidate_actions": candidate_rows,
                    "executed_candidate": candidate_id,
                    "post_action_frames": post,
                    "effect_labels": effect_labels,
                    "route_label": prompt_spec["route"],
                    "task_success": task_success,
                    "privileged_metadata_for_evaluation_only": {
                        "seed": int(seed),
                        "target_location": prompt_spec["location"],
                        "supervision_provenance": provenance,
                        "source_report": (
                            {
                                "path": portable(source_path),
                                "sha256": sha256(source_path),
                            }
                            if source_path is not None
                            else None
                        ),
                        "label_policy": "observable-effect-label-policy.v1",
                    },
                }
                CounterfactualSample.from_mapping(row)
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))


def manifest(
    *,
    output: Path,
    rows: list[dict[str, Any]],
    config_path: Path,
    candidates_path: Path,
    relabel_path: Path,
    direct_cream_relabel_path: Path,
) -> dict[str, Any]:
    support: Counter[tuple[str, str, bool]] = Counter()
    provenance: Counter[str] = Counter()
    for row in rows:
        for factor, value in row["effect_labels"].items():
            support[(row["executed_candidate"], factor, bool(value))] += 1
        provenance[
            row["privileged_metadata_for_evaluation_only"]["supervision_provenance"]
        ] += 1
    candidate_ids = sorted({row["executed_candidate"] for row in rows})
    factors = [factor.value for factor in EffectFactor]
    return {
        "schema_version": "calibrated-interaction.counterfactual-manifest.v1",
        "status": "development_only_not_held_out_test",
        "claim_scope": (
            "Two prompt counterfactuals and three candidate forks for ten fresh "
            "same-scene seeds. DIRECT and OPEN are executed; STOP is an exact "
            "deterministic null option."
        ),
        "rows": len(rows),
        "initial_state_groups": len({row["initial_state_id"] for row in rows}),
        "prompt_conditions": len({row["prompt"] for row in rows}),
        "branches_per_initial_state": 6,
        "dataset": {"path": portable(output), "sha256": sha256(output)},
        "sources": {
            "config": {"path": portable(config_path), "sha256": sha256(config_path)},
            "candidates": {
                "path": portable(candidates_path),
                "sha256": sha256(candidates_path),
            },
            "task_specific_relabel": {
                "path": portable(relabel_path),
                "sha256": sha256(relabel_path),
            },
            "direct_cream_final_relabel": {
                "path": portable(direct_cream_relabel_path),
                "sha256": sha256(direct_cream_relabel_path),
            },
        },
        "supervision_provenance_counts": dict(sorted(provenance.items())),
        "effect_label_support": {
            candidate: {
                factor: {
                    "false": support[(candidate, factor, False)],
                    "true": support[(candidate, factor, True)],
                }
                for factor in factors
            }
            for candidate in candidate_ids
        },
        "constant_unsupported_factors": [
            EffectFactor.CANDIDATE_REJECTED.value,
            EffectFactor.REGION_CONFIRMED_EMPTY.value,
        ],
        "online_oracle_input_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiments/original_drawer_paper_cycle_v2.yaml",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "configs/experiments/original_drawer_candidate_set.yaml",
    )
    parser.add_argument("--task-specific-relabel", type=Path, required=True)
    parser.add_argument("--direct-cream-relabel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    paths = [
        args.config,
        args.candidates,
        args.task_specific_relabel,
        args.direct_cream_relabel,
        args.output,
        args.manifest,
    ]
    (
        config_path,
        candidates_path,
        relabel_path,
        direct_cream_relabel_path,
        output,
        manifest_path,
    ) = [
        path if path.is_absolute() else ROOT / path for path in paths
    ]
    if not args.force:
        for path in (output, manifest_path):
            if path.exists():
                raise FileExistsError(path)
    rows = build(
        config_path,
        candidates_path,
        relabel_path,
        direct_cream_relabel_path,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rows)
    value = manifest(
        output=output,
        rows=rows,
        config_path=config_path,
        candidates_path=candidates_path,
        relabel_path=relabel_path,
        direct_cream_relabel_path=direct_cream_relabel_path,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": portable(output),
                "manifest": portable(manifest_path),
                "rows": len(rows),
                "initial_state_groups": value["initial_state_groups"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
