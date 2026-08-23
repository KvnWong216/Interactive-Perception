#!/usr/bin/env python3
"""Execute every physical public candidate from one exact decision state."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.contracts import (
    PublicTransition,
    load_public_transitions,
    public_observation_sha256,
)
from piu.executor_bridge import SpatialReference, serialize_pi05_subtask
from piu.primitive_registry import (
    load_primitive_qualification_certificate,
    load_qualified_executor_map,
    validate_qualification_candidate_contract,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def slug(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")
    if not result:
        raise ValueError("candidate ID has no portable characters")
    return result


def keyframe(report: dict[str, Any], name: str) -> dict[str, Any]:
    rows = [row for row in report["controller"]["keyframes"] if row["name"] == name]
    if len(rows) != 1:
        raise ValueError(f"counterfactual report needs exactly one {name} keyframe")
    return rows[0]


def public_observation(frame: dict[str, Any]) -> dict[str, Any]:
    images = {}
    for camera, relative in frame["image_paths"].items():
        path = resolve(Path(relative))
        expected = str(frame["image_sha256"][camera])
        if sha256(path) != expected:
            raise ValueError("counterfactual public keyframe file hash differs")
        pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        pixel_digest = hashlib.sha256(
            np.ascontiguousarray(pixels).tobytes()
        ).hexdigest()
        declared = frame.get("image_pixel_sha256", {}).get(camera)
        if declared is not None and declared != pixel_digest:
            raise ValueError("counterfactual public keyframe pixel hash differs")
        images[str(camera)] = {
            "path": portable(path),
            "sha256": expected,
            "pixel_sha256": pixel_digest,
        }
    return {
        "images": images,
        "public_robot_state": [float(value) for value in frame["public_robot_state"]],
    }


def qualifications(path: Path) -> dict[str, Path]:
    return load_qualified_executor_map(path, repository_root=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-transition", type=Path, required=True)
    parser.add_argument("--decision-sample-id", required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--scenario-config", type=Path, required=True)
    parser.add_argument(
        "--baseline-registry",
        type=Path,
        default=ROOT / "configs/experiments/piu_baselines_v1.yaml",
    )
    parser.add_argument("--qualification-map", type=Path, required=True)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--state-key", default="state")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in (
        "decision_transition",
        "execution_plan",
        "scenario_config",
        "baseline_registry",
        "qualification_map",
        "source_state",
        "output_dir",
    ):
        setattr(args, name, resolve(getattr(args, name)))
    if not args.source_state.is_file():
        raise FileNotFoundError(args.source_state)
    rows = [
        row
        for row in load_public_transitions(args.decision_transition)
        if row.sample_id == args.decision_sample_id
    ]
    if len(rows) != 1:
        raise ValueError("decision sample must select one public transition")
    decision = rows[0]
    decision_digest = public_observation_sha256(
        decision.observations["post_interaction"]
    )
    execution_plan = json.loads(args.execution_plan.read_text())
    if (
        execution_plan.get("schema_version")
        != "piu.counterfactual-execution-plan.v1"
        or execution_plan.get("decision_sample_id") != decision.sample_id
        or execution_plan.get("initial_state_group") != decision.initial_state_group
        or execution_plan.get("split") != decision.split.value
        or execution_plan.get("decision_observation_sha256") != decision_digest
        or execution_plan.get("public_inputs_only") is not True
        or execution_plan.get("online_oracle_inputs") != []
    ):
        raise ValueError("counterfactual execution plan differs from decision state")
    if execution_plan.get("inputs", {}).get("public_transition", {}).get(
        "sha256"
    ) != sha256(args.decision_transition):
        raise ValueError("execution plan is bound to another transition artifact")
    plan_rows = execution_plan.get("candidates")
    if not isinstance(plan_rows, list):
        raise TypeError("counterfactual execution plan candidates must be a list")
    plan_by_id = {str(row.get("candidate_id")): row for row in plan_rows}
    decision_ids = {str(row["candidate_id"]) for row in decision.candidate_actions}
    if set(plan_by_id) != decision_ids or len(plan_by_id) != len(plan_rows):
        raise ValueError("execution plan must cover the exact public candidate set")
    registry = yaml.safe_load(args.baseline_registry.read_text())
    if registry.get("schema_version") != "piu.baseline-registry.v1":
        raise ValueError("unsupported PIU baseline registry")
    budgets = {
        str(name).upper(): int(value)
        for name, value in registry["shared_contract"]["option_step_budgets"].items()
    }
    policy_identity = resolve(Path(registry["shared_contract"]["checkpoint_identity"]))
    if not policy_identity.is_file():
        raise FileNotFoundError(policy_identity)
    certificates = qualifications(args.qualification_map)
    commands = []
    for candidate in decision.candidate_actions:
        candidate_id = str(candidate["candidate_id"])
        primitive = str(candidate["primitive"]).upper()
        planned = plan_by_id[candidate_id]
        if str(planned.get("primitive", "")).upper() != primitive or not isinstance(
            planned.get("eligible_for_execution"), bool
        ):
            raise ValueError("execution-plan candidate identity/eligibility differs")
        if not str(planned.get("eligibility_reason", "")).strip():
            raise ValueError("execution-plan candidate lacks an eligibility reason")
        if primitive in {"STOP", "REPORT_NOT_FOUND"}:
            if (
                planned["eligible_for_execution"] is not True
                or planned.get("structured_pi05_subtask") is not None
            ):
                raise ValueError("terminal candidate must be an eligible exact null")
            continue
        if planned["eligible_for_execution"] is False:
            if planned.get("structured_pi05_subtask") is not None:
                raise ValueError("ineligible candidate cannot carry an execution prompt")
            continue
        if primitive not in budgets:
            raise ValueError(
                f"physical candidate {primitive} has no frozen step budget"
            )
        if candidate_id not in certificates:
            raise ValueError(f"physical candidate {candidate_id} lacks qualification")
        subtask = planned.get("structured_pi05_subtask")
        if not isinstance(subtask, str) or not subtask.strip():
            raise ValueError("eligible physical candidate lacks an exact subtask")
        reference_rows = planned.get("spatial_references", ())
        if not isinstance(reference_rows, list):
            raise TypeError("execution-plan spatial references must be a list")
        references = tuple(
            SpatialReference(
                camera=str(row["camera"]),
                selected_patch_indices=tuple(
                    int(value) for value in row["selected_patch_indices"]
                ),
                x_interval=tuple(float(value) for value in row["x_interval"]),
                y_interval=tuple(float(value) for value in row["y_interval"]),
            )
            for row in reference_rows
        )
        expected_subtask = serialize_pi05_subtask(
            candidate,
            spatial_references=(
                references if primitive in {"PICK", "DIRECT"} else ()
            ),
        )
        if subtask != expected_subtask:
            raise ValueError("execution-plan subtask differs from deterministic bridge")
        certificate = load_primitive_qualification_certificate(
            certificates[candidate_id], repository_root=ROOT
        )
        validate_qualification_candidate_contract(
            certificate,
            candidate=candidate,
            spatial_reference_mode=(
                "calibrated_current_frame_boxes" if references else "none"
            ),
        )
        branch = args.output_dir / slug(candidate_id)
        command = [
            sys.executable,
            str(ROOT / "scripts/pipeline/execute.py"),
            "--scenario-config",
            str(args.scenario_config),
            "--role",
            f"COUNTERFACTUAL_{primitive}",
            "--prompt",
            subtask,
            "--initial-state",
            str(args.source_state),
            "--state-key",
            args.state_key,
            "--seed",
            str(args.seed),
            "--steps",
            str(budgets[primitive]),
            "--replan-steps",
            str(registry["shared_contract"]["replanning_steps"]),
            "--report-schema",
            "v2",
            "--external-server",
            "--expected-policy-identity",
            str(policy_identity),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--assets",
            str(branch / "assets"),
            "--work",
            str(branch / "work"),
            "--output",
            str(branch / "report.json"),
            "--final-state",
            str(branch / "final_state.npz"),
        ]
        if primitive in {"PICK", "PICK_TO_INSPECT"}:
            command.append("--preserve-grasp")
        commands.append((candidate_id, primitive, command, branch))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": "piu.counterfactual-collection-plan.v1",
                    "decision_sample_id": decision.sample_id,
                    "decision_observation_sha256": decision_digest,
                    "physical_branches": [
                        {
                            "candidate_id": candidate_id,
                            "primitive": primitive,
                            "command": command,
                        }
                        for candidate_id, primitive, command, _ in commands
                    ],
                    "null_branches": [
                        str(candidate["candidate_id"])
                        for candidate in decision.candidate_actions
                        if str(candidate["primitive"]).upper()
                        in {"STOP", "REPORT_NOT_FOUND"}
                    ],
                    "ineligible_physical_branches": [
                        {
                            "candidate_id": str(candidate["candidate_id"]),
                            "primitive": str(candidate["primitive"]).upper(),
                            "reason": plan_by_id[str(candidate["candidate_id"])][
                                "eligibility_reason"
                            ],
                        }
                        for candidate in decision.candidate_actions
                        if str(candidate["primitive"]).upper()
                        not in {"STOP", "REPORT_NOT_FOUND"}
                        and plan_by_id[str(candidate["candidate_id"])][
                            "eligible_for_execution"
                        ]
                        is False
                    ],
                    "external_pi05_only": True,
                    "local_pi05_loaded": False,
                },
                indent=2,
            )
        )
        return
    if args.output_dir.exists():
        raise FileExistsError("counterfactual collection directories are immutable")
    args.output_dir.mkdir(parents=True)
    branch_manifest = []
    by_candidate = {str(row["candidate_id"]): row for row in decision.candidate_actions}
    command_by_id = {
        candidate_id: (command, branch) for candidate_id, _, command, branch in commands
    }
    for candidate_id, candidate in by_candidate.items():
        primitive = str(candidate["primitive"]).upper()
        if primitive in {"STOP", "REPORT_NOT_FOUND"}:
            branch_manifest.append(
                {
                    "candidate_id": candidate_id,
                    "primitive": primitive,
                    "eligible_for_execution": True,
                    "outcome_transition": None,
                }
            )
            continue
        planned = plan_by_id[candidate_id]
        if planned["eligible_for_execution"] is False:
            branch_manifest.append(
                {
                    "candidate_id": candidate_id,
                    "primitive": primitive,
                    "eligible_for_execution": False,
                    "eligibility_reason": planned["eligibility_reason"],
                    "outcome_transition": None,
                }
            )
            continue
        command, branch = command_by_id[candidate_id]
        subprocess.run(command, cwd=ROOT, check=True)
        report_path = branch / "report.json"
        final_state_path = branch / "final_state.npz"
        report = json.loads(report_path.read_text())
        if not final_state_path.is_file():
            raise FileNotFoundError(final_state_path)
        if report.get("controller", {}).get("online_oracle_inputs") != []:
            raise ValueError("counterfactual public branch consumed an oracle input")
        action_path = resolve(Path(report["controller"]["action_history"]))
        outcome_sample = f"{decision.sample_id}--cf--{slug(candidate_id)}"
        outcome = {
            "schema_version": "piu.public-transition.v1",
            "sample_id": outcome_sample,
            "initial_state_group": decision.initial_state_group,
            "split": decision.split.value,
            "prompt": decision.prompt,
            "observations": {
                "pre_interaction": public_observation(keyframe(report, "00_before")),
                "post_interaction": public_observation(
                    keyframe(report, "05_returned_home")
                ),
            },
            "public_action_history": {
                "last_executed_candidate": dict(candidate),
                "low_level_actions": {
                    "path": portable(action_path),
                    "sha256": sha256(action_path),
                },
                "counterfactual_report": {
                    "path": portable(report_path),
                    "sha256": sha256(report_path),
                },
            },
            "candidate_actions": [dict(row) for row in decision.candidate_actions],
            "online_oracle_inputs": [],
        }
        parsed = PublicTransition.from_mapping(outcome)
        if (
            public_observation_sha256(parsed.observations["pre_interaction"])
            != decision_digest
        ):
            raise ValueError(
                "executed candidate did not reproduce the decision RGB state"
            )
        outcome_path = branch / "public_outcome.jsonl"
        outcome_path.write_text(json.dumps(outcome, sort_keys=True) + "\n")
        branch_manifest.append(
            {
                "candidate_id": candidate_id,
                "primitive": primitive,
                "eligible_for_execution": True,
                "outcome_sample_id": outcome_sample,
                "outcome_transition": {
                    "path": portable(outcome_path),
                    "sha256": sha256(outcome_path),
                },
                "execution_report": {
                    "path": portable(report_path),
                    "sha256": sha256(report_path),
                },
                "final_state": {
                    "path": portable(final_state_path),
                    "sha256": sha256(final_state_path),
                    "state_key": "state",
                },
                "qualification": {
                    "path": portable(certificates[candidate_id]),
                    "sha256": sha256(certificates[candidate_id]),
                },
            }
        )
    value = {
        "schema_version": "piu.counterfactual-branch-manifest.v1",
        "claim_scope": "EXECUTED_DATA_COLLECTION_NOT_METHOD_ROLLOUT",
        "decision_sample_id": decision.sample_id,
        "initial_state_group": decision.initial_state_group,
        "split": decision.split.value,
        "decision_observation_sha256": decision_digest,
        "source_state": {
            "path": portable(args.source_state),
            "sha256": sha256(args.source_state),
            "state_key": args.state_key,
        },
        "execution_plan": {
            "path": portable(args.execution_plan),
            "sha256": sha256(args.execution_plan),
        },
        "branches": branch_manifest,
        "online_oracle_inputs": [],
        "local_pi05_loaded": False,
        "paper_method_claim_allowed": False,
    }
    manifest_path = args.output_dir / "branch_manifest.json"
    manifest_path.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
