#!/usr/bin/env python3
"""Export evaluator-only PIU belief labels from one hash-bound state pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/infra"), str(ROOT / "src")]

from piu.binding_data import BindingLabel, rle_encode
from piu.contracts import load_public_transitions, public_observation_sha256


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def verified_artifact(value: dict[str, Any], *, name: str) -> tuple[Path, str]:
    path = resolve(Path(value["path"]))
    expected = str(value.get("sha256", ""))
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"{name} differs from its content hash")
    return path, str(value.get("state_key", "state"))


def source_pair(
    *, transition: Any, capture_report: Path | None, dispatch_receipt: Path | None
) -> tuple[tuple[Path, str], tuple[Path, str], str, dict[str, Any]]:
    if (capture_report is None) == (dispatch_receipt is None):
        raise ValueError("choose exactly one capture report or dispatch receipt")
    if capture_report is not None:
        report = json.loads(capture_report.read_text())
        if report.get("schema_version") != "piu.initial-observation-capture.v1":
            raise ValueError("unsupported initial-capture report")
        if report.get("sample_id") != transition.sample_id:
            raise ValueError("capture and public-transition sample IDs differ")
        if transition.public_action_history.get("initial_observation") is not True:
            raise ValueError("capture labels require an initial public transition")
        state = verified_artifact(report["source_state"], name="captured state")
        return (
            state,
            state,
            "INITIAL_OBSERVATION",
            {
                "capture_report": {
                    "path": portable(capture_report),
                    "sha256": sha256(capture_report),
                }
            },
        )
    receipt = json.loads(dispatch_receipt.read_text())
    if receipt.get("schema_version") != "piu.executor-dispatch.v1":
        raise ValueError("unsupported PIU dispatch receipt")
    if receipt.get("physical_action_dispatched") is not True:
        raise ValueError("post-action labels require a physical dispatch")
    history_receipt = transition.public_action_history.get("dispatch_receipt", {})
    if not isinstance(history_receipt, dict) or history_receipt.get("sha256") != sha256(
        dispatch_receipt
    ):
        raise ValueError("public transition is not bound to the dispatch receipt")
    candidate = transition.public_action_history.get("last_executed_candidate")
    if not isinstance(candidate, dict):
        raise TypeError("post-action public history lacks an executed candidate")
    primitive = " ".join(str(candidate.get("primitive", "")).split()).upper()
    if primitive != str(receipt.get("primitive", "")).upper():
        raise ValueError("public history and dispatch receipt actions differ")
    pre = receipt.get("source_initial_state_transport")
    post = receipt.get("final_state_transport")
    if not isinstance(pre, dict) or not isinstance(post, dict):
        raise TypeError("dispatch receipt lacks an opaque state pair")
    execution_path, _ = verified_artifact(
        receipt["execution_report"], name="execution report"
    )
    execution = json.loads(execution_path.read_text())
    if execution.get("schema_version") != "piu.semantic-option.v2":
        raise ValueError("belief labels require evaluator metric contract v2")
    if execution.get("controller", {}).get("online_oracle_inputs") != []:
        raise ValueError("public binder labels cannot originate from an oracle rollout")
    return (
        verified_artifact(pre, name="dispatch source state"),
        verified_artifact(post, name="dispatch final state"),
        primitive,
        {
            "dispatch_receipt": {
                "path": portable(dispatch_receipt),
                "sha256": sha256(dispatch_receipt),
            },
            "execution_report": {
                "path": portable(execution_path),
                "sha256": sha256(execution_path),
            },
        },
    )


def annotation_values(
    path: Path | None, *, sample_id: str, digest: str
) -> dict[str, Any]:
    result = {
        "task_sufficient_post": None,
        "region_confirmed_empty_post": None,
    }
    if path is None:
        return result
    value = json.loads(path.read_text())
    if value.get("schema_version") != "piu.post-observation-annotation.v1":
        raise ValueError("unsupported post-observation annotation schema")
    if (
        value.get("sample_id") != sample_id
        or value.get("public_observation_sha256") != digest
    ):
        raise ValueError("annotation is not bound to this public observation")
    if (
        value.get("annotator_blinded_to_method") is not True
        or not str(value.get("annotation_protocol", "")).strip()
    ):
        raise ValueError("annotation requires a declared blinded protocol")
    for name in result:
        item = value.get(name)
        if item is not None and not isinstance(item, bool):
            raise TypeError(f"annotation {name} must be boolean or null")
        result[name] = item
    result["provenance"] = {"path": portable(path), "sha256": sha256(path)}
    return result


def segmentation(observation: dict[str, Any], camera: str):
    import numpy as np

    keys = [
        key for key in observation if key.startswith(camera) and "segmentation" in key
    ]
    if len(keys) != 1:
        raise KeyError(f"expected one {camera} segmentation image, got {keys}")
    values = np.asarray(observation[keys[0]]).squeeze()
    if values.ndim != 2:
        raise ValueError(f"unexpected segmentation shape {values.shape}")
    return values


def policy_mask(mask):
    import numpy as np
    from PIL import Image

    rotated = np.ascontiguousarray(mask[::-1, ::-1])
    return (
        np.asarray(
            Image.fromarray(rotated.astype(np.uint8) * 255).resize(
                (224, 224), Image.Resampling.NEAREST
            )
        )
        > 0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-transition", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--scenario-config", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture-report", type=Path)
    source.add_argument("--dispatch-receipt", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--execution-location", choices=("external_simulator",))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in (
        "public_transition",
        "scenario_config",
        "capture_report",
        "dispatch_receipt",
        "annotation",
        "output",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, resolve(value))
    if args.output.exists():
        raise FileExistsError("post-observation belief labels are immutable")
    rows = [
        row
        for row in load_public_transitions(args.public_transition)
        if row.sample_id == args.sample_id
    ]
    if len(rows) != 1:
        raise ValueError("sample ID must select one public transition")
    transition = rows[0]
    pre, post, action, provenance = source_pair(
        transition=transition,
        capture_report=args.capture_report,
        dispatch_receipt=args.dispatch_receipt,
    )
    post_digest = public_observation_sha256(transition.observations["post_interaction"])
    annotation = annotation_values(
        args.annotation, sample_id=transition.sample_id, digest=post_digest
    )
    plan = {
        "schema_version": "piu.post-observation-label-export-plan.v1",
        "sample_id": transition.sample_id,
        "initial_state_group": transition.initial_state_group,
        "executed_action": action,
        "state_pair_verified": True,
        "task_sufficiency_annotation": annotation["task_sufficient_post"] is not None,
        "region_empty_annotation": annotation["region_confirmed_empty_post"]
        is not None,
        "external_simulator_required": True,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if args.execution_location != "external_simulator":
        raise ValueError(
            "label rendering must run in the external simulator environment"
        )

    import bootstrap  # noqa: F401
    import numpy as np
    import yaml
    from libero.libero.envs import SegmentationRenderEnv

    scenario = yaml.safe_load(args.scenario_config.read_text())
    bddl = resolve(Path(scenario["scene"]["bddl"]))
    target_object = str(scenario["evaluator"]["target_object"])
    env = SegmentationRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
    )
    masks = {}
    raw_pixels = {}
    try:
        env.reset()
        target_id = int(env.instance_to_id[target_object])
        for time_name, (state_path, state_key) in (
            ("pre_interaction", pre),
            ("post_interaction", post),
        ):
            with np.load(state_path) as store:
                state = np.asarray(store[state_key], dtype=float)
            observation = env.set_init_state(state)
            camera_masks = {}
            camera_pixels = {}
            for camera in ("agentview", "robot0_eye_in_hand"):
                mask = segmentation(observation, camera) == target_id
                camera_pixels[camera] = int(mask.sum())
                camera_masks[camera] = rle_encode(policy_mask(mask))
            masks[time_name] = camera_masks
            raw_pixels[time_name] = camera_pixels
        with np.load(post[0]) as store:
            post_state = np.asarray(store[post[1]], dtype=float)
        env.set_init_state(post_state)
        holding = bool(
            env.env._check_grasp(
                env.env.robots[0].gripper,
                env.env.objects_dict[target_object].contact_geoms,
            )
        )
        task_complete = bool(env.env._check_success())
    finally:
        env.close()
    value = {
        "schema_version": "piu.binding-label.v1",
        "sample_id": transition.sample_id,
        "initial_state_group": transition.initial_state_group,
        "split": transition.split.value,
        "target_mask_policy_resolution_rle": masks,
        "target_present_post": any(raw_pixels["post_interaction"].values()),
        "task_sufficient_post": annotation["task_sufficient_post"],
        "holding_requested_target_post": holding,
        "region_confirmed_empty_post": annotation["region_confirmed_empty_post"],
        "task_complete_post": task_complete,
        "executed_action": action,
        "simulator_teacher_only": True,
        "provenance": {
            **provenance,
            "public_transition": {
                "path": portable(args.public_transition),
                "sha256": sha256(args.public_transition),
                "post_observation_sha256": post_digest,
            },
            "scenario": {
                "path": portable(args.scenario_config),
                "sha256": sha256(args.scenario_config),
            },
            "state_pair": {
                "pre": {"path": portable(pre[0]), "sha256": sha256(pre[0])},
                "post": {"path": portable(post[0]), "sha256": sha256(post[0])},
            },
            "raw_target_pixels": raw_pixels,
            "annotation": annotation.get("provenance"),
            "task_complete_teacher": "simulator_task_predicate_post_observation",
            "holding_teacher": "simulator_target_grasp_predicate_post_observation",
        },
    }
    BindingLabel.from_mapping(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, sort_keys=True) + "\n")
    print(
        json.dumps(
            {**plan, "output": portable(args.output), "sha256": sha256(args.output)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
