#!/usr/bin/env python3
"""Export evaluator-only pre/post target masks from retained transition states."""

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
from piu.contracts import load_evaluator_sidecars, load_public_transitions


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _segmentation(observation: dict[str, Any], camera: str):
    import numpy as np

    keys = [
        key
        for key in observation
        if key.startswith(camera) and "segmentation" in key
    ]
    if len(keys) != 1:
        raise KeyError(f"expected one {camera} segmentation image, got {keys}")
    values = np.asarray(observation[keys[0]]).squeeze()
    if values.ndim != 2:
        raise ValueError(f"unexpected segmentation shape {values.shape}")
    return values


def _policy_mask(mask):
    import numpy as np
    from PIL import Image

    rotated = np.ascontiguousarray(mask[::-1, ::-1])
    return np.asarray(
        Image.fromarray(rotated.astype(np.uint8) * 255).resize(
            (224, 224), Image.Resampling.NEAREST
        )
    ) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--scenario-config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--execution-location",
        choices=("external_simulator",),
        help="actual rendering is forbidden on the local 1500 MiB GPU budget",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    public_path = args.public if args.public.is_absolute() else ROOT / args.public
    evaluator_path = (
        args.evaluator if args.evaluator.is_absolute() else ROOT / args.evaluator
    )
    scenario_path = (
        args.scenario_config
        if args.scenario_config.is_absolute()
        else ROOT / args.scenario_config
    )
    public = load_public_transitions(public_path)
    sidecars = load_evaluator_sidecars(evaluator_path)
    by_id = {row.sample_id: row for row in sidecars}
    if {row.sample_id for row in public} != set(by_id):
        raise ValueError("public/evaluator sample sets differ")
    source_states = []
    for row in public:
        open_reference = by_id[row.sample_id].provenance.get("open_report", {})
        report_path = ROOT / str(open_reference.get("path", ""))
        if not report_path.is_file() or sha256(report_path) != open_reference.get(
            "sha256"
        ):
            raise ValueError(f"invalid OPEN provenance for {row.sample_id}")
        report = json.loads(report_path.read_text())
        state_path = ROOT / report["controller"]["opaque_state_transport"]
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        source_states.append((row, by_id[row.sample_id], report, state_path))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN",
                    "groups": len(source_states),
                    "source_states_verified": len(source_states),
                    "simulator_started": False,
                    "gpu_used": False,
                    "task_sufficiency_labels": "unsupported_null",
                },
                indent=2,
            )
        )
        return
    if args.execution_location != "external_simulator":
        raise ValueError(
            "mask rendering must run in the allocated external simulator environment"
        )
    if args.output is None:
        raise ValueError("--output is required outside dry-run")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("binding-label outputs are immutable")

    import bootstrap  # noqa: F401
    import numpy as np
    import yaml
    from libero.libero.envs import SegmentationRenderEnv

    scenario = yaml.safe_load(scenario_path.read_text())
    bddl = ROOT / scenario["scene"]["bddl"]
    target_object = str(scenario["evaluator"]["target_object"])
    environment = SegmentationRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
    )
    rows = []
    try:
        for public_row, sidecar, report, state_path in source_states:
            if max(sidecar.target_visible_pixels_pre.values()) != 0:
                raise ValueError("pre-interaction target mask is not empty")
            seed = int(report["seed"])
            environment.seed(seed)
            environment.reset()
            with np.load(state_path) as store:
                state = np.asarray(store["state"], dtype=float)
            observation = environment.set_init_state(state)
            target_id = int(environment.instance_to_id[target_object])
            post_masks = {}
            raw_pixels = {}
            for camera in ("agentview", "robot0_eye_in_hand"):
                mask = _segmentation(observation, camera) == target_id
                raw_pixels[camera] = int(mask.sum())
                post_masks[camera] = rle_encode(_policy_mask(mask))
            if raw_pixels != dict(sidecar.target_visible_pixels_post):
                raise ValueError(
                    f"post-state segmentation drift for {public_row.sample_id}: "
                    f"{raw_pixels} != {dict(sidecar.target_visible_pixels_post)}"
                )
            zeros = np.zeros((224, 224), dtype=bool)
            value = {
                "schema_version": "piu.binding-label.v1",
                "sample_id": public_row.sample_id,
                "initial_state_group": public_row.initial_state_group,
                "split": public_row.split.value,
                "target_mask_policy_resolution_rle": {
                    "pre_interaction": {
                        "agentview": rle_encode(zeros),
                        "robot0_eye_in_hand": rle_encode(zeros),
                    },
                    "post_interaction": post_masks,
                },
                "target_present_post": any(raw_pixels.values()),
                "task_sufficient_post": None,
                "executed_action": "OPEN",
                "simulator_teacher_only": True,
                "provenance": {
                    "state": {"path": portable(state_path), "sha256": sha256(state_path)},
                    "open_report": sidecar.provenance["open_report"],
                    "raw_target_pixels": raw_pixels,
                },
            }
            BindingLabel.from_mapping(value)
            rows.append(value)
    finally:
        environment.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    manifest = {
        "schema_version": "piu.binding-label-manifest.v1",
        "claim_scope": "EVALUATOR_ONLY_SUPERVISION",
        "labels": {"path": portable(output), "sha256": sha256(output)},
        "public": {"path": portable(public_path), "sha256": sha256(public_path)},
        "evaluator": {
            "path": portable(evaluator_path),
            "sha256": sha256(evaluator_path),
        },
        "scenario": {"path": portable(scenario_path), "sha256": sha256(scenario_path)},
        "groups": len(rows),
        "execution_location": args.execution_location,
        "policy_inputs_contain_masks": False,
        "task_sufficiency_supported": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
