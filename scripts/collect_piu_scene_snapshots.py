#!/usr/bin/env python3
"""Collect scene-disjoint public snapshots and physically separate labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import _bootstrap  # noqa: F401,E402
from interactive_perception.anchors import visible_pixels  # noqa: E402
from interactive_perception.policy_client import build_observation  # noqa: E402
from interactive_perception.seed_registry import load_seed_registry  # noqa: E402


DUMMY_ACTION = [0.0] * 6 + [-1.0]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def expand_range(value: str) -> list[int]:
    start, end = (int(part) for part in value.split("-", maxsplit=1))
    if end < start:
        raise ValueError(f"descending seed range: {value}")
    return list(range(start, end + 1))


def rle_encode(mask: np.ndarray) -> dict:
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    counts = []
    last = 0
    length = 0
    for value in flat:
        current = int(value)
        if current == last:
            length += 1
        else:
            counts.append(length)
            length = 1
            last = current
    counts.append(length)
    return {"size": list(mask.shape), "counts": counts, "starts_with": 0}


def target_mask(env, observation: dict, *, camera: str, instance: str) -> np.ndarray:
    keys = [
        key
        for key in observation
        if key.startswith(camera) and "segmentation" in key
    ]
    if len(keys) != 1:
        raise KeyError(f"expected one {camera} segmentation image, got {keys}")
    raw = np.asarray(observation[keys[0]]).squeeze()
    mask = raw == env.instance_to_id[instance]
    # Match build_observation: 180-degree rotation and 224x224 policy resize.
    rotated = np.ascontiguousarray(mask[::-1, ::-1])
    return np.asarray(
        Image.fromarray(rotated.astype(np.uint8) * 255).resize(
            (224, 224), Image.Resampling.NEAREST
        )
    ) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "benchmarks/piu_v1/scene_disjoint_protocol.yaml",
    )
    parser.add_argument(
        "--split",
        choices=("prototype_train", "conformal_calibration", "clean_scene_disjoint_development"),
        required=True,
    )
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--limit-seeds", type=int)
    parser.add_argument("--allow-clean", action="store_true")
    parser.add_argument(
        "--clean-authorization",
        type=Path,
        default=ROOT / "results/calibration/piu_v1_replacement_clean_authorization.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--image-dir", type=Path)
    args = parser.parse_args()
    if not args.protocol.is_absolute():
        args.protocol = ROOT / args.protocol
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol.get("schema_version") != "interaction-uncertainty.piu-scene-disjoint-protocol.v1":
        raise ValueError("unsupported PIU scene protocol")
    specification = protocol["splits"][args.split]
    if not specification["open_now"]:
        if not args.allow_clean:
            raise PermissionError(
                f"{args.split} is closed: {specification.get('open_condition', 'not authorized')}"
            )
        from verify_piu_v1_clean_authorization import verify_authorization

        verify_authorization(args.clean_authorization)
    registry = load_seed_registry(ROOT / "benchmarks/rss_v1/seed_registry.yaml")
    block = registry[str(specification["seed_block"])]
    seeds = expand_range(str(specification["seeds"]))
    if not set(seeds) <= set(block.seeds):
        raise ValueError("protocol seeds are outside their authoritative registry block")
    if args.limit_seeds is not None:
        if args.limit_seeds < 1:
            raise ValueError("limit-seeds must be positive")
        seeds = seeds[: args.limit_seeds]

    stem = {
        "prototype_train": "prototype_train_v3",
        "conformal_calibration": "conformal_calibration_v3",
        "clean_scene_disjoint_development": "clean_scene_disjoint_v4",
    }[args.split]
    args.output = args.output or ROOT / f"data/piu_v1/{stem}.jsonl"
    args.labels = args.labels or ROOT / f"data/piu_v1/labels/{stem}.jsonl"
    args.image_dir = args.image_dir or ROOT / f"outputs/piu_v1/{stem}/images"
    for name in ("output", "labels", "image_dir"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    input_manifest = args.output.with_suffix(".manifest.json")
    label_manifest = args.labels.with_suffix(".manifest.json")
    for path in (args.output, args.labels, input_manifest, label_manifest):
        if path.exists():
            raise FileExistsError(f"immutable collection output already exists: {path}")

    from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv

    public_rows = []
    label_rows = []
    for scenario in specification["scenarios"]:
        bddl = ROOT / str(scenario["bddl"])
        if not bddl.is_file():
            raise FileNotFoundError(bddl)
        public_env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
        )
        evaluator_env = SegmentationRenderEnv(
            bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
        )
        try:
            for seed in seeds:
                public_env.seed(seed)
                observation = public_env.reset()
                for _ in range(args.wait_steps):
                    observation, _, _, _ = public_env.step(DUMMY_ACTION)
                packet = build_observation(observation, str(scenario["prompt"]))
                sample_id = f"{args.split}:{scenario['id']}:{seed:04d}"
                directory = args.image_dir / str(scenario["id"])
                directory.mkdir(parents=True, exist_ok=True)
                image_paths = {}
                image_hashes = {}
                for view, image in (("agentview", packet.image), ("wrist", packet.wrist_image)):
                    path = directory / f"seed{seed:04d}_{view}.png"
                    imageio.imwrite(path, image)
                    image_paths[view] = str(path.relative_to(ROOT))
                    image_hashes[view] = sha256_bytes(path.read_bytes())
                public_rows.append(
                    {
                        "schema_version": "interaction-uncertainty.piu-scene-snapshot.v1",
                        "sample_id": sample_id,
                        "sample_type": "INITIAL_TASK_BELIEF",
                        "split": args.split,
                        "seed_block": specification["seed_block"],
                        "seed": seed,
                        "scenario_id": scenario["id"],
                        "scenario_path": str(bddl.relative_to(ROOT)),
                        "prompt": scenario["prompt"],
                        "target": scenario["target"],
                        "destination": scenario["destination"],
                        # Only prompt-derived entities are policy inputs.  The
                        # scenario's interaction_target is an evaluator label:
                        # adding it here would reveal which closed container to
                        # search before the controller has acquired evidence.
                        "visual_queries": [scenario["target"], scenario["destination"]],
                        "candidate_actions": [
                            "DIRECT_ACT",
                            "OPEN_TO_INSPECT",
                            "ABSTAIN",
                        ],
                        "policy_inputs": {
                            "image_paths": image_paths,
                            "image_sha256": image_hashes,
                            "public_robot_state": [float(value) for value in packet.state],
                        },
                        "online_oracle_inputs": [],
                    }
                )

                # Evaluator-only replay occurs after the public snapshot is
                # finalized and is written to a separate label index.
                evaluator_env.seed(seed)
                evaluator_observation = evaluator_env.reset()
                for _ in range(args.wait_steps):
                    evaluator_observation, _, _, _ = evaluator_env.step(DUMMY_ACTION)
                target_pixels = {
                    camera: visible_pixels(
                        evaluator_env,
                        evaluator_observation,
                        camera=camera,
                        instance=str(scenario["target_instance"]),
                    )
                    for camera in ("agentview", "robot0_eye_in_hand")
                }
                target_masks = {
                    camera: rle_encode(
                        target_mask(
                            evaluator_env,
                            evaluator_observation,
                            camera=camera,
                            instance=str(scenario["target_instance"]),
                        )
                    )
                    for camera in ("agentview", "robot0_eye_in_hand")
                }
                prompt_resolvable = max(target_pixels.values()) >= 256
                expected_resolvable = scenario["target_location"] == "visible_workspace"
                if prompt_resolvable != expected_resolvable:
                    raise RuntimeError(
                        f"visibility contract mismatch for {sample_id}: "
                        f"registered={scenario['target_location']} pixels={target_pixels}"
                    )
                label_rows.append(
                    {
                        "schema_version": "interaction-uncertainty.piu-scene-label.v1",
                        "sample_id": sample_id,
                        "target_location": scenario["target_location"],
                        "preferred_action": scenario["preferred_action"],
                        "interaction_target": scenario.get("interaction_target"),
                        "target_instance": scenario["target_instance"],
                        "target_pixels": target_pixels,
                        "target_mask_policy_resolution_rle": target_masks,
                        "prompt_resolvable_initially": prompt_resolvable,
                        "minimum_resolvable_pixels": 256,
                        "label_source": "post-public-snapshot SegmentationRenderEnv replay",
                        "simulator_teacher_only": True,
                    }
                )
                print(f"[{len(public_rows)}] {sample_id}", flush=True)
        finally:
            public_env.close()
            evaluator_env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.labels.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in public_rows)
    )
    args.labels.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in label_rows)
    )
    common = {
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "protocol": str(args.protocol.relative_to(ROOT)),
        "protocol_sha256": digest(args.protocol),
        "split": args.split,
        "seed_block": specification["seed_block"],
        "seeds": seeds,
        "scenarios": [scenario["id"] for scenario in specification["scenarios"]],
        "scenario_paths": [scenario["bddl"] for scenario in specification["scenarios"]],
        "samples": len(public_rows),
    }
    input_report = {
        "schema_version": "interaction-uncertainty.piu-scene-snapshot-manifest.v1",
        **common,
        "dataset": str(args.output.relative_to(ROOT)),
        "dataset_sha256": digest(args.output),
        "label_file": str(args.labels.relative_to(ROOT)),
        "label_file_in_policy_index": False,
        "policy_inputs": ["agentview RGB", "wrist RGB", "public robot state", "full prompt"],
        "online_oracle_inputs": [],
        "class_counts_hidden_from_policy": dict(
            Counter(row["target_location"] for row in label_rows)
        ),
    }
    label_report = {
        "schema_version": "interaction-uncertainty.piu-scene-label-manifest.v1",
        **common,
        "labels": str(args.labels.relative_to(ROOT)),
        "labels_sha256": digest(args.labels),
        "source_dataset": str(args.output.relative_to(ROOT)),
        "source_dataset_sha256": digest(args.output),
        "evaluator_only_fields": [
            "instance segmentation",
            "target semantic ID",
            "interaction target",
        ],
    }
    input_manifest.write_text(json.dumps(input_report, indent=2) + "\n")
    label_manifest.write_text(json.dumps(label_report, indent=2) + "\n")
    print(json.dumps(input_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
