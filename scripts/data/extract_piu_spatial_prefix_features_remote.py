#!/usr/bin/env python3
"""Extract full frozen PIU prefix tokens through an identified external server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/infra"), str(ROOT / "src")]

from check_external_pi05 import validate_metadata, wait_for_endpoint

from interactive_perception.policy_client import (
    ObservationPacket,
    OpenPiWebsocketPolicy,
)
from piu.contracts import load_public_transitions, public_observation_sha256
from piu.spatial_prefix import (
    PrefixLayout,
    candidate_conditioned_prompt,
    libero_camera_to_label_view,
    validate_feature_arrays,
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


def executed_action(row: Any) -> str:
    history = row.public_action_history
    candidate = history.get("last_executed_candidate")
    if history.get("initial_observation") is True:
        if candidate is not None:
            raise ValueError("initial observation cannot declare an executed candidate")
        return "INITIAL_OBSERVATION"
    if not isinstance(candidate, dict):
        raise TypeError(f"{row.sample_id}: public history lacks executed candidate")
    identity = (
        " ".join(str(candidate.get("candidate_id", "")).split()),
        " ".join(str(candidate.get("primitive", "")).split()).upper(),
    )
    candidates = {
        (
            " ".join(str(item.get("candidate_id", "")).split()),
            " ".join(str(item.get("primitive", "")).split()).upper(),
        )
        for item in row.candidate_actions
    }
    if not all(identity) or identity not in candidates:
        raise ValueError(f"{row.sample_id}: executed candidate is absent or incomplete")
    return identity[1]


def packet_for(row: Any, time_step: str, prompt: str) -> ObservationPacket:
    observation = row.observations[time_step]
    images = observation["images"]
    wrist_key = "wrist" if "wrist" in images else "robot0_eye_in_hand"
    loaded = {}
    for name, key in (("agentview", "agentview"), ("wrist", wrist_key)):
        path = resolve(Path(images[key]["path"]))
        if sha256(path) != images[key]["sha256"]:
            raise ValueError(f"public image differs from its content hash: {path}")
        loaded[name] = np.asarray(Image.open(path).convert("RGB"))
        declared_pixel = images[key].get("pixel_sha256")
        if declared_pixel is not None:
            observed_pixel = hashlib.sha256(
                np.ascontiguousarray(loaded[name], dtype=np.uint8).tobytes()
            ).hexdigest()
            if observed_pixel != declared_pixel:
                raise ValueError(
                    f"public image pixels differ from canonical hash: {path}"
                )
    return ObservationPacket(
        image=loaded["agentview"],
        wrist_image=loaded["wrist"],
        state=np.asarray(observation["public_robot_state"], dtype=np.float32),
        prompt=prompt,
    )


def assert_layout(reference: dict[str, Any], value: dict[str, Any]) -> None:
    for name in ("camera_names", "tokens_per_camera"):
        if value[name] != reference[name]:
            raise ValueError("external spatial-prefix layout changed within one cache")
    for name in ("image_tokens", "prompt_tokens"):
        if value[name].shape != reference[name].shape:
            raise ValueError(
                "external spatial-prefix token shape changed within one cache"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--api-key")
    parser.add_argument(
        "--identity",
        type=Path,
        default=ROOT / "results/diagnostics/pi05_libero_checkpoint_identity_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    public_path = resolve(args.public)
    identity_path = resolve(args.identity)
    output = resolve(args.output)
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise FileExistsError("remote spatial-prefix outputs are immutable")
    rows = load_public_transitions(public_path)
    wait_for_endpoint(args.host, args.port, args.timeout)
    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port, api_key=args.api_key)
    identity = json.loads(identity_path.read_text())
    metadata = policy.server_metadata
    validate_metadata(metadata, identity)
    if "spatial_prefix_v1" not in metadata.get("capabilities", []):
        raise ValueError("identified endpoint lacks spatial_prefix_v1 capability")

    task_values: list[list[dict[str, Any]]] = []
    reference = None
    for row in rows:
        by_time = []
        for time_step in ("pre_interaction", "post_interaction"):
            value = policy.encode_spatial_prefix(packet_for(row, time_step, row.prompt))
            if reference is None:
                reference = value
            else:
                assert_layout(reference, value)
            by_time.append(value)
        task_values.append(by_time)
    if reference is None:
        raise RuntimeError("no public prefix request was issued")
    layout = PrefixLayout(reference["camera_names"], reference["tokens_per_camera"])
    camera_id, patch_xy = layout.patch_metadata()
    maximum_candidates = max(len(row.candidate_actions) for row in rows)
    candidate_values: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row_index, row in enumerate(rows):
        for candidate_index, candidate in enumerate(row.candidate_actions):
            conditioned = candidate_conditioned_prompt(row.prompt, candidate)
            for time_index, time_step in enumerate(
                ("pre_interaction", "post_interaction")
            ):
                value = policy.encode_spatial_prefix(
                    packet_for(row, time_step, conditioned)
                )
                assert_layout(reference, value)
                candidate_values[(row_index, candidate_index, time_index)] = value

    count = len(rows)
    time_count = 2
    image_count, width = reference["image_tokens"].shape
    prompt_count = reference["prompt_tokens"].shape[0]
    image = np.empty((count, time_count, image_count, width), dtype=np.float16)
    image_mask = np.empty((count, time_count, image_count), dtype=bool)
    prompt = np.empty((count, time_count, prompt_count, width), dtype=np.float16)
    prompt_mask = np.empty((count, time_count, prompt_count), dtype=bool)
    candidate_prompt = np.zeros(
        (count, maximum_candidates, time_count, prompt_count, width),
        dtype=np.float16,
    )
    candidate_prompt_mask = np.zeros(
        (count, maximum_candidates, time_count, prompt_count), dtype=bool
    )
    candidate_valid = np.zeros((count, maximum_candidates), dtype=bool)
    serialized = [
        json.dumps(dict(candidate), sort_keys=True, separators=(",", ":"))
        for row in rows
        for candidate in row.candidate_actions
    ]
    payload_width = max(1, max(map(len, serialized)))
    candidate_id_width = max(
        1,
        max(
            len(str(candidate["candidate_id"]))
            for row in rows
            for candidate in row.candidate_actions
        ),
    )
    primitive_width = max(
        1,
        max(
            len(str(candidate["primitive"]))
            for row in rows
            for candidate in row.candidate_actions
        ),
    )
    candidate_ids = np.full(
        (count, maximum_candidates), "", dtype=f"U{candidate_id_width}"
    )
    candidate_primitives = np.full(
        (count, maximum_candidates), "", dtype=f"U{primitive_width}"
    )
    candidate_payload = np.full(
        (count, maximum_candidates), "", dtype=f"U{payload_width}"
    )
    for row_index, row in enumerate(rows):
        for time_index, value in enumerate(task_values[row_index]):
            image[row_index, time_index] = value["image_tokens"]
            image_mask[row_index, time_index] = value["image_valid_mask"]
            prompt[row_index, time_index] = value["prompt_tokens"]
            prompt_mask[row_index, time_index] = value["prompt_valid_mask"]
        for candidate_index, candidate in enumerate(row.candidate_actions):
            candidate_valid[row_index, candidate_index] = True
            candidate_ids[row_index, candidate_index] = str(candidate["candidate_id"])
            candidate_primitives[row_index, candidate_index] = str(
                candidate["primitive"]
            ).upper()
            candidate_payload[row_index, candidate_index] = json.dumps(
                dict(candidate), sort_keys=True, separators=(",", ":")
            )
            for time_index in range(time_count):
                value = candidate_values[(row_index, candidate_index, time_index)]
                candidate_prompt[row_index, candidate_index, time_index] = value[
                    "prompt_tokens"
                ]
                candidate_prompt_mask[row_index, candidate_index, time_index] = value[
                    "prompt_valid_mask"
                ]
    arrays = {
        "image_tokens": image,
        "image_valid_mask": image_mask,
        "prompt_tokens": prompt,
        "prompt_valid_mask": prompt_mask,
        "patch_xy": patch_xy,
        "camera_id": camera_id,
        "sample_id": np.asarray([row.sample_id for row in rows]),
        "initial_state_group": np.asarray([row.initial_state_group for row in rows]),
        "split": np.asarray([row.split.value for row in rows]),
        "executed_action": np.asarray([executed_action(row) for row in rows]),
        "decision_observation_sha256": np.asarray(
            [
                public_observation_sha256(row.observations["post_interaction"])
                for row in rows
            ]
        ),
        "candidate_prompt_tokens": candidate_prompt,
        "candidate_prompt_valid_mask": candidate_prompt_mask,
        "candidate_valid_mask": candidate_valid,
        "candidate_id": candidate_ids,
        "candidate_primitive": candidate_primitives,
        "candidate_payload": candidate_payload,
    }
    validate_feature_arrays(arrays)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    report = {
        "schema_version": "piu.spatial-prefix-features.v1",
        "claim_scope": "FROZEN_FEATURE_CACHE_NOT_METHOD_RESULT",
        "dataset": {"path": portable(public_path), "sha256": sha256(public_path)},
        "output": {"path": portable(output), "sha256": sha256(output)},
        "server_identity": metadata,
        "identity_lock": {
            "path": portable(identity_path),
            "sha256": sha256(identity_path),
        },
        "layout": {
            "camera_names": list(layout.camera_names),
            "tokens_per_camera": list(layout.tokens_per_camera),
            "spans": {name: list(span) for name, span in layout.spans().items()},
            "camera_to_label_view": libero_camera_to_label_view(layout.camera_names),
            "spatial_coordinates_retained": True,
            "temporal_order": ["pre_interaction", "post_interaction"],
            "candidate_prompt_serialization": "piu.public-candidate-prompt.v1",
            "candidate_prompt_tokens_retained": True,
        },
        "arrays": {
            name: list(np.asarray(value).shape) for name, value in arrays.items()
        },
        "pooling": None,
        "online_oracle_inputs": [],
        "execution_location": "external_identified_server",
        "local_gpu_used": False,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
