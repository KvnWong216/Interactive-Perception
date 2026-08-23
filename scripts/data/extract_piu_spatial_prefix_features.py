#!/usr/bin/env python3
"""Extract full temporal PaliGemma prefix tokens on an external GPU host."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piu.contracts import load_public_transitions


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--external-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    public_path = args.public if args.public.is_absolute() else ROOT / args.public
    rows = load_public_transitions(public_path)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN",
                    "groups": len(rows),
                    "time_steps": ["pre_interaction", "post_interaction"],
                    "checkpoint_loaded": False,
                    "gpu_used": False,
                },
                indent=2,
            )
        )
        return
    if not args.external_gpu:
        raise ValueError(
            "full-prefix extraction is prohibited on the local 1500 MiB budget; "
            "run on the allocated external GPU and pass --external-gpu"
        )
    if args.checkpoint is None or args.output is None:
        raise ValueError("--checkpoint and --output are required outside dry-run")
    checkpoint = (
        args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise FileExistsError("feature outputs are immutable")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    import imageio.v2 as imageio
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from openpi.models import model as model_lib
    from openpi.models import vit as vit_lib
    from openpi.models.pi0 import make_attn_mask
    from openpi.policies import policy_config
    from openpi.shared import nnx_utils
    from openpi.training import config as train_config

    from piu.spatial_prefix import (
        PrefixLayout,
        libero_camera_to_label_view,
        validate_feature_arrays,
    )

    policy = policy_config.create_trained_policy(
        train_config.get_config("pi05_libero"), checkpoint
    )

    class FullPrefixEncoder(nnx.Module):
        def __init__(self, model):
            self.model = model

        def encode(self, observation):
            observation = model_lib.preprocess_observation(
                None, observation, train=False
            )
            tokens, mask, autoregressive = self.model.embed_prefix(observation)
            attention = make_attn_mask(mask, autoregressive)
            positions = jnp.cumsum(mask, axis=1) - 1
            (prefix, _), _ = self.model.PaliGemma.llm(
                [tokens, None], mask=attention, positions=positions
            )
            return prefix, mask

    encoder = FullPrefixEncoder(policy._model)
    encode = nnx_utils.module_jit(encoder.encode)

    def transformed(row, time_step: str):
        observation = row.observations[time_step]
        images = observation["images"]
        agentview = imageio.imread(ROOT / images["agentview"]["path"])
        wrist_key = (
            "wrist" if "wrist" in images else "robot0_eye_in_hand"
        )
        wrist = imageio.imread(ROOT / images[wrist_key]["path"])
        state = np.asarray(observation["public_robot_state"], dtype=np.float32)
        value = policy._input_transform(
            {
                "observation/image": agentview,
                "observation/wrist_image": wrist,
                "observation/state": state,
                "prompt": row.prompt,
            }
        )
        return jax.tree.map(jnp.asarray, value)

    first = transformed(rows[0], "pre_interaction")
    first_batch = jax.tree.map(lambda value: value[None], first)
    first_observation = model_lib.Observation.from_dict(first_batch)
    preprocessed = model_lib.preprocess_observation(
        None, first_observation, train=False
    )
    discovered_counts: dict[str, int] = {}
    for name in preprocessed.images:
        image_tokens, _ = policy._model.PaliGemma.img(
            preprocessed.images[name], train=False
        )
        discovered_counts[name] = int(image_tokens.shape[1])
    layout = PrefixLayout.from_counts(discovered_counts)
    camera_id, patch_xy = layout.patch_metadata()
    camera_to_label_view = libero_camera_to_label_view(layout.camera_names)
    image_token_count = layout.total_image_tokens

    image_by_time: list[np.ndarray] = []
    image_mask_by_time: list[np.ndarray] = []
    prompt_by_time: list[np.ndarray] = []
    prompt_mask_by_time: list[np.ndarray] = []
    for time_step in ("pre_interaction", "post_interaction"):
        image_parts: list[np.ndarray] = []
        image_mask_parts: list[np.ndarray] = []
        prompt_parts: list[np.ndarray] = []
        prompt_mask_parts: list[np.ndarray] = []
        for start in range(0, len(rows), args.batch_size):
            real_rows = rows[start : start + args.batch_size]
            real_size = len(real_rows)
            padded = real_rows + [real_rows[-1]] * (args.batch_size - real_size)
            values = [transformed(row, time_step) for row in padded]
            batch = jax.tree.map(lambda *items: jnp.stack(items), *values)
            observation = model_lib.Observation.from_dict(batch)
            prefix, mask = encode(observation)
            prefix_array = np.asarray(prefix, dtype=np.float16)[:real_size]
            mask_array = np.asarray(mask, dtype=bool)[:real_size]
            image_parts.append(prefix_array[:, :image_token_count])
            image_mask_parts.append(mask_array[:, :image_token_count])
            prompt_parts.append(prefix_array[:, image_token_count:])
            prompt_mask_parts.append(mask_array[:, image_token_count:])
            print(
                f"{time_step}: embedded {start + real_size}/{len(rows)}",
                flush=True,
            )
        image_by_time.append(np.concatenate(image_parts, axis=0))
        image_mask_by_time.append(np.concatenate(image_mask_parts, axis=0))
        prompt_by_time.append(np.concatenate(prompt_parts, axis=0))
        prompt_mask_by_time.append(np.concatenate(prompt_mask_parts, axis=0))

    arrays = {
        "image_tokens": np.stack(image_by_time, axis=1),
        "image_valid_mask": np.stack(image_mask_by_time, axis=1),
        "prompt_tokens": np.stack(prompt_by_time, axis=1),
        "prompt_valid_mask": np.stack(prompt_mask_by_time, axis=1),
        "patch_xy": patch_xy,
        "camera_id": camera_id,
        "sample_id": np.asarray([row.sample_id for row in rows]),
        "initial_state_group": np.asarray(
            [row.initial_state_group for row in rows]
        ),
        "split": np.asarray([row.split.value for row in rows]),
    }
    validate_feature_arrays(arrays)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)

    pi0_source_name = inspect.getsourcefile(policy._model.__class__)
    vit_source_name = inspect.getsourcefile(vit_lib)
    if pi0_source_name is None or vit_source_name is None:
        raise RuntimeError("cannot identify the OpenPI token-layout source files")
    pi0_source = Path(pi0_source_name)
    vit_source = Path(vit_source_name)
    openpi_root = next(
        (parent for parent in pi0_source.parents if (parent / ".git").exists()),
        None,
    )
    openpi_revision = (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=openpi_root, text=True
        ).strip()
        if openpi_root is not None
        else None
    )
    report = {
        "schema_version": "piu.spatial-prefix-features.v1",
        "claim_scope": "FROZEN_FEATURE_CACHE_NOT_METHOD_RESULT",
        "dataset": {
            "path": portable(public_path),
            "sha256": sha256(public_path),
        },
        "output": {
            "path": portable(output),
            "sha256": sha256(output),
        },
        "checkpoint_metadata_sha256": sha256(checkpoint / "params/_METADATA"),
        "openpi_revision": openpi_revision,
        "openpi_pi0_source_sha256": sha256(pi0_source),
        "openpi_vit_token_order_source_sha256": sha256(vit_source),
        "layout": {
            "camera_names": list(layout.camera_names),
            "tokens_per_camera": list(layout.tokens_per_camera),
            "spans": {name: list(span) for name, span in layout.spans().items()},
            "camera_to_label_view": camera_to_label_view,
            "spatial_coordinates_retained": True,
            "temporal_order": ["pre_interaction", "post_interaction"],
        },
        "arrays": {name: list(np.asarray(value).shape) for name, value in arrays.items()},
        "pooling": None,
        "online_oracle_inputs": [],
        "execution_location": "external_gpu",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
