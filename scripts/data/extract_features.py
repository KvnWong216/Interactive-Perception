#!/usr/bin/env python3
"""Extract frozen pi0.5 multimodal prefix features for PIU scene snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    for name in ("dataset", "checkpoint", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    report_path = args.output.with_suffix(".json")
    if args.output.exists() or report_path.exists():
        raise FileExistsError("prefix feature output is immutable")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line]
    if not rows:
        raise ValueError("snapshot dataset is empty")
    if any(row.get("online_oracle_inputs") for row in rows):
        raise ValueError("snapshot input index contains oracle inputs")

    from flax import nnx
    from openpi.models import model as model_lib
    from openpi.models.pi0 import make_attn_mask
    from openpi.policies import policy_config
    from openpi.shared import nnx_utils
    from openpi.training import config as train_config

    policy = policy_config.create_trained_policy(
        train_config.get_config("pi05_libero"), args.checkpoint
    )

    class PrefixEncoder(nnx.Module):
        def __init__(self, model):
            self.model = model

        def encode(self, observation):
            observation = model_lib.preprocess_observation(None, observation, train=False)
            tokens, mask, autoregressive = self.model.embed_prefix(observation)
            attention = make_attn_mask(mask, autoregressive)
            positions = jnp.cumsum(mask, axis=1) - 1
            (prefix, _), _ = self.model.PaliGemma.llm(
                [tokens, None], mask=attention, positions=positions
            )
            prompt_length = observation.tokenized_prompt.shape[1]
            image_length = prefix.shape[1] - prompt_length
            image_count = len(observation.images)
            tokens_per_image = image_length // image_count
            prompt_output = prefix[:, image_length:]
            prompt_mask = mask[:, image_length:]
            prompt_mean = jnp.sum(
                prompt_output * prompt_mask[..., None], axis=1
            ) / jnp.maximum(jnp.sum(prompt_mask, axis=1, keepdims=True), 1)
            last_index = jnp.max(
                jnp.where(
                    prompt_mask,
                    jnp.arange(prompt_length)[None, :],
                    -1,
                ),
                axis=1,
            )
            last_prompt = prompt_output[jnp.arange(prompt_output.shape[0]), last_index]
            agentview = jnp.mean(prefix[:, :tokens_per_image], axis=1)
            wrist = jnp.mean(
                prefix[:, tokens_per_image : 2 * tokens_per_image], axis=1
            )
            return jnp.concatenate(
                (last_prompt, prompt_mean, agentview, wrist), axis=-1
            )

    encode = nnx_utils.module_jit(PrefixEncoder(policy._model).encode)

    def transformed(row):
        images = row["policy_inputs"]["image_paths"]
        agentview = imageio.imread(ROOT / images["agentview"])
        wrist = imageio.imread(ROOT / images["wrist"])
        state = np.asarray(row["policy_inputs"]["public_robot_state"], dtype=np.float32)
        value = policy._input_transform(
            {
                "observation/image": agentview,
                "observation/wrist_image": wrist,
                "observation/state": state,
                "prompt": row["prompt"],
            }
        )
        return jax.tree.map(jnp.asarray, value)

    features = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        real_size = len(batch_rows)
        if real_size < args.batch_size:
            batch_rows = batch_rows + [batch_rows[-1]] * (args.batch_size - real_size)
        values = [transformed(row) for row in batch_rows]
        batch = jax.tree.map(lambda *items: jnp.stack(items), *values)
        observation = model_lib.Observation.from_dict(batch)
        features.append(np.asarray(encode(observation), dtype=np.float32)[:real_size])
        print(f"embedded {start + real_size}/{len(rows)}", flush=True)
    matrix = np.concatenate(features, axis=0)
    if matrix.shape != (len(rows), 8192):
        raise ValueError(f"unexpected prefix feature matrix {matrix.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=matrix,
        sample_id=np.asarray([row["sample_id"] for row in rows]),
        prompt=np.asarray([row["prompt"] for row in rows]),
        split=np.asarray([row["split"] for row in rows]),
        scenario_id=np.asarray([row["scenario_id"] for row in rows]),
        seed=np.asarray([row["seed"] for row in rows], dtype=np.int64),
    )
    report = {
        "schema_version": "interaction-uncertainty.piu-scene-prefix-features.v1",
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dataset": str(args.dataset.relative_to(ROOT)),
        "dataset_sha256": digest(args.dataset),
        "output": str(args.output.relative_to(ROOT)),
        "output_sha256": digest(args.output),
        "shape": list(matrix.shape),
        "checkpoint": "pi05_libero",
        "checkpoint_metadata_sha256": digest(args.checkpoint / "params/_METADATA"),
        "feature": "last prompt + prompt mean + agentview mean + wrist mean after frozen multimodal prefix attention",
        "encoder_inputs": ["agentview RGB", "wrist RGB", "public robot state", "full prompt"],
        "online_oracle_inputs": [],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
