#!/usr/bin/env python3
"""Extract frozen pi0.5 PaliGemma prefix features for prompt-state evidence.

Unlike action-chunk statistics, these features are computed before the action
expert and therefore form a genuinely upstream `(RGB, prompt)` representation.
The script reads only policy-visible images and prompts saved by the calibration
collector.  Evaluator labels are copied into the output metadata after feature
extraction and never enter the frozen encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/calibration/t01_prompt_state_v1.jsonl")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("../checkpoints/checkpoints/pi05_libero"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/t01_prompt_state_v1/pi05_prefix_embeddings.npz")
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    from flax import nnx
    from openpi.models import model as model_lib
    from openpi.models.pi0 import make_attn_mask
    from openpi.policies import policy_config
    from openpi.shared import nnx_utils
    from openpi.training import config as train_config

    root = Path(__file__).resolve().parents[1]
    dataset = args.dataset if args.dataset.is_absolute() else root / args.dataset
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    output = args.output if args.output.is_absolute() else root / args.output
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line]
    if not rows:
        raise ValueError("embedding dataset is empty")

    policy = policy_config.create_trained_policy(
        train_config.get_config("pi05_libero"), checkpoint
    )

    class PrefixEncoder(nnx.Module):
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
            prompt_length = observation.tokenized_prompt.shape[1]
            image_length = prefix.shape[1] - prompt_length
            image_count = len(observation.images)
            tokens_per_image = image_length // image_count
            prompt_output = prefix[:, image_length:]
            prompt_mask = mask[:, image_length:]
            prompt_mean = jnp.sum(
                prompt_output * prompt_mask[..., None], axis=1
            ) / jnp.maximum(jnp.sum(prompt_mask, axis=1, keepdims=True), 1)
            last_prompt_index = jnp.max(
                jnp.where(
                    prompt_mask,
                    jnp.arange(prompt_length)[None, :],
                    -1,
                ),
                axis=1,
            )
            last_prompt = prompt_output[
                jnp.arange(prompt_output.shape[0]), last_prompt_index
            ]
            base_mean = jnp.mean(prefix[:, :tokens_per_image], axis=1)
            wrist_mean = jnp.mean(
                prefix[:, tokens_per_image : 2 * tokens_per_image], axis=1
            )
            return jnp.concatenate(
                [last_prompt, prompt_mean, base_mean, wrist_mean], axis=-1
            )

    encoder = PrefixEncoder(policy._model)  # frozen checkpoint; no gradients
    encode = nnx_utils.module_jit(encoder.encode)

    def transformed(row):
        base = imageio.imread(root / row["image_paths"]["agentview"])
        wrist = imageio.imread(root / row["image_paths"]["wrist"])
        value = policy._input_transform(
            {
                "observation/image": base,
                "observation/wrist_image": wrist,
                "observation/state": np.zeros(8, dtype=np.float32),
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
        embedded = np.asarray(encode(observation), dtype=np.float32)[:real_size]
        features.append(embedded)
        print(f"embedded {start + real_size}/{len(rows)}", flush=True)
    matrix = np.concatenate(features, axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=matrix,
        condition=np.asarray([row["condition"] for row in rows]),
        seed=np.asarray([row["seed"] for row in rows], dtype=np.int64),
        split=np.asarray([row["split"] for row in rows]),
        prompt=np.asarray([row["prompt"] for row in rows]),
        target_state=np.asarray([row["target_state"] for row in rows]),
        resolving_action=np.asarray([row["resolving_action"] for row in rows]),
    )
    report = {
        "schema_version": "interactive-perception.pi05-prefix-embedding.v1",
        "dataset": str(dataset.relative_to(root)),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "checkpoint": "pi05_libero",
        "feature": "last-prompt + prompt-mean + base-image-mean + wrist-image-mean after multimodal prefix attention",
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "output": str(output.relative_to(root)),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "encoder_inputs": ["agentview RGB", "wrist RGB", "prompt"],
        "encoder_oracle_inputs": [],
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
