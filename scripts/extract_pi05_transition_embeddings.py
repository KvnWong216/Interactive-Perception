#!/usr/bin/env python3
"""Extract frozen pi0.5 prefix features for paired action-effect RGB frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/calibration/t01_action_effect_v1.jsonl"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("../checkpoints/checkpoints/pi05_libero"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/t01_action_effect_v1/pi05_transition_embeddings.npz"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--spatial-v2",
        action="store_true",
        help=(
            "Preserve a compact 8x8 map per real camera using a frozen 64-D "
            "random projection, plus prompt-to-patch cosine maps."
        ),
    )
    parser.add_argument(
        "--query-prompt",
        default=None,
        help=(
            "Optional frozen perception sub-task prompt. The action policy is "
            "unchanged; this only conditions the offline/online outcome encoder."
        ),
    )
    parser.add_argument(
        "--cognitive-query-v4",
        action="store_true",
        help=(
            "Append a frozen prompt-to-visual cross-attention readout per camera. "
            "This mirrors a cognitive query without training the VLA."
        ),
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.cognitive_query_v4 and (not args.spatial_v2 or not args.query_prompt):
        raise ValueError(
            "--cognitive-query-v4 requires --spatial-v2 and --query-prompt"
        )

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
        raise ValueError("transition dataset is empty")

    policy = policy_config.create_trained_policy(
        train_config.get_config("pi05_libero"), checkpoint
    )
    projection_rng = np.random.default_rng(260204600)
    projection = jnp.asarray(
        projection_rng.choice((-1.0, 1.0), size=(2048, 64)).astype(np.float32)
        / np.sqrt(64.0)
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
                jnp.where(prompt_mask, jnp.arange(prompt_length)[None, :], -1),
                axis=1,
            )
            last_prompt = prompt_output[
                jnp.arange(prompt_output.shape[0]), last_prompt_index
            ]
            base_mean = jnp.mean(prefix[:, :tokens_per_image], axis=1)
            wrist_mean = jnp.mean(
                prefix[:, tokens_per_image : 2 * tokens_per_image], axis=1
            )
            global_features = jnp.concatenate(
                [last_prompt, prompt_mean, base_mean, wrist_mean], axis=-1
            )
            if not args.spatial_v2:
                return global_features

            grid = int(np.sqrt(tokens_per_image))
            if grid * grid != tokens_per_image or grid % 2:
                raise ValueError(
                    f"expected an even square image-token grid, got {tokens_per_image}"
                )

            def compact_view(start: int) -> jax.Array:
                view = prefix[:, start : start + tokens_per_image]
                spatial = view.reshape(
                    view.shape[0], grid // 2, 2, grid // 2, 2, view.shape[-1]
                ).mean(axis=(2, 4))
                projected = jnp.einsum("bhwd,dr->bhwr", spatial, projection)
                normalized_view = view / jnp.maximum(
                    jnp.linalg.norm(view, axis=-1, keepdims=True), 1e-6
                )
                normalized_mean = prompt_mean / jnp.maximum(
                    jnp.linalg.norm(prompt_mean, axis=-1, keepdims=True), 1e-6
                )
                normalized_last = last_prompt / jnp.maximum(
                    jnp.linalg.norm(last_prompt, axis=-1, keepdims=True), 1e-6
                )
                mean_cosine = jnp.einsum(
                    "bsd,bd->bs", normalized_view, normalized_mean
                )
                last_cosine = jnp.einsum(
                    "bsd,bd->bs", normalized_view, normalized_last
                )
                return jnp.concatenate(
                    [
                        projected.reshape(projected.shape[0], -1),
                        mean_cosine,
                        last_cosine,
                    ],
                    axis=-1,
                )

            def cognitive_view(start: int) -> jax.Array:
                """Parameter-free cross-attention from the prompt to one view."""

                view = prefix[:, start : start + tokens_per_image]
                logits = jnp.einsum("bsd,bd->bs", view, prompt_mean) / jnp.sqrt(
                    float(view.shape[-1])
                )
                weights = jax.nn.softmax(logits, axis=1)
                return jnp.einsum("bs,bsd->bd", weights, view)

            parts = [
                global_features,
                compact_view(0),
                compact_view(tokens_per_image),
            ]
            if args.cognitive_query_v4:
                parts.extend(
                    [cognitive_view(0), cognitive_view(tokens_per_image)]
                )
            return jnp.concatenate(parts, axis=-1)

    encoder = PrefixEncoder(policy._model)
    encode = nnx_utils.module_jit(encoder.encode)

    def transformed(row: dict, phase: str):
        base = imageio.imread(root / row["image_paths"][phase]["agentview"])
        wrist = imageio.imread(root / row["image_paths"][phase]["wrist"])
        value = policy._input_transform(
            {
                "observation/image": base,
                "observation/wrist_image": wrist,
                "observation/state": np.zeros(8, dtype=np.float32),
                "prompt": args.query_prompt or row["final_prompt"],
            }
        )
        return jax.tree.map(jnp.asarray, value)

    def embed_phase(phase: str) -> np.ndarray:
        features = []
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            real_size = len(batch_rows)
            if real_size < args.batch_size:
                batch_rows += [batch_rows[-1]] * (args.batch_size - real_size)
            values = [transformed(row, phase) for row in batch_rows]
            batch = jax.tree.map(lambda *items: jnp.stack(items), *values)
            observation = model_lib.Observation.from_dict(batch)
            features.append(
                np.asarray(encode(observation), dtype=np.float32)[:real_size]
            )
            print(
                f"{phase}: embedded {start + real_size}/{len(rows)}", flush=True
            )
        return np.concatenate(features, axis=0)

    before = embed_phase("before")
    after = embed_phase("after")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        before_features=before,
        after_features=after,
        regime=np.asarray([row["regime"] for row in rows]),
        seed=np.asarray([row["seed"] for row in rows], dtype=np.int64),
        split=np.asarray([row["split"] for row in rows]),
        outcome=np.asarray([row["outcome"] for row in rows]),
        intended_outcome=np.asarray([row["intended_outcome"] for row in rows]),
        full_executor=np.asarray([row["full_executor"] for row in rows], dtype=bool),
        drawer_opened=np.asarray(
            [row["evaluator_only"]["drawer_opened"] for row in rows], dtype=bool
        ),
        after_max_target_pixels=np.asarray(
            [
                max(row["evaluator_only"]["after_target_pixels"].values())
                for row in rows
            ],
            dtype=np.int64,
        ),
        physical_outcome_subtype=np.asarray(
            [
                (
                    row["outcome"]
                    if row["outcome"] != "FAILED"
                    else "OPENED_UNOBSERVED"
                    if row["evaluator_only"]["drawer_opened"]
                    else "NO_EFFECT"
                )
                for row in rows
            ]
        ),
    )
    metadata = checkpoint / "params/_METADATA"
    report = {
        "schema_version": (
            "interactive-perception.pi05-transition-cognitive-query-embedding.v4"
            if args.cognitive_query_v4
            else "interactive-perception.pi05-transition-spatial-embedding.v2"
            if args.spatial_v2
            else "interactive-perception.pi05-transition-embedding.v1"
        ),
        "dataset": str(dataset.relative_to(root)),
        "dataset_sha256": digest(dataset),
        "checkpoint": "pi05_libero",
        "checkpoint_metadata_sha256": digest(metadata),
        "openpi_commit": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
        "feature_per_frame": (
            "8192-D global prefix plus two 8x8x64 frozen-projection maps, "
            "two 16x16 prompt-cosine maps per real camera, and one 2048-D "
            "prompt-attended vector per real camera"
            if args.cognitive_query_v4
            else "8192-D global prefix plus two 8x8x64 frozen-projection maps and "
            "two 16x16 prompt-cosine maps per real camera"
            if args.spatial_v2
            else "last-prompt + prompt-mean + base-image-mean + wrist-image-mean "
            "after multimodal prefix attention"
        ),
        "spatial_v2": args.spatial_v2,
        "cognitive_query_v4": args.cognitive_query_v4,
        "cognitive_query_readout": (
            "parameter-free scaled dot-product attention from prompt-mean to "
            "each camera's frozen visual tokens"
            if args.cognitive_query_v4
            else None
        ),
        "spatial_projection": (
            {
                "type": "fixed Rademacher",
                "seed": 260204600,
                "input_dimension": 2048,
                "output_dimension": 64,
                "scale": "1/sqrt(64)",
                "learned": False,
            }
            if args.spatial_v2
            else None
        ),
        "shape_per_phase": list(before.shape),
        "dtype": str(before.dtype),
        "output": str(output.relative_to(root)),
        "output_sha256": digest(output),
        "encoder_inputs": ["paired stock agentview RGB", "paired stock wrist RGB", "prompt"],
        "encoder_prompt": args.query_prompt or "dataset final_prompt",
        "encoder_oracle_inputs": [],
        "evaluator_only_diagnostics": [
            "drawer_opened",
            "after_max_target_pixels",
            "physical_outcome_subtype",
        ],
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
