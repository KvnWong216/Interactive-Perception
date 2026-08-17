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
    parser.add_argument(
        "--temporal-v5",
        action="store_true",
        help=(
            "Encode the frozen six-point policy-visible RGB/proprioceptive "
            "history saved by OPEN_AND_OBSERVE."
        ),
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.cognitive_query_v4 and (not args.spatial_v2 or not args.query_prompt):
        raise ValueError(
            "--cognitive-query-v4 requires --spatial-v2 and --query-prompt"
        )
    if args.temporal_v5 and (
        not args.spatial_v2 or not args.cognitive_query_v4 or not args.query_prompt
    ):
        raise ValueError(
            "--temporal-v5 requires --spatial-v2, --cognitive-query-v4, and --query-prompt"
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

    def transformed_paths(row: dict, paths: dict):
        base = imageio.imread(root / paths["agentview"])
        wrist = imageio.imread(root / paths["wrist"])
        value = policy._input_transform(
            {
                "observation/image": base,
                "observation/wrist_image": wrist,
                "observation/state": np.zeros(8, dtype=np.float32),
                "prompt": args.query_prompt or row["final_prompt"],
            }
        )
        return jax.tree.map(jnp.asarray, value)

    def transformed(row: dict, phase: str):
        return transformed_paths(row, row["image_paths"][phase])

    def embed_items(items, *, label: str) -> np.ndarray:
        features = []
        for start in range(0, len(items), args.batch_size):
            batch_items = items[start : start + args.batch_size]
            real_size = len(batch_items)
            if real_size < args.batch_size:
                batch_items += [batch_items[-1]] * (args.batch_size - real_size)
            values = [item() for item in batch_items]
            batch = jax.tree.map(lambda *parts: jnp.stack(parts), *values)
            observation = model_lib.Observation.from_dict(batch)
            features.append(
                np.asarray(encode(observation), dtype=np.float32)[:real_size]
            )
            print(f"{label}: embedded {start + real_size}/{len(items)}", flush=True)
        return np.concatenate(features, axis=0)

    def embed_phase(phase: str) -> np.ndarray:
        items = [
            (lambda row=row, phase=phase: transformed(row, phase)) for row in rows
        ]
        return embed_items(items, label=phase)

    history = None
    robot_state_history = None
    if args.temporal_v5:
        history_lengths = {len(row.get("public_history", ())) for row in rows}
        if history_lengths != {6}:
            raise ValueError(
                f"temporal-v5 requires exactly six public history points, got {history_lengths}"
            )
        items = []
        for row in rows:
            for point in row["public_history"]:
                items.append(
                    lambda row=row, point=point: transformed_paths(
                        row, point["image_paths"]
                    )
                )
        flat_history = embed_items(items, label="public-history")
        history = flat_history.reshape(len(rows), 6, flat_history.shape[-1])
        robot_state_history = np.asarray(
            [
                [point["robot_state"] for point in row["public_history"]]
                for row in rows
            ],
            dtype=np.float32,
        )
        if robot_state_history.shape != (len(rows), 6, 8):
            raise ValueError(
                f"expected [N,6,8] public robot history, got {robot_state_history.shape}"
            )
        before = history[:, 0]
        after = history[:, -1]
    else:
        before = embed_phase("before")
        after = embed_phase("after")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
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
    if history is not None:
        payload["history_features"] = history
        payload["robot_state_history"] = robot_state_history
    np.savez_compressed(output, **payload)
    metadata = checkpoint / "params/_METADATA"
    report = {
        "schema_version": (
            "interactive-perception.pi05-transition-temporal-embedding.v5"
            if args.temporal_v5
            else "interactive-perception.pi05-transition-cognitive-query-embedding.v4"
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
        "temporal_v5": args.temporal_v5,
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
        "history_shape": list(history.shape) if history is not None else None,
        "robot_state_history_shape": (
            list(robot_state_history.shape)
            if robot_state_history is not None
            else None
        ),
        "dtype": str(before.dtype),
        "output": str(output.relative_to(root)),
        "output_sha256": digest(output),
        "encoder_inputs": (
            [
                "six stock agentview RGB frames",
                "six stock wrist RGB frames",
                "six policy-visible robot-state vectors",
                "prompt",
                "executed option role",
            ]
            if args.temporal_v5
            else ["paired stock agentview RGB", "paired stock wrist RGB", "prompt"]
        ),
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
