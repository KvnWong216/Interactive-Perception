#!/usr/bin/env python3
"""Serve frozen pi0.5 actions and upstream prefix features from one model copy.

The extension is deliberately read-only with respect to the checkpoint. A
``prefix`` request runs the PaliGemma multimodal prefix and does not split or
advance the base policy's action-sampling PRNG. Requests without an explicit
type retain the stock OpenPI action behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import nnx_utils
from openpi.training import config as train_config


class PrefixEncoder(nnx.Module):
    def __init__(self, model):
        self.model = model
        projection_rng = np.random.default_rng(260204600)
        self.projection = nnx.Variable(
            jnp.asarray(
                projection_rng.choice((-1.0, 1.0), size=(2048, 64)).astype(
                    np.float32
                )
                / np.sqrt(64.0)
            )
        )

    def _parts(self, observation):
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
        prompt_mean = jnp.sum(prompt_output * prompt_mask[..., None], axis=1) / jnp.maximum(
            jnp.sum(prompt_mask, axis=1, keepdims=True), 1
        )
        last_prompt_index = jnp.max(
            jnp.where(prompt_mask, jnp.arange(prompt_length)[None, :], -1), axis=1
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
        return (
            prefix,
            tokens_per_image,
            prompt_mean,
            last_prompt,
            global_features,
        )

    def encode_global(self, observation):
        return self._parts(observation)[-1]

    def encode_cognitive_spatial(self, observation):
        (
            prefix,
            tokens_per_image,
            prompt_mean,
            last_prompt,
            global_features,
        ) = self._parts(observation)
        grid = int(np.sqrt(tokens_per_image))
        if grid * grid != tokens_per_image or grid % 2:
            raise ValueError(
                f"expected an even square image-token grid, got {tokens_per_image}"
            )

        def compact_view(start: int):
            view = prefix[:, start : start + tokens_per_image]
            spatial = view.reshape(
                view.shape[0], grid // 2, 2, grid // 2, 2, view.shape[-1]
            ).mean(axis=(2, 4))
            projected = jnp.einsum(
                "bhwd,dr->bhwr", spatial, self.projection.value
            )
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

        def cognitive_view(start: int):
            view = prefix[:, start : start + tokens_per_image]
            logits = jnp.einsum("bsd,bd->bs", view, prompt_mean) / jnp.sqrt(
                float(view.shape[-1])
            )
            weights = jax.nn.softmax(logits, axis=1)
            return jnp.einsum("bs,bsd->bd", weights, view)

        return jnp.concatenate(
            [
                global_features,
                compact_view(0),
                compact_view(tokens_per_image),
                cognitive_view(0),
                cognitive_view(tokens_per_image),
            ],
            axis=-1,
        )


class PrefixActionPolicy:
    def __init__(self, policy):
        self.policy = policy
        encoder = PrefixEncoder(policy._model)
        self._encode_global = nnx_utils.module_jit(encoder.encode_global)
        self._encode_cognitive_spatial = nnx_utils.module_jit(
            encoder.encode_cognitive_spatial
        )

    def infer(self, raw: dict) -> dict:
        request = str(raw.get("__request_type", "action"))
        feature_schema = str(raw.get("__feature_schema", "global_v1"))
        payload = {
            key: value
            for key, value in raw.items()
            if key not in {"__request_type", "__feature_schema"}
        }
        if request == "action":
            return self.policy.infer(payload)
        if request != "prefix":
            raise ValueError(f"unknown request type {request!r}")
        started = time.monotonic()
        inputs = jax.tree.map(lambda value: value, payload)
        inputs = self.policy._input_transform(inputs)
        inputs = jax.tree.map(lambda value: jnp.asarray(value)[np.newaxis, ...], inputs)
        observation = model_lib.Observation.from_dict(inputs)
        if feature_schema == "global_v1":
            features = np.asarray(
                self._encode_global(observation), dtype=np.float32
            )[0]
            expected = (8192,)
        elif feature_schema == "cognitive_spatial_v5":
            features = np.asarray(
                self._encode_cognitive_spatial(observation), dtype=np.float32
            )[0]
            expected = (21504,)
        else:
            raise ValueError(f"unknown prefix feature schema {feature_schema!r}")
        if features.shape != expected:
            raise ValueError(f"unexpected frozen prefix shape {features.shape}")
        return {
            "prefix_features": features,
            "feature_schema": feature_schema,
            "prefix_timing": {"encode_ms": (time.monotonic() - started) * 1000},
            "action_rng_advanced": False,
        }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root.parent / "checkpoints/checkpoints/pi05_libero",
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    policy = policy_config.create_trained_policy(
        train_config.get_config("pi05_libero"), checkpoint
    )
    metadata = {
        **policy.metadata,
        "extension_schema": "interactive-perception.pi05-prefix-action-server.v1",
        "checkpoint": "pi05_libero",
        "checkpoint_metadata_sha256": digest(checkpoint / "params/_METADATA"),
        "openpi_commit": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
        "prefix_feature_order": [
            "last_prompt[2048]",
            "prompt_mean[2048]",
            "base_mean[2048]",
            "wrist_mean[2048]",
        ],
        "prefix_feature_schemas": {
            "global_v1": 8192,
            "cognitive_spatial_v5": 21504,
        },
        "cognitive_spatial_v5_projection": {
            "type": "fixed Rademacher",
            "seed": 260204600,
            "input_dimension": 2048,
            "output_dimension": 64,
        },
        "prefix_requests_advance_action_rng": False,
        "checkpoint_weights_modified": False,
    }
    server = websocket_policy_server.WebsocketPolicyServer(
        PrefixActionPolicy(policy), host="0.0.0.0", port=args.port, metadata=metadata
    )
    logging.info("Serving frozen actions and prefix features on port %d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
