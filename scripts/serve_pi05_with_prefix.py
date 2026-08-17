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
        return jnp.concatenate(
            [last_prompt, prompt_mean, base_mean, wrist_mean], axis=-1
        )


class PrefixActionPolicy:
    def __init__(self, policy):
        self.policy = policy
        self._encode = nnx_utils.module_jit(PrefixEncoder(policy._model).encode)

    def infer(self, raw: dict) -> dict:
        request = str(raw.get("__request_type", "action"))
        payload = {key: value for key, value in raw.items() if key != "__request_type"}
        if request == "action":
            return self.policy.infer(payload)
        if request != "prefix":
            raise ValueError(f"unknown request type {request!r}")
        started = time.monotonic()
        inputs = jax.tree.map(lambda value: value, payload)
        inputs = self.policy._input_transform(inputs)
        inputs = jax.tree.map(lambda value: jnp.asarray(value)[np.newaxis, ...], inputs)
        observation = model_lib.Observation.from_dict(inputs)
        features = np.asarray(self._encode(observation), dtype=np.float32)[0]
        if features.shape != (8192,):
            raise ValueError(f"unexpected frozen prefix shape {features.shape}")
        return {
            "prefix_features": features,
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
