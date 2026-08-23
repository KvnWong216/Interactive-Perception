#!/usr/bin/env python3
"""Serve pi05_libero with auditable checkpoint identity metadata."""

from __future__ import annotations

import argparse
import logging
import os
import platform
import subprocess
import sys
import uuid
from pathlib import Path

import jax
import jaxlib
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/infra"))

from checkpoint_identity import checkpoint_identity
from flax import nnx
from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import nnx_utils
from openpi.training import config as training_config

SERVER_SCHEMA = "piu.identified-pi05-server.v1"
LOGGER = logging.getLogger(__name__)


def runtime_identity(physical_gpu_index: int) -> dict:
    """Report the process and accelerator that actually loaded the policy."""

    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_gpu_index}",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    rows = [row for row in query.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("identified policy server requires exactly one selected GPU")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 4 or int(fields[0]) != physical_gpu_index:
        raise RuntimeError("nvidia-smi GPU identity differs from the selected GPU")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError("identified policy server requires one visible JAX GPU")
    fraction = float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", ""))
    return {
        "schema_version": "piu.identified-policy-runtime.v1",
        "server_process_id": os.getpid(),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "xla_python_client_mem_fraction": fraction,
        "jax_platform_version": str(devices[0].client.platform_version),
        "gpu": {
            "physical_index": physical_gpu_index,
            "visible_index": int(devices[0].id),
            "name": fields[1],
            "memory_total_mib": int(fields[2]),
            "driver_version": fields[3],
        },
    }


class IdentifiedSpatialPolicy:
    """Add deterministic full-prefix reads without advancing the action RNG."""

    def __init__(self, policy):
        self.policy = policy

        class Encoder(nnx.Module):
            def __init__(self, model):
                self.model = model

            def encode(self, observation):
                observation = model_lib.preprocess_observation(
                    None, observation, train=False
                )
                image_parts = []
                input_masks = []
                autoregressive = []
                for name in observation.images:
                    tokens, _ = self.model.PaliGemma.img(
                        observation.images[name], train=False
                    )
                    image_parts.append(tokens)
                    input_masks.append(
                        jnp.repeat(
                            observation.image_masks[name][:, None],
                            tokens.shape[1],
                            axis=1,
                        )
                    )
                    autoregressive.extend([False] * tokens.shape[1])
                prompt = self.model.PaliGemma.llm(
                    observation.tokenized_prompt, method="embed"
                )
                tokens = jnp.concatenate((*image_parts, prompt), axis=1)
                mask = jnp.concatenate(
                    (*input_masks, observation.tokenized_prompt_mask), axis=1
                )
                autoregressive.extend([False] * prompt.shape[1])
                attention = make_attn_mask(mask, jnp.asarray(autoregressive))
                positions = jnp.cumsum(mask, axis=1) - 1
                (hidden, _), _ = self.model.PaliGemma.llm(
                    [tokens, None], mask=attention, positions=positions
                )
                boundaries = []
                start = 0
                for part in image_parts:
                    boundaries.append(hidden[:, start : start + part.shape[1]])
                    start += part.shape[1]
                return (
                    tuple(boundaries),
                    hidden[:, start:],
                    tuple(input_masks),
                    observation.tokenized_prompt_mask,
                )

        self._encode = nnx_utils.module_jit(Encoder(policy._model).encode)

    def infer(self, observation: dict):
        request_type = observation.get("__request_type")
        if request_type is None:
            return self.policy.infer(observation)
        if (
            request_type != "prefix"
            or observation.get("__feature_schema") != "spatial_prefix_v1"
        ):
            raise ValueError("unsupported identified-server feature request")
        public = {
            key: value
            for key, value in observation.items()
            if not str(key).startswith("__")
        }
        transformed = self.policy._input_transform(
            jax.tree.map(lambda value: value, public)
        )
        batched = jax.tree.map(
            lambda value: jnp.asarray(value)[np.newaxis, ...], transformed
        )
        model_observation = model_lib.Observation.from_dict(batched)
        image_parts, prompt, image_masks, prompt_mask = self._encode(model_observation)
        camera_names = tuple(model_observation.images)
        image_arrays = [np.asarray(part[0], dtype=np.float16) for part in image_parts]
        return {
            "schema_version": "piu.spatial-prefix-response.v1",
            "camera_names": list(camera_names),
            "tokens_per_camera": [int(part.shape[0]) for part in image_arrays],
            "image_tokens": np.concatenate(image_arrays, axis=0),
            "image_valid_mask": np.concatenate(
                [np.asarray(mask[0], dtype=bool) for mask in image_masks]
            ),
            "prompt_tokens": np.asarray(prompt[0], dtype=np.float16),
            "prompt_valid_mask": np.asarray(prompt_mask[0], dtype=bool),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-config", default="pi05_libero")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    identity = checkpoint_identity(checkpoint)
    LOGGER.info("Checkpoint identity: %s", identity)
    policy = policy_config.create_trained_policy(
        training_config.get_config(args.policy_config), checkpoint
    )
    metadata = {
        "schema_version": SERVER_SCHEMA,
        "policy_config": args.policy_config,
        "environment": "LIBERO",
        "checkpoint": identity,
        "capabilities": ["action_chunks", "spatial_prefix_v1"],
        "server_session_id": uuid.uuid4().hex,
        "runtime_identity": runtime_identity(args.physical_gpu_index),
    }
    websocket_policy_server.WebsocketPolicyServer(
        policy=IdentifiedSpatialPolicy(policy),
        host=args.host,
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
