#!/usr/bin/env python3
"""Reproduce the rejected four-token frozen-pi0.5 development features.

The same PaliGemma prefix encoder is used for the task context and every
schema-validated candidate. Candidate prompts contain only public task text and
capability data; route/effect labels are loaded by the later trainer, never by
this process. The fixed global pooling below is retained for historical pilot
reproduction only. It is not the successor target-binding representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from calibrated_interaction.contracts import (
    CandidateAction,
    validate_candidate_set,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise ValueError("snapshot dataset is empty")
    if any(row.get("online_oracle_inputs") for row in rows):
        raise ValueError("snapshot input index contains oracle inputs")
    return rows


def load_candidates(path: Path) -> tuple[CandidateAction, ...]:
    value = yaml.safe_load(path.read_text())
    if value.get("schema_version") != "calibrated-interaction.candidate-set.v1":
        raise ValueError("unsupported candidate-set schema")
    return validate_candidate_set(
        [CandidateAction.from_mapping(row) for row in value.get("candidates", ())]
    )


def candidate_prompt(task: str, candidate: CandidateAction) -> str:
    fields = [
        f"Task: {' '.join(task.split())}",
        f"Candidate primitive: {candidate.primitive.value}",
    ]
    if candidate.target:
        fields.append(f"Candidate target: {candidate.target}")
    if candidate.reference:
        fields.append(f"Candidate reference: {candidate.reference}")
    if candidate.purpose:
        fields.append(f"Candidate purpose: {candidate.purpose}")
    return ". ".join(fields) + "."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    for name in ("dataset", "candidates", "checkpoint", "output"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    report_path = args.output.with_suffix(".json")
    if args.output.exists() or report_path.exists():
        raise FileExistsError("feature outputs are immutable")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    rows = read_rows(args.dataset)
    candidates = load_candidates(args.candidates)

    from flax import nnx
    from openpi.models import model as model_lib
    from openpi.models.pi0 import make_attn_mask
    from openpi.policies import policy_config
    from openpi.shared import nnx_utils
    from openpi.training import config as train_config

    policy = policy_config.create_trained_policy(
        train_config.get_config("pi05_libero"), args.checkpoint
    )

    class PrefixTokenEncoder(nnx.Module):
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
            wrist = jnp.mean(prefix[:, tokens_per_image : 2 * tokens_per_image], axis=1)
            return jnp.stack((last_prompt, prompt_mean, agentview, wrist), axis=1)

    encode = nnx_utils.module_jit(PrefixTokenEncoder(policy._model).encode)

    def transformed(row: dict[str, Any], prompt: str):
        images = row["policy_inputs"]["image_paths"]
        agentview = imageio.imread(ROOT / images["agentview"])
        wrist = imageio.imread(ROOT / images["wrist"])
        state = np.asarray(row["policy_inputs"]["public_robot_state"], dtype=np.float32)
        value = policy._input_transform(
            {
                "observation/image": agentview,
                "observation/wrist_image": wrist,
                "observation/state": state,
                "prompt": prompt,
            }
        )
        return jax.tree.map(jnp.asarray, value)

    def encode_rows(batch_rows: list[dict[str, Any]], prompts: list[str]) -> np.ndarray:
        values = [
            transformed(row, prompt)
            for row, prompt in zip(batch_rows, prompts, strict=True)
        ]
        batch = jax.tree.map(lambda *items: jnp.stack(items), *values)
        observation = model_lib.Observation.from_dict(batch)
        return np.asarray(encode(observation), dtype=np.float32)

    context_parts: list[np.ndarray] = []
    candidate_parts: list[np.ndarray] = []
    for start in range(0, len(rows), args.batch_size):
        real_rows = rows[start : start + args.batch_size]
        real_size = len(real_rows)
        batch_rows = real_rows
        if real_size < args.batch_size:
            batch_rows = real_rows + [real_rows[-1]] * (args.batch_size - real_size)
        context = encode_rows(batch_rows, [str(row["prompt"]) for row in batch_rows])
        candidate_values = []
        for candidate in candidates:
            encoded = encode_rows(
                batch_rows,
                [candidate_prompt(str(row["prompt"]), candidate) for row in batch_rows],
            )
            # The language summary remains in the same 2048-wide representation
            # as the four context tokens. Visual tokens stay in context and are
            # not duplicated in the candidate query passed to the decoder.
            candidate_values.append(encoded[:, :2].mean(axis=1))
        context_parts.append(context[:real_size])
        candidate_parts.append(np.stack(candidate_values, axis=1)[:real_size])
        print(f"embedded {start + real_size}/{len(rows)}", flush=True)

    context_tokens = np.concatenate(context_parts, axis=0)
    candidate_tokens = np.concatenate(candidate_parts, axis=0)
    expected_context = (len(rows), 4, 2048)
    expected_candidates = (len(rows), len(candidates), 2048)
    if context_tokens.shape != expected_context:
        raise ValueError(f"unexpected context shape {context_tokens.shape}")
    if candidate_tokens.shape != expected_candidates:
        raise ValueError(f"unexpected candidate shape {candidate_tokens.shape}")
    if not np.isfinite(context_tokens).all() or not np.isfinite(candidate_tokens).all():
        raise ValueError("feature output contains non-finite values")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        context_tokens=context_tokens,
        candidate_tokens=candidate_tokens,
        sample_id=np.asarray([row["sample_id"] for row in rows]),
        split=np.asarray([row["split"] for row in rows]),
        seed=np.asarray([row["seed"] for row in rows], dtype=np.int64),
        candidate_id=np.asarray([candidate.candidate_id for candidate in candidates]),
    )
    report = {
        "schema_version": "calibrated-interaction.shared-vlm-features.v1",
        "claim_scope": "REJECTED_DEVELOPMENT_PILOT_FIXED_GLOBAL_POOLING",
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dataset": str(args.dataset.relative_to(ROOT)),
        "dataset_sha256": digest(args.dataset),
        "candidate_set": str(args.candidates.relative_to(ROOT)),
        "candidate_set_sha256": digest(args.candidates),
        "output": str(args.output.relative_to(ROOT)),
        "output_sha256": digest(args.output),
        "context_shape": list(context_tokens.shape),
        "candidate_shape": list(candidate_tokens.shape),
        "encoder": "frozen pi05_libero PaliGemma multimodal prefix",
        "shared_encoder_for_context_and_candidates": True,
        "shared_full_prefix_interface_with_action_expert": False,
        "pooling": {
            "context": [
                "prompt_last",
                "prompt_mean",
                "agentview_global_mean",
                "wrist_global_mean",
            ],
            "candidate": "mean(prompt_last, prompt_mean)",
            "spatial_token_indices_retained": False,
            "hand_designed": True,
            "paper_method_claim_allowed": False,
        },
        "checkpoint_metadata_sha256": digest(args.checkpoint / "params/_METADATA"),
        "policy_inputs": [
            "agentview RGB",
            "wrist RGB",
            "public robot state",
            "full task prompt",
            "public capability candidate",
        ],
        "online_oracle_inputs": [],
        "route_or_effect_labels_loaded": False,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
