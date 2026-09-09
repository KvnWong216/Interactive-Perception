"""Weight-free and real-checkpoint preflight checks for frozen PSR V1."""

from __future__ import annotations

import dataclasses
import importlib.util
import math
from typing import Any

from .config import PSRConfig, validate_native_architecture
from .molmo_backend import MolmoPSRBackend
from .types import IntentCandidate, PublicHistory, PublicObservation, RGBReference


@dataclasses.dataclass(frozen=True)
class PreflightReport:
    device: str
    load_model: bool
    checks: dict[str, bool]
    measurements: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "psr-v1-preflight-report-v1",
            "device": self.device,
            "load_model": self.load_model,
            "passed": self.passed,
            "checks": self.checks,
            "measurements": self.measurements,
        }


def cpu_preflight(config: PSRConfig, *, device: str) -> PreflightReport:
    """Validate only local configuration/dependencies; never fetch weights."""

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    checks: dict[str, bool] = {
        "config_frozen_v1": config.to_dict()["method"] == "psr_v1",
        "yaml_parser_available": importlib.util.find_spec("yaml") is not None,
    }
    measurements = {
        "torch_available": importlib.util.find_spec("torch") is not None,
        "transformers_available": importlib.util.find_spec("transformers") is not None,
        "weights_downloaded": "NOT_CHECKED",
        "real_forward": "NOT_RUN",
    }
    return PreflightReport(
        device=device,
        load_model=False,
        checks=checks,
        measurements=measurements,
    )


def _interface_history() -> tuple[PublicHistory, dict[str, Any]]:
    """Create a visibly synthetic batch used only for API/gradient checks."""

    import numpy as np

    images: dict[str, Any] = {}
    refs = []
    for camera, value in (("agent", 32), ("wrist", 96)):
        image = np.full((256, 256, 3), value, dtype=np.uint8)
        digest = f"{value:064x}"[-64:]
        reference = RGBReference(f"psr-canary-{camera}", digest, 256, 256)
        images[reference.frame_id] = image
        refs.append(reference)
    observation = PublicObservation(refs[0], refs[1], (0.0,) * 8, 0)
    return PublicHistory(
        task="move the object to the target area",
        current=observation,
        previous=(),
        executed=(),
        remaining_control_steps=300,
    ), images


def real_model_preflight(
    config: PSRConfig,
    *,
    device: str,
) -> PreflightReport:
    """Load pinned weights and exercise the actual token/KV/AE gradient path.

    The synthetic pixels/actions are an interface canary, never a training or
    evaluation record.  This function creates no checkpoint or outcome file.
    """

    import numpy as np
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real MolmoAct2 preflight requires an available CUDA device")
    history, frames = _interface_history()
    base = config.section("base")
    state = config.section("state")
    intent = config.section("intent")
    execution = config.section("execution")
    adaptation = config.section("adaptation")
    backend = MolmoPSRBackend.from_pretrained(
        frame_resolver=lambda ref: frames[ref.frame_id],
        model_id=str(base["model_id"]),
        revision=str(base["revision"]),
        device=device,
        dtype=str(config.section("training")["dtype"]),
        num_state_tokens=int(state["num_tokens"]),
        previous_observation_bundles=int(state["previous_observation_bundles"]),
        norm_tag=str(base["norm_tag"]),
        generated_candidates=int(intent["generated_candidates"]),
        max_new_tokens=int(intent["max_new_tokens"]),
        temperature=float(intent["temperature"]),
        top_p=float(intent["top_p"]),
        max_generation_attempts=int(intent["max_generation_attempts"]),
        flow_steps=int(execution["flow_steps"]),
        upper_vlm_layers=int(adaptation["upper_vlm_layers"]),
        lora_rank=int(adaptation["lora_rank"]),
        lora_alpha=float(adaptation["lora_alpha"]),
    )
    architecture = validate_native_architecture(
        config,
        {
            "hidden_size": backend.capabilities.hidden_dim,
            "num_hidden_layers": backend.capabilities.num_vlm_layers,
            "num_key_value_heads": backend.capabilities.num_kv_heads,
            "action_horizon": backend.capabilities.max_action_horizon,
            "action_dim": backend.capabilities.max_action_dim,
        },
    )
    encoded = backend.encode(history)
    candidate = IntentCandidate(
        "move toward and inspect the object",
        tuple(backend._plain_token_ids("move toward and inspect the object")),
        "conditioned",
    )
    kv, kv_mask, _, _ = backend._conditioned_kv(encoded, candidate)
    for _, parameter in backend.named_trainable_parameters():
        parameter.grad = None
    flow_loss = backend.flow_matching_loss(
        encoded,
        candidate,
        np.zeros((10, 7), dtype=np.float32),
        timesteps=torch.tensor([0.5], device=device),
        noise=torch.ones(
            (1, architecture.action_horizon, architecture.action_dim),
            device=device,
            dtype=backend._action_expert().action_embed.weight.dtype,
        ),
    )
    flow_loss.backward()
    gradient_names = [
        name
        for name, parameter in backend.named_trainable_parameters()
        if parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.float().abs().sum().item()) > 0
    ]
    frozen_gradient_names = [
        name
        for name, parameter in backend.model.named_parameters()
        if not parameter.requires_grad
        and parameter.grad is not None
        and float(parameter.grad.detach().float().abs().sum().item()) > 0
    ]
    proposed = backend.propose(encoded, episode_seed=17, global_step=0)
    conditioned = [item for item in proposed if item.execution_route == "conditioned"]
    conditioned_chunk = backend.act_chunk(
        encoded,
        candidate,
        seed=9918,
        requested_steps=10,
    )
    conditioned_finite = all(
        len(row) == 7 and all(math.isfinite(value) for value in row)
        for row in conditioned_chunk.actions
    )
    native = next(item for item in proposed if item.execution_route == "native")
    native_seed = 9917
    wrapped_native = np.asarray(
        backend.act_chunk(
            encoded, native, seed=native_seed, requested_steps=10
        ).actions,
        dtype=np.float32,
    )
    direct_generator = torch.Generator(device=device).manual_seed(native_seed)
    with backend.adapter_mode(enabled=False), torch.inference_mode():
        direct = backend.outer.predict_action(
            processor=backend.processor,
            images=list(encoded.current_images),
            task=encoded.task,
            state=np.asarray(encoded.current_state, dtype=np.float32),
            norm_tag=backend.norm_tag,
            inference_action_mode="continuous",
            enable_depth_reasoning=False,
            num_steps=backend.flow_steps,
            generator=direct_generator,
            normalize_language=True,
            enable_cuda_graph=False,
        )
    direct_native = direct.actions
    if hasattr(direct_native, "detach"):
        direct_native = direct_native.detach().float().cpu().numpy()
    direct_native = np.asarray(direct_native, dtype=np.float32)
    if direct_native.ndim == 3:
        direct_native = direct_native[0]
    native_max_abs = float(np.max(np.abs(wrapped_native - direct_native[:10])))
    checks: dict[str, bool] = {
        "config_frozen_v1": True,
        "processor_and_prefix": encoded.state_tokens.shape[1]
        == int(state["num_tokens"]),
        "per_layer_kv": len(kv) == architecture.num_hidden_layers,
        "kv_mask_aligned": int(kv_mask.shape[-1]) == int(kv[0][0].shape[1]),
        "flow_loss_finite": math.isfinite(float(flow_loss.detach().float().item())),
        "state_token_gradient": any(
            "psr_v1_state_adapter.b_embed" in name for name in gradient_names
        ),
        "upper_vlm_lora_gradient": any(
            "transformer.blocks" in name and ".up.weight" in name
            for name in gradient_names
        ),
        "ae_context_lora_gradient": any(
            "action_expert.context_" in name and ".up.weight" in name
            for name in gradient_names
        ),
        "frozen_parameters_no_gradient": not frozen_gradient_names,
        "ordinary_intent_candidates": len(conditioned)
        == int(intent["generated_candidates"]),
        "native_candidate_present": any(
            item.execution_route == "native" for item in proposed
        ),
        "conditioned_action_finite": conditioned_finite,
        "conditioned_action_steps": len(conditioned_chunk.actions) == 10,
        "native_bypass_matches": native_max_abs <= 1e-6,
    }
    measurements = {
        "checkpoint_revision": backend.capabilities.checkpoint_revision,
        "ordinary_intent_candidate_count": len(conditioned),
        "native_bypass_max_abs_difference": native_max_abs,
        "gradient_parameter_names": gradient_names,
        "frozen_gradient_parameter_names": frozen_gradient_names,
        "canary_scope": "INTERFACE_ONLY_NOT_SCIENTIFIC_RESULT",
    }
    return PreflightReport(
        device=device,
        load_model=True,
        checks=checks,
        measurements=measurements,
    )


__all__ = ["PreflightReport", "cpu_preflight", "real_model_preflight"]
