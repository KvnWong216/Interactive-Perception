"""MolmoAct2 compatibility boundary for PSR-VLA v1.

This module owns the only code that reaches into MolmoAct2 remote-code
internals.  It deliberately fails closed when the pinned checkpoint no longer
exposes the native visual-embedding, per-layer KV, or action-expert entry
points.  The rest of :mod:`grounded_interaction.psr` depends on the small
``MolmoPSRBackend`` API rather than on private Hugging Face implementation
details spread throughout the repository.

The conditioned path builds official visual embeddings first, inserts the
repository-owned history/soft positions, and gives the second VLM forward
``inputs_embeds``
only.  Its native *per-layer* KV is then supplied to the checkpoint's action
expert.  The native route remains an exact call to ``predict_action`` with all
PSR additions absent.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import inspect
import math
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Literal

from .types import IntentCandidate, normalize_intent_text

MOLMOACT2_REPO_ID = "allenai/MolmoAct2-LIBERO"
MOLMOACT2_MODEL_REVISION = "0d24a92bd1faf321ef497c3bbd5681af97c65aa2"
MOLMOACT2_CODE_REVISION = "66b87e64efd99dfd103241418113955cf64dfa9c"
INTENT_PROMPT = "Describe the next robot action in one short phrase.\nNext action:"
ROUTE_NATIVE = "native"
ROUTE_CONDITIONED = "conditioned"


class PSRBackendError(RuntimeError):
    """Base error for an incompatible or failed latent/KV integration."""


class PSRCompatibilityError(PSRBackendError):
    """The loaded checkpoint does not implement the frozen PSR contract."""


@dataclasses.dataclass(frozen=True)
class MolmoCapabilities:
    hidden_dim: int
    num_vlm_layers: int
    num_kv_heads: int
    head_dim: int
    max_action_horizon: int
    max_action_dim: int
    action_trigger_id: int
    checkpoint_revision: str
    upstream_revision: str

    def __post_init__(self) -> None:
        numeric = (
            self.hidden_dim,
            self.num_vlm_layers,
            self.num_kv_heads,
            self.head_dim,
            self.max_action_horizon,
            self.max_action_dim,
        )
        if any(not isinstance(value, int) or value < 1 for value in numeric):
            raise ValueError("all discovered MolmoAct2 dimensions must be positive")
        if not isinstance(self.action_trigger_id, int) or self.action_trigger_id < 0:
            raise ValueError("action trigger must be a non-negative token ID")


@dataclasses.dataclass
class EncodedHistory:
    """GPU-resident common prefix; never serialized as JSON."""

    history_identity: str
    task: str
    state_tokens: Any
    prefix_inputs_embeds: Any
    prefix_input_ids: Any
    attention_mask: Any
    token_type_ids: Any | None
    b_positions: Any
    native_inputs: Mapping[str, Any]
    current_images: tuple[Any, Any]
    current_state: tuple[float, ...]
    generation_coverage: Mapping[str, int] | None = None


@dataclasses.dataclass(frozen=True)
class ActionChunk:
    actions: tuple[tuple[float, ...], ...]
    execution_route: Literal["native", "conditioned"]
    requested_steps: int

    def __post_init__(self) -> None:
        rows = tuple(tuple(float(x) for x in row) for row in self.actions)
        if not rows or len(rows) > self.requested_steps:
            raise ValueError("action chunk length is outside the requested prefix")
        if any(len(row) < 1 or any(not math.isfinite(x) for x in row) for row in rows):
            raise ValueError("action chunk must contain finite vectors")
        object.__setattr__(self, "actions", rows)


@dataclasses.dataclass(frozen=True)
class EvidenceTarget:
    """Frozen native visual patch features and their checkpoint-native order."""

    features: Any  # [1, J, D_vlm]
    valid_patch_mask: Any  # [1, J]
    camera_ids: tuple[int, ...]


def _make_toggleable_lora(
    base: Any,
    *,
    rank: int,
    alpha: float,
    enabled: Callable[[], bool],
) -> Any:
    """Wrap one real Linear layer with a zero-initialized, removable delta."""

    torch = _require_torch()
    if not isinstance(base, torch.nn.Linear):
        raise PSRCompatibilityError("LoRA target is not a torch Linear layer")

    class ToggleableLoRALinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = base
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)
            self.down = torch.nn.Linear(
                base.in_features,
                rank,
                bias=False,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
            self.up = torch.nn.Linear(
                rank,
                base.out_features,
                bias=False,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
            torch.nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(self.up.weight)
            self.scale = float(alpha) / float(rank)

        @property
        def weight(self) -> Any:
            return self.base.weight

        @property
        def bias(self) -> Any:
            return self.base.bias

        @property
        def in_features(self) -> int:
            return int(self.base.in_features)

        @property
        def out_features(self) -> int:
            return int(self.base.out_features)

        def forward(self, value: Any) -> Any:
            output = self.base(value)
            if enabled():
                output = output + self.up(self.down(value)) * self.scale
            return output

    return ToggleableLoRALinear()


def content_seed(*, episode_seed: int, global_step: int, attempt: int) -> int:
    """Candidate-order-independent seed derived from public temporal identity."""

    if min(episode_seed, global_step, attempt) < 0:
        raise ValueError("seed components must be non-negative")
    digest = hashlib.sha256(
        f"psr-intent-seed-v1\n{episode_seed}\n{global_step}\n{attempt}\n".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional learned extra.
        raise PSRBackendError("PSR Molmo integration requires PyTorch") from error
    return torch


def _unwrap_backbone(model: Any) -> tuple[Any, Any]:
    """Return ``(conditional_generation_model, MolmoAct2Model)``."""

    outer = getattr(getattr(model, "base_model", None), "model", None) or model
    backbone = getattr(outer, "model", None)
    if backbone is None:
        raise PSRCompatibilityError(
            "MolmoAct2 conditional model has no `.model` backbone"
        )
    return outer, backbone


def discover_capabilities(
    model: Any,
    *,
    checkpoint_revision: str = MOLMOACT2_MODEL_REVISION,
    upstream_revision: str = MOLMOACT2_CODE_REVISION,
) -> MolmoCapabilities:
    """Discover dimensions from the loaded checkpoint; no diagram constants."""

    outer, backbone = _unwrap_backbone(model)
    transformer = getattr(backbone, "transformer", None)
    text_cfg = getattr(transformer, "config", None)
    cfg = getattr(outer, "config", None)
    if text_cfg is None or cfg is None:
        raise PSRCompatibilityError(
            "MolmoAct2 text/checkpoint configuration is unavailable"
        )
    required = {
        "merge_visual_inputs": getattr(backbone, "merge_visual_inputs", None),
        "build_input_embeddings": getattr(backbone, "build_input_embeddings", None),
        "_extract_kv_states": getattr(backbone, "_extract_kv_states", None),
        "generate_actions_from_inputs": getattr(
            backbone, "generate_actions_from_inputs", None
        ),
    }
    absent = sorted(name for name, value in required.items() if not callable(value))
    if absent:
        raise PSRCompatibilityError(
            "pinned MolmoAct2 latent/KV entry points are missing: " + ", ".join(absent)
        )
    action_expert = getattr(backbone, "action_expert", None)
    if action_expert is None:
        require = getattr(backbone, "_require_action_expert", None)
        action_expert = require() if callable(require) else None
    if action_expert is None:
        raise PSRCompatibilityError("checkpoint exposes no continuous Action Expert")
    blocks = getattr(action_expert, "blocks", ())
    if len(blocks) != int(getattr(text_cfg, "num_hidden_layers", -1)):
        raise PSRCompatibilityError(
            "Action Expert and VLM must have one block per layer"
        )
    first_attn = getattr(getattr(transformer, "blocks", [None])[0], "self_attn", None)
    kv_heads = int(
        getattr(text_cfg, "num_key_value_heads", 0)
        or getattr(first_attn, "num_key_value_heads", 0)
    )
    head_dim = int(
        getattr(text_cfg, "head_dim", 0) or getattr(first_attn, "head_dim", 0)
    )
    trigger = getattr(cfg, "action_output_token_id", None)
    if trigger is None:
        raise PSRCompatibilityError("checkpoint has no action_output_token_id")
    return MolmoCapabilities(
        hidden_dim=int(getattr(text_cfg, "hidden_size", 0)),
        num_vlm_layers=int(getattr(text_cfg, "num_hidden_layers", 0)),
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        max_action_horizon=int(getattr(cfg, "max_action_horizon", 0)),
        max_action_dim=int(getattr(cfg, "max_action_dim", 0)),
        action_trigger_id=int(trigger),
        checkpoint_revision=checkpoint_revision,
        upstream_revision=upstream_revision,
    )


def _find_subsequence(row: Sequence[int], needle: Sequence[int]) -> int:
    if not needle:
        raise ValueError("boundary token sequence must be non-empty")
    matches = [
        start
        for start in range(len(row) - len(needle) + 1)
        if list(row[start : start + len(needle)]) == list(needle)
    ]
    if not matches:
        raise PSRCompatibilityError(
            "tokenizer did not emit the native assistant boundary"
        )
    return matches[-1]


def _postprocess_conditioned_actions(
    *,
    outer: Any,
    actions: Any,
    stats: Any,
    tag: Any,
    action_dim: int,
    n_action_steps: int,
) -> Any:
    """Apply the pinned checkpoint's native action-window postprocessing."""

    slice_dim = getattr(outer, "_slice_action_dim", None)
    slice_chunk = getattr(outer, "_slice_action_chunk", None)
    if not callable(slice_dim) or not callable(slice_chunk):
        raise PSRCompatibilityError("checkpoint lacks native action slicing helpers")
    actions = slice_dim(actions, action_dim)
    actions = slice_chunk(actions, int(outer.config.n_obs_steps), n_action_steps)
    return stats.unnormalize_action(actions, tag)


class MolmoPSRBackend:
    """In-process token/KV bridge for one pinned MolmoAct2 checkpoint."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        frame_resolver: Callable[[Any], Any],
        num_state_tokens: int = 6,
        previous_observation_bundles: int = 2,
        norm_tag: str = "libero",
        generated_candidates: int = 2,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_generation_attempts: int = 4,
        flow_steps: int = 10,
        upper_vlm_layers: int = 8,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
    ) -> None:
        torch = _require_torch()
        if num_state_tokens < 1 or previous_observation_bundles not in {0, 1, 2}:
            raise ValueError("invalid frozen predictive-state/history capacity")
        if generated_candidates != 2 or max_new_tokens != 32:
            raise ValueError("PSR v1 freezes two generated candidates and 32 tokens")
        self.model = model
        self.processor = processor
        self.frame_resolver = frame_resolver
        self.capabilities = discover_capabilities(model)
        self.outer, self.backbone = _unwrap_backbone(model)
        self.tokenizer = getattr(processor, "tokenizer", None)
        if self.tokenizer is None:
            raise PSRCompatibilityError("processor has no tokenizer")
        self.num_state_tokens = int(num_state_tokens)
        self.previous_observation_bundles = int(previous_observation_bundles)
        self.norm_tag = str(norm_tag)
        self.generated_candidates = int(generated_candidates)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_generation_attempts = int(max_generation_attempts)
        self.flow_steps = int(flow_steps)
        self._lock = threading.RLock()
        self._adapter_enabled = True
        if min(upper_vlm_layers, lora_rank) < 1 or not math.isfinite(lora_alpha):
            raise ValueError("invalid frozen V1 adaptation configuration")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        embedding = self.outer.get_input_embeddings()
        if embedding is None:
            embedding = self.backbone.get_input_embeddings()
        if embedding is None:
            raise PSRCompatibilityError("MolmoAct2 exposes no input embedding table")
        self.word_embeddings = embedding
        scale = float(getattr(self.outer.config, "initializer_range", 0.02))
        state_adapter = torch.nn.Module()
        state_adapter.register_parameter(
            "b_embed",
            torch.nn.Parameter(
                torch.empty(self.num_state_tokens, self.capabilities.hidden_dim)
            ),
        )
        torch.nn.init.normal_(state_adapter.b_embed, std=scale)
        state_adapter.history_camera_embed = torch.nn.Embedding(
            2, self.capabilities.hidden_dim
        )
        state_adapter.history_age_embed = torch.nn.Embedding(
            max(1, self.previous_observation_bundles), self.capabilities.hidden_dim
        )
        state_adapter.register_parameter(
            "history_marker",
            torch.nn.Parameter(torch.zeros(self.capabilities.hidden_dim)),
        )
        state_adapter.history_projection = torch.nn.Linear(
            self.capabilities.hidden_dim, self.capabilities.hidden_dim
        )
        state_adapter.to(
            device=embedding.weight.device,
            dtype=embedding.weight.dtype,
        )
        if hasattr(self.backbone, "psr_v1_state_adapter"):
            raise PSRCompatibilityError(
                "checkpoint already owns a PSR V1 state adapter"
            )
        self.backbone.psr_v1_state_adapter = state_adapter
        self.state_adapter = state_adapter
        self.b_embed = state_adapter.b_embed
        self.history_camera_embed = state_adapter.history_camera_embed
        self.history_age_embed = state_adapter.history_age_embed
        self.history_marker = state_adapter.history_marker
        self.history_projection = state_adapter.history_projection
        self.adapter_module_names = self._install_adapters(
            upper_vlm_layers=int(upper_vlm_layers),
            rank=int(lora_rank),
            alpha=float(lora_alpha),
        )

    def _install_adapters(
        self, *, upper_vlm_layers: int, rank: int, alpha: float
    ) -> tuple[str, ...]:
        """Install only the frozen V1 upper-VLM and AE-context deltas."""

        transformer = getattr(self.backbone, "transformer", None)
        blocks = getattr(transformer, "blocks", None)
        if blocks is None or len(blocks) != self.capabilities.num_vlm_layers:
            raise PSRCompatibilityError("cannot locate the checkpoint VLM block list")
        start = max(0, len(blocks) - upper_vlm_layers)
        names: list[str] = []
        for index in range(start, len(blocks)):
            attention = getattr(blocks[index], "self_attn", None)
            if attention is None:
                raise PSRCompatibilityError("upper VLM block has no self attention")
            for attribute in ("att_proj", "attn_out"):
                base = getattr(attention, attribute, None)
                setattr(
                    attention,
                    attribute,
                    _make_toggleable_lora(
                        base,
                        rank=rank,
                        alpha=alpha,
                        enabled=lambda: self._adapter_enabled,
                    ),
                )
                names.append(f"transformer.blocks.{index}.self_attn.{attribute}")
        action_expert = self._action_expert()
        for attribute in ("context_k_proj", "context_v_proj"):
            base = getattr(action_expert, attribute, None)
            setattr(
                action_expert,
                attribute,
                _make_toggleable_lora(
                    base,
                    rank=rank,
                    alpha=alpha,
                    enabled=lambda: self._adapter_enabled,
                ),
            )
            names.append(f"action_expert.{attribute}")
        if len(names) != upper_vlm_layers * 2 + 2:
            raise PSRCompatibilityError("PSR LoRA target discovery was incomplete")
        return tuple(names)

    def _action_expert(self) -> Any:
        expert = getattr(self.backbone, "action_expert", None)
        if expert is None:
            require = getattr(self.backbone, "_require_action_expert", None)
            expert = require() if callable(require) else None
        if expert is None:
            raise PSRCompatibilityError("checkpoint exposes no Action Expert")
        return expert

    def named_trainable_parameters(self) -> tuple[tuple[str, Any], ...]:
        """All and only Stage-A parameters owned by the latent execution path."""

        return tuple(
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        )

    def reset(self) -> None:
        """Clear episode-local generation metadata and upstream AE caches."""

        expert = self._action_expert()
        with self._lock:
            for name in ("_modulation_cache_key", "_modulation_cache_value"):
                if hasattr(expert, name):
                    setattr(expert, name, None)

    @classmethod
    def from_pretrained(
        cls,
        *,
        frame_resolver: Callable[[Any], Any],
        model_id: str = MOLMOACT2_REPO_ID,
        revision: str = MOLMOACT2_MODEL_REVISION,
        device: str = "cuda",
        dtype: str = "bfloat16",
        **kwargs: Any,
    ) -> MolmoPSRBackend:
        """Load the real pinned checkpoint; never used by CPU preflight."""

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:  # pragma: no cover - GPU integration.
            raise PSRBackendError(
                "real PSR backend requires torch and transformers molmoact2 extras"
            ) from error
        if model_id != MOLMOACT2_REPO_ID or revision != MOLMOACT2_MODEL_REVISION:
            raise PSRCompatibilityError(
                "PSR v1 requires the frozen MolmoAct2-LIBERO snapshot"
            )
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, trust_remote_code=True, extra_special_tokens={}
        )
        # Some transformers releases use ``torch_dtype`` and newer ones use
        # ``dtype``.  Inspect rather than retrying a partially allocated load.
        loader = AutoModelForImageTextToText.from_pretrained
        dtype_name = (
            "dtype"
            if "dtype" in inspect.signature(loader).parameters
            else "torch_dtype"
        )
        model = loader(
            model_id,
            revision=revision,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **{dtype_name: torch_dtype},
        ).to(device)
        return cls(
            model=model, processor=processor, frame_resolver=frame_resolver, **kwargs
        )

    @contextlib.contextmanager
    def adapter_mode(self, *, enabled: bool) -> Iterator[None]:
        """Atomically and exception-safely switch all repo-owned deltas."""

        with self._lock:
            previous = self._adapter_enabled
            self._adapter_enabled = bool(enabled)
            try:
                yield
            finally:
                self._adapter_enabled = previous

    def _images(self, observation: Any) -> tuple[Any, Any]:
        try:
            agent = self.frame_resolver(observation.agentview_rgb)
            wrist = self.frame_resolver(observation.wrist_rgb)
        except AttributeError as error:
            raise TypeError("observation must expose two RGB references") from error
        try:
            import numpy as np
            from PIL import Image
        except ImportError as error:  # pragma: no cover - integration extra.
            raise PSRBackendError(
                "Molmo image processing requires numpy and Pillow"
            ) from error
        result = []
        for value in (agent, wrist):
            array = np.asarray(value)
            if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
                raise ValueError("resolved RGB must be uint8 HWC")
            result.append(Image.fromarray(array, mode="RGB"))
        return result[0], result[1]

    def _native_inputs(self, history: Any) -> tuple[dict[str, Any], tuple[Any, Any]]:
        """Reproduce the checkpoint's own task/state prompt construction."""

        import numpy as np

        images = self._images(history.current)
        state = np.asarray(history.current.robot_state, dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("MolmoAct2-LIBERO state must be finite 8-D")
        stats = self.outer._get_robot_stats()
        tag = stats.validate_tag(self.norm_tag)
        metadata = stats.get_metadata(tag)
        normalized = np.asarray(stats.normalize_state(state, tag), dtype=np.float32)
        module = inspect.getmodule(type(self.outer))
        if module is None:
            raise PSRCompatibilityError("cannot resolve MolmoAct2 remote-code module")
        required = (
            "_build_discrete_state_string",
            "_normalize_question_text",
            "_build_robot_text",
        )
        helpers = {name: getattr(module, name, None) for name in required}
        if any(not callable(value) for value in helpers.values()):
            raise PSRCompatibilityError(
                "native MolmoAct2 prompt helpers are unavailable"
            )
        discrete = helpers["_build_discrete_state_string"](
            normalized, int(self.outer.config.num_state_tokens)
        )
        task = helpers["_normalize_question_text"](str(history.task))
        text = helpers["_build_robot_text"](
            task=task,
            style="robot_action",
            discrete_state_string=discrete,
            setup_type=str(metadata.get("setup_type", "") or ""),
            control_mode=str(metadata.get("control_mode", "") or ""),
            add_setup_tokens=bool(self.outer.config.add_setup_tokens),
            add_control_tokens=bool(self.outer.config.add_control_tokens),
            num_images=2,
        )
        inputs = dict(
            self.processor(text=text, images=list(images), return_tensors="pt")
        )
        move = getattr(self.outer, "_move_inputs_to_device", None)
        if not callable(move):
            raise PSRCompatibilityError("MolmoAct2 input-device helper is unavailable")
        device = next(self.model.parameters()).device
        inputs = dict(move(inputs, device))
        drop = getattr(self.outer, "_drop_trivial_attention_mask", None)
        if callable(drop):
            inputs = dict(drop(inputs))
        return inputs, images

    def _official_embeddings(self, inputs: Mapping[str, Any]) -> tuple[Any, Any]:
        if inputs.get("inputs_embeds") is not None:
            raise PSRCompatibilityError(
                "official visual pass unexpectedly returned inputs_embeds"
            )
        images, pooling = self.backbone.merge_visual_inputs(
            input_ids=inputs.get("input_ids"),
            pixel_values=inputs.get("pixel_values"),
            image_token_pooling=inputs.get("image_token_pooling"),
            image_grids=inputs.get("image_grids"),
            image_num_crops=inputs.get("image_num_crops"),
            pixel_values_videos=inputs.get("pixel_values_videos"),
            video_token_pooling=inputs.get("video_token_pooling"),
            video_grids=inputs.get("video_grids"),
        )
        embeddings, image_features = self.backbone.build_input_embeddings(
            inputs["input_ids"], images, pooling
        )
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.capabilities.hidden_dim:
            raise PSRCompatibilityError(
                "native input embeddings have an unexpected shape"
            )
        return embeddings, image_features

    def _role_boundaries(self, input_ids: Any) -> tuple[int, int, int]:
        """Locate the final user end, assistant start, and native trigger."""

        row = [int(item) for item in input_ids[0].detach().cpu().tolist()]
        assistant_ids = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        assistant = _find_subsequence(row, assistant_ids)
        user_end_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        user_end = _find_subsequence(row[:assistant], user_end_ids)
        triggers = [
            index
            for index, token_id in enumerate(row)
            if token_id == self.capabilities.action_trigger_id and index >= assistant
        ]
        if len(triggers) != 1:
            raise PSRCompatibilityError(
                "native prompt must contain one action trigger after assistant boundary"
            )
        if not user_end < assistant < triggers[0]:
            raise PSRCompatibilityError("native chat role boundaries are misordered")
        return user_end, assistant, triggers[0]

    def _plain_token_ids(self, text: str) -> list[int]:
        ids = [
            int(item) for item in self.tokenizer.encode(text, add_special_tokens=False)
        ]
        if not ids:
            raise PSRCompatibilityError(
                "tokenizer produced an empty ordinary-text span"
            )
        return ids

    def _history_text(self, history: Any) -> str:
        """Fixed public-only text used inside the native user span."""

        rows = ["Real interaction history:"]
        executed = tuple(getattr(history, "executed", ()))
        if executed:
            for event in executed:
                rows.append(
                    "- "
                    f"steps {int(event.start_step)}-{int(event.end_step)}: "
                    f"{normalize_intent_text(event.text)}; route={event.execution_route}; "
                    f"status={str(event.public_status).strip()}"
                )
        else:
            rows.append("- none")
        rows.append(f"Remaining control steps: {int(history.remaining_control_steps)}")
        return "\n".join(rows) + "\n"

    def _single_image_visual_tokens(self, image: Any) -> Any:
        """Encode a past frame through the same frozen native visual path."""

        torch = _require_torch()
        inputs = dict(
            self.processor(text="<|image|>", images=[image], return_tensors="pt")
        )
        move = getattr(self.outer, "_move_inputs_to_device", None)
        inputs = dict(move(inputs, next(self.model.parameters()).device))
        _, image_features = self._official_embeddings(inputs)
        if image_features is None:
            raise PSRCompatibilityError("native visual path returned no image features")
        selected = image_features
        if selected.ndim != 2 or selected.shape[0] < 1:
            raise PSRCompatibilityError(
                "native visual path returned no patch positions"
            )
        if not bool(torch.isfinite(selected).all()):
            raise PSRCompatibilityError("native visual features are non-finite")
        return selected

    def _history_soft_tokens(self, history: Any) -> Any:
        torch = _require_torch()
        previous = tuple(getattr(history, "previous", ()))
        if len(previous) > self.previous_observation_bundles:
            raise ValueError("public history exceeds the frozen two-boundary window")
        pieces = []
        for age, observation in enumerate(previous):
            for camera, image in enumerate(self._images(observation)):
                patch = self._single_image_visual_tokens(image)
                tagged = (
                    patch
                    + self.history_camera_embed.weight[camera].to(patch)
                    + self.history_age_embed.weight[age].to(patch)
                    + self.history_marker.to(patch)
                )
                pieces.append(self.history_projection(tagged))
        if not pieces:
            return self.b_embed.new_empty((0, self.capabilities.hidden_dim))
        return torch.cat(pieces, dim=0)

    @staticmethod
    def _insert_prefix_field(
        value: Any | None,
        *,
        assistant: int,
        trigger: int,
        history_visual_length: int,
        history_text_length: int,
        state_length: int,
        anchor_length: int,
        fill: int,
    ) -> Any | None:
        """Mirror exact embedding segments for a mask or token-type row.

        The original assistant span is preserved while the original action
        trigger is omitted from this common prefix. Current image token types
        retain MolmoAct2's native visual attention; history, B, and language
        inserts remain ordinary causal tokens.
        """

        if value is None:
            return None
        torch = _require_torch()
        if value.ndim != 2 or value.shape[0] != 1:
            raise PSRCompatibilityError(
                "PSR v1 token assembly currently requires batch one"
            )

        def inserted(length: int) -> Any:
            return torch.full((1, length), fill, dtype=value.dtype, device=value.device)

        return torch.cat(
            (
                value[:, :assistant],
                inserted(history_visual_length),
                inserted(history_text_length),
                value[:, assistant:trigger],
                inserted(state_length),
                inserted(anchor_length),
            ),
            dim=1,
        )

    def encode(self, history: Any) -> EncodedHistory:
        """Build H, insert six soft positions, and return contextualized B."""

        torch = _require_torch()
        with self._lock, self.adapter_mode(enabled=True):
            native, images = self._native_inputs(history)
            native_embed, _ = self._official_embeddings(native)
            user_end, _assistant, trigger = self._role_boundaries(native["input_ids"])
            history_visual = self._history_soft_tokens(history).to(native_embed)
            history_ids = self._plain_token_ids(self._history_text(history))
            history_ids_t = torch.tensor(
                history_ids, dtype=torch.long, device=native_embed.device
            ).unsqueeze(0)
            history_text_embed = self.word_embeddings(history_ids_t).to(native_embed)
            anchor_ids = self._plain_token_ids(INTENT_PROMPT)
            anchor_t = torch.tensor(
                anchor_ids, dtype=torch.long, device=native_embed.device
            ).unsqueeze(0)
            anchor_embed = self.word_embeddings(anchor_t).to(native_embed)
            pre = native_embed[:, :user_end]
            # Preserve the native user-end and assistant-start spans exactly;
            # public history is inserted before the user message closes.
            role_transition_embed = native_embed[:, user_end:trigger]
            b_seed = self.b_embed.to(native_embed).unsqueeze(0)
            prefix = torch.cat(
                (
                    pre,
                    history_visual.unsqueeze(0),
                    history_text_embed,
                    role_transition_embed,
                    b_seed,
                    anchor_embed,
                ),
                dim=1,
            )
            placeholder = int(
                getattr(self.tokenizer, "pad_token_id", None)
                or getattr(self.tokenizer, "unk_token_id", None)
                or 0
            )
            ids = torch.cat(
                (
                    native["input_ids"][:, :user_end],
                    torch.full(
                        (1, history_visual.shape[0]),
                        placeholder,
                        device=native_embed.device,
                        dtype=torch.long,
                    ),
                    history_ids_t,
                    native["input_ids"][:, user_end:trigger],
                    torch.full(
                        (1, self.num_state_tokens),
                        placeholder,
                        device=native_embed.device,
                        dtype=torch.long,
                    ),
                    anchor_t,
                ),
                dim=1,
            )
            original_mask = native.get("attention_mask")
            if original_mask is None:
                original_mask = torch.ones_like(native["input_ids"])
            attention = self._insert_prefix_field(
                original_mask,
                assistant=user_end,
                trigger=trigger,
                history_visual_length=int(history_visual.shape[0]),
                history_text_length=len(history_ids),
                state_length=self.num_state_tokens,
                anchor_length=len(anchor_ids),
                fill=1,
            )
            token_types = self._insert_prefix_field(
                native.get("token_type_ids"),
                assistant=user_end,
                trigger=trigger,
                history_visual_length=int(history_visual.shape[0]),
                history_text_length=len(history_ids),
                state_length=self.num_state_tokens,
                anchor_length=len(anchor_ids),
                fill=0,
            )
            if prefix.shape[1] != ids.shape[1] or prefix.shape[1] != attention.shape[1]:
                raise PSRCompatibilityError(
                    "PSR prefix fields have inconsistent lengths"
                )
            position_ids = attention.long().cumsum(-1).sub(1).clamp_min(0)
            output = self.backbone(
                inputs_embeds=prefix,
                attention_mask=attention,
                token_type_ids=token_types,
                position_ids=position_ids,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=False,
            )
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                hidden_states = getattr(output, "hidden_states", None)
                hidden = hidden_states[-1] if hidden_states else None
            if hidden is None:
                raise PSRCompatibilityError("VLM forward did not return hidden states")
            b_start = (
                pre.shape[1]
                + history_visual.shape[0]
                + len(history_ids)
                + role_transition_embed.shape[1]
            )
            b_positions = torch.arange(
                b_start,
                b_start + self.num_state_tokens,
                device=hidden.device,
                dtype=torch.long,
            ).unsqueeze(0)
            state_tokens = hidden.gather(
                1, b_positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            )
            history_payload = (
                f"psr-history-v1\n{history.task}\n{history.current.control_step}\n"
                f"{history.remaining_control_steps}\n"
                + "\n".join(event.actions_ref for event in history.executed)
            )
            return EncodedHistory(
                history_identity=hashlib.sha256(history_payload.encode()).hexdigest(),
                task=str(history.task),
                state_tokens=state_tokens,
                prefix_inputs_embeds=prefix,
                prefix_input_ids=ids,
                attention_mask=attention,
                token_type_ids=token_types,
                b_positions=b_positions,
                native_inputs=native,
                current_images=images,
                current_state=tuple(float(x) for x in history.current.robot_state),
            )

    def _forbidden_generation_ids(self) -> set[int]:
        cfg = self.outer.config
        forbidden: set[int] = set()
        for name in (
            "action_output_token_id",
            "action_start_token_id",
            "action_end_token_id",
            "depth_start_token_id",
            "depth_end_token_id",
        ):
            value = getattr(cfg, name, None)
            if value is not None:
                forbidden.add(int(value))
        for start_name, count_name in (
            ("action_token_start_id", "num_action_tokens"),
            ("depth_token_start_id", "num_depth_tokens"),
        ):
            start = getattr(cfg, start_name, None)
            count = int(getattr(cfg, count_name, 0) or 0)
            if start is not None:
                forbidden.update(range(int(start), int(start) + count))
        return forbidden

    def _sample_one(
        self, encoded: EncodedHistory, *, seed: int
    ) -> tuple[str, tuple[int, ...]]:
        torch = _require_torch()
        generator = torch.Generator(
            device=encoded.prefix_inputs_embeds.device
        ).manual_seed(seed)
        with torch.inference_mode():
            output = self.outer(
                inputs_embeds=encoded.prefix_inputs_embeds,
                attention_mask=encoded.attention_mask,
                token_type_ids=encoded.token_type_ids,
                use_cache=True,
                output_hidden_states=False,
                output_attentions=False,
            )
            cache = output.past_key_values
            logits = output.logits[:, -1, :].float()
            attention = encoded.attention_mask
            generated: list[int] = []
            forbidden = self._forbidden_generation_ids()
            stop_ids = {
                int(item)
                for item in self.tokenizer.encode("\n", add_special_tokens=False)
            }
            eos = getattr(self.tokenizer, "eos_token_id", None)
            if eos is not None:
                stop_ids.add(int(eos))
            for _ in range(self.max_new_tokens):
                if forbidden:
                    logits[:, list(forbidden)] = -torch.inf
                probabilities = torch.softmax(logits / self.temperature, dim=-1)
                sorted_probs, sorted_ids = probabilities.sort(descending=True)
                cumulative = sorted_probs.cumsum(-1)
                sorted_probs = sorted_probs.masked_fill(
                    cumulative - sorted_probs > self.top_p, 0
                )
                denominator = sorted_probs.sum(-1, keepdim=True)
                if not bool(torch.isfinite(denominator).all()) or bool(
                    (denominator <= 0).any()
                ):
                    raise ValueError("intent decoder has no finite admissible tokens")
                sorted_probs = sorted_probs / denominator
                sampled_rank = torch.multinomial(sorted_probs, 1, generator=generator)
                token = int(sorted_ids.gather(-1, sampled_rank).item())
                if token in stop_ids:
                    break
                generated.append(token)
                next_id = torch.tensor(
                    [[token]], device=logits.device, dtype=torch.long
                )
                attention = torch.cat((attention, attention.new_ones((1, 1))), dim=1)
                step = self.outer(
                    input_ids=next_id,
                    attention_mask=attention,
                    past_key_values=cache,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                cache = step.past_key_values
                logits = step.logits[:, -1, :].float()
        if not generated:
            raise ValueError("intent decoder produced no ordinary tokens")
        text = normalize_intent_text(
            self.tokenizer.decode(generated, skip_special_tokens=True)
        )
        return text, tuple(generated)

    def propose(
        self,
        encoded: EncodedHistory,
        *,
        episode_seed: int,
        global_step: int,
    ) -> list[IntentCandidate]:
        """Independently sample conditioned intents and add the native route."""

        found: dict[tuple[str, str], IntentCandidate] = {}
        attempts = 0
        with self._lock, self.adapter_mode(enabled=True):
            while (
                attempts < self.max_generation_attempts
                and len(found) < self.generated_candidates
            ):
                seed = content_seed(
                    episode_seed=episode_seed,
                    global_step=global_step,
                    attempt=attempts,
                )
                attempts += 1
                try:
                    text, token_ids = self._sample_one(encoded, seed=seed)
                except ValueError:
                    continue
                key = (text, ROUTE_CONDITIONED)
                if key not in found:
                    found[key] = IntentCandidate(
                        text=text,
                        execution_route=ROUTE_CONDITIONED,
                        token_ids=token_ids,
                    )
        native_text = normalize_intent_text(encoded.task)
        native = IntentCandidate(
            text=native_text,
            execution_route=ROUTE_NATIVE,
            token_ids=tuple(self._plain_token_ids(native_text)),
        )
        encoded.generation_coverage = {
            "requested": self.generated_candidates,
            "accepted": len(found),
            "attempted": attempts,
        }
        return list(found.values()) + [native]

    def tokenize_candidates(
        self, candidates: Sequence[IntentCandidate]
    ) -> tuple[Any, Any, Any, Any]:
        """Pad ordinary U IDs for the intent-only predictive readout."""

        torch = _require_torch()
        if not candidates:
            raise ValueError("candidate set must be non-empty")
        length = min(
            self.max_new_tokens, max(len(item.token_ids) for item in candidates)
        )
        pad = int(getattr(self.tokenizer, "pad_token_id", None) or 0)
        ids = torch.full((1, len(candidates), length), pad, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        routes = torch.zeros((1, len(candidates)), dtype=torch.long)
        valid = torch.ones((1, len(candidates)), dtype=torch.bool)
        for index, item in enumerate(candidates):
            span = item.token_ids[:length]
            ids[0, index, : len(span)] = torch.tensor(span, dtype=torch.long)
            mask[0, index, : len(span)] = True
            routes[0, index] = 0 if item.execution_route == ROUTE_NATIVE else 1
        device = next(self.model.parameters()).device
        return ids.to(device), mask.to(device), routes.to(device), valid.to(device)

    def _conditioned_fields(
        self, encoded: EncodedHistory, candidate: IntentCandidate
    ) -> tuple[Any, Any, Any, Any]:
        torch = _require_torch()
        if candidate.execution_route != ROUTE_CONDITIONED:
            raise ValueError("conditioned fields require a conditioned intent")
        intent = torch.tensor(
            candidate.token_ids,
            device=encoded.prefix_inputs_embeds.device,
            dtype=torch.long,
        ).unsqueeze(0)
        trigger = torch.tensor(
            [[self.capabilities.action_trigger_id]],
            device=intent.device,
            dtype=torch.long,
        )
        suffix_ids = torch.cat((intent, trigger), dim=1)
        suffix_embed = self.word_embeddings(suffix_ids).to(encoded.prefix_inputs_embeds)
        embeddings = torch.cat((encoded.prefix_inputs_embeds, suffix_embed), dim=1)
        ids = torch.cat((encoded.prefix_input_ids, suffix_ids), dim=1)
        attention = torch.cat(
            (encoded.attention_mask, encoded.attention_mask.new_ones(suffix_ids.shape)),
            dim=1,
        )
        token_types = None
        if encoded.token_type_ids is not None:
            token_types = torch.cat(
                (
                    encoded.token_type_ids,
                    encoded.token_type_ids.new_zeros(suffix_ids.shape),
                ),
                dim=1,
            )
        return embeddings, ids, attention, token_types

    def _conditioned_kv(
        self, encoded: EncodedHistory, candidate: IntentCandidate
    ) -> tuple[Any, Any, Any, Any]:
        embeddings, ids, attention, token_types = self._conditioned_fields(
            encoded, candidate
        )
        output = self.backbone(
            inputs_embeds=embeddings,
            attention_mask=attention,
            token_type_ids=token_types,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        kv = self.backbone._extract_kv_states(output.past_key_values)
        if len(kv) != self.capabilities.num_vlm_layers:
            raise PSRCompatibilityError(
                "conditioned VLM returned the wrong KV layer count"
            )
        encoder_mask = self.backbone._get_encoder_attention_mask(ids, attention)
        return kv, encoder_mask, ids, attention

    def extract_evidence_target(self, observation: Any, *, task: str) -> EvidenceTarget:
        """Return detached features from the same frozen native visual path.

        MolmoAct2 emits one flattened sequence in the exact order used to add
        visual features to image-patch token positions.  V1 preserves that
        native index instead of inventing a square geometry.  Camera ownership
        is recovered from the ordered image-patch spans in the real prompt.
        """

        torch = _require_torch()

        class _History:
            current = observation

        history = _History()
        history.task = task
        with self._lock, self.adapter_mode(enabled=False), torch.inference_mode():
            native, _ = self._native_inputs(history)
            _, features = self._official_embeddings(native)
        if features is None or features.ndim != 2 or features.shape[0] < 1:
            raise PSRCompatibilityError(
                "native visual path returned no evidence patches"
            )
        ids = native["input_ids"]
        patch_id = int(getattr(self.outer.config, "image_patch_id", -1))
        if int(ids.eq(patch_id).sum().item()) != int(features.shape[0]):
            raise PSRCompatibilityError(
                "visual feature and image-token counts disagree"
            )
        # Upstream defines each image's pooled-token count from its explicit
        # four-value grid and concatenates images in processor order. Reuse
        # that metadata instead of inventing a square patch layout or relying
        # on repeated image-boundary tokens.
        image_grids = native.get("image_grids")
        if (
            image_grids is None
            or image_grids.ndim != 2
            or tuple(image_grids.shape) != (2, 4)
        ):
            raise PSRCompatibilityError(
                "evidence extraction requires two native four-value image grids"
            )
        per_camera = (
            (image_grids[:, :2].prod(dim=1) + image_grids[:, 2:].prod(dim=1))
            .detach()
            .cpu()
            .tolist()
        )
        camera_ids = [
            camera for camera, count in enumerate(per_camera) for _ in range(int(count))
        ]
        if len(camera_ids) != int(features.shape[0]):
            raise PSRCompatibilityError(
                "native grid and visual feature counts disagree"
            )
        target = features.detach().unsqueeze(0).float()
        return EvidenceTarget(
            features=target,
            valid_patch_mask=torch.ones(
                target.shape[:2], device=target.device, dtype=torch.bool
            ),
            camera_ids=tuple(camera_ids),
        )

    def intent_language_loss(
        self, encoded: EncodedHistory, target_token_ids: Sequence[int]
    ) -> Any:
        """Teacher-force one public short intent after ``[H,B,format]``."""

        torch = _require_torch()
        target = torch.as_tensor(
            tuple(int(item) for item in target_token_ids),
            device=encoded.prefix_inputs_embeds.device,
            dtype=torch.long,
        ).unsqueeze(0)
        if target.shape[1] < 1 or target.shape[1] > self.max_new_tokens:
            raise ValueError("intent target length is outside the frozen V1 limit")
        target_embed = self.word_embeddings(target).to(encoded.prefix_inputs_embeds)
        inputs = torch.cat((encoded.prefix_inputs_embeds, target_embed), dim=1)
        mask = torch.cat(
            (encoded.attention_mask, encoded.attention_mask.new_ones(target.shape)),
            dim=1,
        )
        token_types = None
        if encoded.token_type_ids is not None:
            token_types = torch.cat(
                (
                    encoded.token_type_ids,
                    encoded.token_type_ids.new_zeros(target.shape),
                ),
                dim=1,
            )
        output = self.outer(
            inputs_embeds=inputs,
            attention_mask=mask,
            token_type_ids=token_types,
            use_cache=False,
            output_hidden_states=False,
        )
        prefix_length = encoded.prefix_inputs_embeds.shape[1]
        # Token i is predicted by the representation immediately before it.
        logits = output.logits[:, prefix_length - 1 : -1].float()
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1)
        )

    def _normalize_and_pad_actions(self, actions: Any) -> tuple[Any, Any, Any]:
        """Apply the pinned robot statistics and native horizon/dimension masks."""

        import numpy as np

        torch = _require_torch()
        array = np.asarray(actions, dtype=np.float32)
        if array.ndim == 2:
            array = array[None]
        if array.ndim != 3 or array.shape[0] != 1 or array.shape[1] < 1:
            raise ValueError("warmup actions must have shape [1,T,A] or [T,A]")
        stats = self.outer._get_robot_stats()
        tag = stats.validate_tag(self.norm_tag)
        action_dim = int(stats.get_action_dim(tag) or array.shape[-1])
        if array.shape[-1] != action_dim:
            raise ValueError("warmup action width differs from norm-tag metadata")
        normalizers = getattr(stats, "action_normalizers", {})
        normalizer = normalizers.get(tag)
        if normalizer is not None:
            array = np.asarray(normalizer.normalize(array), dtype=np.float32)
        horizon = int(
            stats.get_action_horizon(tag) or self.capabilities.max_action_horizon
        )
        if array.shape[1] > horizon or horizon > self.capabilities.max_action_horizon:
            raise ValueError(
                "warmup action horizon is incompatible with checkpoint metadata"
            )
        device = next(self.model.parameters()).device
        dtype = self._action_expert().action_embed.weight.dtype
        padded = torch.zeros(
            (1, horizon, self.capabilities.max_action_dim),
            device=device,
            dtype=dtype,
        )
        source = torch.as_tensor(array, device=device, dtype=dtype)
        padded[:, : source.shape[1], :action_dim] = source
        horizon_valid = torch.zeros((1, horizon), device=device, dtype=torch.bool)
        horizon_valid[:, : source.shape[1]] = True
        dim_valid = torch.zeros(
            (1, self.capabilities.max_action_dim), device=device, dtype=torch.bool
        )
        dim_valid[:, :action_dim] = True
        return padded, horizon_valid, dim_valid

    def flow_matching_loss(
        self,
        encoded: EncodedHistory,
        candidate: IntentCandidate,
        actions: Any,
        *,
        timesteps: Any | None = None,
        noise: Any | None = None,
    ) -> Any:
        """Differentiable HF Action Expert velocity loss for Stage A.

        This is the small, repository-owned port of the pinned upstream target:
        ``x_t=(1-t)noise+t*action`` and ``v*=action-noise``.  It deliberately
        bypasses both inference-only action-generation entry points.
        """

        torch = _require_torch()
        if candidate.execution_route != ROUTE_CONDITIONED:
            raise ValueError("Stage-A conditioned flow requires a conditioned intent")
        with self._lock, self.adapter_mode(enabled=True):
            kv, encoder_mask, _, _ = self._conditioned_kv(encoded, candidate)
            action, horizon_valid, dim_valid = self._normalize_and_pad_actions(actions)
            cfg = self.outer.config
            if timesteps is None:
                beta = torch.distributions.Beta(
                    torch.tensor(
                        float(cfg.flow_matching_beta_alpha), device=action.device
                    ),
                    torch.tensor(
                        float(cfg.flow_matching_beta_beta), device=action.device
                    ),
                )
                sampled = beta.sample((action.shape[0],)).to(action.dtype)
                lower = float(cfg.flow_matching_time_offset)
                upper = min(
                    float(cfg.flow_matching_cutoff),
                    lower + float(cfg.flow_matching_time_scale),
                )
                timesteps = lower + (upper - lower) * sampled
            else:
                timesteps = torch.as_tensor(
                    timesteps, device=action.device, dtype=action.dtype
                ).reshape(action.shape[0])
            if noise is None:
                noise = torch.randn_like(action)
            else:
                noise = torch.as_tensor(noise, device=action.device, dtype=action.dtype)
                if noise.shape != action.shape:
                    raise ValueError("flow noise must match padded action shape")
            valid = horizon_valid.unsqueeze(-1) & dim_valid.unsqueeze(1)
            action = action * valid
            noise = noise * valid
            t = timesteps[:, None, None]
            noisy = (1.0 - t) * noise + t * action
            target = (action - noise) * valid
            expert = self._action_expert()
            prediction = expert(
                noisy,
                timesteps,
                encoder_kv_states=kv,
                encoder_attention_mask=encoder_mask,
                action_attention_mask=horizon_valid,
                state_embeddings=None,
            )
            if prediction.shape != target.shape or not bool(
                torch.isfinite(prediction).all()
            ):
                raise PSRBackendError("Action Expert returned invalid flow predictions")
            squared = (prediction.float() - target.float()).square() * valid
            return squared.sum() / valid.sum().clamp_min(1)

    def act_chunk(
        self,
        encoded: EncodedHistory,
        candidate: IntentCandidate,
        *,
        seed: int,
        requested_steps: int,
    ) -> ActionChunk:
        """Use exact native inference or selected-intent per-layer VLM KV."""

        torch = _require_torch()
        if requested_steps < 1:
            raise ValueError("requested action prefix must be positive")
        generator = torch.Generator(
            device=next(self.model.parameters()).device
        ).manual_seed(seed)
        with self._lock:
            if candidate.execution_route == ROUTE_NATIVE:
                import numpy as np

                with self.adapter_mode(enabled=False), torch.inference_mode():
                    output = self.outer.predict_action(
                        processor=self.processor,
                        images=list(encoded.current_images),
                        task=encoded.task,
                        state=np.asarray(encoded.current_state, dtype=np.float32),
                        norm_tag=self.norm_tag,
                        inference_action_mode="continuous",
                        enable_depth_reasoning=False,
                        num_steps=self.flow_steps,
                        generator=generator,
                        normalize_language=True,
                        enable_cuda_graph=False,
                    )
                actions = output.actions
            else:
                with self.adapter_mode(enabled=True), torch.inference_mode():
                    kv, encoder_mask, ids, attention = self._conditioned_kv(
                        encoded, candidate
                    )
                    stats = self.outer._get_robot_stats()
                    tag = stats.validate_tag(self.norm_tag)
                    action_dim = int(
                        stats.get_action_dim(tag) or self.capabilities.max_action_dim
                    )
                    resolve_horizon = getattr(
                        self.backbone, "_resolve_action_horizon", None
                    )
                    max_horizon = (
                        int(resolve_horizon())
                        if callable(resolve_horizon)
                        else self.capabilities.max_action_horizon
                    )
                    action_horizon = int(stats.get_action_horizon(tag) or max_horizon)
                    n_action_steps = int(
                        stats.get_n_action_steps(tag) or action_horizon
                    )
                    if (
                        action_horizon > max_horizon
                        or not 1 <= n_action_steps <= action_horizon
                        or requested_steps > n_action_steps
                    ):
                        raise ValueError(
                            "requested action prefix exceeds native executable horizon"
                        )
                    action_dim_is_pad = self.outer._build_action_dim_is_pad(
                        action_dim=action_dim,
                        max_action_dim=self.capabilities.max_action_dim,
                        batch_size=1,
                        device=next(self.model.parameters()).device,
                    )
                    actions = self.backbone.generate_actions_from_inputs(
                        input_ids=ids,
                        attention_mask=attention,
                        action_dim_is_pad=action_dim_is_pad,
                        action_horizon=action_horizon,
                        num_steps=self.flow_steps,
                        generator=generator,
                        encoder_kv_states=kv,
                        encoder_attention_mask=encoder_mask,
                    )
                    actions = _postprocess_conditioned_actions(
                        outer=self.outer,
                        actions=actions,
                        stats=stats,
                        tag=tag,
                        action_dim=action_dim,
                        n_action_steps=n_action_steps,
                    )
            if hasattr(actions, "detach"):
                actions = actions.detach().float().cpu().numpy()
            if getattr(actions, "ndim", 0) == 3:
                actions = actions[0]
            rows = tuple(
                tuple(float(value) for value in row)
                for row in actions[:requested_steps]
            )
            return ActionChunk(
                actions=rows,
                execution_route=candidate.execution_route,
                requested_steps=requested_steps,
            )
