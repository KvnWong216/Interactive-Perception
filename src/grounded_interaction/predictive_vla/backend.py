"""Pinned MolmoAct2 interface: native per-layer KV, no new control bottleneck.

The pretrained backbone, processor and action expert remain upstream assets.
Geometry, history organization and predictive training are implemented here.
"""

import inspect
import threading
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .geometry import GeometryAdapter, backproject, native_patch_layout
from .native import ResidualLinear, discover_capabilities, postprocess_actions
from .types import PolicyContext

RESPONSE_PROMPT = (
    "\nBased only on the observed evidence and the task, is the task complete? "
    "If more physical interaction is needed, reply exactly CONTINUE. "
    "If complete, reply DONE: followed by the answer or completion statement.\n"
    "Response:"
)


def temporal_attention_bias(times, observation_tokens, dtype):
    """Block bidirectional observations, causal text/actions and causal time."""
    if times.ndim != 2 or observation_tokens.shape != times.shape:
        raise ValueError("temporal metadata must be aligned [B,S]")
    n = times.shape[1]
    causal = torch.ones((n, n), device=times.device, dtype=torch.bool).tril()
    same_observation = observation_tokens[:, :, None] & observation_tokens[:, None, :]
    same_time = times[:, :, None] == times[:, None, :]
    earlier = times[:, :, None] > times[:, None, :]
    allowed = earlier | (same_time & (causal | same_observation))
    return torch.where(allowed[:, None], 0.0, torch.finfo(dtype).min).to(dtype)


def _zero_last(module):
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)
    return module


class ContextAdapters(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.geometry = GeometryAdapter(width)
        self.time_view = _zero_last(nn.Linear(2, width))
        self.state = _zero_last(nn.Linear(8, width))
        self.action = _zero_last(nn.Linear(7, width))


@dataclass
class EncodedContext:
    embeddings: torch.Tensor
    ids: torch.Tensor
    times: torch.Tensor
    observation_tokens: torch.Tensor
    hidden: torch.Tensor
    kv: object
    encoder_mask: torch.Tensor


class NativeVLABackend:
    def __init__(self, *, model, processor, config):
        self.model, self.processor, self.config = model, processor, config
        self.outer, self.backbone = model, model.model
        self.tokenizer = processor.tokenizer
        self.capabilities = discover_capabilities(model)
        if config.prediction_steps > self.capabilities.max_action_horizon:
            raise ValueError("prediction_steps exceeds the native action horizon")
        self._lock = threading.RLock()
        self._enabled = True
        for p in model.parameters():
            p.requires_grad_(False)
        self.word_embeddings = model.get_input_embeddings()
        self.adapters = ContextAdapters(self.capabilities.hidden_dim).to(
            self.embedding_reference
        )
        if hasattr(self.backbone, "predictive_vla_adapters"):
            raise RuntimeError("predictive VLA adapters are already installed")
        self.backbone.predictive_vla_adapters = self.adapters
        blocks = self.backbone.transformer.blocks
        if len(blocks) < config.upper_vlm_layers:
            raise ValueError(
                "checkpoint has fewer layers than the requested adaptation"
            )
        targets = [
            (b.self_attn, name)
            for b in blocks[-config.upper_vlm_layers :]
            for name in ("att_proj", "attn_out")
        ]
        targets += [
            (b.mlp, name)
            for b in blocks[-config.upper_vlm_layers :]
            for name in ("ff_proj", "ff_out")
        ]
        for owner, name in targets:
            setattr(
                owner,
                name,
                ResidualLinear(
                    getattr(owner, name),
                    rank=config.lora_rank,
                    alpha=config.lora_alpha,
                    enabled=lambda: self._enabled,
                ),
            )
        self.backbone.action_expert.requires_grad_(config.train_action_expert)
        # Frozen teacher path is the native framewise vision encoder/connector.
        self.model.eval()

    @classmethod
    def from_pretrained(cls, config, *, device="cuda", local_path=None):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; real MolmoAct2 validation has not run"
            )
        source = str(local_path) if local_path else config.model_id
        source_options = {"local_files_only": True} if local_path else {}
        processor = AutoProcessor.from_pretrained(
            source,
            **source_options,
            trust_remote_code=True,
            extra_special_tokens={},
        )
        model = AutoModelForImageTextToText.from_pretrained(
            source,
            **source_options,
            trust_remote_code=True,
            torch_dtype=getattr(torch, config.dtype),
            low_cpu_mem_usage=True,
        ).to(device)
        return cls(model=model, processor=processor, config=config)

    @property
    def device(self):
        return self.embedding_reference.device

    @property
    def embedding_reference(self):
        # MolmoAct2 keeps original/new vocabulary tables separately, unlike
        # nn.Embedding: the pinned native module has no .weight attribute.
        return self.word_embeddings.embedding

    @contextmanager
    def native_mode(self):
        with self._lock:
            previous = self._enabled
            self._enabled = False
            try:
                yield
            finally:
                self._enabled = previous

    def reset(self):
        expert = self.backbone.action_expert
        for name in ("_modulation_cache_key", "_modulation_cache_value"):
            if hasattr(expert, name):
                setattr(expert, name, None)

    def train(self, mode=True):
        # Keep frozen visual features deterministic even while adapters train.
        self.model.eval()
        self.adapters.train(mode)
        self.backbone.action_expert.train(mode)

    def _ids(self, text):
        return torch.tensor(
            [self.tokenizer.encode(text, add_special_tokens=False)],
            device=self.device,
            dtype=torch.long,
        )

    def _native_inputs(self, observation, task):
        module = inspect.getmodule(type(self.outer))
        stats = self.outer._get_robot_stats()
        tag = stats.validate_tag("libero")
        metadata = stats.get_metadata(tag)
        state = np.asarray(
            stats.normalize_state(observation.state, tag), dtype=np.float32
        )
        discrete = module._build_discrete_state_string(
            state, int(self.outer.config.num_state_tokens)
        )
        text = module._build_robot_text(
            task=module._normalize_question_text(task),
            style="robot_action",
            discrete_state_string=discrete,
            setup_type=str(metadata.get("setup_type", "") or ""),
            control_mode=str(metadata.get("control_mode", "") or ""),
            add_setup_tokens=self.outer.config.add_setup_tokens,
            add_control_tokens=self.outer.config.add_control_tokens,
            num_images=2,
        )
        inputs = dict(
            self.processor(
                text=text,
                images=[c.rgb for c in observation.cameras],
                return_tensors="pt",
            )
        )
        return dict(self.outer._move_inputs_to_device(inputs, self.device))

    def _visual(self, inputs):
        images, pooling = self.backbone.merge_visual_inputs(
            **{
                k: inputs.get(k)
                for k in (
                    "input_ids",
                    "pixel_values",
                    "image_token_pooling",
                    "image_grids",
                    "image_num_crops",
                )
            }
        )
        # The teacher's weights never change and frames are encoded independently.
        with torch.no_grad():
            embeddings, features = self.backbone.build_input_embeddings(
                inputs["input_ids"], images, pooling
            )
        return embeddings, features

    def _layout(self, observation, inputs):
        grids = inputs["image_grids"].detach().cpu().numpy()
        pooling = inputs["image_token_pooling"].detach().cpu().numpy()
        if grids.shape != (2, 4):
            raise RuntimeError("expected two native image grids")
        offset, positions, xyz, observed, valid = 0, [], [], [], []
        for view, (frame, grid) in enumerate(zip(observation.cameras, grids)):
            n = int(grid[0] * grid[1] + grid[2] * grid[3])
            uv, mask = native_patch_layout(
                self.processor.image_processor,
                frame,
                grid,
                pooling[offset : offset + n],
            )
            points, seen = backproject(frame, uv)
            h, w = frame.rgb.shape[:2]
            level = np.r_[
                np.zeros(int(grid[0] * grid[1])), np.ones(int(grid[2] * grid[3]))
            ]
            positions.append(
                np.c_[
                    uv / np.array([max(w - 1, 1), max(h - 1, 1)]),
                    np.full(n, view),
                    level,
                ]
            )
            xyz.append(points)
            observed.append(seen & mask)
            valid.append(mask)
            offset += n
        if offset != len(pooling):
            raise RuntimeError("native patch ownership is not aligned")
        return tuple(
            torch.as_tensor(np.concatenate(v), device=self.device)
            for v in (positions, xyz, observed, valid)
        )

    def _frame(self, observation, task, *, age):
        native = self._native_inputs(observation, task)
        embeddings, features = self._visual(native)
        positions, xyz, observed, _valid = self._layout(observation, native)
        patch_mask = native["input_ids"].eq(self.outer.config.image_patch_id)
        if int(patch_mask.sum()) != len(positions) or features.shape[0] != len(
            positions
        ):
            raise RuntimeError("RGB features and geometry differ in patch order/count")
        # Time/view tags are attached to each corresponding patch, not a side stream.
        tags = torch.stack(
            (
                torch.full_like(positions[:, 2], age / self.config.total_steps),
                positions[:, 2],
            ),
            -1,
        )
        delta = self.adapters.time_view(tags.to(self.embedding_reference))
        if self.config.geometry:
            delta = delta + self.adapters.geometry(xyz, observed)
        embeddings = embeddings.clone()
        embeddings[patch_mask] += delta.to(embeddings)
        return native, embeddings

    def _forward(self, embeddings, ids, times, observed):
        if ids.shape[1] > self.config.max_context_tokens:
            raise ValueError(
                "context exceeds max_context_tokens; reduce history/crops explicitly"
            )
        attention = temporal_attention_bias(times, observed, embeddings.dtype)
        output = self.backbone(
            inputs_embeds=embeddings,
            attention_mask=attention,
            position_ids=torch.arange(ids.shape[1], device=self.device)[None],
            use_cache=True,
            output_hidden_states=False,
        )
        kv = self.backbone._extract_kv_states(output.past_key_values)
        mask = self.backbone._get_encoder_attention_mask(ids, torch.ones_like(ids))
        return EncodedContext(
            embeddings, ids, times, observed, output.last_hidden_state, kv, mask
        )

    def encode(self, context: PolicyContext, *, answer=False):
        if (
            not isinstance(context, PolicyContext)
            or len(context.observations) > self.config.history_frames
        ):
            raise ValueError(
                "policy context exceeds the configured real history window"
            )
        parts, ids, times, observed = [], [], [], []

        def append(e, i, step, obs=False):
            parts.append(e)
            ids.append(i)
            times.append(torch.full_like(i, step))
            observed.append(
                torch.full_like(i, obs, dtype=torch.bool)
                if isinstance(obs, bool)
                else obs
            )

        def text(s, step):
            i = self._ids(s)
            append(self.word_embeddings(i), i, step)

        for index, observation in enumerate(context.observations):
            current = index == len(context.observations) - 1
            native, e = self._frame(
                observation, context.task, age=context.current.step - observation.step
            )
            row = native["input_ids"]
            visual = native.get(
                "token_type_ids", row.eq(self.outer.config.image_patch_id)
            ).bool()
            if current:
                trigger = row.eq(self.capabilities.action_trigger_id).nonzero(
                    as_tuple=False
                )
                if len(trigger) != 1:
                    raise RuntimeError("native prompt must have exactly one AE trigger")
                end = int(trigger[0, 1]) + (0 if answer else 1)
                append(e[:, :end], row[:, :end], observation.step, visual[:, :end])
                if answer:
                    text(RESPONSE_PROMPT, observation.step)
            else:
                # Keep native visual delimiters and patch embeddings from this frame.
                start_user = self._ids("<|im_start|>user\n")[0].tolist()
                values = row[0].tolist()
                starts = [
                    j
                    for j in range(len(values) - len(start_user) + 1)
                    if values[j : j + len(start_user)] == start_user
                ]
                if not starts:
                    raise RuntimeError("native user boundary is missing")
                text(f"Observed at step {observation.step}:\n", observation.step)
                append(
                    e[:, : starts[-1]],
                    row[:, : starts[-1]],
                    observation.step,
                    visual[:, : starts[-1]],
                )
                stats = self.outer._get_robot_stats()
                s = torch.as_tensor(
                    stats.normalize_state(observation.state, "libero"),
                    device=self.device,
                    dtype=e.dtype,
                )
                placeholder = self._ids(" ")[:, :1]
                # Newly inserted tokens need a nonzero native embedding.
                # A standalone zero residual enters RMSNorm at its epsilon
                # floor and creates enormous first-step gradients.
                base_token = self.word_embeddings(placeholder)
                state_token = base_token + self.adapters.state(s).reshape(1, 1, -1)
                append(state_token, placeholder, observation.step, True)
                next_step = context.observations[index + 1].step
                for applied in context.applied_actions:
                    if observation.step <= applied.step < next_step:
                        a = self.normalize_actions(applied.value[None])[0]
                        token = base_token + self.adapters.action(a).reshape(1, 1, -1)
                        token = token + self.adapters.time_view(
                            torch.tensor(
                                [
                                    (context.current.step - applied.step - 1)
                                    / self.config.total_steps,
                                    -1.0,
                                ],
                                device=self.device,
                                dtype=e.dtype,
                            )
                        ).reshape(1, 1, -1)
                        append(token, placeholder, applied.step + 1)
        return self._forward(
            *(torch.cat(items, 1) for items in (parts, ids, times, observed))
        )

    def normalize_actions(self, actions):
        a = np.asarray(actions, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != 7 or not np.isfinite(a).all():
            raise ValueError("real actions must be finite T x 7")
        stats = self.outer._get_robot_stats()
        normalizer = stats.action_normalizers.get("libero")
        if normalizer is not None:
            a = normalizer.normalize(a)
        return torch.as_tensor(
            a, device=self.device, dtype=self.embedding_reference.dtype
        )

    def target(self, observation, task):
        with self.native_mode(), torch.no_grad():
            inputs = self._native_inputs(observation, task)
            _, features = self._visual(inputs)
            positions, _, _, valid = self._layout(observation, inputs)
        return features.detach()[None].float(), positions[None].float(), valid[None]

    def flow_loss(self, encoded, actions):
        source = self.normalize_actions(actions)
        stats = self.outer._get_robot_stats()
        horizon = int(
            stats.get_action_horizon("libero") or self.capabilities.max_action_horizon
        )
        offset = int(self.outer.config.n_obs_steps) - 1
        if offset != 0:
            raise RuntimeError(
                "training alignment requires the pinned single-observation action window"
            )
        if not 0 < len(source) <= horizon:
            raise ValueError("recorded action prefix exceeds the native horizon")
        action = source.new_zeros(
            (self.config.num_flow_samples, horizon, self.capabilities.max_action_dim)
        )
        action[:, : len(source), :7] = source
        valid = torch.zeros_like(action, dtype=torch.bool)
        valid[:, : len(source), :7] = True
        cfg = self.outer.config
        beta = torch.distributions.Beta(
            float(cfg.flow_matching_beta_alpha), float(cfg.flow_matching_beta_beta)
        )
        t = beta.sample((self.config.num_flow_samples,)).to(action)
        lo = float(cfg.flow_matching_time_offset)
        hi = min(
            float(cfg.flow_matching_cutoff), lo + float(cfg.flow_matching_time_scale)
        )
        t = lo + (hi - lo) * t
        noise = torch.randn_like(action) * valid
        noisy = (1 - t[:, None, None]) * noise + t[:, None, None] * action
        prediction = self.backbone.action_expert(
            noisy,
            t,
            encoder_kv_states=encoded.kv,
            encoder_attention_mask=encoded.encoder_mask,
            action_attention_mask=valid.any(-1),
            state_embeddings=None,
        )
        if prediction.shape != action.shape or not torch.isfinite(prediction).all():
            raise RuntimeError("native action expert produced invalid flow predictions")
        return ((prediction.float() - (action - noise).float()).square()[valid]).mean()

    def language_loss(self, context, response):
        from .runtime import parse_response

        parse_response(response)
        encoded = self.encode(context, answer=True)
        suffix = self._ids(response + "<|im_end|>")
        all_ids = torch.cat((encoded.ids, suffix), 1)
        embeddings = torch.cat((encoded.embeddings, self.word_embeddings(suffix)), 1)
        times = torch.cat(
            (encoded.times, torch.full_like(suffix, context.current.step)), 1
        )
        observed = torch.cat(
            (encoded.observation_tokens, torch.zeros_like(suffix, dtype=torch.bool)), 1
        )
        output = self._forward(embeddings, all_ids, times, observed)
        start = encoded.ids.shape[1] - 1
        logits = self.outer.lm_head(output.hidden[:, start:-1]).float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), suffix.reshape(-1))

    @torch.no_grad()
    def respond(self, context):
        self.train(False)
        encoded = self.encode(context, answer=True)
        output_ids = []
        boundary = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        for _ in range(self.config.response_tokens):
            token = int(self.outer.lm_head(encoded.hidden[:, -1]).argmax(-1).item())
            output_ids.append(token)
            if output_ids[-len(boundary) :] == boundary:
                return self.tokenizer.decode(
                    output_ids[: -len(boundary)], skip_special_tokens=False
                ).strip()
            if token == self.tokenizer.eos_token_id:
                return self.tokenizer.decode(
                    output_ids[:-1], skip_special_tokens=False
                ).strip()
            ids = torch.tensor([[token]], device=self.device)
            encoded = self._forward(
                torch.cat((encoded.embeddings, self.word_embeddings(ids)), 1),
                torch.cat((encoded.ids, ids), 1),
                torch.cat(
                    (encoded.times, torch.full_like(ids, context.current.step)), 1
                ),
                torch.cat(
                    (
                        encoded.observation_tokens,
                        torch.zeros_like(ids, dtype=torch.bool),
                    ),
                    1,
                ),
            )
        # An unterminated/truncated answer is not a completion claim.
        return "INVALID_UNTERMINATED_RESPONSE"

    @torch.no_grad()
    def act(self, context, *, seed, steps):
        if not self._enabled:
            return self.native_act(context, seed=seed, steps=steps)
        self.train(False)
        encoded = self.encode(context)
        stats = self.outer._get_robot_stats()
        horizon = int(
            stats.get_action_horizon("libero") or self.capabilities.max_action_horizon
        )
        n = int(stats.get_n_action_steps("libero") or horizon)
        if not 1 <= steps <= n:
            raise ValueError("requested prefix exceeds native executable horizon")
        padding = self.outer._build_action_dim_is_pad(
            action_dim=7,
            max_action_dim=self.capabilities.max_action_dim,
            batch_size=1,
            device=self.device,
        )
        actions = self.backbone.generate_actions_from_inputs(
            input_ids=encoded.ids,
            attention_mask=torch.ones_like(encoded.ids),
            action_dim_is_pad=padding,
            action_horizon=horizon,
            num_steps=self.config.flow_steps,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            encoder_kv_states=encoded.kv,
            encoder_attention_mask=encoded.encoder_mask,
        )
        actions = postprocess_actions(
            outer=self.outer,
            actions=actions,
            stats=stats,
            tag="libero",
            action_dim=7,
            n_action_steps=n,
        )
        if hasattr(actions, "detach"):
            actions = actions.detach().float().cpu().numpy()
        actions = np.asarray(actions)
        return (actions[0] if actions.ndim == 3 else actions)[:steps]

    @torch.no_grad()
    def native_act(self, context, *, seed, steps):
        self.train(False)
        with self.native_mode():
            output = self.outer.predict_action(
                processor=self.processor,
                images=[c.rgb for c in context.current.cameras],
                task=context.task,
                state=context.current.state,
                norm_tag="libero",
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=self.config.flow_steps,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                normalize_language=True,
                enable_cuda_graph=False,
            )
        actions = output.actions
        if hasattr(actions, "detach"):
            actions = actions.detach().float().cpu().numpy()
        return (actions[0] if actions.ndim == 3 else actions)[:steps]
