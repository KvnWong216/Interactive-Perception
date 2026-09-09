"""Predictive-state readout used by the PSR-VLA v1 path.

This module deliberately has a narrow information boundary.  The predictor
reads only contextualized predictive-state tokens ``B``, ordinary intent token
IDs, and an execution-route ID.  It cannot accept the VLM history tokens,
contextual hidden states from intent generation, action-decoder caches, or
future observations.

``C`` is a supervised failure indicator under a frozen execution protocol.
``E`` is an auxiliary distribution over frozen future visual patch features.
Neither output is a hand-designed uncertainty score and the evidence decoder
is not used as an online image-rollout model in v1.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

try:  # Keep the repository importable when the learned extra is not installed.
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - exercised in torch-free installations.
    torch = None
    F = None
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


_Module = nn.Module if nn is not None else object


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError("PSR predictive models require the PyTorch extra")


def _check_bool_tensor(value: Tensor, *, name: str, shape: tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype is not torch.bool:
        raise TypeError(f"{name} must be a bool tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")


@dataclasses.dataclass(frozen=True)
class PredictiveOutput:
    """Per-candidate predictions with no future-derived inputs.

    Shapes use ``N`` for batch, ``M`` for candidates, ``K`` for mixture
    components, ``J`` for native visual positions, and ``P`` for the frozen
    projected target dimension.
    """

    failure_logits: Tensor  # [N, M]
    mixture_logits: Tensor  # [N, M, K]
    evidence_mean: Tensor  # [N, M, K, J, P]
    evidence_logstd: Tensor  # [N, M, K, J, P]
    candidate_valid_mask: Tensor  # [N, M]
    intent_queries: Tensor | None = None  # [N, M, d], diagnostic only
    readout_states: Tensor | None = None  # [N, M, d], diagnostic only

    @property
    def expected_failure(self) -> Tensor:
        """Expected binary task cost used for online finite-candidate ranking."""

        return torch.sigmoid(self.failure_logits)


@dataclasses.dataclass(frozen=True)
class MaskedLoss:
    """A loss that distinguishes missing supervision from a numeric zero.

    ``value`` is ``None`` when there are no valid supervised items.  Training
    code must skip its optimizer update rather than treating that batch as a
    zero-loss observation.
    """

    value: Tensor | None
    per_item: Tensor
    valid_items: Tensor
    valid_item_count: int

    @property
    def has_supervision(self) -> bool:
        return self.value is not None

    def require_value(self) -> Tensor:
        if self.value is None:
            raise RuntimeError("this batch contains no valid supervision")
        return self.value


class FrozenOrthogonalProjection(_Module):
    """Checkpointed random orthogonal projection for fixed evidence targets."""

    def __init__(
        self,
        *,
        input_dim: int,
        requested_output_dim: int = 128,
        seed: int = 17,
    ) -> None:
        _require_torch()
        super().__init__()
        if input_dim < 1 or requested_output_dim < 1:
            raise ValueError("projection dimensions must be positive")
        output_dim = min(int(input_dim), int(requested_output_dim))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        sample = torch.randn(input_dim, output_dim, generator=generator)
        q, r = torch.linalg.qr(sample, mode="reduced")
        # Resolve QR's column-sign ambiguity so serialized construction is
        # deterministic for a given torch implementation and seed.
        diagonal = torch.diagonal(r)
        signs = torch.where(
            diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal)
        )
        projection = (q * signs.unsqueeze(0)).to(dtype=torch.float32)
        self.input_dim = int(input_dim)
        self.output_dim = output_dim
        self.seed = int(seed)
        self.register_buffer("projection", projection, persistent=True)

    def forward(self, features: Tensor) -> Tensor:
        if not isinstance(features, torch.Tensor) or features.ndim < 2:
            raise ValueError("features must be a tensor with a final channel axis")
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected feature dimension {self.input_dim}, got {features.shape[-1]}"
            )
        return features @ self.projection.to(dtype=features.dtype)


class FrozenEvidenceNormalizer(_Module):
    """Training-split channel statistics for projected evidence targets."""

    def __init__(self, feature_dim: int, *, minimum_scale: float = 1e-6) -> None:
        _require_torch()
        super().__init__()
        if feature_dim < 1 or not math.isfinite(minimum_scale) or minimum_scale <= 0:
            raise ValueError("feature_dim and minimum_scale must be positive")
        self.feature_dim = int(feature_dim)
        self.minimum_scale = float(minimum_scale)
        self.register_buffer("mean", torch.zeros(feature_dim), persistent=True)
        self.register_buffer("scale", torch.ones(feature_dim), persistent=True)
        self.register_buffer(
            "constant_channels",
            torch.zeros(feature_dim, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer("statistics_ready", torch.tensor(False), persistent=True)

    def fit(self, features: Tensor, valid_patch_mask: Tensor) -> None:
        """Fit once from valid training patches and then freeze the statistics."""

        if bool(self.statistics_ready.item()):
            raise RuntimeError("evidence normalization statistics are already frozen")
        if not isinstance(features, torch.Tensor) or features.ndim < 2:
            raise ValueError("features must have shape [..., patch, channel]")
        if features.shape[-1] != self.feature_dim:
            raise ValueError("feature channel dimension does not match the normalizer")
        _check_bool_tensor(
            valid_patch_mask,
            name="valid_patch_mask",
            shape=tuple(features.shape[:-1]),
        )
        selected = features.detach()[valid_patch_mask]
        if selected.numel() == 0:
            raise ValueError("cannot fit evidence normalization without valid patches")
        if not bool(torch.isfinite(selected).all()):
            raise ValueError("valid evidence features must be finite")
        selected = selected.to(dtype=torch.float64)
        mean = selected.mean(dim=0)
        raw_scale = selected.std(dim=0, correction=0)
        constant = raw_scale < self.minimum_scale
        scale = raw_scale.clamp_min(self.minimum_scale)
        with torch.no_grad():
            self.mean.copy_(mean.to(device=self.mean.device, dtype=self.mean.dtype))
            self.scale.copy_(scale.to(device=self.scale.device, dtype=self.scale.dtype))
            self.constant_channels.copy_(
                constant.to(device=self.constant_channels.device)
            )
            self.statistics_ready.fill_(True)

    def forward(self, features: Tensor) -> Tensor:
        if not bool(self.statistics_ready.item()):
            raise RuntimeError(
                "training-split evidence statistics have not been fitted"
            )
        if (
            not isinstance(features, torch.Tensor)
            or features.shape[-1] != self.feature_dim
        ):
            raise ValueError("feature channel dimension does not match the normalizer")
        return (features - self.mean.to(dtype=features.dtype)) / self.scale.to(
            dtype=features.dtype
        )


class IntentOnlyEncoder(_Module):
    """Encode ordinary intent IDs and route, without any scene-state input."""

    def __init__(
        self,
        *,
        word_embeddings: _Module,
        native_embedding_dim: int,
        readout_dim: int = 512,
        num_heads: int = 8,
        encoder_layers: int = 1,
        max_intent_tokens: int = 32,
        route_count: int = 2,
    ) -> None:
        _require_torch()
        super().__init__()
        if native_embedding_dim < 1 or readout_dim < 1 or max_intent_tokens < 1:
            raise ValueError("encoder dimensions must be positive")
        if num_heads < 1 or readout_dim % num_heads:
            raise ValueError("readout_dim must be divisible by num_heads")
        if encoder_layers < 1 or route_count < 2:
            raise ValueError("encoder_layers must be positive and route_count >= 2")
        if not isinstance(word_embeddings, nn.Module):
            raise TypeError("word_embeddings must be a torch module")
        self.word_embeddings = word_embeddings
        for parameter in self.word_embeddings.parameters():
            parameter.requires_grad_(False)
        self.native_embedding_dim = int(native_embedding_dim)
        self.readout_dim = int(readout_dim)
        self.max_intent_tokens = int(max_intent_tokens)
        self.route_count = int(route_count)
        self.word_projection = nn.Linear(native_embedding_dim, readout_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, readout_dim))
        self.position_embedding = nn.Parameter(
            torch.empty(1, max_intent_tokens + 1, readout_dim)
        )
        self.route_embedding = nn.Embedding(route_count, readout_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=readout_dim,
            nhead=num_heads,
            dim_feedforward=4 * readout_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=encoder_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(readout_dim)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def train(self, mode: bool = True) -> IntentOnlyEncoder:
        super().train(mode)
        # A shared native embedding table is a fixed feature source in this
        # module.  Keep it in eval mode even while the PSR heads train.
        self.word_embeddings.eval()
        return self

    def forward(
        self,
        intent_token_ids: Tensor,
        intent_token_mask: Tensor,
        route_ids: Tensor,
        candidate_valid_mask: Tensor,
    ) -> Tensor:
        """Return ``[N,M,d]`` queries from the only four permitted tensors."""

        if not isinstance(intent_token_ids, torch.Tensor) or intent_token_ids.ndim != 3:
            raise ValueError(
                "intent_token_ids must have shape [batch, candidate, token]"
            )
        n, m, length = intent_token_ids.shape
        if length > self.max_intent_tokens:
            raise ValueError(
                f"intent length {length} exceeds frozen maximum {self.max_intent_tokens}"
            )
        _check_bool_tensor(
            intent_token_mask,
            name="intent_token_mask",
            shape=(n, m, length),
        )
        _check_bool_tensor(
            candidate_valid_mask,
            name="candidate_valid_mask",
            shape=(n, m),
        )
        if not isinstance(route_ids, torch.Tensor) or tuple(route_ids.shape) != (n, m):
            raise ValueError("route_ids must have shape [batch, candidate]")
        if intent_token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("intent_token_ids must contain integer token IDs")
        if route_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("route_ids must contain integer route IDs")
        invalid_registered_route = (
            (route_ids < 0) | (route_ids >= self.route_count)
        ) & candidate_valid_mask
        if bool(invalid_registered_route.any()):
            raise ValueError("route_ids contain an unregistered execution route")
        if not (
            intent_token_ids.device
            == intent_token_mask.device
            == route_ids.device
            == candidate_valid_mask.device
        ):
            raise ValueError("all intent inputs must be on one device")

        flat_ids = intent_token_ids.reshape(n * m, length)
        flat_mask = intent_token_mask.reshape(n * m, length)
        flat_valid = candidate_valid_mask.reshape(n * m)
        flat_route = torch.where(
            flat_valid,
            route_ids.reshape(n * m),
            torch.zeros_like(flat_valid, dtype=route_ids.dtype),
        )
        # Do not let the frozen table receive gradients; the learned projection
        # still receives gradients from the detached embeddings.
        with torch.no_grad():
            native_words = self.word_embeddings(flat_ids)
        if not isinstance(native_words, torch.Tensor) or native_words.shape != (
            n * m,
            length,
            self.native_embedding_dim,
        ):
            raise RuntimeError("native word embedding returned an incompatible shape")
        words = self.word_projection(
            native_words.detach().to(dtype=self.word_projection.weight.dtype)
        )
        cls = self.cls_token.expand(n * m, -1, -1)
        sequence = torch.cat((cls, words), dim=1)
        sequence = sequence + self.position_embedding[:, : length + 1]
        sequence = sequence + self.route_embedding(flat_route).unsqueeze(1)
        padding = torch.cat(
            (
                torch.zeros(n * m, 1, dtype=torch.bool, device=flat_mask.device),
                ~flat_mask,
            ),
            dim=1,
        )
        encoded = self.encoder(sequence, src_key_padding_mask=padding)
        query = self.output_norm(encoded[:, 0]).reshape(n, m, self.readout_dim)
        return query * candidate_valid_mask.unsqueeze(-1)


class _CrossAttentionReadoutBlock(_Module):
    def __init__(self, *, hidden_dim: int, num_heads: int) -> None:
        _require_torch()
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query: Tensor, state_tokens: Tensor) -> Tensor:
        attended, _ = self.cross_attention(
            query=query,
            key=state_tokens,
            value=state_tokens,
            need_weights=False,
        )
        query = self.attention_norm(query + attended)
        return self.output_norm(query + self.feedforward(query))


class PredictiveStateReadout(_Module):
    """Read six contextualized state tokens independently for each intent."""

    def __init__(
        self,
        *,
        state_dim: int,
        readout_dim: int = 512,
        num_heads: int = 8,
        predictor_layers: int = 2,
        num_state_tokens: int = 6,
    ) -> None:
        _require_torch()
        super().__init__()
        if (
            min(state_dim, readout_dim, num_heads, predictor_layers, num_state_tokens)
            < 1
        ):
            raise ValueError("readout dimensions must be positive")
        if readout_dim % num_heads:
            raise ValueError("readout_dim must be divisible by num_heads")
        self.state_dim = int(state_dim)
        self.readout_dim = int(readout_dim)
        self.num_state_tokens = int(num_state_tokens)
        self.state_projection = nn.Sequential(
            nn.Linear(state_dim, readout_dim), nn.LayerNorm(readout_dim)
        )
        self.blocks = nn.ModuleList(
            _CrossAttentionReadoutBlock(hidden_dim=readout_dim, num_heads=num_heads)
            for _ in range(predictor_layers)
        )

    def forward(
        self,
        state_tokens: Tensor,
        intent_queries: Tensor,
        candidate_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(state_tokens, torch.Tensor) or state_tokens.ndim != 3:
            raise ValueError(
                "state_tokens must have shape [batch, state_token, channel]"
            )
        n, b_count, state_dim = state_tokens.shape
        if b_count != self.num_state_tokens or state_dim != self.state_dim:
            raise ValueError(
                f"expected state_tokens [N,{self.num_state_tokens},{self.state_dim}]"
            )
        if not isinstance(intent_queries, torch.Tensor) or intent_queries.ndim != 3:
            raise ValueError(
                "intent_queries must have shape [batch, candidate, channel]"
            )
        if intent_queries.shape[0] != n or intent_queries.shape[2] != self.readout_dim:
            raise ValueError("intent query dimensions do not match the readout")
        m = intent_queries.shape[1]
        _check_bool_tensor(
            candidate_valid_mask,
            name="candidate_valid_mask",
            shape=(n, m),
        )
        projected_state = self.state_projection(state_tokens)
        # Flatten candidates into the batch.  No operation has an M sequence
        # axis, so distinct candidates can never self-attend to one another.
        memory = (
            projected_state.unsqueeze(1)
            .expand(n, m, b_count, self.readout_dim)
            .reshape(n * m, b_count, self.readout_dim)
        )
        query = intent_queries.reshape(n * m, 1, self.readout_dim)
        for block in self.blocks:
            query = block(query, memory)
        readout = query[:, 0].reshape(n, m, self.readout_dim)
        readout = readout * candidate_valid_mask.unsqueeze(-1)
        return readout, projected_state


class PredictiveStateModel(_Module):
    """PSR-v1 intent-conditioned ``q(E,C | B,U,route)`` model."""

    def __init__(
        self,
        *,
        word_embeddings: _Module,
        native_embedding_dim: int,
        state_dim: int,
        patch_camera_ids: Sequence[int],
        readout_dim: int = 512,
        num_heads: int = 8,
        encoder_layers: int = 1,
        predictor_layers: int = 2,
        num_state_tokens: int = 6,
        max_intent_tokens: int = 32,
        route_count: int = 2,
        mixture_components: int = 4,
        evidence_dim: int = 128,
        minimum_logstd: float = -5.0,
        maximum_logstd: float = 2.0,
    ) -> None:
        _require_torch()
        super().__init__()
        if mixture_components < 1 or evidence_dim < 1:
            raise ValueError("mixture_components and evidence_dim must be positive")
        if not math.isfinite(minimum_logstd) or not math.isfinite(maximum_logstd):
            raise ValueError("log-standard-deviation bounds must be finite")
        if minimum_logstd >= maximum_logstd:
            raise ValueError("minimum_logstd must be less than maximum_logstd")
        camera_ids = tuple(int(camera_id) for camera_id in patch_camera_ids)
        if not camera_ids or min(camera_ids) < 0:
            raise ValueError(
                "patch_camera_ids must be a non-empty non-negative sequence"
            )
        self.intent_encoder = IntentOnlyEncoder(
            word_embeddings=word_embeddings,
            native_embedding_dim=native_embedding_dim,
            readout_dim=readout_dim,
            num_heads=num_heads,
            encoder_layers=encoder_layers,
            max_intent_tokens=max_intent_tokens,
            route_count=route_count,
        )
        self.readout = PredictiveStateReadout(
            state_dim=state_dim,
            readout_dim=readout_dim,
            num_heads=num_heads,
            predictor_layers=predictor_layers,
            num_state_tokens=num_state_tokens,
        )
        self.failure_head = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, readout_dim),
            nn.GELU(),
            nn.Linear(readout_dim, 1),
        )
        self.mixture_head = nn.Sequential(
            nn.LayerNorm(readout_dim), nn.Linear(readout_dim, mixture_components)
        )
        self.num_positions = len(camera_ids)
        self.evidence_dim = int(evidence_dim)
        self.mixture_components = int(mixture_components)
        self.minimum_logstd = float(minimum_logstd)
        self.maximum_logstd = float(maximum_logstd)
        self.position_queries = nn.Parameter(
            torch.empty(1, self.num_positions, readout_dim)
        )
        self.camera_embedding = nn.Embedding(max(camera_ids) + 1, readout_dim)
        self.register_buffer(
            "patch_camera_ids",
            torch.tensor(camera_ids, dtype=torch.long),
            persistent=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=readout_dim,
            nhead=num_heads,
            dim_feedforward=4 * readout_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.evidence_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.evidence_parameter_head = nn.Linear(
            readout_dim, 2 * mixture_components * evidence_dim
        )
        nn.init.normal_(self.position_queries, std=0.02)

    def forward(
        self,
        state_tokens: Tensor,
        intent_token_ids: Tensor,
        intent_token_mask: Tensor,
        route_ids: Tensor,
        candidate_valid_mask: Tensor,
    ) -> PredictiveOutput:
        """Predict outcomes; these are the complete permitted forward inputs."""

        intent_queries = self.intent_encoder(
            intent_token_ids,
            intent_token_mask,
            route_ids,
            candidate_valid_mask,
        )
        readout, projected_state = self.readout(
            state_tokens, intent_queries, candidate_valid_mask
        )
        n, m, dim = readout.shape
        failure_logits = self.failure_head(readout).squeeze(-1)
        mixture_logits = self.mixture_head(readout)

        # Each candidate gets its own [R, projected B] memory.  Fixed position
        # and camera queries never consume future feature values or masks.
        memory = torch.cat(
            (
                readout.unsqueeze(2),
                projected_state.unsqueeze(1).expand(
                    n, m, projected_state.shape[1], projected_state.shape[2]
                ),
            ),
            dim=2,
        ).reshape(n * m, projected_state.shape[1] + 1, dim)
        position = self.position_queries + self.camera_embedding(
            self.patch_camera_ids
        ).unsqueeze(0)
        position = (
            position.unsqueeze(1)
            .expand(n, m, self.num_positions, dim)
            .reshape(n * m, self.num_positions, dim)
        )
        decoded = self.evidence_decoder(tgt=position, memory=memory)
        parameters = self.evidence_parameter_head(decoded)
        parameters = parameters.reshape(
            n,
            m,
            self.num_positions,
            2,
            self.mixture_components,
            self.evidence_dim,
        ).permute(0, 1, 3, 4, 2, 5)
        evidence_mean = parameters[:, :, 0]
        evidence_logstd = parameters[:, :, 1].clamp(
            self.minimum_logstd, self.maximum_logstd
        )
        valid = candidate_valid_mask
        failure_logits = failure_logits.masked_fill(~valid, 0.0)
        mixture_logits = mixture_logits.masked_fill(~valid.unsqueeze(-1), 0.0)
        evidence_mean = evidence_mean * valid[:, :, None, None, None]
        evidence_logstd = evidence_logstd * valid[:, :, None, None, None]
        return PredictiveOutput(
            failure_logits=failure_logits,
            mixture_logits=mixture_logits,
            evidence_mean=evidence_mean,
            evidence_logstd=evidence_logstd,
            candidate_valid_mask=valid,
            intent_queries=intent_queries,
            readout_states=readout,
        )


def evidence_mixture_nll(
    prediction: PredictiveOutput,
    targets: Tensor,
    valid_dimension_mask: Tensor,
    *,
    supervised_candidates: Tensor | None = None,
    minimum_logstd: float = -5.0,
    maximum_logstd: float = 2.0,
) -> MaskedLoss:
    """Global-component diagonal-Gaussian NLL for real future evidence.

    The mixture component is selected once for the entire set of valid
    patch/channel dimensions for a candidate.  It is *not* independently
    selected per patch.  Each candidate's NLL is normalized by its number of
    valid scalar target dimensions before valid candidates are averaged.
    """

    _require_torch()
    if (
        not math.isfinite(minimum_logstd)
        or not math.isfinite(maximum_logstd)
        or minimum_logstd >= maximum_logstd
    ):
        raise ValueError("log-standard-deviation bounds must be finite and ordered")
    mean = prediction.evidence_mean
    logstd = prediction.evidence_logstd
    if mean.ndim != 5 or logstd.shape != mean.shape:
        raise ValueError("evidence parameters must have shape [N,M,K,J,P]")
    n, m, components, positions, feature_dim = mean.shape
    if tuple(prediction.mixture_logits.shape) != (n, m, components):
        raise ValueError("mixture logits do not match evidence parameters")
    if tuple(targets.shape) != (n, m, positions, feature_dim):
        raise ValueError("targets must have shape [N,M,J,P]")
    if valid_dimension_mask.dtype is not torch.bool:
        raise TypeError("valid_dimension_mask must be boolean")
    if tuple(valid_dimension_mask.shape) == (n, m, positions):
        valid_dimensions = valid_dimension_mask.unsqueeze(-1).expand(
            n, m, positions, feature_dim
        )
    elif tuple(valid_dimension_mask.shape) == (n, m, positions, feature_dim):
        valid_dimensions = valid_dimension_mask
    else:
        raise ValueError("valid_dimension_mask must have shape [N,M,J] or [N,M,J,P]")
    if supervised_candidates is None:
        supervised_candidates = torch.ones(
            n, m, dtype=torch.bool, device=targets.device
        )
    _check_bool_tensor(
        supervised_candidates,
        name="supervised_candidates",
        shape=(n, m),
    )
    if not (
        mean.device
        == logstd.device
        == prediction.mixture_logits.device
        == prediction.candidate_valid_mask.device
        == targets.device
        == valid_dimensions.device
        == supervised_candidates.device
    ):
        raise ValueError("all evidence-loss tensors must share one device")
    if (
        not bool(torch.isfinite(mean).all())
        or bool(torch.isnan(logstd).any())
        or not bool(torch.isfinite(prediction.mixture_logits).all())
    ):
        raise ValueError("predicted evidence parameters must be finite")
    if bool(valid_dimensions.any()) and not bool(
        torch.isfinite(targets[valid_dimensions]).all()
    ):
        raise ValueError("valid future evidence targets must be finite")

    valid_dimensions = valid_dimensions & supervised_candidates[:, :, None, None]
    valid_dimensions = (
        valid_dimensions & prediction.candidate_valid_mask[:, :, None, None]
    )
    dimension_count = valid_dimensions.sum(dim=(-2, -1))
    valid_items = dimension_count > 0
    safe_targets = torch.where(valid_dimensions, targets, torch.zeros_like(targets))
    target32 = safe_targets.float().unsqueeze(2)
    mean32 = mean.float()
    logstd32 = logstd.float().clamp(minimum_logstd, maximum_logstd)
    inv_variance = torch.exp(-2.0 * logstd32)
    log_density = -0.5 * (
        (target32 - mean32).square() * inv_variance
        + 2.0 * logstd32
        + math.log(2.0 * math.pi)
    )
    log_density = log_density * valid_dimensions.unsqueeze(2)
    component_log_likelihood = log_density.sum(dim=(-2, -1))
    log_weights = F.log_softmax(prediction.mixture_logits.float(), dim=-1)
    item_log_likelihood = torch.logsumexp(
        log_weights + component_log_likelihood, dim=-1
    )
    denominator = dimension_count.clamp_min(1).float()
    normalized_nll = -item_log_likelihood / denominator
    per_item = torch.where(
        valid_items,
        normalized_nll,
        torch.full_like(normalized_nll, torch.nan),
    )
    valid_count = int(valid_items.sum().item())
    value = per_item[valid_items].mean() if valid_count else None
    return MaskedLoss(
        value=value,
        per_item=per_item,
        valid_items=valid_items,
        valid_item_count=valid_count,
    )


def failure_bce_loss(
    failure_logits: Tensor,
    failure_targets: Tensor,
    supervised_candidates: Tensor,
    *,
    candidate_valid_mask: Tensor | None = None,
) -> MaskedLoss:
    """Binary failure NLL with explicit support for partially missing labels."""

    _require_torch()
    if not isinstance(failure_logits, torch.Tensor) or failure_logits.ndim != 2:
        raise ValueError("failure_logits must have shape [N,M]")
    shape = tuple(failure_logits.shape)
    if (
        not isinstance(failure_targets, torch.Tensor)
        or tuple(failure_targets.shape) != shape
    ):
        raise ValueError("failure_targets must match failure_logits")
    _check_bool_tensor(
        supervised_candidates,
        name="supervised_candidates",
        shape=shape,
    )
    if candidate_valid_mask is None:
        candidate_valid_mask = torch.ones_like(supervised_candidates)
    _check_bool_tensor(
        candidate_valid_mask,
        name="candidate_valid_mask",
        shape=shape,
    )
    if not (
        failure_logits.device
        == failure_targets.device
        == supervised_candidates.device
        == candidate_valid_mask.device
    ):
        raise ValueError("all failure-loss tensors must share one device")
    valid_items = supervised_candidates & candidate_valid_mask
    if bool(valid_items.any()):
        selected_targets = failure_targets[valid_items]
        if not bool(torch.isfinite(selected_targets).all()) or bool(
            ((selected_targets != 0) & (selected_targets != 1)).any()
        ):
            raise ValueError("valid failure targets must be finite binary values")
    safe_targets = torch.where(
        valid_items, failure_targets, torch.zeros_like(failure_targets)
    )
    raw = F.binary_cross_entropy_with_logits(
        failure_logits, safe_targets.to(dtype=failure_logits.dtype), reduction="none"
    )
    per_item = torch.where(valid_items, raw, torch.full_like(raw, torch.nan))
    valid_count = int(valid_items.sum().item())
    value = per_item[valid_items].mean() if valid_count else None
    return MaskedLoss(
        value=value,
        per_item=per_item,
        valid_items=valid_items,
        valid_item_count=valid_count,
    )


@dataclasses.dataclass(frozen=True)
class EvidenceSeparabilityReport:
    """Target-side diagnostic for paired decisive future observations."""

    pair_rms_distance: Tensor
    valid_pairs: Tensor
    mean_rms_distance: float | None


def paired_evidence_separability(
    first: Tensor,
    second: Tensor,
    first_valid_mask: Tensor,
    second_valid_mask: Tensor,
) -> EvidenceSeparabilityReport:
    """Measure whether the frozen target retains paired visual differences.

    This is a diagnostic, not a calibrated decision threshold.  A low value
    indicates that downstream capacity cannot recover evidence discarded by
    the frozen target representation.
    """

    _require_torch()
    if (
        not isinstance(first, torch.Tensor)
        or first.ndim != 3
        or second.shape != first.shape
    ):
        raise ValueError("paired evidence must have matching shape [N,J,P]")
    n, positions, _ = first.shape
    _check_bool_tensor(
        first_valid_mask,
        name="first_valid_mask",
        shape=(n, positions),
    )
    _check_bool_tensor(
        second_valid_mask,
        name="second_valid_mask",
        shape=(n, positions),
    )
    common = first_valid_mask & second_valid_mask
    finite = torch.isfinite(first).all(dim=-1) & torch.isfinite(second).all(dim=-1)
    common = common & finite
    counts = common.sum(dim=1) * first.shape[-1]
    difference = torch.where(
        common.unsqueeze(-1),
        first.float() - second.float(),
        torch.zeros_like(first, dtype=torch.float32),
    )
    squared = difference.square()
    valid_pairs = counts > 0
    rms = torch.sqrt(squared.sum(dim=(-2, -1)) / counts.clamp_min(1).float())
    rms = torch.where(valid_pairs, rms, torch.full_like(rms, torch.nan))
    mean = float(rms[valid_pairs].mean().item()) if bool(valid_pairs.any()) else None
    return EvidenceSeparabilityReport(
        pair_rms_distance=rms,
        valid_pairs=valid_pairs,
        mean_rms_distance=mean,
    )


__all__ = [
    "EvidenceSeparabilityReport",
    "FrozenEvidenceNormalizer",
    "FrozenOrthogonalProjection",
    "IntentOnlyEncoder",
    "MaskedLoss",
    "PredictiveOutput",
    "PredictiveStateModel",
    "PredictiveStateReadout",
    "evidence_mixture_nll",
    "failure_bce_loss",
    "paired_evidence_separability",
]
