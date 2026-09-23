"""Action-conditioned transport of frozen current visual features.

This restricted affine candidate tests appearance reuse before adding an
unconstrained appearance decoder. It cannot synthesize newly visible content.
No future pixels, depth, camera poses, or states are accepted as inputs.
"""

import torch
from torch import nn
from torch.nn import functional as F


def native_grids(positions):
    """Validate native row-major grids; never infer topology from count alone.

    Current LIBERO inputs have two uncropped grids with uneven edge spacing.
    Multi-crop inputs require a separate correspondence rule and fail explicitly.
    """
    if positions.ndim != 2 or positions.shape[-1] != 4:
        raise ValueError("positions must be J x 4")
    if not torch.isfinite(positions).all() or not (positions[:, 3] == 0).all():
        raise ValueError("transport requires finite uncropped patch positions")
    if set(positions[:, 2].tolist()) != {0.0, 1.0}:
        raise ValueError("transport requires ordered agent and wrist views")
    grids = []
    for view in [0, 1]:
        ids = torch.where(positions[:, 2] == view)[0]
        # Row boundaries, not sqrt(token count), determine native grid topology.
        uv = positions[ids, :2]
        first_row = torch.isclose(uv[:, 1], uv[0, 1], atol=2e-6, rtol=0)
        width = int(first_row.sum())
        if width < 2 or len(ids) % width or len(ids) // width < 2:
            raise ValueError("transport needs a complete 2D native patch grid")
        height = len(ids) // width
        uv = uv.reshape(height, width, 2)
        xs, ys = uv[0, :, 0], uv[:, 0, 1]
        if not (
            torch.all(xs[1:] > xs[:-1])
            and torch.all(ys[1:] > ys[:-1])
            and torch.allclose(
                uv[..., 0], xs[None].expand(height, width), atol=2e-6, rtol=0
            )
            and torch.allclose(
                uv[..., 1], ys[:, None].expand(height, width), atol=2e-6, rtol=0
            )
        ):
            raise ValueError("native patch order is not a separable row-major grid")
        grids.append((ids, xs, ys))
    return grids


def fractional_index(axis, values):
    right = torch.searchsorted(axis.contiguous(), values.contiguous()).clamp(
        1, len(axis) - 1
    )
    left = right - 1
    return left + (values - axis[left]) / (axis[right] - axis[left])


def resample_current(features, positions, uv):
    """Bilinear transport in native patch coordinates, independently per view."""
    if features.ndim != 3 or uv.shape != (*features.shape[:2], 2):
        raise ValueError("transport tensors are misaligned")
    if positions.shape != (*features.shape[:2], 4):
        raise ValueError("transport positions are misaligned")
    results = []
    for b in range(len(features)):
        result = torch.zeros_like(features[b], dtype=torch.float32)
        for ids, xs, ys in native_grids(positions[b]):
            gx = fractional_index(xs, uv[b, ids, 0]) / (len(xs) - 1) * 2 - 1
            gy = fractional_index(ys, uv[b, ids, 1]) / (len(ys) - 1) * 2 - 1
            grid = torch.stack((gx, gy), -1).reshape(1, len(ys), len(xs), 2)
            field = (
                features[b, ids].reshape(len(ys), len(xs), -1).permute(2, 0, 1)[None]
            )
            # FP32 sampling also works on CPU and avoids BF16 coordinate quantization.
            with torch.autocast(features.device.type, enabled=False):
                moved = F.grid_sample(
                    field.float(),
                    grid.float(),
                    align_corners=True,
                    padding_mode="border",
                )
            result[ids] = moved[0].permute(1, 2, 0).flatten(0, 1)
        results.append(result)
    return torch.stack(results)


class TransportPredictor(nn.Module):
    """Pooled VLM history + ordered actual controls + current proprio -> 2 affines.

    Native VLM--AE conditioning is untouched. Unlike the constant-action
    diagnostic MLP, the GRU preserves action order and actual prefix duration.
    Current teacher features provide appearance; gradients reach VLM via the
    transform parameters after the zero-initialized readout starts learning.
    """

    def __init__(
        self,
        native_width,
        *,
        width=64,
        time_scale=300,
        normalize_context=True,
        fusion="control_residual",
    ):
        super().__init__()
        if min(native_width, width, time_scale) < 1:
            raise ValueError("invalid transport dimensions")
        self.time_scale = time_scale
        self.normalize_context = normalize_context
        if fusion not in {"concat", "control_residual"}:
            raise ValueError("invalid transport fusion")
        self.fusion = fusion
        self.context = nn.Linear(native_width, width)
        self.action = nn.GRU(8, width, batch_first=True)
        self.state = nn.Linear(8, width)
        self.output = nn.Sequential(
            nn.Linear(3 * width + 1, width), nn.SiLU(), nn.Linear(width, 12)
        )
        if fusion == "control_residual":
            self.control_output = nn.Sequential(
                nn.Linear(2 * width + 1, width), nn.SiLU(), nn.Linear(width, 12)
            )
            nn.init.zeros_(self.control_output[-1].weight)
            nn.init.zeros_(self.control_output[-1].bias)
        self.register_buffer("state_mean", torch.zeros(8))
        self.register_buffer("state_std", torch.ones(8))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        shared,
        context_valid,
        actions,
        action_valid,
        positions,
        horizon,
        *,
        current,
        current_valid,
        state,
    ):
        if (
            shared.ndim != 3
            or context_valid.shape != shared.shape[:2]
            or context_valid.dtype != torch.bool
        ):
            raise ValueError("shared context mask is misaligned")
        if (
            actions.ndim != 3
            or actions.shape[-1] != 7
            or action_valid.shape != actions.shape[:2]
            or action_valid.dtype != torch.bool
        ):
            raise ValueError("actions need an aligned boolean prefix mask")
        batch = len(shared)
        if len(actions) != batch or state.shape != (batch, 8):
            raise ValueError("transport action/state batch mismatch")
        if (
            current.shape != (batch, positions.shape[1], shared.shape[-1])
            or current_valid.shape != current.shape[:2]
            or current_valid.dtype != torch.bool
        ):
            raise ValueError("current appearance is misaligned")
        if not current_valid.all() or not context_valid.any(-1).all():
            raise ValueError(
                "transport requires complete source grids and nonempty context"
            )
        lengths = action_valid.sum(-1)
        h = torch.as_tensor(horizon, device=actions.device).reshape(-1)
        times = torch.arange(actions.shape[1], device=actions.device)[None]
        if (
            h.shape != (batch,)
            or not torch.equal(h, lengths)
            or (h < 1).any()
            or not torch.equal(action_valid, times < lengths[:, None])
        ):
            raise ValueError("horizon must match a contiguous actual action prefix")
        clean_shared = torch.where(context_valid[..., None], shared, 0.0)
        clean_actions = torch.where(action_valid[..., None], actions, 0.0)
        if not all(
            torch.isfinite(x).all()
            for x in (clean_shared, clean_actions, current, state, positions)
        ):
            raise ValueError("valid transport inputs must be finite")
        if not torch.isfinite(self.state_std).all() or (self.state_std <= 0).any():
            raise ValueError("invalid training-only state normalization")
        dtype = self.context.weight.dtype
        pooled = clean_shared.float().sum(1) / context_valid.sum(1, keepdim=True)
        if self.normalize_context:
            pooled = F.layer_norm(pooled.float(), (pooled.shape[-1],))
        c = self.context(pooled.to(dtype))
        action_input = torch.cat(
            (clean_actions, ((times + 1) / h[:, None])[..., None]), -1
        ).to(dtype)
        sequence, _ = self.action(action_input)
        a = sequence[torch.arange(batch, device=actions.device), lengths - 1]
        s = self.state(((state - self.state_mean) / self.state_std).to(dtype))
        time = (h[:, None] / self.time_scale).to(dtype)
        correction = self.output(torch.cat((c, a, s, time), -1))
        if self.fusion == "control_residual":
            parameters = (
                self.control_output(torch.cat((a, s, time), -1)) + 0.1 * correction
            )
        else:
            parameters = correction
        moved_uv = self.motion_coordinates(
            parameters, positions, current, c, a, s, time
        )
        appearance = F.layer_norm(current.detach().float(), (current.shape[-1],))
        return self.predict_features(appearance, positions, moved_uv, c, a, s, time)

    def predict_features(
        self, appearance, positions, moved_uv, context, action, state, time
    ):
        return resample_current(appearance, positions, moved_uv)

    def motion_coordinates(
        self, parameters, positions, current, context, action, state, time
    ):
        batch = len(parameters)
        affine = parameters.float().reshape(batch, 2, 2, 3)
        uv = positions[..., :2].float()
        basis = torch.cat((uv - 0.5, torch.ones_like(uv[..., :1])), -1)
        for row in positions:
            native_grids(row)
        matrices = affine[
            torch.arange(batch, device=positions.device)[:, None],
            positions[..., 2].long(),
        ]
        return uv + torch.einsum("bjkl,bjl->bjk", matrices, basis)


class LocalTransportPredictor(TransportPredictor):
    """Affine motion plus a bounded per-patch correction from current features.

    This tests the spatial bottleneck directly. It still cannot generate new
    appearance; no future observation or privileged motion is an input.
    """

    def __init__(self, native_width, *, width=64, **kwargs):
        super().__init__(native_width, width=width, **kwargs)
        self.local_feature = nn.Linear(native_width, width)
        self.local_output = nn.Sequential(
            nn.Linear(4 * width + 4, width), nn.SiLU(), nn.Linear(width, 2)
        )
        nn.init.zeros_(self.local_output[-1].weight)
        nn.init.zeros_(self.local_output[-1].bias)

    def motion_coordinates(
        self, parameters, positions, current, context, action, state, time
    ):
        base = super().motion_coordinates(
            parameters, positions, current, context, action, state, time
        )
        appearance = F.layer_norm(current.detach().float(), (current.shape[-1],))
        local = self.local_feature(appearance.to(context.dtype))
        condition = torch.cat((context, action, state, time), -1)
        condition = condition[:, None].expand(-1, current.shape[1], -1)
        inputs = torch.cat((local, condition, positions[..., :3].to(context.dtype)), -1)
        return base + 0.25 * torch.tanh(self.local_output(inputs).float())


class ResidualTransportPredictor(LocalTransportPredictor):
    """Test whether transport needs an appearance correction at each current patch.

    Unlike the retired global query decoder, the correction directly reads the
    current patch and its local neighbors. No target or future pose is an input.
    Both motion and residual readouts start at the exact copy-current function.
    """

    def __init__(self, native_width, *, width=64, **kwargs):
        super().__init__(native_width, width=width, **kwargs)
        self.neighborhood = nn.Conv2d(width, width, 3, padding=1)
        self.residual_output = nn.Sequential(
            nn.Linear(4 * width + 4, width), nn.SiLU(), nn.Linear(width, native_width)
        )
        nn.init.zeros_(self.residual_output[-1].weight)
        nn.init.zeros_(self.residual_output[-1].bias)

    def predict_features(
        self, appearance, positions, moved_uv, context, action, state, time
    ):
        local = self.local_feature(appearance.to(context.dtype))
        neighborhood = torch.zeros_like(local)
        for b in range(len(local)):
            for ids, xs, ys in native_grids(positions[b]):
                field = (
                    local[b, ids].reshape(len(ys), len(xs), -1).permute(2, 0, 1)[None]
                )
                neighborhood[b, ids] = (
                    self.neighborhood(field)[0].permute(1, 2, 0).flatten(0, 1)
                )
        condition = torch.cat((context, action, state, time), -1)[:, None].expand(
            -1, local.shape[1], -1
        )
        inputs = torch.cat(
            (local + neighborhood, condition, positions[..., :3].to(context.dtype)), -1
        )
        return (
            self.appearance_base(appearance, positions, moved_uv)
            + self.residual_output(inputs).float()
        )

    def appearance_base(self, appearance, positions, moved_uv):
        return resample_current(appearance, positions, moved_uv)


class PatchResidualPredictor(ResidualTransportPredictor):
    """Matched no-transport diagnostic; isolates the new appearance correction."""

    def appearance_base(self, appearance, positions, moved_uv):
        return appearance


PREDICTOR_TYPES = {
    "affine": TransportPredictor,
    "local": LocalTransportPredictor,
    "local_residual": ResidualTransportPredictor,
    "patch_residual": PatchResidualPredictor,
}
