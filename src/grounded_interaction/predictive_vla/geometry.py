"""Calibrated geometry at native visual patch positions, including missingness."""

import inspect

import numpy as np
import torch
from torch import nn

from .types import CameraFrame


def backproject(frame: CameraFrame, uv: np.ndarray):
    """Sample metric depth at native patch centers; return XYZ + observed bit.

    Pixel coordinates use integer pixel centers. Invalid/missing depth never
    means free space; it yields observed=False, even if XYZ's storage is zero.
    """
    uv = np.asarray(uv, dtype=np.float32)
    if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
        raise ValueError("patch centers must be finite J x 2 pixel coordinates")
    xyz = np.zeros((len(uv), 3), dtype=np.float32)
    valid = np.zeros(len(uv), dtype=bool)
    if frame.depth_m is None:
        return xyz, valid
    h, w = frame.rgb.shape[:2]
    u, v = np.rint(uv).astype(np.int64).T
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    z = frame.depth_m[np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)]
    valid = inside & np.isfinite(z) & (z > 0)
    rays = np.c_[uv[valid], np.ones(valid.sum())] @ np.linalg.inv(frame.intrinsics).T
    camera = rays * z[valid, None]
    xyz[valid] = camera @ frame.camera_to_world[:3, :3].T + frame.camera_to_world[:3, 3]
    return xyz, valid


def native_patch_layout(
    image_processor, frame: CameraFrame, expected_grid, expected_pooling
):
    """Replay the pinned processor's real crop/pool map on a coordinate raster.

    This avoids guessing a square grid from the number of flattened tokens.
    No RGB/depth target or simulator state enters layout construction.
    """
    module = inspect.getmodule(type(image_processor))
    helper = getattr(module, "image_to_patches_and_grids", None)
    if helper is None:
        raise RuntimeError("native image processor has no crop/pooling layout helper")
    h, w = frame.rgb.shape[:2]
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.stack(
        (u / max(w - 1, 1), v / max(h - 1, 1), np.ones_like(u)), -1
    ).astype(np.float32)
    p = image_processor
    grid, patches, pooling = helper(
        coords,
        p.max_crops,
        p.overlap_margins,
        [p.size["height"], p.size["width"]],
        p.resample,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        p.patch_size,
        p.pooling_size[1],
        p.pooling_size[0],
        p.crop_mode,
    )
    if not np.array_equal(
        np.asarray(grid).reshape(-1), np.asarray(expected_grid).reshape(-1)
    ):
        raise RuntimeError("geometry crop grids differ from the native RGB processor")
    if not np.array_equal(pooling, expected_pooling):
        raise RuntimeError(
            "geometry pooling indices differ from native visual positions"
        )
    centers = patches.reshape(-1, p.patch_size * p.patch_size, 3).mean(1)
    present = pooling >= 0
    gathered = centers[np.maximum(pooling, 0)]
    pooled = (gathered * present[..., None]).sum(1) / np.maximum(
        present.sum(1, keepdims=True), 1
    )
    uv = pooled[:, :2] * np.array([max(w - 1, 1), max(h - 1, 1)])
    return uv.astype(np.float32), present.any(1)


class GeometryAdapter(nn.Module):
    """Zero-initialized residual preserves native RGB semantics at initialization."""

    def __init__(self, width, frequencies=4):
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(frequencies) * torch.pi)
        self.project = nn.Sequential(
            nn.Linear(3 + 6 * frequencies + 1, 128), nn.SiLU(), nn.Linear(128, width)
        )
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

    def forward(self, xyz, observed):
        if xyz.shape[:-1] != observed.shape or xyz.shape[-1] != 3:
            raise ValueError("geometry and observed masks are misaligned")
        if not torch.isfinite(xyz[observed]).all():
            raise ValueError("observed geometry must be finite")
        clean = torch.where(observed[..., None], xyz, torch.zeros_like(xyz))
        angles = clean[..., None] * self.frequencies.to(clean)
        features = torch.cat(
            (
                clean,
                angles.sin().flatten(-2),
                angles.cos().flatten(-2),
                observed[..., None].to(clean),
            ),
            -1,
        )
        return self.project(features.to(self.project[0].weight))
