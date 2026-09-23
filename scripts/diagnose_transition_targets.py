"""State-transfer and feature-transport interventions on existing development forks.

Future pose/depth is used ONLY by the explicitly privileged oracle, never by a
learned model. Old confirmation cases are excluded. All fitting uses train only.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from grounded_interaction.predictive_vla.config import set_manual_seed

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/stage2_forks_s7227"
OUT = ROOT / "runs/transition_targets_s8227"


def save(name, value):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def norm(x):
    return F.layer_norm(x.float(), (x.shape[-1],))


def fractional_index(axis, values):
    right = torch.searchsorted(axis.contiguous(), values.contiguous()).clamp(
        1, len(axis) - 1
    )
    left = right - 1
    return left + (values - axis[left]) / (axis[right] - axis[left])


def layout(positions):
    """Validate the cached native layout, including its uneven last interval."""
    grids = []
    for view in [0, 1]:
        ids = torch.where(positions[:, 2] == view)[0]
        if not torch.all(positions[ids, 3] == 0):
            raise ValueError("diagnostic only supports uncropped single grids")
        n = int(len(ids) ** 0.5)
        uv = positions[ids, :2].reshape(n, n, 2)
        xs, ys = uv[0, :, 0], uv[:, 0, 1]
        assert torch.allclose(uv[..., 0], xs[None].expand(n, n), atol=2e-6)
        assert torch.allclose(uv[..., 1], ys[:, None].expand(n, n), atol=2e-6)
        grids.append((ids, xs, ys))
    return grids


def sample(features, grids, uv):
    result = torch.zeros_like(features)
    for ids, xs, ys in grids:
        n = len(xs)
        gx = fractional_index(xs, uv[:, ids, 0]) / (n - 1) * 2 - 1
        gy = fractional_index(ys, uv[:, ids, 1]) / (n - 1) * 2 - 1
        grid = torch.stack((gx, gy), -1).reshape(len(features), n, n, 2)
        field = features[:, ids].reshape(len(features), n, n, -1).permute(0, 3, 1, 2)
        result[:, ids] = (
            F.grid_sample(field, grid, align_corners=True, padding_mode="border")
            .permute(0, 2, 3, 1)
            .flatten(1, 2)
        )
    return result


def camera_map(current, future, uv, view):
    """Backward world reprojection; future geometry makes this a privileged diagnostic."""
    h, w = future[f"{view}_depth"].shape
    pixels = uv * [w - 1, h - 1]
    ij = np.rint(pixels).astype(int)
    z = future[f"{view}_depth"][ij[:, 1], ij[:, 0]]
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(future[f"{view}_K"]).T
    camera = rays * z[:, None]
    tf, tc = future[f"{view}_T_world"], current[f"{view}_T_world"]
    world = camera @ tf[:3, :3].T + tf[:3, 3]
    old = (world - tc[:3, 3]) @ tc[:3, :3]
    projected = old @ current[f"{view}_K"].T
    pix = projected[:, :2] / np.maximum(projected[:, 2:], 1e-8)
    inside = (
        (old[:, 2] > 0)
        & (pix[:, 0] >= 0)
        & (pix[:, 0] <= w - 1)
        & (pix[:, 1] >= 0)
        & (pix[:, 1] <= h - 1)
    )
    ij_old = np.rint(np.clip(pix, 0, [w - 1, h - 1])).astype(int)
    old_z = current[f"{view}_depth"][ij_old[:, 1], ij_old[:, 0]]
    # Static-scene depth consistency. Moving objects can fail this test; report coverage.
    visible = inside & np.isfinite(z) & (z > 0) & (np.abs(old_z - old[:, 2]) < 0.02)
    return pix / [w - 1, h - 1], visible


def load(
    device,
    source=SOURCE,
    feature_directory=None,
    splits=("train", "development"),
    *,
    privileged=False,
):
    """Load selected splits only; future geometry exists solely in oracle diagnostics."""
    cases = []
    feature_directory = feature_directory or source / "features"
    plan = json.loads((source / "plan.json").read_text())
    selected = {
        f"case_{c['case_id']:03d}.pt": c for c in plan["cases"] if c["split"] in splits
    }
    provenance_path = feature_directory / "encoding.json"
    provenance = (
        json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    )
    for name, case in sorted(selected.items()):
        cache = torch.load(
            feature_directory / name, weights_only=True, map_location="cpu"
        )
        if cache["case"] != case:
            raise ValueError("cache case differs from locked plan")
        if provenance is not None and cache.get("encoding") != provenance:
            raise ValueError("mixed or absent cache policy provenance")
        report = cache["source_report"]
        if not report.get("complete") or not report.get("initial_arrays_identical"):
            raise ValueError("fork replay audit is incomplete")
        if report.get("render_profile", "native") != plan.get(
            "render_profile", "native"
        ):
            raise ValueError("cache render profile differs from locked plan")
        if any(
            max(e["repeat_max_abs"].values()) != 0
            for b in report["records"][6:]
            for e in b["endpoints"]
        ):
            raise ValueError("fork repeatability audit failed")
        folder = source / f"case_{case['case_id']:03d}"
        with np.load(folder / "history_2.npz", allow_pickle=False) as arrays:
            current = dict(arrays)
        targets, sequences, states, oracle_uv, visible = [], [], [], [], []
        positions = cache["positions"][0]
        for branch, record in zip(
            cache["branches"][:6], report["records"][:6], strict=True
        ):
            endpoint = next(e for e in branch["endpoints"] if e["horizon"] == 10)
            targets.append(endpoint["target"][0])
            sequences.append(endpoint["actions"][0])
            if privileged:
                file = next(
                    e["file"] for e in record["endpoints"] if e["horizon"] == 10
                )
                with np.load(folder / file, allow_pickle=False) as arrays:
                    future = dict(arrays)
                states.append(future["states"][:3] - current["states"][:3])
                maps, masks = [], []
                for view, key in enumerate(["agent", "wrist"]):
                    uv = positions[positions[:, 2] == view, :2].numpy()
                    mapping, mask = camera_map(current, future, uv, key)
                    maps.append(mapping)
                    masks.append(mask)
                oracle_uv.append(np.concatenate(maps))
                visible.append(np.concatenate(masks))
        actions = torch.stack(sequences).float().to(device)
        item = {
            "case": case,
            "current": norm(cache["current"]).to(device),
            "target": norm(torch.stack(targets)).to(device),
            "positions": positions.to(device),
            "action_sequences": actions,
            "valid": cache["valid"][0].to(device),
            "actions": actions.mean(1),
            "state": torch.tensor(current["states"], device=device),
            "shared": cache["shared"].float().mean(1).to(device),
        }
        if privileged:
            item.update(
                delta=torch.tensor(np.stack(states), device=device),
                oracle_uv=torch.tensor(
                    np.stack(oracle_uv), device=device, dtype=torch.float32
                ),
                visible=torch.tensor(np.stack(visible), device=device),
            )
        cases.append(item)
    return cases


def physical(cases):
    train = torch.tensor(
        [c["case"]["split"] == "train" for c in cases], device=cases[0]["delta"].device
    ).repeat_interleave(6)
    y = torch.cat([c["delta"] for c in cases]).double()
    a = torch.cat([c["actions"] for c in cases]).double()
    s = torch.cat([c["state"][None].expand(6, -1) for c in cases]).double()
    shared = torch.cat([c["shared"].expand(6, -1) for c in cases]).double()
    rows = []
    for name, x in [
        ("action_only", a),
        ("state_only", s),
        ("state_action", torch.cat((s, a), -1)),
        ("vlm_action", torch.cat((shared, a), -1)),
    ]:
        mean, std = x[train].mean(0), x[train].std(0).clamp_min(0.01)
        x = torch.cat(((x - mean) / std, torch.ones((len(x), 1), device=x.device)), 1)
        # Fixed regularizer, train-only centering/scaling. No development selection.
        xt, yt = x[train], y[train]
        ym = yt.mean(0)
        alpha = torch.linalg.solve(
            xt @ xt.T / xt.shape[1] + 0.01 * torch.eye(len(xt), device=x.device),
            yt - ym,
        )
        p = (x @ xt.T / xt.shape[1]) @ alpha + ym
        for split, mask in [("train", train), ("development", ~train)]:
            truth, pred = y[mask].reshape(-1, 6, 3), p[mask].reshape(-1, 6, 3)
            centered = truth - truth.mean(1, keepdim=True)
            error = ((pred - pred.mean(1, keepdim=True)) - centered).square().mean()
            rows.append(
                {
                    "model": name,
                    "split": split,
                    "rmse_m": float((pred - truth).square().mean().sqrt()),
                    "copy_rmse_m": float(truth.square().mean().sqrt()),
                    "action_variance_explained": float(
                        1 - error / centered.square().mean()
                    ),
                }
            )
    return rows


def oracle(cases):
    rows = []
    for c in cases:
        grid = layout(c["positions"])
        current = c["current"].expand(6, -1, -1)
        identity = sample(current, grid, c["positions"][None, :, :2].expand(6, -1, -1))
        assert torch.allclose(identity, current, atol=1e-5, rtol=1e-5), (
            "identity warp changed features"
        )
        pred = sample(current, grid, c["oracle_uv"])
        for v, (ids, _, _) in enumerate(grid):
            mask = c["visible"][:, ids] & c["valid"][None, ids]
            target = c["target"][:, ids]
            copied = current[:, ids]
            if mask.any():
                rows.append(
                    {
                        "case": c["case"],
                        "view": v,
                        "coverage": float(mask.float().mean()),
                        "oracle_loss": float(
                            F.smooth_l1_loss(pred[:, ids][mask], target[mask])
                        ),
                        "copy_loss": float(
                            F.smooth_l1_loss(copied[mask], target[mask])
                        ),
                    }
                )
    return rows


class Transport(torch.nn.Module):
    """Small action/state-conditioned per-view affine resampling, no future inputs."""

    def __init__(self, state_mean, state_std, blind=False, action_only=False):
        super().__init__()
        self.register_buffer("mean", state_mean)
        self.register_buffer("std", state_std)
        self.blind = blind
        self.action_only = action_only
        self.net = torch.nn.Sequential(
            torch.nn.Linear(15, 64), torch.nn.SiLU(), torch.nn.Linear(64, 12)
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, c):
        a = torch.zeros_like(c["actions"]) if self.blind else c["actions"]
        s = ((c["state"] - self.mean) / self.std)[None].expand(6, -1)
        if self.action_only:
            s = torch.zeros_like(s)
        affine = self.net(torch.cat((s, a), -1)).reshape(6, 2, 2, 3)
        uv = c["positions"][None, :, :2].expand(6, -1, -1)
        basis = torch.cat((uv - 0.5, torch.ones_like(uv[:, :, :1])), -1)
        matrices = affine[:, c["positions"][:, 2].long()]
        shift = torch.einsum("bjkl,bjl->bjk", matrices, basis)
        return sample(
            c["current"].expand(6, -1, -1), layout(c["positions"]), uv + shift
        )


@torch.no_grad()
def evaluate(model, cases):
    rows = []
    for c in cases:
        p = model(c)
        y = c["target"]
        valid = c["valid"]
        p = p[:, valid]
        y = y[:, valid]
        pc = p - p.mean(0)
        yc = y - y.mean(0)
        swapped = model({**c, "actions": c["actions"].roll(1, 0)})[:, valid]
        rows.append(
            {
                "case": c["case"],
                "swapped_loss": float(F.smooth_l1_loss(swapped, y)),
                "loss": float(F.smooth_l1_loss(p, y)),
                "copy_loss": float(
                    F.smooth_l1_loss(c["current"][:, valid].expand_as(y), y)
                ),
                "action_variance_explained": float(
                    1 - (pc - yc).square().mean() / yc.square().mean()
                ),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["audit", "fit", "controls", "blur"], required=True
    )
    args = parser.parse_args()
    set_manual_seed(8227)
    torch.set_num_threads(2)
    start = time.monotonic()
    cases = load("cuda", privileged=True)
    if args.mode == "blur":
        rows = []
        for c in cases:
            copied = c["current"]
            blurred = torch.zeros_like(copied)
            for ids, xs, ys in layout(c["positions"]):
                field = (
                    copied[:, ids].reshape(1, len(ys), len(xs), -1).permute(0, 3, 1, 2)
                )
                b = F.avg_pool2d(
                    F.pad(field, (1, 1, 1, 1), mode="replicate"), 3, stride=1
                )
                blurred[:, ids] = b.permute(0, 2, 3, 1).flatten(1, 2)
            rows.append(
                {
                    "case": c["case"],
                    "loss": float(
                        F.smooth_l1_loss(
                            blurred.expand_as(c["target"])[:, c["valid"]],
                            c["target"][:, c["valid"]],
                        )
                    ),
                }
            )
        save("blur.json", {"rows": rows, "seconds": time.monotonic() - start})
    elif args.mode == "audit":
        save(
            "audit.json",
            {
                "manual_seed": 8227,
                "physical": physical(cases),
                "oracle": oracle(cases),
                "seconds": time.monotonic() - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            },
        )
    else:
        train = [c for c in cases if c["case"]["split"] == "train"]
        states = torch.stack([c["state"] for c in train])
        rows = []
        for seed in [17, 29, 43]:
            for kind in (
                ["live", "blind", "action_only"]
                if args.mode == "controls"
                else ["live", "blind"]
            ):
                blind = kind == "blind"
                set_manual_seed(seed)
                model = Transport(
                    states.mean(0),
                    states.std(0).clamp_min(0.01),
                    blind,
                    kind == "action_only",
                ).cuda()
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=0.0005, weight_decay=0
                )
                generator = np.random.default_rng(seed)
                for step in range(400):
                    c = train[int(generator.integers(len(train)))]
                    optimizer.zero_grad()
                    loss = F.smooth_l1_loss(
                        model(c)[:, c["valid"]], c["target"][:, c["valid"]]
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                torch.save(model.state_dict(), OUT / f"{args.mode}_{kind}_s{seed}.pt")
                rows.append(
                    {
                        "manual_seed": seed,
                        "kind": kind,
                        "blind": blind,
                        "steps": 400,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "rows": evaluate(model, cases),
                    }
                )
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "blind": blind,
                            "seconds": time.monotonic() - start,
                        }
                    ),
                    flush=True,
                )
        save(
            f"{args.mode}.json",
            {
                "fits": rows,
                "seconds": time.monotonic() - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            },
        )


if __name__ == "__main__":
    main()
