"""Privileged fitting diagnostic: estimates transport expressivity, not prediction.

Only train/development caches are loaded. Future labels optimize each oracle
independently; these fitted coordinates must NEVER initialize a deployable head.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from diagnose_transition_targets import load
from torch.nn import functional as F

from grounded_interaction.predictive_vla.config import set_manual_seed
from grounded_interaction.predictive_vla.transport import fractional_index, native_grids

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/transition_scaling_no_msaa_s10227"


def sample(current, positions, coords):
    result = []
    for ids, xs, ys in native_grids(positions):
        grid = torch.stack(
            (
                fractional_index(xs, coords[:, ids, 0]) / (len(xs) - 1) * 2 - 1,
                fractional_index(ys, coords[:, ids, 1]) / (len(ys) - 1) * 2 - 1,
            ),
            -1,
        ).reshape(6, len(ys), len(xs), 2)
        field = (
            current[:, ids]
            .reshape(1, len(ys), len(xs), -1)
            .permute(0, 3, 1, 2)
            .expand(6, -1, -1, -1)
        )
        result.append(
            F.grid_sample(field, grid, align_corners=True, padding_mode="border")
            .permute(0, 2, 3, 1)
            .flatten(1, 2)
        )
    return torch.cat(result, 1)


def metrics(p, y, current, positions):
    result = {}
    for view in [-1, 0, 1]:
        valid = (
            torch.ones(len(positions), dtype=torch.bool, device=y.device)
            if view == -1
            else positions[:, 2] == view
        )
        a, b = p[:, valid], y[:, valid]
        pc = a - a.mean(0)
        yc = b - b.mean(0)
        result[str(view)] = {
            "loss": float(F.smooth_l1_loss(a, b)),
            "copy_loss": float(F.smooth_l1_loss(current[:, valid].expand_as(b), b)),
            "effect_error": float((pc - yc).square().mean()),
            "effect_energy": float(yc.square().mean()),
            "predicted_effect_energy": float(pc.square().mean()),
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["affine", "dense"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--source", default=str(SOURCE))
    p.add_argument(
        "--splits", nargs="+", default=["train", "development", "id_development"]
    )
    p.add_argument("--cases-per-family", type=int)
    a = p.parse_args()
    out = (ROOT / a.output).resolve()
    assert out.is_relative_to(ROOT)
    out.mkdir(exist_ok=False)
    set_manual_seed(11227)
    torch.set_num_threads(2)
    started = time.monotonic()
    source = (ROOT / a.source).resolve()
    if not source.is_relative_to(ROOT):
        raise ValueError("source must stay inside repository")
    cases = load("cuda", source, source / "features_stage1", splits=tuple(a.splits))
    rng = random.Random(11227)
    chosen = []
    for split in a.splits:
        families = sorted(
            {c["case"]["family"] for c in cases if c["case"]["split"] == split}
        )
        for family in families:
            pool = [
                c
                for c in cases
                if c["case"]["family"] == family and c["case"]["split"] == split
            ]
            chosen.extend(
                rng.sample(
                    pool, a.cases_per_family or (1 if split != "development" else 3)
                )
            )
    # Identical 24 cases in both modes, fixed before viewing oracle results.
    rows = []
    for index, c in enumerate(chosen):
        y = c["target"]
        current = c["current"]
        positions = c["positions"]
        base = positions[:, :2][None].expand(6, -1, -1)
        basis = torch.cat((base - 0.5, torch.ones_like(base[..., :1])), -1)
        parameter = torch.zeros(
            (6, 2, 2, 3) if a.mode == "affine" else (6, len(positions), 2),
            device="cuda",
            requires_grad=True,
        )
        opt = torch.optim.Adam([parameter], lr=0.003)
        best = float("inf")
        best_state = None
        curve = []
        for step in range(a.steps):
            if a.mode == "affine":
                matrices = parameter[:, positions[:, 2].long()]
                coords = base + torch.einsum("bjkl,bjl->bjk", matrices, basis)
            else:
                coords = base + parameter
            prediction = sample(current, positions, coords)
            # MSE on same-state action contrasts makes the oracle seek the exact failed metric.
            pc = prediction - prediction.mean(0)
            yc = y - y.mean(0)
            loss = F.smooth_l1_loss(prediction, y) + (pc - yc).square().mean()
            value = float(loss.detach())
            if value < best:
                best = value
                best_state = prediction.detach().clone()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (step + 1) % 100 == 0:
                curve.append({"step": step + 1, "objective": value})
        rows.append(
            {
                "case": c["case"],
                "metrics": metrics(best_state, y, current, positions),
                "curve": curve,
            }
        )
        report = {
            "complete": len(rows) == len(chosen),
            "oracle_uses_future_targets": True,
            "deployable": False,
            "manual_seed": 11227,
            "mode": a.mode,
            "steps": a.steps,
            "rows": rows,
            "seconds": time.monotonic() - started,
        }
        (out / "report.partial").write_text(json.dumps(report, indent=2) + "\n")
        (out / "report.partial").replace(out / "report.json")
        print(
            json.dumps(
                {
                    "case": c["case"]["case_id"],
                    "completed": index + 1,
                    "total": len(chosen),
                    "metrics": rows[-1]["metrics"]["-1"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
