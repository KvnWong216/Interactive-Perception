"""CPU-only controlled tests of the production predictor; never a VLA benchmark.

Synthetic pairs share exactly the same observed state and have opposite controls.
Only the action-dependent condition has different futures. No robot data is fitted.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from grounded_interaction.predictive_vla.config import set_manual_seed
from grounded_interaction.predictive_vla.model import ActionConditionedPredictor

ROOT = Path(__file__).resolve().parents[1]


def inside(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("diagnostic output must stay inside the repository")
    return path


def normalize(x):
    return F.layer_norm(x, (x.shape[-1],))


def make_pairs(*, manual_seed=6227, families=24, context_tokens=64, static=False):
    """Known dynamics in feature space, with disjoint initial-state families.

    This is deliberately NOT a simulator or a video encoder reproduction.
    Each pair has an identical current state, context, positions and horizon.
    """
    if families < 2 or context_tokens < 16:
        raise ValueError("need at least two families and 16 context tokens")
    generator = torch.Generator().manual_seed(manual_seed)
    width, patches, horizon = 32, 16, 4
    positions = torch.zeros(patches, 4)
    positions[:, 0] = torch.linspace(-1, 1, patches)
    spatial = torch.stack((torch.ones(patches), positions[:, 0]), -1)
    basis = torch.randn(2, width, generator=generator)
    direction = normalize(torch.randn(width, generator=generator))
    records = []
    for family in range(families):
        state = 0.3 * torch.randn(width, generator=generator)
        current = normalize(spatial @ basis + state)
        context = torch.cat((current, state.expand(context_tokens - patches, -1)), 0)
        magnitude = 0.5 + 0.5 * torch.rand((), generator=generator)
        actions = torch.zeros(2, horizon, 7)
        actions[0, :, 0] = magnitude
        actions[1, :, 0] = -magnitude
        # A fixed spatial effect makes target differences interpretable.
        delta = actions[:, :, 0].mean(1)[:, None, None] * direction
        delta = delta * (1 + positions[None, :, :1])
        target = normalize(current[None] + (0 if static else delta))
        if static:
            target = target.expand(2, -1, -1).clone()
        records.append(
            {
                "family": family,
                "shared": context[None].expand(2, -1, -1).clone(),
                "actions": actions,
                "positions": positions[None].expand(2, -1, -1).clone(),
                "current": current[None].expand(2, -1, -1).clone(),
                "target": target,
            }
        )
    return records


def stack(records):
    return {
        key: torch.cat([r[key] for r in records])
        for key in ("shared", "actions", "positions", "current", "target")
    }


def predict(model, batch, *, blind=False, swapped=False):
    actions = batch["actions"]
    if blind:
        actions = torch.zeros_like(actions)
    elif swapped:
        actions = actions.reshape(-1, 2, *actions.shape[1:]).flip(1).flatten(0, 1)
    return model(
        batch["shared"],
        torch.ones(batch["shared"].shape[:2], dtype=torch.bool),
        actions,
        torch.ones(actions.shape[:2], dtype=torch.bool),
        batch["positions"],
        torch.full((len(actions),), actions.shape[1]),
    )


def pair_metrics(prediction, target):
    """Target is already normalized. Report every pair, not just an average."""
    p, y = (v.reshape(-1, 2, *v.shape[1:]) for v in (prediction, target))
    correct = F.smooth_l1_loss(p, y, reduction="none").mean((1, 2, 3))
    crossed = F.smooth_l1_loss(p, y.flip(1), reduction="none").mean((1, 2, 3))
    midpoint = y.mean(1, keepdim=True).expand_as(y)
    # Convex, symmetric equal-weight two-target SmoothL1 has a midpoint minimizer.
    # This is an empirical pair bound, not the optimum for unseen states.
    blind_bound = F.smooth_l1_loss(midpoint, y, reduction="none").mean((1, 2, 3))
    separation = (y[:, 0] - y[:, 1]).square().mean((1, 2)).sqrt()
    predicted_separation = (p[:, 0] - p[:, 1]).square().mean((1, 2)).sqrt()
    return {
        "loss": float(correct.mean()),
        "paired_assignment_advantage": float((crossed - correct).mean()),
        "strict_pair_accuracy": float((crossed > correct + 1e-7).float().mean()),
        "tie_fraction": float(((crossed - correct).abs() <= 1e-7).float().mean()),
        "empirical_action_blind_bound": float(blind_bound.mean()),
        "target_pair_separation_rms": float(separation.mean()),
        "predicted_pair_separation_rms": float(predicted_separation.mean()),
        "pairs": [
            {"correct": float(c), "crossed": float(w), "blind_bound": float(b)}
            for c, w, b in zip(correct, crossed, blind_bound)
        ],
    }


@torch.no_grad()
def evaluate(model, batch, blind):
    prediction = predict(model, batch, blind=blind)
    result = pair_metrics(prediction, batch["target"])
    wrong = predict(model, batch, blind=blind, swapped=True)
    result["wrong_action_loss"] = float(F.smooth_l1_loss(wrong, batch["target"]))
    result["copy_loss"] = float(F.smooth_l1_loss(batch["current"], batch["target"]))
    result["oracle_loss"] = float(F.smooth_l1_loss(batch["target"], batch["target"]))
    return result


def run(*, records, manual_seed, steps, learning_rate):
    set_manual_seed(manual_seed)
    initial = ActionConditionedPredictor(
        32, width=32, heads=4, layers=2, time_scale=1000
    )
    train, development = stack(records[:16]), stack(records[16:])
    # Family sampling is paired and identical across the live/blind controls.
    generator = torch.Generator().manual_seed(manual_seed)
    orders = [torch.randperm(16, generator=generator)[:4] for _ in range(steps)]
    results = []
    for blind in (False, True):
        model = copy.deepcopy(initial)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0, betas=(0.9, 0.95)
        )
        row = {
            "mode": "action_blind" if blind else "actual_actions",
            "manual_seed": manual_seed,
            "parameters": sum(p.numel() for p in model.parameters()),
            "initial": evaluate(model, train, blind),
            "curve": [],
        }
        started = time.monotonic()
        for step, order in enumerate(orders, 1):
            batch = stack([records[int(i)] for i in order])
            optimizer.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(predict(model, batch, blind=blind), batch["target"])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite diagnostic loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            if step == 1 or step % 50 == 0 or step == steps:
                row["curve"].append(
                    {
                        "step": step,
                        "loss": float(loss.detach()),
                        "gradient_norm": float(norm),
                    }
                )
        row.update(
            train=evaluate(model, train, blind),
            development=evaluate(model, development, blind),
            seconds=time.monotonic() - started,
        )
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--manual-seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--data-manual-seed", type=int, default=6227)
    parser.add_argument("--context-tokens", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000 or not 16 <= args.context_tokens <= 2048:
        raise ValueError("diagnostic budget exceeded")
    if args.learning_rate <= 0 or not torch.isfinite(torch.tensor(args.learning_rate)):
        raise ValueError("learning rate must be positive and finite")
    destination = inside(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    report = {
        "schema": "predictor-controlled-diagnostic-v1",
        "complete": False,
        "scope": "synthetic feature-space diagnostic; not robot performance or a formal architecture ablation",
        "device": "cpu",
        "dtype": "float32",
        "threads": 4,
        "train_families": 16,
        "development_families": 8,
        "batch_windows": 8,
        "config": vars(args),
        "rows": [],
    }
    for static in (False, True):
        records = make_pairs(
            manual_seed=args.data_manual_seed,
            static=static,
            context_tokens=args.context_tokens,
        )
        for seed in args.manual_seeds:
            rows = run(
                records=records,
                manual_seed=seed,
                steps=args.steps,
                learning_rate=args.learning_rate,
            )
            for row in rows:
                row["scenario"] = (
                    "static_negative_control" if static else "action_required"
                )
                report["rows"].append(row)
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "scenario",
                                "mode",
                                "manual_seed",
                                "seconds",
                                "development",
                            )
                        }
                    ),
                    flush=True,
                )
            (destination / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
    report["complete"] = True
    (destination / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
