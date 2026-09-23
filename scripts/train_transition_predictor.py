"""Frozen-cache predictor qualification; never updates or loads an AE optimizer.

Diagnostic caches encoded by stage two are allowed for local experiments only.
Promotion requires fresh confirmation from the intended policy encoder and is
NOT inferred from this development report.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from diagnose_transition_targets import ROOT, SOURCE, load
from torch.nn import functional as F

from grounded_interaction.predictive_vla.config import set_manual_seed
from grounded_interaction.predictive_vla.transport import (
    PREDICTOR_TYPES,
)


def predict(model, c, blind=False, swap=False, zero_context=False):
    actions = c["action_sequences"]
    if blind:
        actions = torch.zeros_like(actions)
    elif swap:
        actions = actions.roll(1, 0)
    shared = c["shared"][None].expand(6, -1, -1)
    if zero_context:
        shared = torch.zeros_like(shared)
    return model(
        shared,
        torch.ones(shared.shape[:2], dtype=torch.bool, device=shared.device),
        actions,
        torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device),
        c["positions"][None].expand(6, -1, -1),
        torch.full((6,), actions.shape[1], device=actions.device),
        current=c["current"].expand(6, -1, -1),
        current_valid=c["valid"][None].expand(6, -1),
        state=c["state"][None].expand(6, -1),
    )


@torch.no_grad()
def evaluate(model, cases, blind=False, zero_context=False):
    rows = []
    for c in cases:
        valid = c["valid"]
        p = predict(model, c, blind, zero_context=zero_context)[:, valid]
        y = c["target"][:, valid]
        wrong = predict(model, c, blind, swap=True, zero_context=zero_context)[:, valid]
        pc = p - p.mean(0)
        yc = y - y.mean(0)
        copied = c["current"][:, valid].expand_as(y)
        rows.append(
            {
                "case": c["case"],
                "loss": float(F.smooth_l1_loss(p, y)),
                "swapped_loss": float(F.smooth_l1_loss(wrong, y)),
                "copy_loss": float(F.smooth_l1_loss(copied, y)),
                "action_mse": float((pc - yc).square().mean()),
                "target_action_mse": float(yc.square().mean()),
            }
        )
    result = {}
    for split in sorted({r["case"]["split"] for r in rows}):
        selected = [r for r in rows if r["case"]["split"] == split]
        mean = {
            k: sum(r[k] for r in selected) / len(selected)
            for k in [
                "loss",
                "swapped_loss",
                "copy_loss",
                "action_mse",
                "target_action_mse",
            ]
        }
        mean["action_variance_explained"] = (
            1 - mean["action_mse"] / mean["target_action_mse"]
        )
        mean["copy_relative_gain"] = 1 - mean["loss"] / mean["copy_loss"]
        result[split] = mean
    return {"rows": rows, "summary": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--output", required=True)
    parser.add_argument("--features-directory")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--training-cases", type=int)
    parser.add_argument("--include-id-development", action="store_true")
    parser.add_argument("--manual-seed", type=int, default=17)
    parser.add_argument("--raw-context", action="store_true")
    parser.add_argument(
        "--fusion", choices=["concat", "control_residual"], default="control_residual"
    )
    parser.add_argument("--model-kind", choices=list(PREDICTOR_TYPES), default="affine")
    parser.add_argument("--effect-weight", type=float, default=0.0)
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--zero-context", action="store_true")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    out = Path(args.output).resolve()
    if not source.is_relative_to(ROOT) or not out.is_relative_to(ROOT):
        raise ValueError("paths must stay inside repository")
    if args.effect_weight < 0:
        raise ValueError("effect-weight must be nonnegative")
    if args.steps < 1:
        raise ValueError("steps must be positive")
    out.mkdir(parents=True, exist_ok=False)
    set_manual_seed(args.manual_seed)
    torch.set_num_threads(2)
    features = (
        Path(args.features_directory).resolve()
        if args.features_directory
        else source / "features"
    )
    if not features.is_relative_to(ROOT):
        raise ValueError("feature directory must be in repository")
    provenance_path = features / "encoding.json"
    provenance = (
        json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    )
    started = time.monotonic()
    splits = (
        ("train", "development", "id_development")
        if args.include_id_development
        else ("train", "development")
    )
    cases = load("cuda", source, features, splits=splits)
    if args.training_cases is not None:
        if args.training_cases < 2:
            raise ValueError("training-cases must be at least two")
        selected = [
            c["case"]["case_id"] for c in cases if c["case"]["split"] == "train"
        ][: args.training_cases]
        if len(selected) != args.training_cases:
            raise ValueError("insufficient training cases")
        cases = [
            c
            for c in cases
            if c["case"]["split"] != "train" or c["case"]["case_id"] in selected
        ]
    train = [c for c in cases if c["case"]["split"] == "train"]
    model_class = PREDICTOR_TYPES[args.model_kind]
    model = (
        model_class(
            cases[0]["current"].shape[-1],
            width=64,
            time_scale=1000,
            normalize_context=not args.raw_context,
            fusion=args.fusion,
        )
        .cuda()
        .float()
    )
    states = torch.stack([c["state"] for c in train])
    model.state_mean.copy_(states.mean(0))
    model.state_std.copy_(states.std(0).clamp_min(0.01))
    opt = torch.optim.AdamW(
        model.parameters(), lr=0.0005, weight_decay=0, betas=(0.9, 0.95)
    )
    rng = np.random.default_rng(args.manual_seed)
    curve = []
    for step in range(args.steps):
        c = train[int(rng.integers(len(train)))]
        opt.zero_grad()
        p = predict(model, c, args.blind, zero_context=args.zero_context)
        predicted, target = p[:, c["valid"]], c["target"][:, c["valid"]]
        loss = F.smooth_l1_loss(predicted, target)
        if args.effect_weight:
            # Same physical initial state, six distinct actual action branches.
            pc, yc = predicted - predicted.mean(0), target - target.mean(0)
            loss = loss + args.effect_weight * F.mse_loss(pc, yc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if (step + 1) % 50 == 0:
            curve.append({"step": step + 1, "loss": float(loss.detach())})
            progress = {
                "step": step + 1,
                "total_steps": args.steps,
                "loss": curve[-1]["loss"],
                "seconds": time.monotonic() - started,
                "manual_seed": args.manual_seed,
                "training_cases": len(train),
                "blind": args.blind,
            }
            temporary = out / "progress.partial"
            temporary.write_text(json.dumps(progress) + "\n")
            temporary.replace(out / "progress.json")
    result = evaluate(model, cases, args.blind, args.zero_context)
    report = {
        "schema": "frozen-transport-diagnostic-v1",
        "manual_seed": args.manual_seed,
        "source": str(source),
        "render_profile": json.loads((source / "plan.json").read_text()).get(
            "render_profile", "native"
        ),
        "source_policy": provenance,
        "features_directory": str(features),
        "effective_batch": 6,
        "learning_rate": 0.0005,
        "optimizer": "AdamW(.9,.95), weight_decay=0",
        "normalize_context": not args.raw_context,
        "fusion": args.fusion,
        "steps": args.steps,
        "training_cases": len(train),
        "blind": args.blind,
        "zero_context": args.zero_context,
        "curve": curve,
        "result": result,
        "model_kind": args.model_kind,
        "effect_weight": args.effect_weight,
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "eligible_for_joint_training": False,
        "reason": "Development diagnostics only; independent confirmation and policy-matched encoding required.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(
        {
            "schema": "transport-head-v1",
            "model_kind": args.model_kind,
            "effect_weight": args.effect_weight,
            "predictor": model.state_dict(),
            "manual_seed": args.manual_seed,
            "normalize_context": not args.raw_context,
            "fusion": args.fusion,
            "native_width": cases[0]["current"].shape[-1],
            "width": 64,
            "time_scale": 1000,
            "report": str(out / "report.json"),
        },
        out / "head.pt",
    )
    print(json.dumps(report["result"]["summary"]), flush=True)


if __name__ == "__main__":
    main()
