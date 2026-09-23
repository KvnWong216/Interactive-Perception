"""Two-GPU cached-feature diagnostics, with independently queued head experiments."""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from gpu_devices import verify_cuda_target
from torch import nn
from torch.nn import functional as F

from grounded_interaction.predictive_vla.config import VLAConfig, set_manual_seed
from grounded_interaction.predictive_vla.model import (
    ActionConditionedPredictor,
    patch_prediction_loss,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/stage2_s17_gpu4567/best.pt"


def save(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(out):
    rng = random.Random(6227)
    manifest = ROOT / "data/prepared/stage1_full/manifest.json"
    entries = [
        x for x in json.loads(manifest.read_text())["episodes"] if x["split"] == "train"
    ]
    families = sorted({x["reset_family"] for x in entries})
    # Reserve three smaller families; preserve the two large families for fitting.
    small = [f for f in families if sum(x["reset_family"] == f for x in entries) < 100]
    rng.shuffle(small)
    development = small[:3]
    rows = []
    for split in ["train", "development"]:
        available = [
            x
            for x in entries
            if (x["reset_family"] in development) == (split == "development")
        ]
        tasks = sorted({x["task"] for x in available})
        rng.shuffle(tasks)
        groups = {t: [x for x in available if x["task"] == t] for t in tasks}
        for group in groups.values():
            rng.shuffle(group)
        chosen = []
        while len(chosen) < 32:
            for task in tasks:
                if groups[task] and len(chosen) < 32:
                    chosen.append(groups[task].pop())
        for entry in chosen:
            path = (manifest.parent / entry["path"]).resolve()
            with np.load(path, allow_pickle=False) as data:
                length = len(data["actions"])
            if length < 11:
                raise ValueError("trajectory too short")
            step = rng.randint(1, length - 10)
            rows.append(
                {
                    "index": len(rows),
                    "diagnostic_split": split,
                    "entry": entry,
                    "step": step,
                    "horizon": 10,
                }
            )
    out.mkdir(parents=True, exist_ok=False)
    (out / "cache").mkdir()
    (out / "jobs").mkdir()
    plan = {
        "manual_seed": 6227,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_update": 1500,
        "development_families": development,
        "original_split": "train",
        "rows": rows,
        "claim_limit": "development used here was seen by original stage2 training; not held-out policy generalization",
    }
    save(out / "plan.json", plan)
    jobs = []
    # First wave: existing vs random head, size and LR, all paired across seeds.
    for seed in [17, 29, 43]:
        for n in [1, 8, 32]:
            for lr in [0.0001, 0.0005]:
                for init in ["stage2", "random"]:
                    jobs.append(
                        {
                            "variant": "original",
                            "initialization": init,
                            "n": n,
                            "lr": lr,
                            "manual_seed": seed,
                            "steps": 300,
                        }
                    )
    # Four candidates share current-patch input: isolate residual/fusion effects.
    for seed in [17, 29, 43]:
        for variant in [
            "current_absolute_mixed",
            "current_residual_mixed",
            "current_absolute_separate",
            "current_residual_separate",
        ]:
            jobs.append(
                {
                    "variant": variant,
                    "initialization": "random",
                    "n": 32,
                    "lr": 0.0001,
                    "manual_seed": seed,
                    "steps": 300,
                }
            )
    for i, job in enumerate(jobs):
        job["job_id"] = i
    save(out / "queue.json", jobs)
    print(
        json.dumps(
            {
                "windows": len(rows),
                "jobs": len(jobs),
                "development_families": development,
            }
        ),
        flush=True,
    )


def encode(out, shard, physical_gpu):
    from grounded_interaction.predictive_vla.backend import NativeVLABackend
    from grounded_interaction.predictive_vla.data import Trajectory, TrajectoryDataset
    from grounded_interaction.predictive_vla.training import load_checkpoint

    mapping = verify_cuda_target(physical_gpu)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
    config = VLAConfig(**payload["config"])
    del payload
    backend = NativeVLABackend.from_pretrained(
        config, device="cuda:0", local_path=ROOT / "checkpoints/base/MolmoAct2-LIBERO"
    )
    for p in backend.model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    load_checkpoint(CHECKPOINT, backend)
    backend.model.requires_grad_(False)
    backend.train(False)
    dataset = TrajectoryDataset(
        ROOT / "data/prepared/stage1_full/manifest.json", config
    )
    plan = json.loads((out / "plan.json").read_text())
    started = time.monotonic()
    for row in plan["rows"][shard::2]:
        path = out / "cache" / f"{row['index']:03d}.pt"
        if path.exists():
            raise ValueError("refusing stale cache reuse")
        trajectory = Trajectory(
            dataset.resolve(row["entry"]), row["entry"], total_steps=config.total_steps
        )
        step = row["step"]
        h = row["horizon"]
        context = trajectory.context(step, config)
        actual = trajectory.arrays["actions"][step : step + h].copy()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            shared = backend.encode(context)
            target, positions, valid = backend.target(
                trajectory.observation(step + h), context.task
            )
            current, cp, cv = backend.target(context.current, context.task)
            if not torch.equal(positions, cp):
                raise ValueError("teacher grids changed")
            cache = {
                "provenance": row,
                "shared": shared.hidden.cpu(),
                "target": target.cpu(),
                "positions": positions.cpu(),
                "valid": (valid & cv).cpu(),
                "current": current.cpu(),
                "actions": backend.normalize_actions(actual.copy())[None].float().cpu(),
            }
            if not np.array_equal(
                actual, trajectory.arrays["actions"][step : step + h]
            ):
                raise ValueError("normalizer mutated actions")
            temporary = path.with_suffix(".partial")
            torch.save(cache, temporary)
            temporary.replace(path)
        print(
            json.dumps(
                {
                    "cached": row["index"],
                    "split": row["diagnostic_split"],
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
        del trajectory, shared, cache
    save(
        out / f"encoding_{shard}_complete.json",
        {
            "complete": True,
            **mapping,
            "seconds": time.monotonic() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        },
    )


class Candidate(nn.Module):
    """Diagnostic-only 2x2; native action policy never calls this class."""

    def __init__(self, base, variant):
        super().__init__()
        self.base = base
        self.variant = variant
        if variant in {"fourier_queries", "linear_query_control", "fourier_separate"}:
            self.position = nn.Linear(24, base.output.in_features)
            nn.init.zeros_(self.position.weight)
            nn.init.zeros_(self.position.bias)
        elif variant not in {"original", "separate_actions"}:
            self.current = nn.Linear(base.output.out_features, base.output.in_features)

    def forward(self, x, *, wrong=False):
        b = self.base
        shared = x["shared"]
        actions = x["actions"]
        # Within-queue donor is supplied explicitly; not a counterfactual future.
        if wrong:
            actions = x["donor_actions"]
        h = actions.shape[1]
        cm = torch.ones(shared.shape[:2], dtype=torch.bool, device=shared.device)
        am = torch.ones(actions.shape[:2], dtype=torch.bool, device=shared.device)
        if self.variant == "original":
            return b(
                shared,
                cm,
                actions,
                am,
                x["positions"],
                torch.tensor([h], device=shared.device),
            )
        times = torch.arange(1, h + 1, device=shared.device)[None, :, None] / h
        action = b.action(torch.cat((actions, times), -1))
        context = b.context(shared)
        query = b.query(
            torch.cat(
                (
                    x["positions"],
                    torch.full(
                        (*x["positions"].shape[:2], 1),
                        h / b.time_scale,
                        device=shared.device,
                    ),
                ),
                dim=-1,
            )
        )
        if hasattr(self, "position"):
            uv = x["positions"][..., :2, None]
            if self.variant.startswith("fourier"):
                phase = (
                    uv * torch.tensor([1, 2, 4, 8, 16, 32], device=uv.device) * torch.pi
                )
                features = torch.cat((phase.sin(), phase.cos()), -1).flatten(-2)
            else:
                features = torch.cat(
                    (
                        uv.expand(*uv.shape[:-1], 6),
                        torch.ones_like(uv).expand(*uv.shape[:-1], 6),
                    ),
                    -1,
                ).flatten(-2)
            query = query + self.position(features)
        elif hasattr(self, "current"):
            current = F.layer_norm(x["current"], (x["current"].shape[-1],))
            query = query + self.current(current)
        for block in b.readouts:
            if self.variant.endswith("separate") or self.variant == "separate_actions":
                # Same parameters and parallel readouts, independently normalized memories.
                query = (block(query, context, cm) + block(query, action, am)) * 0.5
            else:
                query = block(
                    query, torch.cat((context, action), 1), torch.cat((cm, am), 1)
                )
        result = b.output(query)
        return current + result if "_residual_" in self.variant else result


def load_caches(out):
    plan = json.loads((out / "plan.json").read_text())
    result = []
    for row in plan["rows"]:
        x = torch.load(
            out / "cache" / f"{row['index']:03d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        if x.pop("provenance") != row:
            raise ValueError("cache provenance differs")
        x = {
            k: v.to(
                "cuda", dtype=torch.bool if v.dtype == torch.bool else torch.float32
            )
            for k, v in x.items()
        }
        result.append(x)
    for split in [result[:32], result[32:]]:
        for i, x in enumerate(split):
            x["donor_actions"] = split[(i + 1) % len(split)]["actions"]
    return result


@torch.no_grad()
def evaluate(model, data):
    rows = []
    for x in data:
        p = model(x)
        target = x["target"]
        valid = x["valid"]
        truth = F.layer_norm(target, (target.shape[-1],))
        current = F.layer_norm(x["current"], (target.shape[-1],))
        change = (truth - current).square().mean(-1)
        changed = torch.zeros_like(valid)
        flat = change[valid]
        cut = torch.quantile(flat, 0.75)
        changed = valid & (change >= cut)
        rows.append(
            {
                "loss": float(patch_prediction_loss(p, target, valid)),
                "changed_loss": float(patch_prediction_loss(p, target, changed)),
                "copy_loss": float(patch_prediction_loss(current, target, valid)),
                "copy_changed_loss": float(
                    patch_prediction_loss(current, target, changed)
                ),
                "wrong_loss": float(
                    patch_prediction_loss(model(x, wrong=True), target, valid)
                ),
            }
        )
    return {
        "windows": rows,
        "mean": {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]},
    }


def worker(out, worker_id, physical_gpu):
    mapping = verify_cuda_target(physical_gpu)
    data = load_caches(out)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
    head = {k: v.clone() for k, v in payload["predictor"].items()}
    del payload
    queue = json.loads((out / "queue.json").read_text())
    for job in queue:
        if job["job_id"] % 8 != worker_id:
            continue
        set_manual_seed(job["manual_seed"])
        base = ActionConditionedPredictor(
            head["context.weight"].shape[1],
            width=256,
            heads=8,
            layers=2,
            time_scale=1000,
        )
        if job["initialization"] == "stage2":
            base.load_state_dict(head, strict=True)
        model = Candidate(base, job["variant"]).cuda()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=job["lr"], weight_decay=0, betas=(0.9, 0.95)
        )
        train = data[: job["n"]]
        batch = min(8, len(train))
        rng = random.Random(job["manual_seed"])
        torch.cuda.reset_peak_memory_stats()
        report = {
            "complete": False,
            "job": job,
            "pid": os.getpid(),
            **mapping,
            "microbatch": 1,
            "accumulation": batch,
            "parameters": sum(p.numel() for p in model.parameters()),
            "initial_train": evaluate(model, train),
            "curve": [],
            "warning": "donor action sensitivity is observational, not a same-state causal test",
        }
        path = out / "jobs" / f"job_{job['job_id']:03d}.json"
        if path.exists():
            raise ValueError("refusing to overwrite experiment")
        save(path, report)
        started = time.monotonic()
        for step in range(1, job["steps"] + 1):
            optimizer.zero_grad(set_to_none=True)
            selected = rng.sample(range(len(train)), batch)
            loss_sum = 0.0
            for i in selected:
                x = train[i]
                loss = patch_prediction_loss(model(x), x["target"], x["valid"]) / batch
                loss.backward()
                loss_sum += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            if step == 1 or step % 50 == 0:
                report["curve"].append(
                    {
                        "step": step,
                        "loss": loss_sum,
                        "grad_norm": float(norm),
                        "seconds": time.monotonic() - started,
                    }
                )
                save(path, report)
                print(
                    json.dumps(
                        {
                            "worker": worker_id,
                            "job": job["job_id"],
                            "step": step,
                            "loss": loss_sum,
                        }
                    ),
                    flush=True,
                )
        report.update(
            complete=True,
            train=evaluate(model, train),
            development=evaluate(model, data[32:]),
            seconds=time.monotonic() - started,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        )
        save(path, report)
        torch.save(
            {"job": job, "state": model.cpu().state_dict()},
            out / "jobs" / f"job_{job['job_id']:03d}.pt",
        )
        del model, optimizer
        torch.cuda.empty_cache()
    save(
        out / f"worker_{worker_id}_complete.json",
        {"complete": True, "worker": worker_id, **mapping},
    )


def baseline(out, physical_gpu):
    mapping = verify_cuda_target(physical_gpu)
    data = load_caches(out)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
    base = ActionConditionedPredictor(
        payload["predictor"]["context.weight"].shape[1],
        width=256,
        heads=8,
        layers=2,
        time_scale=1000,
    )
    base.load_state_dict(payload["predictor"], strict=True)
    del payload
    model = Candidate(base, "original").cuda()
    save(
        out / "baseline.json",
        {
            "checkpoint_update": 1500,
            "optimizer_steps": 0,
            **mapping,
            "train": evaluate(model, data[:32]),
            "development": evaluate(model, data[32:]),
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "encode", "worker", "baseline"])
    p.add_argument("--output", required=True)
    p.add_argument("--id", type=int, default=0)
    p.add_argument("--physical-gpu", type=int, default=2)
    a = p.parse_args()
    out = (ROOT / a.output).resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("output escaped repository")
    torch.set_num_threads(2)
    set_manual_seed(6227)
    if a.mode == "prepare":
        prepare(out)
    elif a.mode == "encode":
        encode(out, a.id, a.physical_gpu)
    elif a.mode == "baseline":
        baseline(out, a.physical_gpu)
    else:
        worker(out, a.id, a.physical_gpu)


if __name__ == "__main__":
    main()
