"""Train/evaluate action predictors on audited same-state robot forks."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from gpu_devices import verify_cuda_target
from run_real_predictor_debug import Candidate, save
from torch.nn import functional as F

from grounded_interaction.predictive_vla.config import set_manual_seed
from grounded_interaction.predictive_vla.model import (
    ActionConditionedPredictor,
    patch_prediction_loss,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/stage2_forks_s7227"


def normalize(x):
    return F.layer_norm(x, (x.shape[-1],))


def dataset(horizon=10, *, confirmation=False):
    cases = []
    for path in sorted((OUT / "features").glob("case_*.pt")):
        cache = torch.load(path, map_location="cpu", weights_only=True)
        if (cache["case"]["split"] == "confirmation") != confirmation:
            continue
        inputs = []
        common = {
            k: cache[k].to(
                "cuda",
                dtype=torch.bool if cache[k].dtype == torch.bool else torch.float32,
            )
            for k in ["shared", "current", "positions", "valid"]
        }
        for branch in cache["branches"][:6]:
            endpoint = next(e for e in branch["endpoints"] if e["horizon"] == horizon)
            inputs.append(
                {
                    **common,
                    "target": endpoint["target"].to("cuda", dtype=torch.float32),
                    "actions": endpoint["actions"].to("cuda", dtype=torch.float32),
                }
            )
        noise = max(
            max(e["repeat_max_abs"].values())
            for b in cache["source_report"]["records"][6:]
            for e in b["endpoints"]
        )
        cases.append(
            {"case": cache["case"], "inputs": inputs, "repeat_noise_max": noise}
        )
    return cases


def make_model(variant, seed):
    set_manual_seed(seed)
    payload = torch.load(
        ROOT / "runs/stage2_s17_gpu4567/best.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    head = payload["predictor"]
    base = ActionConditionedPredictor(
        head["context.weight"].shape[1], width=256, heads=8, layers=2, time_scale=1000
    )
    base.load_state_dict(head, strict=True)
    return Candidate(base, variant).cuda()


def infer(model, x, blind=False):
    if blind:
        x = {**x, "actions": torch.zeros_like(x["actions"])}
    return model(x)


@torch.no_grad()
def evaluate(model, cases, blind=False):
    rows = []
    for case in cases:
        inputs = case["inputs"]
        valid = inputs[0]["valid"]
        p = torch.cat([infer(model, x, blind) for x in inputs])[:, valid[0]]
        y = torch.cat([normalize(x["target"]) for x in inputs])[:, valid[0]]
        current = normalize(inputs[0]["current"])[:, valid[0]].expand_as(y)
        error = F.smooth_l1_loss(p, y, reduction="none").mean((1, 2))
        copy = F.smooth_l1_loss(current, y, reduction="none").mean((1, 2))
        pairs = []
        for i in range(6):
            for j in range(i + 1, 6):
                target_separation = float((y[i] - y[j]).square().mean().sqrt())
                actual = float((error[i] + error[j]) / 2)
                wrong = float(
                    (F.smooth_l1_loss(p[i], y[j]) + F.smooth_l1_loss(p[j], y[i])) / 2
                )
                midpoint = (y[i] + y[j]) / 2
                bound = float(
                    (
                        F.smooth_l1_loss(midpoint, y[i])
                        + F.smooth_l1_loss(midpoint, y[j])
                    )
                    / 2
                )
                pairs.append(
                    {
                        "i": i,
                        "j": j,
                        "correct_loss": actual,
                        "wrong_loss": wrong,
                        "blind_bound": bound,
                        "target_separation_rms": target_separation,
                        "prediction_separation_rms": float(
                            (p[i] - p[j]).square().mean().sqrt()
                        ),
                    }
                )
        rows.append(
            {
                "case": case["case"],
                "loss": float(error.mean()),
                "copy_loss": float(copy.mean()),
                "repeat_noise_max": case["repeat_noise_max"],
                "pairs": pairs,
                "mean_component_mse": float((p.mean(0) - y.mean(0)).square().mean()),
                "action_component_mse": float(
                    ((p - p.mean(0)) - (y - y.mean(0))).square().mean()
                ),
                "target_action_mse": float((y - y.mean(0)).square().mean()),
            }
        )
    pairs = [p for r in rows for p in r["pairs"] if p["target_separation_rms"] > 1e-5]
    return {
        "rows": rows,
        "mean_loss": sum(r["loss"] for r in rows) / len(rows),
        "copy_loss": sum(r["copy_loss"] for r in rows) / len(rows),
        "separable_pairs": len(pairs),
        "all_pairs": len(rows) * 15,
        "pair_assignment_accuracy": sum(
            p["wrong_loss"] > p["correct_loss"] + 1e-7 for p in pairs
        )
        / max(1, len(pairs)),
        "pair_tie_fraction": sum(
            abs(p["wrong_loss"] - p["correct_loss"]) <= 1e-7 for p in pairs
        )
        / max(1, len(pairs)),
        "wrong_minus_correct": sum(p["wrong_loss"] - p["correct_loss"] for p in pairs)
        / max(1, len(pairs)),
        "pair_blind_bound": sum(p["blind_bound"] for p in pairs) / max(1, len(pairs)),
        "separation_ratio": sum(
            p["prediction_separation_rms"] / p["target_separation_rms"] for p in pairs
        )
        / max(1, len(pairs)),
    }


@torch.no_grad()
def teacher_probe(cases):
    """Train-only normalized ridge probe; true future is permitted only in this diagnostic."""
    features = []
    current_features = []
    labels = []
    splits = []
    for case in cases:
        for branch, x in enumerate(case["inputs"]):
            # Preserve coarse spatial moments, separately in each camera.
            positions = x["positions"][0]
            valid = x["valid"][0]
            values = []
            currents = []
            for view in torch.unique(positions[:, 2]):
                mask = valid & (positions[:, 2] == view)
                uv = positions[mask, :2]
                weights = torch.stack(
                    (
                        torch.ones_like(uv[:, 0]),
                        uv[:, 0],
                        uv[:, 1],
                        uv[:, 0] * uv[:, 1],
                    ),
                    0,
                )
                values.append(
                    weights
                    @ (
                        normalize(x["target"])[0, mask]
                        - normalize(x["current"])[0, mask]
                    )
                    / mask.sum()
                )
                currents.append(weights @ normalize(x["current"])[0, mask] / mask.sum())
            features.append(torch.cat(values).flatten())
            current_features.append(torch.cat(currents).flatten())
            labels.append(branch)
            splits.append(case["case"]["split"])
    y = torch.tensor(labels, device="cuda")
    train = torch.tensor([s == "train" for s in splits], device="cuda")
    results = {}
    for name, values in [
        ("future_minus_current", features),
        ("current_only", current_features),
    ]:
        x = torch.stack(values).double()
        mean = x[train].mean(0)
        scale = x[train].std(0).clamp_min(0.05)
        x = (x - mean) / scale
        x = torch.cat((x, torch.ones((len(x), 1), device="cuda")), 1)
        xt = x[train]
        kernel = xt @ xt.T / xt.shape[1]
        alpha = torch.linalg.solve(
            kernel + 0.01 * torch.eye(len(xt), device="cuda", dtype=torch.float64),
            F.one_hot(y[train], 6).double(),
        )
        logits = (x @ xt.T / xt.shape[1]) @ alpha
        results[name] = {
            split: float((logits[mask].argmax(1) == y[mask]).double().mean())
            for split, mask in [("train", train), ("development", ~train)]
        }
    return {
        "branch_accuracy": results,
        "chance": 1 / 6,
        "ridge": 0.01,
        "feature": "teacher spatial moments by view; no actions or robot state input",
    }


def prepare():
    out = OUT / "fits"
    out.mkdir(exist_ok=False)
    (out / "jobs").mkdir()
    jobs = []
    for seed in [17, 29, 43]:
        for variant in [
            "original",
            "separate_actions",
            "fourier_queries",
            "fourier_separate",
            "current_residual_separate",
        ]:
            for blind in [False, True]:
                jobs.append(
                    {
                        "job_id": len(jobs),
                        "variant": variant,
                        "blind": blind,
                        "manual_seed": seed,
                        "steps": 600,
                        "lr": 0.0005,
                    }
                )
    save(out / "queue.json", jobs)


def train_worker(worker_id, run_name="fits"):
    out = OUT / run_name
    cases = dataset()
    train = [c for c in cases if c["case"]["split"] == "train"]
    dev = [c for c in cases if c["case"]["split"] == "development"]
    for job in json.loads((out / "queue.json").read_text()):
        if job["job_id"] % 8 != worker_id:
            continue
        model = make_model(job["variant"], job["manual_seed"])
        selected_train = train[: job.get("n_cases", len(train))]
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=job["lr"], weight_decay=0, betas=(0.9, 0.95)
        )
        rng = random.Random(job["manual_seed"])
        started = time.monotonic()
        path = out / "jobs" / f"job_{job['job_id']:03d}.json"
        if path.exists():
            raise ValueError("refusing result overwrite")
        report = {
            "complete": False,
            "job": job,
            "curve": [],
            "training_cases": len(selected_train),
            "development_cases": len(dev),
            "effective_batch": 6,
            "init": "stage2 best common weights; zero new query projection if present",
        }
        for step in range(1, job["steps"] + 1):
            case = selected_train[rng.randrange(len(selected_train))]
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for x in case["inputs"]:
                loss = (
                    patch_prediction_loss(
                        infer(model, x, job["blind"]), x["target"], x["valid"]
                    )
                    / 6
                )
                loss.backward()
                total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            if step == 1 or step % 100 == 0:
                report["curve"].append(
                    {"step": step, "loss": total, "seconds": time.monotonic() - started}
                )
                save(path, report)
                print(
                    json.dumps({"job": job["job_id"], "step": step, "loss": total}),
                    flush=True,
                )
        report.update(
            complete=True,
            train=evaluate(model, selected_train, job["blind"]),
            development=evaluate(model, dev, job["blind"]),
            seconds=time.monotonic() - started,
        )
        save(path, report)
        torch.save(
            {"job": job, "state": model.cpu().state_dict()}, path.with_suffix(".pt")
        )
        del model, optimizer
        torch.cuda.empty_cache()
    save(out / f"worker_{worker_id}_complete.json", {"complete": True})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "audit", "train"])
    p.add_argument("--id", type=int, default=0)
    p.add_argument("--gpu", type=int, default=2)
    p.add_argument("--run-name", default="fits")
    a = p.parse_args()
    set_manual_seed(7227)
    torch.set_num_threads(2)
    if a.mode == "prepare":
        prepare()
        return
    verify_cuda_target(a.gpu)
    if a.mode == "train":
        if not a.run_name.replace("_", "").isalnum():
            raise ValueError("invalid diagnostic run name")
        train_worker(a.id, a.run_name)
        return
    report = {
        "complete": False,
        "manual_seed": 7227,
        "confirmation_read": False,
        "horizons": {},
    }
    for h in [5, 10, 30]:
        cases = dataset(h)
        model = make_model("original", 17)
        report["horizons"][h] = {
            "baseline": evaluate(model, cases),
            "teacher_probe": teacher_probe(cases),
        }
        save(OUT / "audit.json", report)
        del cases, model
        torch.cuda.empty_cache()
    report["complete"] = True
    save(OUT / "audit.json", report)


if __name__ == "__main__":
    main()
