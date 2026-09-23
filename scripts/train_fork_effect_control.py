"""Matched control: absolute latent loss versus explicitly weighted action effects."""

import argparse
import json
import random
import time

import torch
from fork_predictor_lab import OUT, dataset, evaluate, infer, make_model, normalize
from gpu_devices import verify_cuda_target
from run_real_predictor_debug import Candidate, save
from torch.nn import functional as F

from grounded_interaction.predictive_vla.config import set_manual_seed
from grounded_interaction.predictive_vla.model import ActionConditionedPredictor


def effect_loss(predicted, target, scale):
    """Center over same-state actions; never center across unrelated states."""
    p = predicted - predicted.mean(0, keepdim=True)
    y = target.detach() - target.detach().mean(0, keepdim=True)
    return F.smooth_l1_loss(p / scale, y / scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--run-name", default="effect_fits")
    args = parser.parse_args()
    verify_cuda_target(args.gpu)
    torch.set_num_threads(2)
    cases = dataset()
    train = [c for c in cases if c["case"]["split"] == "train"]
    dev = [c for c in cases if c["case"]["split"] == "development"]
    with torch.no_grad():
        variances = []
        for case in train:
            y = torch.cat([normalize(x["target"]) for x in case["inputs"]])[
                :, case["inputs"][0]["valid"][0]
            ]
            variances.append((y - y.mean(0)).square().mean())
        scale = float(torch.stack(variances).mean().sqrt())
    if scale < 1e-5:
        raise ValueError("training teacher has no action effects")
    if not args.run_name.replace("_", "").isalnum():
        raise ValueError("invalid run name")
    out = OUT / args.run_name
    for job in json.loads((out / "queue.json").read_text()):
        if job["job_id"] % 8 != args.id:
            continue
        if "width" in job:
            set_manual_seed(job["manual_seed"])
            base = ActionConditionedPredictor(
                train[0]["inputs"][0]["shared"].shape[-1],
                width=job["width"],
                heads=8,
                layers=2,
                time_scale=1000,
            )
            torch.nn.init.zeros_(base.output.weight)
            torch.nn.init.zeros_(base.output.bias)
            model = Candidate(base, job["variant"]).cuda()
        else:
            model = make_model(job["variant"], job["manual_seed"])
        opt = torch.optim.AdamW(
            model.parameters(), lr=0.0005, betas=(0.9, 0.95), weight_decay=0
        )
        rng = random.Random(job["manual_seed"])
        started = time.monotonic()
        path = out / "jobs" / f"job_{job['job_id']:03d}.json"
        if path.exists():
            raise ValueError("refusing to overwrite results")
        report = {
            "complete": False,
            "job": job,
            "train_only_effect_scale": scale,
            "curve": [],
            "initialization": "random with zero output; identical copy-current initial function"
            if "width" in job
            else "stage2 best shared weights, same initializer and batch stream within each seed",
            "parameters": sum(p.numel() for p in model.parameters()),
            "batch": 6,
            "dtype": "float32",
            "grad_clip": 1.0,
        }
        for step in range(1, 601):
            case = train[rng.randrange(len(train))]
            valid = case["inputs"][0]["valid"][0]
            opt.zero_grad(set_to_none=True)
            p = torch.cat([infer(model, x, job["blind"]) for x in case["inputs"]])[
                :, valid
            ]
            y = torch.cat([normalize(x["target"]) for x in case["inputs"]])[:, valid]
            absolute = F.smooth_l1_loss(p, y)
            effect = effect_loss(p, y, scale)
            loss = absolute + job["effect_weight"] * effect
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
            if step == 1 or step % 100 == 0:
                report["curve"].append(
                    {
                        "step": step,
                        "absolute_loss": float(absolute.detach()),
                        "effect_loss": float(effect.detach()),
                        "grad_norm": float(norm),
                        "seconds": time.monotonic() - started,
                    }
                )
                save(path, report)
                print(json.dumps({"job": job["job_id"], "step": step}), flush=True)
        report.update(
            complete=True,
            train=evaluate(model, train, job["blind"]),
            development=evaluate(model, dev, job["blind"]),
            seconds=time.monotonic() - started,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        )
        save(path, report)
        torch.save(
            {"job": job, "state": model.cpu().state_dict()}, path.with_suffix(".pt")
        )
        del model, opt
        torch.cuda.empty_cache()
    save(out / f"worker_{args.id}_complete.json", {"complete": True})


if __name__ == "__main__":
    main()
