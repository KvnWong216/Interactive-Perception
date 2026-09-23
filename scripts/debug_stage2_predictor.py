"""Read-only CPU probes of the actual stage-two predictor and real validation contexts."""

import argparse
import json
import time

import numpy as np
import torch
from evaluate_stage1 import ROOT, inside, save_report
from torch.nn import functional as F

from grounded_interaction.predictive_vla.backend import NativeVLABackend
from grounded_interaction.predictive_vla.config import load_config, set_manual_seed
from grounded_interaction.predictive_vla.data import Trajectory, TrajectoryDataset
from grounded_interaction.predictive_vla.model import (
    ActionConditionedPredictor,
    patch_prediction_loss,
)


def probe(predictor, cache):
    shared = cache["shared"].float().clone().requires_grad_()
    actions = cache["actions"].float().clone().requires_grad_()
    positions = cache["positions"].float()
    cm = torch.ones(shared.shape[:2], dtype=torch.bool)
    am = torch.ones(actions.shape[:2], dtype=torch.bool)
    horizon = torch.tensor([actions.shape[1]])
    target = cache["target"].float()
    valid = cache["valid"]
    truth = F.layer_norm(target, (target.shape[-1],))
    attention = []

    def request_weights(_module, args, kwargs):
        return args, {**kwargs, "need_weights": True, "average_attn_weights": False}

    def record_attention(_module, _args, output):
        mass = output[1].detach()[..., shared.shape[1] :].sum(-1)
        attention.append(
            {
                "action_mass_mean": float(mass.mean()),
                "action_mass_by_head": mass.mean((0, 2)).tolist(),
            }
        )

    hooks = []
    for block in predictor.readouts:
        hooks.extend(
            [
                block.attention.register_forward_pre_hook(
                    request_weights, with_kwargs=True
                ),
                block.attention.register_forward_hook(record_attention),
            ]
        )
    prediction = predictor(shared, cm, actions, am, positions, horizon)
    loss = patch_prediction_loss(prediction, target, valid)
    loss.backward()
    gradients = {
        n: float(p.grad.norm()) if p.grad is not None else None
        for n, p in predictor.named_parameters()
    }
    for hook in hooks:
        hook.remove()
    with torch.no_grad():
        ordinary = predictor(shared, cm, actions, am, positions, horizon)
        variants = {
            "actual": ordinary,
            "normalized_actual": F.layer_norm(ordinary, (ordinary.shape[-1],)),
            "zero_control": predictor(
                shared, cm, cache["zero_actions"], am, positions, horizon
            ),
            "reverse_time": predictor(
                shared, cm, actions.flip(1), am, positions, horizon
            ),
            "donor_actions": predictor(
                shared, cm, cache["donor_actions"], am, positions, horizon
            ),
            "zero_context": predictor(
                torch.zeros_like(shared), cm, actions, am, positions, horizon
            ),
            "mean_context": predictor(
                shared.mean(1, keepdim=True).expand_as(shared),
                cm,
                actions,
                am,
                positions,
                horizon,
            ),
            "constant_queries": predictor(
                shared,
                cm,
                actions,
                am,
                positions.mean(1, keepdim=True).expand_as(positions),
                horizon,
            ),
            "copy_current": F.layer_norm(cache["current"].float(), (target.shape[-1],)),
        }
        scores = {
            name: {
                "loss": float(patch_prediction_loss(v, target, valid)),
                "prediction_rms_delta": float(
                    (v - ordinary)[valid].square().mean().sqrt()
                ),
            }
            for name, v in variants.items()
        }

        def spatial_rms(value):
            values = value[valid]
            return float((values - values.mean(0, keepdim=True)).square().mean().sqrt())

        result = {
            "context_tokens": shared.shape[1],
            "action_tokens": actions.shape[1],
            "uniform_action_mass": actions.shape[1]
            / (shared.shape[1] + actions.shape[1]),
            "teacher_width": target.shape[-1],
            "patches": int(valid.sum()),
            "attention": attention,
            "instrumentation_max_output_difference": float(
                (prediction - ordinary).abs().max()
            ),
            "shared_gradient_rms": float(shared.grad.square().mean().sqrt()),
            "action_gradient_rms": float(actions.grad.square().mean().sqrt()),
            "parameter_gradient_norms": gradients,
            "scores": scores,
            "predicted_spatial_rms": spatial_rms(ordinary),
            "target_spatial_rms": spatial_rms(truth),
        }
    predictor.zero_grad(set_to_none=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/stage2_predictor_debug_s5227")
    parser.add_argument("--windows", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.windows <= 4:
        raise ValueError("debug is bounded to 1-4 validation tasks")
    output = inside(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    set_manual_seed(5227)
    config = load_config(ROOT / "experiments/stage2_policy.yaml")
    payload = torch.load(
        ROOT / "runs/stage2_s17_gpu4567/best.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    assert payload["updates"] == 1500 and payload["config"] == config.to_dict()
    predictor = ActionConditionedPredictor(
        payload["predictor"]["context.weight"].shape[1],
        width=config.predictor_dim,
        heads=config.predictor_heads,
        layers=config.predictor_layers,
        time_scale=config.total_steps,
    )
    predictor.load_state_dict(payload["predictor"], strict=True)
    predictor.eval()
    plan = json.loads(
        (ROOT / "runs/stage2_assessment_s3217_gpu1/prediction_windows.json").read_text()
    )
    selected = []
    tasks = set()
    for row in plan:
        if row["split"] == "validation" and row["task"] not in tasks:
            tasks.add(row["task"])
            selected.append(row)
    selected = selected[: args.windows]
    report = {
        "complete": False,
        "manual_seed": 5227,
        "device": "cpu",
        "updates": 1500,
        "optimizer_steps": 0,
        "selection": "first frozen diagnostic window of each of the first validation tasks; no outcome selection",
        "precision": "CPU native backbone bfloat16, predictor float32; mechanism debug, not replacement benchmark",
        "rows": [],
    }
    save_report(output / "report.json", report)
    missing = any(
        not (output / f"context_{i}.pt").exists() for i in range(len(selected))
    )
    backend = None
    if missing:
        print("Loading frozen policy on CPU; GPU jobs are untouched", flush=True)
        backend = NativeVLABackend.from_pretrained(
            config, device="cpu", local_path=ROOT / "checkpoints/base/MolmoAct2-LIBERO"
        )
        for parameter in backend.model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        state = backend.model.state_dict()
        if any(
            k not in state or state[k].shape != v.shape
            for k, v in payload["adapters"].items()
        ):
            raise ValueError("checkpoint does not match policy")
        backend.model.load_state_dict(payload["adapters"], strict=False)
        if any(
            not torch.equal(backend.model.state_dict()[k], v)
            for k, v in payload["adapters"].items()
        ):
            raise ValueError("policy checkpoint restoration not exact")
        backend.model.requires_grad_(False)
        backend.train(False)
    del payload
    dataset = TrajectoryDataset(
        ROOT / "data/prepared/stage1_full/manifest.json", config
    )
    entries = {r["episode_id"]: r for r in dataset.entries}
    started = time.monotonic()
    for i, row in enumerate(selected):
        path = output / f"context_{i}.pt"
        if not path.exists():
            entry = entries[row["episode_id"]]
            trajectory = Trajectory(
                dataset.resolve(entry), entry, total_steps=config.total_steps
            )
            actual = np.asarray(row["actual_actions"], dtype=np.float32)
            step = row["step"]
            assert np.array_equal(
                actual, trajectory.arrays["actions"][step : step + 10]
            )
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                context = trajectory.context(step, config)
                print(
                    json.dumps(
                        {
                            "encoding": i,
                            "episode": row["episode_id"],
                            "steps": [o.step for o in context.observations],
                        }
                    ),
                    flush=True,
                )
                shared = backend.encode(context)
                target, positions, valid = backend.target(
                    trajectory.observation(step + 10), context.task
                )
                current, cp, cv = backend.target(context.current, context.task)
                if not torch.equal(positions, cp):
                    raise ValueError("teacher grid changed")
                normalized = backend.normalize_actions(actual.copy())[None]
                assert np.array_equal(
                    actual, np.asarray(row["actual_actions"], dtype=np.float32)
                )
                cache = {
                    "shared": shared.hidden.detach().cpu(),
                    "target": target.cpu(),
                    "current": current.cpu(),
                    "positions": positions.cpu(),
                    "valid": (valid & cv).cpu(),
                    "actions": normalized.cpu(),
                    "zero_actions": backend.normalize_actions(np.zeros_like(actual))[
                        None
                    ]
                    .float()
                    .cpu(),
                    "donor_actions": backend.normalize_actions(
                        np.asarray(row["donor_actions"], dtype=np.float32)
                    )[None]
                    .float()
                    .cpu(),
                }
                torch.save(cache, path)
            del trajectory, shared, cache
        cache = torch.load(path, map_location="cpu", weights_only=True)
        result = probe(predictor, cache)
        result.update(
            episode_id=row["episode_id"], step=row["step"], split="validation"
        )
        report["rows"].append(result)
        report["seconds"] = time.monotonic() - started
        save_report(output / "report.json", report)
        print(
            json.dumps(
                {
                    "window": i,
                    "scores": result["scores"],
                    "attention": result["attention"],
                }
            ),
            flush=True,
        )
    report["complete"] = True
    save_report(output / "report.json", report)


if __name__ == "__main__":
    main()
