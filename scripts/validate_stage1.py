"""Compare best/last on two fixed windows from every validation trajectory."""

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch
from evaluate_stage1 import ROOT, inside, save_report
from gpu_devices import verify_cuda_target

from grounded_interaction.predictive_vla.backend import NativeVLABackend
from grounded_interaction.predictive_vla.config import load_config, set_manual_seed
from grounded_interaction.predictive_vla.data import (
    TrainingExample,
    Trajectory,
    TrajectoryDataset,
)
from grounded_interaction.predictive_vla.training import joint_loss, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manual-seed", type=int, default=117)
    args = parser.parse_args()
    device = verify_cuda_target(7)
    config = load_config(ROOT / "experiments/stage1_policy.yaml")
    if config.prediction_weight or config.language_weight:
        raise ValueError(
            "this diagnostic requires the action-only stage-one configuration"
        )
    output = inside(args.output)
    output.mkdir(parents=True, exist_ok=False)
    dataset = TrajectoryDataset(
        ROOT / "data/prepared/stage1_full/manifest.json", config
    )
    rows = sorted(
        (r for r in dataset.entries if r["split"] == "validation"),
        key=lambda r: r["episode_id"],
    )
    rng = np.random.default_rng(args.manual_seed)
    plan = []
    for index, row in enumerate(rows):
        with np.load(dataset.resolve(row), allow_pickle=False) as arrays:
            length = len(arrays["actions"])
        windows = np.arange(0, length, config.execute_steps)
        if len(windows) < 2:
            raise ValueError("validation trajectory needs at least two windows")
        # One draw from each temporal half; no selection by model error.
        steps = [int(rng.choice(part)) for part in np.array_split(windows, 2)]
        plan.append(
            {
                "episode_id": row["episode_id"],
                "task": row["task"],
                "steps": steps,
                "flow_manual_seeds": [
                    args.manual_seed + 2 * index + j for j in range(2)
                ],
            }
        )
    report = {
        "kind": "expanded held-out validation diagnostic, not a test-set result",
        "manual_seed": args.manual_seed,
        "device": device,
        "plan": plan,
        "episodes": len(plan),
        "windows": 2 * len(plan),
        "flow_samples_per_window": config.num_flow_samples,
        "checkpoint_selection": "retain original best; compare without selecting on test data",
        "results": {},
        "complete": False,
    }
    save_report(output / "report.json", report)
    torch.set_num_threads(4)
    set_manual_seed(config.manual_seed)
    backend = NativeVLABackend.from_pretrained(
        config, device="cuda:0", local_path=ROOT / "checkpoints/base/MolmoAct2-LIBERO"
    )
    for parameter in backend.model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    for name in ("best", "last"):
        payload = load_checkpoint(ROOT / f"runs/stage1_s17_gpu7/{name}.pt", backend)
        updates = payload["updates"]
        del payload
        backend.reset()
        backend.train(False)
        torch.cuda.reset_peak_memory_stats()
        started, scores, tasks = time.monotonic(), [], defaultdict(list)
        with (output / f"{name}.jsonl").open("w") as records:
            for index, (row, selected) in enumerate(zip(rows, plan, strict=True)):
                trajectory = Trajectory(
                    dataset.resolve(row), row, total_steps=config.total_steps
                )
                for step, seed in zip(
                    selected["steps"], selected["flow_manual_seeds"], strict=True
                ):
                    end = min(step + config.prediction_steps, trajectory.length)
                    example = TrainingExample(
                        trajectory.context(step, config),
                        trajectory.arrays["actions"][step:end],
                        trajectory.observation(end),
                        None,
                    )
                    set_manual_seed(seed)
                    with (
                        torch.inference_mode(),
                        torch.autocast("cuda", dtype=torch.bfloat16),
                    ):
                        loss, _ = joint_loss(backend, None, example)
                    value = float(loss)
                    scores.append(value)
                    tasks[row["task"]].append(value)
                    records.write(
                        json.dumps(
                            {
                                "episode_id": row["episode_id"],
                                "task": row["task"],
                                "step": step,
                                "manual_seed": seed,
                                "loss": value,
                            }
                        )
                        + "\n"
                    )
                records.flush()
                del example, trajectory
                if (index + 1) % 20 == 0:
                    print(
                        json.dumps(
                            {
                                "checkpoint": name,
                                "episodes_completed": index + 1,
                                "episodes_total": len(rows),
                                "mean_loss": float(np.mean(scores)),
                            }
                        ),
                        flush=True,
                    )
        task_scores = {
            task: {"windows": len(values), "mean_loss": float(np.mean(values))}
            for task, values in tasks.items()
        }
        report["results"][name] = {
            "updates": updates,
            "mean_loss": float(np.mean(scores)),
            "per_task": task_scores,
            "task_macro_mean_loss": float(
                np.mean([v["mean_loss"] for v in task_scores.values()])
            ),
            "wall_seconds": time.monotonic() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        save_report(output / "report.json", report)
        print(json.dumps({"checkpoint": name, **report["results"][name]}), flush=True)
    report["complete"] = True
    save_report(output / "report.json", report)


if __name__ == "__main__":
    main()
