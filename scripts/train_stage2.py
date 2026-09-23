"""Four-GPU stage two: exact stage-one policy transfer and joint future learning."""

import argparse
import json
import math
import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from gpu_devices import verify_cuda_target

from grounded_interaction.predictive_vla.backend import NativeVLABackend
from grounded_interaction.predictive_vla.config import load_config, set_manual_seed
from grounded_interaction.predictive_vla.data import (
    TrainingExample,
    Trajectory,
    TrajectoryDataset,
)
from grounded_interaction.predictive_vla.parallel import (
    assert_replicas_equal,
    average_gradients,
    prediction_scale,
    sharded_examples,
)
from grounded_interaction.predictive_vla.qualification import load_qualified_predictor
from grounded_interaction.predictive_vla.training import (
    CHECKPOINT_SCHEMA,
    adapter_state,
    joint_loss,
    load_checkpoint,
    make_predictor,
    warm_start_policy,
)

ROOT = Path(__file__).resolve().parents[1]


def inside(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("stage-two paths must stay inside the repository")
    return path


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validation_examples(dataset, rank, world, limit):
    rows = sorted(
        (r for r in dataset.entries if r["split"] == "validation"),
        key=lambda r: r["episode_id"],
    )
    rng = random.Random(dataset.config.manual_seed + 100_000)
    for index, row in enumerate(rows[:limit]):
        # Each trajectory has its own local seed, independent of worker count.
        seed = rng.randrange(2**31)
        if index % world != rank:
            continue
        trajectory = Trajectory(
            dataset.resolve(row), row, total_steps=dataset.config.total_steps
        )
        step = random.Random(seed).choice(
            range(0, trajectory.length, dataset.config.execute_steps)
        )
        end = min(step + dataset.config.prediction_steps, trajectory.length)
        yield (
            index,
            row["episode_id"],
            step,
            TrainingExample(
                trajectory.context(step, dataset.config),
                trajectory.arrays["actions"][step:end],
                trajectory.observation(end),
                None,
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/stage2_transition.yaml")
    parser.add_argument("--data", default="data/prepared/stage1_full/manifest.json")
    parser.add_argument(
        "--data-audit", default="data/prepared/stage1_full/final_audit.json"
    )
    parser.add_argument("--model-path", default="checkpoints/base/MolmoAct2-LIBERO")
    parser.add_argument("--warm-start", default="runs/stage1_s17_gpu7/best.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-updates", type=int, default=2000)
    parser.add_argument("--warmup-updates", type=int, default=200)
    parser.add_argument("--prediction-warmup-updates", type=int, default=200)
    parser.add_argument("--predictor-lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--validation-episodes", type=int, default=198)
    parser.add_argument("--resume")
    parser.add_argument("--predictor-init")
    parser.add_argument("--qualification-report")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    requested_config = load_config(inside(args.config))
    if requested_config.predictor_kind in {"transport", "local_transport"} and not (
        args.predictor_init and args.qualification_report
    ):
        raise ValueError(
            "transport must first pass frozen-head qualification; provide --predictor-init and --qualification-report"
        )
    if min(args.max_updates, args.eval_every, args.validation_episodes) < 1:
        raise ValueError("invalid update/evaluation budget")
    if (
        min(args.warmup_updates, args.prediction_warmup_updates) < 0
        or args.predictor_lr <= 0
    ):
        raise ValueError("invalid warmup or predictor learning rate")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4,5,6,7":
        raise RuntimeError("this launch must expose physical GPUs 4,5,6,7")
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 4 or rank != local_rank:
        raise RuntimeError("expected four local workers")
    mapping = verify_cuda_target(
        local_rank + 4, cuda_ordinal=local_rank, visible_count=world
    )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=20), device_id=device)
    torch.set_num_threads(4)
    config = load_config(inside(args.config))
    if config.prediction_weight <= 0 or config.language_weight != 0:
        raise ValueError(
            "stage two requires future supervision and no unlabelled language loss"
        )
    output = inside(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=args.resume is not None)
        save(output / "config.json", config.to_dict())
    dist.barrier()
    dataset = TrajectoryDataset(inside(args.data), config)
    summary = dataset.read_validation_report(inside(args.data_audit))
    if summary["rgbd_episodes"] != summary["episodes"] or summary["answer_annotations"]:
        raise ValueError("unexpected geometry/annotation inventory")
    set_manual_seed(config.manual_seed)
    print(
        json.dumps(
            {"rank": rank, "phase": "loading_model", "physical_gpu": local_rank + 4}
        ),
        flush=True,
    )
    backend = NativeVLABackend.from_pretrained(
        config, device=str(device), local_path=inside(args.model_path)
    )
    groups = {"vlm_lora": [], "context_adapters": [], "action_expert": []}
    for name, parameter in backend.model.named_parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
            key = (
                "action_expert"
                if "action_expert" in name
                else "vlm_lora"
                if "transformer.blocks" in name
                else "context_adapters"
            )
            groups[key].append(parameter)
    initialization = warm_start_policy(inside(args.warm_start), backend)
    if initialization["source_updates"] != 1500:
        raise ValueError("the selected stage-one best must be update 1500")
    predictor = make_predictor(backend).float()
    if config.predictor_kind in {"transport", "local_transport"}:
        initialization["predictor"] = load_qualified_predictor(
            inside(args.predictor_init),
            inside(args.qualification_report),
            predictor,
            inside(args.warm_start),
            config,
            policy_metadata=initialization,
        )
    groups["predictor"] = list(predictor.parameters())
    parameters = [p for group in groups.values() for p in group]
    rates = [
        config.vlm_learning_rate,
        config.learning_rate,
        config.action_expert_learning_rate,
        args.predictor_lr,
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": group, "lr": lr}
            for group, lr in zip(groups.values(), rates, strict=True)
        ],
        betas=(0.9, 0.95),
        eps=1e-6,
        weight_decay=config.weight_decay,
    )

    def lr_scale(step):
        if step < args.warmup_updates:
            return (step + 1) / args.warmup_updates
        progress = min(
            1.0,
            (step - args.warmup_updates)
            / max(1, args.max_updates - args.warmup_updates),
        )
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    plan = {
        key: getattr(args, key)
        for key in (
            "max_updates",
            "warmup_updates",
            "prediction_warmup_updates",
            "predictor_lr",
            "eval_every",
            "validation_episodes",
            "preflight",
        )
    } | {
        "world_size": world,
        "data": str(inside(args.data)),
        "warm_start": str(inside(args.warm_start)),
    }
    updates, epoch, offset, best = 0, 0, 0, math.inf
    set_manual_seed(config.manual_seed + 1000 * rank)
    if args.resume:
        payload = load_checkpoint(inside(args.resume), backend, predictor)
        if payload["run_plan"] != plan or payload.get("predictor") is None:
            raise ValueError("resume schedule, data or predictor differs")
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        updates, epoch, offset, best = (
            payload["updates"],
            payload["epoch"],
            payload["next_example"],
            payload["best"],
        )
        rng = payload["rank_rng"][rank]
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"], device)
        random.setstate(rng["python"])
        del payload
    assert_replicas_equal(parameters)
    metadata = {
        "manual_seed": config.manual_seed,
        "rank_noise_seed_rule": "manual_seed + 1000 * rank",
        "dataset": summary,
        "initialization": initialization,
        "world_size": world,
        "physical_gpus": [4, 5, 6, 7],
        "microbatch_per_rank": 1,
        "accumulation_per_rank": config.accumulation,
        "effective_batch": config.accumulation * world,
        "parallel_method": "synchronous summed-gradient all-reduce before global averaging, clipping and Adam",
        "parameters": {
            name: sum(p.numel() for p in group) for name, group in groups.items()
        },
        "trainable_dtype": "float32",
        "autocast_dtype": config.dtype,
        "validation": "one fixed window per validation trajectory, no test trajectories",
        "validation_score": "action (qualified transport)"
        if config.predictor_kind in {"transport", "local_transport"}
        else "action + final_prediction_weight * prediction; fixed across ramp",
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "run_plan": plan,
        "pid_by_rank": [None] * world,
        "device_by_rank": [None] * world,
        "epoch_sharding": "same seeded global window stream, blocks of four; discard final incomplete block (one window for this dataset)",
    }
    dist.all_gather_object(metadata["pid_by_rank"], os.getpid())
    dist.all_gather_object(metadata["device_by_rank"], mapping)
    if rank == 0:
        save(output / "run_metadata.json", metadata)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)

    def log(value):
        if rank == 0:
            value = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **value}
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(value, allow_nan=False) + "\n")
            print(json.dumps(value, allow_nan=False), flush=True)
            save(output / "status.json", value)

    def journal(message):
        if rank == 0:
            timestamp = datetime.now(timezone(timedelta(hours=8))).isoformat(
                timespec="seconds"
            )
            with (ROOT / "docs/experiment_log.md").open("a") as stream:
                stream.write(f"\n{timestamp}：阶段二（{output.name}）{message}\n")

    def evaluate():
        backend.train(False)
        predictor.eval()
        sums = torch.zeros(3, device=device, dtype=torch.float64)
        selections = []
        with torch.random.fork_rng(devices=[local_rank]), torch.no_grad():
            for index, identity, step, example in validation_examples(
                dataset, rank, world, args.validation_episodes
            ):
                torch.manual_seed(config.manual_seed + 100_000 + index)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, parts = joint_loss(backend, predictor, example)
                sums += sums.new_tensor([parts["action"], parts["prediction"], 1])
                selections.append({"episode_id": identity, "step": step})
        dist.all_reduce(sums)
        all_selections = [None] * world
        dist.all_gather_object(all_selections, selections)
        if rank == 0 and not (output / "validation_windows.json").exists():
            save(
                output / "validation_windows.json",
                [row for shard in all_selections for row in shard],
            )
        backend.train(True)
        predictor.train()
        action, prediction = (sums[:2] / sums[2]).tolist()
        score = (
            action
            if config.predictor_kind in {"transport", "local_transport"}
            else action + config.prediction_weight * prediction
        )
        return score, {
            "action": action,
            "prediction": prediction,
            "windows": int(sums[2]),
        }

    def checkpoint(improved):
        rank_rng = [None] * world
        dist.all_gather_object(
            rank_rng,
            {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(device),
                "python": random.getstate(),
            },
        )
        if rank == 0:
            payload = {
                "schema": CHECKPOINT_SCHEMA,
                "config": config.to_dict(),
                "manual_seed": config.manual_seed,
                "updates": updates,
                "epoch": epoch,
                "next_example": offset,
                "best": best,
                "run_plan": plan,
                "initialization": initialization,
                "rank_rng": rank_rng,
                "adapters": adapter_state(backend),
                "predictor": {
                    k: v.detach().cpu() for k, v in predictor.state_dict().items()
                },
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            temporary = output / "last.pt.tmp"
            torch.save(payload, temporary)
            temporary.replace(output / "last.pt")
            if improved:
                # Preserve independent best/last paths without serializing twice.
                import shutil

                shutil.copyfile(output / "last.pt", output / "best.pt.tmp")
                (output / "best.pt.tmp").replace(output / "best.pt")
            journal(
                f"已完成 {updates}/{args.max_updates} 更新，验证目标={best:.6f}（目前最佳）；检查点已保存。"
            )
        dist.barrier()

    log({"update": updates, "initialization": initialization, "replicas_equal": True})
    if not args.resume:
        score, parts = evaluate()
        log(
            {
                "update": 0,
                "validation_loss": score,
                "validation_losses": parts,
                "baseline_before_updates": True,
            }
        )
        journal(
            f"启动，继承阶段一 best 第1500步；GPU 4/5/6/7，有效 batch={world * config.accumulation}，目标 {args.max_updates} 更新。"
        )
    backend.train(True)
    predictor.train()
    optimizer.zero_grad(set_to_none=True)
    while updates < args.max_updates:
        iterator = sharded_examples(
            dataset.examples("train", seed=config.manual_seed + epoch), rank, world
        )
        iterator = iter(iterator)
        for _ in range(offset):
            next(iterator)
        exhausted = False
        while updates < args.max_updates:
            begin, count = time.monotonic(), 0
            totals = torch.zeros(3, device=device, dtype=torch.float64)
            weight = prediction_scale(
                updates, args.prediction_warmup_updates, config.prediction_weight
            )
            for _ in range(config.accumulation):
                example = next(iterator, None)
                if example is None:
                    exhausted = True
                    break
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, parts = joint_loss(
                        backend, predictor, example, prediction_weight=weight
                    )
                if loss is None:
                    raise ValueError("unsupervised window in stage-two data")
                loss.backward()  # summed gradients; divide once by global sample count
                totals += totals.new_tensor(
                    [float(loss.detach()), parts["action"], parts["prediction"]]
                )
                count += 1
                offset += 1
                del example, loss
            if not count:
                break
            global_count = average_gradients(parameters, count)
            if updates <= 1:
                grad_report = {
                    name: sum(
                        p.grad is not None and bool(p.grad.count_nonzero())
                        for p in group
                    )
                    for name, group in groups.items()
                }
                frozen_gradients = sum(
                    p.grad is not None
                    for p in backend.model.parameters()
                    if not p.requires_grad
                )
                if frozen_gradients or not all(
                    grad_report[name]
                    for name in ("vlm_lora", "context_adapters", "action_expert")
                ):
                    raise RuntimeError(
                        "frozen/trainable policy gradient contract failed"
                    )
                if weight > 0 and not grad_report["predictor"]:
                    raise RuntimeError("future predictor received no gradient")
                log(
                    {
                        "update": updates,
                        "gradient_check": grad_report,
                        "frozen_parameters_with_grad": frozen_gradients,
                    }
                )
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, config.grad_clip, error_if_nonfinite=True
            )
            applied_lr = [g["lr"] for g in optimizer.param_groups]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            dist.all_reduce(totals)
            timing = torch.tensor(
                [
                    time.monotonic() - begin,
                    torch.cuda.max_memory_allocated(device) / 2**30,
                    torch.cuda.max_memory_reserved(device) / 2**30,
                ],
                device=device,
            )
            dist.all_reduce(timing, op=dist.ReduceOp.MAX)
            log(
                {
                    "update": updates,
                    "epoch": epoch,
                    "global_batch": global_count,
                    "loss": float(totals[0] / global_count),
                    "losses": dict(
                        zip(
                            ("action", "prediction"),
                            (totals[1:] / global_count).tolist(),
                            strict=True,
                        )
                    ),
                    "prediction_weight": weight,
                    "grad_norm": float(norm),
                    "lr": applied_lr,
                    "seconds_per_update": float(timing[0]),
                    "peak_allocated_gib": float(timing[1]),
                    "peak_reserved_gib": float(timing[2]),
                }
            )
            if (
                updates in (1, 2)
                or updates % args.eval_every == 0
                or updates == args.max_updates
            ):
                assert_replicas_equal(parameters)
                score, parts = evaluate()
                improved = score < best
                best = min(best, score)
                log(
                    {
                        "update": updates,
                        "validation_loss": score,
                        "validation_losses": parts,
                        "replicas_equal": True,
                    }
                )
                checkpoint(improved)
            if exhausted:
                break
        if exhausted:
            epoch, offset = epoch + 1, 0
    log(
        {
            "update": updates,
            "training_complete": True,
            "wall_seconds": time.monotonic() - started,
        }
    )
    if args.preflight and rank == 0:
        save(
            output / "preflight.json",
            {
                "passed": True,
                "manual_seed": config.manual_seed,
                "world_size": world,
                "initialization": initialization,
                "config": config.to_dict(),
                "run_plan": plan,
                "checks": [
                    "exact policy transfer",
                    "real four-GPU forward/backward",
                    "gradient averaging",
                    "replica equality after Adam",
                    "frozen weights have no gradients",
                    "predictor gradient at nonzero weight",
                    "held-out validation",
                    "checkpoint serialization",
                ],
            },
        )
    journal(f"本次更新预算完成，耗时 {(time.monotonic() - started) / 3600:.3f} 小时。")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
