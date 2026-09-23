"""Joint flow/action-conditioned prediction/language training on real trajectories."""

import json
import math
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

from .config import VLAConfig, set_manual_seed
from .model import ActionConditionedPredictor, patch_prediction_loss
from .transport import LocalTransportPredictor, TransportPredictor

CHECKPOINT_SCHEMA = "predictive-vla-adapters-v1"


def make_predictor(backend):
    c = backend.config
    if c.predictor_kind in {"transport", "local_transport"}:
        model_class = (
            LocalTransportPredictor
            if c.predictor_kind == "local_transport"
            else TransportPredictor
        )
        return model_class(
            backend.capabilities.hidden_dim,
            width=c.predictor_dim,
            time_scale=c.total_steps,
        ).to(backend.device)
    return ActionConditionedPredictor(
        backend.capabilities.hidden_dim,
        width=c.predictor_dim,
        heads=c.predictor_heads,
        layers=c.predictor_layers,
        time_scale=c.total_steps,
    ).to(backend.device)


def joint_loss(backend, predictor, example, *, prediction_weight=None):
    """Future actions and frames never enter encode/flow/language policy inputs."""
    losses = {}
    if len(example.actual_future_actions):
        shared = backend.encode(example.context)
        losses["action"] = backend.flow_loss(shared, example.actual_future_actions)
        if backend.config.prediction_weight:
            # The teacher encodes a separate single future observation. No
            # bidirectional video encoding is sliced into apparent past/future.
            target, positions, valid = backend.target(
                example.future_observation, example.context.task
            )
            actions = backend.normalize_actions(example.actual_future_actions)[None]
            extra = {}
            if backend.config.predictor_kind in {"transport", "local_transport"}:
                current, cp, cv = backend.target(
                    example.context.current, example.context.task
                )
                if not torch.equal(positions, cp):
                    raise ValueError(
                        "transport requires matching current/future native grids"
                    )
                extra = {
                    "current": current,
                    "current_valid": cv,
                    "state": torch.as_tensor(
                        example.context.current.state, device=actions.device
                    )[None],
                }
            prediction = predictor(
                shared.hidden,
                torch.ones_like(shared.ids, dtype=torch.bool),
                actions,
                torch.ones(actions.shape[:2], device=actions.device, dtype=torch.bool),
                positions,
                torch.tensor([actions.shape[1]], device=actions.device),
                **extra,
            )
            losses["prediction"] = patch_prediction_loss(prediction, target, valid)
    if example.response is not None and backend.config.language_weight:
        losses["language"] = backend.language_loss(example.context, example.response)
    if not losses:
        return None, {}
    weights = {
        "action": 1.0,
        "prediction": backend.config.prediction_weight
        if prediction_weight is None
        else prediction_weight,
        "language": backend.config.language_weight,
    }
    total = sum(weights[name] * loss for name, loss in losses.items())
    if not torch.isfinite(total):
        raise RuntimeError("non-finite joint training loss")
    return total, {name: float(loss.detach()) for name, loss in losses.items()}


def adapter_state(backend):
    names = {name for name, p in backend.model.named_parameters() if p.requires_grad}
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in backend.model.state_dict().items()
        if name in names or "predictive_vla_adapters" in name
    }


def read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            "checkpoint is not from the current geometry/history VLA route"
        )
    config = VLAConfig(**payload["config"])
    if config.manual_seed != payload["manual_seed"]:
        raise ValueError("checkpoint manual_seed differs from its configuration")
    return payload, config


def load_checkpoint(path, backend, predictor=None):
    payload, config = read_checkpoint(path)
    if config != backend.config:
        raise ValueError("checkpoint and runtime configurations differ")
    expected = adapter_state(backend)
    if set(payload["adapters"]) != set(expected):
        raise ValueError("checkpoint adapters do not match the native execution path")
    for name, tensor in payload["adapters"].items():
        if tensor.shape != expected[name].shape or not torch.isfinite(tensor).all():
            raise ValueError(f"invalid adapter tensor {name}")
    saved_predictor = payload.get("predictor")
    if saved_predictor is not None and (
        not isinstance(saved_predictor, dict)
        or any(
            not torch.is_tensor(t) or not torch.isfinite(t).all()
            for t in saved_predictor.values()
        )
    ):
        raise ValueError("invalid or nonfinite checkpoint predictor")
    if bool(config.prediction_weight) != bool(saved_predictor):
        raise ValueError("checkpoint predictor is inconsistent with prediction_weight")
    if predictor is not None:
        if saved_predictor is None:
            raise ValueError(
                "cannot restore a predictor from an action-only checkpoint"
            )
        expected_predictor = predictor.state_dict()
        if set(saved_predictor) != set(expected_predictor) or any(
            t.shape != expected_predictor[n].shape for n, t in saved_predictor.items()
        ):
            raise ValueError("checkpoint predictor does not match the requested module")
    backend.model.load_state_dict(payload["adapters"], strict=False)
    if predictor is not None:
        predictor.load_state_dict(saved_predictor, strict=True)
    return payload


def warm_start_policy(path, backend):
    """Transfer policy weights between stages; never transfer optimizer or RNG.

    Training and auxiliary-predictor settings may change. All policy architecture and
    observation/action alignment settings must match the source checkpoint.
    Promote trainable parameters to FP32 before calling to preserve exact values.
    """
    payload, source = read_checkpoint(path)
    allowed = {
        "prediction_weight",
        "predictor_kind",
        "predictor_dim",
        "predictor_heads",
        "predictor_layers",
        "language_weight",
        "learning_rate",
        "vlm_learning_rate",
        "action_expert_learning_rate",
        "weight_decay",
        "grad_clip",
        "accumulation",
    }
    changed = {
        key
        for key, value in source.to_dict().items()
        if value != backend.config.to_dict()[key]
    }
    if changed - allowed:
        raise ValueError(
            f"warm-start architecture differs: {sorted(changed - allowed)}"
        )
    if source.prediction_weight != 0 or payload.get("predictor") is not None:
        raise ValueError(
            "stage two must warm-start from an action-only stage-one checkpoint"
        )
    current = backend.model.state_dict()
    expected = {
        name
        for name, parameter in backend.model.named_parameters()
        if parameter.requires_grad
    } | {name for name in current if "predictive_vla_adapters" in name}
    if set(payload["adapters"]) != expected:
        raise ValueError("warm-start policy tensor names differ")
    for name, tensor in payload["adapters"].items():
        if current[name].shape != tensor.shape or not torch.isfinite(tensor).all():
            raise ValueError(f"invalid warm-start tensor: {name}")
        if current[name].dtype != tensor.dtype:
            raise ValueError(f"warm-start dtype would lose precision: {name}")
    backend.model.load_state_dict(payload["adapters"], strict=False)
    for name, tensor in payload["adapters"].items():
        if not torch.equal(backend.model.state_dict()[name].cpu(), tensor):
            raise RuntimeError(f"warm-start exact restoration failed: {name}")
    return {
        "source": str(Path(path).resolve()),
        "source_updates": payload["updates"],
        "source_manual_seed": source.manual_seed,
        "source_config": source.to_dict(),
        "restored_tensors": len(expected),
        "exact_restoration": True,
        "changed_training_fields": sorted(changed),
        "optimizer_restored": False,
        "predictor_restored": False,
    }


def train(
    backend,
    dataset,
    *,
    output,
    epochs=1,
    max_updates=None,
    warmup_updates=200,
    eval_every=100,
    validation_examples=64,
    resume=None,
    experiment_log=None,
    data_summary=None,
):
    c = backend.config
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be positive")
    summary = dataset.validate() if data_summary is None else data_summary
    if not {"train", "validation"}.issubset(summary["splits"]):
        raise ValueError("training requires disjoint train and validation episodes")
    if c.geometry and summary["rgbd_episodes"] != summary["episodes"]:
        raise ValueError(
            "geometry training requires calibrated RGB-D for every episode"
        )
    set_manual_seed(c.manual_seed)
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("max_updates must be positive")
    if eval_every < 1 or validation_examples < 1 or warmup_updates < 0:
        raise ValueError("invalid evaluation/warmup settings")
    predictor = make_predictor(backend) if c.prediction_weight else None
    vlm, added, expert = [], [], []
    for name, p in backend.model.named_parameters():
        if p.requires_grad:
            group = (
                expert
                if "action_expert" in name
                else vlm
                if "transformer.blocks" in name
                else added
            )
            group.append(p)
    if predictor is not None:
        added.extend(predictor.parameters())
    parameters = vlm + added + expert
    # Keep trainable weights and Adam moments in FP32. Autocast uses BF16 for
    # model arithmetic while avoiding small Adam updates rounding to zero.
    for parameter in parameters:
        parameter.data = parameter.data.float()
    run_plan = {
        "max_updates": max_updates,
        "warmup_updates": warmup_updates,
        "validation_examples": validation_examples,
        "eval_every": eval_every,
    }
    optimizer = torch.optim.AdamW(
        [
            {"params": vlm, "lr": c.vlm_learning_rate},
            {"params": added, "lr": c.learning_rate},
            {"params": expert, "lr": c.action_expert_learning_rate},
        ],
        weight_decay=c.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-6,
    )
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=resume is not None)
    (destination / "config.json").write_text(json.dumps(c.to_dict(), indent=2) + "\n")
    best, updates, first_epoch, first_offset = math.inf, 0, 0, 0
    started = time.monotonic()

    def lr_scale(step):
        if warmup_updates and step < warmup_updates:
            return (step + 1) / warmup_updates
        if max_updates is None:
            return 1.0
        progress = min(
            1.0, (step - warmup_updates) / max(1, max_updates - warmup_updates)
        )
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * max(0.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    if resume is not None:
        payload = load_checkpoint(resume, backend, predictor)
        if payload.get("run_plan") != run_plan:
            raise ValueError("resume training schedule differs from checkpoint")
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        updates, first_epoch, first_offset = (
            payload["updates"],
            payload["epoch"],
            payload["next_example"],
        )
        best = payload["best"]
        torch.set_rng_state(payload["torch_rng"])
        random.setstate(payload["python_rng"])
        if backend.device.type == "cuda":
            torch.cuda.set_rng_state(payload["cuda_rng"], backend.device)
    metadata = {
        "manual_seed": c.manual_seed,
        "dataset": summary,
        "trainable_parameters": {
            "vlm_lora": sum(p.numel() for p in vlm),
            "new_modules": sum(p.numel() for p in added),
            "action_expert": sum(p.numel() for p in expert),
        },
        "microbatch": 1,
        "gradient_accumulation": c.accumulation,
        "effective_batch": c.accumulation,
        "world_size": 1,
        "flow_samples_per_chunk": c.num_flow_samples,
        "max_updates": max_updates,
        "warmup_updates": warmup_updates,
        "validation_examples": validation_examples,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "trainable_parameter_dtype": "float32",
        "optimizer_state_dtype": "float32",
    }
    if backend.device.type == "cuda":
        metadata["gpu"] = torch.cuda.get_device_name(backend.device)
        torch.cuda.reset_peak_memory_stats(backend.device)
    (destination / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    latest = {}
    context_lengths = []
    collect_context = False

    def track_context(_module, _args, kwargs):
        embeddings = kwargs.get("inputs_embeds")
        if collect_context and embeddings is not None:
            context_lengths.append(int(embeddings.shape[1]))

    context_hook = (
        backend.backbone.register_forward_pre_hook(track_context, with_kwargs=True)
        if hasattr(backend, "backbone")
        else None
    )

    def journal(message):
        if experiment_log is not None:
            timestamp = datetime.now(timezone(timedelta(hours=8))).isoformat(
                timespec="seconds"
            )
            with Path(experiment_log).open("a") as stream:
                stream.write(f"\n{timestamp}：{message}\n")

    def log(value):
        if "seconds_per_update" in value and context_lengths:
            ordered = sorted(context_lengths)
            value["context_tokens"] = {
                "min": ordered[0],
                "median": ordered[len(ordered) // 2],
                "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
                "max": ordered[-1],
            }
            context_lengths.clear()
        value = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **value,
        }
        with (destination / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
        print(json.dumps(value, allow_nan=False), flush=True)
        latest.update(value)
        lines = [
            "**阶段一实时训练记录**",
            "",
            f"最后更新：{latest['timestamp_utc']}。",
            f"优化器更新：{latest.get('update', updates)} / {max_updates or '按 epoch'}；有效 batch 上限 {c.accumulation}。",
            f"训练 loss：{latest.get('loss', '尚未记录')}；验证 loss：{latest.get('validation_loss', '尚未记录')}。",
            f"最近一次更新耗时：{latest.get('seconds_per_update', '尚未记录')} 秒。",
            f"显存峰值 allocated / reserved：{latest.get('peak_allocated_gib', '尚未记录')} / {latest.get('peak_reserved_gib', '尚未记录')} GiB。",
            "",
            "完整数值见 [metrics.jsonl](metrics.jsonl)，恢复点见 [last.pt](last.pt)。",
            "已达到本次更新预算。"
            if value.get("training_complete")
            else "以上是最近一次写盘记录；进程状态需结合控制台日志确认。",
        ]
        temporary = destination / "progress.md.tmp"
        temporary.write_text("\n\n".join(lines) + "\n")
        temporary.replace(destination / "progress.md")

    def checkpoint(name, epoch, next_example):
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "config": c.to_dict(),
            "manual_seed": c.manual_seed,
            "epoch": epoch,
            "next_example": next_example,
            "updates": updates,
            "run_plan": run_plan,
            "best": best,
            "adapters": adapter_state(backend),
            "predictor": None if predictor is None else predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state(backend.device)
            if backend.device.type == "cuda"
            else None,
        }
        temporary = destination / (name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(destination / name)
        if name == "last.pt":
            journal(
                f"阶段一已完成 {updates} 次更新；最近训练 loss={latest.get('loss')}，"
                f"验证 loss={latest.get('validation_loss')}，已保存 {destination / name}。"
                f"训练循环累计用时 {(time.monotonic() - started) / 3600:.3f} 小时。"
            )

    def step(count):
        # Correct the last partial accumulation, without dropping real samples.
        if count != c.accumulation:
            for p in parameters:
                if p.grad is not None:
                    p.grad.mul_(c.accumulation / count)
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, c.grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return float(norm)

    def evaluate():
        nonlocal collect_context
        previous_collection, collect_context = collect_context, False
        backend.train(False)
        if predictor is not None:
            predictor.eval()
        values, metrics = [], {}
        devices = [backend.device.index or 0] if backend.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(c.manual_seed)
            for index, example in enumerate(
                dataset.examples("validation", seed=c.manual_seed)
            ):
                if index >= validation_examples:
                    break
                with torch.autocast(
                    backend.device.type,
                    dtype=torch.bfloat16,
                    enabled=backend.device.type == "cuda" and c.dtype == "bfloat16",
                ):
                    loss, parts = joint_loss(backend, predictor, example)
                if loss is not None:
                    values.append(float(loss))
                    for key, value in parts.items():
                        metrics.setdefault(key, []).append(value)
        if not values:
            raise ValueError("validation split yielded no supervised samples")
        backend.train(True)
        if predictor is not None:
            predictor.train()
        collect_context = previous_collection
        return sum(values) / len(values), {
            k: sum(v) / len(v) for k, v in metrics.items()
        }

    if resume is None:
        journal(
            f"阶段一训练进程启动，输出 {destination}，目标 {max_updates} 次更新，"
            f"有效 batch={c.accumulation}，manual_seed={c.manual_seed}。"
        )
        score, metrics = evaluate()
        log(
            {
                "update": 0,
                "validation_loss": score,
                "validation_losses": metrics,
                "baseline_before_updates": True,
            }
        )
    epoch = first_epoch
    while (updates < max_updates) if max_updates is not None else (epoch < epochs):
        collect_context = True
        backend.train(True)
        if predictor is not None:
            predictor.train()
        optimizer.zero_grad(set_to_none=True)
        pending, values, parts_sum, update_start = 0, [], {}, time.monotonic()
        yielded = 0
        for index, example in enumerate(
            dataset.examples("train", seed=c.manual_seed + epoch)
        ):
            if epoch == first_epoch and index < first_offset:
                continue
            yielded += 1
            with torch.autocast(
                backend.device.type,
                dtype=torch.bfloat16,
                enabled=backend.device.type == "cuda" and c.dtype == "bfloat16",
            ):
                loss, parts = joint_loss(backend, predictor, example)
            if loss is None:
                continue
            (loss / c.accumulation).backward()
            pending += 1
            values.append(float(loss.detach()))
            for key, value in parts.items():
                parts_sum[key] = parts_sum.get(key, 0.0) + value
            if pending == c.accumulation:
                if updates == 0:
                    gradient_report = {}
                    for name, group in (
                        ("vlm_lora", vlm),
                        ("added", added),
                        ("action_expert", expert),
                    ):
                        gradient_report[name] = {
                            "parameters_with_grad": sum(
                                p.grad is not None for p in group
                            ),
                            "parameters_with_nonzero_grad": sum(
                                p.grad is not None and bool(p.grad.count_nonzero())
                                for p in group
                            ),
                        }
                    log({"update": 0, "gradient_check": gradient_report})
                applied_lr = [g["lr"] for g in optimizer.param_groups]
                grad_norm = step(pending)
                updates += 1
                elapsed = time.monotonic() - update_start
                row = {
                    "update": updates,
                    "epoch": epoch,
                    "seconds_per_update": elapsed,
                    "loss": sum(values[-pending:]) / pending,
                    "grad_norm": grad_norm,
                    "lr": applied_lr,
                    "next_lr": [g["lr"] for g in optimizer.param_groups],
                    "losses": {k: v / pending for k, v in parts_sum.items()},
                }
                if backend.device.type == "cuda":
                    row["peak_allocated_gib"] = (
                        torch.cuda.max_memory_allocated(backend.device) / 2**30
                    )
                    row["peak_reserved_gib"] = (
                        torch.cuda.max_memory_reserved(backend.device) / 2**30
                    )
                log(row)
                pending = 0
                parts_sum = {}
                if updates == 1 or updates % eval_every == 0 or updates == max_updates:
                    score, metrics = evaluate()
                    log(
                        {
                            "update": updates,
                            "validation_loss": score,
                            "validation_losses": metrics,
                        }
                    )
                    if score < best:
                        best = score
                        checkpoint("best.pt", epoch, index + 1)
                    checkpoint("last.pt", epoch, index + 1)
                update_start = time.monotonic()
                if updates == max_updates:
                    break
        if pending:
            # At an epoch boundary the true last partial batch has its own
            # normalization; a checkpoint is saved only after this update.
            applied_lr = [g["lr"] for g in optimizer.param_groups]
            grad_norm = step(pending)
            updates += 1
            row = {
                "update": updates,
                "epoch": epoch,
                "partial_batch": pending,
                "seconds_per_update": time.monotonic() - update_start,
                "loss": sum(values[-pending:]) / pending,
                "grad_norm": grad_norm,
                "lr": applied_lr,
                "next_lr": [g["lr"] for g in optimizer.param_groups],
                "losses": {k: v / pending for k, v in parts_sum.items()},
            }
            if backend.device.type == "cuda":
                row["peak_allocated_gib"] = (
                    torch.cuda.max_memory_allocated(backend.device) / 2**30
                )
                row["peak_reserved_gib"] = (
                    torch.cuda.max_memory_reserved(backend.device) / 2**30
                )
            log(row)
        if not values:
            if first_offset and yielded == 0:
                epoch += 1
                continue
            raise ValueError("training split yielded no supervised samples")
        if pending or max_updates is None:
            score, metrics = evaluate()
            log(
                {
                    "update": updates,
                    "validation_loss": score,
                    "validation_losses": metrics,
                }
            )
            if score < best:
                best = score
                checkpoint("best.pt", epoch + 1, 0)
            checkpoint("last.pt", epoch + 1, 0)
        epoch += 1
    log(
        {
            "update": updates,
            "training_complete": True,
            "wall_seconds": time.monotonic() - started,
        }
    )
    journal(
        f"阶段一训练达到本次预算：{updates} 次更新；最佳验证 loss={best}。"
        "这表示优化过程完成，闭环性能仍需单独评测。"
    )
    if context_hook is not None:
        context_hook.remove()
    return {
        "epochs": epoch,
        "updates": updates,
        "wall_seconds": time.monotonic() - started,
        "best_validation_loss": best,
        "dataset": summary,
        "checkpoint": str(destination / "best.pt"),
        "robot_evaluated": False,
    }
