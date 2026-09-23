"""Real pretrained-model checks and a bounded four-window overfit probe on GPU 7."""

import argparse
import json
import os
import resource
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["HF_MODULES_CACHE"] = str(ROOT / ".cache/huggingface/modules")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch

from grounded_interaction.predictive_vla.backend import NativeVLABackend
from grounded_interaction.predictive_vla.config import load_config
from grounded_interaction.predictive_vla.data import Trajectory
from grounded_interaction.predictive_vla.training import (
    CHECKPOINT_SCHEMA,
    adapter_state,
    joint_loss,
    load_checkpoint,
)


def save_report(report):
    p = ROOT / "data/preparation/preflight.json"
    tmp = p.with_suffix(".json.partial")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    tmp.replace(p)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", default="data/prepared/audit256")
    parser.add_argument(
        "--wait-assets",
        type=int,
        default=0,
        help="Maximum seconds to wait for verified model files",
    )
    parser.add_argument("--updates", type=int, default=8)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
        raise RuntimeError("preflight must be restricted to physical GPU 7")
    if not 1 <= args.updates <= 32:
        raise ValueError("preflight update budget must be 1..32")
    torch.set_num_threads(4)
    cfg = load_config(ROOT / "experiments/stage1_policy.yaml")
    report = {
        "passed": False,
        "phase": "assets",
        "manual_seed": cfg.manual_seed,
        "gpu_physical": 7,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": cfg.model_id,
        "overfit_updates": args.updates,
    }
    save_report(report)
    deadline = time.monotonic() + args.wait_assets
    rows = json.loads((ROOT / "experiments/stage1_assets.json").read_text())["files"]
    rows = [r for r in rows if r["source"] == cfg.model_id]
    while not all((ROOT / r["path"]).is_file() for r in rows):
        if time.monotonic() >= deadline:
            raise RuntimeError("pretrained model files are incomplete")
        print("Waiting for complete pretrained model files", flush=True)
        time.sleep(30)
    for row in rows:
        path = ROOT / row["path"]
        if path.stat().st_size != row["size"]:
            raise ValueError("model asset verification failed")
    source = (ROOT / args.audit_dir).resolve()
    if not source.is_relative_to(ROOT):
        raise ValueError("audit directory escapes repository")
    provenance = json.loads((source / "latest_audit.json").read_text())
    if not provenance["passed"] or provenance["image_size"] != 256:
        raise ValueError("256-pixel replay/calibration audit must pass first")
    episode = (
        source
        / provenance["source"].split("data/libero_raw/")[1].replace(".hdf5", "")
        / (provenance["episode"] + ".npz")
    )
    trajectory = Trajectory(
        episode, {"task": provenance["task"]}, total_steps=cfg.total_steps
    )
    windows = list(trajectory.examples(cfg))[2:6]
    if len(windows) != 4:
        raise ValueError("probe needs four real windows with full history")
    report["episode"] = str(episode.relative_to(ROOT))
    report["phase"] = "loading_model"
    save_report(report)
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_path = ROOT / "checkpoints/base/MolmoAct2-LIBERO"
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        extra_special_tokens={},
    )
    torch.manual_seed(cfg.manual_seed)
    started = time.monotonic()
    model = (
        AutoModelForImageTextToText.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .to("cuda:0")
        .eval()
    )
    report["load_seconds"] = time.monotonic() - started
    report["gpu_name"] = torch.cuda.get_device_name(0)
    report["phase"] = "native_bypass"
    save_report(report)
    context = trajectory.context(0, cfg)
    with torch.no_grad():
        original = model.predict_action(
            processor=processor,
            images=[c.rgb for c in context.current.cameras],
            task=context.task,
            state=context.current.state,
            norm_tag="libero",
            inference_action_mode="continuous",
            enable_depth_reasoning=False,
            num_steps=cfg.flow_steps,
            generator=torch.Generator(device="cuda:0").manual_seed(cfg.manual_seed),
            normalize_language=True,
            enable_cuda_graph=False,
        ).actions
    if torch.is_tensor(original):
        original = original.float().cpu().numpy()
    original = np.asarray(original).reshape(-1, 7)[: cfg.prediction_steps]
    backend = NativeVLABackend(model=model, processor=processor, config=cfg)
    with backend.native_mode():
        bypass = backend.act(context, seed=cfg.manual_seed, steps=cfg.prediction_steps)
    report["native_bypass_max_abs_error"] = float(np.max(np.abs(original - bypass)))
    if not np.allclose(original, bypass, atol=1e-5, rtol=1e-5):
        raise RuntimeError("native bypass changed pretrained actions")
    groups = {"vlm": [], "new": [], "expert": []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
            groups[
                "expert"
                if "action_expert" in name
                else "vlm"
                if "transformer.blocks" in name
                else "new"
            ].append(parameter)
    parameters = [p for group in groups.values() for p in group]
    optimizer = torch.optim.AdamW(
        parameters, lr=5e-5, betas=(0.9, 0.95), eps=1e-6, weight_decay=0
    )
    report["trainable_parameters"] = {
        k: sum(p.numel() for p in g) for k, g in groups.items()
    }
    report["phase"] = "forward_backward"
    save_report(report)
    torch.cuda.reset_peak_memory_stats()

    def measure_loss():
        backend.train(False)
        with (
            torch.random.fork_rng(devices=[0]),
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            torch.manual_seed(cfg.manual_seed)
            losses = [
                float(joint_loss(backend, None, example)[0]) for example in windows
            ]
        return float(np.mean(losses))

    report["fixed_noise_loss_before"] = measure_loss()
    report["updates"] = []
    for update in range(args.updates):
        started = time.monotonic()
        backend.train(True)
        optimizer.zero_grad(set_to_none=True)
        loss_values = []
        for example in windows:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = joint_loss(backend, None, example)
            (loss / len(windows)).backward()
            loss_values.append(float(loss.detach()))
        if update == 0:
            report["gradient_norm_by_group"] = {
                k: float(
                    torch.linalg.vector_norm(
                        torch.stack(
                            [p.grad.float().norm() for p in group if p.grad is not None]
                        )
                    )
                )
                for k, group in groups.items()
            }
            if any(
                v <= 0 or not np.isfinite(v)
                for v in report["gradient_norm_by_group"].values()
            ):
                raise RuntimeError(
                    "flow gradients did not reach every intended parameter group"
                )
            if any(
                p.grad is not None for p in model.parameters() if not p.requires_grad
            ):
                raise RuntimeError(
                    "frozen backbone/teacher parameters received gradients"
                )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        torch.cuda.synchronize()
        row = {
            "update": update + 1,
            "loss": float(np.mean(loss_values)),
            "grad_norm": float(grad_norm),
            "seconds": time.monotonic() - started,
            "microbatches": len(windows),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        report["updates"].append(row)
        save_report(report)
        print(json.dumps(row, allow_nan=False), flush=True)
    report["fixed_noise_loss_after"] = measure_loss()
    report["phase"] = "checkpoint_roundtrip"
    save_report(report)
    checkpoint = ROOT / "checkpoints/stage1_preflight.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "config": cfg.to_dict(),
        "manual_seed": cfg.manual_seed,
        "adapters": adapter_state(backend),
        "predictor": None,
        "optimizer": optimizer.state_dict(),
    }
    expected = parameters[0].detach().clone()
    torch.save(payload, checkpoint)
    del payload
    with torch.no_grad():
        parameters[0].zero_()
    payload = load_checkpoint(checkpoint, backend)
    optimizer.load_state_dict(payload["optimizer"])
    if not torch.equal(parameters[0], expected):
        raise RuntimeError("actual model checkpoint did not restore trainable weights")
    report["checkpoint_roundtrip"] = True
    report["peak_cpu_rss_gib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    )
    report["short_overfit_improved"] = (
        report["fixed_noise_loss_after"] < report["fixed_noise_loss_before"]
    )
    report["passed"] = report["short_overfit_improved"]
    report["phase"] = "complete"
    report["robot_evaluated"] = False
    save_report(report)
    if not report["passed"]:
        raise RuntimeError(
            "fixed-noise short-overfit loss did not decrease; inspect before formal training"
        )
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, TypeError, ValueError, OSError, ImportError) as error:
        path = ROOT / "data/preparation/preflight.json"
        value = json.loads(path.read_text()) if path.exists() else {}
        value.update(passed=False, error=f"{type(error).__name__}: {error}")
        save_report(value)
        with (ROOT / "data/preparation/preflight_history.jsonl").open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
        raise
