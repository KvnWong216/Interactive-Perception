"""Stream completed simulator forks into frozen VLA/teacher features."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from gpu_devices import verify_cuda_target

from grounded_interaction.predictive_vla.config import VLAConfig, set_manual_seed
from grounded_interaction.predictive_vla.types import (
    AppliedAction,
    CameraFrame,
    Observation,
    PolicyContext,
)

ROOT = Path(__file__).resolve().parents[1]
for name, path in [
    ("HF_HOME", ".cache/huggingface"),
    ("HF_MODULES_CACHE", ".cache/huggingface/modules"),
]:
    os.environ[name] = str(ROOT / path)
os.environ["HF_HUB_OFFLINE"] = "1"


def observation(path, step):
    with np.load(path, allow_pickle=False) as a:
        return Observation(
            step,
            tuple(
                CameraFrame(*(a[f"{v}_{k}"] for k in ["rgb", "depth", "K", "T_world"]))
                for v in ["agent", "wrist"]
            ),
            a["states"],
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--physical-gpu", type=int, required=True)
    p.add_argument("--source", default="runs/stage2_forks_s7227")
    p.add_argument("--features-directory")
    p.add_argument("--checkpoint", default="runs/stage2_s17_gpu4567/best.pt")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    verify_cuda_target(a.physical_gpu)
    set_manual_seed(7227)
    torch.set_num_threads(2)
    out = (ROOT / a.source).resolve()
    feature_dir = (
        (ROOT / a.features_directory).resolve()
        if a.features_directory
        else out / "features"
    )
    checkpoint = (ROOT / a.checkpoint).resolve()
    if any(not path.is_relative_to(ROOT) for path in (out, feature_dir, checkpoint)):
        raise ValueError("encoding paths must stay in repository")
    feature_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads((out / "plan.json").read_text())
    from grounded_interaction.predictive_vla.backend import NativeVLABackend
    from grounded_interaction.predictive_vla.training import load_checkpoint

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    config = VLAConfig(**payload["config"])
    provenance = {
        "checkpoint": str(checkpoint),
        "updates": payload["updates"],
        "manual_seed": payload["manual_seed"],
        "config": config.to_dict(),
    }
    destination = feature_dir / "encoding.json"
    if any(feature_dir.glob("case_*.pt")) and not destination.exists():
        raise ValueError(
            "legacy cache has no lineage metadata; use a fresh feature directory"
        )
    if destination.exists() and json.loads(destination.read_text()) != provenance:
        raise ValueError("refusing mixed policy encoding provenance")
    if a.shard == 0:
        destination = feature_dir / "encoding.json"
        if destination.exists() and json.loads(destination.read_text()) != provenance:
            raise ValueError("refusing mixed policy encoding provenance")
        destination.write_text(json.dumps(provenance, indent=2) + "\n")
    del payload
    backend = NativeVLABackend.from_pretrained(
        config, device="cuda", local_path=ROOT / "checkpoints/base/MolmoAct2-LIBERO"
    )
    for param in backend.model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    load_checkpoint(checkpoint, backend)
    backend.model.requires_grad_(False)
    backend.train(False)
    started = time.monotonic()
    pending = list(plan["cases"][a.shard :: 2])
    count = 0
    while pending:
        advanced = False
        for case in pending[:]:
            folder = out / f"case_{case['case_id']:03d}"
            report_path = folder / "report.json"
            if not report_path.exists():
                continue
            report = json.loads(report_path.read_text())
            if not report["complete"]:
                continue
            destination = feature_dir / f"case_{case['case_id']:03d}.pt"
            if destination.exists():
                if not a.resume:
                    raise ValueError("refusing stale feature overwrite")
                existing = torch.load(
                    destination, map_location="cpu", weights_only=True, mmap=True
                )
                if (
                    existing.get("case") != case
                    or existing.get("encoding") != provenance
                    or existing.get("source_report") != report
                ):
                    raise ValueError("resume cache differs from its source or policy")
                del existing
                pending.remove(case)
                advanced = True
                continue
            context = PolicyContext(
                report["task"],
                tuple(
                    observation(folder / f"history_{i}.npz", i * 5) for i in range(3)
                ),
                tuple(
                    AppliedAction(i, np.asarray(value))
                    for i, value in enumerate(report["history_actions"])
                ),
                990,
            )
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                shared = backend.encode(context)
                current, positions, valid = backend.target(
                    context.current, context.task
                )
                cached = {
                    "case": case,
                    "encoding": provenance,
                    "shared": shared.hidden.cpu(),
                    "current": current.cpu(),
                    "positions": positions.cpu(),
                    "valid": valid.cpu(),
                    "branches": [],
                    "source_report": report,
                }
                for branch in report["records"]:
                    targets = []
                    for endpoint in branch["endpoints"]:
                        target, tp, tv = backend.target(
                            observation(
                                folder / endpoint["file"], 10 + endpoint["horizon"]
                            ),
                            context.task,
                        )
                        if not torch.equal(positions, tp) or not torch.equal(valid, tv):
                            raise ValueError("teacher grid/mask changed")
                        actions = np.tile(
                            np.asarray(branch["control"], dtype=np.float32),
                            (endpoint["horizon"], 1),
                        )
                        targets.append(
                            {
                                "horizon": endpoint["horizon"],
                                "target": target.cpu(),
                                "actions": backend.normalize_actions(actions)[None]
                                .float()
                                .cpu(),
                            }
                        )
                    cached["branches"].append(
                        {"branch": branch["branch"], "endpoints": targets}
                    )
                temporary = destination.with_suffix(".partial")
                torch.save(cached, temporary)
                temporary.replace(destination)
            del cached, shared
            pending.remove(case)
            count += 1
            advanced = True
            print(
                json.dumps(
                    {
                        "encoded_case": case["case_id"],
                        "count": count,
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
        if not advanced:
            if time.monotonic() - started > 2400:
                raise TimeoutError("collection made insufficient progress")
            time.sleep(3)
    (feature_dir / f"encoding_{a.shard}_complete.json").write_text(
        json.dumps(
            {
                "complete": True,
                "cases": count,
                "seconds": time.monotonic() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
        )
    )


if __name__ == "__main__":
    main()
