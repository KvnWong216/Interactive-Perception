"""Validate privileged geometric transport on RGB, independent of latent features."""

import json
import time

import numpy as np
import torch
from diagnose_transition_targets import SOURCE, camera_map, save
from torch.nn import functional as F

started = time.monotonic()
torch.set_num_threads(2)
axis = np.arange(0, 256, 2) / 255
v, u = np.meshgrid(axis, axis, indexing="ij")
uv = np.stack((u.ravel(), v.ravel()), -1)
rows = []
for case in json.loads((SOURCE / "plan.json").read_text())["cases"]:
    if case["split"] == "confirmation":
        continue
    folder = SOURCE / f"case_{case['case_id']:03d}"
    current = dict(np.load(folder / "history_2.npz"))
    report = json.loads((folder / "report.json").read_text())
    for record in report["records"][:6]:
        e = next(e for e in record["endpoints"] if e["horizon"] == 10)
        future = dict(np.load(folder / e["file"]))
        for view in ["agent", "wrist"]:
            mapping, mask = camera_map(current, future, uv, view)
            image = (
                torch.tensor(current[f"{view}_rgb"].copy(), device="cuda")
                .float()
                .permute(2, 0, 1)[None]
                / 255
            )
            grid = (
                torch.tensor(mapping, device="cuda", dtype=torch.float32).reshape(
                    1, 128, 128, 2
                )
                * 2
                - 1
            )
            warped = (
                F.grid_sample(image, grid, align_corners=True, padding_mode="border")[0]
                .permute(1, 2, 0)
                .reshape(-1, 3)
            )
            old = (
                torch.tensor(current[f"{view}_rgb"][::2, ::2].copy(), device="cuda")
                .float()
                .reshape(-1, 3)
                / 255
            )
            target = (
                torch.tensor(future[f"{view}_rgb"][::2, ::2].copy(), device="cuda")
                .float()
                .reshape(-1, 3)
                / 255
            )
            valid = torch.tensor(mask, device="cuda")
            rows.append(
                {
                    "case": case,
                    "view": view,
                    "branch": record["branch"],
                    "coverage": float(valid.float().mean()),
                    "oracle_mse": float(
                        (warped[valid] - target[valid]).square().mean()
                    ),
                    "copy_mse": float((old[valid] - target[valid]).square().mean()),
                }
            )
save(
    "rgb_oracle.json",
    {
        "rows": rows,
        "seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    },
)
