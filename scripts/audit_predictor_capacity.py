"""Read-only MSE lower bounds for the stage-two linear output bottleneck."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from grounded_interaction.predictive_vla.config import set_manual_seed

ROOT = Path(__file__).resolve().parents[1]


def spectrum_bound(matrix, width):
    """Best per-window affine rank-width approximation, NOT a SmoothL1 bound."""
    centered = matrix.double() - matrix.double().mean(0, keepdim=True)
    eigenvalues = torch.linalg.eigvalsh(centered @ centered.T).clamp_min(0).flip(0)
    total = eigenvalues.sum()
    probabilities = eigenvalues / total.clamp_min(1e-30)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return {
        "best_affine_rank_mse_bound": float(eigenvalues[width:].sum() / matrix.numel()),
        "spatial_variance_mse": float(total / matrix.numel()),
        "tail_energy_fraction": float(
            eigenvalues[width:].sum() / total.clamp_min(1e-30)
        ),
        "effective_rank": float(entropy.exp()) if total > 1e-20 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="runs/stage2_predictor_debug_s5227")
    parser.add_argument("--output", default="runs/stage2_predictor_capacity_s6227.json")
    args = parser.parse_args()
    source, output = [(ROOT / value).resolve() for value in (args.cache, args.output)]
    if not all(p.is_relative_to(ROOT) for p in (source, output)):
        raise ValueError("diagnostic paths must stay inside repository")
    if output.exists():
        raise ValueError("refusing to overwrite a diagnostic result")
    torch.set_num_threads(4)
    set_manual_seed(6227)
    provenance = json.loads((source / "report.json").read_text())
    if not provenance["complete"] or provenance["updates"] != 1500:
        raise ValueError("expected the completed stage-two best diagnostic cache")
    checkpoint = torch.load(
        ROOT / "runs/stage2_s17_gpu4567/best.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if checkpoint["updates"] != 1500:
        raise ValueError("unexpected checkpoint update")
    weight = checkpoint["predictor"]["output.weight"].double()
    bias = checkpoint["predictor"]["output.bias"].double()
    u, singular, _ = torch.linalg.svd(weight, full_matrices=False)
    basis = u[:, singular > singular.max() * 1e-10]
    rows = []
    for i, item in enumerate(provenance["rows"]):
        cache = torch.load(
            source / f"context_{i}.pt", map_location="cpu", weights_only=True
        )
        valid = cache["valid"]
        target = F.layer_norm(cache["target"].float(), (weight.shape[0],))[
            valid
        ].double()
        current = F.layer_norm(cache["current"].float(), (weight.shape[0],))[
            valid
        ].double()
        centered = target - bias
        residual = centered - (centered @ basis) @ basis.T
        rows.append(
            {
                "episode_id": item["episode_id"],
                "step": item["step"],
                "target": spectrum_bound(target, weight.shape[1]),
                "future_minus_current": spectrum_bound(
                    target - current, weight.shape[1]
                ),
                "fixed_learned_output_mse_bound": float(residual.square().mean()),
                "copy_current_mse": float((target - current).square().mean()),
            }
        )
    report = {
        "schema": "predictor-capacity-audit-v1",
        "manual_seed": 6227,
        "checkpoint_update": 1500,
        "output_width": weight.shape[1],
        "teacher_width": weight.shape[0],
        "learned_output_rank": basis.shape[1],
        "optimizer_steps": 0,
        "rows": rows,
        "interpretation": "Bounds are for MSE, not training SmoothL1. Per-window optimal rank bounds permit a different subspace per window and are optimistic. Fixed-output bounds diagnose the learned output subspace, not every width-equivalent architecture. These four development windows do not establish generalization.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
