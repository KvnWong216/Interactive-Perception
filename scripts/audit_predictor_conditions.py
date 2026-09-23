"""Frozen-head sensitivity probes; interventions are diagnostics, not qualification."""

import argparse
import json
from pathlib import Path

import torch
from diagnose_transition_targets import load
from train_transition_predictor import evaluate

from grounded_interaction.predictive_vla.transport import LocalTransportPredictor

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    source, head, output = [(ROOT / x).resolve() for x in [a.source, a.head, a.output]]
    if any(not x.is_relative_to(ROOT) for x in [source, head, output]):
        raise ValueError("paths must stay inside repository")
    torch.manual_seed(17)
    torch.set_num_threads(2)
    payload = torch.load(head, map_location="cpu", weights_only=True)
    model = (
        LocalTransportPredictor(
            payload["native_width"],
            width=payload["width"],
            time_scale=payload["time_scale"],
        )
        .cuda()
        .eval()
    )
    model.load_state_dict(payload["predictor"])
    cases = load("cuda", source, source / "features_stage1", splits=("confirmation",))
    results = []
    for mode in ["unchanged", "state_clip3", "state_clip1", "zero_context"]:
        altered = []
        for c in cases:
            state = c["state"]
            if mode.startswith("state_clip"):
                limit = int(mode[-1])
                state = model.state_mean + model.state_std * (
                    (state - model.state_mean) / model.state_std
                ).clamp(-limit, limit)
            altered.append({**c, "state": state})
        result = evaluate(model, altered, zero_context=mode == "zero_context")
        results.append({"mode": mode, **result})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump({"diagnostic_only": True, "rows": results}, f, indent=2)


if __name__ == "__main__":
    main()
