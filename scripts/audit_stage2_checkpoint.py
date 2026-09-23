"""Read back the real preflight checkpoint before allowing the main launch."""

import argparse
import json
from pathlib import Path

import torch

from grounded_interaction.predictive_vla.config import load_config
from grounded_interaction.predictive_vla.training import read_checkpoint


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/stage2_transition_preflight_s17_gpu4567")
    parser.add_argument("--config", default="experiments/stage2_transition.yaml")
    args = parser.parse_args()
    output = (root / args.run).resolve()
    config_path = (root / args.config).resolve()
    if not output.is_relative_to(root) or not config_path.is_relative_to(root):
        raise ValueError("audit paths must stay in repository")
    report = json.loads((output / "preflight.json").read_text())
    if not report["passed"]:
        raise RuntimeError("preflight did not finish successfully")
    torch.set_num_threads(4)
    payload, config = read_checkpoint(output / "last.pt")
    if config != load_config(config_path):
        raise ValueError("saved preflight configuration differs")
    if (
        payload["updates"] != 2
        or len(payload["rank_rng"]) != 4
        or payload["initialization"]["source_updates"] != 1500
        or not payload["initialization"]["exact_restoration"]
        or payload["predictor"] is None
        or len(payload["optimizer"]["param_groups"]) != 4
    ):
        raise ValueError("incomplete stage-two checkpoint")
    for name, tensor in {**payload["adapters"], **payload["predictor"]}.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"nonfinite saved parameter: {name}")
    result = {
        "passed": True,
        "manual_seed": config.manual_seed,
        "updates": payload["updates"],
        "policy_tensors": len(payload["adapters"]),
        "predictor_tensors": len(payload["predictor"]),
        "optimizer_groups": len(payload["optimizer"]["param_groups"]),
        "rank_random_states": len(payload["rank_rng"]),
        "all_saved_weights_finite": True,
    }
    temporary = output / "checkpoint_audit.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output / "checkpoint_audit.json")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
