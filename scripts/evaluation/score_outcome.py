#!/usr/bin/env python3
"""Score one six-frame option trace with a frozen public-RGB outcome critic."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from interactive_perception.policy_client import ObservationPacket  # noqa: E402
from interactive_perception.rgb_outcome_critic import (  # noqa: E402
    PublicRGBOutcomeCritic,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--option-report", type=Path, required=True)
    parser.add_argument("--composite", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--fusion",
        choices=("strict", "complementary"),
        default="strict",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("option_report", "composite", "output"):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, ROOT / path)
    if args.output.exists():
        raise FileExistsError("outcome report is immutable")
    option = json.loads(args.option_report.read_text())
    if option.get("online_oracle_inputs"):
        raise ValueError("option report declares forbidden online oracle inputs")
    packets = []
    for row in option["controller"]["keyframes"]:
        packets.append(
            ObservationPacket(
                image=np.asarray(imageio.imread(ROOT / row["image_paths"]["agentview"])),
                wrist_image=np.asarray(imageio.imread(ROOT / row["image_paths"]["wrist"])),
                state=np.asarray(row["public_robot_state"], dtype=np.float64),
                prompt=args.prompt,
            )
        )
    critic = PublicRGBOutcomeCritic(
        args.composite, root=ROOT, fusion=args.fusion
    )
    prediction = critic.predict(packets)
    report = {
        "schema_version": "interaction-uncertainty.public-rgb-option-outcome.v1",
        "claim_status": "disposable inference with frozen candidate; not clean/sealed",
        "camera_fusion": args.fusion,
        "prompt": args.prompt,
        "option_report": {
            "path": str(args.option_report.relative_to(ROOT)),
            "sha256": digest(args.option_report),
        },
        "composite": {
            "path": str(args.composite.relative_to(ROOT)),
            "sha256": digest(args.composite),
        },
        "prediction": prediction.to_dict(),
        "online_inputs": [
            "six agentview RGB frames",
            "six wrist RGB frames",
            "public robot state",
        ],
        "online_oracle_inputs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["prediction"], indent=2))


if __name__ == "__main__":
    main()
