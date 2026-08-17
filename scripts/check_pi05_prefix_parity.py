#!/usr/bin/env python3
"""Block deployment unless online v5 prefix features reproduce offline sets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interactive_perception.action_outcome import (  # noqa: E402
    HierarchicalActionOutcomePredictor,
)
from interactive_perception.policy_client import (  # noqa: E402
    ObservationPacket,
    OpenPiWebsocketPolicy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/calibration/t01_open_and_observe_effect_v3.jsonl",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT
        / "outputs/t01_open_and_observe_effect_v3/pi05_temporal_embeddings_v5.npz",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT
        / "results/calibration/t01_open_and_observe_outcome_critic_v8.json",
    )
    parser.add_argument("--absolute-tolerance", type=float, default=5e-4)
    args = parser.parse_args()
    for name in ("dataset", "embeddings", "artifact"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)

    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line]
    data = np.load(args.embeddings)
    predictor = HierarchicalActionOutcomePredictor.from_artifact(
        json.loads(args.artifact.read_text())
    )
    policy = OpenPiWebsocketPolicy(host=args.host, port=args.port)
    metadata = policy.server_metadata
    if metadata.get("prefix_requests_advance_action_rng") is not False:
        raise ValueError("prefix parity requests must not advance action RNG")

    indices = [index for index, row in enumerate(rows) if row["split"] == "heldout_development"]
    if len(indices) != 21:
        raise ValueError(f"expected 21 held-out development rows, got {len(indices)}")
    maximum_error = 0.0
    for count, index in enumerate(indices, start=1):
        row = rows[index]
        offline = np.asarray(data["history_features"][index], dtype=np.float32)
        online = []
        for point in row["public_history"]:
            paths = point["image_paths"]
            packet = ObservationPacket(
                image=np.asarray(imageio.imread(ROOT / paths["agentview"])),
                wrist_image=np.asarray(imageio.imread(ROOT / paths["wrist"])),
                state=np.asarray(point["robot_state"], dtype=np.float32),
                prompt="Find the butter",
            )
            online.append(
                policy.encode_prefix(packet, feature_schema="cognitive_spatial_v5")
            )
        online_values = np.asarray(online, dtype=np.float32)
        error = float(np.max(np.abs(online_values - offline)))
        maximum_error = max(maximum_error, error)
        offline_set = predictor.predict_history(
            offline, data["robot_state_history"][index]
        ).prediction_set
        online_set = predictor.predict_history(
            online_values, data["robot_state_history"][index]
        ).prediction_set
        if offline_set != online_set:
            raise ValueError(
                f"row {index} changes conformal set offline={offline_set} online={online_set}"
            )
        print(
            f"[{count:02d}/{len(indices)}] {row['regime']} seed={row['seed']} "
            f"max_abs={error:.3g} set={online_set}",
            flush=True,
        )
    if maximum_error > args.absolute_tolerance:
        raise ValueError(
            f"online/offline maximum error {maximum_error} exceeds "
            f"{args.absolute_tolerance}"
        )
    print(
        json.dumps(
            {
                "rows": len(indices),
                "frames": len(indices) * 6,
                "maximum_absolute_error": maximum_error,
                "conformal_set_mismatches": 0,
                "action_rng_advanced": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
