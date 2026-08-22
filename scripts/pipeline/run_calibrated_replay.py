#!/usr/bin/env python3
"""Run the learned-pipeline control contract against a no-training replay.

This validates candidate schemas, conformal arbitration, text serialization,
history updates, and closed-loop termination.  It deliberately does not claim
that the fixture probabilities came from a trained model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from calibrated_interaction.calibration import BinaryEffectCalibration, LACCalibrator
from calibrated_interaction.capabilities import CapabilityRegistry
from calibrated_interaction.contracts import CandidateAction
from calibrated_interaction.controller import (
    CalibratedSelector,
    ClosedLoopController,
    ModelPrediction,
)
from calibrated_interaction.data import assert_policy_input_clean


class ReplayBackend:
    def __init__(
        self, candidates: Sequence[CandidateAction], steps: Sequence[Mapping[str, Any]]
    ):
        self.candidates = tuple(candidates)
        self.steps = tuple(steps)
        self.index = 0
        self.pending: Mapping[str, Any] | None = None

    def predict(self, *, prompt: str, history: Sequence[Mapping[str, Any]]):
        assert_policy_input_clean({"prompt": prompt, "history": history})
        if self.index >= len(self.steps):
            raise RuntimeError("replay exhausted before the controller terminated")
        row = self.steps[self.index]
        self.pending = row
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        prediction = ModelPrediction(
            candidate_ids=candidate_ids,
            route_probabilities=dict(row["route_probabilities"]),
            effect_positive_probabilities={
                candidate_id: dict(probabilities)
                for candidate_id, probabilities in row[
                    "effect_positive_probabilities"
                ].items()
            },
        )
        return self.candidates, prediction, tuple(row["public_observation_frames"])

    def execute(self, *, candidate: CandidateAction, instruction: str):
        if self.pending is None:
            raise RuntimeError("execute called without a pending replay prediction")
        result = dict(self.pending["execution"])
        self.index += 1
        self.pending = None
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", type=Path, default=ROOT / "configs/scenarios/original_drawer.yaml"
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=ROOT / "tests/fixtures/original_drawer_calibrated_replay.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = yaml.safe_load(args.scenario.read_text(encoding="utf-8"))
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    if replay.get(
        "schema_version"
    ) != "calibrated-interaction.replay-fixture.v1" or not replay.get("fixture_only"):
        raise ValueError("replay must be an explicitly fixture-only artifact")
    method = scenario["calibrated_interaction"]
    candidates = tuple(
        CandidateAction.from_mapping(row) for row in method["candidates"]
    )
    registry_path = ROOT / method["capability_registry"]
    selector = CalibratedSelector(
        LACCalibrator.from_dict(replay["route_calibration"]),
        BinaryEffectCalibration.from_dict(replay["effect_calibration"]),
    )
    controller = ClosedLoopController(
        selector=selector,
        capabilities=CapabilityRegistry.load(registry_path),
        backend=ReplayBackend(candidates, replay["steps"]),
        max_steps=len(replay["steps"]),
    )
    trace = controller.run(prompt=scenario["task"]["prompt"])
    trace["validation_scope"] = replay["claim"]
    trace["scenario"] = str(args.scenario.resolve().relative_to(ROOT))
    trace["replay_fixture"] = str(args.replay.resolve().relative_to(ROOT))
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable trace: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "terminal": trace["terminal"],
                "steps": len(trace["steps"]),
            }
        )
    )


if __name__ == "__main__":
    main()
