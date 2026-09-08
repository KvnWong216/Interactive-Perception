#!/usr/bin/env python3
"""Regenerate the unexecuted E1a pilot plan from auditable local inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "experiments/e1_referent_ceiling/pilot_state0_v1.json"
RUNNER_FILES = tuple(
    sorted(
        {
            "pyproject.toml",
            "uv.lock",
            "src/grounded_interaction/conditioning.py",
            "src/grounded_interaction/contracts.py",
            "src/grounded_interaction/data.py",
            "src/grounded_interaction/e1.py",
            "src/grounded_interaction/e1_training.py",
            "src/grounded_interaction/execution.py",
            "src/grounded_interaction/libero_runtime.py",
            "src/grounded_interaction/losses.py",
            "src/grounded_interaction/molmoact2.py",
            "src/grounded_interaction/rgb.py",
            "src/grounded_interaction/serialization.py",
        }
    )
)
SCORED_STATE = [
    -0.21005375683307648,
    -0.0007738260901533067,
    1.1812676191329956,
    3.140786647796631,
    0.0021030099596828222,
    -0.09107550233602524,
    0.038723304867744446,
    -0.038722287863492966,
]
CANARY_STATE = [
    -0.2049785554409027,
    -0.0002499072579666972,
    1.1861578226089478,
    3.1400699615478516,
    0.001452555530704558,
    -0.08685237169265747,
    0.03872224688529968,
    -0.03872334212064743,
]


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    existing = json.loads(PLAN.read_text(encoding="utf-8"))
    canonical_runs = ROOT / "runs/e1/e1-moka-state0-pilot-v1"
    canary_report = ROOT / "runs/e1_canary/e1-moka-state0-pilot-v1.json"
    if canonical_runs.exists() or canary_report.exists():
        raise RuntimeError(
            "E1 plan is outcome-bearing or canary-bound and cannot change"
        )

    source_files = [
        {"path": relative, "sha256": file_sha256(ROOT / relative)}
        for relative in RUNNER_FILES
    ]
    runner_source = {
        "schema_version": "e1-runner-source-tree-v1",
        "files": source_files,
    }
    runner_source["tree_sha256"] = canonical_sha256(runner_source)

    candidate_ids = {
        "moka_pot_1": "candidate-5c89f4a2",
        "moka_pot_2": "candidate-9e31d6b7",
    }
    trials = existing["trials"]
    for trial in trials:
        target = trial["evaluator_sidecar"]["target_object"]
        trial["candidate_id"] = candidate_ids[target]

    payload = {
        "schema_version": "e1-referent-executor-plan-v1",
        "plan_id": "e1-moka-state0-pilot-v1",
        "runner_source": runner_source,
        "model_canary": {
            "init_state_index": 49,
            "env_seed": 0,
            "reset_state_sha256": (
                "03d21e9eda5027be6b72f057dfb8370c54e8a42156e43eab2530e72680f9d486"
            ),
            "expected_agentview_sha256": (
                "730792faeea46cb8432c87dd5308c98ff3834dc1b96643b719ca565063f72145"
            ),
            "expected_wrist_sha256": (
                "f1a915e973c317b0de162ca047e7ba18bf58f588ce808b85d4f6b46d4cb56345"
            ),
            "expected_state": CANARY_STATE,
            "expected_state_sha256": (
                "0238144dc00ebb6e46667ba3dd741a0c3a064befdfd71255039d140f9a48965c"
            ),
            "instruction": "Put the right moka pot on the stove.",
            "model_seed": 26090799,
            "report_relative_path": ("runs/e1_canary/e1-moka-state0-pilot-v1.json"),
        },
        "prompt": "Put the selected moka pot on the stove.",
        "simulator": {
            **existing["simulator"],
            "control_mode": "relative",
        },
        "public_observation": {
            **existing["public_observation"],
            "expected_state": SCORED_STATE,
            "expected_state_sha256": (
                "889fdde3a405e067fb3dad044885bf60163a68fe6cfb6df8a31b8004cd1118d5"
            ),
        },
        "executor": {
            **existing["executor"],
            "config_sha256": (
                "2e521b21ac0a74eb124e20242704957177bf32a7c4296c01d7180b74994c40b4"
            ),
            "norm_stats_sha256": (
                "de7de428abfd1af0aff0e97ff499a4a27012106b3982d22e8b8d029de312db80"
            ),
            "checkpoint_manifest_sha256": (
                "bfdfaa64ada2a050e24cc89a3e8a40c3776d07405329f4a0588dbead3e094ec7"
            ),
            "checkpoint_file_count": 19,
            "checkpoint_total_bytes": 21781401448,
            "norm_stats_format": "molmoact2_norm_stats.v1",
            "norm_mode": "q01_q99",
            "control_mode": "delta end-effector pose",
        },
        "execution": existing["execution"],
        "trials": trials,
    }
    payload["frozen_plan_sha256"] = canonical_sha256(payload)
    PLAN.write_text(
        json.dumps(
            payload,
            sort_keys=False,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(payload["frozen_plan_sha256"])


if __name__ == "__main__":
    main()
