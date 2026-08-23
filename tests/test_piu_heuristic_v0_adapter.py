from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from piu.contracts import public_observation_sha256

ROOT = Path(__file__).resolve().parents[1]
FROZEN_COMMIT = "e7db12b7f35d9be416fc3ed57d36b12560e40cf0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capture(
    tmp_path: Path, *, split: str = "sealed_test"
) -> tuple[Path, Path, dict]:
    images = {}
    for camera, value in (("agentview", 17), ("wrist", 23)):
        path = tmp_path / f"{camera}.png"
        pixels = np.full((4, 5, 3), value, dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        images[camera] = {
            "path": str(path),
            "sha256": _sha256(path),
            "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        }
    observation = {"images": images, "public_robot_state": [0.0] * 8}
    transition = tmp_path / "transition.jsonl"
    transition.write_text(
        json.dumps(
            {
                "schema_version": "piu.public-transition.v1",
                "sample_id": "sample",
                "initial_state_group": "group",
                "split": split,
                "prompt": "Place the butter in the basket",
                "observations": {
                    "pre_interaction": observation,
                    "post_interaction": observation,
                },
                "public_action_history": {
                    "initial_observation": True,
                    "last_executed_candidate": None,
                },
                "candidate_actions": [
                    {
                        "candidate_id": "open",
                        "primitive": "OPEN",
                        "target": "middle drawer",
                    },
                    {"candidate_id": "stop", "primitive": "STOP"},
                ],
                "online_oracle_inputs": [],
            }
        )
        + "\n"
    )
    state = tmp_path / "state.npz"
    state.write_bytes(b"opaque state")
    capture = tmp_path / "capture.json"
    capture.write_text(
        json.dumps(
            {
                "schema_version": "piu.initial-observation-capture.v1",
                "sample_id": "sample",
                "online_oracle_inputs": [],
                "evaluator_fields_copied": [],
                "source_state": {
                    "path": str(state),
                    "sha256": _sha256(state),
                    "state_key": "state",
                },
                "public_transition": {
                    "path": str(transition),
                    "sha256": _sha256(transition),
                },
            }
        )
    )
    return capture, state, observation


def _attestation(
    tmp_path: Path, *, capture: Path, state: Path, observation: dict, action: str
) -> Path:
    contracts = {
        "OPEN_CONTAINER": {
            "mode": "OBSERVE",
            "primitive": "OPEN_CONTAINER",
            "target_id": "drawer-track",
            "subtask": "Open the middle drawer.",
        },
        "MOVE_CLOSER": {
            "mode": "OBSERVE",
            "primitive": "MOVE_CLOSER",
            "target_id": "drawer-track",
            "query": "Observe the drawer more closely.",
        },
    }
    report = tmp_path / f"{action}.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": (
                    "interaction-uncertainty.qwen-observation-pipeline.v0"
                ),
                "prompt": "Place the butter in the basket",
                "selected_action": {
                    "action": action,
                    "target_id": "drawer-track",
                },
                "execution_contract": contracts[action],
                "online_oracle_inputs": [],
            }
        )
    )
    model = {
        "schema_version": "piu.checkpoint-tree-sha256.v1",
        "sha256": "a" * 64,
        "file_count": 1,
        "total_bytes": 1,
    }
    transition_spec = json.loads(capture.read_text())["public_transition"]
    attestation = tmp_path / f"{action}_attestation.json"
    attestation.write_text(
        json.dumps(
            {
                "schema_version": "piu.heuristic-v0-inference-attestation.v1",
                "method_id": "B2",
                "frozen_commit": FROZEN_COMMIT,
                "one_step_baseline": True,
                "model_identities": {
                    name: model
                    for name in ("grounding_dino", "sam", "dinov2", "siglip", "qwen")
                },
                "decision_observation_sha256": public_observation_sha256(observation),
                "source_state": {
                    "path": str(state),
                    "sha256": _sha256(state),
                    "state_key": "state",
                },
                "inputs": {
                    "capture_report": {
                        "path": str(capture),
                        "sha256": _sha256(capture),
                    },
                    "public_transition": transition_spec,
                },
                "inference_report": {
                    "path": str(report),
                    "sha256": _sha256(report),
                },
                "online_oracle_inputs": [],
                "paper_method_claim_allowed": False,
            }
        )
    )
    return attestation


def test_frozen_v0_inference_plan_uses_exact_tag_worktree(tmp_path: Path) -> None:
    capture, _, _ = _capture(tmp_path)
    worktree = tmp_path / "heuristic-v0"
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", str(ROOT), str(worktree)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(worktree), "checkout", "--detach", FROZEN_COMMIT],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/pipeline/run_piu_heuristic_v0_inference.py"),
            "--worktree",
            str(worktree),
            "--capture-report",
            str(capture),
            "--output-dir",
            str(worktree / "runs/b2"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)
    assert plan["frozen_commit"] == FROZEN_COMMIT
    assert plan["external_gpu_required"] is True
    assert plan["local_gpu_run_allowed"] is False
    assert plan["command"][1].startswith(str(worktree))


def test_b2_adapter_executes_only_one_shared_physical_option(tmp_path: Path) -> None:
    capture, state, observation = _capture(tmp_path, split="development")
    attestation = _attestation(
        tmp_path,
        capture=capture,
        state=state,
        observation=observation,
        action="OPEN_CONTAINER",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/pipeline/run_piu_heuristic_v0_once.py"),
            "--attestation",
            str(attestation),
            "--scenario-config",
            str(ROOT / "configs/scenarios/original_drawer.yaml"),
            "--seed",
            "7",
            "--host",
            "pi05.example.internal",
            "--output-dir",
            str(tmp_path / "b2"),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)
    assert plan["method_id"] == "B2"
    assert plan["selected_action"] == "OPEN_CONTAINER"
    assert plan["one_step_baseline"] is True
    assert plan["command"][plan["command"].index("--steps") + 1] == "300"


def test_b2_adapter_abstains_on_a_legacy_option_outside_shared_budget(
    tmp_path: Path,
) -> None:
    capture, state, observation = _capture(tmp_path, split="development")
    attestation = _attestation(
        tmp_path,
        capture=capture,
        state=state,
        observation=observation,
        action="MOVE_CLOSER",
    )
    command = [
        sys.executable,
        str(ROOT / "scripts/pipeline/run_piu_heuristic_v0_once.py"),
        "--attestation",
        str(attestation),
        "--scenario-config",
        str(ROOT / "configs/scenarios/original_drawer.yaml"),
        "--seed",
        "7",
        "--host",
        "pi05.example.internal",
        "--output-dir",
        str(tmp_path / "b2"),
    ]
    completed = subprocess.run(
        [*command, "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(completed.stdout)
    assert plan["physical_action_supported_by_shared_contract"] is False
    assert plan["command"] is None
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    episode = json.loads((tmp_path / "b2/episode.json").read_text())
    assert episode["method_id"] == "B2"
    assert episode["rollout_status"] == "ABSTAINED"
    assert episode["outcomes"]["abstention"] is True
    assert episode["outcomes"]["interaction_count"] == 0
