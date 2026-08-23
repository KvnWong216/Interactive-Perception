from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from piu.contracts import public_observation_sha256
from piu.temporal_memory import PublicObservationEvent, PublicTemporalMemory

ROOT = Path(__file__).resolve().parents[1]
DIGEST = "a" * 64


def _empty_event(step: int, source: str, *, certain: bool) -> PublicObservationEvent:
    return PublicObservationEvent.create(
        step=step,
        candidate_id=source,
        primitive="OPEN",
        information_source_id=source,
        region_confirmed_empty_set=(True,) if certain else (False, True),
        task_complete_set=(False,),
        post_observation_sha256=chr(ord("b") + step) * 64,
    )


def test_search_exhaustion_requires_every_post_observation_verifier() -> None:
    memory = PublicTemporalMemory(("drawer", "cabinet"), DIGEST)
    memory = memory.append(_empty_event(0, "drawer", certain=True))
    memory = memory.append(_empty_event(1, "cabinet", certain=False))
    assert memory.search_coverage_sufficient is False
    memory = memory.append(_empty_event(2, "cabinet", certain=True))
    assert memory.search_coverage_sufficient is True


def test_task_completion_is_distinct_from_search_coverage() -> None:
    memory = PublicTemporalMemory(("drawer",), DIGEST).append(
        PublicObservationEvent.create(
            step=0,
            candidate_id="place_target",
            primitive="PLACE",
            information_source_id=None,
            region_confirmed_empty_set=(),
            task_complete_set=(True,),
            post_observation_sha256="b" * 64,
        )
    )
    assert memory.task_complete is True
    assert memory.search_coverage_sufficient is False


def test_effect_forecast_fields_are_rejected_by_v2_memory() -> None:
    value = PublicTemporalMemory(("drawer",), DIGEST).to_public_history()
    value["events"] = [
        {
            "step": 0,
            "candidate_id": "open_drawer",
            "primitive": "OPEN",
            "information_source_id": "drawer",
            "execution_set": [True],
            "task_progress_set": [True],
            "region_confirmed_empty_set": [True],
            "task_complete_set": [False],
            "post_observation_sha256": "b" * 64,
        }
    ]
    value["current_observation_sha256"] = "b" * 64
    value["confirmed_empty_sources"] = ["drawer"]
    value["search_coverage_sufficient"] = True
    # Unknown fields are not interpreted as evidence; round-tripping projects
    # only post-observation verifier state.
    parsed = PublicTemporalMemory.from_mapping(value)
    serialized = parsed.to_public_history()
    assert "execution_set" not in serialized["events"][0]
    assert "task_progress_set" not in serialized["events"][0]


def test_controller_public_state_cli_recomputes_declared_memory(tmp_path: Path) -> None:
    memory = PublicTemporalMemory(("drawer",), DIGEST)
    source = tmp_path / "memory.jsonl"
    source.write_text(
        json.dumps(
            {
                "schema_version": "piu.public-controller-memory.v2",
                "sample_id": "sample",
                "initial_state_group": "group",
                "public_inputs_only": True,
                "online_oracle_inputs": [],
                "memory": memory.to_public_history(),
            }
        )
        + "\n"
    )
    output = tmp_path / "states.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/data/build_piu_controller_public_states.py"),
            "--memory",
            str(source),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    row = json.loads(output.read_text())
    assert row["state_sets"] == {"search_coverage_sufficient": [False]}
    assert row["derivation"]["schema"] == "piu.public-temporal-memory.v2"


def test_memory_update_rejects_a_broken_observation_chain(tmp_path: Path) -> None:
    transition = tmp_path / "transition.jsonl"
    observation = {
        "images": {"agentview": {"sha256": "1" * 64}},
        "public_robot_state": [0.0],
    }
    transition.write_text(
        json.dumps(
            {
                "schema_version": "piu.public-transition.v1",
                "sample_id": "post",
                "initial_state_group": "group",
                "split": "development",
                "prompt": "pick butter",
                "observations": {
                    "pre_interaction": observation,
                    "post_interaction": observation,
                },
                "public_action_history": {
                    "last_executed_candidate": {
                        "candidate_id": "open_drawer",
                        "primitive": "OPEN",
                    }
                },
                "candidate_actions": [
                    {"candidate_id": "open_drawer", "primitive": "OPEN"}
                ],
                "online_oracle_inputs": [],
            }
        )
        + "\n"
    )
    predictions = tmp_path / "binding.npz"
    import numpy as np

    np.savez_compressed(
        predictions,
        sample_id=np.asarray(["post"]),
        initial_state_group=np.asarray(["group"]),
        split=np.asarray(["development"]),
        image_valid_mask=np.asarray([[True]]),
        spatial_logits=np.asarray([[0.0]]),
        target_present_logit=np.asarray([0.0]),
        task_sufficiency_logit=np.asarray([0.0]),
        holding_requested_target_logit=np.asarray([0.0]),
        region_confirmed_empty_logit=np.asarray([10.0]),
        task_complete_logit=np.asarray([-10.0]),
    )
    report = tmp_path / "binding.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "piu.target-binder-online-predictions.v1",
                "inputs": {"checkpoint": {"sha256": "c" * 64}},
                "output": {
                    "path": str(predictions),
                    "sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                },
            }
        )
    )
    unsupported = {"status": "UNSUPPORTED"}
    binary = {
        "status": "SUPPORTED",
        "temperature": 1.0,
        "conformal": {
            "0.1": {
                "calibrator": {
                    "method": "class_conditional_LAC",
                    "alpha": 0.1,
                    "class_quantiles": {"0": 0.1, "1": 0.1},
                    "class_counts": {"0": 10, "1": 10},
                }
            }
        },
    }
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema_version": "piu.target-binder-calibration.v1",
                "checkpoint_sha256": "c" * 64,
                "primary_alpha": 0.1,
                "risk_contract": {"reported_alpha": [0.1]},
                "spatial": {
                    "temperature": 1.0,
                    "conformal": {"0.1": {"calibrator": {"quantile": 1.0}}},
                },
                "target_presence": unsupported,
                "task_sufficiency": unsupported,
                "holding_requested_target": unsupported,
                "region_confirmed_empty": binary,
                "task_complete": binary,
            }
        )
    )
    previous = tmp_path / "previous.jsonl"
    previous.write_text(
        json.dumps(
            {
                "schema_version": "piu.public-controller-memory.v2",
                "sample_id": "initial",
                "initial_state_group": "group",
                "public_inputs_only": True,
                "online_oracle_inputs": [],
                "memory": PublicTemporalMemory(
                    ("open_drawer",), "f" * 64
                ).to_public_history(),
            }
        )
        + "\n"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/data/update_piu_temporal_memory.py"),
            "--public-transition",
            str(transition),
            "--sample-id",
            "post",
            "--binding-predictions",
            str(predictions),
            "--binding-report",
            str(report),
            "--binder-calibration",
            str(calibration),
            "--previous-memory",
            str(previous),
            "--output",
            str(tmp_path / "updated.jsonl"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "does not continue memory head" in completed.stderr

    previous.unlink()
    previous.write_text(
        json.dumps(
            {
                "schema_version": "piu.public-controller-memory.v2",
                "sample_id": "initial",
                "initial_state_group": "group",
                "public_inputs_only": True,
                "online_oracle_inputs": [],
                "memory": PublicTemporalMemory(
                    ("open_drawer",), public_observation_sha256(observation)
                ).to_public_history(),
            }
        )
        + "\n"
    )
    output = tmp_path / "updated.jsonl"
    subprocess.run(
        [*completed.args[:-2], "--output", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    updated = json.loads(output.read_text())
    assert updated["memory"]["confirmed_empty_sources"] == ["open_drawer"]
    assert updated["memory"]["search_coverage_sufficient"] is True
    assert updated["memory"]["task_complete"] is False
    assert updated["verifier_semantics"].startswith("calibrated_post_observation")
