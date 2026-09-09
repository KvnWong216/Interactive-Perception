from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from grounded_interaction.psr.cli import (
    CALIBRATION_ROW_SCHEMA,
    EvaluationTrialResult,
    build_parser,
    main,
)
from grounded_interaction.psr.data import (
    PSR_BRANCH_SCHEMA,
    CollectionStatus,
    PSRBranchRecord,
    RecordKind,
)
from grounded_interaction.psr.evaluation import TrialOutcome
from grounded_interaction.psr.training import (
    CheckpointIdentity,
    CheckpointStage,
    freeze_execution_snapshot,
    save_training_checkpoint,
)
from grounded_interaction.psr.types import (
    IntentCandidate,
    PublicHistory,
    PublicObservation,
    RGBReference,
)

ROOT = Path(__file__).resolve().parents[1]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _observation(step: int) -> PublicObservation:
    return PublicObservation(
        RGBReference(f"agent-{step}", _digest(f"agent-{step}"), 32, 32),
        RGBReference(f"wrist-{step}", _digest(f"wrist-{step}"), 32, 32),
        (float(step), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        step,
    )


def _warmup_record() -> PSRBranchRecord:
    conditioned = IntentCandidate("inspect the label", (11, 12), "conditioned")
    native = IntentCandidate("put the item in the basket", (21, 22), "native")
    history = PublicHistory(
        task=native.text,
        current=_observation(0),
        previous=(),
        executed=(),
        remaining_control_steps=300,
    )
    return PSRBranchRecord(
        schema_version=PSR_BRANCH_SCHEMA,
        record_kind=RecordKind.WARMUP_DEMONSTRATIONS,
        record_id="warmup-real-0001",
        episode_id="episode-real-0001",
        group_id="family-real-0001/step-0",
        split="train",
        decision_step=0,
        remaining_steps=300,
        public_history=history,
        candidate_set=(conditioned, native),
        chosen_candidate_id=conditioned.candidate_id,
        chosen_intent_ids=conditioned.token_ids,
        candidate_generation_version="human-demonstration-v1",
        behavior_selection_rule="recorded-demonstrator-v1",
        behavior_probability=1.0,
        execution_snapshot_id="pre-s0-warmup-controller-v1",
        continuation_id="native-molmoact2-continuation-v1",
        target_encoder_id="frozen-native-visual-patches-v1",
        initial_reset_ref="collector-private-reset-real-0001",
        actual_action_chunks=("actions://real/0001",),
        actual_step_count=1,
        actual_observations=(_observation(1),),
        future_evidence_ref="evidence://real/0001",
        evidence_valid=True,
        final_success=None,
        cost_valid=False,
        terminal_reason="warmup_window_recorded",
        seed_schedule={"demonstration_seed": 17},
        collection_status=CollectionStatus.COMPLETED,
    )


def _identity(*, snapshot_id: str | None = None) -> CheckpointIdentity:
    return CheckpointIdentity(
        base_model_id="allenai/MolmoAct2-LIBERO",
        base_revision="0d24a92",
        upstream_revision="66b87e64",
        config_sha256="a" * 64,
        split_sha256="b" * 64,
        protocol_sha256="c" * 64,
        projection_sha256="d" * 64,
        normalization_sha256="e" * 64,
        seed=17,
        execution_snapshot_id=snapshot_id,
    )


def test_parser_exposes_the_frozen_command_surface() -> None:
    help_text = build_parser().format_help()
    for command in (
        "preflight",
        "import-data",
        "train-warmup",
        "collect",
        "train-outcomes",
        "calibrate",
        "evaluate",
    ):
        assert command in help_text
    assert "python -m grounded_interaction.psr" in help_text


def test_module_entrypoint_exposes_help_without_loading_weights() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "grounded_interaction.psr", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0
    assert "train-warmup" in completed.stdout
    assert "real data/model/environment commands fail closed" in completed.stdout


def test_cpu_preflight_is_weight_free(monkeypatch, capsys) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU preflight attempted to load the model")

    monkeypatch.setattr("grounded_interaction.psr.cli.real_model_preflight", forbidden)
    code = main(
        [
            "preflight",
            "--config",
            str(ROOT / "experiments" / "psr_v1.yaml"),
            "--device",
            "cpu",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert code in {0, 2}
    assert report["load_model"] is False
    assert report["measurements"]["weights_downloaded"] == "NOT_CHECKED"
    assert report["measurements"]["real_forward"] == "NOT_RUN"


def test_import_data_validates_and_copies_real_records(tmp_path, capsys) -> None:
    source = tmp_path / "real-source"
    source.mkdir()
    record = _warmup_record()
    source_bytes = json.dumps(record.to_dict(), sort_keys=True).encode() + b"\n"
    (source / "records.jsonl").write_bytes(source_bytes)
    (source / "frames").mkdir()
    (source / "frames" / "README.txt").write_text("real assets live here\n")
    output = tmp_path / "imported"
    assert main(["import-data", "--source", str(source), "--output", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dataset"]["record_kind"] == "warmup_demonstrations"
    assert report["dataset"]["records"] == 1
    assert (output / "records.jsonl").read_bytes() == source_bytes
    assert (output / "import_manifest.json").is_file()


def test_calibration_writes_bound_checkpoint_from_valid_prediction_rows(
    tmp_path, capsys
) -> None:
    torch = pytest.importorskip("torch")
    execution = torch.nn.Linear(2, 2)
    snapshot = freeze_execution_snapshot(execution, identity=_identity())
    snapshot_dir = tmp_path / "s0"
    snapshot_dir.mkdir()
    (snapshot_dir / "execution_snapshot.json").write_text(
        json.dumps(snapshot.to_dict(), sort_keys=True) + "\n"
    )

    predictor_dir = tmp_path / "predictor"
    predictor_dir.mkdir()
    predictor_path = predictor_dir / "predictor.pt"
    predictor = torch.nn.Linear(2, 1)
    save_training_checkpoint(
        predictor_path,
        stage=CheckpointStage.OUTCOME_PREDICTOR,
        model=predictor,
        identity=_identity(snapshot_id=snapshot.snapshot_id),
    )
    predictor_digest = hashlib.sha256(predictor_path.read_bytes()).hexdigest()
    data = tmp_path / "calibration_predictions.jsonl"
    rows = [
        {
            "schema_version": CALIBRATION_ROW_SCHEMA,
            "record_id": f"real-calibration-{index}",
            "split": "calibration",
            "execution_snapshot_id": snapshot.snapshot_id,
            "predictor_file_sha256": predictor_digest,
            "failure_logit": logit,
            "failure_target": target,
            "cost_valid": True,
        }
        for index, (logit, target) in enumerate(
            ((-2.0, 0), (-0.2, 1), (1.0, 0), (2.5, 1))
        )
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "calibrated"
    assert (
        main(
            [
                "calibrate",
                "--snapshot",
                str(snapshot_dir),
                "--predictor",
                str(predictor_dir),
                "--data",
                str(data),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["calibration"]["temperature"] > 0
    payload = torch.load(
        output / "calibrated_predictor.pt", map_location="cpu", weights_only=False
    )
    assert payload["stage"] == "calibrated_predictor"
    assert payload["calibration"]["examples"] == 4


def test_native_evaluation_uses_driver_then_repository_metrics(
    tmp_path, capsys, monkeypatch
) -> None:
    module = types.ModuleType("psr_cli_test_driver")

    def factory(**kwargs):
        assert kwargs["command"] == "evaluate"
        assert kwargs["mode"] == "native"

        def run_trial(spec):
            return EvaluationTrialResult(
                trial_id=spec.trial_id,
                outcome=TrialOutcome(
                    reset_family=spec.reset_family,
                    completed=True,
                    success=True,
                    actual_steps=300,
                    elapsed_seconds=1.0,
                ),
            )

        base = kwargs["config"].section("base")
        return {
            "run_trial": run_trial,
            "attestation": {
                "mode": "native",
                "base_model_id": base["model_id"],
                "base_revision": base["revision"],
                "execution_snapshot_id": None,
                "predictor_file_sha256": None,
            },
        }

    module.factory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    plan = tmp_path / "eval.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": "psr-v1-evaluation-plan-v1",
                "component_factory": "psr_cli_test_driver:factory",
                "trials": [
                    {
                        "trial_id": "real-trial-1",
                        "reset_family": "real-family-1",
                        "initial_reset_ref": "/registered/resets/real-1.npz",
                        "task": "put the requested item in the basket",
                        "episode_seed": 17,
                    }
                ],
            }
        )
    )
    output = tmp_path / "eval-output"
    assert (
        main(
            [
                "evaluate",
                "--config",
                str(ROOT / "experiments" / "psr_v1.yaml"),
                "--mode",
                "native",
                "--plan",
                str(plan),
                "--output",
                str(output),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["execution"]["planned"] == 1
    assert report["execution"]["task_successes"] == 1
    assert report["probability"] is None


def test_plan_examples_contain_no_outcome_or_result_fields() -> None:
    for name in ("collection_plan.example.json", "evaluation_plan.example.json"):
        value = json.loads((ROOT / "experiments" / "psr_v1" / name).read_text())
        serialized = json.dumps(value).casefold()
        for forbidden in (
            '"success"',
            '"failure"',
            '"reward"',
            '"outcome"',
            '"prediction"',
        ):
            assert forbidden not in serialized
