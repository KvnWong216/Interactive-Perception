from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from grounded_interaction.contracts import (
    GroundedIntervention,
    GroundingReference,
    PolicyContext,
    Primitive,
    PublicFrame,
    canonical_json_bytes,
    canonical_sha256,
)
from grounded_interaction.method_v1_data import (
    DecisionGroupManifest,
    build_branch_schedule,
    make_method_v1_outcome_contract,
)
from grounded_interaction.model import GroundedOutcomeModel
from grounded_interaction.qwen_provider import (
    QwenFeatureCache,
    candidate_feature_cache_key,
    qwen_cache_artifact_sha256,
)
from grounded_interaction.selection import ExpectedSuccessSelector, ValuePrediction
from grounded_interaction.selection_provenance import (
    CALIBRATOR_SCHEMA,
    _build_frozen_selection_artifact,
    _main,
    create_frozen_scorer_selection_for_model_seed,
    resolve_selected_schedule_entry,
    verify_frozen_scorer_selection,
)
from grounded_interaction.tokens import CandidateTokenField, FrozenTokenField

torch = pytest.importorskip("torch")


def _digest(character: str) -> str:
    return character * 64


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _manifest() -> DecisionGroupManifest:
    frame = PublicFrame(
        frame_id="agent-0",
        camera="agentview",
        frame_index=0,
        image_sha256=_digest("a"),
        width=256,
        height=256,
    )
    context = PolicyContext(
        prompt="Put the butter in the basket.",
        frames=(frame,),
        proprioception=(0.0,) * 8,
    )
    candidates = tuple(
        GroundedIntervention(
            candidate_id=candidate_id,
            primitive=primitive,
            referent=referent,
            parameters=(),
            grounding=GroundingReference(
                camera=frame.camera,
                frame_id=frame.frame_id,
                frame_index=frame.frame_index,
                image_sha256=frame.image_sha256,
                box_xyxy=(x - 0.1, 0.3, x + 0.1, 0.7),
                point_xy=(x, 0.5),
            ),
        )
        for candidate_id, primitive, referent, x in (
            ("direct-butter", Primitive.DIRECT, "butter package", 0.3),
            ("open-drawer", Primitive.OPEN, "middle drawer", 0.7),
        )
    )
    contract = make_method_v1_outcome_contract(
        continuation_policy_id=_digest("b"),
        executor_id="molmoact2-test",
        serializer_id="grounded-precise-text-v1",
    )
    return DecisionGroupManifest(
        manifest_id="manifest-a",
        experiment_id="method-v1-test",
        initial_state_group="state-a",
        decision_group_id="decision-a",
        split_group_id="split-a",
        split="train",
        information_stratum="INFORMATION_NECESSARY",
        scene_id="drawer-scene",
        layout_id="layout-a",
        reset_state_sha256=_digest("c"),
        context=context,
        candidates=candidates,
        outcome_contract=contract,
        proposal_provider_id="qwen-proposal-test",
        proposal_request_sha256=_digest("d"),
        token_provider_id="qwen-token-test",
        token_cache_sha256=_digest("e"),
        configuration_sha256=_digest("f"),
        model_seeds=(101, 202),
    )


def _frozen_inputs(tmp_path: Path) -> dict[str, object]:
    manifest = _manifest()
    logical_cache_key = candidate_feature_cache_key(
        provider_id=manifest.token_provider_id,
        context=manifest.context,
        candidates=manifest.candidates,
    )
    cache_tensor_bytes = b"detached tensor cache bytes"
    tensor_artifact_sha256 = hashlib.sha256(cache_tensor_bytes).hexdigest()
    cache_sidecar_value = {
        "schema": QwenFeatureCache.SIDECAR_SCHEMA,
        "logical_key": logical_cache_key,
        "provider_id": manifest.token_provider_id,
        "context_fingerprint": manifest.context.fingerprint(),
        "candidate_ids": [item.candidate_id for item in manifest.candidates],
        "candidate_fingerprints": [item.fingerprint() for item in manifest.candidates],
        "primitives": [item.primitive.value for item in manifest.candidates],
        "camera_ids": [[]],
        "frame_ids": [[]],
        "tensor_sha256": {
            name: hashlib.sha256(name.encode()).hexdigest()
            for name in QwenFeatureCache._tensor_names()
        },
        "tensor_artifact_sha256": tensor_artifact_sha256,
    }
    cache_sidecar_bytes = canonical_json_bytes(cache_sidecar_value) + b"\n"
    sidecar_artifact_sha256 = hashlib.sha256(cache_sidecar_bytes).hexdigest()
    cache_artifact_sha256 = qwen_cache_artifact_sha256(
        logical_key=logical_cache_key,
        tensor_artifact_sha256=tensor_artifact_sha256,
        sidecar_artifact_sha256=sidecar_artifact_sha256,
    )
    manifest = dataclasses.replace(manifest, token_cache_sha256=cache_artifact_sha256)
    schedule = build_branch_schedule((manifest,), schedule_id="schedule-a")
    entry = next(item for item in schedule if item.candidate_id == "open-drawer")
    checkpoint = tmp_path / "checkpoint" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"actual immutable checkpoint bytes")
    checkpoint_sha256 = _file_sha256(checkpoint)
    identity = {
        "schema_version": "method-v1-outcome-checkpoint-v1",
        "seed": 17,
        "experiment_id": manifest.experiment_id,
        "configuration_sha256": manifest.configuration_sha256,
        "proposal_provider_id": manifest.proposal_provider_id,
        "provider_id": manifest.token_provider_id,
        "outcome_contract_sha256": manifest.outcome_contract.fingerprint(),
        "training_dataset_sha256": _digest("1"),
        "training_admission_evidence_sha256": _digest("2"),
        "collection_plan_sha256": _digest("3"),
        "scorer_verifier_auth_key_id": _digest("4"),
        "training_split_group_ids": ["training-family-a", "validation-family-a"],
        "model": {"class": "GroundedOutcomeModel"},
        "training": {"optimizer": "AdamW"},
        "best_epoch": 2,
        "best_validation_nll": 0.4,
        "trainable_parameter_count": 10,
    }
    identity_sha256 = canonical_sha256(identity)
    identity_path = checkpoint.parent / "identity.json"
    _write_json(identity_path, {**identity, "identity_sha256": identity_sha256})

    cache_root = tmp_path / "cache"
    cache_dir = cache_root / "sha256" / manifest.token_cache_sha256[:2]
    cache_tensor = cache_dir / f"{manifest.token_cache_sha256}.pt"
    cache_tensor.parent.mkdir(parents=True)
    cache_tensor.write_bytes(cache_tensor_bytes)
    cache_sidecar = cache_dir / f"{manifest.token_cache_sha256}.json"
    cache_sidecar.write_bytes(cache_sidecar_bytes)
    predictions = (
        ValuePrediction(
            candidate_id=manifest.candidates[0].candidate_id,
            candidate_fingerprint=manifest.candidates[0].fingerprint(),
            success_probability=0.2,
            feasible=True,
        ),
        ValuePrediction(
            candidate_id=manifest.candidates[1].candidate_id,
            candidate_fingerprint=manifest.candidates[1].fingerprint(),
            success_probability=0.8,
            feasible=True,
        ),
    )
    decision = ExpectedSuccessSelector().select(manifest.candidates, predictions)
    selection = {
        "schema_version": "method-v1-scorer-selection-v1",
        "evidence_scope": "PRE_EXECUTION_MODEL_DECISION",
        "manifest_sha256": manifest.fingerprint(),
        "token_cache_sha256": manifest.token_cache_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_identity_sha256": identity_sha256,
        "outcome_contract_sha256": manifest.outcome_contract.fingerprint(),
        "temperature": 1.0,
        "predictions": [item.to_dict() for item in predictions],
        "decision": decision.to_dict(),
    }
    return {
        "manifest": manifest,
        "entry": entry,
        "schedule": schedule,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "identity_path": identity_path,
        "identity_sha256": identity_sha256,
        "cache_root": cache_root,
        "selection": selection,
    }


def _build(inputs: dict[str, object], **changes: object) -> dict[str, object]:
    values = {
        "selection_value": inputs["selection"],
        "manifest": inputs["manifest"],
        "entry": inputs["entry"],
        "checkpoint_path": inputs["checkpoint"],
        "checkpoint_identity_path": inputs["identity_path"],
        "cache_root": inputs["cache_root"],
        "expected_checkpoint_sha256": inputs["checkpoint_sha256"],
        "expected_checkpoint_identity_sha256": inputs["identity_sha256"],
    }
    values.update(changes)
    return _build_frozen_selection_artifact(**values)  # type: ignore[arg-type]


def _install_replay(
    monkeypatch: pytest.MonkeyPatch, selection: object
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_select_manifest_candidate(
        received_manifest: DecisionGroupManifest, **kwargs: object
    ) -> SimpleNamespace:
        calls.append({"manifest": received_manifest, **kwargs})
        return SimpleNamespace(to_dict=lambda: selection)

    monkeypatch.setattr(
        "grounded_interaction.train_outcomes.select_manifest_candidate",
        fake_select_manifest_candidate,
    )
    return calls


def test_frozen_selection_binds_real_files_cache_and_schedule_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _frozen_inputs(tmp_path)
    calls = _install_replay(monkeypatch, inputs["selection"])
    artifact = _build(inputs)
    path = tmp_path / "selection.frozen.json"
    _write_json(path, artifact)
    verified = verify_frozen_scorer_selection(
        path,
        manifest=inputs["manifest"],  # type: ignore[arg-type]
        entry=inputs["entry"],  # type: ignore[arg-type]
    )
    assert verified.artifact_sha256 == artifact["artifact_sha256"]
    assert verified.checkpoint_sha256 == inputs["checkpoint_sha256"]
    assert verified.training_dataset_sha256 == _digest("1")
    assert verified.training_admission_evidence_sha256 == _digest("2")
    assert verified.training_collection_plan_sha256 == _digest("3")
    assert verified.scorer_verifier_auth_key_id == _digest("4")
    assert verified.candidate_id == "open-drawer"
    assert verified.temperature == 1.0
    assert len(calls) == 1
    assert calls[0]["device"] == "cpu"

    checkpoint = inputs["checkpoint"]
    assert isinstance(checkpoint, Path)
    checkpoint.write_bytes(b"different checkpoint bytes")
    with pytest.raises(ValueError, match="checkpoint file changed"):
        verify_frozen_scorer_selection(
            path,
            manifest=inputs["manifest"],  # type: ignore[arg-type]
            entry=inputs["entry"],  # type: ignore[arg-type]
        )


def test_selection_creation_rejects_self_declared_hashes_and_unfrozen_temperature(
    tmp_path: Path,
) -> None:
    inputs = _frozen_inputs(tmp_path)
    bare = tmp_path / "bare-selection.json"
    _write_json(bare, inputs["selection"])
    with pytest.raises(ValueError, match="frozen schema"):
        verify_frozen_scorer_selection(
            bare,
            manifest=inputs["manifest"],  # type: ignore[arg-type]
            entry=inputs["entry"],  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="actual checkpoint differs"):
        _build(inputs, expected_checkpoint_sha256=_digest("9"))
    with pytest.raises(ValueError, match="checkpoint identity differs"):
        _build(inputs, expected_checkpoint_identity_sha256=_digest("9"))

    identity_path = inputs["identity_path"]
    assert isinstance(identity_path, Path)
    original_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    incomplete_identity = dict(original_identity)
    incomplete_identity.pop("collection_plan_sha256")
    incomplete_body = {
        key: value
        for key, value in incomplete_identity.items()
        if key != "identity_sha256"
    }
    incomplete_identity["identity_sha256"] = canonical_sha256(incomplete_body)
    _write_json(identity_path, incomplete_identity)
    with pytest.raises(ValueError, match="identity fields differ"):
        _build(
            inputs,
            expected_checkpoint_identity_sha256=incomplete_identity["identity_sha256"],
        )
    _write_json(identity_path, original_identity)

    selection = dict(inputs["selection"])  # type: ignore[arg-type]
    selection["checkpoint_sha256"] = _digest("8")
    with pytest.raises(ValueError, match="does not name the actual checkpoint"):
        _build(inputs, selection_value=selection)

    selection = dict(inputs["selection"])  # type: ignore[arg-type]
    selection["temperature"] = 1.7
    with pytest.raises(ValueError, match="requires a frozen calibrator"):
        _build(inputs, selection_value=selection)


def test_frozen_temperature_calibrator_is_rehashed_at_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _frozen_inputs(tmp_path)
    selection = dict(inputs["selection"])  # type: ignore[arg-type]
    selection["temperature"] = 1.7
    _install_replay(monkeypatch, selection)
    calibrator_body = {
        "schema_version": CALIBRATOR_SCHEMA,
        "method": "positive_scalar_temperature",
        "temperature": 1.7,
        "checkpoint_sha256": inputs["checkpoint_sha256"],
        "checkpoint_identity_sha256": inputs["identity_sha256"],
        "outcome_contract_sha256": inputs["manifest"].outcome_contract.fingerprint(),  # type: ignore[union-attr]
        "calibration_dataset_sha256": _digest("3"),
        "calibration_split_group_ids_sha256": _digest("4"),
        "fit_config_sha256": _digest("5"),
        "calibration_group_count": 4,
        "calibration_example_count": 20,
    }
    calibrator = tmp_path / "temperature.json"
    _write_json(
        calibrator,
        {**calibrator_body, "identity_sha256": canonical_sha256(calibrator_body)},
    )
    artifact = _build(
        inputs,
        selection_value=selection,
        calibrator_path=calibrator,
        expected_calibrator_sha256=_file_sha256(calibrator),
    )
    path = tmp_path / "selection-calibrated.frozen.json"
    _write_json(path, artifact)
    verified = verify_frozen_scorer_selection(
        path,
        manifest=inputs["manifest"],  # type: ignore[arg-type]
        entry=inputs["entry"],  # type: ignore[arg-type]
    )
    assert verified.temperature == 1.7

    changed = json.loads(calibrator.read_text(encoding="utf-8"))
    changed["calibration_example_count"] = 21
    _write_json(tmp_path / "unused.json", changed)
    calibrator.write_bytes(canonical_json_bytes(changed) + b"\n")
    with pytest.raises(ValueError, match="identity digest mismatch"):
        verify_frozen_scorer_selection(
            path,
            manifest=inputs["manifest"],  # type: ignore[arg-type]
            entry=inputs["entry"],  # type: ignore[arg-type]
        )


def test_model_seed_path_scores_once_then_resolves_argmax_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _frozen_inputs(tmp_path)
    selection = inputs["selection"]
    manifest = inputs["manifest"]
    assert isinstance(selection, dict)
    assert isinstance(manifest, DecisionGroupManifest)
    decision = ExpectedSuccessSelector().select(
        manifest.candidates,
        tuple(
            ValuePrediction(
                candidate_id=str(value["candidate_id"]),
                candidate_fingerprint=str(value["candidate_fingerprint"]),
                success_probability=float(value["success_probability"]),
                feasible=bool(value["feasible"]),
            )
            for value in selection["predictions"]
        ),
    )
    calls: list[dict[str, object]] = []

    def fake_select_manifest_candidate(
        received_manifest: DecisionGroupManifest, **kwargs: object
    ) -> SimpleNamespace:
        calls.append({"manifest": received_manifest, **kwargs})
        return SimpleNamespace(decision=decision, to_dict=lambda: selection)

    monkeypatch.setattr(
        "grounded_interaction.train_outcomes.select_manifest_candidate",
        fake_select_manifest_candidate,
    )
    output = tmp_path / "automatic-selection.json"
    verified, entry = create_frozen_scorer_selection_for_model_seed(
        manifest=manifest,
        schedule=inputs["schedule"],  # type: ignore[arg-type]
        model_seed=202,
        cache_root=inputs["cache_root"],  # type: ignore[arg-type]
        checkpoint_path=inputs["checkpoint"],  # type: ignore[arg-type]
        checkpoint_identity_path=inputs["identity_path"],  # type: ignore[arg-type]
        expected_checkpoint_sha256=str(inputs["checkpoint_sha256"]),
        expected_checkpoint_identity_sha256=str(inputs["identity_sha256"]),
        output_path=output,
    )

    assert len(calls) == 2
    assert calls[0]["manifest"] is manifest
    assert entry.candidate_id == "open-drawer"
    assert entry.model_seed == 202
    assert entry.execution_index == 3
    assert verified.candidate_fingerprint == entry.candidate_fingerprint
    assert output.is_file()


def test_rehashed_forged_scores_are_rejected_by_checkpoint_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _frozen_inputs(tmp_path)
    artifact = _build(inputs)
    frozen_selection = artifact["selection"]
    assert isinstance(frozen_selection, dict)
    forged_selection = dict(frozen_selection)
    forged_predictions = [dict(item) for item in frozen_selection["predictions"]]
    forged_predictions[0]["success_probability"] = 0.3
    forged_predictions[1]["success_probability"] = 0.7
    forged_selection["predictions"] = forged_predictions
    forged_selection["decision"] = (
        ExpectedSuccessSelector()
        .select(
            inputs["manifest"].candidates,  # type: ignore[union-attr]
            tuple(
                ValuePrediction(
                    candidate_id=item["candidate_id"],
                    candidate_fingerprint=item["candidate_fingerprint"],
                    success_probability=item["success_probability"],
                    feasible=item["feasible"],
                )
                for item in forged_predictions
            ),
        )
        .to_dict()
    )
    artifact["selection"] = forged_selection
    artifact["selection_sha256"] = canonical_sha256(forged_selection)
    body = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    artifact["artifact_sha256"] = canonical_sha256(body)
    path = tmp_path / "forged-selection.json"
    _write_json(path, artifact)

    _install_replay(monkeypatch, inputs["selection"])
    with pytest.raises(ValueError, match="checkpoint replay"):
        verify_frozen_scorer_selection(
            path,
            manifest=inputs["manifest"],  # type: ignore[arg-type]
            entry=inputs["entry"],  # type: ignore[arg-type]
        )


def test_frozen_selection_is_replayed_by_a_real_checkpoint_and_cache(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    context = FrozenTokenField(
        tokens=torch.arange(32, dtype=torch.float32).reshape(1, 4, 8),
        valid_mask=torch.ones(1, 4, dtype=torch.bool),
        current_patch_mask=torch.tensor([[True, True, False, False]]),
        camera_ids=(("agentview", "agentview", None, None),),
        frame_ids=(("agent-0", "agent-0", None, None),),
        patch_xyxy=torch.tensor(
            [[[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0], [0.0] * 4, [0.0] * 4]],
            dtype=torch.float32,
        ),
        context_fingerprints=(manifest.context.fingerprint(),),
        provider_id=manifest.token_provider_id,
    )
    field = CandidateTokenField(
        tokens=torch.arange(16, dtype=torch.float32).reshape(1, 2, 8),
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        candidate_ids=(tuple(item.candidate_id for item in manifest.candidates),),
        candidate_fingerprints=(
            tuple(item.fingerprint() for item in manifest.candidates),
        ),
        primitives=(tuple(item.primitive for item in manifest.candidates),),
        grounding_support=torch.tensor(
            [[[True, False, False, False], [False, True, False, False]]]
        ),
        public_context=context,
    )
    cache_root = tmp_path / "real-cache"
    logical_key = candidate_feature_cache_key(
        provider_id=manifest.token_provider_id,
        context=manifest.context,
        candidates=manifest.candidates,
    )
    cache_identity = QwenFeatureCache(cache_root).put(logical_key, field)
    manifest = dataclasses.replace(
        manifest, token_cache_sha256=cache_identity.artifact_sha256
    )
    schedule = build_branch_schedule((manifest,), schedule_id="real-schedule")

    torch.manual_seed(31)
    model = GroundedOutcomeModel(
        context_dim=8,
        candidate_dim=8,
        hidden_dim=8,
        num_heads=2,
        feedforward_dim=16,
        use_public_state=False,
    )
    checkpoint_identity = {
        "schema_version": "method-v1-outcome-checkpoint-v1",
        "seed": 31,
        "experiment_id": manifest.experiment_id,
        "configuration_sha256": manifest.configuration_sha256,
        "proposal_provider_id": manifest.proposal_provider_id,
        "provider_id": manifest.token_provider_id,
        "outcome_contract_sha256": manifest.outcome_contract.fingerprint(),
        "training_dataset_sha256": _digest("1"),
        "training_admission_evidence_sha256": _digest("2"),
        "collection_plan_sha256": _digest("3"),
        "scorer_verifier_auth_key_id": _digest("4"),
        "training_split_group_ids": ["disjoint-training-family"],
        "model": {
            "class": "grounded_interaction.model.GroundedOutcomeModel",
            "context_dim": 8,
            "candidate_dim": 8,
            "hidden_dim": 8,
            "num_heads": 2,
            "feedforward_dim": 16,
            "use_public_state": False,
            "public_state_dim": 9,
            "public_state_train_statistics": None,
        },
        "training": {"test_fixture": True},
        "best_epoch": 0,
        "best_validation_nll": 0.5,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }
    identity_sha256 = canonical_sha256(checkpoint_identity)
    checkpoint_dir = tmp_path / "real-checkpoint"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "best.pt"
    torch.save(
        {
            "schema_version": "method-v1-outcome-checkpoint-v1",
            "identity": checkpoint_identity,
            "identity_sha256": identity_sha256,
            "model_state_dict": model.state_dict(),
            "history": [],
        },
        checkpoint_path,
    )
    identity_path = checkpoint_dir / "identity.json"
    _write_json(
        identity_path,
        {**checkpoint_identity, "identity_sha256": identity_sha256},
    )
    output = tmp_path / "real-selection.json"
    verified, entry = create_frozen_scorer_selection_for_model_seed(
        manifest=manifest,
        schedule=schedule,
        model_seed=202,
        cache_root=cache_root,
        checkpoint_path=checkpoint_path,
        checkpoint_identity_path=identity_path,
        expected_checkpoint_sha256=_file_sha256(checkpoint_path),
        expected_checkpoint_identity_sha256=identity_sha256,
        output_path=output,
    )
    replayed = verify_frozen_scorer_selection(output, manifest=manifest, entry=entry)
    assert replayed == verified

    forged = json.loads(output.read_text(encoding="utf-8"))
    selected_id = forged["selection"]["decision"]["candidate_id"]
    unselected = next(
        item
        for item in forged["selection"]["predictions"]
        if item["candidate_id"] != selected_id
    )
    unselected["success_probability"] = float(unselected["success_probability"]) * 0.5
    forged["selection_sha256"] = canonical_sha256(forged["selection"])
    forged_body = {
        key: value for key, value in forged.items() if key != "artifact_sha256"
    }
    forged["artifact_sha256"] = canonical_sha256(forged_body)
    forged_path = tmp_path / "real-selection-forged.json"
    _write_json(forged_path, forged)
    with pytest.raises(ValueError, match="checkpoint replay"):
        verify_frozen_scorer_selection(forged_path, manifest=manifest, entry=entry)


def test_model_seed_resolution_fails_closed_without_one_matching_row(
    tmp_path: Path,
) -> None:
    inputs = _frozen_inputs(tmp_path)
    manifest = inputs["manifest"]
    assert isinstance(manifest, DecisionGroupManifest)
    selected = manifest.candidates[1]
    with pytest.raises(ValueError, match="exactly one frozen schedule entry"):
        resolve_selected_schedule_entry(
            manifest=manifest,
            schedule=inputs["schedule"],  # type: ignore[arg-type]
            candidate_id=selected.candidate_id,
            candidate_fingerprint=selected.fingerprint(),
            model_seed=999,
        )


def test_cli_rejects_outcome_augmented_schedule_before_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _frozen_inputs(tmp_path)
    manifest = inputs["manifest"]
    assert isinstance(manifest, DecisionGroupManifest)
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest.to_dict())
    schedule_path = tmp_path / "schedule.json"
    rows = [entry.to_dict() for entry in inputs["schedule"]]  # type: ignore[union-attr]
    rows[0]["observed_outcome"] = True
    _write_json(schedule_path, {"entries": rows})

    def forbidden_scorer(*args: object, **kwargs: object) -> None:
        raise AssertionError("outcome-bearing schedule reached the scorer")

    monkeypatch.setattr(
        "grounded_interaction.train_outcomes.select_manifest_candidate",
        forbidden_scorer,
    )
    with pytest.raises(ValueError, match="keys mismatch"):
        _main(
            [
                "--manifest",
                str(manifest_path),
                "--schedule",
                str(schedule_path),
                "--model-seed",
                "101",
                "--cache-root",
                str(inputs["cache_root"]),
                "--checkpoint",
                str(inputs["checkpoint"]),
                "--checkpoint-identity",
                str(inputs["identity_path"]),
                "--expected-checkpoint-sha256",
                str(inputs["checkpoint_sha256"]),
                "--expected-checkpoint-identity-sha256",
                str(inputs["identity_sha256"]),
                "--output",
                str(tmp_path / "must-not-exist.json"),
            ]
        )
    assert not (tmp_path / "must-not-exist.json").exists()
