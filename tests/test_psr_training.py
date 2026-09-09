from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from grounded_interaction.psr.model import PredictiveOutput
from grounded_interaction.psr.training import (
    CheckpointIdentity,
    CheckpointStage,
    FrozenExecutionGuard,
    StageCBatch,
    assert_disjoint_stage_c_parameters,
    assert_execution_snapshot_unchanged,
    compose_stage_a_loss,
    fit_positive_temperature,
    freeze_execution_snapshot,
    load_training_checkpoint,
    save_training_checkpoint,
    train_stage_c_step,
)
from grounded_interaction.psr.types import IntentCandidate


def _identity(*, snapshot: str | None = None) -> CheckpointIdentity:
    digest = "a" * 64
    return CheckpointIdentity(
        base_model_id="allenai/MolmoAct2-LIBERO",
        base_revision="0d24a92",
        upstream_revision="66b87e64",
        config_sha256=digest,
        split_sha256="b" * 64,
        protocol_sha256="c" * 64,
        projection_sha256="d" * 64,
        normalization_sha256="e" * 64,
        seed=17,
        execution_snapshot_id=snapshot,
    )


def _prediction(parameter: torch.Tensor | None = None) -> PredictiveOutput:
    if parameter is None:
        parameter = torch.tensor(0.1, requires_grad=True)
    return PredictiveOutput(
        failure_logits=parameter.expand(1, 2),
        mixture_logits=torch.stack((parameter, -parameter))
        .reshape(1, 1, 2)
        .expand(1, 2, 2),
        evidence_mean=parameter.expand(1, 2, 2, 3, 2),
        evidence_logstd=torch.zeros(1, 2, 2, 3, 2) + parameter * 0,
        candidate_valid_mask=torch.ones(1, 2, dtype=torch.bool),
    )


class _WarmupBackend:
    def __init__(self) -> None:
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.flow_calls = 0
        self.intent_calls = 0

    def flow_matching_loss(self, encoded, candidate, actions, **kwargs):
        assert encoded == "real-prefix"
        assert candidate.execution_route == "conditioned"
        assert actions == "real-actions"
        self.flow_calls += 1
        return self.weight.square()

    def intent_language_loss(self, encoded, token_ids):
        assert encoded == "real-prefix"
        assert tuple(token_ids) == (7, 8)
        self.intent_calls += 1
        return (self.weight - 1).square()


def test_stage_a_composes_real_backend_flow_intent_and_evidence() -> None:
    backend = _WarmupBackend()
    prediction = _prediction(backend.weight)
    result = compose_stage_a_loss(
        backend=backend,
        encoded_history="real-prefix",
        executed_intent=IntentCandidate("move the cup", (7, 8), "conditioned"),
        actions="real-actions",
        intent_target_token_ids=(7, 8),
        prediction=prediction,
        evidence_targets=torch.zeros(1, 2, 3, 2),
        evidence_valid_dimension_mask=torch.ones(1, 2, 3, dtype=torch.bool),
    )
    assert backend.flow_calls == backend.intent_calls == 1
    assert torch.allclose(
        result.total,
        result.native_flow + result.future_evidence_nll + result.intent_ce,
    )
    result.total.backward()
    assert backend.weight.grad is not None and torch.isfinite(backend.weight.grad)


def test_freeze_snapshot_detects_overlap_and_later_mutation() -> None:
    execution = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    snapshot = freeze_execution_snapshot(execution, identity=_identity())
    assert snapshot.trainable_before_freeze
    assert all(not parameter.requires_grad for parameter in execution.parameters())
    assert_execution_snapshot_unchanged(execution, snapshot)

    predictor = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(predictor.parameters(), lr=0.1)
    with FrozenExecutionGuard(execution, snapshot, optimizer):
        pass
    overlapping = torch.optim.SGD(execution.parameters(), lr=0.1)
    with pytest.raises(RuntimeError, match="overlaps"):
        assert_disjoint_stage_c_parameters(execution, overlapping)
    with torch.no_grad():
        next(execution.parameters()).add_(1)
    with pytest.raises(RuntimeError, match="values changed"):
        assert_execution_snapshot_unchanged(execution, snapshot)


class _TinyPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.2))

    def forward(self, state, ids, mask, route, valid):
        del state, ids, mask, route
        output = _prediction(self.value)
        return dataclasses.replace(output, candidate_valid_mask=valid)


def _stage_c_batch(*, supervised: bool) -> StageCBatch:
    labels = torch.full((1, 2), supervised, dtype=torch.bool)
    return StageCBatch(
        state_tokens=torch.zeros(1, 6, 4),
        intent_token_ids=torch.ones(1, 2, 2, dtype=torch.long),
        intent_token_mask=torch.ones(1, 2, 2, dtype=torch.bool),
        route_ids=torch.tensor([[0, 1]]),
        candidate_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        evidence_targets=torch.zeros(1, 2, 3, 2),
        evidence_valid_dimension_mask=torch.ones(1, 2, 3, dtype=torch.bool),
        evidence_supervised_candidates=labels,
        failure_targets=torch.tensor([[0.0, 1.0]]),
        failure_supervised_candidates=labels,
    )


def test_stage_c_skips_unlabeled_batch_without_weight_decay_update() -> None:
    execution = nn.Linear(2, 2)
    freeze_execution_snapshot(execution, identity=_identity())
    predictor = _TinyPredictor()
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=0.1, weight_decay=1.0)
    before = predictor.value.detach().clone()
    skipped = train_stage_c_step(
        predictor=predictor,
        optimizer=optimizer,
        batch=_stage_c_batch(supervised=False),
        execution_model=execution,
    )
    assert not skipped.updated
    assert skipped.skip_reason == "no_valid_evidence_or_failure_supervision"
    assert torch.equal(predictor.value.detach(), before)
    assert not optimizer.state

    updated = train_stage_c_step(
        predictor=predictor,
        optimizer=optimizer,
        batch=_stage_c_batch(supervised=True),
        execution_model=execution,
    )
    assert updated.updated and updated.total_loss is not None
    assert updated.evidence_items == updated.failure_items == 2
    assert not torch.equal(predictor.value.detach(), before)


def test_temperature_is_positive_refuses_single_class_and_preserves_order() -> None:
    logits = torch.tensor([-3.0, -1.0, 1.0, 3.0])
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
    result = fit_positive_temperature(
        logits,
        targets,
        calibration_sha256="f" * 64,
        max_iterations=30,
    )
    assert result.temperature > 0
    assert result.bce_after <= result.bce_before + 1e-10
    assert torch.equal(torch.argsort(logits), torch.argsort(result.apply(logits)))
    with pytest.raises(ValueError, match="single-class"):
        fit_positive_temperature(
            logits,
            torch.zeros_like(targets),
            calibration_sha256="f" * 64,
        )


def test_checkpoint_verifies_hash_and_identity_before_loading(tmp_path) -> None:
    torch.manual_seed(3)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    destination = tmp_path / "predictor.pt"
    identity = _identity(snapshot="1" * 64)
    saved = save_training_checkpoint(
        destination,
        stage=CheckpointStage.OUTCOME_PREDICTOR,
        model=model,
        optimizer=optimizer,
        identity=identity,
    )
    expected = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    with torch.no_grad():
        model.weight.add_(4)
    loaded = load_training_checkpoint(
        destination,
        model=model,
        optimizer=optimizer,
        expected_stage="predictor",
        expected_execution_snapshot_id="1" * 64,
        expected_file_sha256=saved.file_sha256,
    )
    assert loaded.payload_sha256 == saved.payload_sha256
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name])
    with pytest.raises(ValueError, match="file SHA-256"):
        load_training_checkpoint(
            destination,
            model=model,
            expected_file_sha256="0" * 64,
        )
