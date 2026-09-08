from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
import types
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
import pytest

import grounded_interaction.collect_outcomes as collector_module
import grounded_interaction.method_v1_runtime as runtime_module
from grounded_interaction.collect_outcomes import (
    FREEZE_RECEIPT_SCHEMA,
    SCENE_SPEC_SCHEMA,
    SCHEDULE_FILE_SCHEMA,
    SceneResetSpec,
    _claim_execution_once,
    _tree_sha256,
    _write_json_once,
    load_verified_collection_attempt,
    render_public_reset,
    run_schedule_entry,
)
from grounded_interaction.continuation import (
    BudgetedExecution,
    FixedContinuationIdentity,
    FixedHorizonExecution,
    FixedHorizonStatus,
)
from grounded_interaction.contracts import (
    ExecutionStatus,
    GroundedIntervention,
    GroundingReference,
    PolicyContext,
    Primitive,
    PublicActionEvent,
    PublicFrame,
    canonical_sha256,
)
from grounded_interaction.data import ObservedBranch
from grounded_interaction.execution import ExecutorReceipt, ExecutorRequest
from grounded_interaction.method_v1_config import (
    load_method_v1_config,
    resolve_method_v1_identity,
)
from grounded_interaction.method_v1_data import (
    BranchScheduleEntry,
    CollectionAttempt,
    CollectionAttemptStatus,
    DecisionGroupManifest,
    MethodV1OutcomeDataset,
    build_branch_schedule,
    make_method_v1_outcome_contract,
    validate_branch_schedule,
)
from grounded_interaction.method_v1_runtime import (
    LiberoMolmoBudgetedExecutor,
    PolicyExecutionFailure,
    method_v1_executor_id,
    public_context_from_observation,
)
from grounded_interaction.method_v1_trace import validate_method_v1_execution_trace
from grounded_interaction.molmoact2 import (
    MolmoAct2ActionChunk,
    MolmoAct2PolicyOutputError,
    MolmoAct2ServerIdentity,
)
from grounded_interaction.proposals import (
    proposal_request_text,
    qwen_proposal_prompt_contract,
)
from grounded_interaction.qwen_provider import QwenProposalGeneration
from grounded_interaction.qwen_service import (
    QwenProposalHTTPClient,
    _QwenServer,
)
from grounded_interaction.rgb import RGBFrameStore, canonical_rgb_sha256
from grounded_interaction.serialization import (
    GroundedTextSerializer,
    SpatialConditioningMode,
)


def _digest(character: str) -> str:
    return character * 64


def _frame(frame_id: str, camera: str, index: int, character: str) -> PublicFrame:
    return PublicFrame(
        frame_id=frame_id,
        camera=camera,
        frame_index=index,
        image_sha256=_digest(character),
        width=256,
        height=256,
    )


def _context(suffix: str) -> PolicyContext:
    return PolicyContext(
        prompt="Put the butter in the basket.",
        frames=(
            _frame(f"agent-{suffix}", "agentview", 0, suffix),
            _frame(f"wrist-{suffix}", "wrist", 0, chr(ord(suffix) + 1)),
        ),
        proprioception=(0.0,) * 8,
    )


def _candidate(context: PolicyContext, candidate_id: str) -> GroundedIntervention:
    frame = context.frames[0]
    return GroundedIntervention(
        candidate_id=candidate_id,
        primitive=Primitive.DIRECT,
        referent="butter package",
        parameters=(),
        grounding=GroundingReference(
            camera=frame.camera,
            frame_id=frame.frame_id,
            frame_index=frame.frame_index,
            image_sha256=frame.image_sha256,
            box_xyxy=(0.2, 0.2, 0.5, 0.6),
            point_xy=(0.35, 0.4),
        ),
    )


def _manifest(suffix: str) -> DecisionGroupManifest:
    context = _context(suffix)
    return DecisionGroupManifest(
        manifest_id=f"manifest-{suffix}",
        experiment_id="method-v1-development",
        initial_state_group=f"state-{suffix}",
        decision_group_id=f"decision-{suffix}",
        split_group_id=f"split-group-{suffix}",
        split="train",
        information_stratum="INFORMATION_NECESSARY",
        scene_id=f"scene-{suffix}",
        layout_id=f"layout-{suffix}",
        reset_state_sha256=_digest(chr(ord(suffix) + 2)),
        context=context,
        candidates=(_candidate(context, f"direct-{suffix}"),),
        outcome_contract=make_method_v1_outcome_contract(
            continuation_policy_id=_digest("c"),
            executor_id="executor-v1",
            serializer_id="grounded-precise-text-v1",
        ),
        proposal_provider_id="qwen-proposer-v1",
        proposal_request_sha256=_digest("d"),
        token_provider_id="qwen-token-provider-v1",
        token_cache_sha256=_digest("e"),
        configuration_sha256=_digest("f"),
        model_seeds=(11, 22),
    )


def _schedule_file(manifest: DecisionGroupManifest) -> dict[str, object]:
    schedule = build_branch_schedule(
        (manifest,), schedule_id=f"{manifest.decision_group_id}:branches"
    )
    body = {
        "schema_version": SCHEDULE_FILE_SCHEMA,
        "schedule_id": schedule[0].schedule_id,
        "manifest_sha256": manifest.fingerprint(),
        "entries": [row.to_dict() for row in schedule],
    }
    return {**body, "schedule_sha256": canonical_sha256(body)}


def test_scene_spec_rejects_non_digest_and_nonpaired_seeds() -> None:
    base = {
        "scene_id": "drawer",
        "layout_id": "layout-a",
        "initial_state_group": "state-a",
        "decision_group_id": "decision-a",
        "split_group_id": "family-a",
        "split": "train",
        "information_stratum": "INFORMATION_NECESSARY",
        "prompt": "Put the butter in the basket.",
        "bddl_file": "scene.bddl",
        "init_states_file": "states.pt",
        "init_state_index": 0,
        "env_seed": 7,
        "reset_state_sha256": _digest("a"),
        "evaluator_predicate": ["In", "butter_1", "basket_1"],
        "model_seeds": [101, 202],
        "image_size": 256,
        "settle_steps": 10,
        "control_mode": "relative",
        "schema_version": SCENE_SPEC_SCHEMA,
    }
    assert SceneResetSpec.from_mapping(base).model_seeds == (101, 202)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        SceneResetSpec.from_mapping({**base, "reset_state_sha256": "z" * 64})
    with pytest.raises(ValueError, match="exactly two"):
        SceneResetSpec.from_mapping({**base, "model_seeds": [101]})
    with pytest.raises(ValueError, match="unique"):
        SceneResetSpec.from_mapping({**base, "model_seeds": [101, 101]})


def test_dataset_schema_aggregates_multiple_prefrozen_local_schedules() -> None:
    # Canonical receipt-backed admission, including the prospective global
    # plan, is tested in test_method_v1_data.py.  This compact schema test
    # isolates the fact that schedule-local execution indices may repeat
    # across decision freezes without colliding.
    manifests = (_manifest("a"), _manifest("d"))
    schedules = tuple(_schedule_file(manifest) for manifest in manifests)
    rows = tuple(
        row
        for schedule in schedules
        for row in (
            BranchScheduleEntry.from_mapping(item) for item in schedule["entries"]
        )
    )
    assert [row.execution_index for row in rows] == [0, 1, 0, 1]
    validate_branch_schedule(manifests, rows)

    attempts: list[CollectionAttempt] = []
    for index, (manifest, schedule) in enumerate(
        zip(manifests, schedules, strict=True)
    ):
        row = rows[index * 2]
        post = _frame(f"post-{index}", "agentview", 1, str(index + 1))
        branch = ObservedBranch(
            branch_id=row.entry_id,
            initial_state_group=row.initial_state_group,
            decision_group_id=row.decision_group_id,
            split=row.split,
            reset_state_sha256=row.reset_state_sha256,
            repeat_index=row.repeat_index,
            context=manifest.context,
            executed_intervention=manifest.candidate(row.candidate_id),
            post_action_frames=(post,),
            outcome_contract=manifest.outcome_contract,
            observed_outcome=bool(index % 2 == 0),
            execution_status=ExecutionStatus.COMPLETED,
            execution_receipt_id=f"receipt-{index}",
            diagnostics={"control_steps_used": 300},
        )
        attempts.append(
            CollectionAttempt(
                attempt_id=row.entry_id,
                schedule_entry=row,
                status=CollectionAttemptStatus.OUTCOME_EVALUATED,
                artifact_tree_sha256=_digest(str(index + 1)),
                branch=branch,
            )
        )
    dataset = MethodV1OutcomeDataset(
        dataset_id="method-v1-two-resets",
        manifests=manifests,
        schedule=rows,
        attempts=tuple(attempts),
        complete=False,
    )
    loaded = MethodV1OutcomeDataset.from_mapping(dataset.to_dict())
    assert len(loaded.manifests) == 2
    assert len(loaded.schedule) == 4
    assert len(loaded.observed_branches) == 2


def test_execution_claim_is_single_use_even_if_output_root_changes(
    tmp_path: Path,
) -> None:
    manifest = _manifest("a")
    row = build_branch_schedule((manifest,), schedule_id="local-schedule")[0]
    freeze_dir = tmp_path / "freeze"
    first = _claim_execution_once(
        freeze_dir=freeze_dir,
        output_root=tmp_path / "out-a",
        entry=row,
        collection_plan_sha256=_digest("8"),
        scorer_verifier_auth_key_id=_digest("9"),
    )
    assert first.is_file()
    with pytest.raises(FileExistsError):
        _claim_execution_once(
            freeze_dir=freeze_dir,
            output_root=tmp_path / "out-b",
            entry=row,
            collection_plan_sha256=_digest("8"),
            scorer_verifier_auth_key_id=_digest("9"),
        )


class _FakeIdentity:
    def to_dict(self) -> dict[str, str]:
        return {"kind": "fake-public-qwen"}


class _FakeQwenRuntime:
    provider_id = "fake-qwen-provider"
    identity = _FakeIdentity()

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_proposal_json(self, **kwargs: object) -> QwenProposalGeneration:
        self.calls.append(dict(kwargs))
        enabled = tuple(kwargs["enabled_primitives"])
        max_candidates = int(kwargs["max_candidates"])
        max_per_primitive = int(kwargs["max_per_primitive"])
        contract = qwen_proposal_prompt_contract(
            enabled_primitives=enabled,
            max_candidates=max_candidates,
            max_per_primitive=max_per_primitive,
        )
        user_text = proposal_request_text(
            task_prompt=str(kwargs["task_prompt"]),
            public_history_text=str(kwargs["public_history_text"]),
            camera_label=str(kwargs["camera_label"]),
            frame_id=str(kwargs["frame_id"]),
            processed_width=32,
            processed_height=24,
            enabled_primitives=enabled,
            max_candidates=max_candidates,
            max_per_primitive=max_per_primitive,
        )
        return QwenProposalGeneration(
            raw_json='{"candidates":[]}',
            processed_width=32,
            processed_height=24,
            prompt_contract_sha256=contract.fingerprint,
            rendered_user_prompt_sha256=hashlib.sha256(
                user_text.encode("utf-8")
            ).hexdigest(),
        )


def test_qwen_http_boundary_transports_public_history_and_rgb_only() -> None:
    fake = _FakeQwenRuntime()
    server = _QwenServer(("127.0.0.1", 0), runtime=fake)  # type: ignore[arg-type]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        client = QwenProposalHTTPClient(
            f"http://{host}:{port}", expected_provider_id=fake.provider_id
        )
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        generation = client.generate_proposal_json(
            task_prompt="Put the butter in the basket.",
            public_history_text=(
                "Completed public action history:\n"
                "0. OPEN: Open the middle drawer.; status=COMPLETED"
            ),
            image=image,
            camera_label="agentview",
            frame_id="agent-post-open",
            enabled_primitives=(Primitive.DIRECT,),
            max_candidates=3,
            max_per_primitive=3,
        )
        assert generation.processed_width == 32
        assert len(fake.calls) == 1
        assert set(fake.calls[0]) == {
            "task_prompt",
            "public_history_text",
            "image",
            "camera_label",
            "frame_id",
            "enabled_primitives",
            "max_candidates",
            "max_per_primitive",
        }
        assert fake.calls[0]["enabled_primitives"] == (Primitive.DIRECT,)
        assert canonical_rgb_sha256(fake.calls[0]["image"]) == canonical_rgb_sha256(
            image
        )
        assert "evaluator" not in json.dumps(
            {key: value for key, value in fake.calls[0].items() if key != "image"}
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@dataclasses.dataclass
class _FakeObservation:
    agentview_rgb: np.ndarray
    wrist_rgb: np.ndarray
    state: tuple[float, ...]


class _FakeEnvironment:
    def __init__(self) -> None:
        self.step_count = 0
        self.public_observation = _FakeObservation(
            agentview_rgb=np.zeros((16, 16, 3), dtype=np.uint8),
            wrist_rgb=np.ones((16, 16, 3), dtype=np.uint8),
            state=(0.0,) * 8,
        )

    def step(self, action: object) -> tuple[_FakeObservation, tuple[float, ...]]:
        values = tuple(float(item) for item in action)  # type: ignore[arg-type]
        self.step_count += 1
        level = min(self.step_count, 255)
        self.public_observation = _FakeObservation(
            agentview_rgb=np.full((16, 16, 3), level, dtype=np.uint8),
            wrist_rgb=np.full((16, 16, 3), min(level + 1, 255), dtype=np.uint8),
            state=(float(self.step_count),) + (0.0,) * 7,
        )
        return self.public_observation, values


class _FakeMolmoClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.expected_identity = MolmoAct2ServerIdentity()
        self.fail = fail
        self.reset_ids: list[str] = []
        self.seeds: list[int] = []

    def reset(self, session_id: str) -> None:
        self.reset_ids.append(session_id)

    def predict_action_chunk(self, **kwargs: object) -> MolmoAct2ActionChunk:
        self.seeds.append(int(kwargs["seed"]))
        if self.fail:
            raise MolmoAct2PolicyOutputError("malformed action tokens")
        return MolmoAct2ActionChunk(
            request_id=canonical_sha256(
                {"seed": kwargs["seed"], "call": len(self.seeds)}
            ),
            server_identity_sha256=self.expected_identity.digest,
            actions=tuple((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0) for _ in range(10)),
            latency_ms=1,
        )


def _runtime_candidate(context: PolicyContext) -> GroundedIntervention:
    frame = context.frames[0]
    return GroundedIntervention(
        candidate_id="open-middle",
        primitive=Primitive.OPEN,
        referent="middle drawer",
        parameters=(),
        grounding=GroundingReference(
            camera=frame.camera,
            frame_id=frame.frame_id,
            frame_index=frame.frame_index,
            image_sha256=frame.image_sha256,
            box_xyxy=(0.2, 0.2, 0.8, 0.8),
            point_xy=(0.5, 0.5),
        ),
    )


def test_libero_molmo_runtime_applies_exact_chunks_and_reobserves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_module, "LiberoE1Environment", _FakeEnvironment)
    monkeypatch.setattr(runtime_module, "MolmoAct2HTTPClient", _FakeMolmoClient)
    environment = _FakeEnvironment()
    client = _FakeMolmoClient()
    store = RGBFrameStore(tmp_path / "frames")
    context = public_context_from_observation(
        prompt="Put the butter in the basket.",
        observation=environment.public_observation,  # type: ignore[arg-type]
        frame_store=store,
        frame_namespace="initial",
        frame_index=0,
    )
    executor = LiberoMolmoBudgetedExecutor(
        environment=environment,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        frame_store=store,
        branch_namespace="branch-a",
    )
    result = executor.execute_candidate(
        _runtime_candidate(context),
        context,
        control_step_budget=20,
        model_seed=7,
    )
    assert result.control_steps_used == 20
    assert environment.step_count == 20
    assert client.seeds == [7, 8]
    assert (
        result.next_context.public_history[-1].execution_status
        is ExecutionStatus.COMPLETED
    )
    assert len(result.receipt.post_frames) == 2
    assert len(executor.public_traces) == 1


def test_libero_molmo_runtime_exposes_policy_failure_for_final_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_module, "LiberoE1Environment", _FakeEnvironment)
    monkeypatch.setattr(runtime_module, "MolmoAct2HTTPClient", _FakeMolmoClient)
    environment = _FakeEnvironment()
    client = _FakeMolmoClient(fail=True)
    store = RGBFrameStore(tmp_path / "frames")
    context = public_context_from_observation(
        prompt="Put the butter in the basket.",
        observation=environment.public_observation,  # type: ignore[arg-type]
        frame_store=store,
        frame_namespace="initial",
        frame_index=0,
    )
    executor = LiberoMolmoBudgetedExecutor(
        environment=environment,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        frame_store=store,
        branch_namespace="branch-failure",
    )
    with pytest.raises(PolicyExecutionFailure) as captured:
        executor.execute_candidate(
            _runtime_candidate(context),
            context,
            control_step_budget=20,
            model_seed=3,
        )
    failure = captured.value
    assert failure.control_steps_used == 0
    assert failure.model_calls == 1
    assert (
        failure.final_context.public_history[-1].execution_status
        is ExecutionStatus.FAILED
    )
    assert len(failure.final_context.frames) == len(context.frames) + 2


def test_execution_trace_validator_recomputes_real_chunk_and_receipt_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_module, "LiberoE1Environment", _FakeEnvironment)
    monkeypatch.setattr(runtime_module, "MolmoAct2HTTPClient", _FakeMolmoClient)
    attempt_dir = tmp_path / "attempt"
    store = RGBFrameStore(attempt_dir / "public_frames")
    environment = _FakeEnvironment()
    client = _FakeMolmoClient()
    context = public_context_from_observation(
        prompt="Put the butter in the basket.",
        observation=environment.public_observation,  # type: ignore[arg-type]
        frame_store=store,
        frame_namespace="trace:initial",
        frame_index=0,
    )
    open_candidate = _runtime_candidate(context)
    candidate = dataclasses.replace(
        open_candidate,
        candidate_id="direct-butter",
        primitive=Primitive.DIRECT,
        referent="visible butter package",
    )
    manifest = DecisionGroupManifest(
        manifest_id="trace-manifest",
        experiment_id="trace-integration",
        initial_state_group="trace-state",
        decision_group_id="trace-decision",
        split_group_id="trace-family",
        split="test",
        information_stratum="INFORMATION_NECESSARY",
        scene_id="trace-scene",
        layout_id="trace-layout",
        reset_state_sha256=_digest("8"),
        context=context,
        candidates=(candidate,),
        outcome_contract=make_method_v1_outcome_contract(
            continuation_policy_id=_digest("7"),
            executor_id=method_v1_executor_id(client.expected_identity.digest),
            serializer_id="grounded-precise-text-v1",
        ),
        proposal_provider_id="trace-proposer",
        proposal_request_sha256=_digest("6"),
        token_provider_id="trace-token-provider",
        token_cache_sha256=_digest("5"),
        configuration_sha256=_digest("4"),
        model_seeds=(3, 4),
    )
    entry = build_branch_schedule((manifest,), schedule_id="trace-schedule")[0]
    executor = LiberoMolmoBudgetedExecutor(
        environment=environment,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        frame_store=store,
        branch_namespace=entry.entry_id,
    )
    stage = executor.execute_candidate(
        candidate,
        context,
        control_step_budget=300,
        model_seed=entry.model_seed,
    )
    fixed = FixedHorizonExecution(
        continuation_policy_id=manifest.outcome_contract.continuation_policy_id,
        model_seed=entry.model_seed,
        selected_candidate_id=candidate.candidate_id,
        selected_candidate_fingerprint=candidate.fingerprint(),
        status=FixedHorizonStatus.DIRECT_EXECUTED,
        stages=(stage,),
        continuation_candidate_ids=(),
        continuation_candidate_fingerprints=(),
        final_context=stage.next_context,
    )
    trace = {
        "fixed_horizon": fixed.to_dict(),
        "stage_traces": dict(executor.public_traces),
        "continuation_proposal_audit": [],
    }
    _write_json_once(attempt_dir / "public_execution_trace.json", trace)
    branch = ObservedBranch(
        branch_id=entry.entry_id,
        initial_state_group=entry.initial_state_group,
        decision_group_id=entry.decision_group_id,
        split=entry.split,
        reset_state_sha256=entry.reset_state_sha256,
        repeat_index=entry.repeat_index,
        context=context,
        executed_intervention=candidate,
        post_action_frames=stage.next_context.frames[len(context.frames) :],
        outcome_contract=manifest.outcome_contract,
        observed_outcome=True,
        execution_status=ExecutionStatus.COMPLETED,
        execution_receipt_id=canonical_sha256([stage.receipt.receipt_id]),
        diagnostics={
            "fixed_horizon_status": FixedHorizonStatus.DIRECT_EXECUTED.value,
            "control_steps_used": 300,
            "model_calls": 30,
            "policy_output_failure": False,
            "continuation_abstained": False,
            "continuation_proposal_failure": False,
            "selection_sha256": None,
        },
    )
    attempt = CollectionAttempt(
        attempt_id=entry.entry_id,
        schedule_entry=entry,
        status=CollectionAttemptStatus.OUTCOME_EVALUATED,
        artifact_tree_sha256=_digest("3"),
        branch=branch,
    )
    assert validate_method_v1_execution_trace(
        attempt_dir=attempt_dir, attempt=attempt
    ) == canonical_sha256(trace)

    _write_json_once(
        attempt_dir / "observed_branch.public.json",
        branch.to_dict(include_private=False),
    )
    _write_json_once(
        attempt_dir / "private" / "evaluator_sidecar.json",
        {
            "evaluator_predicate": ["In", "butter_1", "basket_1"],
            "final_task_success": True,
            "policy_input_changed_by_evaluator": False,
        },
    )
    _write_json_once(
        attempt_dir / "started.json",
        {
            "status": "STARTED",
            "entry": entry.to_dict(),
            "manifest_sha256": manifest.fingerprint(),
            "selection_sha256": None,
            "collection_plan_sha256": _digest("8"),
            "scorer_verifier_auth_key_id": _digest("9"),
        },
    )
    sealed = dataclasses.replace(
        attempt, artifact_tree_sha256=_tree_sha256(attempt_dir)
    )
    attempt_path = attempt_dir / "attempt.json"
    _write_json_once(attempt_path, sealed.to_dict(include_private_branch=False))
    assert load_verified_collection_attempt(attempt_path) == sealed

    trace_path = attempt_dir / "public_execution_trace.json"
    tampered = json.loads(trace_path.read_text(encoding="utf-8"))
    stage_trace = next(iter(tampered["stage_traces"].values()))
    stage_trace["chunks"][0]["input_state_sha256"] = _digest("0")
    trace_path.write_text(json.dumps(tampered), encoding="utf-8")
    resealed = dataclasses.replace(
        sealed,
        artifact_tree_sha256=_tree_sha256(attempt_dir, exclude=("attempt.json",)),
    )
    attempt_path.write_text(
        json.dumps(resealed.to_dict(include_private_branch=False)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="stage trace key|state digest"):
        load_verified_collection_attempt(attempt_path)


class _CollectorEnvironment:
    predicate_steps: ClassVar[list[int]] = []

    def __init__(self, **_: object) -> None:
        self.step_count = 0
        self.public_observation = _FakeObservation(
            agentview_rgb=np.zeros((16, 16, 3), dtype=np.uint8),
            wrist_rgb=np.ones((16, 16, 3), dtype=np.uint8),
            state=(0.0,) * 8,
        )

    def advance(self, steps: int) -> None:
        self.step_count += steps
        self.public_observation = _FakeObservation(
            agentview_rgb=np.full((16, 16, 3), 31, dtype=np.uint8),
            wrist_rgb=np.full((16, 16, 3), 47, dtype=np.uint8),
            state=(float(steps),) + (0.0,) * 7,
        )

    def predicate(self, _: tuple[str, ...]) -> bool:
        self.predicate_steps.append(self.step_count)
        return self.step_count == 300

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class _CollectorMolmoClient:
    def __init__(self, _: str, *, expected_identity: MolmoAct2ServerIdentity) -> None:
        self.expected_identity = expected_identity

    def health(self) -> dict[str, str]:
        return {"status": "ok"}


class _CollectorQwenClient:
    def __init__(self, _: str, *, expected_provider_id: str) -> None:
        self.provider_id = expected_provider_id

    def ready(self) -> dict[str, object]:
        return {"status": "ok", "model_loaded": True}


class _CollectorScorerClient:
    calls: ClassVar[list[str]] = []
    reject: ClassVar[bool] = False
    training_key_override: ClassVar[str | None] = None

    def __init__(self, _: str, *, auth_key_path: Path) -> None:
        assert auth_key_path.is_file()
        self.auth_key_id = hashlib.sha256(auth_key_path.read_bytes()).hexdigest()

    def health(self) -> dict[str, object]:
        return {"status": "ok"}

    def verify(
        self,
        *,
        selection_path: Path,
        manifest_path: Path,
        schedule_path: Path,
        entry_id: str,
    ) -> dict[str, object]:
        assert selection_path.is_file()
        assert manifest_path.is_file()
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        row = next(item for item in schedule["entries"] if item["entry_id"] == entry_id)
        self.calls.append(entry_id)
        if self.reject:
            raise ValueError("learned selection rejected before execution claim")
        return {
            "artifact_sha256": _digest("e"),
            "selection_sha256": _digest("f"),
            "checkpoint_sha256": _digest("1"),
            "checkpoint_identity_sha256": _digest("2"),
            "training_dataset_sha256": _digest("3"),
            "training_admission_evidence_sha256": _digest("4"),
            "training_collection_plan_sha256": _digest("5"),
            "scorer_verifier_auth_key_id": (
                self.training_key_override or self.auth_key_id
            ),
            "candidate_id": row["candidate_id"],
            "candidate_fingerprint": row["candidate_fingerprint"],
            "temperature": 1.0,
        }


class _CollectorBudgetedExecutor:
    budgets: ClassVar[list[int]] = []

    def __init__(
        self,
        *,
        environment: _CollectorEnvironment,
        client: _CollectorMolmoClient,
        frame_store: RGBFrameStore,
        branch_namespace: str,
    ) -> None:
        self.environment = environment
        self.client = client
        self.frame_store = frame_store
        self.branch_namespace = branch_namespace
        self.public_traces: dict[str, dict[str, object]] = {}

    @property
    def executor_id(self) -> str:
        return method_v1_executor_id(self.client.expected_identity.digest)

    @property
    def serializer_id(self) -> str:
        return "grounded-precise-text-v1"

    @property
    def replan_interval(self) -> int:
        return 10

    def execute_candidate(
        self,
        candidate: GroundedIntervention,
        context: PolicyContext,
        *,
        control_step_budget: int,
        model_seed: int,
    ) -> BudgetedExecution:
        self.budgets.append(control_step_budget)
        serializer = GroundedTextSerializer(
            self.serializer_id,
            spatial_mode=SpatialConditioningMode.PRECISE_TEXT,
        )
        request = ExecutorRequest.from_serialized(
            serializer.serialize(candidate, context)
        )
        self.environment.advance(control_step_budget)
        event = PublicActionEvent(
            step_index=len(context.public_history),
            primitive=candidate.primitive,
            subtask_text=request.subtask_text,
            execution_status=ExecutionStatus.COMPLETED,
        )
        next_context = public_context_from_observation(
            prompt=context.prompt,
            observation=self.environment.public_observation,  # type: ignore[arg-type]
            frame_store=self.frame_store,
            frame_namespace=f"{self.branch_namespace}:post",
            frame_index=max(frame.frame_index for frame in context.frames) + 1,
            previous=context,
            event=event,
        )
        post_frames = next_context.frames[len(context.frames) :]
        receipt = ExecutorReceipt(
            receipt_id=canonical_sha256(
                {
                    "candidate": candidate.fingerprint(),
                    "budget": control_step_budget,
                    "seed": model_seed,
                }
            ),
            executor_id=self.executor_id,
            candidate_id=candidate.candidate_id,
            candidate_fingerprint=candidate.fingerprint(),
            request_digest=request.request_digest,
            status=ExecutionStatus.COMPLETED,
            post_frames=post_frames,
        )
        trace = {
            "candidate": candidate.fingerprint(),
            "budget": control_step_budget,
            "seed": model_seed,
        }
        trace_sha256 = canonical_sha256(trace)
        self.public_traces[trace_sha256] = trace
        return BudgetedExecution(
            request=request,
            receipt=receipt,
            previous_context=context,
            next_context=next_context,
            requested_control_steps=control_step_budget,
            control_steps_used=control_step_budget,
            model_seed=model_seed,
            public_trace_sha256=trace_sha256,
        )


@pytest.mark.parametrize(
    "selection_mode", ["none", "verified", "rejected", "wrong_training_key"]
)
def test_collector_direct_branch_runs_once_for_300_then_evaluates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection_mode: str,
) -> None:
    bddl = tmp_path / "scene.bddl"
    states = tmp_path / "states.pt"
    bddl.write_text("fake public scene", encoding="utf-8")
    states.write_bytes(b"fake exact reset")
    reset_digest = _digest("a")
    decision_group = "decision-live"
    spec = SceneResetSpec(
        scene_id="scene-live",
        layout_id="layout-live",
        initial_state_group="state-live",
        decision_group_id=decision_group,
        split_group_id="family-live",
        split="train",
        information_stratum="INFORMATION_NECESSARY",
        prompt="Put the butter in the basket.",
        bddl_file=str(bddl),
        init_states_file=str(states),
        init_state_index=0,
        env_seed=17,
        reset_state_sha256=reset_digest,
        evaluator_predicate=("In", "butter_1", "basket_1"),
        model_seeds=(101, 202),
    )
    spec_path = tmp_path / "scene_spec.json"
    _write_json_once(spec_path, spec.to_dict())

    initial_observation = _CollectorEnvironment().public_observation
    preparation_store = RGBFrameStore(tmp_path / "prepared-frames")
    context = public_context_from_observation(
        prompt=spec.prompt,
        observation=initial_observation,  # type: ignore[arg-type]
        frame_store=preparation_store,
        frame_namespace=f"{decision_group}:initial",
        frame_index=0,
    )
    frame = context.frames[0]
    candidate = GroundedIntervention(
        candidate_id="direct-visible-butter",
        primitive=Primitive.DIRECT,
        referent="visible butter package",
        parameters=(),
        grounding=GroundingReference(
            camera=frame.camera,
            frame_id=frame.frame_id,
            frame_index=frame.frame_index,
            image_sha256=frame.image_sha256,
            box_xyxy=(0.2, 0.2, 0.6, 0.7),
            point_xy=(0.4, 0.45),
        ),
    )
    executor_identity = MolmoAct2ServerIdentity()
    config = load_method_v1_config(
        Path(__file__).resolve().parents[1] / "experiments" / "method_v1.yaml"
    )
    resolved = resolve_method_v1_identity(config, executor_identity=executor_identity)
    continuation = FixedContinuationIdentity(**resolved["continuation"]["identity"])
    manifest = DecisionGroupManifest(
        manifest_id="manifest-live",
        experiment_id=config["experiment_id"],
        initial_state_group=spec.initial_state_group,
        decision_group_id=decision_group,
        split_group_id=spec.split_group_id,
        split=spec.split,
        information_stratum=spec.information_stratum,
        scene_id=spec.scene_id,
        layout_id=spec.layout_id,
        reset_state_sha256=reset_digest,
        context=context,
        candidates=(candidate,),
        outcome_contract=continuation.outcome_contract(),
        proposal_provider_id="initial-public-proposer",
        proposal_request_sha256=_digest("b"),
        token_provider_id="initial-public-token-provider",
        token_cache_sha256=_digest("c"),
        configuration_sha256=resolved["resolved_identity_sha256"],
        model_seeds=spec.model_seeds,
    )
    schedule_file = _schedule_file(manifest)
    freeze_dir = tmp_path / "freeze"
    _write_json_once(freeze_dir / "decision_manifest.json", manifest.to_dict())
    _write_json_once(freeze_dir / "branch_schedule.json", schedule_file)
    _write_json_once(freeze_dir / "resolved_method_v1.json", resolved)
    freeze_body = {
        "schema_version": FREEZE_RECEIPT_SCHEMA,
        "scene_spec_sha256": canonical_sha256(spec.to_dict()),
        "prepared_sha256": _digest("d"),
        "bddl_file_sha256": collector_module._file_sha256(bddl),
        "init_states_file_sha256": collector_module._file_sha256(states),
        "manifest_sha256": manifest.fingerprint(),
        "schedule_sha256": schedule_file["schedule_sha256"],
        "resolved_identity_sha256": resolved["resolved_identity_sha256"],
        "token_cache_key": manifest.token_cache_sha256,
    }
    _write_json_once(
        freeze_dir / "freeze_receipt.json",
        {
            **freeze_body,
            "freeze_receipt_sha256": canonical_sha256(freeze_body),
        },
    )

    _CollectorEnvironment.predicate_steps = []
    _CollectorBudgetedExecutor.budgets = []
    monkeypatch.setattr(collector_module, "LiberoE1Environment", _CollectorEnvironment)
    monkeypatch.setattr(collector_module, "MolmoAct2HTTPClient", _CollectorMolmoClient)
    monkeypatch.setattr(
        collector_module, "QwenProposalHTTPClient", _CollectorQwenClient
    )
    monkeypatch.setattr(
        collector_module,
        "ScorerVerificationHTTPClient",
        _CollectorScorerClient,
    )
    monkeypatch.setattr(
        collector_module,
        "LiberoMolmoBudgetedExecutor",
        _CollectorBudgetedExecutor,
    )
    # This integration test exercises the post-freeze execution boundary.  A
    # separate test suite validates the full current freeze receipt and cache
    # provenance; keep this synthetic legacy fixture focused on exactly-once
    # execution and evaluator timing.
    monkeypatch.setattr(collector_module, "_validate_freeze_receipt", lambda **_: {})
    auth_key_path = tmp_path / "scorer-verifier.key"
    auth_key_path.write_bytes(b"method-v1-test-authentication-key")
    auth_key_id = hashlib.sha256(auth_key_path.read_bytes()).hexdigest()
    collection_plan = types.SimpleNamespace(
        collection_plan_sha256=_digest("9"),
        scorer_verifier_auth_key_id=auth_key_id,
        freezes=(types.SimpleNamespace(freeze_dir=freeze_dir.resolve()),),
        document={
            "groups": [
                {
                    "manifest_id": manifest.manifest_id,
                    "entry_ids": [
                        item["entry_id"] for item in schedule_file["entries"]
                    ],
                }
            ]
        },
    )
    monkeypatch.setattr(
        collector_module,
        "load_verified_collection_plan",
        lambda *_, **__: collection_plan,
    )
    output_root = tmp_path / "outcomes"
    _CollectorScorerClient.calls = []
    _CollectorScorerClient.reject = selection_mode == "rejected"
    _CollectorScorerClient.training_key_override = (
        _digest("0") if selection_mode == "wrong_training_key" else None
    )
    selection_path = None
    if selection_mode != "none":
        selection_path = tmp_path / "selection.json"
        selection_path.write_text("{}\n", encoding="utf-8")
    run_arguments = {
        "scene_spec_path": spec_path,
        "freeze_dir": freeze_dir,
        "collection_plan_path": tmp_path / "collection-plan.json",
        "collection_freeze_dirs": (freeze_dir,),
        "collection_config_path": (
            Path(__file__).resolve().parents[1] / "experiments" / "method_v1.yaml"
        ),
        "scorer_auth_key_path": auth_key_path,
        "execution_index": 0,
        "molmo_endpoint": "http://molmo.invalid",
        "qwen_endpoint": "http://qwen.invalid",
        "scorer_endpoint": "http://scorer.invalid",
        "output_root": output_root,
        "selection_path": selection_path,
    }
    if selection_mode == "rejected":
        with pytest.raises(ValueError, match="before execution claim"):
            run_schedule_entry(**run_arguments)
        assert _CollectorScorerClient.calls == [schedule_file["entries"][0]["entry_id"]]
        assert not (freeze_dir / "execution_claims").exists()
        assert _CollectorBudgetedExecutor.budgets == []
        assert _CollectorEnvironment.predicate_steps == []
        return
    if selection_mode == "wrong_training_key":
        with pytest.raises(ValueError, match="different scorer-verifier"):
            run_schedule_entry(**run_arguments)
        assert _CollectorScorerClient.calls == [schedule_file["entries"][0]["entry_id"]]
        assert not (freeze_dir / "execution_claims").exists()
        assert _CollectorBudgetedExecutor.budgets == []
        assert _CollectorEnvironment.predicate_steps == []
        return
    attempt = run_schedule_entry(
        **run_arguments,
    )
    assert attempt.status is CollectionAttemptStatus.OUTCOME_EVALUATED
    assert attempt.branch is not None and attempt.branch.observed_outcome is True
    assert _CollectorBudgetedExecutor.budgets == [300]
    assert _CollectorEnvironment.predicate_steps == [300]
    expected_selection = _digest("e") if selection_mode == "verified" else None
    assert attempt.branch.diagnostics["selection_sha256"] == expected_selection
    assert _CollectorScorerClient.calls == (
        [attempt.schedule_entry.entry_id] if selection_mode == "verified" else []
    )

    with pytest.raises(FileExistsError):
        run_schedule_entry(
            **{
                **run_arguments,
                "output_root": tmp_path / "different-output-root",
            },
        )


@pytest.mark.skipif(
    not os.environ.get("METHOD_V1_LIVE_SCENE_SPEC"),
    reason="real LIBERO Method-V1 scene is opt-in",
)
def test_live_method_v1_public_reset_boundary(tmp_path: Path) -> None:
    scene_spec = Path(os.environ["METHOD_V1_LIVE_SCENE_SPEC"]).resolve()
    result = render_public_reset(
        scene_spec=scene_spec,
        output_dir=tmp_path / "prepared-live-reset",
    )
    assert result["schema_version"] == "method-v1-prepared-public-reset-v1"
    assert "evaluator_predicate" not in json.dumps(result)
