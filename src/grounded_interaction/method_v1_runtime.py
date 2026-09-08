"""Concrete MolmoAct2/LIBERO execution adapter for Method-V1.

The adapter reuses the pinned HTTP client, official LIBERO action
normalization, precise grounded serializer, and content-addressed RGB store.
It applies real action chunks to one live exact-reset environment. Private
predicates are absent; callers evaluate the final state only after public
execution returns.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from .continuation import BudgetedExecution
from .contracts import (
    ExecutionStatus,
    GroundedIntervention,
    PolicyContext,
    PublicActionEvent,
    canonical_sha256,
)
from .execution import ExecutorReceipt, ExecutorRequest
from .libero_runtime import (
    LiberoE1Environment,
    LiberoPublicObservation,
    canonical_libero_state_sha256,
)
from .molmoact2 import (
    MolmoAct2HTTPClient,
    MolmoAct2PolicyOutputError,
    molmoact2_public_request_payload,
)
from .rgb import RGBFrameStore, canonical_rgb_sha256
from .serialization import GroundedTextSerializer, SpatialConditioningMode

METHOD_V1_RUNTIME_SCHEMA = "method-v1-libero-molmo-runtime-v1"
METHOD_V1_SERIALIZER_ID = "grounded-precise-text-v1"
METHOD_V1_REPLAN_INTERVAL = 10


def method_v1_executor_id(server_identity_sha256: str) -> str:
    """Content identity for the exact public Stage-2 execution semantics."""

    return canonical_sha256(
        {
            "schema_version": METHOD_V1_RUNTIME_SCHEMA,
            "server_identity_sha256": server_identity_sha256,
            "serializer_id": METHOD_V1_SERIALIZER_ID,
            "replan_interval": METHOD_V1_REPLAN_INTERVAL,
            "session_policy": "reset_each_high_level_stage_v1",
            "chunk_seed_policy": "base_plus_stage10000_plus_chunk_v1",
        }
    )


def _latest_frame_index(context: PolicyContext, camera: str) -> int:
    try:
        return context.latest_frame_index(camera)
    except ValueError:
        return -1


def public_context_from_observation(
    *,
    prompt: str,
    observation: LiberoPublicObservation,
    frame_store: RGBFrameStore,
    frame_namespace: str,
    frame_index: int,
    previous: PolicyContext | None = None,
    event: PublicActionEvent | None = None,
) -> PolicyContext:
    """Store one real two-camera observation and form a public context."""

    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    if previous is None and event is not None:
        raise ValueError("an action event requires a previous context")
    if previous is not None:
        if prompt != previous.prompt:
            raise ValueError("the original user prompt cannot change")
        if event is None:
            raise ValueError("post-action context requires one public event")
    agent = frame_store.put(
        observation.agentview_rgb,
        frame_id=f"{frame_namespace}:agentview",
        camera="agentview",
        frame_index=frame_index,
    )
    wrist = frame_store.put(
        observation.wrist_rgb,
        frame_id=f"{frame_namespace}:wrist",
        camera="wrist",
        frame_index=frame_index,
    )
    prior_frames = () if previous is None else previous.frames
    prior_history = () if previous is None else previous.public_history
    return PolicyContext(
        prompt=prompt,
        frames=prior_frames + (agent, wrist),
        public_history=prior_history + (() if event is None else (event,)),
        proprioception=observation.state,
    )


@dataclasses.dataclass(frozen=True)
class PolicyExecutionFailure(RuntimeError):
    """Malformed VLA output after a still-evaluable physical prefix."""

    message: str
    candidate: GroundedIntervention
    request: ExecutorRequest
    previous_context: PolicyContext
    final_context: PolicyContext
    control_steps_used: int
    model_calls: int
    public_trace: Mapping[str, Any]

    def __str__(self) -> str:
        return self.message


class LiberoMolmoBudgetedExecutor:
    """Apply one grounded instruction for an exact fixed number of steps."""

    def __init__(
        self,
        *,
        environment: LiberoE1Environment,
        client: MolmoAct2HTTPClient,
        frame_store: RGBFrameStore,
        branch_namespace: str,
    ) -> None:
        if not isinstance(environment, LiberoE1Environment):
            raise TypeError("environment must be a LiberoE1Environment")
        if not isinstance(client, MolmoAct2HTTPClient):
            raise TypeError("client must be a MolmoAct2HTTPClient")
        if not isinstance(frame_store, RGBFrameStore):
            raise TypeError("frame_store must be an RGBFrameStore")
        self.environment = environment
        self.client = client
        self.frame_store = frame_store
        self.branch_namespace = " ".join(str(branch_namespace).split())
        if not self.branch_namespace:
            raise ValueError("branch_namespace must be non-empty")
        self.serializer = GroundedTextSerializer(
            METHOD_V1_SERIALIZER_ID,
            spatial_mode=SpatialConditioningMode.PRECISE_TEXT,
        )
        self._stage_index = 0
        self.public_traces: dict[str, dict[str, Any]] = {}

    @property
    def replan_interval(self) -> int:
        return METHOD_V1_REPLAN_INTERVAL

    @property
    def serializer_id(self) -> str:
        return self.serializer.serializer_id

    @property
    def executor_id(self) -> str:
        return method_v1_executor_id(self.client.expected_identity.digest)

    def _verify_live_context(self, context: PolicyContext) -> None:
        observation = self.environment.public_observation
        if context.proprioception != observation.state:
            raise ValueError("public context state differs from live LIBERO state")
        expected = {
            "agentview": canonical_rgb_sha256(observation.agentview_rgb),
            "wrist": canonical_rgb_sha256(observation.wrist_rgb),
        }
        for camera, digest in expected.items():
            frame = max(
                (item for item in context.frames if item.camera == camera),
                key=lambda item: item.frame_index,
            )
            if frame.image_sha256 != digest:
                raise ValueError(
                    f"public context {camera} differs from live LIBERO observation"
                )

    def _post_context(
        self,
        *,
        context: PolicyContext,
        request: ExecutorRequest,
        status: ExecutionStatus,
        stage_index: int,
    ) -> PolicyContext:
        event = PublicActionEvent(
            step_index=len(context.public_history),
            primitive=request.primitive,
            subtask_text=request.subtask_text,
            execution_status=status,
        )
        next_index = (
            max(
                _latest_frame_index(context, "agentview"),
                _latest_frame_index(context, "wrist"),
            )
            + 1
        )
        return public_context_from_observation(
            prompt=context.prompt,
            observation=self.environment.public_observation,
            frame_store=self.frame_store,
            frame_namespace=f"{self.branch_namespace}:stage-{stage_index}:post",
            frame_index=next_index,
            previous=context,
            event=event,
        )

    def execute_candidate(
        self,
        candidate: GroundedIntervention,
        context: PolicyContext,
        *,
        control_step_budget: int,
        model_seed: int,
    ) -> BudgetedExecution:
        if (
            not isinstance(control_step_budget, int)
            or isinstance(control_step_budget, bool)
            or control_step_budget < 1
            or control_step_budget % self.replan_interval
        ):
            raise ValueError("control budget must be a positive multiple of ten")
        if not isinstance(model_seed, int) or isinstance(model_seed, bool):
            raise TypeError("model_seed must be an integer")
        candidate.validate_against(context)
        self._verify_live_context(context)
        stage_index = self._stage_index
        self._stage_index += 1
        serialized = self.serializer.serialize(candidate, context)
        request = ExecutorRequest.from_serialized(serialized)
        session_id = canonical_sha256(
            {
                "schema_version": METHOD_V1_RUNTIME_SCHEMA,
                "branch_namespace": self.branch_namespace,
                "stage_index": stage_index,
                "context_fingerprint": context.fingerprint(),
                "candidate_fingerprint": candidate.fingerprint(),
                "model_seed": model_seed,
            }
        )
        self.client.reset(session_id)

        chunks: list[dict[str, Any]] = []
        actions_applied: list[list[float]] = []
        steps_before = self.environment.step_count
        for chunk_index in range(control_step_budget // self.replan_interval):
            public = self.environment.public_observation
            request_state_sha256 = canonical_libero_state_sha256(public.state)
            input_frame_index = self.environment.step_count
            input_agentview = self.frame_store.put(
                public.agentview_rgb,
                frame_id=(
                    f"{self.branch_namespace}:stage-{stage_index}:"
                    f"chunk-{chunk_index}:input:agentview"
                ),
                camera="agentview",
                frame_index=input_frame_index,
            )
            input_wrist = self.frame_store.put(
                public.wrist_rgb,
                frame_id=(
                    f"{self.branch_namespace}:stage-{stage_index}:"
                    f"chunk-{chunk_index}:input:wrist"
                ),
                camera="wrist",
                frame_index=input_frame_index,
            )
            seed = model_seed + stage_index * 10_000 + chunk_index
            request_id = canonical_sha256(
                molmoact2_public_request_payload(
                    agentview_rgb_sha256=input_agentview.image_sha256,
                    wrist_rgb_sha256=input_wrist.image_sha256,
                    state=public.state,
                    instruction=request.subtask_text,
                    session_id=session_id,
                    seed=seed,
                    expected_identity=self.client.expected_identity,
                )
            )
            chunk_input = {
                "chunk_index": chunk_index,
                "seed": seed,
                "instruction": request.subtask_text,
                "session_id": session_id,
                "request_id": request_id,
                "server_identity_sha256": self.client.expected_identity.digest,
                "input_agentview": input_agentview.to_dict(),
                "input_wrist": input_wrist.to_dict(),
                "input_state": list(public.state),
                "input_state_sha256": request_state_sha256,
            }
            try:
                chunk = self.client.predict_action_chunk(
                    agentview_rgb=public.agentview_rgb,
                    wrist_rgb=public.wrist_rgb,
                    state=public.state,
                    instruction=request.subtask_text,
                    session_id=session_id,
                    seed=seed,
                )
            except MolmoAct2PolicyOutputError as error:
                chunks.append(
                    {
                        **chunk_input,
                        "status": "POLICY_OUTPUT_FAILURE",
                        "failure_type": type(error).__name__,
                        "message": str(error),
                        "actions": [],
                    }
                )
                final_context = self._post_context(
                    context=context,
                    request=request,
                    status=ExecutionStatus.FAILED,
                    stage_index=stage_index,
                )
                trace = {
                    "schema_version": METHOD_V1_RUNTIME_SCHEMA,
                    "status": "POLICY_OUTPUT_FAILURE",
                    "stage_index": stage_index,
                    "candidate_id": candidate.candidate_id,
                    "candidate_fingerprint": candidate.fingerprint(),
                    "context_fingerprint": context.fingerprint(),
                    "next_context_fingerprint": final_context.fingerprint(),
                    "post_context": final_context.to_dict(),
                    "request": request.to_dict(),
                    "executor_id": self.executor_id,
                    "server_identity": self.client.expected_identity.to_dict(),
                    "session_id": session_id,
                    "replan_interval": self.replan_interval,
                    "control_step_budget": control_step_budget,
                    "control_steps_used": self.environment.step_count - steps_before,
                    "model_seed": model_seed,
                    "chunks": chunks,
                    "actions_applied": actions_applied,
                }
                self.public_traces[canonical_sha256(trace)] = trace
                raise PolicyExecutionFailure(
                    message=str(error),
                    candidate=candidate,
                    request=request,
                    previous_context=context,
                    final_context=final_context,
                    control_steps_used=self.environment.step_count - steps_before,
                    model_calls=len(chunks),
                    public_trace=trace,
                ) from error
            except Exception as error:
                chunks.append(
                    {
                        **chunk_input,
                        "status": "INFRASTRUCTURE_FAILURE",
                        "failure_phase": "molmo_request",
                        "failure_type": type(error).__name__,
                        "message": str(error),
                        "actions": [],
                    }
                )
                trace = {
                    "schema_version": METHOD_V1_RUNTIME_SCHEMA,
                    "status": "INFRASTRUCTURE_FAILURE",
                    "stage_index": stage_index,
                    "candidate_id": candidate.candidate_id,
                    "candidate_fingerprint": candidate.fingerprint(),
                    "context_fingerprint": context.fingerprint(),
                    "request": request.to_dict(),
                    "executor_id": self.executor_id,
                    "server_identity": self.client.expected_identity.to_dict(),
                    "session_id": session_id,
                    "replan_interval": self.replan_interval,
                    "control_step_budget": control_step_budget,
                    "control_steps_used": self.environment.step_count - steps_before,
                    "model_seed": model_seed,
                    "chunks": chunks,
                }
                self.public_traces[canonical_sha256(trace)] = trace
                raise
            applied_this_chunk: list[list[float]] = []
            try:
                for action in chunk.actions:
                    _, applied = self.environment.step(action)
                    values = [float(item) for item in applied]
                    applied_this_chunk.append(values)
                    actions_applied.append(values)
            except Exception as error:
                chunks.append(
                    {
                        **chunk_input,
                        "status": "INFRASTRUCTURE_FAILURE",
                        "failure_phase": "environment_step",
                        "failure_type": type(error).__name__,
                        "message": str(error),
                        "actions": applied_this_chunk,
                        "latency_ms": chunk.latency_ms,
                    }
                )
                trace = {
                    "schema_version": METHOD_V1_RUNTIME_SCHEMA,
                    "status": "INFRASTRUCTURE_FAILURE",
                    "stage_index": stage_index,
                    "candidate_id": candidate.candidate_id,
                    "candidate_fingerprint": candidate.fingerprint(),
                    "context_fingerprint": context.fingerprint(),
                    "request": request.to_dict(),
                    "executor_id": self.executor_id,
                    "server_identity": self.client.expected_identity.to_dict(),
                    "session_id": session_id,
                    "replan_interval": self.replan_interval,
                    "control_step_budget": control_step_budget,
                    "control_steps_used": self.environment.step_count - steps_before,
                    "model_seed": model_seed,
                    "chunks": chunks,
                    "actions_applied": actions_applied,
                }
                self.public_traces[canonical_sha256(trace)] = trace
                raise
            if len(applied_this_chunk) != self.replan_interval:
                raise RuntimeError("MolmoAct2 action chunk did not apply ten steps")
            chunks.append(
                {
                    **chunk_input,
                    "status": "COMPLETED",
                    "actions": applied_this_chunk,
                    "latency_ms": chunk.latency_ms,
                }
            )

        used = self.environment.step_count - steps_before
        if used != control_step_budget:
            raise RuntimeError("live LIBERO execution did not consume fixed budget")
        next_context = self._post_context(
            context=context,
            request=request,
            status=ExecutionStatus.COMPLETED,
            stage_index=stage_index,
        )
        trace = {
            "schema_version": METHOD_V1_RUNTIME_SCHEMA,
            "status": "COMPLETED",
            "stage_index": stage_index,
            "candidate_id": candidate.candidate_id,
            "candidate_fingerprint": candidate.fingerprint(),
            "context_fingerprint": context.fingerprint(),
            "next_context_fingerprint": next_context.fingerprint(),
            "post_context": next_context.to_dict(),
            "request": request.to_dict(),
            "executor_id": self.executor_id,
            "server_identity": self.client.expected_identity.to_dict(),
            "session_id": session_id,
            "replan_interval": self.replan_interval,
            "control_step_budget": control_step_budget,
            "control_steps_used": used,
            "model_seed": model_seed,
            "chunks": chunks,
            "actions_applied": actions_applied,
        }
        trace_sha256 = canonical_sha256(trace)
        self.public_traces[trace_sha256] = trace
        receipt = ExecutorReceipt(
            receipt_id=canonical_sha256(
                {
                    "runtime": self.executor_id,
                    "request": request.request_digest,
                    "trace": trace_sha256,
                }
            ),
            executor_id=self.executor_id,
            candidate_id=candidate.candidate_id,
            candidate_fingerprint=candidate.fingerprint(),
            request_digest=request.request_digest,
            status=ExecutionStatus.COMPLETED,
            post_frames=next_context.frames[-2:],
        )
        return BudgetedExecution(
            request=request,
            receipt=receipt,
            previous_context=context,
            next_context=next_context,
            requested_control_steps=control_step_budget,
            control_steps_used=used,
            model_seed=model_seed,
            public_trace_sha256=trace_sha256,
        )
