"""Semantic validation for Method-V1 execution evidence.

Artifact-tree hashes prove that bytes did not change after sealing.  They do
not prove that those bytes describe a possible run.  This module therefore
recomputes the public execution chain before an outcome is admitted to a
training dataset: RGB files, public state hashes, fixed stage budgets, chunk
seeds, 7-D actions, request/receipt identities, context transitions, and the
DIRECT or OPEN->DIRECT continuation topology.

The validator consumes only public execution artifacts.  The evaluator
predicate and final Boolean label remain a separate private sidecar.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .continuation import (
    INITIAL_DIRECT_STEPS,
    OPEN_STEPS,
    POST_OPEN_DIRECT_STEPS,
    REPLAN_INTERVAL,
)
from .contracts import (
    ExecutionStatus,
    PolicyContext,
    Primitive,
    PublicFrame,
    canonical_sha256,
)
from .execution import ExecutorReceipt, ExecutorRequest
from .libero_runtime import canonical_libero_state_sha256
from .method_v1_data import CollectionAttempt, CollectionAttemptStatus
from .method_v1_runtime import METHOD_V1_RUNTIME_SCHEMA, method_v1_executor_id
from .molmoact2 import (
    MolmoAct2ServerIdentity,
    molmoact2_public_request_payload,
)
from .rgb import RGBFrameStore


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name}: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _exact(
    value: Any,
    *,
    keys: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    observed = set(value)
    if observed != keys:
        raise ValueError(
            f"{name} fields differ from schema; "
            f"missing={sorted(keys - observed)}, extra={sorted(observed - keys)}"
        )
    return value


def _sha256(value: Any, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_actions(value: Any, *, name: str) -> list[list[float]]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    result: list[list[float]] = []
    for row_index, raw_row in enumerate(value):
        if not isinstance(raw_row, list) or len(raw_row) != 7:
            raise ValueError(f"{name}[{row_index}] must be one 7-D action")
        try:
            row = [float(item) for item in raw_row]
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name}[{row_index}] must be numeric") from error
        if any(not math.isfinite(item) for item in row):
            raise ValueError(f"{name}[{row_index}] contains a non-finite action")
        result.append(row)
    return result


def _request(value: Any) -> ExecutorRequest:
    mapping = _exact(
        value,
        keys={
            "serializer_id",
            "serialized_subtask_digest",
            "candidate_id",
            "candidate_fingerprint",
            "primitive",
            "context_fingerprint",
            "subtask_text",
            "spatial_audit_payload",
            "request_digest",
        },
        name="executor request",
    )
    request = ExecutorRequest(
        serializer_id=str(mapping["serializer_id"]),
        serialized_subtask_digest=str(mapping["serialized_subtask_digest"]),
        candidate_id=str(mapping["candidate_id"]),
        candidate_fingerprint=str(mapping["candidate_fingerprint"]),
        primitive=Primitive(str(mapping["primitive"])),
        context_fingerprint=str(mapping["context_fingerprint"]),
        subtask_text=str(mapping["subtask_text"]),
        spatial_audit_payload=mapping["spatial_audit_payload"],
    )
    if request.to_dict() != dict(mapping):
        raise ValueError("executor request digest does not reproduce")
    return request


def _receipt(value: Any) -> ExecutorReceipt:
    mapping = _exact(
        value,
        keys={
            "receipt_id",
            "executor_id",
            "candidate_id",
            "candidate_fingerprint",
            "request_digest",
            "status",
            "post_frames",
        },
        name="executor receipt",
    )
    raw_frames = mapping["post_frames"]
    if not isinstance(raw_frames, list):
        raise TypeError("executor receipt post_frames must be a list")
    receipt = ExecutorReceipt(
        receipt_id=str(mapping["receipt_id"]),
        executor_id=str(mapping["executor_id"]),
        candidate_id=str(mapping["candidate_id"]),
        candidate_fingerprint=str(mapping["candidate_fingerprint"]),
        request_digest=str(mapping["request_digest"]),
        status=ExecutionStatus(str(mapping["status"])),
        post_frames=tuple(PublicFrame.from_mapping(item) for item in raw_frames),
    )
    if receipt.to_dict() != dict(mapping):
        raise ValueError("executor receipt does not reproduce")
    return receipt


def _validate_context_frames(context: PolicyContext, store: RGBFrameStore) -> None:
    canonical_libero_state_sha256(context.proprioception)
    for frame in context.frames:
        store.resolve(frame)


def _validate_chunk(
    value: Any,
    *,
    chunk_index: int,
    stage_index: int,
    model_seed: int,
    request: ExecutorRequest,
    session_id: str,
    server_identity: MolmoAct2ServerIdentity,
    store: RGBFrameStore,
    terminal_status: str,
) -> list[list[float]]:
    common = {
        "chunk_index",
        "seed",
        "instruction",
        "session_id",
        "request_id",
        "server_identity_sha256",
        "input_agentview",
        "input_wrist",
        "input_state",
        "input_state_sha256",
        "status",
        "actions",
    }
    if terminal_status == "COMPLETED":
        keys = common | {"latency_ms"}
    elif terminal_status == "POLICY_OUTPUT_FAILURE":
        keys = common | {"failure_type", "message"}
    else:
        keys = common | {"failure_phase", "failure_type", "message"}
        if value.get("failure_phase") == "environment_step":
            keys.add("latency_ms")
    mapping = _exact(value, keys=keys, name=f"stage chunk {chunk_index}")
    if mapping["status"] != terminal_status:
        raise ValueError("chunk terminal status differs from its schema")
    if _integer(mapping["chunk_index"], name="chunk_index") != chunk_index:
        raise ValueError("chunk indices are not contiguous")
    expected_seed = model_seed + stage_index * 10_000 + chunk_index
    if _integer(mapping["seed"], name="chunk seed") != expected_seed:
        raise ValueError("chunk seed differs from the frozen derivation")
    if mapping["instruction"] != request.subtask_text:
        raise ValueError("chunk instruction differs from executor request")
    if mapping["session_id"] != session_id:
        raise ValueError("chunk session differs from stage session")
    observed_server = _sha256(mapping["server_identity_sha256"], name="server identity")
    if observed_server != server_identity.digest:
        raise ValueError("chunk server identity differs from stage identity")
    agent = PublicFrame.from_mapping(mapping["input_agentview"])
    wrist = PublicFrame.from_mapping(mapping["input_wrist"])
    if agent.camera != "agentview" or wrist.camera != "wrist":
        raise ValueError("chunk must bind agentview and wrist input frames")
    store.resolve(agent)
    store.resolve(wrist)
    state = mapping["input_state"]
    state_digest = canonical_libero_state_sha256(state)
    if state_digest != mapping["input_state_sha256"]:
        raise ValueError("chunk public state digest does not reproduce")
    expected_request_id = canonical_sha256(
        molmoact2_public_request_payload(
            agentview_rgb_sha256=agent.image_sha256,
            wrist_rgb_sha256=wrist.image_sha256,
            state=state,
            instruction=request.subtask_text,
            session_id=session_id,
            seed=expected_seed,
            expected_identity=server_identity,
        )
    )
    if mapping["request_id"] != expected_request_id:
        raise ValueError("MolmoAct2 request ID does not reproduce from public inputs")
    actions = _finite_actions(mapping["actions"], name="chunk actions")
    if terminal_status == "COMPLETED" and len(actions) != REPLAN_INTERVAL:
        raise ValueError("completed chunk must apply exactly ten actions")
    if terminal_status == "POLICY_OUTPUT_FAILURE" and actions:
        raise ValueError("policy-output failure cannot apply its invalid chunk")
    if "latency_ms" in mapping:
        _integer(mapping["latency_ms"], name="chunk latency_ms")
    return actions


_OUTCOME_STAGE_KEYS = {
    "schema_version",
    "status",
    "stage_index",
    "candidate_id",
    "candidate_fingerprint",
    "context_fingerprint",
    "next_context_fingerprint",
    "post_context",
    "request",
    "executor_id",
    "server_identity",
    "session_id",
    "replan_interval",
    "control_step_budget",
    "control_steps_used",
    "model_seed",
    "chunks",
    "actions_applied",
}


def _validate_outcome_stage(
    value: Any,
    *,
    trace_sha256: str,
    stage_index: int,
    expected_status: str,
    expected_budget: int,
    expected_seed: int,
    previous_context: PolicyContext,
    branch_namespace: str,
    expected_executor_id: str,
    store: RGBFrameStore,
) -> tuple[PolicyContext, ExecutorRequest, str]:
    mapping = _exact(
        value,
        keys=_OUTCOME_STAGE_KEYS,
        name=f"stage trace {stage_index}",
    )
    if canonical_sha256(mapping) != trace_sha256:
        raise ValueError("stage trace key does not match trace bytes")
    if mapping["schema_version"] != METHOD_V1_RUNTIME_SCHEMA:
        raise ValueError("stage trace runtime schema changed")
    if mapping["status"] != expected_status:
        raise ValueError("stage trace status changed")
    if _integer(mapping["stage_index"], name="stage_index") != stage_index:
        raise ValueError("stage indices are not contiguous")
    if mapping["context_fingerprint"] != previous_context.fingerprint():
        raise ValueError("stage previous context fingerprint changed")
    request = _request(mapping["request"])
    request.validate_context(previous_context)
    if (
        mapping["candidate_id"] != request.candidate_id
        or mapping["candidate_fingerprint"] != request.candidate_fingerprint
    ):
        raise ValueError("stage candidate differs from executor request")
    if mapping["executor_id"] != expected_executor_id:
        raise ValueError("stage executor differs from outcome contract")
    server_identity = MolmoAct2ServerIdentity.from_mapping(mapping["server_identity"])
    if method_v1_executor_id(server_identity.digest) != expected_executor_id:
        raise ValueError("stage server identity does not reproduce executor identity")
    if _integer(mapping["replan_interval"], name="replan_interval", minimum=1) != 10:
        raise ValueError("stage replan interval changed")
    if (
        _integer(mapping["control_step_budget"], name="control_step_budget", minimum=1)
        != expected_budget
    ):
        raise ValueError("stage budget differs from frozen continuation")
    if _integer(mapping["model_seed"], name="model_seed") != expected_seed:
        raise ValueError("stage model seed differs from schedule")
    expected_session = canonical_sha256(
        {
            "schema_version": METHOD_V1_RUNTIME_SCHEMA,
            "branch_namespace": branch_namespace,
            "stage_index": stage_index,
            "context_fingerprint": previous_context.fingerprint(),
            "candidate_fingerprint": request.candidate_fingerprint,
            "model_seed": expected_seed,
        }
    )
    if mapping["session_id"] != expected_session:
        raise ValueError("stage session ID does not reproduce")
    raw_chunks = mapping["chunks"]
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ValueError("outcome stage must contain at least one model-call chunk")
    expected_chunk_count = expected_budget // REPLAN_INTERVAL
    if expected_status == "COMPLETED" and len(raw_chunks) != expected_chunk_count:
        raise ValueError("completed stage chunk count differs from fixed budget")
    if (
        expected_status == "POLICY_OUTPUT_FAILURE"
        and len(raw_chunks) > expected_chunk_count
    ):
        raise ValueError("failed stage exceeds its fixed chunk budget")
    actions: list[list[float]] = []
    for chunk_index, raw_chunk in enumerate(raw_chunks):
        status = (
            "POLICY_OUTPUT_FAILURE"
            if expected_status == "POLICY_OUTPUT_FAILURE"
            and chunk_index == len(raw_chunks) - 1
            else "COMPLETED"
        )
        actions.extend(
            _validate_chunk(
                raw_chunk,
                chunk_index=chunk_index,
                stage_index=stage_index,
                model_seed=expected_seed,
                request=request,
                session_id=expected_session,
                server_identity=server_identity,
                store=store,
                terminal_status=status,
            )
        )
    if mapping["actions_applied"] != actions:
        raise ValueError("stage action summary differs from chunk actions")
    used = _integer(mapping["control_steps_used"], name="control_steps_used")
    if used != len(actions):
        raise ValueError("stage control-step count differs from applied actions")
    if expected_status == "COMPLETED" and used != expected_budget:
        raise ValueError("completed stage did not consume the exact budget")
    post_context = PolicyContext.from_mapping(mapping["post_context"])
    _validate_context_frames(post_context, store)
    if mapping["next_context_fingerprint"] != post_context.fingerprint():
        raise ValueError("stage post-context fingerprint does not reproduce")
    if post_context.prompt != previous_context.prompt:
        raise ValueError("stage changed the original task prompt")
    if post_context.frames[:-2] != previous_context.frames:
        raise ValueError("stage post-context did not append exactly two frames")
    if post_context.public_history[:-1] != previous_context.public_history:
        raise ValueError("stage post-context changed public action history")
    if len(post_context.public_history) != len(previous_context.public_history) + 1:
        raise ValueError("stage post-context did not append one action event")
    event = post_context.public_history[-1]
    expected_event_status = (
        ExecutionStatus.COMPLETED
        if expected_status == "COMPLETED"
        else ExecutionStatus.FAILED
    )
    if (
        event.step_index != len(previous_context.public_history)
        or event.primitive is not request.primitive
        or event.subtask_text != request.subtask_text
        or event.execution_status is not expected_event_status
    ):
        raise ValueError("stage post-context event differs from request/status")
    return post_context, request, trace_sha256


def _validate_stage_summary(
    value: Any,
    *,
    trace: Mapping[str, Any],
    trace_sha256: str,
    previous_context: PolicyContext,
    post_context: PolicyContext,
    request: ExecutorRequest,
    expected_budget: int,
    expected_seed: int,
) -> str:
    mapping = _exact(
        value,
        keys={
            "request",
            "receipt",
            "previous_context_fingerprint",
            "next_context_fingerprint",
            "requested_control_steps",
            "control_steps_used",
            "model_seed",
            "public_trace_sha256",
        },
        name="fixed-horizon stage summary",
    )
    if dict(mapping["request"]) != request.to_dict():
        raise ValueError("fixed-horizon summary request differs from stage trace")
    if (
        mapping["previous_context_fingerprint"] != previous_context.fingerprint()
        or mapping["next_context_fingerprint"] != post_context.fingerprint()
        or mapping["requested_control_steps"] != expected_budget
        or mapping["control_steps_used"] != expected_budget
        or mapping["model_seed"] != expected_seed
        or mapping["public_trace_sha256"] != trace_sha256
    ):
        raise ValueError("fixed-horizon stage summary changed execution identity")
    receipt = _receipt(mapping["receipt"])
    receipt.validate_request(request)
    if receipt.executor_id != trace["executor_id"]:
        raise ValueError("receipt executor differs from stage trace")
    if receipt.status is not ExecutionStatus.COMPLETED:
        raise ValueError("successful fixed-horizon stage has a failed receipt")
    if receipt.post_frames != post_context.frames[len(previous_context.frames) :]:
        raise ValueError("receipt frames differ from the public post-context")
    expected_receipt = canonical_sha256(
        {
            "runtime": trace["executor_id"],
            "request": request.request_digest,
            "trace": trace_sha256,
        }
    )
    if receipt.receipt_id != expected_receipt:
        raise ValueError("executor receipt ID does not reproduce")
    return receipt.receipt_id


def _expected_topology(
    primitive: Primitive, status: str
) -> tuple[list[Primitive], list[int]]:
    if status == "DIRECT_EXECUTED":
        if primitive is not Primitive.DIRECT:
            raise ValueError("DIRECT execution status requires a DIRECT candidate")
        return [Primitive.DIRECT], [INITIAL_DIRECT_STEPS]
    if status == "OPEN_THEN_DIRECT_EXECUTED":
        if primitive is not Primitive.OPEN:
            raise ValueError("OPEN continuation status requires an OPEN candidate")
        return [Primitive.OPEN, Primitive.DIRECT], [OPEN_STEPS, POST_OPEN_DIRECT_STEPS]
    if status == "OPEN_WITH_NO_DIRECT_CANDIDATE":
        if primitive is not Primitive.OPEN:
            raise ValueError("no-continuation status requires an OPEN candidate")
        return [Primitive.OPEN], [OPEN_STEPS]
    raise ValueError("unknown fixed-horizon execution status")


def _validate_continuation_audit(
    value: Any,
    *,
    required: bool,
    open_context_fingerprint: str | None,
    continuation_candidate_fingerprint: str | None,
) -> None:
    if not isinstance(value, list):
        raise TypeError("continuation proposal audit must be a list")
    if not required:
        if value:
            raise ValueError("DIRECT branch must not call the continuation proposer")
        return
    if len(value) != 1 or not isinstance(value[0], Mapping):
        raise ValueError("OPEN branch must record exactly one continuation proposal")
    record = value[0]
    if (
        record.get("context_fingerprint") != open_context_fingerprint
        or record.get("deterministic_seed") != 0
        or record.get("max_candidates") != 3
    ):
        raise ValueError("continuation proposal audit changed frozen inputs")
    if continuation_candidate_fingerprint is not None and (
        record.get("status") != "PROPOSED"
        or record.get("selected_top1_fingerprint") != continuation_candidate_fingerprint
    ):
        raise ValueError("continuation top-1 differs from executed DIRECT stage")


def _validate_completed_trace(
    trace: Mapping[str, Any],
    *,
    attempt: CollectionAttempt,
    store: RGBFrameStore,
) -> None:
    top = _exact(
        trace,
        keys={"fixed_horizon", "stage_traces", "continuation_proposal_audit"},
        name="completed public execution trace",
    )
    branch = attempt.branch
    if branch is None:  # pragma: no cover - CollectionAttempt guards this.
        raise RuntimeError("completed attempt has no branch")
    fixed = _exact(
        top["fixed_horizon"],
        keys={
            "continuation_policy_id",
            "model_seed",
            "selected_candidate_id",
            "selected_candidate_fingerprint",
            "status",
            "stages",
            "continuation_candidate_ids",
            "continuation_candidate_fingerprints",
            "final_context_fingerprint",
            "control_steps_used",
        },
        name="fixed-horizon execution",
    )
    entry = attempt.schedule_entry
    if (
        fixed["continuation_policy_id"]
        != branch.outcome_contract.continuation_policy_id
        or fixed["model_seed"] != entry.model_seed
        or fixed["selected_candidate_id"] != branch.candidate_id
        or fixed["selected_candidate_fingerprint"] != branch.candidate_fingerprint
    ):
        raise ValueError("fixed-horizon identity differs from branch/schedule")
    primitives, budgets = _expected_topology(
        branch.executed_intervention.primitive, str(fixed["status"])
    )
    summaries = fixed["stages"]
    raw_stage_traces = top["stage_traces"]
    if not isinstance(summaries, list) or len(summaries) != len(primitives):
        raise ValueError("fixed-horizon stage count differs from status")
    if not isinstance(raw_stage_traces, Mapping) or len(raw_stage_traces) != len(
        primitives
    ):
        raise ValueError("stage trace coverage differs from fixed-horizon stages")
    previous = branch.context
    _validate_context_frames(previous, store)
    receipt_ids: list[str] = []
    continuation_fingerprint: str | None = None
    open_context_fingerprint: str | None = None
    for stage_index, (primitive, budget, summary) in enumerate(
        zip(primitives, budgets, summaries, strict=True)
    ):
        if not isinstance(summary, Mapping):
            raise TypeError("fixed-horizon stage summary must be a mapping")
        trace_sha256 = _sha256(
            summary.get("public_trace_sha256"), name="stage public trace digest"
        )
        if trace_sha256 not in raw_stage_traces:
            raise ValueError("fixed-horizon stage has no matching detailed trace")
        detailed = raw_stage_traces[trace_sha256]
        post, request, _ = _validate_outcome_stage(
            detailed,
            trace_sha256=trace_sha256,
            stage_index=stage_index,
            expected_status="COMPLETED",
            expected_budget=budget,
            expected_seed=entry.model_seed,
            previous_context=previous,
            branch_namespace=entry.entry_id,
            expected_executor_id=branch.outcome_contract.executor_id,
            store=store,
        )
        if request.primitive is not primitive:
            raise ValueError("stage primitive differs from fixed continuation topology")
        if stage_index == 0 and (
            request.candidate_id != branch.candidate_id
            or request.candidate_fingerprint != branch.candidate_fingerprint
        ):
            raise ValueError("first stage changed the selected candidate")
        if stage_index == 0 and primitive is Primitive.OPEN:
            open_context_fingerprint = post.fingerprint()
        if stage_index == 1:
            continuation_fingerprint = request.candidate_fingerprint
        receipt_ids.append(
            _validate_stage_summary(
                summary,
                trace=detailed,
                trace_sha256=trace_sha256,
                previous_context=previous,
                post_context=post,
                request=request,
                expected_budget=budget,
                expected_seed=entry.model_seed,
            )
        )
        previous = post
    if fixed["final_context_fingerprint"] != previous.fingerprint():
        raise ValueError("fixed-horizon final context does not reproduce")
    if fixed["control_steps_used"] != sum(budgets):
        raise ValueError("fixed-horizon control budget does not reproduce")
    if branch.post_action_frames != previous.frames[len(branch.context.frames) :]:
        raise ValueError("branch post-action frames differ from execution trace")
    if branch.execution_status is not ExecutionStatus.COMPLETED:
        raise ValueError("completed trace has a failed branch status")
    if branch.execution_receipt_id != canonical_sha256(receipt_ids):
        raise ValueError("branch receipt chain does not reproduce")
    continuation_ids = fixed["continuation_candidate_ids"]
    continuation_fingerprints = fixed["continuation_candidate_fingerprints"]
    if not isinstance(continuation_ids, list) or not isinstance(
        continuation_fingerprints, list
    ):
        raise TypeError("continuation candidate identities must be lists")
    if len(continuation_ids) != len(continuation_fingerprints):
        raise ValueError("continuation candidate IDs/fingerprints are misaligned")
    if continuation_fingerprint is None:
        if continuation_ids or continuation_fingerprints:
            raise ValueError("trace claims unexecuted continuation candidates")
    elif (
        not continuation_fingerprints
        or continuation_fingerprints[0] != continuation_fingerprint
    ):
        raise ValueError("executed continuation is not proposer top-1")
    _validate_continuation_audit(
        top["continuation_proposal_audit"],
        required=branch.executed_intervention.primitive is Primitive.OPEN,
        open_context_fingerprint=open_context_fingerprint,
        continuation_candidate_fingerprint=continuation_fingerprint,
    )
    diagnostics = branch.diagnostics
    if (
        diagnostics.get("control_steps_used") != sum(budgets)
        or diagnostics.get("model_calls") != sum(budgets) // REPLAN_INTERVAL
        or diagnostics.get("policy_output_failure") is not False
    ):
        raise ValueError("branch diagnostics differ from completed execution trace")


def _validate_policy_failure_trace(
    trace: Mapping[str, Any],
    *,
    attempt: CollectionAttempt,
    store: RGBFrameStore,
) -> None:
    top = _exact(
        trace,
        keys={
            "status",
            "failed_stage",
            "stage_traces",
            "continuation_proposal_audit",
        },
        name="policy-failure public execution trace",
    )
    if top["status"] != "POLICY_OUTPUT_FAILURE":
        raise ValueError("policy-failure trace status changed")
    branch = attempt.branch
    if branch is None:  # pragma: no cover
        raise RuntimeError("policy-failure attempt has no branch")
    raw_traces = top["stage_traces"]
    if not isinstance(raw_traces, Mapping) or len(raw_traces) not in {1, 2}:
        raise ValueError("policy failure must contain one or two stage traces")
    ordered = sorted(
        raw_traces.items(),
        key=lambda item: _integer(item[1].get("stage_index"), name="stage_index"),
    )
    previous = branch.context
    _validate_context_frames(previous, store)
    total_steps = 0
    total_calls = 0
    open_context_fingerprint: str | None = None
    continuation_fingerprint: str | None = None
    for stage_index, (trace_sha256, detailed) in enumerate(ordered):
        digest = _sha256(trace_sha256, name="stage trace digest")
        terminal = (
            "POLICY_OUTPUT_FAILURE" if stage_index == len(ordered) - 1 else "COMPLETED"
        )
        if branch.executed_intervention.primitive is Primitive.DIRECT:
            if len(ordered) != 1:
                raise ValueError("DIRECT failure cannot contain a continuation stage")
            primitive, budget = Primitive.DIRECT, INITIAL_DIRECT_STEPS
        else:
            primitive = Primitive.OPEN if stage_index == 0 else Primitive.DIRECT
            budget = OPEN_STEPS if stage_index == 0 else POST_OPEN_DIRECT_STEPS
        post, request, _ = _validate_outcome_stage(
            detailed,
            trace_sha256=digest,
            stage_index=stage_index,
            expected_status=terminal,
            expected_budget=budget,
            expected_seed=attempt.schedule_entry.model_seed,
            previous_context=previous,
            branch_namespace=attempt.schedule_entry.entry_id,
            expected_executor_id=branch.outcome_contract.executor_id,
            store=store,
        )
        if request.primitive is not primitive:
            raise ValueError("policy-failure stage primitive changed")
        if stage_index == 0 and (
            request.candidate_id != branch.candidate_id
            or request.candidate_fingerprint != branch.candidate_fingerprint
        ):
            raise ValueError("failed execution changed the selected candidate")
        if stage_index == 0 and primitive is Primitive.OPEN:
            open_context_fingerprint = post.fingerprint()
        if stage_index == 1:
            continuation_fingerprint = request.candidate_fingerprint
        total_steps += int(detailed["control_steps_used"])
        total_calls += len(detailed["chunks"])
        previous = post
    if dict(top["failed_stage"]) != dict(ordered[-1][1]):
        raise ValueError("failed_stage is not the terminal detailed trace")
    if branch.post_action_frames != previous.frames[len(branch.context.frames) :]:
        raise ValueError("failed branch frames differ from execution trace")
    if branch.execution_status is not ExecutionStatus.FAILED:
        raise ValueError("policy-output failure must be a failed branch")
    if branch.execution_receipt_id != canonical_sha256(
        {"entry_id": attempt.schedule_entry.entry_id, "policy_failure": dict(top)}
    ):
        raise ValueError("policy-failure branch receipt does not reproduce")
    _validate_continuation_audit(
        top["continuation_proposal_audit"],
        required=len(ordered) == 2,
        open_context_fingerprint=open_context_fingerprint,
        continuation_candidate_fingerprint=continuation_fingerprint,
    )
    diagnostics = branch.diagnostics
    if (
        diagnostics.get("control_steps_used") != total_steps
        or diagnostics.get("model_calls") != total_calls
        or diagnostics.get("policy_output_failure") is not True
    ):
        raise ValueError("failed branch diagnostics differ from execution trace")


def validate_method_v1_execution_trace(
    *,
    attempt_dir: str | Path,
    attempt: CollectionAttempt,
) -> str | None:
    """Validate public execution semantics and return the trace digest.

    Outcome-bearing attempts require a complete semantic trace.  Unlabelled
    infrastructure attempts may have no trace when failure occurred before a
    live executor existed; if a trace exists, its outer digest remains checked
    by the collection loader and this function rejects label-like topology.
    """

    if not isinstance(attempt, CollectionAttempt):
        raise TypeError("attempt must be a CollectionAttempt")
    root = Path(attempt_dir).expanduser().resolve()
    trace_path = root / "public_execution_trace.json"
    if attempt.status is CollectionAttemptStatus.INFRASTRUCTURE_FAILURE:
        if not trace_path.exists():
            return None
        trace = _read_json(trace_path, name="infrastructure public trace")
        # A failure can occur while a VLA request is in flight, after a
        # completed OPEN prefix, or even after the completed public trace was
        # written but before evaluator artifacts were sealed.  Such attempts
        # remain unlabelled.  Recheck every detailed-stage content address
        # without reclassifying a completed physical prefix as supervision.
        raw_stages = trace.get("stage_traces")
        if not isinstance(raw_stages, Mapping):
            raise ValueError("infrastructure trace lacks detailed stage records")
        for digest, stage in raw_stages.items():
            if canonical_sha256(stage) != _sha256(digest, name="stage trace digest"):
                raise ValueError("infrastructure stage trace digest mismatch")
            if not isinstance(stage, Mapping) or stage.get("status") not in {
                "COMPLETED",
                "INFRASTRUCTURE_FAILURE",
                "POLICY_OUTPUT_FAILURE",
            }:
                raise ValueError(
                    "infrastructure trace contains an invalid stage status"
                )
        return canonical_sha256(trace)
    if not trace_path.is_file():
        raise ValueError("outcome attempt lacks public execution trace")
    trace = _read_json(trace_path, name="public execution trace")
    branch = attempt.branch
    if branch is None:  # pragma: no cover
        raise RuntimeError("outcome attempt has no branch")
    store = RGBFrameStore(root / "public_frames")
    if branch.execution_status is ExecutionStatus.COMPLETED:
        _validate_completed_trace(trace, attempt=attempt, store=store)
    else:
        _validate_policy_failure_trace(trace, attempt=attempt, store=store)
    return canonical_sha256(trace)
