"""E1: two-referent MolmoAct2/LIBERO execution-interface ceiling.

E1 deliberately starts with human boxes drawn on public RGB.  It tests whether
one already-grounded intention survives the Stage-2 interface and controls the
physical referent.  It does not test autonomous candidate proposal, Stage-1
selection, or the learned outcome model.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import struct
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .conditioning import ReferentConditioner, draw_public_grounding_marker
from .contracts import (
    ExecutionStatus,
    GroundedIntervention,
    GroundingReference,
    OutcomeContract,
    PolicyContext,
    Primitive,
    PublicFrame,
    canonical_sha256,
)
from .data import ObservedBranch
from .execution import ExecutorReceipt, ExecutorRequest
from .libero_runtime import (
    LiberoE1Environment,
    canonical_libero_state_sha256,
    libero_runtime_identity,
)
from .molmoact2 import (
    MOLMOACT2_CUDA_VERSION,
    MOLMOACT2_HTTP_SCHEMA,
    MOLMOACT2_LIVE_BACKEND_KIND,
    MOLMOACT2_TORCH_VERSION,
    MOLMOACT2_TRANSFORMERS_VERSION,
    MolmoAct2ActionChunk,
    MolmoAct2HTTPClient,
    MolmoAct2ProtocolError,
    MolmoAct2ServerIdentity,
    molmoact2_adapter_sha256,
    molmoact2_public_request_payload,
)
from .rgb import RGBFrameStore, canonical_rgb_sha256
from .serialization import (
    GroundedTextSerializer,
    SpatialConditioningMode,
)

E1_PLAN_SCHEMA = "e1-referent-executor-plan-v1"
E1_RUNNER_SOURCE_SCHEMA = "e1-runner-source-tree-v1"
_SERIALIZER_IDS = {
    SpatialConditioningMode.COARSE_TEXT: "grounded-coarse-text-v1",
    SpatialConditioningMode.PRECISE_TEXT: "grounded-precise-text-v1",
    SpatialConditioningMode.VISUAL_MARKER: "grounded-visual-marker-v1",
}


def _clean_text(value: object, *, name: str) -> str:
    result = " ".join(str(value or "").split())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    path.write_text(encoded + "\n", encoding="utf-8")


def _write_json_once(path: Path, value: object) -> None:
    """Create one immutable JSON receipt without an overwrite path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _seal_artifact_tree(output_dir: Path) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact_seal.json":
            continue
        relative = path.relative_to(output_dir).as_posix()
        entries.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    payload: dict[str, object] = {
        "schema_version": "e1-artifact-seal-v1",
        "files": entries,
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    _write_json(output_dir / "artifact_seal.json", payload)
    return payload


def _live_backend_attestation(
    health: Mapping[str, object],
    *,
    expected_identity: MolmoAct2ServerIdentity,
) -> tuple[bool, tuple[str, ...]]:
    """Check the minimum runtime evidence required for a live-model claim."""

    runtime = health.get("runtime_identity")
    if not isinstance(runtime, Mapping):
        return False, ("runtime_identity_missing",)
    expected = {
        "backend_kind": MOLMOACT2_LIVE_BACKEND_KIND,
        "checkpoint_id": expected_identity.checkpoint_id,
        "checkpoint_revision": expected_identity.checkpoint_revision,
        "upstream_code_revision": expected_identity.upstream_code_revision,
        "model_dtype": expected_identity.dtype,
        "model_device": expected_identity.device,
        "snapshot_revision": expected_identity.checkpoint_revision,
        "adapter_source_sha256": molmoact2_adapter_sha256(),
        "transformers": MOLMOACT2_TRANSFORMERS_VERSION,
        "cuda_runtime": MOLMOACT2_CUDA_VERSION,
        "config_sha256": expected_identity.config_sha256,
        "norm_stats_sha256": expected_identity.norm_stats_sha256,
        "checkpoint_manifest_sha256": (expected_identity.checkpoint_manifest_sha256),
        "checkpoint_file_count": expected_identity.checkpoint_file_count,
        "checkpoint_total_bytes": expected_identity.checkpoint_total_bytes,
        "norm_stats_format": expected_identity.norm_stats_format,
        "norm_mode": expected_identity.norm_mode,
        "state_dim": expected_identity.state_dim,
        "action_dim": expected_identity.action_dim,
        "action_horizon": expected_identity.action_horizon,
        "n_action_steps": expected_identity.num_steps,
        "control_mode": expected_identity.control_mode,
    }
    failures = [
        f"{key}_mismatch"
        for key, value in expected.items()
        if runtime.get(key) != value
    ]
    torch_version = str(runtime.get("torch", "")).split("+", maxsplit=1)[0]
    if torch_version != MOLMOACT2_TORCH_VERSION:
        failures.append("torch_mismatch")
    for key in ("model_class", "device_name"):
        if not str(runtime.get(key, "")).strip():
            failures.append(f"{key}_missing")
    return not failures, tuple(failures)


def _normalized_action_for_audit(value: object) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 7:
        raise ValueError("recorded MolmoAct2 action must be a 7-D sequence")
    try:
        row = [struct.unpack("<f", struct.pack("<f", float(item)))[0] for item in value]
    except (OverflowError, TypeError, ValueError, struct.error) as error:
        raise ValueError("recorded MolmoAct2 action must fit finite float32") from error
    if any(not math.isfinite(item) for item in row):
        raise ValueError("recorded MolmoAct2 action must be finite")
    row[-1] = -1.0 if row[-1] < 0 else 1.0
    return row


def _derive_e1_diagnostics(
    *,
    trial: E1Trial,
    per_step: object,
    chunks: object,
    horizon: int,
) -> tuple[dict[str, object], tuple[str, ...]]:
    if not isinstance(per_step, list) or not isinstance(chunks, list):
        raise TypeError("E1 traces must contain per-step and chunk lists")
    first_contact: tuple[str, ...] | None = None
    first_contact_step: int | None = None
    any_distractor = False
    distractor_before_or_at_goal = False
    target_grasped = False
    goal_step: int | None = None
    allowed_objects = {trial.target_object, *trial.distractor_objects}
    for index, raw in enumerate(per_step, start=1):
        step = _require_exact_keys(
            raw,
            expected={
                "step",
                "contact_objects",
                "target_grasped",
                "target_predicate",
            },
            name=f"E1 evaluator step {index}",
        )
        if step["step"] != index:
            raise ValueError("E1 evaluator step indices must be consecutive")
        contacts_raw = step["contact_objects"]
        if not isinstance(contacts_raw, list):
            raise TypeError("E1 contact_objects must be a list")
        contacts = tuple(str(item) for item in contacts_raw)
        if len(set(contacts)) != len(contacts) or any(
            item not in allowed_objects for item in contacts
        ):
            raise ValueError("E1 evaluator contacts contain invalid object identities")
        if not isinstance(step["target_grasped"], bool) or not isinstance(
            step["target_predicate"], bool
        ):
            raise TypeError("E1 evaluator flags must be booleans")
        if contacts and first_contact is None:
            first_contact = contacts
            first_contact_step = index
        if any(item in trial.distractor_objects for item in contacts):
            any_distractor = True
            if goal_step is None:
                distractor_before_or_at_goal = True
        if step["target_grasped"]:
            target_grasped = True
        if goal_step is None and step["target_predicate"]:
            goal_step = index
    intended = bool(
        first_contact is not None
        and trial.target_object in first_contact
        and not any(item in trial.distractor_objects for item in first_contact)
    )
    wrong_first = bool(
        first_contact is not None
        and any(item in trial.distractor_objects for item in first_contact)
    )
    diagnostics: dict[str, object] = {
        "first_contact_intended": intended,
        "wrong_first_contact": wrong_first,
        "no_candidate_contact": first_contact is None,
        "any_distractor_contact_within_horizon": any_distractor,
        "distractor_contact_before_or_at_goal_or_horizon_end": (
            distractor_before_or_at_goal
        ),
        "first_contact_step": first_contact_step,
        "target_grasped": target_grasped,
        "full_physical_goal": goal_step is not None,
        "goal_step": goal_step,
        "simulator_steps": len(per_step),
        "model_calls": len(chunks),
        "fixed_horizon_completed": len(per_step) == horizon,
        "evaluator_feedback_changed_policy_input": False,
        "policy_output_failure": None,
    }
    return diagnostics, tuple(first_contact or ())


def validate_e1_artifacts(
    output_dir: str | Path,
    *,
    require_empirical: bool = False,
    expected_plan: E1Plan | None = None,
    expected_trial_id: str | None = None,
) -> dict[str, object]:
    """Verify one immutable E1 result tree without running model or simulator."""

    root = Path(output_dir).expanduser().resolve()
    seal = _read_json(root / "artifact_seal.json")
    manifest_sha256 = seal.pop("manifest_sha256", None)
    if manifest_sha256 != canonical_sha256(seal):
        raise ValueError("E1 artifact manifest fingerprint mismatch")
    if seal.get("schema_version") != "e1-artifact-seal-v1":
        raise ValueError("unsupported E1 artifact seal")
    entries = seal.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("E1 artifact seal has no files")
    expected_paths: set[str] = set()
    for raw in entries:
        if not isinstance(raw, dict):
            raise TypeError("E1 artifact entry must be an object")
        relative = str(raw.get("path", ""))
        path = (root / relative).resolve()
        if root not in path.parents or relative in expected_paths or not path.is_file():
            raise ValueError("invalid, duplicate, or missing E1 artifact path")
        expected_paths.add(relative)
        if path.stat().st_size != raw.get("size") or _file_sha256(path) != raw.get(
            "sha256"
        ):
            raise ValueError(f"E1 artifact bytes changed: {relative}")
    observed_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_seal.json"
    }
    if observed_paths != expected_paths:
        raise ValueError("E1 artifact tree contains unsealed files")

    started = _require_exact_keys(
        _read_json(root / "started.json"),
        expected={
            "status",
            "plan_id",
            "plan_sha256",
            "trial_id",
            "runner_repository",
        },
        name="E1 started artifact",
    )
    if started["status"] != "STARTED":
        raise ValueError("E1 started artifact has an invalid status")
    runner_repository = _require_exact_keys(
        started["runner_repository"],
        expected={"git_revision", "dirty"},
        name="E1 runner repository identity",
    )
    if expected_plan is None and expected_trial_id is not None:
        raise ValueError("expected_trial_id requires expected_plan")
    expected_trial = (
        expected_plan.trial(expected_trial_id)
        if expected_plan is not None and expected_trial_id is not None
        else None
    )
    if expected_plan is not None:
        if expected_trial is None:
            raise ValueError("expected_plan validation requires expected_trial_id")
        expected_identity = {
            "plan_id": expected_plan.plan_id,
            "plan_sha256": expected_plan.frozen_plan_sha256,
            "trial_id": expected_trial.trial_id,
        }
        if any(started.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("started artifact does not match the expected E1 plan row")

    if (root / "failed.json").is_file():
        failed = _require_exact_keys(
            _read_json(root / "failed.json"),
            expected={
                "status",
                "plan_id",
                "plan_sha256",
                "trial_id",
                "error_type",
                "error",
                "observed_branch_generated",
            },
            name="E1 failed artifact",
        )
        if (
            failed["status"] != "INFRASTRUCTURE_FAILURE"
            or failed["observed_branch_generated"] is not False
        ):
            raise ValueError("E1 failed artifact has invalid failure semantics")
        if (root / "completed.json").exists() or (
            root / "observed_branch.public.json"
        ).exists():
            raise ValueError("infrastructure-failed E1 run must not contain a label")
        if (
            expected_plan is not None
            and expected_trial is not None
            and (
                failed.get("plan_id") != expected_plan.plan_id
                or failed.get("plan_sha256") != expected_plan.frozen_plan_sha256
                or failed.get("trial_id") != expected_trial.trial_id
            )
        ):
            raise ValueError("failed artifact does not match expected E1 plan row")
        return {
            "status": "VERIFIED_INFRASTRUCTURE_FAILURE",
            "manifest_sha256": manifest_sha256,
        }

    completed = _require_exact_keys(
        _read_json(root / "completed.json"),
        expected={
            "status",
            "schema_version",
            "plan_id",
            "plan_sha256",
            "trial_id",
            "conditioning_mode",
            "candidate_id",
            "candidate_fingerprint",
            "request_digest",
            "receipt_id",
            "execution_status",
            "public_execution_trace_sha256",
            "private_evaluator_trace_sha256",
            "observed_branch_public_fingerprint",
            "diagnostics",
            "empirical_execution",
            "autonomous_grounding",
            "stage1_selection_executed",
        },
        name="E1 completed artifact",
    )
    public_trace = _require_exact_keys(
        _read_json(root / "public_execution_trace.json"),
        expected={
            "schema_version",
            "plan_id",
            "plan_sha256",
            "trial_id",
            "execution_policy_id",
            "server_health",
            "executor_request",
            "initial_conditioned_input",
            "action_chunks",
            "actions_applied",
            "milestones",
            "post_frames",
            "execution_status",
        },
        name="E1 public execution trace",
    )
    private_trace = _require_exact_keys(
        _read_json(root / "private" / "evaluator_trace.json"),
        expected={
            "schema_version",
            "target_object",
            "distractor_objects",
            "target_predicate",
            "per_step",
            "summary",
        },
        name="E1 private evaluator trace",
    )
    if (
        completed["status"] != "COMPLETED"
        or completed["schema_version"] != MOLMOACT2_HTTP_SCHEMA
        or completed["autonomous_grounding"] is not False
        or completed["stage1_selection_executed"] is not False
        or public_trace["schema_version"] != "e1-public-execution-trace-v1"
        or private_trace["schema_version"] != "e1-private-evaluator-trace-v1"
    ):
        raise ValueError("E1 artifact schema or pilot-boundary declaration changed")
    public_digest = canonical_sha256(public_trace)
    private_digest = canonical_sha256(private_trace)
    if completed.get("public_execution_trace_sha256") != public_digest:
        raise ValueError("public execution trace is not bound to completed report")
    if completed.get("private_evaluator_trace_sha256") != private_digest:
        raise ValueError("private evaluator trace is not bound to completed report")
    receipt = _require_exact_keys(
        _read_json(root / "execution_receipt.json"),
        expected={
            "receipt_id",
            "executor_id",
            "candidate_id",
            "candidate_fingerprint",
            "request_digest",
            "status",
            "post_frames",
        },
        name="E1 execution receipt",
    )
    request = _read_json(root / "executor_request.json")
    request_payload = {
        key: value for key, value in request.items() if key != "request_digest"
    }
    if request.get("request_digest") != canonical_sha256(request_payload):
        raise ValueError("executor request digest is internally inconsistent")
    expected_receipt_id = canonical_sha256(
        {
            "request_digest": request["request_digest"],
            "server_identity": public_trace["server_health"]["identity_sha256"],
            "public_execution_trace_sha256": public_digest,
            "private_evaluator_trace_sha256": private_digest,
        }
    )
    if receipt.get("receipt_id") != expected_receipt_id:
        raise ValueError("execution receipt does not bind the complete E1 trace")
    if completed.get("receipt_id") != expected_receipt_id:
        raise ValueError("completed report does not bind the execution receipt")
    server_identity = MolmoAct2ServerIdentity.from_mapping(
        public_trace["server_health"]["identity"]
    )
    if public_trace["server_health"].get("identity_sha256") != server_identity.digest:
        raise ValueError("server health identity is internally inconsistent")
    live_backend, attestation_failures = _live_backend_attestation(
        public_trace["server_health"], expected_identity=server_identity
    )
    if completed.get("empirical_execution") is not live_backend:
        raise ValueError("empirical flag does not match the backend runtime evidence")
    if require_empirical and not live_backend:
        raise ValueError("canonical E1 ledger accepts live MolmoAct2 execution only")
    if live_backend and runner_repository["dirty"] is not False:
        raise ValueError("E1 empirical runner must begin from a clean checkout")
    if expected_plan is not None and expected_trial is not None:
        expected_identity = {
            "plan_id": expected_plan.plan_id,
            "plan_sha256": expected_plan.frozen_plan_sha256,
            "trial_id": expected_trial.trial_id,
        }
        for name, payload in (
            ("completed", completed),
            ("public trace", public_trace),
        ):
            if any(
                payload.get(key) != value for key, value in expected_identity.items()
            ):
                raise ValueError(f"{name} does not match the expected E1 plan row")
        if completed.get("conditioning_mode") != expected_trial.mode.value:
            raise ValueError(
                "completed conditioning mode does not match expected trial"
            )
        if completed.get("candidate_id") != expected_trial.candidate_id:
            raise ValueError("completed candidate does not match expected trial")
        if public_trace["execution_policy_id"] != expected_plan.execution_policy_id:
            raise ValueError("public trace execution policy changed after plan freeze")
        if server_identity.digest != expected_plan.server_identity.digest:
            raise ValueError(
                "artifact executor identity does not match expected E1 plan"
            )
        if (
            private_trace.get("target_object") != expected_trial.target_object
            or tuple(private_trace.get("distractor_objects", ()))
            != expected_trial.distractor_objects
        ):
            raise ValueError("private evaluator identity does not match expected trial")
        if (
            tuple(private_trace.get("target_predicate", ()))
            != expected_trial.target_predicate
        ):
            raise ValueError(
                "private evaluator predicate does not match expected trial"
            )
    mirrors = {
        "executor_request.json": public_trace["executor_request"],
        "executor_health.json": public_trace["server_health"],
        "conditioned_input.json": public_trace["initial_conditioned_input"],
        "action_chunks.json": public_trace["action_chunks"],
        "actions_applied.json": public_trace["actions_applied"],
        "milestones.json": public_trace["milestones"],
    }
    for relative, expected in mirrors.items():
        actual = json.loads((root / relative).read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"{relative} does not match the bound execution trace")
    branch_payload = _read_json(root / "observed_branch.public.json")
    branch = ObservedBranch.from_public_mapping(branch_payload)
    if branch_payload != branch.to_dict(include_private=False):
        raise ValueError("public observed branch is not in canonical typed form")
    candidate_fingerprint = branch.executed_intervention.fingerprint()
    context_fingerprint = branch.context.fingerprint()
    if (
        completed.get("observed_branch_public_fingerprint")
        != branch.public_fingerprint()
    ):
        raise ValueError("completed report does not bind the public observed branch")
    if (
        request.get("candidate_fingerprint") != candidate_fingerprint
        or receipt.get("candidate_fingerprint") != candidate_fingerprint
        or completed.get("candidate_fingerprint") != candidate_fingerprint
    ):
        raise ValueError("candidate fingerprint chain is internally inconsistent")
    if request.get("context_fingerprint") != context_fingerprint:
        raise ValueError("request context fingerprint is internally inconsistent")
    if receipt.get("request_digest") != request.get("request_digest"):
        raise ValueError("receipt request digest does not match request")
    if (
        branch.execution_receipt_id != expected_receipt_id
        or branch.candidate_id != request.get("candidate_id")
    ):
        raise ValueError("observed branch identity does not match execution chain")
    if expected_plan is None or expected_trial is None:
        raise ValueError(
            "semantic E1 validation requires the frozen expected plan and trial"
        )
    expected_branch_identity = {
        "branch_id": f"{expected_plan.plan_id}:{expected_trial.trial_id}",
        "initial_state_group": (
            f"{expected_plan.plan_id}:state-{expected_plan.init_state_index}"
        ),
        "decision_group_id": (
            f"{expected_plan.plan_id}:state-{expected_plan.init_state_index}:"
            f"{expected_trial.mode.value}"
        ),
        "split": "pilot",
        "reset_state_sha256": expected_plan.reset_state_sha256,
        "repeat_index": 0,
    }
    if any(
        getattr(branch, key) != value for key, value in expected_branch_identity.items()
    ):
        raise ValueError("observed branch grouping does not match the frozen plan")
    if branch.context.prompt != expected_plan.prompt:
        raise ValueError("observed branch prompt does not match the frozen plan")
    if branch.context.public_history:
        raise ValueError("initial-state E1 context must have empty public history")
    if branch.context.proprioception != expected_plan.expected_state:
        raise ValueError("observed branch public state does not match the frozen plan")
    if len(branch.context.frames) != 2:
        raise ValueError("E1 context must contain exactly agentview and wrist frames")
    frame_by_camera = {frame.camera: frame for frame in branch.context.frames}
    if set(frame_by_camera) != {"agentview", "wrist"}:
        raise ValueError("E1 context cameras do not match the frozen plan")
    if (
        frame_by_camera["agentview"].image_sha256
        != expected_plan.expected_agentview_sha256
        or frame_by_camera["wrist"].image_sha256 != expected_plan.expected_wrist_sha256
        or frame_by_camera["agentview"].frame_id
        != f"{expected_plan.plan_id}:state-{expected_plan.init_state_index}:pre:agentview"
        or frame_by_camera["wrist"].frame_id
        != f"{expected_plan.plan_id}:state-{expected_plan.init_state_index}:pre:wrist"
        or frame_by_camera["agentview"].frame_index != 0
        or frame_by_camera["wrist"].frame_index != 0
        or (frame_by_camera["agentview"].width, frame_by_camera["agentview"].height)
        != (256, 256)
        or (frame_by_camera["wrist"].width, frame_by_camera["wrist"].height)
        != (256, 256)
    ):
        raise ValueError("E1 context frame identities do not match the frozen plan")
    grounding = branch.executed_intervention.grounding
    if grounding is None:
        raise ValueError("expected E1 candidate grounding is missing")
    if (
        branch.candidate_id != expected_trial.candidate_id
        or branch.executed_intervention.primitive is not Primitive.DIRECT
        or branch.executed_intervention.referent != expected_trial.referent
        or branch.executed_intervention.parameters
        != (("destination", "stove cook region"),)
        or grounding.box_xyxy != expected_trial.box_xyxy
        or grounding.point_xy != expected_trial.point_xy
        or grounding.camera != expected_trial.grounding_camera
        or grounding.image_sha256 != expected_plan.expected_agentview_sha256
    ):
        raise ValueError("observed candidate does not match expected E1 trial")
    expected_contract = _outcome_contract(expected_plan, expected_trial)
    if branch.outcome_contract.to_dict() != expected_contract.to_dict():
        raise ValueError("E1 outcome contract changed after plan freeze")
    serializer = GroundedTextSerializer(
        _SERIALIZER_IDS[expected_trial.mode], spatial_mode=expected_trial.mode
    )
    serialized = serializer.serialize(branch.executed_intervention, branch.context)
    expected_request = ExecutorRequest.from_serialized(serialized)
    if request != expected_request.to_dict():
        raise ValueError("executor request is not reproducible from the frozen branch")

    public_store = RGBFrameStore(root / "public_frames")
    conditioned_store = RGBFrameStore(root / "conditioned_frames")
    for frame in (*branch.context.frames, *branch.post_action_frames):
        public_store.resolve(frame)
    expected_initial_conditioned = ReferentConditioner(public_store).prepare(
        context=branch.context,
        intervention=branch.executed_intervention,
        serialized=serialized,
        mode=expected_trial.mode,
    )
    if (
        public_trace["initial_conditioned_input"]
        != expected_initial_conditioned.public_identity()
    ):
        raise ValueError("initial conditioned input cannot be regenerated")

    raw_chunks = public_trace["action_chunks"]
    raw_milestones = public_trace["milestones"]
    if not isinstance(raw_chunks, list) or not isinstance(raw_milestones, list):
        raise TypeError("E1 public chunks and milestones must be lists")
    if not raw_chunks:
        raise ValueError("E1 completed trace must contain at least one model call")
    expected_actions: list[list[float]] = []
    expected_milestone_index = 0
    current_agent = public_store.resolve(frame_by_camera["agentview"])
    current_wrist = public_store.resolve(frame_by_camera["wrist"])
    current_state = expected_plan.expected_state
    session_id = canonical_sha256(
        {
            "plan": expected_plan.frozen_plan_sha256,
            "trial": expected_trial.trial_id,
            "candidate": candidate_fingerprint,
        }
    )
    for chunk_index, raw_chunk in enumerate(raw_chunks):
        if not isinstance(raw_chunk, dict):
            raise TypeError("E1 action chunk must be an object")
        status = raw_chunk.get("status")
        common_keys = {
            "chunk_index",
            "status",
            "request_id",
            "server_identity_sha256",
            "seed",
            "instruction",
            "marker_applied",
            "input_agentview_sha256",
            "input_wrist_sha256",
            "input_agentview_frame",
            "input_wrist_frame",
            "input_state",
            "input_state_sha256",
            "actions_returned",
            "actions_applied",
            "action_start_index",
            "actions",
        }
        expected_keys = common_keys | {"latency_ms"}
        _require_exact_keys(
            raw_chunk,
            expected=expected_keys,
            name=f"E1 action chunk {chunk_index}",
        )
        if raw_chunk["chunk_index"] != chunk_index:
            raise ValueError("E1 action chunk indices must be consecutive")
        marker_applied = expected_trial.mode is SpatialConditioningMode.VISUAL_MARKER
        expected_agent = (
            draw_public_grounding_marker(
                current_agent,
                box_xyxy=expected_trial.box_xyxy,
                point_xy=expected_trial.point_xy,
            )
            if marker_applied
            else current_agent
        )
        expected_agent_digest = canonical_rgb_sha256(expected_agent)
        expected_wrist_digest = canonical_rgb_sha256(current_wrist)
        input_agent = PublicFrame.from_mapping(raw_chunk["input_agentview_frame"])
        input_wrist = PublicFrame.from_mapping(raw_chunk["input_wrist_frame"])
        conditioned_store.resolve(input_agent)
        conditioned_store.resolve(input_wrist)
        if (
            input_agent.image_sha256 != expected_agent_digest
            or input_wrist.image_sha256 != expected_wrist_digest
            or input_agent.frame_id
            != f"{expected_trial.trial_id}:conditioned:{chunk_index}:agentview"
            or input_wrist.frame_id
            != f"{expected_trial.trial_id}:conditioned:{chunk_index}:wrist"
            or input_agent.camera != "agentview"
            or input_wrist.camera != "wrist"
            or input_agent.frame_index != chunk_index
            or input_wrist.frame_index != chunk_index
            or (input_agent.width, input_agent.height) != (256, 256)
            or (input_wrist.width, input_wrist.height) != (256, 256)
            or raw_chunk["input_agentview_sha256"] != expected_agent_digest
            or raw_chunk["input_wrist_sha256"] != expected_wrist_digest
            or raw_chunk["instruction"] != serialized.subtask_text
            or raw_chunk["seed"] != expected_trial.seed + chunk_index
            or raw_chunk["marker_applied"] is not marker_applied
            or raw_chunk["action_start_index"] != len(expected_actions)
            or tuple(float(item) for item in raw_chunk["input_state"]) != current_state
            or raw_chunk["input_state_sha256"]
            != canonical_libero_state_sha256(current_state)
        ):
            raise ValueError("E1 action chunk inputs do not follow the frozen policy")
        public_request = molmoact2_public_request_payload(
            agentview_rgb_sha256=expected_agent_digest,
            wrist_rgb_sha256=expected_wrist_digest,
            state=current_state,
            instruction=serialized.subtask_text,
            session_id=session_id,
            seed=expected_trial.seed + chunk_index,
            expected_identity=expected_plan.server_identity,
        )
        if (
            raw_chunk["request_id"] != canonical_sha256(public_request)
            or raw_chunk["server_identity_sha256"]
            != expected_plan.server_identity.digest
        ):
            raise ValueError("E1 action chunk request identity is inconsistent")
        if status != "COMPLETED":
            raise ValueError("unsupported E1 action-chunk status")
        if len(expected_actions) >= expected_plan.horizon:
            raise ValueError("E1 cannot call the model after completing the horizon")
        action_chunk = MolmoAct2ActionChunk(
            request_id=str(raw_chunk["request_id"]),
            server_identity_sha256=str(raw_chunk["server_identity_sha256"]),
            actions=tuple(tuple(row) for row in raw_chunk["actions"]),
            latency_ms=int(raw_chunk["latency_ms"]),
        )
        remaining = expected_plan.horizon - len(expected_actions)
        applied_count = min(len(action_chunk.actions), remaining)
        if (
            raw_chunk["actions_returned"] != 10
            or raw_chunk["actions_applied"] != applied_count
        ):
            raise ValueError("E1 applied-action count is inconsistent")
        expected_actions.extend(
            _normalized_action_for_audit(row)
            for row in action_chunk.actions[:applied_count]
        )
        if expected_milestone_index >= len(raw_milestones):
            raise ValueError("E1 completed chunk lacks a milestone")
        milestone = _require_exact_keys(
            raw_milestones[expected_milestone_index],
            expected={
                "chunk_index",
                "agentview",
                "wrist",
                "state",
                "state_sha256",
            },
            name=f"E1 milestone {expected_milestone_index}",
        )
        if milestone["chunk_index"] != chunk_index:
            raise ValueError("E1 milestone does not follow its action chunk")
        milestone_agent = PublicFrame.from_mapping(milestone["agentview"])
        milestone_wrist = PublicFrame.from_mapping(milestone["wrist"])
        current_agent = public_store.resolve(milestone_agent)
        current_wrist = public_store.resolve(milestone_wrist)
        if (
            milestone_agent.frame_id
            != f"{expected_trial.trial_id}:milestone:{chunk_index}:agentview"
            or milestone_wrist.frame_id
            != f"{expected_trial.trial_id}:milestone:{chunk_index}:wrist"
            or milestone_agent.camera != "agentview"
            or milestone_wrist.camera != "wrist"
            or milestone_agent.frame_index != chunk_index + 1
            or milestone_wrist.frame_index != chunk_index + 1
            or (milestone_agent.width, milestone_agent.height) != (256, 256)
            or (milestone_wrist.width, milestone_wrist.height) != (256, 256)
        ):
            raise ValueError("E1 milestone frame identity is inconsistent")
        current_state = tuple(float(item) for item in milestone["state"])
        if milestone["state_sha256"] != canonical_libero_state_sha256(current_state):
            raise ValueError("E1 milestone state fingerprint is inconsistent")
        expected_milestone_index += 1
    if expected_milestone_index != len(raw_milestones):
        raise ValueError("E1 trace contains unmatched milestones")
    if len(expected_actions) != expected_plan.horizon:
        raise ValueError("normal E1 completion must reach the frozen horizon")
    if public_trace["actions_applied"] != expected_actions:
        raise ValueError(
            "applied actions are not the normalized returned-action prefixes"
        )

    expected_diagnostics, first_contact = _derive_e1_diagnostics(
        trial=expected_trial,
        per_step=private_trace["per_step"],
        chunks=raw_chunks,
        horizon=expected_plan.horizon,
    )
    if len(expected_actions) != expected_diagnostics["simulator_steps"]:
        raise ValueError("public applied actions and private evaluator steps disagree")
    if canonical_sha256(private_trace["summary"]) != canonical_sha256(
        expected_diagnostics
    ):
        raise ValueError("E1 evaluator summary is not derivable from per-step trace")
    observed_outcome = branch.observed_outcome
    if observed_outcome is not expected_diagnostics["first_contact_intended"]:
        raise ValueError("E1 training label is not derivable from per-step trace")
    if canonical_sha256(branch.diagnostics) != canonical_sha256(expected_diagnostics):
        raise ValueError("E1 branch diagnostics differ from recomputed diagnostics")
    expected_execution_status = ExecutionStatus.COMPLETED
    if (
        branch.execution_status is not expected_execution_status
        or public_trace["execution_status"] != expected_execution_status.value
        or completed.get("execution_status") != expected_execution_status.value
        or canonical_sha256(completed.get("diagnostics"))
        != canonical_sha256(expected_diagnostics)
        or completed.get("request_digest") != expected_request.request_digest
    ):
        raise ValueError("E1 status/diagnostic chain is inconsistent")
    post_payloads = [frame.to_dict() for frame in branch.post_action_frames]
    if public_trace["post_frames"] != post_payloads:
        raise ValueError("E1 branch post frames do not match public trace")
    if len(post_payloads) != 2:
        raise ValueError("E1 requires exactly two post-action frames")
    post_by_camera = {frame.camera: frame for frame in branch.post_action_frames}
    if set(post_by_camera) != {"agentview", "wrist"}:
        raise ValueError("E1 post-action cameras are invalid")
    if (
        post_by_camera["agentview"].image_sha256 != canonical_rgb_sha256(current_agent)
        or post_by_camera["wrist"].image_sha256 != canonical_rgb_sha256(current_wrist)
        or post_by_camera["agentview"].frame_id
        != f"{expected_trial.trial_id}:post:agentview"
        or post_by_camera["wrist"].frame_id != f"{expected_trial.trial_id}:post:wrist"
        or post_by_camera["agentview"].frame_index != len(raw_chunks) + 1
        or post_by_camera["wrist"].frame_index != len(raw_chunks) + 1
        or (post_by_camera["agentview"].width, post_by_camera["agentview"].height)
        != (256, 256)
        or (post_by_camera["wrist"].width, post_by_camera["wrist"].height) != (256, 256)
    ):
        raise ValueError("E1 final frames do not match final milestone observation")
    parsed_receipt = ExecutorReceipt(
        receipt_id=str(receipt["receipt_id"]),
        executor_id=str(receipt["executor_id"]),
        candidate_id=str(receipt["candidate_id"]),
        candidate_fingerprint=str(receipt["candidate_fingerprint"]),
        request_digest=str(receipt["request_digest"]),
        status=ExecutionStatus(str(receipt["status"])),
        post_frames=tuple(
            PublicFrame.from_mapping(item) for item in receipt["post_frames"]
        ),
    )
    parsed_receipt.validate_request(expected_request)
    if receipt != parsed_receipt.to_dict():
        raise ValueError("E1 execution receipt is not in canonical typed form")
    if (
        parsed_receipt.executor_id
        != f"molmoact2-libero-http-v1@{expected_plan.server_identity.digest}"
        or parsed_receipt.status is not expected_execution_status
        or [frame.to_dict() for frame in parsed_receipt.post_frames] != post_payloads
    ):
        raise ValueError("E1 execution receipt is semantically inconsistent")
    expected_sidecar = {
        "target_object": expected_trial.target_object,
        "distractor_objects": list(expected_trial.distractor_objects),
        "target_predicate": list(expected_trial.target_predicate),
        "first_contact_objects": list(first_contact),
        "evaluator_trace_sha256": private_digest,
    }
    if _read_json(root / "private" / "evaluator_sidecar.json") != expected_sidecar:
        raise ValueError("E1 evaluator sidecar is inconsistent")
    private_tokens = (
        str(private_trace["target_object"]),
        *(str(item) for item in private_trace["distractor_objects"]),
    )
    public_text = json.dumps(public_trace, sort_keys=True)
    if any(token in public_text for token in private_tokens):
        raise ValueError(
            "evaluator object identity leaked into the public execution trace"
        )
    return {
        "status": "VERIFIED_COMPLETED",
        "evidence_class": (
            "LIVE_MOLMOACT2_LIBERO" if live_backend else "SOFTWARE_TEST_DOUBLE"
        ),
        "backend_attestation_failures": list(attestation_failures),
        "manifest_sha256": manifest_sha256,
        "public_execution_trace_sha256": public_digest,
        "private_evaluator_trace_sha256": private_digest,
        "observed_outcome": observed_outcome,
    }


def _require_exact_keys(
    value: object, *, expected: set[str], name: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{name} keys mismatch; missing={missing}, extra={extra}")
    return value


@dataclasses.dataclass(frozen=True)
class E1Trial:
    trial_id: str
    seed: int
    mode: SpatialConditioningMode
    candidate_id: str
    referent: str
    grounding_source_id: str
    grounding_camera: str
    box_xyxy: tuple[float, float, float, float]
    point_xy: tuple[float, float]
    target_object: str
    distractor_objects: tuple[str, ...]
    target_predicate: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "trial_id",
            "candidate_id",
            "referent",
            "grounding_source_id",
            "grounding_camera",
            "target_object",
        ):
            object.__setattr__(self, name, _clean_text(getattr(self, name), name=name))
        if self.grounding_source_id != "human-public-rgb-v1":
            raise ValueError("E1 grounding source must be human-public-rgb-v1")
        if self.grounding_camera != "agentview":
            raise ValueError("the E1 v1 annotation is frozen to agentview")
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            raise ValueError("trial seed must be a non-negative integer")
        if not isinstance(self.mode, SpatialConditioningMode):
            raise TypeError("trial mode must be SpatialConditioningMode")
        distractors = tuple(
            _clean_text(item, name="distractor object")
            for item in self.distractor_objects
        )
        if not distractors or self.target_object in distractors:
            raise ValueError("E1 needs at least one distinct distractor object")
        object.__setattr__(self, "distractor_objects", distractors)
        predicate = tuple(
            _clean_text(item, name="target predicate item")
            for item in self.target_predicate
        )
        if len(predicate) not in {2, 3} or self.target_object not in predicate:
            raise ValueError(
                "target predicate must explicitly evaluate the target object"
            )
        object.__setattr__(self, "target_predicate", predicate)
        # Reuse the production grounding validator with a synthetic frame identity.
        GroundingReference(
            camera=self.grounding_camera,
            frame_id="validation-frame",
            frame_index=0,
            image_sha256="0" * 64,
            box_xyxy=self.box_xyxy,
            point_xy=self.point_xy,
        )

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> E1Trial:
        value = _require_exact_keys(
            value,
            expected={
                "trial_id",
                "seed",
                "conditioning_mode",
                "candidate_id",
                "referent",
                "public_rgb_annotation",
                "evaluator_sidecar",
            },
            name="E1 trial",
        )
        annotation = _require_exact_keys(
            value["public_rgb_annotation"],
            expected={"source", "camera", "box_xyxy", "point_xy"},
            name="E1 public RGB annotation",
        )
        evaluator = _require_exact_keys(
            value["evaluator_sidecar"],
            expected={"target_object", "distractor_objects", "target_predicate"},
            name="E1 evaluator sidecar",
        )
        return cls(
            trial_id=str(value["trial_id"]),
            seed=int(value["seed"]),
            mode=SpatialConditioningMode(str(value["conditioning_mode"])),
            candidate_id=str(value["candidate_id"]),
            referent=str(value["referent"]),
            grounding_source_id=str(annotation["source"]),
            grounding_camera=str(annotation["camera"]),
            box_xyxy=tuple(float(item) for item in annotation["box_xyxy"]),
            point_xy=tuple(float(item) for item in annotation["point_xy"]),
            target_object=str(evaluator["target_object"]),
            distractor_objects=tuple(
                str(item) for item in evaluator["distractor_objects"]
            ),
            target_predicate=tuple(str(item) for item in evaluator["target_predicate"]),
        )


@dataclasses.dataclass(frozen=True)
class E1ModelCanary:
    """Outcome-free model/runtime check on an init state excluded from E1."""

    init_state_index: int
    env_seed: int
    reset_state_sha256: str
    expected_agentview_sha256: str
    expected_wrist_sha256: str
    expected_state: tuple[float, ...]
    expected_state_sha256: str
    instruction: str
    model_seed: int
    report_relative_path: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (self.init_state_index, self.env_seed, self.model_seed)
        ):
            raise ValueError("canary indices and seeds must be non-negative integers")
        object.__setattr__(
            self,
            "instruction",
            _clean_text(self.instruction, name="canary instruction"),
        )
        path = Path(_clean_text(self.report_relative_path, name="canary report path"))
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("canary report path must be repository-relative")
        object.__setattr__(self, "report_relative_path", path.as_posix())
        for name in (
            "reset_state_sha256",
            "expected_agentview_sha256",
            "expected_wrist_sha256",
            "expected_state_sha256",
        ):
            digest = _clean_text(getattr(self, name), name=name)
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be lowercase SHA-256")
            object.__setattr__(self, name, digest)
        state = tuple(float(item) for item in self.expected_state)
        if canonical_libero_state_sha256(state) != self.expected_state_sha256:
            raise ValueError("canary expected state fingerprint is inconsistent")
        object.__setattr__(self, "expected_state", state)

    @classmethod
    def from_mapping(cls, value: object) -> E1ModelCanary:
        mapping = _require_exact_keys(
            value,
            expected={
                "init_state_index",
                "env_seed",
                "reset_state_sha256",
                "expected_agentview_sha256",
                "expected_wrist_sha256",
                "expected_state",
                "expected_state_sha256",
                "instruction",
                "model_seed",
                "report_relative_path",
            },
            name="E1 model canary",
        )
        return cls(
            init_state_index=int(mapping["init_state_index"]),
            env_seed=int(mapping["env_seed"]),
            reset_state_sha256=str(mapping["reset_state_sha256"]),
            expected_agentview_sha256=str(mapping["expected_agentview_sha256"]),
            expected_wrist_sha256=str(mapping["expected_wrist_sha256"]),
            expected_state=tuple(float(item) for item in mapping["expected_state"]),
            expected_state_sha256=str(mapping["expected_state_sha256"]),
            instruction=str(mapping["instruction"]),
            model_seed=int(mapping["model_seed"]),
            report_relative_path=str(mapping["report_relative_path"]),
        )


@dataclasses.dataclass(frozen=True)
class E1Plan:
    plan_id: str
    runner_source_schema: str
    runner_source_files: tuple[tuple[str, str], ...]
    runner_source_tree_sha256: str
    libero_git_revision: str
    bddl_relative_path: str
    bddl_sha256: str
    init_states_relative_path: str
    init_states_sha256: str
    init_state_index: int
    env_seed: int
    reset_state_sha256: str
    expected_agentview_sha256: str
    expected_wrist_sha256: str
    expected_state: tuple[float, ...]
    expected_state_sha256: str
    image_size: int
    settle_steps: int
    control_mode: str
    simulator_runtime: Mapping[str, str]
    model_canary: E1ModelCanary
    prompt: str
    horizon: int
    execution_policy_id: str
    canonical_run_root: str
    execution_order: tuple[str, ...]
    server_identity: MolmoAct2ServerIdentity
    trials: tuple[E1Trial, ...]
    frozen_plan_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "plan_id",
            "runner_source_schema",
            "runner_source_tree_sha256",
            "libero_git_revision",
            "bddl_relative_path",
            "bddl_sha256",
            "init_states_relative_path",
            "init_states_sha256",
            "reset_state_sha256",
            "expected_agentview_sha256",
            "expected_wrist_sha256",
            "expected_state_sha256",
            "prompt",
            "execution_policy_id",
            "control_mode",
            "canonical_run_root",
            "frozen_plan_sha256",
        ):
            object.__setattr__(self, name, _clean_text(getattr(self, name), name=name))
        if self.init_state_index < 0 or self.env_seed < 0 or self.settle_steps < 0:
            raise ValueError(
                "state index, environment seed, and settle steps must be non-negative"
            )
        if self.image_size != 256:
            raise ValueError("MolmoAct2-LIBERO E1 image size is frozen to 256")
        if self.horizon != 300:
            raise ValueError("E1 execution horizon is frozen to 300 simulator steps")
        if (
            self.execution_policy_id
            != "fixed_conditioning_replan10_horizon300_no_evaluator_feedback_v1"
        ):
            raise ValueError("unsupported E1 execution policy")
        if self.control_mode != "relative":
            raise ValueError("E1 simulator control mode must be relative")
        run_root = Path(self.canonical_run_root)
        if run_root.is_absolute() or ".." in run_root.parts or not run_root.parts:
            raise ValueError(
                "canonical E1 run root must be a safe repository-relative path"
            )
        if not isinstance(self.server_identity, MolmoAct2ServerIdentity):
            raise TypeError("server_identity must be MolmoAct2ServerIdentity")
        if not isinstance(self.model_canary, E1ModelCanary):
            raise TypeError("model_canary must be E1ModelCanary")
        if self.model_canary.init_state_index == self.init_state_index:
            raise ValueError(
                "model canary must use an init state excluded from scored E1"
            )
        if self.runner_source_schema != E1_RUNNER_SOURCE_SCHEMA:
            raise ValueError("unsupported E1 runner source schema")
        source_files: list[tuple[str, str]] = []
        for raw_path, raw_digest in self.runner_source_files:
            relative = Path(_clean_text(raw_path, name="runner source path"))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError(
                    "runner source paths must be safe repository-relative paths"
                )
            digest = _clean_text(raw_digest, name="runner source digest")
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("runner source digest must be lowercase SHA-256")
            source_files.append((relative.as_posix(), digest))
        if not source_files or source_files != sorted(source_files):
            raise ValueError("runner source files must be non-empty and path-sorted")
        if len({path for path, _ in source_files}) != len(source_files):
            raise ValueError("runner source paths must be unique")
        object.__setattr__(self, "runner_source_files", tuple(source_files))
        expected_tree = canonical_sha256(
            {
                "schema_version": self.runner_source_schema,
                "files": [
                    {"path": path, "sha256": digest} for path, digest in source_files
                ],
            }
        )
        if self.runner_source_tree_sha256 != expected_tree:
            raise ValueError(
                "runner source tree fingerprint is internally inconsistent"
            )
        runtime = {
            _clean_text(key, name="simulator runtime key"): _clean_text(
                child, name=f"simulator runtime {key}"
            )
            for key, child in self.simulator_runtime.items()
        }
        if set(runtime) != {"python", "numpy", "torch", "robosuite", "mujoco"}:
            raise ValueError("simulator runtime identity has unexpected keys")
        object.__setattr__(self, "simulator_runtime", MappingProxyType(runtime))
        for name in (
            "bddl_sha256",
            "init_states_sha256",
            "reset_state_sha256",
            "expected_agentview_sha256",
            "expected_wrist_sha256",
            "expected_state_sha256",
            "frozen_plan_sha256",
            "runner_source_tree_sha256",
        ):
            digest = getattr(self, name)
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        state = tuple(float(item) for item in self.expected_state)
        if canonical_libero_state_sha256(state) != self.expected_state_sha256:
            raise ValueError("scored expected state fingerprint is inconsistent")
        object.__setattr__(self, "expected_state", state)
        trials = tuple(self.trials)
        if not trials:
            raise ValueError("E1 plan must contain trials")
        if len({trial.trial_id for trial in trials}) != len(trials):
            raise ValueError("E1 trial IDs must be unique")
        expected_modes = set(SpatialConditioningMode)
        if {trial.mode for trial in trials} != expected_modes:
            raise ValueError("E1 plan must contain every conditioning mode")
        target_set: set[str] | None = None
        for mode in expected_modes:
            rows = [trial for trial in trials if trial.mode is mode]
            if len(rows) != 2:
                raise ValueError("E1 plan requires two referents per conditioning mode")
            current_targets = {trial.target_object for trial in rows}
            if len(current_targets) != 2:
                raise ValueError("each E1 mode must test two distinct target objects")
            if len({trial.seed for trial in rows}) != 1:
                raise ValueError("the paired E1 referents must share one model seed")
            if target_set is None:
                target_set = current_targets
            elif current_targets != target_set:
                raise ValueError("every E1 mode must test the same referent pair")
        if len({trial.seed for trial in trials}) != 1:
            raise ValueError("all paired E1 conditions must use common random numbers")
        assert target_set is not None
        for target in target_set:
            rows = [trial for trial in trials if trial.target_object == target]
            physical_bindings = {
                (
                    trial.candidate_id,
                    trial.referent,
                    trial.grounding_camera,
                    trial.box_xyxy,
                    trial.point_xy,
                )
                for trial in rows
            }
            if len(rows) != len(expected_modes) or len(physical_bindings) != 1:
                raise ValueError(
                    "one physical referent must keep one candidate identity "
                    "and grounding across all interface conditions"
                )
        object.__setattr__(self, "trials", trials)
        execution_order = tuple(
            _clean_text(item, name="execution-order trial ID")
            for item in self.execution_order
        )
        trial_ids = {trial.trial_id for trial in trials}
        if len(execution_order) != len(trials) or set(execution_order) != trial_ids:
            raise ValueError("execution_order must list every E1 trial exactly once")
        object.__setattr__(self, "execution_order", execution_order)

    @staticmethod
    def _fingerprinted_payload(value: dict[str, Any]) -> dict[str, Any]:
        return {
            key: child for key, child in value.items() if key != "frozen_plan_sha256"
        }

    @classmethod
    def load(cls, path: str | Path) -> E1Plan:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        value = _require_exact_keys(
            value,
            expected={
                "schema_version",
                "plan_id",
                "frozen_plan_sha256",
                "runner_source",
                "model_canary",
                "prompt",
                "simulator",
                "public_observation",
                "executor",
                "execution",
                "trials",
            },
            name="E1 plan",
        )
        if value.get("schema_version") != E1_PLAN_SCHEMA:
            raise ValueError("unsupported E1 plan schema")
        expected = canonical_sha256(cls._fingerprinted_payload(value))
        if value.get("frozen_plan_sha256") != expected:
            raise ValueError(
                "E1 plan fingerprint mismatch; freeze a new plan instead of editing it"
            )
        simulator = _require_exact_keys(
            value["simulator"],
            expected={
                "libero_git_revision",
                "bddl_relative_path",
                "bddl_sha256",
                "init_states_relative_path",
                "init_states_sha256",
                "init_state_index",
                "env_seed",
                "settle_steps",
                "control_mode",
                "reset_state_sha256",
                "runtime",
            },
            name="E1 simulator contract",
        )
        runner_source = _require_exact_keys(
            value["runner_source"],
            expected={"schema_version", "files", "tree_sha256"},
            name="E1 runner source contract",
        )
        raw_source_files = runner_source["files"]
        if not isinstance(raw_source_files, list):
            raise TypeError("E1 runner source files must be a list")
        source_files: list[tuple[str, str]] = []
        for index, raw in enumerate(raw_source_files):
            item = _require_exact_keys(
                raw,
                expected={"path", "sha256"},
                name=f"E1 runner source file {index}",
            )
            source_files.append((str(item["path"]), str(item["sha256"])))
        observation = _require_exact_keys(
            value["public_observation"],
            expected={
                "image_size",
                "expected_agentview_sha256",
                "expected_wrist_sha256",
                "expected_state",
                "expected_state_sha256",
            },
            name="E1 public observation contract",
        )
        execution = _require_exact_keys(
            value["execution"],
            expected={
                "horizon",
                "execution_policy_id",
                "canonical_run_root",
                "execution_order",
            },
            name="E1 execution contract",
        )
        _require_exact_keys(
            value["executor"],
            expected={
                "checkpoint_id",
                "checkpoint_revision",
                "upstream_code_revision",
                "dtype",
                "device",
                "norm_tag",
                "inference_action_mode",
                "normalize_language",
                "camera_order",
                "image_size",
                "state_dim",
                "action_dim",
                "action_horizon",
                "num_steps",
                "enable_depth_reasoning",
                "enable_cuda_graph",
                "config_sha256",
                "norm_stats_sha256",
                "checkpoint_manifest_sha256",
                "checkpoint_file_count",
                "checkpoint_total_bytes",
                "norm_stats_format",
                "norm_mode",
                "control_mode",
            },
            name="E1 executor contract",
        )
        return cls(
            plan_id=str(value["plan_id"]),
            runner_source_schema=str(runner_source["schema_version"]),
            runner_source_files=tuple(source_files),
            runner_source_tree_sha256=str(runner_source["tree_sha256"]),
            libero_git_revision=str(simulator["libero_git_revision"]),
            bddl_relative_path=str(simulator["bddl_relative_path"]),
            bddl_sha256=str(simulator["bddl_sha256"]),
            init_states_relative_path=str(simulator["init_states_relative_path"]),
            init_states_sha256=str(simulator["init_states_sha256"]),
            init_state_index=int(simulator["init_state_index"]),
            env_seed=int(simulator["env_seed"]),
            reset_state_sha256=str(simulator["reset_state_sha256"]),
            expected_agentview_sha256=str(observation["expected_agentview_sha256"]),
            expected_wrist_sha256=str(observation["expected_wrist_sha256"]),
            expected_state=tuple(float(item) for item in observation["expected_state"]),
            expected_state_sha256=str(observation["expected_state_sha256"]),
            image_size=int(observation["image_size"]),
            settle_steps=int(simulator["settle_steps"]),
            control_mode=str(simulator["control_mode"]),
            simulator_runtime={
                str(key): str(child) for key, child in simulator["runtime"].items()
            },
            model_canary=E1ModelCanary.from_mapping(value["model_canary"]),
            prompt=str(value["prompt"]),
            horizon=int(execution["horizon"]),
            execution_policy_id=str(execution["execution_policy_id"]),
            canonical_run_root=str(execution["canonical_run_root"]),
            execution_order=tuple(str(item) for item in execution["execution_order"]),
            server_identity=MolmoAct2ServerIdentity.from_mapping(value["executor"]),
            trials=tuple(E1Trial.from_mapping(item) for item in value["trials"]),
            frozen_plan_sha256=str(value["frozen_plan_sha256"]),
        )

    def trial(self, trial_id: str) -> E1Trial:
        matches = [item for item in self.trials if item.trial_id == trial_id]
        if len(matches) != 1:
            raise ValueError(f"unknown or ambiguous E1 trial {trial_id!r}")
        return matches[0]


def _libero_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repository_identity() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"git_revision": revision, "dirty": dirty}


def _canonical_run_root(plan: E1Plan) -> Path:
    repository = Path(__file__).resolve().parents[2]
    root = (repository / plan.canonical_run_root).resolve()
    if repository not in root.parents:
        raise ValueError("canonical E1 run root escapes the repository")
    return root


def _canonical_canary_path(plan: E1Plan) -> Path:
    repository = Path(__file__).resolve().parents[2]
    path = (repository / plan.model_canary.report_relative_path).resolve()
    if repository not in path.parents:
        raise ValueError("canonical E1 canary path escapes the repository")
    return path


def _model_canary_session_id(plan: E1Plan) -> str:
    return canonical_sha256(
        {
            "schema_version": "e1-model-canary-session-v1",
            "plan_sha256": plan.frozen_plan_sha256,
            "init_state_index": plan.model_canary.init_state_index,
        }
    )


def validate_e1_model_canary_report(plan: E1Plan) -> dict[str, object]:
    """Validate the one outcome-free live-model canary required by E1."""

    path = _canonical_canary_path(plan)
    if not path.is_file():
        raise FileNotFoundError(
            f"E1 live-model canary is pending: {path}; run --run-model-canary first"
        )
    report = _read_json(path)
    report = _require_exact_keys(
        report,
        expected={
            "schema_version",
            "status",
            "plan_id",
            "plan_sha256",
            "outcome_bearing",
            "init_state_index",
            "env_seed",
            "reset_state_sha256",
            "agentview_sha256",
            "wrist_sha256",
            "state",
            "state_sha256",
            "instruction",
            "model_seed",
            "session_id",
            "request_payload",
            "action_chunk",
            "action_chunk_sha256",
            "actions_applied",
            "server_health",
            "runner_repository",
            "report_sha256",
        },
        name="E1 model canary report",
    )
    fingerprint = report.pop("report_sha256")
    if fingerprint != canonical_sha256(report):
        raise ValueError("E1 model canary report fingerprint mismatch")
    canary = plan.model_canary
    expected_scalar = {
        "schema_version": "e1-live-model-canary-v1",
        "status": "PASSED",
        "plan_id": plan.plan_id,
        "plan_sha256": plan.frozen_plan_sha256,
        "outcome_bearing": False,
        "init_state_index": canary.init_state_index,
        "env_seed": canary.env_seed,
        "reset_state_sha256": canary.reset_state_sha256,
        "agentview_sha256": canary.expected_agentview_sha256,
        "wrist_sha256": canary.expected_wrist_sha256,
        "state_sha256": canary.expected_state_sha256,
        "instruction": canary.instruction,
        "model_seed": canary.model_seed,
        "session_id": _model_canary_session_id(plan),
        "actions_applied": 0,
    }
    mismatches = [
        key for key, value in expected_scalar.items() if report.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "E1 model canary does not match the frozen plan: " + ", ".join(mismatches)
        )
    if tuple(float(item) for item in report["state"]) != canary.expected_state:
        raise ValueError("E1 model canary state values changed")
    health = report["server_health"]
    if not isinstance(health, dict):
        raise TypeError("E1 model canary server health must be an object")
    observed_identity = MolmoAct2ServerIdentity.from_mapping(health["identity"])
    if (
        observed_identity.digest != plan.server_identity.digest
        or health.get("identity_sha256") != plan.server_identity.digest
    ):
        raise ValueError("E1 model canary used a different executor identity")
    live, failures = _live_backend_attestation(
        health, expected_identity=plan.server_identity
    )
    if not live:
        raise ValueError(f"E1 model canary backend is not live: {list(failures)}")
    expected_request = molmoact2_public_request_payload(
        agentview_rgb_sha256=canary.expected_agentview_sha256,
        wrist_rgb_sha256=canary.expected_wrist_sha256,
        state=canary.expected_state,
        instruction=canary.instruction,
        session_id=_model_canary_session_id(plan),
        seed=canary.model_seed,
        expected_identity=plan.server_identity,
    )
    if report["request_payload"] != expected_request:
        raise ValueError("E1 model canary request is not reproducible from the plan")
    raw_chunk = report["action_chunk"]
    if not isinstance(raw_chunk, dict):
        raise TypeError("E1 model canary action chunk must be an object")
    chunk = MolmoAct2ActionChunk(
        request_id=str(raw_chunk["request_id"]),
        server_identity_sha256=str(raw_chunk["server_identity_sha256"]),
        actions=tuple(tuple(row) for row in raw_chunk["actions"]),
        latency_ms=int(raw_chunk["latency_ms"]),
    )
    if chunk.request_id != canonical_sha256(expected_request):
        raise ValueError("E1 model canary action chunk does not match request")
    if chunk.server_identity_sha256 != plan.server_identity.digest:
        raise ValueError("E1 model canary action chunk came from another executor")
    if report["action_chunk_sha256"] != canonical_sha256(chunk.to_dict()):
        raise ValueError("E1 model canary action chunk fingerprint mismatch")
    return {
        "status": "VERIFIED_PASSED",
        "report_path": str(path),
        "report_sha256": fingerprint,
        "action_chunk_sha256": report["action_chunk_sha256"],
        "actions_applied": 0,
        "outcomes_generated": 0,
    }


def run_e1_model_canary(
    *,
    plan: E1Plan,
    libero_root: str | Path,
    endpoint: str,
) -> dict[str, object]:
    """Run one real model call on an excluded state and apply zero actions."""

    repository = _repository_identity()
    if repository["dirty"] is not False:
        raise RuntimeError("E1 model canary requires a clean repository checkout")
    _verify_runner_source(plan)
    output_path = _canonical_canary_path(plan)
    if output_path.exists():
        raise FileExistsError(
            f"single-use E1 model canary already exists: {output_path}"
        )
    bddl, init_states = _resolve_libero_paths(plan, Path(libero_root))
    client = MolmoAct2HTTPClient(endpoint, expected_identity=plan.server_identity)
    health = client.health()
    live, failures = _live_backend_attestation(
        health, expected_identity=plan.server_identity
    )
    if not live:
        raise MolmoAct2ProtocolError(
            f"E1 canary requires an attested live backend; failures={list(failures)}"
        )
    canary = plan.model_canary
    with LiberoE1Environment(
        bddl_file=bddl,
        init_states_file=init_states,
        init_state_index=canary.init_state_index,
        env_seed=canary.env_seed,
        expected_reset_state_sha256=canary.reset_state_sha256,
        image_size=plan.image_size,
        settle_steps=plan.settle_steps,
        control_mode=plan.control_mode,
    ) as environment:
        public = environment.public_observation
        agent_digest = canonical_rgb_sha256(public.agentview_rgb)
        wrist_digest = canonical_rgb_sha256(public.wrist_rgb)
        state_digest = canonical_libero_state_sha256(public.state)
        if (
            agent_digest != canary.expected_agentview_sha256
            or wrist_digest != canary.expected_wrist_sha256
            or tuple(public.state) != canary.expected_state
            or state_digest != canary.expected_state_sha256
        ):
            raise RuntimeError("E1 model canary observation does not match frozen plan")
        session_id = _model_canary_session_id(plan)
        request_payload = molmoact2_public_request_payload(
            agentview_rgb_sha256=agent_digest,
            wrist_rgb_sha256=wrist_digest,
            state=public.state,
            instruction=canary.instruction,
            session_id=session_id,
            seed=canary.model_seed,
            expected_identity=plan.server_identity,
        )
        client.reset(session_id)
        chunk = client.predict_action_chunk(
            agentview_rgb=public.agentview_rgb,
            wrist_rgb=public.wrist_rgb,
            state=public.state,
            instruction=canary.instruction,
            session_id=session_id,
            seed=canary.model_seed,
        )
        if environment.step_count != 0:
            raise RuntimeError("E1 model canary must not apply simulator actions")
    payload: dict[str, object] = {
        "schema_version": "e1-live-model-canary-v1",
        "status": "PASSED",
        "plan_id": plan.plan_id,
        "plan_sha256": plan.frozen_plan_sha256,
        "outcome_bearing": False,
        "init_state_index": canary.init_state_index,
        "env_seed": canary.env_seed,
        "reset_state_sha256": canary.reset_state_sha256,
        "agentview_sha256": agent_digest,
        "wrist_sha256": wrist_digest,
        "state": list(canary.expected_state),
        "state_sha256": state_digest,
        "instruction": canary.instruction,
        "model_seed": canary.model_seed,
        "session_id": session_id,
        "request_payload": request_payload,
        "action_chunk": chunk.to_dict(),
        "action_chunk_sha256": canonical_sha256(chunk.to_dict()),
        "actions_applied": 0,
        "server_health": health,
        "runner_repository": repository,
    }
    payload["report_sha256"] = canonical_sha256(payload)
    _write_json_once(output_path, payload)
    return validate_e1_model_canary_report(plan)


def e1_ledger_status(plan: E1Plan) -> dict[str, object]:
    """Return the single canonical next index without creating artifacts."""

    _verify_runner_source(plan)
    validate_e1_model_canary_report(plan)
    run_root = _canonical_run_root(plan) / plan.plan_id
    consumed: list[str] = []
    blocked_by_failure: str | None = None
    for index, trial_id in enumerate(plan.execution_order):
        path = run_root / trial_id
        if not path.exists():
            later = [
                child
                for child in plan.execution_order[index + 1 :]
                if (run_root / child).exists()
            ]
            if later:
                raise RuntimeError("E1 canonical ledger has out-of-order artifacts")
            return {
                "consumed": consumed,
                "next_execution_index": index,
                "next_trial_id": trial_id,
                "blocked_by_failure": blocked_by_failure,
            }
        validate_e1_artifacts(
            path,
            require_empirical=True,
            expected_plan=plan,
            expected_trial_id=trial_id,
        )
        consumed.append(trial_id)
        if (path / "failed.json").is_file():
            blocked_by_failure = trial_id
            break
    complete = len(consumed) == len(plan.execution_order)
    return {
        "consumed": consumed,
        "next_execution_index": (
            None if blocked_by_failure or complete else len(consumed)
        ),
        "next_trial_id": None,
        "blocked_by_failure": blocked_by_failure,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_runner_source(plan: E1Plan) -> dict[str, object]:
    """Require the outcome-generating source bytes frozen by the E1 plan."""

    repository = Path(__file__).resolve().parents[2]
    observed: list[dict[str, str]] = []
    for relative, expected_digest in plan.runner_source_files:
        path = (repository / relative).resolve()
        if path != repository and repository not in path.parents:
            raise ValueError("runner source path escapes the repository")
        if not path.is_file():
            raise FileNotFoundError(f"frozen runner source is missing: {relative}")
        digest = _file_sha256(path)
        if digest != expected_digest:
            raise RuntimeError(
                f"runner source bytes do not match frozen plan: {relative}"
            )
        observed.append({"path": relative, "sha256": digest})
    tree_sha256 = canonical_sha256(
        {"schema_version": plan.runner_source_schema, "files": observed}
    )
    if tree_sha256 != plan.runner_source_tree_sha256:
        raise RuntimeError("runner source tree does not match frozen E1 plan")
    return {
        "schema_version": plan.runner_source_schema,
        "tree_sha256": tree_sha256,
        "file_count": len(observed),
    }


def _resolve_libero_paths(plan: E1Plan, libero_root: Path) -> tuple[Path, Path]:
    root = libero_root.expanduser().resolve()
    if _libero_revision(root) != plan.libero_git_revision:
        raise RuntimeError("LIBERO checkout revision does not match the frozen E1 plan")
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty.strip():
        raise RuntimeError("LIBERO checkout must be clean for frozen E1 execution")
    bddl = (root / plan.bddl_relative_path).resolve()
    init_states = (root / plan.init_states_relative_path).resolve()
    for path in (bddl, init_states):
        if root not in path.parents:
            raise ValueError("E1 asset path escapes the LIBERO checkout")
        if not path.is_file():
            raise FileNotFoundError(path)
    if _file_sha256(bddl) != plan.bddl_sha256:
        raise RuntimeError("E1 BDDL bytes do not match the frozen plan")
    if _file_sha256(init_states) != plan.init_states_sha256:
        raise RuntimeError("E1 initial-state bytes do not match the frozen plan")
    observed_runtime = libero_runtime_identity()
    if observed_runtime != plan.simulator_runtime:
        raise RuntimeError(
            "LIBERO runtime identity mismatch: "
            f"expected {plan.simulator_runtime}, observed {observed_runtime}"
        )
    return bddl, init_states


def _build_context_and_candidate(
    *,
    plan: E1Plan,
    trial: E1Trial,
    environment: LiberoE1Environment,
    frame_store: RGBFrameStore,
) -> tuple[PolicyContext, GroundedIntervention]:
    public = environment.public_observation
    observed_agent_digest = canonical_rgb_sha256(public.agentview_rgb)
    observed_wrist_digest = canonical_rgb_sha256(public.wrist_rgb)
    if observed_agent_digest != plan.expected_agentview_sha256:
        raise RuntimeError("initial agentview RGB does not match frozen E1 plan")
    if observed_wrist_digest != plan.expected_wrist_sha256:
        raise RuntimeError("initial wrist RGB does not match frozen E1 plan")
    if tuple(public.state) != plan.expected_state:
        raise RuntimeError("initial public state values do not match frozen E1 plan")
    if canonical_libero_state_sha256(public.state) != plan.expected_state_sha256:
        raise RuntimeError("initial public state digest does not match frozen E1 plan")
    agent_frame = frame_store.put(
        public.agentview_rgb,
        frame_id=(f"{plan.plan_id}:state-{plan.init_state_index}:pre:agentview"),
        camera=trial.grounding_camera,
        frame_index=0,
    )
    wrist_frame = frame_store.put(
        public.wrist_rgb,
        frame_id=f"{plan.plan_id}:state-{plan.init_state_index}:pre:wrist",
        camera="wrist",
        frame_index=0,
    )
    context = PolicyContext(
        prompt=plan.prompt,
        frames=(agent_frame, wrist_frame),
        proprioception=public.state,
    )
    grounding = GroundingReference(
        camera="agentview",
        frame_id=agent_frame.frame_id,
        frame_index=agent_frame.frame_index,
        image_sha256=agent_frame.image_sha256,
        box_xyxy=trial.box_xyxy,
        point_xy=trial.point_xy,
    )
    candidate = GroundedIntervention(
        candidate_id=trial.candidate_id,
        primitive=Primitive.DIRECT,
        referent=trial.referent,
        parameters=(("destination", "stove cook region"),),
        grounding=grounding,
    )
    candidate.validate_against(context)
    return context, candidate


def validate_e1_runtime_preflight(
    plan: E1Plan, *, libero_root: str | Path
) -> dict[str, object]:
    """Reload the exact reset and verify both real public RGB observations."""

    runner_source = _verify_runner_source(plan)
    bddl, init_states = _resolve_libero_paths(plan, Path(libero_root))
    with LiberoE1Environment(
        bddl_file=bddl,
        init_states_file=init_states,
        init_state_index=plan.init_state_index,
        env_seed=plan.env_seed,
        expected_reset_state_sha256=plan.reset_state_sha256,
        image_size=plan.image_size,
        settle_steps=plan.settle_steps,
        control_mode=plan.control_mode,
    ) as environment:
        agentview_sha256 = canonical_rgb_sha256(
            environment.public_observation.agentview_rgb
        )
        wrist_sha256 = canonical_rgb_sha256(environment.public_observation.wrist_rgb)
        if agentview_sha256 != plan.expected_agentview_sha256:
            raise RuntimeError("preflight agentview RGB does not match frozen E1 plan")
        if wrist_sha256 != plan.expected_wrist_sha256:
            raise RuntimeError("preflight wrist RGB does not match frozen E1 plan")
        state = environment.public_observation.state
        state_sha256 = canonical_libero_state_sha256(state)
        if (
            tuple(state) != plan.expected_state
            or state_sha256 != plan.expected_state_sha256
        ):
            raise RuntimeError("preflight public state does not match frozen E1 plan")
        scored = {
            "init_state_index": plan.init_state_index,
            "reset_state_sha256": environment.reset_state_sha256,
            "agentview_sha256": agentview_sha256,
            "wrist_sha256": wrist_sha256,
            "state": list(state),
            "state_sha256": state_sha256,
            "control_mode": environment.control_mode,
        }
    canary = plan.model_canary
    with LiberoE1Environment(
        bddl_file=bddl,
        init_states_file=init_states,
        init_state_index=canary.init_state_index,
        env_seed=canary.env_seed,
        expected_reset_state_sha256=canary.reset_state_sha256,
        image_size=plan.image_size,
        settle_steps=plan.settle_steps,
        control_mode=plan.control_mode,
    ) as environment:
        canary_agent = canonical_rgb_sha256(
            environment.public_observation.agentview_rgb
        )
        canary_wrist = canonical_rgb_sha256(environment.public_observation.wrist_rgb)
        canary_state = environment.public_observation.state
        canary_state_sha = canonical_libero_state_sha256(canary_state)
        if (
            canary_agent != canary.expected_agentview_sha256
            or canary_wrist != canary.expected_wrist_sha256
            or tuple(canary_state) != canary.expected_state
            or canary_state_sha != canary.expected_state_sha256
        ):
            raise RuntimeError(
                "preflight canary observation does not match frozen plan"
            )
        canary_observation = {
            "init_state_index": canary.init_state_index,
            "reset_state_sha256": environment.reset_state_sha256,
            "agentview_sha256": canary_agent,
            "wrist_sha256": canary_wrist,
            "state": list(canary_state),
            "state_sha256": canary_state_sha,
            "control_mode": environment.control_mode,
        }
    return {
        "status": "VALID",
        "plan_id": plan.plan_id,
        "plan_sha256": plan.frozen_plan_sha256,
        "scored_observation": scored,
        "canary_observation": canary_observation,
        "simulator_runtime": dict(plan.simulator_runtime),
        "runner_source": runner_source,
        "outcomes_generated": 0,
    }


def _outcome_contract(
    plan: E1Plan,
    trial: E1Trial,
) -> OutcomeContract:
    return OutcomeContract(
        outcome_name="first_candidate_contact_is_exclusively_intended_within_300_steps",
        continuation_policy_id=plan.execution_policy_id,
        horizon=plan.horizon,
        executor_id=f"molmoact2-libero-http-v1@{plan.server_identity.digest}",
        serializer_id=_SERIALIZER_IDS[trial.mode],
        failure_handling=(
            "any_policy_output_or_infrastructure_failure_produces_no_label; "
            "first_candidate_contact_not_exclusively_intended_is_negative"
        ),
    )


def run_e1_trial(
    *,
    plan: E1Plan,
    trial_id: str,
    libero_root: str | Path,
    endpoint: str,
    output_root: str | Path,
    allow_test_backend: bool = False,
) -> dict[str, object]:
    """Run exactly one preregistered branch and export a trainable result."""

    _verify_runner_source(plan)
    trial = plan.trial(trial_id)
    output_dir = (
        Path(output_root).expanduser().resolve() / plan.plan_id / trial.trial_id
    )
    if output_dir.exists():
        raise FileExistsError(
            f"single-use E1 output already exists: {output_dir}; never overwrite or rerun"
        )
    output_dir.mkdir(parents=True)
    _write_json(
        output_dir / "started.json",
        {
            "status": "STARTED",
            "plan_id": plan.plan_id,
            "plan_sha256": plan.frozen_plan_sha256,
            "trial_id": trial.trial_id,
            "runner_repository": _repository_identity(),
        },
    )

    try:
        bddl, init_states = _resolve_libero_paths(plan, Path(libero_root))
        client = MolmoAct2HTTPClient(endpoint, expected_identity=plan.server_identity)
        health = client.health()
        live_backend, attestation_failures = _live_backend_attestation(
            health, expected_identity=plan.server_identity
        )
        if not live_backend and not allow_test_backend:
            raise MolmoAct2ProtocolError(
                "formal E1 execution requires an attested live MolmoAct2 backend; "
                f"failures={list(attestation_failures)}"
            )
        frame_store = RGBFrameStore(output_dir / "public_frames")
        all_actions: list[list[float]] = []
        chunks: list[dict[str, object]] = []
        milestones: list[dict[str, object]] = []
        first_contact: tuple[str, ...] | None = None
        evaluator_trace: list[dict[str, object]] = []
        with LiberoE1Environment(
            bddl_file=bddl,
            init_states_file=init_states,
            init_state_index=plan.init_state_index,
            env_seed=plan.env_seed,
            expected_reset_state_sha256=plan.reset_state_sha256,
            image_size=plan.image_size,
            settle_steps=plan.settle_steps,
            control_mode=plan.control_mode,
        ) as environment:
            context, candidate = _build_context_and_candidate(
                plan=plan,
                trial=trial,
                environment=environment,
                frame_store=frame_store,
            )
            serializer = GroundedTextSerializer(
                _SERIALIZER_IDS[trial.mode], spatial_mode=trial.mode
            )
            serialized = serializer.serialize(candidate, context)
            request = ExecutorRequest.from_serialized(serialized)
            initial_conditioned = ReferentConditioner(frame_store).prepare(
                context=context,
                intervention=candidate,
                serialized=serialized,
                mode=trial.mode,
            )
            conditioned_store = RGBFrameStore(output_dir / "conditioned_frames")
            conditioned_store.put(
                initial_conditioned.agentview_rgb,
                frame_id=f"{trial.trial_id}:conditioned:agentview",
                camera="agentview",
                frame_index=0,
            )
            conditioned_store.put(
                initial_conditioned.wrist_rgb,
                frame_id=f"{trial.trial_id}:conditioned:wrist",
                camera="wrist",
                frame_index=0,
            )
            session_id = canonical_sha256(
                {
                    "plan": plan.frozen_plan_sha256,
                    "trial": trial.trial_id,
                    "candidate": candidate.fingerprint(),
                }
            )
            client.reset(session_id)
            candidate_objects = (trial.target_object,) + trial.distractor_objects
            public = environment.public_observation
            while environment.step_count < plan.horizon:
                # The executed conditioning is fixed for the complete branch.
                # Evaluator-only contact / grasp / predicate state is logged below,
                # but may not change language, pixels, actions, or termination.
                instruction = serialized.subtask_text
                agentview = public.agentview_rgb
                wrist = public.wrist_rgb
                marker_applied = trial.mode is SpatialConditioningMode.VISUAL_MARKER
                if marker_applied:
                    agentview = draw_public_grounding_marker(
                        agentview,
                        box_xyxy=trial.box_xyxy,
                        point_xy=trial.point_xy,
                    )
                chunk_index = len(chunks)
                request_state = list(public.state)
                request_state_sha256 = canonical_libero_state_sha256(public.state)
                request_agentview_sha256 = canonical_rgb_sha256(agentview)
                request_wrist_sha256 = canonical_rgb_sha256(wrist)
                conditioned_agent = conditioned_store.put(
                    agentview,
                    frame_id=(f"{trial.trial_id}:conditioned:{chunk_index}:agentview"),
                    camera="agentview",
                    frame_index=chunk_index,
                )
                conditioned_wrist = conditioned_store.put(
                    wrist,
                    frame_id=f"{trial.trial_id}:conditioned:{chunk_index}:wrist",
                    camera="wrist",
                    frame_index=chunk_index,
                )
                action_chunk = client.predict_action_chunk(
                    agentview_rgb=agentview,
                    wrist_rgb=wrist,
                    state=public.state,
                    instruction=instruction,
                    session_id=session_id,
                    seed=trial.seed + chunk_index,
                )
                applied_in_chunk = 0
                for action in action_chunk.actions:
                    if environment.step_count >= plan.horizon:
                        break
                    public, applied = environment.step(action)
                    all_actions.append(list(applied))
                    applied_in_chunk += 1
                    contacts = environment.contacts(candidate_objects)
                    grasping_now = environment.is_grasping(trial.target_object)
                    goal_now = environment.predicate(trial.target_predicate)
                    evaluator_trace.append(
                        {
                            "step": environment.step_count,
                            "contact_objects": list(contacts),
                            "target_grasped": grasping_now,
                            "target_predicate": goal_now,
                        }
                    )
                    if contacts and first_contact is None:
                        first_contact = contacts
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "status": "COMPLETED",
                        "request_id": action_chunk.request_id,
                        "server_identity_sha256": (action_chunk.server_identity_sha256),
                        "seed": trial.seed + chunk_index,
                        "instruction": instruction,
                        "marker_applied": marker_applied,
                        "input_agentview_sha256": request_agentview_sha256,
                        "input_wrist_sha256": request_wrist_sha256,
                        "input_agentview_frame": conditioned_agent.to_dict(),
                        "input_wrist_frame": conditioned_wrist.to_dict(),
                        "input_state": request_state,
                        "input_state_sha256": request_state_sha256,
                        "actions_returned": len(action_chunk.actions),
                        "actions_applied": applied_in_chunk,
                        "action_start_index": len(all_actions) - applied_in_chunk,
                        "actions": [
                            [float(item) for item in row]
                            for row in action_chunk.actions
                        ],
                        "latency_ms": action_chunk.latency_ms,
                    }
                )
                milestone_agent = frame_store.put(
                    public.agentview_rgb,
                    frame_id=f"{trial.trial_id}:milestone:{chunk_index}:agentview",
                    camera="agentview",
                    frame_index=chunk_index + 1,
                )
                milestone_wrist = frame_store.put(
                    public.wrist_rgb,
                    frame_id=f"{trial.trial_id}:milestone:{chunk_index}:wrist",
                    camera="wrist",
                    frame_index=chunk_index + 1,
                )
                milestones.append(
                    {
                        "chunk_index": chunk_index,
                        "agentview": milestone_agent.to_dict(),
                        "wrist": milestone_wrist.to_dict(),
                        "state": list(public.state),
                        "state_sha256": canonical_libero_state_sha256(public.state),
                    }
                )
            final_agent = frame_store.put(
                public.agentview_rgb,
                frame_id=f"{trial.trial_id}:post:agentview",
                camera="agentview",
                frame_index=len(chunks) + 1,
            )
            final_wrist = frame_store.put(
                public.wrist_rgb,
                frame_id=f"{trial.trial_id}:post:wrist",
                camera="wrist",
                frame_index=len(chunks) + 1,
            )
            diagnostics, derived_first_contact = _derive_e1_diagnostics(
                trial=trial,
                per_step=evaluator_trace,
                chunks=chunks,
                horizon=plan.horizon,
            )
            if derived_first_contact != tuple(first_contact or ()):
                raise RuntimeError(
                    "online and replayed E1 first-contact logic disagree"
                )
            intended_first = bool(diagnostics["first_contact_intended"])
            execution_status = ExecutionStatus.COMPLETED
            public_trace = {
                "schema_version": "e1-public-execution-trace-v1",
                "plan_id": plan.plan_id,
                "plan_sha256": plan.frozen_plan_sha256,
                "trial_id": trial.trial_id,
                "execution_policy_id": plan.execution_policy_id,
                "server_health": health,
                "executor_request": request.to_dict(),
                "initial_conditioned_input": initial_conditioned.public_identity(),
                "action_chunks": chunks,
                "actions_applied": all_actions,
                "milestones": milestones,
                "post_frames": [final_agent.to_dict(), final_wrist.to_dict()],
                "execution_status": execution_status.value,
            }
            public_trace_sha256 = canonical_sha256(public_trace)
            private_trace = {
                "schema_version": "e1-private-evaluator-trace-v1",
                "target_object": trial.target_object,
                "distractor_objects": list(trial.distractor_objects),
                "target_predicate": list(trial.target_predicate),
                "per_step": evaluator_trace,
                "summary": diagnostics,
            }
            private_trace_sha256 = canonical_sha256(private_trace)
            receipt_id = canonical_sha256(
                {
                    "request_digest": request.request_digest,
                    "server_identity": plan.server_identity.digest,
                    "public_execution_trace_sha256": public_trace_sha256,
                    "private_evaluator_trace_sha256": private_trace_sha256,
                }
            )
            receipt = ExecutorReceipt(
                receipt_id=receipt_id,
                executor_id=f"molmoact2-libero-http-v1@{plan.server_identity.digest}",
                candidate_id=candidate.candidate_id,
                candidate_fingerprint=candidate.fingerprint(),
                request_digest=request.request_digest,
                status=execution_status,
                post_frames=(final_agent, final_wrist),
            )
            receipt.validate_request(request)
            branch = ObservedBranch(
                branch_id=f"{plan.plan_id}:{trial.trial_id}",
                initial_state_group=f"{plan.plan_id}:state-{plan.init_state_index}",
                decision_group_id=(
                    f"{plan.plan_id}:state-{plan.init_state_index}:{trial.mode.value}"
                ),
                split="pilot",
                reset_state_sha256=environment.reset_state_sha256,
                repeat_index=0,
                context=context,
                executed_intervention=candidate,
                post_action_frames=(final_agent, final_wrist),
                outcome_contract=_outcome_contract(plan, trial),
                observed_outcome=intended_first,
                execution_status=execution_status,
                execution_receipt_id=receipt_id,
                diagnostics=diagnostics,
                private_evaluator_metadata={
                    "target_object": trial.target_object,
                    "distractor_objects": list(trial.distractor_objects),
                    "target_predicate": list(trial.target_predicate),
                    "first_contact_objects": list(first_contact or ()),
                    "evaluator_trace_sha256": private_trace_sha256,
                },
            )
            _write_json(output_dir / "public_execution_trace.json", public_trace)
            _write_json(output_dir / "private" / "evaluator_trace.json", private_trace)
            _write_json(output_dir / "executor_request.json", request.to_dict())
            _write_json(output_dir / "executor_health.json", health)
            _write_json(
                output_dir / "conditioned_input.json",
                initial_conditioned.public_identity(),
            )
            _write_json(output_dir / "action_chunks.json", chunks)
            _write_json(output_dir / "actions_applied.json", all_actions)
            _write_json(output_dir / "milestones.json", milestones)
            _write_json(output_dir / "execution_receipt.json", receipt.to_dict())
            _write_json(
                output_dir / "observed_branch.public.json",
                branch.to_dict(include_private=False),
            )
            _write_json(
                output_dir / "private" / "evaluator_sidecar.json",
                branch.to_dict(include_private=True)["private_evaluator_metadata"],
            )
            complete = {
                "status": "COMPLETED",
                "schema_version": MOLMOACT2_HTTP_SCHEMA,
                "plan_id": plan.plan_id,
                "plan_sha256": plan.frozen_plan_sha256,
                "trial_id": trial.trial_id,
                "conditioning_mode": trial.mode.value,
                "candidate_id": candidate.candidate_id,
                "candidate_fingerprint": candidate.fingerprint(),
                "request_digest": request.request_digest,
                "receipt_id": receipt_id,
                "execution_status": execution_status.value,
                "public_execution_trace_sha256": public_trace_sha256,
                "private_evaluator_trace_sha256": private_trace_sha256,
                "observed_branch_public_fingerprint": branch.public_fingerprint(),
                "diagnostics": diagnostics,
                "empirical_execution": live_backend,
                "autonomous_grounding": False,
                "stage1_selection_executed": False,
            }
            _write_json(output_dir / "completed.json", complete)
            seal = _seal_artifact_tree(output_dir)
            returned = dict(complete)
            returned["artifact_manifest_sha256"] = seal["manifest_sha256"]
            return returned
    except Exception as error:
        if (output_dir / "completed.json").exists():
            raise
        _write_json(
            output_dir / "failed.json",
            {
                "status": "INFRASTRUCTURE_FAILURE",
                "plan_id": plan.plan_id,
                "plan_sha256": plan.frozen_plan_sha256,
                "trial_id": trial.trial_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "observed_branch_generated": False,
            },
        )
        _seal_artifact_tree(output_dir)
        raise


def run_e1_index(
    *,
    plan: E1Plan,
    execution_index: int,
    libero_root: str | Path,
    endpoint: str,
) -> dict[str, object]:
    """Run only the next index in the plan's canonical single-use ledger."""

    if (
        not isinstance(execution_index, int)
        or isinstance(execution_index, bool)
        or not 0 <= execution_index < len(plan.execution_order)
    ):
        raise ValueError("execution_index is outside the frozen E1 schedule")
    repository_identity = _repository_identity()
    if repository_identity["dirty"] is not False:
        raise RuntimeError("formal E1 execution requires a clean repository checkout")
    _verify_runner_source(plan)
    ledger = e1_ledger_status(plan)
    if ledger["blocked_by_failure"] is not None:
        raise RuntimeError(
            "E1 is sealed after infrastructure failure; freeze an amendment, do not rerun"
        )
    if ledger["next_execution_index"] != execution_index:
        raise RuntimeError(
            "execution_index is not the next unused entry in the frozen E1 schedule"
        )
    trial_id = plan.execution_order[execution_index]
    return run_e1_trial(
        plan=plan,
        trial_id=trial_id,
        libero_root=libero_root,
        endpoint=endpoint,
        output_root=_canonical_run_root(plan),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen E1 execution branch")
    parser.add_argument("--plan")
    parser.add_argument("--execution-index", type=int)
    parser.add_argument("--libero-root", default=os.environ.get("LIBERO_REPO_ROOT"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8003")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validate-output")
    parser.add_argument("--trial-id")
    parser.add_argument("--validate-canary", action="store_true")
    parser.add_argument("--run-model-canary", action="store_true")
    parser.add_argument("--allow-model-canary", action="store_true")
    parser.add_argument("--allow-execution", action="store_true")
    args = parser.parse_args()
    if args.validate_output:
        if not args.plan or not args.trial_id:
            parser.error(
                "--validate-output requires --plan and --trial-id for semantic replay"
            )
        validation_plan = E1Plan.load(args.plan)
        print(
            json.dumps(
                validate_e1_artifacts(
                    args.validate_output,
                    expected_plan=validation_plan,
                    expected_trial_id=args.trial_id,
                ),
                sort_keys=True,
            )
        )
        return
    if not args.plan:
        parser.error("--plan is required unless --validate-output is used")
    plan = E1Plan.load(args.plan)
    if not args.libero_root:
        parser.error("--libero-root or LIBERO_REPO_ROOT is required")
    if args.validate_canary:
        print(json.dumps(validate_e1_model_canary_report(plan), sort_keys=True))
        return
    if args.run_model_canary:
        if not args.allow_model_canary:
            parser.error("model canary requires explicit --allow-model-canary")
        report = run_e1_model_canary(
            plan=plan,
            libero_root=args.libero_root,
            endpoint=args.endpoint,
        )
        print(json.dumps(report, sort_keys=True))
        return
    if args.validate_only:
        report = validate_e1_runtime_preflight(plan, libero_root=args.libero_root)
        report["trials"] = [trial.trial_id for trial in plan.trials]
        report["execution_order"] = list(plan.execution_order)
        try:
            report["model_canary"] = validate_e1_model_canary_report(plan)
            report["ledger"] = e1_ledger_status(plan)
        except FileNotFoundError:
            report["model_canary"] = {
                "status": "PENDING",
                "report_path": str(_canonical_canary_path(plan)),
            }
            report["ledger"] = {
                "status": "BLOCKED_PENDING_MODEL_CANARY",
                "next_execution_index": None,
            }
        print(json.dumps(report, sort_keys=True))
        return
    if not args.allow_execution:
        parser.error("formal execution requires explicit --allow-execution")
    if args.execution_index is None:
        parser.error("--execution-index is required for single-use execution")
    report = run_e1_index(
        plan=plan,
        execution_index=args.execution_index,
        libero_root=args.libero_root,
        endpoint=args.endpoint,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
