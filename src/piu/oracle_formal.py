"""Prospective, single-use formal oracle target-prompt experiment contracts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .executor_bridge import serialize_pi05_subtask
from .formal_design import summarize_paired_pilot
from .formal_states import validate_state_archive
from .policy_identity import load_checkpoint_identity, validate_server_metadata
from .primitive_registry import load_primitive_qualification_certificate
from .reproducibility import validate_repro_lock
from .splits import load_split_manifest
from .statistics import paired_binary_summary

PLAN_SCHEMA = "calibrated-interaction.oracle-formal-test-plan.v1"
INITIAL_STATE_SCHEMA = "piu.oracle-formal-initial-state-manifest.v1"
SCHEDULE_SCHEMA = "piu.oracle-formal-execution-schedule.v1"
RECEIPT_SCHEMA = "piu.oracle-formal-group-receipt.v1"
RESULT_SCHEMA = "piu.oracle-formal-result.v1"
ARMS = ("oracle_target_prompt", "raw_post_open_direct")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: Path, *, repository_root: Path) -> Path:
    return path if path.is_absolute() else repository_root / path


def portable(path: Path, *, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def artifact(path: Path, *, repository_root: Path) -> dict[str, str]:
    return {
        "path": portable(path, repository_root=repository_root),
        "sha256": sha256(path),
    }


def _verified_reference(
    value: Any, *, name: str, repository_root: Path
) -> Path:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} reference must be a mapping")
    raw = " ".join(str(value.get("path", "")).split())
    if not raw:
        raise ValueError(f"{name} reference lacks a path")
    path = resolve(Path(raw), repository_root=repository_root)
    if not path.is_file() or value.get("sha256") != sha256(path):
        raise ValueError(f"{name} artifact differs from its content hash")
    return path


def load_oracle_formal_plan(
    path: Path, *, repository_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate a frozen plan and the exact independent pilot that sized it."""

    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != PLAN_SCHEMA
        or value.get("claim_scope") != "DESIGN_ONLY_NO_FORMAL_OUTCOME_DATA"
    ):
        raise ValueError("unsupported oracle formal plan")
    protocol_path = _verified_reference(
        value.get("protocol"),
        name="oracle formal protocol",
        repository_root=repository_root,
    )
    config = yaml.safe_load(protocol_path.read_text())
    if (
        not isinstance(config, Mapping)
        or config.get("schema_version")
        != "calibrated-interaction.oracle-target-prompt-pilot.v2"
    ):
        raise ValueError("oracle formal plan uses another experiment protocol")
    pilot_path = _verified_reference(
        value.get("pilot"),
        name="oracle formal pilot",
        repository_root=repository_root,
    )
    pilot = json.loads(pilot_path.read_text())
    selected_style = pilot.get("confirmation", {}).get("selected_style")
    if (
        pilot.get("schema_version")
        != "calibrated-interaction.oracle-target-prompt-result.v1"
        or pilot.get("status") != "INDEPENDENT_DEVELOPMENT_PILOT_COMPLETE"
        or selected_style not in config.get("screen", {}).get("styles", ())
        or pilot.get("formal_method_claim") is not False
        or pilot.get("automatic_method_branch") is not False
    ):
        raise ValueError("oracle formal plan lacks a valid independent pilot")
    formal = config.get("formal_followup", {})
    if (
        value.get("test") != formal.get("test")
        or float(value.get("alpha")) != float(formal.get("alpha"))
        or float(value.get("target_power")) != float(formal.get("target_power"))
    ):
        raise ValueError("oracle formal plan differs from its frozen test")
    status = value.get("status")
    count = value.get("prospective_group_count")
    if status == "PROSPECTIVE_GROUP_COUNT_FROZEN":
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("oracle formal plan has no valid prospective count")
    elif status == "NO_PLAN_WITHIN_FROZEN_RESOURCE_CAP":
        if count is not None:
            raise ValueError("blocked oracle formal plan invents a group count")
    else:
        raise ValueError("oracle formal plan has an unsupported design status")
    if value.get("warning") != (
        "Pilot effect estimates determine design only. Pilot groups are excluded "
        "from the formal p-value and effect estimate."
    ):
        raise ValueError("oracle formal plan weakened pilot/formal isolation")
    return value, dict(config), pilot


def load_oracle_formal_initial_states(
    path: Path, *, repository_root: Path
) -> dict[str, Any]:
    """Validate exact, pre-outcome simulator states allocated to oracle formal."""

    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != INITIAL_STATE_SCHEMA
        or value.get("status") != "FROZEN_BEFORE_ORACLE_FORMAL_OUTCOMES"
        or value.get("claim_scope") != "OPAQUE_TRANSPORT_ONLY_NOT_POLICY_INPUT"
        or value.get("outcomes_loaded") is not False
    ):
        raise ValueError("unsupported oracle formal initial-state manifest")
    split_path = _verified_reference(
        value.get("split_manifest"),
        name="oracle formal split manifest",
        repository_root=repository_root,
    )
    split = load_split_manifest(split_path)
    formal_assignments = {
        str(row["initial_state_group"]): int(row["seed"])
        for row in split["assignments"]
        if row["split_role"] == "oracle_formal"
    }
    if not formal_assignments:
        raise ValueError("split manifest has no oracle-formal cohort")
    lock_reference = value.get("offline_repro_lock")
    lock_path = _verified_reference(
        lock_reference,
        name="oracle formal offline release lock",
        repository_root=repository_root,
    )
    manifest_path = repository_root / "configs/experiments/piu_offline_repro_v3.yaml"
    if lock_reference.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("oracle formal states use another reproduction manifest")
    validate_repro_lock(
        lock_path,
        manifest_path=manifest_path,
        repository_root=repository_root,
    )
    rows = value.get("states")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("oracle formal initial states must be a sequence")
    observed: dict[str, int] = {}
    digests: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("oracle formal state rows must be mappings")
        group = " ".join(str(row.get("initial_state_group", "")).split())
        seed = row.get("simulator_seed")
        if group in observed or formal_assignments.get(group) != seed:
            raise ValueError("oracle formal state group/seed differs from split")
        reference = row.get("source_state")
        state_path = _verified_reference(
            reference,
            name=f"oracle formal source state {group}",
            repository_root=repository_root,
        )
        state_key = " ".join(str(row.get("state_key", "")).split())
        shape, dtype = validate_state_archive(state_path, state_key=state_key)
        if reference.get("shape") != shape or reference.get("dtype") != dtype:
            raise ValueError("oracle formal state metadata differs")
        if reference["sha256"] in digests:
            raise ValueError("oracle formal groups reuse an opaque state")
        digests.add(str(reference["sha256"]))
        observed[group] = int(seed)
    if observed != formal_assignments:
        raise ValueError("oracle formal states differ from the split cohort")
    return value


def _key(namespace: str, binding: str, name: str) -> str:
    return hashlib.sha256(f"{namespace}\0{binding}\0{name}".encode()).hexdigest()


def load_oracle_formal_schedule(
    path: Path, *, repository_root: Path
) -> dict[str, Any]:
    """Revalidate an outcome-independent attempted-OPEN paired schedule."""

    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != SCHEDULE_SCHEMA
        or value.get("status") != "FROZEN_BEFORE_ORACLE_FORMAL_OUTCOMES"
        or value.get("claim_scope") != "CAUSAL_ORACLE_DESIGN_ONLY_NO_OUTCOMES"
        or value.get("outcomes_loaded") is not False
    ):
        raise ValueError("unsupported oracle formal execution schedule")
    inputs = value.get("inputs")
    if not isinstance(inputs, Mapping):
        raise TypeError("oracle formal schedule lacks input provenance")
    plan_path = _verified_reference(
        inputs.get("formal_plan"),
        name="oracle formal plan",
        repository_root=repository_root,
    )
    plan, config, pilot = load_oracle_formal_plan(
        plan_path, repository_root=repository_root
    )
    if plan["status"] != "PROSPECTIVE_GROUP_COUNT_FROZEN":
        raise ValueError("blocked oracle plan cannot produce an execution schedule")
    config_path = _verified_reference(
        inputs.get("experiment"),
        name="oracle experiment",
        repository_root=repository_root,
    )
    if sha256(config_path) != plan["protocol"]["sha256"]:
        raise ValueError("oracle formal schedule uses another experiment")
    formal_protocol_path = _verified_reference(
        inputs.get("formal_execution_protocol"),
        name="oracle formal execution protocol",
        repository_root=repository_root,
    )
    formal_protocol = yaml.safe_load(formal_protocol_path.read_text())
    if (
        not isinstance(formal_protocol, Mapping)
        or formal_protocol.get("schema_version")
        != "piu.oracle-formal-experiment.v1"
        or formal_protocol.get("status") != "frozen_before_formal_oracle_outcomes"
        or formal_protocol.get("claim_scope")
        != "privileged_causal_mechanism_only_not_public_method"
        or resolve(
            Path(str(formal_protocol.get("oracle_pilot_protocol", ""))),
            repository_root=repository_root,
        ).resolve()
        != config_path.resolve()
    ):
        raise ValueError("oracle formal execution protocol is not prospectively frozen")
    claims = formal_protocol.get("claim_contract", {})
    if claims != {
        "condition_on_successful_open": False,
        "reuse_screen_or_confirmation_groups": False,
        "incomplete_denominator_allowed": False,
        "interrupted_group_may_rerun": False,
        "privileged_result_is_public_method_evidence": False,
        "within_group_server_session_fixed": True,
        "cross_group_server_session_restart_allowed": True,
    }:
        raise ValueError("oracle formal execution claim firewall was weakened")
    interpretation = formal_protocol.get("causal_interpretation")
    if interpretation != {
        "require_positive_paired_risk_difference": True,
        "reject_when_exact_two_sided_p_is_at_most_plan_alpha": True,
        "automatic_public_method_success_claim": False,
    }:
        raise ValueError("oracle formal causal interpretation was changed")
    pilot_path = _verified_reference(
        inputs.get("pilot"),
        name="oracle independent pilot",
        repository_root=repository_root,
    )
    if sha256(pilot_path) != plan["pilot"]["sha256"]:
        raise ValueError("oracle formal schedule uses another pilot")
    state_manifest_path = _verified_reference(
        inputs.get("initial_state_manifest"),
        name="oracle formal initial-state manifest",
        repository_root=repository_root,
    )
    state_manifest = load_oracle_formal_initial_states(
        state_manifest_path, repository_root=repository_root
    )
    split_path = _verified_reference(
        inputs.get("split_manifest"),
        name="oracle formal split manifest",
        repository_root=repository_root,
    )
    if state_manifest["split_manifest"]["sha256"] != sha256(split_path):
        raise ValueError("oracle formal schedule split/state manifests differ")
    scenario_path = _verified_reference(
        inputs.get("scenario_config"),
        name="oracle formal scenario",
        repository_root=repository_root,
    )
    scenario = yaml.safe_load(scenario_path.read_text())
    baseline_path = _verified_reference(
        inputs.get("baseline_registry"),
        name="oracle formal baseline registry",
        repository_root=repository_root,
    )
    baseline = yaml.safe_load(baseline_path.read_text())
    if (
        scenario.get("schema_version") != "piu.scenario.v1"
        or baseline.get("schema_version") != "piu.baseline-registry.v1"
        or split_path is None
        or load_split_manifest(split_path).get("scenario") != scenario.get("id")
        or config.get("scenario_config")
        != portable(scenario_path, repository_root=repository_root)
        or formal_protocol.get("scenario_config")
        != portable(scenario_path, repository_root=repository_root)
    ):
        raise ValueError("oracle formal schedule uses another scenario")
    if resolve(
        Path(str(baseline.get("scenario", ""))), repository_root=repository_root
    ).resolve() != resolve(
        Path(str(scenario.get("scene", {}).get("bddl", ""))),
        repository_root=repository_root,
    ).resolve():
        raise ValueError("oracle formal baseline registry uses another scenario")
    identity_path = _verified_reference(
        inputs.get("policy_identity"),
        name="oracle formal policy identity",
        repository_root=repository_root,
    )
    expected_identity = resolve(
        Path(config["resource_contract"]["checkpoint_identity"]),
        repository_root=repository_root,
    )
    if identity_path.resolve() != expected_identity.resolve():
        raise ValueError("oracle formal schedule uses another policy identity")
    lock_reference = inputs.get("offline_repro_lock")
    lock_path = _verified_reference(
        lock_reference,
        name="oracle formal offline release lock",
        repository_root=repository_root,
    )
    manifest_path = repository_root / "configs/experiments/piu_offline_repro_v3.yaml"
    if (
        lock_reference.get("manifest_sha256") != sha256(manifest_path)
        or state_manifest["offline_repro_lock"]["sha256"] != sha256(lock_path)
    ):
        raise ValueError("oracle formal schedule release locks differ")
    validate_repro_lock(
        lock_path,
        manifest_path=manifest_path,
        repository_root=repository_root,
    )
    certificate_path = _verified_reference(
        inputs.get("open_qualification_certificate"),
        name="oracle formal OPEN certificate",
        repository_root=repository_root,
    )
    certificate = load_primitive_qualification_certificate(
        certificate_path, repository_root=repository_root
    )
    formal = formal_protocol
    if (
        certificate.get("status") != "FORMALLY_QUALIFIED"
        or certificate.get("paper_method_action_authorized") is not True
        or certificate.get("primitive") != formal.get("source_primitive")
        or certificate.get("candidate_id") != formal.get("source_candidate_id")
    ):
        raise ValueError("oracle formal schedule lacks the exact qualified OPEN")
    candidate_contract = certificate.get("candidate_contract")
    if not isinstance(candidate_contract, Mapping):
        raise TypeError("oracle formal OPEN certificate lacks a candidate contract")
    candidate = candidate_contract.get("selected_candidate")
    if not isinstance(candidate, Mapping):
        raise TypeError("oracle formal OPEN candidate is malformed")
    subtask = serialize_pi05_subtask(candidate, spatial_references=())
    if (
        not subtask
        or candidate_contract.get("spatial_reference_mode") != "none"
        or value.get("source_open_subtask") != subtask
        or value.get("source_open_candidate") != candidate
    ):
        raise ValueError("oracle formal source OPEN differs from its certificate")
    selected_style = pilot["confirmation"]["selected_style"]
    if (
        value.get("selected_style") != selected_style
        or value.get("primary_outcome") != formal.get("outcome")
        or value.get("arms") != list(ARMS)
        or value.get("failure_accounting")
        != "attempted_open_or_arm_failure_retained_as_false"
        or value.get("causal_interpretation") != interpretation
    ):
        raise ValueError("oracle formal estimand differs from its protocol")
    schedule_contract = formal.get("execution_schedule", {})
    randomization = value.get("randomization", {})
    if (
        schedule_contract.get("method")
        != "sha256_keyed_outcome_independent_permutation"
        or schedule_contract.get("outcomes_loaded") is not False
        or schedule_contract.get("group_order_randomized") is not True
        or schedule_contract.get("within_group_arm_order_randomized") is not True
        or randomization.get("method") != schedule_contract.get("method")
        or randomization.get("namespace") != schedule_contract.get("namespace")
    ):
        raise ValueError("oracle formal execution randomization was weakened")
    namespace = str(schedule_contract["namespace"])
    binding = "\0".join(
        sha256(item)
        for item in (
            plan_path,
            pilot_path,
            state_manifest_path,
            split_path,
            config_path,
            formal_protocol_path,
            scenario_path,
            baseline_path,
            identity_path,
            certificate_path,
            lock_path,
        )
    )
    if randomization.get("binding_sha256") != hashlib.sha256(binding.encode()).hexdigest():
        raise ValueError("oracle formal schedule binding differs")
    state_rows = {
        row["initial_state_group"]: row for row in state_manifest["states"]
    }
    prior_oracle_seeds = {int(seed) for seed in config["preflight"]["source_seeds"]}
    formal_groups = set(state_rows)
    formal_seeds = {
        int(row["simulator_seed"]) for row in state_manifest["states"]
    }
    qualification_schedule_path = _verified_reference(
        certificate["schedule"],
        name="oracle formal source qualification schedule",
        repository_root=repository_root,
    )
    qualification_schedule = json.loads(qualification_schedule_path.read_text())
    qualification_groups = {
        str(row["initial_state_group"]) for row in qualification_schedule["entries"]
    }
    qualification_seeds = {
        int(row["simulator_seed"]) for row in qualification_schedule["entries"]
    }
    if prior_oracle_seeds & formal_seeds:
        raise ValueError("oracle formal cohort reuses a preflight/pilot seed")
    if qualification_groups & formal_groups or qualification_seeds & formal_seeds:
        raise ValueError("oracle formal cohort reuses an OPEN qualification group")
    expected_groups = sorted(
        state_rows,
        key=lambda group: _key(namespace, binding, group),
    )
    entries = value.get("entries")
    if (
        not isinstance(entries, list)
        or len(entries) != int(plan["prospective_group_count"])
        or len(entries) != len(state_rows)
    ):
        raise ValueError("oracle formal schedule has another cohort size")
    run_root_raw = " ".join(str(value.get("run_root", "")).split())
    if not run_root_raw or run_root_raw != formal.get("run_root"):
        raise ValueError("oracle formal schedule uses another run root")
    run_root = resolve(Path(run_root_raw), repository_root=repository_root).resolve()
    for index, (entry, group) in enumerate(zip(entries, expected_groups, strict=True)):
        if not isinstance(entry, Mapping):
            raise TypeError("oracle formal schedule entries must be mappings")
        state = state_rows[group]
        expected_arms = sorted(
            ARMS,
            key=lambda arm: _key(namespace, f"{binding}\0{group}", arm),
        )
        group_slug = hashlib.sha256(group.encode()).hexdigest()[:12]
        group_root = run_root / f"{index:05d}_{group_slug}"
        if (
            entry.get("execution_index") != index
            or entry.get("initial_state_group") != group
            or entry.get("simulator_seed") != state["simulator_seed"]
            or entry.get("state_key") != state["state_key"]
            or entry.get("source_state") != state["source_state"]
            or entry.get("arm_order") != expected_arms
            or resolve(
                Path(str(entry.get("expected_group_receipt", ""))),
                repository_root=repository_root,
            ).resolve()
            != group_root / "receipt.json"
            or resolve(
                Path(str(entry.get("expected_started_ticket", ""))),
                repository_root=repository_root,
            ).resolve()
            != group_root / "started.json"
            or resolve(
                Path(str(entry.get("expected_post_open_state", ""))),
                repository_root=repository_root,
            ).resolve()
            != group_root / "open_source/final_state.npz"
            or resolve(
                Path(str(entry.get("expected_open_report", ""))),
                repository_root=repository_root,
            ).resolve()
            != group_root / "open_source/report.json"
        ):
            raise ValueError("oracle formal schedule entry differs from its binding")
        reports = entry.get("expected_arm_reports")
        if not isinstance(reports, Mapping) or set(reports) != set(ARMS):
            raise ValueError("oracle formal schedule arm reports are incomplete")
        for arm in ARMS:
            expected = group_root / arm / "report.json"
            observed = resolve(
                Path(str(reports[arm])), repository_root=repository_root
            ).resolve()
            if observed != expected:
                raise ValueError("oracle formal schedule arm report path differs")
    return value


def _keyframe_hashes(report: Mapping[str, Any], name: str) -> dict[str, str]:
    controller = report.get("controller", {})
    rows = [row for row in controller.get("keyframes", ()) if row.get("name") == name]
    if len(rows) != 1 or not isinstance(rows[0].get("image_sha256"), Mapping):
        raise ValueError(f"oracle formal report lacks one {name} keyframe")
    return dict(rows[0]["image_sha256"])


def oracle_formal_report_metrics(
    report: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    evaluator = report.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise TypeError("oracle formal report lacks evaluator metrics")
    target_name = str(config["execution"]["target_object"])
    wrong_name = str(config["execution"]["wrong_object"])
    objects = evaluator.get("objects")
    if not isinstance(objects, Mapping) or target_name not in objects or wrong_name not in objects:
        raise ValueError("oracle formal evaluator object set differs")
    target = objects[target_name]
    wrong = objects[wrong_name]
    return {
        "target_grasp_contact": bool(target["grasp_contact_steps"] > 0),
        "wrong_object_grasp_contact": bool(wrong["grasp_contact_steps"] > 0),
        "target_destination_final": bool(evaluator["target_in_destination_final"]),
        "task_success": bool(evaluator["task_success"]),
        "target_maximum_lift_m": float(target["maximum_lift_m"]),
    }


def validate_oracle_formal_execution_report(
    path: Path,
    *,
    entry: Mapping[str, Any],
    arm: str,
    schedule: Mapping[str, Any],
    config: Mapping[str, Any],
    scenario: Mapping[str, Any],
    identity: Mapping[str, Any],
    post_open_state_path: Path | None,
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(path.read_text())
    controller = report.get("controller")
    if not isinstance(controller, Mapping):
        raise TypeError("oracle formal execution report lacks controller metadata")
    expected_prompt = (
        schedule["source_open_subtask"]
        if arm == "source_open"
        else (
            config["execution"]["prompt"]
            if arm == "oracle_target_prompt"
            else scenario["task"]["prompt"]
        )
    )
    expected_role = {
        "source_open": "PIU_ORACLE_FORMAL_OPEN_SOURCE",
        "oracle_target_prompt": "PIU_ORACLE_FORMAL_ORACLE",
        "raw_post_open_direct": "PIU_ORACLE_FORMAL_BASELINE",
    }[arm]
    expected_claim = (
        "EVALUATOR_ONLY_ORACLE_UPPER_BOUND"
        if arm == "oracle_target_prompt"
        else "PUBLIC_INPUT_EXECUTION"
    )
    if (
        report.get("schema_version") != "piu.semantic-option.v2"
        or report.get("claim_scope") != expected_claim
        or report.get("seed") != entry["simulator_seed"]
        or report.get("role") != expected_role
        or report.get("prompt") != expected_prompt
        or controller.get("server_mode") != "external"
    ):
        raise ValueError(f"oracle formal {arm} report identity differs")
    validate_server_metadata(dict(controller.get("server_metadata", {})), identity)
    oracle_inputs = controller.get("online_oracle_inputs")
    if arm == "oracle_target_prompt":
        oracle = controller.get("oracle_visual_prompt")
        if (
            not isinstance(oracle_inputs, Sequence)
            or isinstance(oracle_inputs, (str, bytes))
            or len(oracle_inputs) != 2
            or not isinstance(oracle, Mapping)
            or oracle.get("style") != schedule["selected_style"]
        ):
            raise ValueError("oracle formal treatment lacks its declared oracle input")
    elif oracle_inputs != [] or controller.get("oracle_visual_prompt") is not None:
        raise ValueError("oracle formal public execution consumed oracle inputs")
    source = controller.get("source_initial_state_transport")
    expected_source = (
        resolve(Path(entry["source_state"]["path"]), repository_root=repository_root)
        if arm == "source_open"
        else post_open_state_path
    )
    if (
        expected_source is None
        or not isinstance(source, Mapping)
        or resolve(
            Path(str(source.get("path", ""))), repository_root=repository_root
        ).resolve()
        != expected_source.resolve()
        or source.get("sha256") != sha256(expected_source)
        or source.get("state_key") != entry["state_key"]
    ):
        raise ValueError(f"oracle formal {arm} source-state provenance differs")
    if arm == "oracle_target_prompt":
        oracle = controller["oracle_visual_prompt"]
        oracle_source = oracle.get("source_initial_state")
        audits = oracle.get("policy_call_audit")
        minimum_visible = int(config["execution"]["target_presence_minimum_pixels"])
        if (
            not isinstance(oracle_source, Mapping)
            or resolve(
                Path(str(oracle_source.get("path", ""))),
                repository_root=repository_root,
            ).resolve()
            != expected_source.resolve()
            or oracle_source.get("sha256") != sha256(expected_source)
            or not isinstance(audits, Sequence)
            or isinstance(audits, (str, bytes))
            or not audits
            or len(audits) != controller.get("policy_calls")
            or max(int(item) for item in audits[0]["visible_pixels"].values())
            < minimum_visible
        ):
            raise ValueError("oracle formal target-prompt audit is incomplete")
    if arm == "source_open":
        final_path = resolve(
            Path(entry["expected_post_open_state"]), repository_root=repository_root
        )
        if (
            not final_path.is_file()
            or resolve(
                Path(str(controller.get("opaque_state_transport", ""))),
                repository_root=repository_root,
            ).resolve()
            != final_path.resolve()
            or controller.get("opaque_state_transport_sha256") != sha256(final_path)
        ):
            raise ValueError("oracle formal source OPEN final-state transport differs")
    action_path = resolve(
        Path(str(controller.get("action_history", ""))),
        repository_root=repository_root,
    )
    if not action_path.is_file():
        raise FileNotFoundError(action_path)
    keyframe_collections = [controller.get("keyframes", ())]
    if arm == "oracle_target_prompt":
        keyframe_collections.append(
            controller["oracle_visual_prompt"].get("keyframes", ())
        )
    for row in (item for rows in keyframe_collections for item in rows):
        if not isinstance(row, Mapping):
            raise TypeError("oracle formal keyframe must be a mapping")
        paths = row.get("image_paths", {})
        digests = row.get("image_sha256", {})
        if not isinstance(paths, Mapping) or not isinstance(digests, Mapping):
            raise TypeError("oracle formal keyframe lacks image provenance")
        for view, raw_path in paths.items():
            image_path = resolve(Path(str(raw_path)), repository_root=repository_root)
            if not image_path.is_file() or digests.get(view) != sha256(image_path):
                raise ValueError("oracle formal keyframe differs from its hash")
    metrics = oracle_formal_report_metrics(report, config)
    if any(not math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("oracle formal report contains nonfinite metrics")
    return report, metrics


def load_oracle_formal_group_receipt(
    path: Path,
    *,
    schedule_path: Path,
    schedule: Mapping[str, Any],
    execution_index: int,
    repository_root: Path,
) -> dict[str, Any]:
    """Validate one irreversible group receipt and recompute its arm outcomes."""

    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != RECEIPT_SCHEMA
        or value.get("status") != "CLOSED_SINGLE_USE"
        or value.get("execution_index") != execution_index
        or value.get("schedule_sha256") != sha256(schedule_path)
        or value.get("outcomes_entered_manually") is not False
    ):
        raise ValueError("unsupported oracle formal group receipt")
    entries = schedule["entries"]
    if execution_index >= len(entries):
        raise ValueError("oracle formal receipt index exceeds its schedule")
    entry = entries[execution_index]
    if value.get("initial_state_group") != entry["initial_state_group"]:
        raise ValueError("oracle formal receipt uses another group")
    started_path = resolve(
        Path(entry["expected_started_ticket"]), repository_root=repository_root
    )
    if not started_path.is_file() or value.get("started_ticket_sha256") != sha256(
        started_path
    ):
        raise ValueError("oracle formal receipt lacks its immutable start ticket")
    started = json.loads(started_path.read_text())
    endpoint_path = _verified_reference(
        value.get("endpoint_check"),
        name="oracle formal endpoint check",
        repository_root=repository_root,
    )
    previous_digest = None
    if execution_index > 0:
        previous_path = resolve(
            Path(entries[execution_index - 1]["expected_group_receipt"]),
            repository_root=repository_root,
        )
        if not previous_path.is_file():
            raise ValueError("oracle formal group closed before its predecessor")
        previous_digest = sha256(previous_path)
    if (
        started.get("schema_version") != "piu.oracle-formal-group-start.v1"
        or started.get("status") != "STARTED_SINGLE_USE"
        or started.get("execution_index") != execution_index
        or started.get("schedule_sha256") != sha256(schedule_path)
        or started.get("entry") != entry
        or started.get("previous_receipt_sha256") != previous_digest
        or started.get("outcomes_loaded") is not False
        or started.get("endpoint_check_sha256") != sha256(endpoint_path)
    ):
        raise ValueError("oracle formal start ticket differs from the schedule")
    config_path = _verified_reference(
        schedule["inputs"]["experiment"],
        name="oracle experiment",
        repository_root=repository_root,
    )
    config = yaml.safe_load(config_path.read_text())
    scenario_path = _verified_reference(
        schedule["inputs"]["scenario_config"],
        name="oracle scenario",
        repository_root=repository_root,
    )
    scenario = yaml.safe_load(scenario_path.read_text())
    identity_path = _verified_reference(
        schedule["inputs"]["policy_identity"],
        name="oracle policy identity",
        repository_root=repository_root,
    )
    identity = load_checkpoint_identity(identity_path)
    endpoint = json.loads(endpoint_path.read_text())
    session = value.get("server_session_id")
    endpoint_probe = endpoint.get("action_probe")
    if (
        endpoint.get("schema_version")
        not in {"piu.external-pi05-check.v1", "piu.external-pi05-check.v2"}
        or endpoint.get("status") != "PASS"
        or not isinstance(endpoint_probe, Mapping)
        or endpoint_probe.get("finite") is not True
        or not isinstance(session, str)
        or not session
        or endpoint.get("identity", {}).get("server_session_id") != session
    ):
        raise ValueError("oracle formal receipt endpoint/session is invalid")
    validate_server_metadata(dict(endpoint.get("identity", {})), identity)
    source_status = value.get("source_open_status")
    arm_status = value.get("arm_status")
    if source_status not in {"COMPLETE", "FAILED", "INTERRUPTED_UNVERIFIED"} or not isinstance(
        arm_status, Mapping
    ) or set(arm_status) != set(ARMS):
        raise ValueError("oracle formal receipt has invalid execution statuses")
    for arm in ARMS:
        if arm_status[arm] not in {
            "COMPLETE",
            "FAILED",
            "NOT_RUN_OPEN_FAILED",
            "INTERRUPTED_UNVERIFIED",
        }:
            raise ValueError("oracle formal receipt has an unsupported arm status")
    metrics: dict[str, dict[str, Any]] = {
        arm: {
            "target_grasp_contact": False,
            "wrong_object_grasp_contact": False,
            "target_destination_final": False,
            "task_success": False,
            "target_maximum_lift_m": 0.0,
        }
        for arm in ARMS
    }
    reports = value.get("reports")
    if not isinstance(reports, Mapping) or set(reports) != {"source_open", *ARMS}:
        raise ValueError("oracle formal receipt report map is incomplete")
    post_open_path = resolve(
        Path(entry["expected_post_open_state"]), repository_root=repository_root
    )
    if source_status == "COMPLETE":
        open_path = _verified_reference(
            reports["source_open"],
            name="oracle formal OPEN report",
            repository_root=repository_root,
        )
        if open_path.resolve() != resolve(
            Path(entry["expected_open_report"]), repository_root=repository_root
        ).resolve():
            raise ValueError("oracle formal OPEN report path differs")
        if not post_open_path.is_file() or value.get("post_open_state_sha256") != sha256(
            post_open_path
        ):
            raise ValueError("oracle formal post-OPEN state differs")
        open_report, _ = validate_oracle_formal_execution_report(
            open_path,
            entry=entry,
            arm="source_open",
            schedule=schedule,
            config=config,
            scenario=scenario,
            identity=identity,
            post_open_state_path=None,
            repository_root=repository_root,
        )
        if open_report["controller"]["server_metadata"].get(
            "server_session_id"
        ) != session:
            raise ValueError("oracle formal OPEN report used another server session")
    else:
        if reports["source_open"] is not None:
            failed_open_path = _verified_reference(
                reports["source_open"],
                name="failed oracle formal OPEN report",
                repository_root=repository_root,
            )
            if failed_open_path.resolve() != resolve(
                Path(entry["expected_open_report"]), repository_root=repository_root
            ).resolve():
                raise ValueError("failed oracle formal OPEN report path differs")
        if value.get("post_open_state_sha256") is not None:
            raise ValueError("failed oracle formal OPEN authorizes a post-OPEN state")
        expected_arm_status = (
            "NOT_RUN_OPEN_FAILED"
            if source_status == "FAILED"
            else "INTERRUPTED_UNVERIFIED"
        )
        if any(arm_status[arm] != expected_arm_status for arm in ARMS):
            raise ValueError("oracle formal failed/interrupted arm status differs")
    initial_hashes: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        if arm_status[arm] == "COMPLETE":
            report_path = _verified_reference(
                reports[arm],
                name=f"oracle formal {arm} report",
                repository_root=repository_root,
            )
            expected_path = resolve(
                Path(entry["expected_arm_reports"][arm]),
                repository_root=repository_root,
            )
            if report_path.resolve() != expected_path.resolve():
                raise ValueError("oracle formal arm report path differs")
            report, metrics[arm] = validate_oracle_formal_execution_report(
                report_path,
                entry=entry,
                arm=arm,
                schedule=schedule,
                config=config,
                scenario=scenario,
                identity=identity,
                post_open_state_path=post_open_path,
                repository_root=repository_root,
            )
            if report["controller"]["server_metadata"].get(
                "server_session_id"
            ) != session:
                raise ValueError("oracle formal arm report used another server session")
            initial_hashes[arm] = _keyframe_hashes(report, "00_before")
        elif reports[arm] is not None:
            failed_path = _verified_reference(
                reports[arm],
                name=f"failed oracle formal {arm} report",
                repository_root=repository_root,
            )
            expected_path = resolve(
                Path(entry["expected_arm_reports"][arm]),
                repository_root=repository_root,
            )
            if failed_path.resolve() != expected_path.resolve():
                raise ValueError("failed oracle formal arm report path differs")
    if set(initial_hashes) == set(ARMS) and len(
        {tuple(sorted(value.items())) for value in initial_hashes.values()}
    ) != 1:
        raise ValueError("oracle formal arms did not start from identical public RGB")
    if value.get("derived_outcomes") != metrics:
        raise ValueError("oracle formal receipt outcomes differ from recomputation")
    return value


def analyze_oracle_formal_schedule(
    schedule_path: Path, *, repository_root: Path
) -> dict[str, Any]:
    """Analyze the complete intention-to-treat paired causal oracle cohort."""

    schedule = load_oracle_formal_schedule(
        schedule_path, repository_root=repository_root
    )
    plan_path = resolve(
        Path(schedule["inputs"]["formal_plan"]["path"]),
        repository_root=repository_root,
    )
    plan, config, _ = load_oracle_formal_plan(
        plan_path, repository_root=repository_root
    )
    rows = []
    sessions: set[str] = set()
    for index, entry in enumerate(schedule["entries"]):
        receipt_path = resolve(
            Path(entry["expected_group_receipt"]), repository_root=repository_root
        )
        if not receipt_path.is_file():
            raise ValueError("oracle formal analysis requires the complete denominator")
        receipt = load_oracle_formal_group_receipt(
            receipt_path,
            schedule_path=schedule_path,
            schedule=schedule,
            execution_index=index,
            repository_root=repository_root,
        )
        session = receipt.get("server_session_id")
        if not isinstance(session, str) or not session:
            raise ValueError("oracle formal receipt lacks a server session")
        sessions.add(session)
        rows.append(
            {
                "execution_index": index,
                "initial_state_group": entry["initial_state_group"],
                "simulator_seed": entry["simulator_seed"],
                "source_open_status": receipt["source_open_status"],
                "arm_status": receipt["arm_status"],
                "derived_outcomes": receipt["derived_outcomes"],
                "receipt": artifact(receipt_path, repository_root=repository_root),
            }
        )
    primary = str(schedule["primary_outcome"])
    treatment = [
        bool(row["derived_outcomes"]["oracle_target_prompt"][primary]) for row in rows
    ]
    comparator = [
        bool(row["derived_outcomes"]["raw_post_open_direct"][primary]) for row in rows
    ]
    confidence = 1.0 - float(plan["alpha"])
    summary = summarize_paired_pilot(
        treatment, comparator, confidence=confidence
    )
    supported = bool(
        summary["paired_risk_difference"] > 0.0
        and summary["exact_two_sided_paired_binomial_p"] <= float(plan["alpha"])
    )
    secondary = {}
    for outcome in (
        "wrong_object_grasp_contact",
        "target_destination_final",
        "task_success",
    ):
        secondary[outcome] = paired_binary_summary(
            [
                bool(row["derived_outcomes"]["oracle_target_prompt"][outcome])
                for row in rows
            ],
            [
                bool(row["derived_outcomes"]["raw_post_open_direct"][outcome])
                for row in rows
            ],
        )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": (
            "FORMAL_ORACLE_CAUSAL_EFFECT_SUPPORTED"
            if supported
            else "FORMAL_ORACLE_CAUSAL_EFFECT_NOT_SUPPORTED"
        ),
        "claim_scope": "PRIVILEGED_CAUSAL_MECHANISM_ONLY_NOT_PUBLIC_METHOD",
        "formal_method_claim": False,
        "automatic_public_method_success_claim": False,
        "schedule": artifact(schedule_path, repository_root=repository_root),
        "plan": artifact(plan_path, repository_root=repository_root),
        "complete_frozen_denominator": True,
        "groups": len(rows),
        "server_session_ids": sorted(sessions),
        "server_session_count": len(sessions),
        "primary": {
            "outcome": primary,
            "treatment": "oracle_target_prompt",
            "comparator": "raw_post_open_direct",
            "test": plan["test"],
            "alpha": float(plan["alpha"]),
            **summary,
            "causal_effect_supported": supported,
        },
        "secondary_descriptive": secondary,
        "rows": rows,
        "interpretation": (
            "A positive result establishes only that privileged target binding can "
            "change frozen-executor behavior in this fixed scenario."
        ),
    }
    return result
