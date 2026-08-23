"""Auditable primitive reliability estimates and prospective binomial design."""

from __future__ import annotations

import hashlib
import json
import math
import ast
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .formal_states import validate_state_archive
from .contracts import assert_public_policy_value
from .executor_bridge import SpatialReference, serialize_pi05_subtask
from .reproducibility import validate_repro_lock
from .policy_identity import load_checkpoint_identity, validate_server_metadata


def allocate_episode_primitive_risk(
    *,
    maximum_episode_failure_probability: float,
    maximum_physical_dispatches: int,
) -> dict[str, Any]:
    """Allocate an external episode risk budget without dependence assumptions."""

    delta = float(maximum_episode_failure_probability)
    if not 0.0 < delta < 1.0:
        raise ValueError("episode primitive-failure budget must lie strictly in (0,1)")
    if (
        not isinstance(maximum_physical_dispatches, int)
        or isinstance(maximum_physical_dispatches, bool)
        or maximum_physical_dispatches < 1
    ):
        raise ValueError("maximum physical dispatches must be a positive integer")
    per_dispatch_failure = delta / maximum_physical_dispatches
    return {
        "method": "bonferroni_union_bound_equal_per_dispatch",
        "maximum_episode_failure_probability": delta,
        "maximum_physical_dispatches": maximum_physical_dispatches,
        "per_dispatch_failure_probability": per_dispatch_failure,
        "minimum_reliable_rate": 1.0 - per_dispatch_failure,
        "dependence_assumption": "none",
        "union_bound_maximum_episode_failure_probability": (
            maximum_physical_dispatches * per_dispatch_failure
        ),
    }


def validate_derived_primitive_risk_contract(
    value: dict[str, Any],
) -> dict[str, Any]:
    """Reject hand-entered rates masquerading as a derived risk contract."""

    if value.get("schema_version") != "piu.primitive-risk-contract.v1":
        raise ValueError("unsupported primitive risk contract")
    if value.get("status") != "FROZEN_BEFORE_PRIMITIVE_QUALIFICATION_OUTCOMES":
        raise ValueError("primitive risk contract was not prospectively frozen")
    if value.get("outcomes_loaded") is not False:
        raise ValueError("primitive risk derivation may not load qualification outcomes")
    if value.get("minimum_reliable_rate_provenance") != (
        "derived_union_bound_from_external_episode_budget"
    ):
        raise ValueError("primitive minimum reliability is not externally derived")
    allocation = value.get("risk_allocation")
    if not isinstance(allocation, dict):
        raise TypeError("primitive risk contract lacks a risk allocation")
    expected = allocate_episode_primitive_risk(
        maximum_episode_failure_probability=float(
            allocation["maximum_episode_failure_probability"]
        ),
        maximum_physical_dispatches=allocation["maximum_physical_dispatches"],
    )
    for name, required in expected.items():
        observed = allocation.get(name)
        if isinstance(required, float):
            if not isinstance(observed, (int, float)) or isinstance(observed, bool):
                raise TypeError(f"risk-allocation field {name} must be numeric")
            if not math.isclose(float(observed), required, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"risk-allocation field {name} was hand-modified")
        elif observed != required:
            raise ValueError(f"risk-allocation field {name} differs from derivation")
    if not math.isclose(
        float(value.get("minimum_reliable_rate")),
        float(expected["minimum_reliable_rate"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("primitive minimum reliable rate differs from derivation")
    for name in ("primitive", "context", "candidate_id"):
        if not " ".join(str(value.get(name, "")).split()):
            raise ValueError(f"primitive risk contract requires {name}")
    alpha = float(value.get("alpha"))
    target_power = float(value.get("target_power"))
    alternative = float(value.get("design_alternative_success_probability"))
    minimum = float(value.get("minimum_reliable_rate"))
    if not 0.0 < alpha < 1.0 or not 0.0 < target_power < 1.0:
        raise ValueError("primitive qualification alpha/power are invalid")
    if not minimum < alternative <= 1.0:
        raise ValueError(
            "primitive design alternative must exceed the derived reliability rate"
        )
    if value.get("design_alternative_provenance") != "external_task_owner_contract":
        raise ValueError("primitive design alternative is not externally declared")
    if value.get("retrospective_pilot_used_for_effect_size") is not False:
        raise ValueError("primitive risk contract reused a retrospective pilot")
    maximum_groups = value.get("maximum_qualification_groups")
    if (
        not isinstance(maximum_groups, int)
        or isinstance(maximum_groups, bool)
        or maximum_groups < 1
    ):
        raise ValueError("primitive risk contract lacks an external group cap")
    if value.get("maximum_qualification_groups_provenance") != (
        "external_task_owner_resource_contract"
    ):
        raise ValueError("primitive qualification group cap is not external")
    return value


def wilson_interval(
    successes: int, trials: int, *, z: float = 1.959963984540054
) -> list[float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    rate = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (rate + z**2 / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def reliability_record(values: Sequence[bool]) -> dict[str, Any]:
    if not values:
        raise ValueError("reliability evidence must be non-empty")
    successes = sum(bool(value) for value in values)
    trials = len(values)
    return {
        "successes": successes,
        "trials": trials,
        "rate": successes / trials,
        "wilson_95": wilson_interval(successes, trials),
    }


def binomial_upper_tail(successes: int, trials: int, probability: float) -> float:
    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial tail counts")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("binomial probability must lie in [0,1]")
    return float(
        sum(
            math.comb(trials, count)
            * probability**count
            * (1.0 - probability) ** (trials - count)
            for count in range(successes, trials + 1)
        )
    )


def exact_binomial_rejection_count(
    trials: int, *, null_success_probability: float, alpha: float
) -> int | None:
    """Smallest success count rejecting an absolute reliability null."""

    if trials < 1 or not 0.0 < alpha < 1.0:
        raise ValueError("trials and alpha must be positive")
    for successes in range(trials + 1):
        if (
            binomial_upper_tail(
                successes, trials, null_success_probability
            )
            <= alpha
        ):
            return successes
    return None


def exact_binomial_power(
    trials: int,
    *,
    null_success_probability: float,
    alternative_success_probability: float,
    alpha: float,
) -> dict[str, float | int | None]:
    if not 0.0 <= null_success_probability < alternative_success_probability <= 1.0:
        raise ValueError("alternative reliability must exceed the null contract")
    rejection_count = exact_binomial_rejection_count(
        trials,
        null_success_probability=null_success_probability,
        alpha=alpha,
    )
    power = (
        0.0
        if rejection_count is None
        else binomial_upper_tail(
            rejection_count, trials, alternative_success_probability
        )
    )
    return {
        "trials": trials,
        "rejection_success_count": rejection_count,
        "power": power,
    }


def smallest_binomial_design(
    *,
    null_success_probability: float,
    alternative_success_probability: float,
    alpha: float,
    target_power: float,
    search_limit: int,
) -> dict[str, float | int | None] | None:
    """Freeze the first exact one-sided binomial design reaching target power."""

    if not 0.0 < target_power < 1.0 or search_limit < 1:
        raise ValueError("invalid power target or numerical search limit")
    for trials in range(1, search_limit + 1):
        design = exact_binomial_power(
            trials,
            null_success_probability=null_success_probability,
            alternative_success_probability=alternative_success_probability,
            alpha=alpha,
        )
        if float(design["power"]) >= target_power:
            return design
    return None


def evaluate_frozen_binomial_design(
    values: Sequence[bool],
    *,
    null_success_probability: float,
    alpha: float,
    expected_trials: int,
    rejection_success_count: int,
) -> dict[str, Any]:
    """Evaluate a prospectively frozen exact-binomial rejection region."""

    if len(values) != expected_trials:
        raise ValueError("formal primitive outcomes do not match the frozen trial count")
    if not 0 <= rejection_success_count <= expected_trials:
        raise ValueError("invalid frozen primitive rejection count")
    if any(not isinstance(value, bool) for value in values):
        raise TypeError("formal primitive outcomes must be booleans")
    successes = sum(values)
    p_value = binomial_upper_tail(
        successes, expected_trials, null_success_probability
    )
    qualified = successes >= rejection_success_count and p_value <= alpha
    return {
        "successes": successes,
        "trials": expected_trials,
        "rate": successes / expected_trials,
        "exact_one_sided_p_value": p_value,
        "rejection_success_count": rejection_success_count,
        "qualified": qualified,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_artifact(value: str, *, repository_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def _verified_reference(
    value: Any, *, name: str, repository_root: Path
) -> Path:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} artifact reference must be a mapping")
    path = _resolve_artifact(str(value.get("path", "")), repository_root=repository_root)
    if not path.is_file() or _sha256(path) != value.get("sha256"):
        raise ValueError(f"{name} artifact differs from its hash")
    return path


def load_derived_primitive_risk_contract(
    path: Path, *, repository_root: Path
) -> dict[str, Any]:
    """Load and re-derive every input-bound executor reliability value."""

    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("primitive risk contract root must be a mapping")
    validate_derived_primitive_risk_contract(value)
    inputs = value.get("inputs")
    if not isinstance(inputs, Mapping):
        raise TypeError("primitive risk contract lacks input provenance")
    paths = {
        name: _verified_reference(
            inputs.get(name), name=f"primitive risk {name}", repository_root=repository_root
        )
        for name in (
            "allocation_config",
            "baseline_registry",
            "primitive_registry_protocol",
            "external_budget",
        )
    }
    allocation_config = yaml.safe_load(paths["allocation_config"].read_text())
    baseline = yaml.safe_load(paths["baseline_registry"].read_text())
    protocol = yaml.safe_load(paths["primitive_registry_protocol"].read_text())
    budget = yaml.safe_load(paths["external_budget"].read_text())
    if allocation_config.get("schema_version") != "piu.executor-risk-allocation.v1":
        raise ValueError("primitive risk allocation config is unsupported")
    if allocation_config.get("status") != (
        "frozen_derivation_waiting_for_external_episode_risk_budget"
    ):
        raise ValueError("primitive risk allocation config is not frozen")
    allocation_rule = allocation_config.get("allocation", {})
    if (
        allocation_rule.get("method")
        != "bonferroni_union_bound_equal_per_dispatch"
        or allocation_rule.get("dependence_assumption") != "none"
        or allocation_rule.get("outcome_dependent_choice") is not False
    ):
        raise ValueError("primitive risk allocation rule differs")
    claims = allocation_config.get("claim_contract", {})
    if (
        claims.get("external_budget_required") is not True
        or claims.get("budget_may_be_inferred_from_retrospective_successes")
        is not False
        or claims.get("budget_may_reuse_historical_count_gate") is not False
        or claims.get("calibration_alpha_is_executor_failure_budget") is not False
        or claims.get("derived_rate_is_task_success_probability") is not False
        or claims.get("qualification_outcomes_loaded_during_derivation") is not False
        or claims.get("retrospective_pilot_may_set_design_alternative") is not False
        or claims.get("qualification_group_resource_cap_required") is not True
        or claims.get("qualification_group_resource_cap_may_use_cli_default")
        is not False
        or claims.get("qualification_group_resource_cap_is_success_threshold")
        is not False
    ):
        raise ValueError("primitive risk claim firewall was weakened")
    if baseline.get("schema_version") != "piu.baseline-registry.v1":
        raise ValueError("primitive risk baseline registry is unsupported")
    if protocol.get("schema_version") != "piu.primitive-registry-protocol.v1":
        raise ValueError("primitive risk registry protocol is unsupported")
    if str(value["primitive"]) not in protocol.get(
        "prospective_success_contracts", {}
    ):
        raise ValueError("primitive risk contract lacks a registered success event")
    if budget.get("schema_version") != "piu.external-execution-risk-budget.v1":
        raise ValueError("primitive external risk budget is unsupported")
    if budget.get("status") != "FROZEN_BEFORE_PRIMITIVE_QUALIFICATION_OUTCOMES":
        raise ValueError("primitive external risk budget is not frozen")
    if budget.get("outcomes_loaded") is not False:
        raise ValueError("primitive external risk budget loaded outcomes")
    alternative = float(
        budget["design_alternative_per_dispatch_success_probability"]
    )
    maximum_groups = budget.get("maximum_qualification_groups_per_primitive")
    if (
        not isinstance(maximum_groups, int)
        or isinstance(maximum_groups, bool)
        or maximum_groups < 1
    ):
        raise ValueError("primitive external risk budget has no valid group cap")
    authority = " ".join(str(budget.get("authority", "")).split())
    rationale = " ".join(str(budget.get("rationale", "")).split())
    if (
        not authority
        or not rationale
        or authority != " ".join(str(value.get("external_authority", "")).split())
        or rationale != " ".join(str(value.get("external_rationale", "")).split())
    ):
        raise ValueError("primitive risk authority/rationale differs")
    if (
        value.get("claim_scope") != "EXECUTOR_RELIABILITY_ONLY_NOT_TASK_SUCCESS"
        or value.get("paper_method_claim_allowed") is not False
    ):
        raise ValueError("primitive risk contract overclaims its scope")
    if baseline["shared_contract"]["maximum_controller_decisions"] != value[
        "risk_allocation"
    ]["maximum_physical_dispatches"]:
        raise ValueError("primitive risk contract differs from dispatch cap")
    if float(budget["maximum_episode_probability_of_any_primitive_failure"]) != float(
        value["risk_allocation"]["maximum_episode_failure_probability"]
    ):
        raise ValueError("primitive risk contract differs from external budget")
    if alternative != float(value["design_alternative_success_probability"]):
        raise ValueError("primitive design alternative differs from external budget")
    if maximum_groups != value.get("maximum_qualification_groups"):
        raise ValueError("primitive qualification group cap differs from budget")
    if value.get("maximum_qualification_groups_provenance") != (
        "external_task_owner_resource_contract"
    ):
        raise ValueError("primitive qualification group cap lacks provenance")
    if protocol["formal_qualification"].get("minimum_reliable_rate") is not None:
        raise ValueError("primitive registry protocol contains a hand-entered rate")
    if float(protocol["formal_qualification"]["alpha"]) != float(value["alpha"]):
        raise ValueError("primitive risk alpha differs from registry protocol")
    if float(protocol["formal_qualification"]["target_power"]) != float(
        value["target_power"]
    ):
        raise ValueError("primitive risk power differs from registry protocol")
    return value


def load_primitive_qualification_plan(
    path: Path, *, repository_root: Path
) -> dict[str, Any]:
    """Recompute an immutable prospective qualification design from its inputs."""

    value = json.loads(path.read_text())
    if value.get("schema_version") != "piu.primitive-qualification-plan.v1":
        raise ValueError("unsupported primitive qualification plan")
    if value.get("status") != "PROSPECTIVE_GROUP_COUNT_FROZEN":
        raise ValueError("primitive qualification plan has no frozen design")
    if value.get("claim_scope") != "DESIGN_ONLY_NO_FORMAL_OUTCOME_DATA":
        raise ValueError("primitive qualification plan crossed its claim firewall")
    registry_path = _verified_reference(
        value.get("registry"), name="primitive registry", repository_root=repository_root
    )
    risk_path = _verified_reference(
        value.get("risk_contract"),
        name="primitive risk contract",
        repository_root=repository_root,
    )
    registry = json.loads(registry_path.read_text())
    if registry.get("schema_version") != "piu.primitive-reliability-registry.v1":
        raise ValueError("primitive plan registry is unsupported")
    risk = load_derived_primitive_risk_contract(
        risk_path, repository_root=repository_root
    )
    protocol_reference = risk["inputs"]["primitive_registry_protocol"]
    registry_config_path = _verified_reference(
        registry.get("config"),
        name="retrospective primitive registry protocol",
        repository_root=repository_root,
    )
    if (
        registry_config_path.resolve()
        != _resolve_artifact(
            str(protocol_reference["path"]), repository_root=repository_root
        ).resolve()
        or registry["config"]["sha256"] != protocol_reference["sha256"]
        or registry.get("paper_method_claim_allowed") is not False
        or registry.get("historical_count_gates_used") is not False
        or registry.get("paper_method_action_set_authorized") != []
    ):
        raise ValueError("primitive plan uses an unbound retrospective registry")
    registry_seeds = registry.get("seeds")
    if (
        not isinstance(registry_seeds, list)
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in registry_seeds)
        or len(set(registry_seeds)) != len(registry_seeds)
    ):
        raise ValueError("retrospective primitive registry seeds are invalid")
    for name in ("primitive", "context", "candidate_id"):
        if str(value.get(name)) != str(risk.get(name)):
            raise ValueError(f"primitive plan differs from risk contract at {name}")
    risk_summary = value["risk_contract"]
    if (
        float(risk_summary["minimum_reliable_rate"])
        != float(risk["minimum_reliable_rate"])
        or risk_summary.get("provenance")
        != risk["minimum_reliable_rate_provenance"]
        or risk_summary.get("external_budget") != risk["inputs"]["external_budget"]
        or risk_summary.get("risk_allocation") != risk["risk_allocation"]
    ):
        raise ValueError("primitive plan risk summary differs from its contract")
    if (
        value.get("retrospective_registry_role")
        != "diagnostic_and_seed_exclusion_only"
        or value.get("retrospective_pilot_used_for_effect_size") is not False
        or value.get("test") != "exact_one_sided_binomial"
    ):
        raise ValueError("primitive plan registry/test contract differs")
    alternative = float(risk["design_alternative_success_probability"])
    null_rate = float(risk["minimum_reliable_rate"])
    if alternative <= null_rate:
        raise ValueError("primitive frozen plan has a nonseparated alternative")
    if (
        float(value.get("alternative_success_probability")) != alternative
        or value.get("alternative_success_probability_provenance")
        != "external_task_owner_contract"
    ):
        raise ValueError("primitive plan alternative differs from its risk contract")
    if float(value.get("alpha")) != float(risk["alpha"]) or float(
        value.get("target_power")
    ) != float(risk["target_power"]):
        raise ValueError("primitive plan alpha/power differs from its risk contract")
    if (
        value.get("maximum_qualification_groups_provenance")
        != risk["maximum_qualification_groups_provenance"]
        or value.get("maximum_qualification_groups")
        != risk["maximum_qualification_groups"]
    ):
        raise ValueError("primitive plan group cap differs from its risk contract")
    expected = smallest_binomial_design(
        null_success_probability=null_rate,
        alternative_success_probability=alternative,
        alpha=float(risk["alpha"]),
        target_power=float(risk["target_power"]),
        search_limit=int(value["maximum_qualification_groups"]),
    )
    if expected is None or value.get("design") != expected:
        raise ValueError("primitive plan design differs from exact recomputation")
    return value


def primitive_qualification_permutation_key(
    *,
    namespace: str,
    plan_sha256: str,
    initial_state_group: str,
    simulator_seed: int,
    source_state_sha256: str,
) -> str:
    payload = "\0".join(
        (
            namespace,
            plan_sha256,
            initial_state_group,
            str(simulator_seed),
            source_state_sha256,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _spatial_reference_from_mapping(value: Any) -> SpatialReference:
    if not isinstance(value, Mapping):
        raise TypeError("qualification spatial reference must be a mapping")
    camera = " ".join(str(value.get("camera", "")).split())
    indices = value.get("selected_patch_indices")
    if (
        not camera
        or not isinstance(indices, Sequence)
        or isinstance(indices, (str, bytes))
        or not indices
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in indices
        )
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("qualification spatial-reference identity is invalid")
    intervals = []
    for name in ("x_interval", "y_interval"):
        raw = value.get(name)
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or len(raw) != 2
        ):
            raise ValueError(f"qualification {name} is invalid")
        interval = tuple(float(item) for item in raw)
        if (
            any(not math.isfinite(item) for item in interval)
            or not 0.0 <= interval[0] <= interval[1] <= 1.0
        ):
            raise ValueError(f"qualification {name} lies outside normalized RGB")
        intervals.append(interval)
    return SpatialReference(
        camera=camera,
        selected_patch_indices=tuple(indices),
        x_interval=intervals[0],
        y_interval=intervals[1],
    )


def load_qualification_controller_decision(
    path: Path,
    *,
    candidate_id: str,
    primitive: str,
    initial_state_group: str,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Bind qualification to the exact public candidate and serializer output."""

    report = json.loads(path.read_text())
    schema = report.get("schema_version")
    if schema not in {
        "piu.calibrated-controller-report.v1",
        "piu.uncalibrated-ablation-controller-report.v1",
        "piu.prompted-vlm-router-report.v1",
        "piu.primitive-qualification-probe.v1",
    }:
        raise ValueError("unsupported qualification controller report")
    if report.get("evaluator_labels_loaded") is not False:
        raise ValueError("qualification controller report contains evaluator labels")
    decisions = report.get("decisions")
    if not isinstance(decisions, Sequence) or isinstance(decisions, (str, bytes)):
        raise TypeError("qualification controller report lacks decisions")
    matches = [
        row
        for row in decisions
        if isinstance(row, Mapping)
        and " ".join(str(row.get("initial_state_group", "")).split())
        == initial_state_group
        and str(row.get("selected_candidate_id")) == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError("qualification group/candidate does not select one decision")
    decision = dict(matches[0])
    normalized_primitive = " ".join(primitive.split()).upper()
    if schema == "piu.primitive-qualification-probe.v1":
        if repository_root is None:
            raise ValueError("qualification probe validation requires repository root")
        if (
            len(decisions) != 1
            or decision.get("decision_kind") != "INTERACT"
            or not " ".join(str(decision.get("sample_id", "")).split())
            or report.get("status")
            != "FROZEN_BEFORE_PRIMITIVE_QUALIFICATION_OUTCOMES"
            or report.get("claim_scope")
            != "EXECUTOR_STIMULUS_ONLY_NOT_METHOD_SELECTION"
            or report.get("outcomes_loaded") is not False
            or report.get("selection_source")
            != "preregistered_executor_probe_not_method_decision"
            or report.get("candidate_choice_outcome_dependent") is not False
            or report.get("paper_method_selection_claim_allowed") is not False
            or report.get("trained_model_loaded") is not False
            or report.get("calibration_loaded") is not False
            or report.get("online_oracle_inputs") != []
            or normalized_primitive != "OPEN"
        ):
            raise ValueError("qualification probe crossed its selection firewall")
        inputs = report.get("inputs")
        if not isinstance(inputs, Mapping):
            raise TypeError("qualification probe lacks input provenance")
        plan_path = _verified_reference(
            inputs.get("plan"),
            name="qualification probe plan",
            repository_root=repository_root,
        )
        candidate_set_path = _verified_reference(
            inputs.get("candidate_set"),
            name="qualification probe candidate set",
            repository_root=repository_root,
        )
        plan = load_primitive_qualification_plan(
            plan_path, repository_root=repository_root
        )
        if (
            str(plan.get("candidate_id")) != candidate_id
            or str(plan.get("primitive", "")).upper() != normalized_primitive
        ):
            raise ValueError("qualification probe differs from its frozen plan")
        source_rows = [
            json.loads(line)
            for line in candidate_set_path.read_text().splitlines()
            if line
        ]
        if any(not isinstance(row, Mapping) for row in source_rows):
            raise TypeError("qualification probe candidate-set rows must be objects")
        source_matches = [
            row
            for row in source_rows
            if row.get("schema_version") == "piu.public-candidate-set.v1"
            and row.get("sample_id") == decision.get("sample_id")
            and row.get("initial_state_group") == initial_state_group
        ]
        if len(source_matches) != 1:
            raise ValueError("qualification probe lacks one public candidate row")
        source = source_matches[0]
        if (
            source.get("public_inputs_only") is not True
            or source.get("online_oracle_inputs") != []
            or source.get("candidates") != decision.get("public_candidates")
        ):
            raise ValueError("qualification probe candidate source is not public")
    if (
        decision.get("decision_kind") not in {"EXECUTE", "INTERACT"}
        or " ".join(
            str(decision.get("selected_candidate_primitive", "")).split()
        ).upper()
        != normalized_primitive
    ):
        raise ValueError("qualification controller decision is not executable")
    candidate = decision.get("selected_candidate")
    candidates = decision.get("public_candidates")
    if not isinstance(candidate, Mapping) or not isinstance(candidates, list):
        raise TypeError("qualification decision lacks public candidate payloads")
    candidate = dict(candidate)
    assert_public_policy_value(candidate, path="qualification.selected_candidate")
    assert_public_policy_value(candidates, path="qualification.public_candidates")
    selected = [
        row
        for row in candidates
        if isinstance(row, Mapping)
        and row.get("candidate_id") == candidate_id
        and " ".join(str(row.get("primitive", "")).split()).upper()
        == normalized_primitive
    ]
    if selected != [candidate]:
        raise ValueError("qualification selected candidate differs from its public set")
    references = tuple(
        _spatial_reference_from_mapping(row)
        for row in decision.get("spatial_references", ())
    )
    if normalized_primitive not in {"PICK", "DIRECT"} and references:
        raise ValueError("qualification uses spatial boxes for a non-spatial primitive")
    expected_subtask = serialize_pi05_subtask(
        candidate,
        spatial_references=(
            references if normalized_primitive in {"PICK", "DIRECT"} else ()
        ),
    )
    if not expected_subtask or decision.get("structured_pi05_subtask") != expected_subtask:
        raise ValueError("qualification subtask differs from the frozen serializer")
    return {
        "candidate": candidate,
        "spatial_reference_mode": (
            "calibrated_current_frame_boxes" if references else "none"
        ),
        "structured_subtask": expected_subtask,
        "structured_subtask_sha256": hashlib.sha256(
            expected_subtask.encode()
        ).hexdigest(),
    }


def load_primitive_qualification_schedule(
    path: Path, *, repository_root: Path
) -> dict[str, Any]:
    """Validate an outcome-independent, exact-state qualification cohort."""

    value = json.loads(path.read_text())
    if (
        value.get("schema_version") != "piu.primitive-qualification-schedule.v1"
        or value.get("status")
        != "FROZEN_BEFORE_PRIMITIVE_QUALIFICATION_OUTCOMES"
        or value.get("outcomes_loaded") is not False
    ):
        raise ValueError("unsupported primitive qualification schedule")
    if value.get("claim_scope") != "EXECUTOR_QUALIFICATION_ONLY_NOT_TASK_SUCCESS":
        raise ValueError("primitive qualification schedule overclaims its scope")
    inputs = value.get("inputs")
    if not isinstance(inputs, Mapping):
        raise TypeError("primitive qualification schedule lacks inputs")
    plan_path = _verified_reference(
        inputs.get("plan"), name="primitive qualification plan", repository_root=repository_root
    )
    plan = load_primitive_qualification_plan(plan_path, repository_root=repository_root)
    split_path = _verified_reference(
        inputs.get("split_manifest"),
        name="primitive qualification split manifest",
        repository_root=repository_root,
    )
    from .splits import load_split_manifest

    split = load_split_manifest(split_path)
    qualification_groups = {
        str(row["initial_state_group"]): int(row["seed"])
        for row in split["assignments"]
        if row["split_role"] == "primitive_qualification"
    }
    if len(qualification_groups) != int(plan["design"]["trials"]):
        raise ValueError("qualification split does not match frozen trial count")
    baseline_path = _verified_reference(
        inputs.get("baseline_registry"),
        name="primitive qualification baseline registry",
        repository_root=repository_root,
    )
    scenario_path = _verified_reference(
        inputs.get("scenario_config"),
        name="primitive qualification scenario config",
        repository_root=repository_root,
    )
    identity_path = _verified_reference(
        inputs.get("policy_identity"),
        name="primitive qualification policy identity",
        repository_root=repository_root,
    )
    baseline = yaml.safe_load(baseline_path.read_text())
    scenario = yaml.safe_load(scenario_path.read_text())
    identity = json.loads(identity_path.read_text())
    if baseline.get("schema_version") != "piu.baseline-registry.v1":
        raise ValueError("qualification baseline registry is unsupported")
    if scenario.get("schema_version") != "piu.scenario.v1":
        raise ValueError("qualification scenario config is unsupported")
    if split.get("scenario") != scenario.get("id"):
        raise ValueError("qualification split uses another scenario")
    if identity.get("schema_version") != "piu.pi05-checkpoint-identity.v1":
        raise ValueError("qualification policy identity is unsupported")
    registered_bddl = _resolve_artifact(
        str(baseline.get("scenario", "")), repository_root=repository_root
    ).resolve()
    configured_bddl = _resolve_artifact(
        str(scenario.get("scene", {}).get("bddl", "")),
        repository_root=repository_root,
    ).resolve()
    if registered_bddl != configured_bddl:
        raise ValueError("qualification schedule uses another scenario")
    lock_reference = inputs.get("offline_repro_lock")
    lock_path = _verified_reference(
        lock_reference,
        name="primitive qualification offline release lock",
        repository_root=repository_root,
    )
    manifest_path = repository_root / "configs/experiments/piu_offline_repro_v1.yaml"
    if lock_reference.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("qualification schedule uses another reproduction manifest")
    validate_repro_lock(
        lock_path,
        manifest_path=manifest_path,
        repository_root=repository_root,
    )
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) != len(qualification_groups):
        raise ValueError("qualification schedule has another cohort size")
    namespace = " ".join(str(value.get("permutation_namespace", "")).split())
    if not namespace:
        raise ValueError("qualification schedule lacks a permutation namespace")
    observed_groups = set()
    state_digests = set()
    report_paths = set()
    recomputed = []
    candidate_contract = value.get("candidate_contract")
    if not isinstance(candidate_contract, Mapping):
        raise TypeError("qualification schedule lacks a candidate contract")
    run_root = _resolve_artifact(
        str(value.get("run_root", "")), repository_root=repository_root
    ).resolve()
    if not str(value.get("run_root", "")):
        raise ValueError("qualification schedule lacks a run root")
    plan_digest = _sha256(plan_path)
    for index, row in enumerate(entries):
        if not isinstance(row, Mapping) or row.get("execution_index") != index:
            raise ValueError("qualification execution indices are not contiguous")
        group = " ".join(str(row.get("initial_state_group", "")).split())
        seed = row.get("simulator_seed")
        if group not in qualification_groups or qualification_groups[group] != seed:
            raise ValueError("qualification schedule group/seed differs from split")
        if group in observed_groups:
            raise ValueError("qualification schedule repeats a group")
        observed_groups.add(group)
        if any(str(row.get(name)) != str(plan[name]) for name in ("candidate_id", "primitive", "context")):
            raise ValueError("qualification schedule identity differs from plan")
        controller_path = _verified_reference(
            row.get("controller_report"),
            name=f"qualification controller report {group}",
            repository_root=repository_root,
        )
        controller = load_qualification_controller_decision(
            controller_path,
            candidate_id=str(plan["candidate_id"]),
            primitive=str(plan["primitive"]),
            initial_state_group=group,
            repository_root=repository_root,
        )
        if (
            row.get("structured_subtask_sha256")
            != controller["structured_subtask_sha256"]
            or candidate_contract.get("selected_candidate")
            != controller["candidate"]
            or candidate_contract.get("spatial_reference_mode")
            != controller["spatial_reference_mode"]
            or candidate_contract.get("serializer")
            != "src/piu/executor_bridge.py:serialize_pi05_subtask"
        ):
            raise ValueError("qualification controller differs from candidate contract")
        source = row.get("source_state")
        source_path = _verified_reference(
            source,
            name=f"qualification source state {group}",
            repository_root=repository_root,
        )
        shape, dtype = validate_state_archive(
            source_path, state_key=str(row.get("state_key", ""))
        )
        if source.get("shape") != shape or source.get("dtype") != dtype:
            raise ValueError("qualification source-state metadata differs")
        if source["sha256"] in state_digests:
            raise ValueError("qualification groups reuse an opaque source state")
        state_digests.add(source["sha256"])
        expected_report = str(row.get("expected_execution_receipt", ""))
        if not expected_report or expected_report in report_paths:
            raise ValueError("qualification report destinations are invalid")
        expected_path = run_root / f"{index:05d}_{hashlib.sha256(group.encode()).hexdigest()[:12]}" / "report.json"
        if _resolve_artifact(expected_report, repository_root=repository_root).resolve() != expected_path:
            raise ValueError("qualification execution receipt path differs from schedule")
        report_paths.add(expected_report)
        key = primitive_qualification_permutation_key(
            namespace=namespace,
            plan_sha256=plan_digest,
            initial_state_group=group,
            simulator_seed=int(seed),
            source_state_sha256=str(source["sha256"]),
        )
        if row.get("permutation_key") != key:
            raise ValueError("qualification schedule permutation key differs")
        recomputed.append((key, group))
    if observed_groups != set(qualification_groups):
        raise ValueError("qualification schedule misses a frozen group")
    if [group for _, group in sorted(recomputed)] != [
        str(row["initial_state_group"]) for row in entries
    ]:
        raise ValueError("qualification schedule order is not hash-keyed")
    registry_path = _resolve_artifact(
        str(plan["registry"]["path"]), repository_root=repository_root
    )
    pilot_seeds = set(json.loads(registry_path.read_text()).get("seeds", ()))
    if pilot_seeds & set(qualification_groups.values()):
        raise ValueError("qualification cohort reuses a retrospective pilot seed")
    return value


def _libero_default_open_range(path: Path, *, class_name: str) -> tuple[float, float]:
    """Read the simulator's declared range without importing its GPU stack."""

    tree = ast.parse(path.read_text(), filename=str(path))
    matches: list[tuple[float, float]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign) or len(child.targets) != 1:
                continue
            target = child.targets[0]
            if (
                not isinstance(target, ast.Subscript)
                or "default_open_ranges" not in ast.dump(target)
                or not isinstance(child.value, (ast.List, ast.Tuple))
                or len(child.value.elts) != 2
            ):
                continue
            values = tuple(float(ast.literal_eval(item)) for item in child.value.elts)
            matches.append(values)
    if len(matches) != 1 or not matches[0][0] < matches[0][1]:
        raise ValueError("cannot uniquely recover the LIBERO default open range")
    return matches[0]


def _qualification_success_contract(
    plan: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    risk_path = _resolve_artifact(
        str(plan["risk_contract"]["path"]), repository_root=repository_root
    )
    risk = load_derived_primitive_risk_contract(
        risk_path, repository_root=repository_root
    )
    protocol_path = _resolve_artifact(
        str(risk["inputs"]["primitive_registry_protocol"]["path"]),
        repository_root=repository_root,
    )
    protocol = yaml.safe_load(protocol_path.read_text())
    try:
        contract = protocol["prospective_success_contracts"][str(plan["primitive"])]
    except KeyError as exc:
        raise ValueError("primitive has no simulator-grounded success contract") from exc
    if not isinstance(contract, Mapping):
        raise TypeError("primitive success contract must be a mapping")
    return dict(contract)


def load_primitive_qualification_execution_receipt(
    path: Path,
    *,
    schedule_path: Path,
    schedule: Mapping[str, Any],
    execution_index: int,
    repository_root: Path,
) -> dict[str, Any]:
    """Validate one single-use qualification attempt and its semantic report."""

    value = json.loads(path.read_text())
    if value.get("schema_version") != "piu.primitive-qualification-execution.v1":
        raise ValueError("unsupported primitive qualification execution receipt")
    if value.get("status") not in {"COMPLETED", "FAILED_ATTEMPT_COUNTED"}:
        raise ValueError("primitive qualification attempt is not terminal")
    entries = schedule["entries"]
    if not 0 <= execution_index < len(entries):
        raise ValueError("primitive qualification receipt index is outside schedule")
    entry = entries[execution_index]
    expected_path = _resolve_artifact(
        str(entry["expected_execution_receipt"]), repository_root=repository_root
    )
    if path.resolve() != expected_path.resolve():
        raise ValueError("primitive qualification receipt is outside its scheduled path")
    attempt_path = _verified_reference(
        value.get("attempt"),
        name=f"qualification attempt ticket {execution_index}",
        repository_root=repository_root,
    )
    attempt = json.loads(attempt_path.read_text())
    if (
        attempt_path.resolve() != path.parent / "started.json"
        or attempt.get("schema_version")
        != "piu.primitive-qualification-attempt.v1"
        or attempt.get("status") != "STARTED_SINGLE_USE"
        or attempt.get("execution_index") != execution_index
        or attempt.get("entry") != entry
        or attempt.get("schedule_sha256") != _sha256(schedule_path)
    ):
        raise ValueError("primitive qualification attempt ticket differs")
    expected_previous = (
        None
        if execution_index == 0
        else _sha256(
            _resolve_artifact(
                str(entries[execution_index - 1]["expected_execution_receipt"]),
                repository_root=repository_root,
            )
        )
    )
    if attempt.get("previous_receipt_sha256") != expected_previous:
        raise ValueError("primitive qualification attempt breaks the receipt chain")
    schedule_reference = value.get("schedule")
    if (
        not isinstance(schedule_reference, Mapping)
        or schedule_reference.get("sha256") != _sha256(schedule_path)
        or _resolve_artifact(
            str(schedule_reference.get("path", "")), repository_root=repository_root
        ).resolve()
        != schedule_path.resolve()
        or value.get("execution_index") != execution_index
    ):
        raise ValueError("primitive qualification receipt differs from schedule")
    for name in (
        "initial_state_group",
        "simulator_seed",
        "candidate_id",
        "primitive",
        "context",
    ):
        if value.get(name) != entry[name]:
            raise ValueError(f"primitive qualification receipt differs at {name}")
    if (
        value.get("source_state_sha256") != entry["source_state"]["sha256"]
        or value.get("controller_report_sha256")
        != entry["controller_report"]["sha256"]
        or value.get("structured_subtask_sha256")
        != entry["structured_subtask_sha256"]
    ):
        raise ValueError("primitive qualification receipt provenance differs")
    if value["status"] == "FAILED_ATTEMPT_COUNTED":
        failure = value.get("failure")
        if (
            not isinstance(failure, Mapping)
            or not " ".join(str(failure.get("reason", "")).split())
            or value.get("semantic_report") is not None
        ):
            raise ValueError("failed primitive attempt lacks a closed failure record")
        return value
    semantic_path = _verified_reference(
        value.get("semantic_report"),
        name=f"qualification semantic report {execution_index}",
        repository_root=repository_root,
    )
    report = json.loads(semantic_path.read_text())
    if report.get("schema_version") != "piu.semantic-option.v2":
        raise ValueError("qualification semantic report must use metric contract v2")
    baseline_path = _resolve_artifact(
        str(schedule["inputs"]["baseline_registry"]["path"]),
        repository_root=repository_root,
    )
    scenario_path = _resolve_artifact(
        str(schedule["inputs"]["scenario_config"]["path"]),
        repository_root=repository_root,
    )
    identity_path = _resolve_artifact(
        str(schedule["inputs"]["policy_identity"]["path"]),
        repository_root=repository_root,
    )
    baseline = yaml.safe_load(baseline_path.read_text())
    scenario = yaml.safe_load(scenario_path.read_text())
    controller = report.get("controller", {})
    frozen_decision = load_qualification_controller_decision(
        _resolve_artifact(
            str(entry["controller_report"]["path"]), repository_root=repository_root
        ),
        candidate_id=str(entry["candidate_id"]),
        primitive=str(entry["primitive"]),
        initial_state_group=str(entry["initial_state_group"]),
        repository_root=repository_root,
    )
    expected_identity = load_checkpoint_identity(identity_path)
    validate_server_metadata(controller.get("server_metadata", {}), expected_identity)
    source = controller.get("source_initial_state_transport")
    if (
        report.get("scenario") != baseline["scenario"]
        or report.get("seed") != entry["simulator_seed"]
        or report.get("role") != f"PIU_QUALIFY_{entry['primitive']}"
        or report.get("prompt") != frozen_decision["structured_subtask"]
        or report.get("claim_scope") != "PUBLIC_INPUT_EXECUTION"
        or controller.get("server_mode") != "external"
        or controller.get("online_oracle_inputs") != []
        or controller.get("subtask_steps")
        != baseline["shared_contract"]["option_step_budgets"][entry["primitive"]]
        or not isinstance(source, Mapping)
        or source.get("sha256") != entry["source_state"]["sha256"]
        or source.get("state_key") != entry["state_key"]
        or controller.get("expected_policy_identity", {}).get("sha256")
        != schedule["inputs"]["policy_identity"]["sha256"]
        or scenario.get("scene", {}).get("bddl") != report.get("scenario")
    ):
        raise ValueError("qualification semantic execution differs from frozen entry")
    return value


def primitive_qualification_outcome(
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    repository_root: Path,
) -> tuple[bool, dict[str, Any]]:
    """Derive success only from a registered simulator/task predicate."""

    if receipt.get("status") == "FAILED_ATTEMPT_COUNTED":
        return False, {
            "metric": "single_use_attempt_completed",
            "value": False,
            "provenance": "closed_runtime_failure_counted_in_denominator",
        }
    semantic_path = _resolve_artifact(
        str(receipt["semantic_report"]["path"]), repository_root=repository_root
    )
    report = json.loads(semantic_path.read_text())
    evaluator = report.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise TypeError("qualification semantic report lacks evaluator replay")
    contract = _qualification_success_contract(plan, repository_root=repository_root)
    metric = contract.get("metric")
    if metric == "simulator_articulated_object_is_open":
        source_path = _resolve_artifact(
            str(contract["implementation_source"]), repository_root=repository_root
        )
        low, high = _libero_default_open_range(
            source_path, class_name=str(contract["articulated_object_class"])
        )
        joint = evaluator.get("joints", {}).get(str(contract["tracked_joint"]), {})
        minimum = float(joint["minimum"])
        success = minimum < high
        detail = {
            "metric": metric,
            "observed_minimum_qpos": minimum,
            "simulator_default_open_range": [low, high],
            "comparison": contract["comparison"],
            "implementation_source": {
                "path": str(contract["implementation_source"]),
                "sha256": _sha256(source_path),
            },
        }
    elif metric == "target_grasp_contact_success":
        target = str(evaluator.get("target_object", ""))
        steps = int(evaluator.get("objects", {}).get(target, {}).get("grasp_contact_steps", -1))
        success = steps > 0
        if evaluator.get(metric) is not success:
            raise ValueError("qualification contact metric differs from raw replay")
        detail = {"metric": metric, "target_object": target, "contact_steps": steps}
    elif metric == "target_in_destination_final":
        success = evaluator.get(metric)
        if not isinstance(success, bool):
            raise TypeError("qualification terminal destination metric is not boolean")
        detail = {"metric": metric, "value": success}
    elif metric == "task_success_final":
        success = evaluator.get(metric)
        if not isinstance(success, bool):
            raise TypeError("qualification terminal task metric is not boolean")
        detail = {"metric": metric, "value": success}
    else:
        raise ValueError("unsupported primitive qualification success metric")
    return bool(success), {**detail, "registered_contract": contract}


def load_primitive_qualification_certificate(
    path: Path, *, repository_root: Path
) -> dict[str, Any]:
    """Revalidate the entire plan/outcome chain before authorizing dispatch."""

    value = json.loads(path.read_text())
    if value.get("schema_version") != "piu.primitive-qualification-certificate.v1":
        raise ValueError("unsupported primitive qualification certificate")
    plan_path = _verified_reference(
        value.get("plan"), name="primitive qualification plan", repository_root=repository_root
    )
    schedule_path = _verified_reference(
        value.get("schedule"),
        name="primitive qualification schedule",
        repository_root=repository_root,
    )
    outcomes_path = _verified_reference(
        value.get("outcomes"),
        name="primitive qualification outcomes",
        repository_root=repository_root,
    )
    plan = load_primitive_qualification_plan(plan_path, repository_root=repository_root)
    schedule = load_primitive_qualification_schedule(
        schedule_path, repository_root=repository_root
    )
    if schedule["inputs"]["plan"]["sha256"] != _sha256(plan_path):
        raise ValueError("primitive certificate schedule uses another plan")
    expected_identity = {
        name: str(plan[name]) for name in ("candidate_id", "primitive", "context")
    }
    if any(str(value.get(name)) != item for name, item in expected_identity.items()):
        raise ValueError("primitive certificate identity differs from its plan")
    rows = [
        json.loads(line)
        for line in outcomes_path.read_text().splitlines()
        if line.strip()
    ]
    groups = []
    outcomes = []
    schedule_digest = _sha256(schedule_path)
    entries = schedule["entries"]
    for index, row in enumerate(rows):
        if row.get("schema_version") != "piu.primitive-qualification-outcome.v1":
            raise ValueError("unsupported primitive qualification outcome")
        if any(str(row.get(name)) != item for name, item in expected_identity.items()):
            raise ValueError("primitive outcome identity differs from its plan")
        if not isinstance(row.get("success"), bool):
            raise TypeError("primitive outcome success must be boolean")
        if row.get("success_derived_from_registered_evaluator_contract") is not True:
            raise ValueError("primitive outcome bypassed its evaluator contract")
        if row.get("execution_index") != index or index >= len(entries):
            raise ValueError("primitive outcomes are not in frozen execution order")
        entry = entries[index]
        group = " ".join(str(row.get("initial_state_group", "")).split())
        if not group:
            raise ValueError("primitive qualification outcome lacks a group")
        groups.append(group)
        outcomes.append(bool(row["success"]))
        if (
            group != entry["initial_state_group"]
            or row.get("simulator_seed") != entry["simulator_seed"]
            or row.get("source_state_sha256") != entry["source_state"]["sha256"]
            or row.get("schedule_sha256") != schedule_digest
        ):
            raise ValueError("primitive outcome differs from its scheduled entry")
        receipt_path = _verified_reference(
            row.get("execution_receipt"),
            name=f"primitive execution receipt {index}",
            repository_root=repository_root,
        )
        expected_report = _resolve_artifact(
            str(entry["expected_execution_receipt"]), repository_root=repository_root
        )
        if receipt_path.resolve() != expected_report.resolve():
            raise ValueError("primitive outcome uses an unscheduled execution receipt")
        receipt = load_primitive_qualification_execution_receipt(
            receipt_path,
            schedule_path=schedule_path,
            schedule=schedule,
            execution_index=index,
            repository_root=repository_root,
        )
        derived_success, evaluator = primitive_qualification_outcome(
            receipt, plan=plan, repository_root=repository_root
        )
        if row["success"] is not derived_success or row.get("evaluator") != evaluator:
            raise ValueError("primitive outcome differs from evaluator recomputation")
    if len(groups) != len(set(groups)):
        raise ValueError("primitive qualification outcome groups are not unique")
    design = plan["design"]
    result = evaluate_frozen_binomial_design(
        outcomes,
        null_success_probability=float(plan["risk_contract"]["minimum_reliable_rate"]),
        alpha=float(plan["alpha"]),
        expected_trials=int(design["trials"]),
        rejection_success_count=int(design["rejection_success_count"]),
    )
    expected_status = "FORMALLY_QUALIFIED" if result["qualified"] else "NOT_QUALIFIED"
    if (
        value.get("status") != expected_status
        or value.get("initial_state_groups") != groups
        or value.get("complete_frozen_denominator") is not True
        or value.get("risk_contract") != plan["risk_contract"]
        or value.get("candidate_contract") != schedule["candidate_contract"]
        or float(value.get("alpha")) != float(plan["alpha"])
        or value.get("test") != "exact_one_sided_binomial"
        or value.get("result") != result
        or value.get("paper_method_action_authorized") != bool(result["qualified"])
    ):
        raise ValueError("primitive certificate differs from exact recomputation")
    return value


def load_qualified_executor_map(
    path: Path, *, repository_root: Path
) -> dict[str, Path]:
    """Load only certificates whose full evidence chains still validate."""

    value = json.loads(path.read_text())
    if value.get("schema_version") != "piu.qualified-executor-map.v1":
        raise ValueError("unsupported qualified-executor map")
    if (
        value.get("status") != "FORMALLY_QUALIFIED_CANDIDATES_ONLY"
        or value.get("paper_method_action_authorized") is not True
    ):
        raise ValueError("qualified-executor map is not dispatch-authorizing")
    candidates = value.get("candidates")
    primitives = value.get("primitives")
    if not isinstance(candidates, Mapping) or not isinstance(primitives, Mapping):
        raise TypeError("qualified-executor map entries must be mappings")
    if set(candidates) != set(primitives) or not candidates:
        raise ValueError("qualified-executor map candidate/primitive keys differ")
    result = {}
    for candidate_id, reference in candidates.items():
        certificate_path = _verified_reference(
            reference,
            name=f"qualification for {candidate_id}",
            repository_root=repository_root,
        )
        certificate = load_primitive_qualification_certificate(
            certificate_path, repository_root=repository_root
        )
        if (
            certificate.get("status") != "FORMALLY_QUALIFIED"
            or certificate.get("paper_method_action_authorized") is not True
            or certificate.get("candidate_id") != candidate_id
            or str(certificate.get("primitive", "")).upper()
            != str(primitives[candidate_id]).upper()
        ):
            raise ValueError(f"candidate {candidate_id} is not formally qualified")
        result[str(candidate_id)] = certificate_path
    return result


def validate_qualification_candidate_contract(
    certificate: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    spatial_reference_mode: str,
) -> None:
    """Prevent a valid certificate from being reused for another subtask family."""

    normalized = dict(candidate)
    assert_public_policy_value(normalized, path="qualification.dispatch_candidate")
    contract = certificate.get("candidate_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("selected_candidate") != normalized
        or contract.get("spatial_reference_mode") != spatial_reference_mode
        or contract.get("serializer")
        != "src/piu/executor_bridge.py:serialize_pi05_subtask"
    ):
        raise ValueError(
            "primitive certificate does not cover this exact executor contract"
        )
