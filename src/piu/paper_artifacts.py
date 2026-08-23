"""Evidence-locked PIU paper tables with explicit missing-data semantics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .ablations import BINDING_ABLATIONS
from .statistics import (
    REQUIRED_BINARY_OUTCOMES,
    REQUIRED_CONTINUOUS_OUTCOMES,
    load_analysis_config,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(path: str | Path, repository_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repository_root / value


def _portable(path: Path, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def load_reporting_config(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, Mapping) or value.get("schema_version") != (
        "piu.robustness-reporting.v1"
    ):
        raise ValueError("unsupported PIU robustness/reporting config")
    scope = _mapping(value.get("scope"), name="scope")
    claims = _mapping(value.get("claim_contract"), name="claim_contract")
    if scope.get("interpretation") != "same_scenario_controlled_stress_not_broad_ood":
        raise ValueError("robustness scope must remain a same-scenario stress test")
    if (
        scope.get("scenario_expansion_allowed") is not False
        or scope.get("counterfactual_branches_stay_in_one_group") is not True
        or scope.get("calibration_groups_isolated") is not True
        or scope.get("sealed_groups_opened_once") is not True
    ):
        raise ValueError("robustness split/firewall contract was weakened")
    if (
        claims.get("pending_marker") != "PENDING"
        or claims.get("missing_artifact_imputation") != "forbidden"
        or claims.get("missing_metric_imputation") != "forbidden"
        or claims.get("synthetic_fixture_is_performance_evidence") is not False
        or claims.get("development_ablation_is_sealed_evidence") is not False
        or claims.get("oracle_is_public_method") is not False
        or claims.get("automatic_success_threshold") is not None
        or claims.get("posthoc_subgroup_selection") != "forbidden"
        or claims.get("same_scenario_stress_is_ood_generalization") is not False
    ):
        raise ValueError("reporting config permits unsupported result claims")
    development = _mapping(value.get("development_only"), name="development_only")
    binding = _mapping(
        development.get("binding_input_ablations"),
        name="binding_input_ablations",
    )
    if tuple(binding.get("order", ())) != BINDING_ABLATIONS:
        raise ValueError("reporting ablations differ from executable binder ablations")
    effect = _mapping(
        development.get("effect_training_variants"),
        name="effect_training_variants",
    )
    if tuple(effect.get("order", ())) != (
        "route_only",
        "stop_gradient_effect",
        "joint_effect",
    ):
        raise ValueError("effect reporting variants differ from the frozen experiment")
    formal = _mapping(value.get("formal_method_table"), name="formal_method_table")
    method_order = tuple(str(item) for item in formal.get("method_order", ()))
    if method_order != tuple(f"B{index}" for index in range(9)):
        raise ValueError("formal paper table must contain exactly B0--B8")
    if set(formal.get("public_methods", ())) & set(formal.get("oracle_methods", ())):
        raise ValueError("public and oracle paper columns overlap")
    if set(formal.get("public_methods", ())) | set(formal.get("oracle_methods", ())) != set(
        method_order
    ):
        raise ValueError("paper method classes do not cover B0--B8")
    if any(item not in REQUIRED_BINARY_OUTCOMES for item in formal["binary_columns"]):
        raise ValueError("paper table requests an unsupported binary outcome")
    if any(
        item not in REQUIRED_CONTINUOUS_OUTCOMES
        for item in formal["continuous_columns"]
    ):
        raise ValueError("paper table requests an unsupported continuous outcome")
    registry_path = _resolve(formal["registry"], repository_root)
    registry = yaml.safe_load(registry_path.read_text())
    if not isinstance(registry, Mapping) or registry.get("schema_version") != (
        "piu.baseline-registry.v1"
    ):
        raise ValueError("unsupported baseline registry")
    if registry.get("scenario") != scope.get("scenario"):
        raise ValueError("reporting protocol and baseline registry use different scenarios")
    registry_methods = tuple(str(row["id"]) for row in registry.get("methods", ()))
    if registry_methods != method_order:
        raise ValueError("baseline registry order differs from the paper table")
    analysis_path = _resolve(formal["analysis"], repository_root)
    analysis = load_analysis_config(analysis_path)
    if tuple(analysis["population"]["required_method_ids"]) != method_order:
        raise ValueError("formal analysis and paper table use different methods")
    if tuple(analysis["reporting"]["oracle_methods"]) != tuple(
        formal["oracle_methods"]
    ):
        raise ValueError("formal analysis and paper table disagree about oracles")
    prompt_stress = _mapping(
        value.get("same_scenario_prompt_stress"), name="same_scenario_prompt_stress"
    )
    variants = prompt_stress.get("variants")
    if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
        raise TypeError("prompt stress variants must be a sequence")
    ids = [str(_mapping(item, name="prompt stress row").get("id", "")) for item in variants]
    prompts = [str(item.get("prompt", "")).strip() for item in variants]
    if len(ids) < 2 or len(set(ids)) != len(ids) or not all(ids + prompts):
        raise ValueError("prompt stress variants must be nonempty and unique")
    if prompts[0] != str(prompt_stress.get("source_prompt", "")).strip():
        raise ValueError("first prompt stress variant must be the source prompt")
    if (
        prompt_stress.get("selection_use") != "none"
        or prompt_stress.get("grouping")
        != "cluster_all_prompt_variants_by_initial_state_group"
    ):
        raise ValueError("prompt stress may not select a model or split paired groups")
    outputs = _mapping(value.get("outputs"), name="outputs")
    if outputs.get("immutable") is not True:
        raise ValueError("paper table outputs must be immutable")
    json_output = Path(str(outputs.get("json", "")))
    markdown_output = Path(str(outputs.get("markdown", "")))
    if (
        json_output.suffix != ".json"
        or markdown_output.suffix != ".md"
        or json_output == markdown_output
        or json_output.is_absolute()
        or markdown_output.is_absolute()
    ):
        raise ValueError("paper table output paths are malformed")
    return dict(value)


def _load_json_with_schema(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping) or value.get("schema_version") != schema:
        raise ValueError(f"{path} does not match schema {schema}")
    return dict(value)


def _verify_reference(reference: Mapping[str, Any], repository_root: Path) -> Path:
    path = _resolve(str(reference.get("path", "")), repository_root)
    digest = str(reference.get("sha256", ""))
    if not path.is_file() or sha256(path) != digest:
        raise ValueError(f"artifact reference differs from its hash: {path}")
    return path


def _available_artifact(
    path: Path,
    *,
    schema: str,
    repository_root: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not path.exists():
        return None, {
            "status": "PENDING",
            "path": _portable(path, repository_root),
            "sha256": None,
        }
    value = _load_json_with_schema(path, schema)
    return value, {
        "status": "AVAILABLE",
        "path": _portable(path, repository_root),
        "sha256": sha256(path),
    }


def _verify_training_report(
    value: Mapping[str, Any], *, kind: str, repository_root: Path
) -> None:
    if value.get("paper_method_claim_allowed") is not False:
        raise ValueError(f"{kind} development report overclaims paper evidence")
    if value.get("sealed_test_loaded") is not False or value.get(
        "calibration_loaded"
    ) is not False:
        raise ValueError(f"{kind} development report crossed a split firewall")
    for reference in _mapping(value.get("inputs"), name=f"{kind} inputs").values():
        _verify_reference(_mapping(reference, name=f"{kind} input"), repository_root)
    if kind == "binding":
        _verify_reference(
            _mapping(value.get("config"), name="binding config"), repository_root
        )
        _verify_reference(
            _mapping(value.get("checkpoint"), name="binding checkpoint"),
            repository_root,
        )
        _verify_reference(
            _mapping(
                value.get("development_predictions"),
                name="binding development predictions",
            ),
            repository_root,
        )
    elif kind == "effect":
        for variant_name, variant in _mapping(
            value.get("variants"), name="effect variants"
        ).items():
            row = _mapping(variant, name=str(variant_name))
            _verify_reference(
                _mapping(row.get("checkpoint"), name="effect checkpoint"),
                repository_root,
            )
            _verify_reference(
                _mapping(
                    row.get("development_predictions"),
                    name="effect development predictions",
                ),
                repository_root,
            )


def _verify_sealed_evaluation(
    value: Mapping[str, Any], *, kind: str, repository_root: Path
) -> None:
    if value.get("sealed_test_opened") is not True:
        raise ValueError(f"{kind} report is not a sealed evaluation")
    if value.get("paper_method_claim_allowed") is not False:
        raise ValueError(f"{kind} component report overclaims end-to-end evidence")
    inputs = _mapping(value.get("inputs"), name=f"{kind} inputs")
    for reference in inputs.values():
        _verify_reference(_mapping(reference, name=f"{kind} input"), repository_root)
    prediction_report_path = _verify_reference(
        _mapping(inputs.get("prediction_report"), name="prediction report"),
        repository_root,
    )
    prediction_report = json.loads(prediction_report_path.read_text())
    authorization = _mapping(
        prediction_report.get("sealed_authorization"), name="sealed authorization"
    )
    authorization_path = _verify_reference(authorization, repository_root)
    authorization_value = json.loads(authorization_path.read_text())
    if kind == "binding":
        expected_schema = "piu.sealed-test-authorization.v1"
        expected_hashes = {
            "checkpoint_sha256": prediction_report["inputs"]["checkpoint"][
                "sha256"
            ],
            "feature_sha256": prediction_report["inputs"]["features"]["sha256"],
            "label_sha256": prediction_report["inputs"]["labels"]["sha256"],
        }
    else:
        expected_schema = "piu.action-effect-sealed-authorization.v1"
        expected_hashes = {
            "checkpoint_sha256": prediction_report["inputs"]["checkpoint"][
                "sha256"
            ],
            "feature_sha256": prediction_report["inputs"]["features"]["sha256"],
            "binding_prediction_sha256": prediction_report["inputs"][
                "binding_predictions"
            ]["sha256"],
            "effect_label_sha256": prediction_report["inputs"]["labels"][
                "sha256"
            ],
        }
    if authorization_value.get("schema_version") != expected_schema:
        raise ValueError(f"{kind} sealed authorization has another schema")
    expected_hashes["single_use_output"] = prediction_report["output"]["path"]
    for name, expected in expected_hashes.items():
        if authorization_value.get(name) != expected:
            raise ValueError(f"{kind} sealed authorization differs at {name}")


def _verify_formal_report(
    value: Mapping[str, Any], *, report_path: Path, repository_root: Path
) -> None:
    if value.get("automatic_method_pass") is not None:
        raise ValueError("formal report introduced an automatic method pass")
    if tuple(value.get("public_methods", ())) != (
        "B0",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B8",
    ):
        raise ValueError("formal report public methods differ from the registry")
    if tuple(value.get("oracle_upper_bound_methods", ())) != ("B6", "B7"):
        raise ValueError("formal report oracle methods differ from the registry")
    inputs = _mapping(value.get("inputs"), name="formal inputs")
    for reference in inputs.values():
        _verify_reference(_mapping(reference, name="formal input"), repository_root)
    authorization_path = _verify_reference(
        _mapping(value.get("sealed_authorization"), name="formal authorization"),
        repository_root,
    )
    authorization = json.loads(authorization_path.read_text())
    if authorization.get("schema_version") != (
        "piu.formal-analysis-sealed-authorization.v1"
    ):
        raise ValueError("formal report carries another authorization schema")
    expected = {
        "outcomes_sha256": inputs["outcomes"]["sha256"],
        "config_sha256": inputs["config"]["sha256"],
        "single_use_output": _portable(report_path, repository_root),
    }
    for name, item in expected.items():
        if authorization.get(name) != item:
            raise ValueError(f"formal authorization differs at {name}")


def _development_binding_rows(
    config: Mapping[str, Any], report: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    spec = config["development_only"]["binding_input_ablations"]
    metrics = tuple(str(item) for item in spec["metrics"])
    if report is None:
        return [
            {"ablation": name, **{metric: "PENDING" for metric in metrics}}
            for name in spec["order"]
        ]
    observed = _mapping(report.get("development_ablations"), name="binding ablations")
    if tuple(observed) != tuple(spec["order"]):
        raise ValueError("binding training report is missing a frozen ablation")
    rows = []
    for name in spec["order"]:
        values = _mapping(observed[name].get("development_metrics"), name=name)
        missing = set(metrics) - set(values)
        if missing:
            raise ValueError(f"binding ablation {name} lacks metrics {sorted(missing)}")
        rows.append(
            {
                "ablation": name,
                **{
                    metric: (
                        "UNSUPPORTED" if values[metric] is None else values[metric]
                    )
                    for metric in metrics
                },
            }
        )
    return rows


def _development_effect_rows(
    config: Mapping[str, Any], report: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    spec = config["development_only"]["effect_training_variants"]
    metrics = tuple(str(item) for item in spec["metrics"])
    if report is None:
        return [
            {"variant": name, **{metric: "PENDING" for metric in metrics}}
            for name in spec["order"]
        ]
    observed = _mapping(report.get("variants"), name="effect variants")
    if tuple(observed) != tuple(spec["order"]):
        raise ValueError("effect training report is missing a frozen variant")
    rows = []
    for name in spec["order"]:
        variant = _mapping(observed[name], name=name)
        selected = int(variant["selected_trial"])
        trials = variant.get("trials")
        if not isinstance(trials, Sequence) or isinstance(trials, (str, bytes)):
            raise TypeError("effect variant trials must be a sequence")
        matches = [item for item in trials if int(item.get("trial", -1)) == selected]
        if len(matches) != 1:
            raise ValueError(f"effect variant {name} selected trial is ambiguous")
        values = _mapping(matches[0].get("development_metrics"), name=name)
        missing = set(metrics) - set(values)
        if missing:
            raise ValueError(f"effect variant {name} lacks metrics {sorted(missing)}")
        rows.append(
            {
                "variant": name,
                **{
                    metric: (
                        "UNSUPPORTED" if values[metric] is None else values[metric]
                    )
                    for metric in metrics
                },
            }
        )
    return rows


def _formal_rows(
    config: Mapping[str, Any], report: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    spec = config["formal_method_table"]
    columns = [*spec["binary_columns"], *spec["continuous_columns"]]
    if report is None:
        return [
            {
                "method_id": method,
                "evidence_class": (
                    "oracle_upper_bound"
                    if method in spec["oracle_methods"]
                    else "public_method"
                ),
                **{column: "PENDING" for column in columns},
            }
            for method in spec["method_order"]
        ]
    binary = _mapping(
        report.get("binary_descriptive_by_method"),
        name="formal binary descriptions",
    )
    continuous = _mapping(
        report.get("continuous_descriptive_only"),
        name="formal continuous descriptions",
    )
    rows = []
    for method in spec["method_order"]:
        row: dict[str, Any] = {
            "method_id": method,
            "evidence_class": (
                "oracle_upper_bound"
                if method in spec["oracle_methods"]
                else "public_method"
            ),
        }
        for outcome in spec["binary_columns"]:
            row[outcome] = dict(
                _mapping(
                    _mapping(binary.get(outcome), name=outcome).get(method),
                    name=f"{outcome}/{method}",
                )
            )
        for outcome in spec["continuous_columns"]:
            row[outcome] = dict(
                _mapping(
                    _mapping(continuous.get(outcome), name=outcome)
                    .get("arms", {})
                    .get(method),
                    name=f"{outcome}/{method}",
                )
            )
        rows.append(row)
    return rows


def _retained_negative_rows(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    aggregates = _mapping(value.get("aggregates"), name="retained aggregates")
    requested = (
        ("direct_closed_butter", "target_pick"),
        ("direct_closed_butter", "wrong_object_contact"),
        ("open_closed_drawer", "drawer_open"),
        ("open_closed_drawer", "information_acquired"),
        ("direct_after_actual_open", "target_visible_initial"),
        ("direct_after_actual_open", "target_pick"),
        ("direct_after_actual_open", "wrong_object_contact"),
        ("direct_after_actual_open", "task_success"),
        ("direct_visible_cream_cheese", "target_pick"),
        ("direct_visible_cream_cheese", "target_destination_final"),
    )
    rows = []
    for condition, metric in requested:
        summary = _mapping(
            _mapping(aggregates.get(condition), name=condition).get(metric),
            name=f"{condition}/{metric}",
        )
        rows.append({"condition": condition, "metric": metric, **dict(summary)})
    return rows


def _verify_retained_negative(
    value: Mapping[str, Any],
    *,
    expected_scenario: str,
    repository_root: Path,
) -> None:
    if value.get("scenario") != expected_scenario:
        raise ValueError("retained evidence uses another scenario")
    if value.get("online_oracle_input_count") != 0:
        raise ValueError("retained public evidence unexpectedly uses an oracle")
    if value.get("overall_executor_gate_passed") is not False:
        raise ValueError("retained negative evidence no longer records a failed gate")
    _verify_reference(
        _mapping(value.get("config"), name="retained config"), repository_root
    )
    for reference in _mapping(
        value.get("task_specific_relabels"), name="retained relabels"
    ).values():
        _verify_reference(
            _mapping(reference, name="retained relabel"), repository_root
        )


def build_evidence_tables(
    config_path: Path, *, repository_root: Path
) -> dict[str, Any]:
    """Build table data without inventing absent empirical values."""

    config = load_reporting_config(config_path, repository_root=repository_root)
    development = config["development_only"]
    binding_path = _resolve(
        development["binding_input_ablations"]["source"], repository_root
    )
    effect_path = _resolve(
        development["effect_training_variants"]["source"], repository_root
    )
    binding_report, binding_status = _available_artifact(
        binding_path,
        schema="piu.target-binder-training.v1",
        repository_root=repository_root,
    )
    if binding_report is not None:
        _verify_training_report(
            binding_report, kind="binding", repository_root=repository_root
        )
    effect_report, effect_status = _available_artifact(
        effect_path,
        schema="piu.action-effect-training.v1",
        repository_root=repository_root,
    )
    if effect_report is not None:
        _verify_training_report(
            effect_report, kind="effect", repository_root=repository_root
        )
    sealed_values: dict[str, dict[str, Any] | None] = {}
    sealed_status: dict[str, dict[str, Any]] = {}
    for name, specification in config["sealed_reports"].items():
        path = _resolve(specification["path"], repository_root)
        value, status = _available_artifact(
            path,
            schema=specification["schema_version"],
            repository_root=repository_root,
        )
        sealed_values[name] = value
        sealed_status[name] = status
    if sealed_values["target_binding"] is not None:
        _verify_sealed_evaluation(
            sealed_values["target_binding"],
            kind="binding",
            repository_root=repository_root,
        )
    if sealed_values["action_effect"] is not None:
        _verify_sealed_evaluation(
            sealed_values["action_effect"],
            kind="effect",
            repository_root=repository_root,
        )
    if sealed_values["closed_loop"] is not None:
        _verify_formal_report(
            sealed_values["closed_loop"],
            report_path=_resolve(
                config["sealed_reports"]["closed_loop"]["path"], repository_root
            ),
            repository_root=repository_root,
        )
    retained_spec = config["retained_negative_evidence"]
    retained_path = _resolve(retained_spec["path"], repository_root)
    retained = _load_json_with_schema(retained_path, retained_spec["schema_version"])
    _verify_retained_negative(
        retained,
        expected_scenario=config["scope"]["scenario"],
        repository_root=repository_root,
    )
    retained_status = {
        "status": "AVAILABLE",
        "path": _portable(retained_path, repository_root),
        "sha256": sha256(retained_path),
    }
    evidence_complete = all(
        item["status"] == "AVAILABLE" for item in sealed_status.values()
    )
    return {
        "schema_version": "piu.paper-evidence-tables.v1",
        "claim_scope": (
            "AUTOGENERATED_EVIDENCE_TABLES_WITHOUT_AUTOMATIC_METHOD_CLAIM"
        ),
        "config": {
            "path": _portable(config_path, repository_root),
            "sha256": sha256(config_path),
        },
        "scenario": config["scope"]["scenario"],
        "scope_interpretation": config["scope"]["interpretation"],
        "artifacts": {
            "retained_negative": retained_status,
            "binding_development": binding_status,
            "effect_development": effect_status,
            **{f"{name}_sealed": status for name, status in sealed_status.items()},
        },
        "retained_negative_evidence": _retained_negative_rows(retained),
        "binding_development_ablations": _development_binding_rows(
            config, binding_report
        ),
        "effect_development_variants": _development_effect_rows(
            config, effect_report
        ),
        "formal_method_table": _formal_rows(
            config, sealed_values["closed_loop"]
        ),
        "same_scenario_prompt_stress": {
            "status": "PENDING",
            "variants": config["same_scenario_prompt_stress"]["variants"],
            "selection_use": "none",
            "broad_ood_claim_allowed": False,
        },
        "main_table_evidence_complete": evidence_complete,
        "automatic_method_success": None,
        "paper_method_claim_allowed": False,
        "paper_method_claim_blocker": (
            "Missing sealed artifacts remain PENDING. Even after the evidence "
            "matrix is complete, the preregistered multi-metric interpretation "
            "contract requires a claim audit; no metric threshold auto-creates "
            "a successful method claim."
        ),
        "missing_values_encoded_as_zero": False,
        "local_gpu_actions_performed": False,
    }


def _format_scalar(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "UNSUPPORTED"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    raise TypeError(f"cannot render table value {value!r}")


def _format_binary(value: Any) -> str:
    if isinstance(value, str):
        return value
    item = _mapping(value, name="binary table cell")
    return f"{int(item['successes'])}/{int(item['trials'])} ({float(item['rate']):.1%})"


def _format_continuous(value: Any) -> str:
    if isinstance(value, str):
        return value
    item = _mapping(value, name="continuous table cell")
    return f"{float(item['mean']):.2f} / {float(item['median']):.2f}"


def render_markdown(value: Mapping[str, Any]) -> str:
    """Render deterministic Markdown from a validated evidence-table payload."""

    lines = [
        "# PIU evidence tables",
        "",
        "> Generated from hash-checked artifacts. `PENDING` means no admissible "
        "artifact exists; it never means zero. Development results are not sealed "
        "test evidence. B6/B7 are oracle upper bounds.",
        "",
        "## Evidence readiness",
        "",
        "| artifact | status | SHA-256 |",
        "|---|---:|---|",
    ]
    for name, artifact in value["artifacts"].items():
        lines.append(
            f"| {name} | {artifact['status']} | {artifact['sha256'] or 'PENDING'} |"
        )
    lines.extend(
        [
            "",
            "## Retained fixed-scenario negative/qualification evidence",
            "",
            "| condition | metric | result |",
            "|---|---|---:|",
        ]
    )
    for row in value["retained_negative_evidence"]:
        lines.append(
            f"| {row['condition']} | {row['metric']} | "
            f"{row['successes']}/{row['trials']} ({row['rate']:.1%}) |"
        )
    binding_rows = value["binding_development_ablations"]
    binding_metrics = [key for key in binding_rows[0] if key != "ablation"]
    lines.extend(
        [
            "",
            "## Binding input ablations (development only)",
            "",
            "| ablation | " + " | ".join(binding_metrics) + " |",
            "|---|" + "---:|" * len(binding_metrics),
        ]
    )
    for row in binding_rows:
        lines.append(
            f"| {row['ablation']} | "
            + " | ".join(_format_scalar(row[name]) for name in binding_metrics)
            + " |"
        )
    effect_rows = value["effect_development_variants"]
    effect_metrics = [key for key in effect_rows[0] if key != "variant"]
    lines.extend(
        [
            "",
            "## Effect/route variants (development only)",
            "",
            "| variant | " + " | ".join(effect_metrics) + " |",
            "|---|" + "---:|" * len(effect_metrics),
        ]
    )
    for row in effect_rows:
        lines.append(
            f"| {row['variant']} | "
            + " | ".join(_format_scalar(row[name]) for name in effect_metrics)
            + " |"
        )
    formal_rows = value["formal_method_table"]
    formal_columns = [
        key for key in formal_rows[0] if key not in {"method_id", "evidence_class"}
    ]
    lines.extend(
        [
            "",
            "## Sealed paired B0--B8 table",
            "",
            "| method | class | " + " | ".join(formal_columns) + " |",
            "|---|---|" + "---:|" * len(formal_columns),
        ]
    )
    binary_columns = set(REQUIRED_BINARY_OUTCOMES)
    for row in formal_rows:
        cells = []
        for name in formal_columns:
            cells.append(
                _format_binary(row[name])
                if name in binary_columns
                else _format_continuous(row[name])
            )
        lines.append(
            f"| {row['method_id']} | {row['evidence_class']} | "
            + " | ".join(cells)
            + " |"
        )
    lines.extend(
        [
            "",
            "Continuous cells are `mean / median`. This table contains no "
            "automatic success decision.",
            "",
            "## Claim boundary",
            "",
            f"- Main-table evidence complete: `{str(value['main_table_evidence_complete']).lower()}`.",
            "- Automatic method success: `null`.",
            "- Broad OOD claim allowed: `false`.",
            "- Missing values encoded as zero: `false`.",
            "",
        ]
    )
    return "\n".join(lines)
