"""Collect real reset-controlled Method-V1 branch outcomes.

The CLI is intentionally split across environments:

render
    Reproduce one exact LIBERO reset and export only public RGB/state.
freeze
    In the isolated Qwen environment, propose once, extract frozen features,
    and freeze the complete candidate by paired-seed schedule.
run
    In the LIBERO environment, execute exactly one schedule entry with
    MolmoAct2, the online Qwen proposal service, and an isolated learned-scorer
    verifier when a frozen learned selection is supplied.
finalize
    Validate consumed attempts and summarize real labels without inventing
    missing branches.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .continuation import (
    FixedContinuationIdentity,
    FixedHorizonContinuation,
    FixedHorizonStatus,
)
from .contracts import (
    ExecutionStatus,
    GroundedIntervention,
    PolicyContext,
    Primitive,
    canonical_json_bytes,
    canonical_sha256,
)
from .data import ObservedBranch
from .libero_runtime import LiberoE1Environment
from .method_v1_config import (
    load_method_v1_config,
    method_v1_information_stratum_counts,
    resolve_method_v1_identity,
    validate_resolved_method_v1_identity,
)
from .method_v1_data import (
    BranchScheduleEntry,
    CollectionAttempt,
    CollectionAttemptStatus,
    DecisionGroupManifest,
    InformationStratum,
    InfrastructureFailure,
    build_branch_schedule,
    validate_branch_schedule,
    validate_manifest_information_strata,
    validate_manifest_splits,
)
from .method_v1_data import (
    ProposalRejection as ManifestProposalRejection,
)
from .method_v1_runtime import (
    LiberoMolmoBudgetedExecutor,
    PolicyExecutionFailure,
    public_context_from_observation,
)
from .method_v1_trace import validate_method_v1_execution_trace
from .molmoact2 import MolmoAct2HTTPClient, MolmoAct2ServerIdentity
from .proposals import (
    ProposalFailure,
    Qwen25VLProposer,
    parse_qwen_proposal_json,
    qwen_proposer_id,
)
from .qwen_provider import (
    QWEN25VL_TRANSFORMERS_VERSION,
    Qwen25VLRuntime,
    Qwen25VLTokenProvider,
    QwenFeatureCache,
    QwenProviderIdentity,
    candidate_feature_cache_key,
    save_grounding_support_overlay,
)
from .qwen_service import QwenProposalHTTPClient
from .rgb import RGBFrameStore
from .scorer_verification_client import ScorerVerificationHTTPClient

SCENE_SPEC_SCHEMA = "method-v1-scene-reset-spec-v2"
PREPARED_SCHEMA = "method-v1-prepared-public-reset-v1"
SCHEDULE_FILE_SCHEMA = "method-v1-schedule-file-v1"
FREEZE_RECEIPT_SCHEMA = "method-v1-decision-freeze-receipt-v4"
INVENTORY_FREEZE_RECEIPT_SCHEMA = "method-v1-decision-freeze-receipt-v5"
COLLECTION_PLAN_SCHEMA = "method-v1-global-collection-plan-v1"
INVENTORY_COLLECTION_PLAN_SCHEMA = "method-v1-global-collection-plan-v2"
RESET_INVENTORY_SCHEMA = "method-v1-pre-proposal-reset-inventory-v1"
EXECUTION_CLAIM_SCHEMA = "method-v1-single-use-execution-claim-v2"

_COLLECTION_INFRASTRUCTURE_POLICY = {
    "before_claim": "fail_without_consuming_schedule_entry",
    "after_claim": "record_unlabelled_infrastructure_failure_and_never_rerun",
    "policy_output_failure": "evaluate_final_task_outcome_when_physically_obtainable",
    "missing_attempt": "reject_canonical_dataset_admission",
    "single_use_schedule_entries": True,
}

_METHOD_V1_RUNTIME_SOURCE_NAMES = (
    "collect_outcomes.py",
    "contracts.py",
    "continuation.py",
    "data.py",
    "execution.py",
    "libero_runtime.py",
    "losses.py",
    "method_v1_config.py",
    "method_v1_data.py",
    "method_v1_runtime.py",
    "method_v1_trace.py",
    "model.py",
    "molmoact2.py",
    "proposals.py",
    "qwen_provider.py",
    "qwen_service.py",
    "rgb.py",
    "scorer_verification_client.py",
    "scorer_verification_service.py",
    "selection.py",
    "selection_provenance.py",
    "serialization.py",
    "tokens.py",
    "train_outcomes.py",
)


def _clean_text(value: Any, *, name: str) -> str:
    value = " ".join(str(value or "").split())
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read strict JSON file {source}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{source} must contain a JSON object")
    return value


def _write_json_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path, *, exclude: Sequence[str] = ()) -> str:
    excluded = set(exclude)
    rows: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append(f"{_file_sha256(path)} {path.stat().st_size} {relative}\n")
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def _method_v1_runtime_source_sha256() -> str:
    """Bind decision freeze and execution to the same Method-V1 source bytes."""

    package = Path(__file__).resolve().parent
    rows = []
    for name in _METHOD_V1_RUNTIME_SOURCE_NAMES:
        path = package / name
        rows.append({"path": name, "sha256": _file_sha256(path)})
    return canonical_sha256(rows)


def load_verified_collection_attempt(path: str | Path) -> CollectionAttempt:
    """Load one canonical attempt after verifying its sealed artifact tree.

    ``attempt.json`` is the manifest for the other immutable files in its
    directory, so it is excluded from the tree digest.  Centralizing this
    admission check keeps collection finalization and training consistent.
    """

    source = Path(path).expanduser().resolve()
    if source.name != "attempt.json":
        raise ValueError("attempt path must name the canonical attempt.json")
    attempt = CollectionAttempt.from_mapping(_read_json(source))
    observed_tree = _tree_sha256(source.parent, exclude=("attempt.json",))
    if observed_tree != attempt.artifact_tree_sha256:
        raise ValueError(
            f"attempt artifact tree digest mismatch for {attempt.attempt_id}"
        )
    validate_method_v1_execution_trace(attempt_dir=source.parent, attempt=attempt)
    public_branch_path = source.parent / "observed_branch.public.json"
    evaluator_sidecar_path = source.parent / "private" / "evaluator_sidecar.json"
    infrastructure_path = source.parent / "infrastructure_failure.json"
    if attempt.status is CollectionAttemptStatus.OUTCOME_EVALUATED:
        if attempt.branch is None:  # pragma: no cover - dataclass guards this.
            raise RuntimeError("outcome-evaluated attempt lost its branch")
        if not public_branch_path.is_file() or not evaluator_sidecar_path.is_file():
            raise ValueError(
                "outcome attempt lacks canonical branch/evaluator artifacts"
            )
        if infrastructure_path.exists():
            raise ValueError("outcome attempt also contains infrastructure failure")
        public_branch = _read_json(public_branch_path)
        if public_branch != attempt.branch.to_dict(include_private=False):
            raise ValueError(
                "attempt branch differs from sealed public branch artifact"
            )
        sidecar = _read_json(evaluator_sidecar_path)
        if set(sidecar) != {
            "evaluator_predicate",
            "final_task_success",
            "policy_input_changed_by_evaluator",
        }:
            raise ValueError("evaluator sidecar fields differ from schema")
        if not isinstance(sidecar["final_task_success"], bool):
            raise TypeError("evaluator final_task_success must be bool")
        predicate = sidecar["evaluator_predicate"]
        if (
            not isinstance(predicate, list)
            or not predicate
            or any(
                not isinstance(value, str) or not value.strip() for value in predicate
            )
        ):
            raise ValueError("evaluator predicate must be a non-empty string list")
        if sidecar["policy_input_changed_by_evaluator"] is not False:
            raise ValueError("private evaluator data entered the policy path")
        if sidecar["final_task_success"] is not attempt.branch.observed_outcome:
            raise ValueError("sealed evaluator label differs from attempt branch")
    else:
        if public_branch_path.exists() or evaluator_sidecar_path.exists():
            raise ValueError("infrastructure failure may not contain outcome artifacts")
        if not infrastructure_path.is_file():
            raise ValueError("infrastructure failure lacks canonical diagnostics")
        if attempt.infrastructure_failure is None:  # pragma: no cover - guarded.
            raise RuntimeError("infrastructure attempt lost its failure metadata")
        failure = _read_json(infrastructure_path)
        if set(failure) != {
            "status",
            "failure_type",
            "message",
            "traceback",
            "selection_sha256",
            "collection_plan_sha256",
            "scorer_verifier_auth_key_id",
            "public_trace_sha256",
        }:
            raise ValueError("infrastructure diagnostics fields differ from schema")
        expected_failure = attempt.infrastructure_failure
        if (
            failure["status"] != "INFRASTRUCTURE_FAILURE"
            or failure["failure_type"] != expected_failure.failure_type
            or failure["message"] != expected_failure.message
            or canonical_sha256(failure) != expected_failure.diagnostics_sha256
        ):
            raise ValueError("infrastructure diagnostics differ from attempt")
        trace_digest = failure["public_trace_sha256"]
        trace_path = source.parent / "public_execution_trace.json"
        if trace_digest is None:
            if trace_path.exists():
                raise ValueError("infrastructure trace exists without a bound digest")
        else:
            if (
                not isinstance(trace_digest, str)
                or len(trace_digest) != 64
                or any(
                    character not in "0123456789abcdef" for character in trace_digest
                )
                or not trace_path.is_file()
                or canonical_sha256(_read_json(trace_path)) != trace_digest
            ):
                raise ValueError("infrastructure public trace digest mismatch")
    started_path = source.parent / "started.json"
    if (
        attempt.status is CollectionAttemptStatus.OUTCOME_EVALUATED
        and not started_path.is_file()
    ):
        raise ValueError("outcome attempt lacks collection-plan-bound started artifact")
    if started_path.is_file():
        started = _read_json(started_path)
        if set(started) != {
            "status",
            "entry",
            "manifest_sha256",
            "selection_sha256",
            "collection_plan_sha256",
            "scorer_verifier_auth_key_id",
        }:
            raise ValueError("started artifact fields differ from schema")
        if (
            started["status"] != "STARTED"
            or BranchScheduleEntry.from_mapping(started["entry"])
            != attempt.schedule_entry
        ):
            raise ValueError("started artifact changes schedule entry")
        for name in (
            "manifest_sha256",
            "collection_plan_sha256",
            "scorer_verifier_auth_key_id",
        ):
            _lower_sha256(started[name], name=f"started artifact {name}")
    return attempt


@dataclasses.dataclass(frozen=True)
class SceneResetSpec:
    """Exact simulator input plus evaluator-only final predicate."""

    scene_id: str
    layout_id: str
    initial_state_group: str
    decision_group_id: str
    split_group_id: str
    split: str
    information_stratum: InformationStratum
    prompt: str
    bddl_file: str
    init_states_file: str
    init_state_index: int
    env_seed: int
    reset_state_sha256: str
    evaluator_predicate: tuple[str, ...]
    model_seeds: tuple[int, int]
    image_size: int = 256
    settle_steps: int = 10
    control_mode: str = "relative"
    schema_version: str = SCENE_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SCENE_SPEC_SCHEMA:
            raise ValueError("unsupported scene reset spec schema")
        for name in (
            "scene_id",
            "layout_id",
            "initial_state_group",
            "decision_group_id",
            "split_group_id",
            "split",
            "prompt",
            "bddl_file",
            "init_states_file",
        ):
            object.__setattr__(self, name, _clean_text(getattr(self, name), name=name))
        if self.split not in {"train", "validation", "calibration", "test"}:
            raise ValueError("scene split is invalid")
        try:
            stratum = InformationStratum(self.information_stratum)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "information_stratum must be one of "
                f"{[item.value for item in InformationStratum]}"
            ) from error
        object.__setattr__(self, "information_stratum", stratum)
        for name in ("init_state_index", "env_seed", "settle_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.image_size != 256 or self.control_mode != "relative":
            raise ValueError("Method-V1 uses 256px RGB and relative LIBERO control")
        if len(self.reset_state_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.reset_state_sha256
        ):
            raise ValueError("reset_state_sha256 must be a lowercase SHA-256 digest")
        predicate = tuple(
            _clean_text(item, name="predicate item")
            for item in self.evaluator_predicate
        )
        if len(predicate) not in {2, 3}:
            raise ValueError("evaluator predicate must have two or three items")
        object.__setattr__(self, "evaluator_predicate", predicate)
        seeds = tuple(self.model_seeds)
        if len(seeds) != 2:
            raise ValueError("Method-V1 requires exactly two paired model seeds")
        if any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
            for seed in seeds
        ):
            raise ValueError("model_seeds must contain non-negative integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("paired model seeds must be unique")
        object.__setattr__(self, "model_seeds", seeds)

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["information_stratum"] = self.information_stratum.value
        value["evaluator_predicate"] = list(self.evaluator_predicate)
        value["model_seeds"] = list(self.model_seeds)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SceneResetSpec:
        expected = {field.name for field in dataclasses.fields(cls)}
        if set(value) != expected:
            raise ValueError("scene reset spec fields differ from schema")
        return cls(
            **{
                field.name: (
                    tuple(value[field.name])
                    if field.name in {"evaluator_predicate", "model_seeds"}
                    else InformationStratum(value[field.name])
                    if field.name == "information_stratum"
                    else value[field.name]
                )
                for field in dataclasses.fields(cls)
            }
        )


def _resolve_scene_paths(spec: SceneResetSpec, *, spec_path: Path) -> tuple[Path, Path]:
    def resolve(value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = spec_path.parent / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    return resolve(spec.bddl_file), resolve(spec.init_states_file)


@dataclasses.dataclass(frozen=True)
class VerifiedResetInventory:
    """Pre-proposal study population, frozen independently of Qwen outcomes."""

    inventory_path: Path
    inventory_sha256: str
    source_config_sha256: str
    rows: tuple[Mapping[str, Any], ...]
    document: Mapping[str, Any]


def _method_v1_population_contract(
    config_path: str | Path,
) -> tuple[str, dict[str, int], dict[str, dict[str, int]]]:
    """Return the source-config identity and its exact reset population."""

    config = load_method_v1_config(config_path)
    source_sha256 = canonical_sha256(config)
    collection = config.get("collection")
    if not isinstance(collection, Mapping):
        raise TypeError("Method-V1 source config collection section is missing")
    raw_counts = collection.get("reset_groups")
    split_names = ("train", "validation", "calibration", "test")
    if not isinstance(raw_counts, Mapping) or set(raw_counts) != set(split_names):
        raise ValueError("source config reset_groups must name all Method-V1 splits")
    counts: dict[str, int] = {}
    for split in split_names:
        value = raw_counts[split]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                "source config reset-group counts must be non-negative integers"
            )
        counts[split] = value
    return source_sha256, counts, method_v1_information_stratum_counts(config)


def _reset_inventory_row(
    spec: SceneResetSpec,
    *,
    spec_path: Path,
) -> dict[str, Any]:
    """Bind one public reset without copying evaluator-only semantic fields."""

    bddl_file, init_states_file = _resolve_scene_paths(spec, spec_path=spec_path)
    body = {
        "row_id": f"reset:{spec.decision_group_id}",
        "scene_spec_sha256": canonical_sha256(spec.to_dict()),
        "scene_id": spec.scene_id,
        "layout_id": spec.layout_id,
        "initial_state_group": spec.initial_state_group,
        "decision_group_id": spec.decision_group_id,
        "split_group_id": spec.split_group_id,
        "split": spec.split,
        "information_stratum": spec.information_stratum.value,
        "prompt": spec.prompt,
        "bddl_file_sha256": _file_sha256(bddl_file),
        "init_states_file_sha256": _file_sha256(init_states_file),
        "init_state_index": spec.init_state_index,
        "env_seed": spec.env_seed,
        "reset_state_sha256": spec.reset_state_sha256,
        "model_seeds": list(spec.model_seeds),
    }
    return {**body, "inventory_row_sha256": canonical_sha256(body)}


def _validate_reset_inventory_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_split_counts: Mapping[str, int],
    expected_stratum_counts: Mapping[str, Mapping[str, int]],
) -> None:
    expected_fields = {
        "row_id",
        "scene_spec_sha256",
        "scene_id",
        "layout_id",
        "initial_state_group",
        "decision_group_id",
        "split_group_id",
        "split",
        "information_stratum",
        "prompt",
        "bddl_file_sha256",
        "init_states_file_sha256",
        "init_state_index",
        "env_seed",
        "reset_state_sha256",
        "model_seeds",
        "inventory_row_sha256",
    }
    if not rows:
        raise ValueError("reset inventory must contain at least one row")
    split_names = ("train", "validation", "calibration", "test")
    seen: dict[str, set[Any]] = {
        "row_id": set(),
        "scene_spec_sha256": set(),
        "decision_group_id": set(),
        "initial_state_group": set(),
        "physical_reset": set(),
    }
    split_group_owner: dict[str, str] = {}
    scene_layout_owner: dict[tuple[str, str], str] = {}
    observed_split_counts = {split: 0 for split in split_names}
    observed_strata = {
        split: {item.value: 0 for item in InformationStratum} for split in split_names
    }
    for row in rows:
        if set(row) != expected_fields:
            raise ValueError("reset inventory row fields differ from schema")
        body = {
            key: value for key, value in row.items() if key != "inventory_row_sha256"
        }
        row_sha256 = _lower_sha256(
            row["inventory_row_sha256"], name="reset inventory row SHA-256"
        )
        if canonical_sha256(body) != row_sha256:
            raise ValueError("reset inventory row digest mismatch")
        for digest_name in (
            "scene_spec_sha256",
            "bddl_file_sha256",
            "init_states_file_sha256",
            "reset_state_sha256",
        ):
            _lower_sha256(row[digest_name], name=f"reset inventory {digest_name}")
        split = str(row["split"])
        if split not in observed_split_counts:
            raise ValueError("reset inventory row uses an invalid split")
        try:
            stratum = InformationStratum(str(row["information_stratum"]))
        except ValueError as error:
            raise ValueError("reset inventory row uses an invalid stratum") from error
        for name in (
            "row_id",
            "scene_id",
            "layout_id",
            "initial_state_group",
            "decision_group_id",
            "split_group_id",
            "prompt",
        ):
            if _clean_text(row[name], name=f"reset inventory {name}") != row[name]:
                raise ValueError(f"reset inventory {name} is not canonical")
        for name in ("init_state_index", "env_seed"):
            value = row[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"reset inventory {name} must be non-negative")
        seeds = row["model_seeds"]
        if (
            not isinstance(seeds, list)
            or len(seeds) != 2
            or len(set(seeds)) != 2
            or any(
                not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
                for seed in seeds
            )
        ):
            raise ValueError("reset inventory must retain two unique model seeds")
        identities = {
            "row_id": row["row_id"],
            "scene_spec_sha256": row["scene_spec_sha256"],
            "decision_group_id": row["decision_group_id"],
            "initial_state_group": row["initial_state_group"],
            "physical_reset": (
                row["bddl_file_sha256"],
                row["init_states_file_sha256"],
                row["init_state_index"],
                row["env_seed"],
                row["reset_state_sha256"],
            ),
        }
        for name, identity in identities.items():
            if identity in seen[name]:
                raise ValueError(f"reset inventory contains duplicate {name}")
            seen[name].add(identity)
        split_group = str(row["split_group_id"])
        previous = split_group_owner.setdefault(split_group, split)
        if previous != split:
            raise ValueError("one split_group_id crosses the split firewall")
        scene_layout = (str(row["scene_id"]), str(row["layout_id"]))
        previous = scene_layout_owner.setdefault(scene_layout, split)
        if previous != split:
            raise ValueError("one scene/layout pair crosses the split firewall")
        observed_split_counts[split] += 1
        observed_strata[split][stratum.value] += 1
    expected_counts = {
        split: int(expected_split_counts[split]) for split in split_names
    }
    if observed_split_counts != expected_counts:
        raise ValueError(
            "reset inventory population does not match source config: "
            f"expected={expected_counts}, observed={observed_split_counts}"
        )
    canonical_expected_strata = {
        split: {
            item.value: int(expected_stratum_counts[split][item.value])
            for item in InformationStratum
        }
        for split in split_names
    }
    if observed_strata != canonical_expected_strata:
        raise ValueError(
            "reset inventory information strata do not match source config: "
            f"expected={canonical_expected_strata}, observed={observed_strata}"
        )


def _build_reset_inventory_body(
    *,
    scene_spec_paths: Sequence[str | Path],
    config_path: str | Path,
    inventory_id: str,
) -> dict[str, Any]:
    paths = tuple(Path(path).expanduser().resolve() for path in scene_spec_paths)
    if not paths:
        raise ValueError("reset inventory requires at least one --scene-spec")
    if len(set(paths)) != len(paths):
        raise ValueError("reset inventory scene-spec paths must be unique")
    source_sha256, split_counts, stratum_counts = _method_v1_population_contract(
        config_path
    )
    rows = tuple(
        sorted(
            (
                _reset_inventory_row(
                    SceneResetSpec.from_mapping(_read_json(path)), spec_path=path
                )
                for path in paths
            ),
            key=lambda row: (str(row["decision_group_id"]), str(row["row_id"])),
        )
    )
    _validate_reset_inventory_rows(
        rows,
        expected_split_counts=split_counts,
        expected_stratum_counts=stratum_counts,
    )
    return {
        "schema_version": RESET_INVENTORY_SCHEMA,
        "inventory_id": _clean_text(inventory_id, name="reset inventory ID"),
        "pre_proposal_freeze": True,
        "source_config_sha256": source_sha256,
        "expected_split_counts": split_counts,
        "expected_information_stratum_counts": stratum_counts,
        "row_count": len(rows),
        "rows": list(rows),
    }


def freeze_reset_inventory(
    *,
    scene_spec_paths: Sequence[str | Path],
    config_path: str | Path,
    inventory_id: str,
    output: str | Path,
) -> dict[str, Any]:
    """Freeze the complete reset population before the first Qwen proposal."""

    body = _build_reset_inventory_body(
        scene_spec_paths=scene_spec_paths,
        config_path=config_path,
        inventory_id=inventory_id,
    )
    document = {**body, "reset_inventory_sha256": canonical_sha256(body)}
    destination = Path(output).expanduser().resolve()
    _write_json_once(destination, document)
    return dict(
        load_verified_reset_inventory(
            destination,
            config_path=config_path,
            scene_spec_paths=scene_spec_paths,
        ).document
    )


def load_verified_reset_inventory(
    path: str | Path,
    *,
    config_path: str | Path,
    scene_spec_paths: Sequence[str | Path] | None = None,
) -> VerifiedResetInventory:
    """Re-open an inventory and, when supplied, re-hash every source scene spec."""

    source = Path(path).expanduser().resolve()
    document = _read_json(source)
    expected_keys = {
        "schema_version",
        "inventory_id",
        "pre_proposal_freeze",
        "source_config_sha256",
        "expected_split_counts",
        "expected_information_stratum_counts",
        "row_count",
        "rows",
        "reset_inventory_sha256",
    }
    if set(document) != expected_keys:
        raise ValueError("reset inventory fields differ from schema")
    body = {
        key: value for key, value in document.items() if key != "reset_inventory_sha256"
    }
    inventory_sha256 = _lower_sha256(
        document["reset_inventory_sha256"], name="reset inventory SHA-256"
    )
    if (
        document["schema_version"] != RESET_INVENTORY_SCHEMA
        or document["pre_proposal_freeze"] is not True
        or canonical_sha256(body) != inventory_sha256
    ):
        raise ValueError("reset inventory digest or pre-proposal identity is invalid")
    source_sha256, split_counts, stratum_counts = _method_v1_population_contract(
        config_path
    )
    if document["source_config_sha256"] != source_sha256:
        raise ValueError("reset inventory source config identity changed")
    rows = document["rows"]
    if not isinstance(rows, list) or document["row_count"] != len(rows):
        raise ValueError("reset inventory row count is inconsistent")
    if (
        document["expected_split_counts"] != split_counts
        or document["expected_information_stratum_counts"] != stratum_counts
    ):
        raise ValueError("reset inventory population contract changed")
    _validate_reset_inventory_rows(
        rows,
        expected_split_counts=split_counts,
        expected_stratum_counts=stratum_counts,
    )
    if scene_spec_paths is not None:
        rebuilt = _build_reset_inventory_body(
            scene_spec_paths=scene_spec_paths,
            config_path=config_path,
            inventory_id=str(document["inventory_id"]),
        )
        if body != rebuilt:
            raise ValueError("reset inventory differs from re-opened scene specs")
    return VerifiedResetInventory(
        inventory_path=source,
        inventory_sha256=inventory_sha256,
        source_config_sha256=source_sha256,
        rows=tuple(dict(row) for row in rows),
        document=document,
    )


def _inventory_binding_for_scene_spec(
    *,
    inventory_path: str | Path,
    config_path: str | Path,
    spec: SceneResetSpec,
    spec_path: Path,
) -> dict[str, str]:
    inventory = load_verified_reset_inventory(inventory_path, config_path=config_path)
    expected_row = _reset_inventory_row(spec, spec_path=spec_path)
    matches = [
        row
        for row in inventory.rows
        if row["inventory_row_sha256"] == expected_row["inventory_row_sha256"]
    ]
    if len(matches) != 1 or dict(matches[0]) != expected_row:
        raise ValueError("scene reset is not an exact member of reset inventory")
    return {
        "reset_inventory_sha256": inventory.inventory_sha256,
        "inventory_row_sha256": str(expected_row["inventory_row_sha256"]),
    }


def _load_executor_identity(path: str | Path) -> MolmoAct2ServerIdentity:
    value = _read_json(path)
    payload = value.get("executor", value)
    if not isinstance(payload, Mapping):
        raise TypeError("executor identity must be an object")
    return MolmoAct2ServerIdentity.from_mapping(payload)


def render_public_reset(*, scene_spec: Path, output_dir: Path) -> dict[str, Any]:
    """Materialize one exact public decision context without outcomes."""

    spec = SceneResetSpec.from_mapping(_read_json(scene_spec))
    bddl, init_states = _resolve_scene_paths(spec, spec_path=scene_spec)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite prepared reset {output_dir}")
    output_dir.mkdir(parents=True)
    store = RGBFrameStore(output_dir / "public_frames")
    with LiberoE1Environment(
        bddl_file=bddl,
        init_states_file=init_states,
        init_state_index=spec.init_state_index,
        env_seed=spec.env_seed,
        expected_reset_state_sha256=spec.reset_state_sha256,
        image_size=spec.image_size,
        settle_steps=spec.settle_steps,
        control_mode=spec.control_mode,
    ) as environment:
        context = public_context_from_observation(
            prompt=spec.prompt,
            observation=environment.public_observation,
            frame_store=store,
            frame_namespace=f"{spec.decision_group_id}:initial",
            frame_index=0,
        )
    body = {
        "schema_version": PREPARED_SCHEMA,
        "scene_spec_sha256": canonical_sha256(spec.to_dict()),
        "scene_id": spec.scene_id,
        "layout_id": spec.layout_id,
        "initial_state_group": spec.initial_state_group,
        "decision_group_id": spec.decision_group_id,
        "split_group_id": spec.split_group_id,
        "split": spec.split,
        "reset_state_sha256": spec.reset_state_sha256,
        "bddl_file_sha256": _file_sha256(bddl),
        "init_states_file_sha256": _file_sha256(init_states),
        "context": context.to_dict(),
    }
    prepared = {**body, "prepared_sha256": canonical_sha256(body)}
    _write_json_once(output_dir / "prepared_public_context.json", prepared)
    return prepared


def _prepared_context(
    *, scene_spec: SceneResetSpec, prepared: Mapping[str, Any]
) -> PolicyContext:
    body = {key: value for key, value in prepared.items() if key != "prepared_sha256"}
    if (
        prepared.get("schema_version") != PREPARED_SCHEMA
        or canonical_sha256(body) != prepared.get("prepared_sha256")
        or prepared.get("scene_spec_sha256") != canonical_sha256(scene_spec.to_dict())
    ):
        raise ValueError("prepared public reset identity is inconsistent")
    context = PolicyContext.from_mapping(prepared["context"])
    if (
        context.prompt != scene_spec.prompt
        or prepared["reset_state_sha256"] != scene_spec.reset_state_sha256
    ):
        raise ValueError("prepared public reset changed scene prompt or state")
    return context


def freeze_decision_group(
    *,
    scene_spec_path: Path,
    prepared_dir: Path,
    config_path: Path,
    executor_identity_path: Path,
    output_dir: Path,
    qwen_device_map: str | None,
    local_files_only: bool,
    reset_inventory_path: Path | None = None,
) -> dict[str, Any]:
    """Run real Qwen proposal/features once and freeze the branch schedule."""

    spec = SceneResetSpec.from_mapping(_read_json(scene_spec_path))
    prepared = _read_json(prepared_dir / "prepared_public_context.json")
    context = _prepared_context(scene_spec=spec, prepared=prepared)
    config = load_method_v1_config(config_path)
    inventory_binding = (
        None
        if reset_inventory_path is None
        else _inventory_binding_for_scene_spec(
            inventory_path=reset_inventory_path,
            config_path=config_path,
            spec=spec,
            spec_path=scene_spec_path,
        )
    )
    executor_identity = _load_executor_identity(executor_identity_path)
    resolved = resolve_method_v1_identity(config, executor_identity=executor_identity)
    validate_resolved_method_v1_identity(resolved, source_config=config)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite decision freeze {output_dir}")
    output_dir.mkdir(parents=True)

    identity = QwenProviderIdentity(transformers_version=QWEN25VL_TRANSFORMERS_VERSION)
    runtime = Qwen25VLRuntime(
        identity=identity,
        device_map=qwen_device_map,
        local_files_only=local_files_only,
    )
    frame_store = RGBFrameStore(prepared_dir / "public_frames")
    proposer = Qwen25VLProposer(runtime=runtime, frame_store=frame_store)
    try:
        proposal = proposer.propose(context)
    except ProposalFailure as error:
        failure_body = {
            "schema_version": (
                "method-v1-proposal-failure-v1"
                if inventory_binding is None
                else "method-v1-proposal-failure-v2"
            ),
            "status": "PROPOSAL_FAILURE",
            "scene_spec_sha256": canonical_sha256(spec.to_dict()),
            "context_fingerprint": context.fingerprint(),
            "proposer_id": proposer.proposer_id,
            "failure": error.audit_dict(),
            "runtime_source_sha256": _method_v1_runtime_source_sha256(),
            **({} if inventory_binding is None else inventory_binding),
        }
        _write_json_once(
            output_dir / "proposal_failure.json",
            {
                **failure_body,
                "proposal_failure_sha256": canonical_sha256(failure_body),
            },
        )
        raise
    candidates = proposal.candidates
    cache = QwenFeatureCache(output_dir / "feature_cache")
    provider = Qwen25VLTokenProvider(
        runtime=runtime, frame_store=frame_store, cache=cache
    )
    field = provider.encode(context, candidates)
    logical_cache_key = candidate_feature_cache_key(
        provider_id=provider.provider_id,
        context=context,
        candidates=candidates,
    )
    cache_artifact = cache.artifact_identity(
        logical_cache_key,
        context=context,
        candidates=candidates,
        provider_id=provider.provider_id,
    )
    continuation = FixedContinuationIdentity(**resolved["continuation"]["identity"])
    contract = continuation.outcome_contract()
    rejections = tuple(
        ManifestProposalRejection(
            proposal_index=item.index,
            reason_code=item.code,
            proposal_sha256=canonical_sha256(item.to_dict()),
        )
        for item in proposal.rejections
        if item.index is not None
    )
    request_sha256 = canonical_sha256(
        {
            "context_fingerprint": context.fingerprint(),
            "proposer_id": proposer.proposer_id,
        }
    )
    overlay_rows: list[dict[str, Any]] = []
    overlay_dir = output_dir / "grounding_overlays"
    patch_boxes = field.public_context.patch_xyxy[0].detach().cpu().tolist()
    for candidate_index, candidate in enumerate(candidates):
        if candidate.grounding is None:  # pragma: no cover - manifest validates.
            raise RuntimeError("Method-V1 candidate lost its grounding")
        matching_frames = [
            frame
            for frame in context.frames
            if frame.camera == candidate.grounding.camera
            and frame.frame_id == candidate.grounding.frame_id
            and frame.frame_index == candidate.grounding.frame_index
            and frame.image_sha256 == candidate.grounding.image_sha256
        ]
        if len(matching_frames) != 1:
            raise ValueError("candidate grounding has no unique public source frame")
        overlay_path = overlay_dir / (
            f"{candidate_index:02d}-{candidate.fingerprint()[:16]}.png"
        )
        save_grounding_support_overlay(
            frame_store.resolve(matching_frames[0]),
            patch_boxes=patch_boxes,
            support_mask=field.grounding_support[0, candidate_index].detach().cpu(),
            candidate_box=candidate.grounding.box_xyxy,
            output_path=overlay_path,
        )
        overlay_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "candidate_fingerprint": candidate.fingerprint(),
                "path": str(overlay_path.relative_to(output_dir)),
                "file_sha256": _file_sha256(overlay_path),
            }
        )
    proposal_audit_body = {
        "schema_version": "method-v1-qwen-proposal-audit-v1",
        "context_fingerprint": context.fingerprint(),
        "proposer_id": proposer.proposer_id,
        "proposal_request_sha256": request_sha256,
        "proposal": proposal.audit_dict(),
        "grounding_overlays": overlay_rows,
    }
    proposal_audit = {
        **proposal_audit_body,
        "proposal_audit_sha256": canonical_sha256(proposal_audit_body),
    }
    _write_json_once(output_dir / "proposal_audit.json", proposal_audit)
    manifest = DecisionGroupManifest(
        manifest_id=f"{spec.decision_group_id}:method-v1",
        experiment_id=config["experiment_id"],
        initial_state_group=spec.initial_state_group,
        decision_group_id=spec.decision_group_id,
        split_group_id=spec.split_group_id,
        split=spec.split,
        information_stratum=spec.information_stratum,
        scene_id=spec.scene_id,
        layout_id=spec.layout_id,
        reset_state_sha256=spec.reset_state_sha256,
        context=context,
        candidates=candidates,
        outcome_contract=contract,
        proposal_provider_id=proposer.proposer_id,
        proposal_request_sha256=request_sha256,
        token_provider_id=provider.provider_id,
        token_cache_sha256=cache_artifact.artifact_sha256,
        configuration_sha256=resolved["resolved_identity_sha256"],
        model_seeds=spec.model_seeds,
        proposal_rejections=rejections,
    )
    schedule = build_branch_schedule(
        (manifest,), schedule_id=f"{spec.decision_group_id}:branches"
    )
    primitive_set = {candidate.primitive for candidate in candidates}
    proposal_population_status = (
        "VALID_CHOICE_SET"
        if {Primitive.DIRECT, Primitive.OPEN}.issubset(primitive_set)
        else "VALID_SINGLE_PRIMITIVE_SET"
    )
    _write_json_once(output_dir / "resolved_method_v1.json", resolved)
    _write_json_once(output_dir / "decision_manifest.json", manifest.to_dict())
    schedule_body = {
        "schema_version": SCHEDULE_FILE_SCHEMA,
        "schedule_id": schedule[0].schedule_id,
        "manifest_sha256": manifest.fingerprint(),
        "entries": [item.to_dict() for item in schedule],
    }
    schedule_file = {
        **schedule_body,
        "schedule_sha256": canonical_sha256(schedule_body),
    }
    _write_json_once(output_dir / "branch_schedule.json", schedule_file)
    freeze_body = {
        "schema_version": (
            FREEZE_RECEIPT_SCHEMA
            if inventory_binding is None
            else INVENTORY_FREEZE_RECEIPT_SCHEMA
        ),
        "scene_spec_sha256": canonical_sha256(spec.to_dict()),
        "prepared_sha256": prepared["prepared_sha256"],
        "bddl_file_sha256": prepared["bddl_file_sha256"],
        "init_states_file_sha256": prepared["init_states_file_sha256"],
        "manifest_sha256": manifest.fingerprint(),
        "schedule_sha256": schedule_file["schedule_sha256"],
        "resolved_identity_sha256": resolved["resolved_identity_sha256"],
        "runtime_source_sha256": _method_v1_runtime_source_sha256(),
        "proposal_audit_sha256": proposal_audit["proposal_audit_sha256"],
        "token_cache_logical_key": cache_artifact.logical_key,
        "token_cache_artifact_sha256": cache_artifact.artifact_sha256,
        "token_cache_tensor_artifact_sha256": (cache_artifact.tensor_artifact_sha256),
        "token_cache_sidecar_artifact_sha256": (cache_artifact.sidecar_artifact_sha256),
        **(
            {}
            if inventory_binding is None
            else {
                **inventory_binding,
                "proposal_population_status": proposal_population_status,
            }
        ),
    }
    freeze_receipt = {
        **freeze_body,
        "freeze_receipt_sha256": canonical_sha256(freeze_body),
    }
    _write_json_once(output_dir / "freeze_receipt.json", freeze_receipt)
    return {
        "manifest_sha256": manifest.fingerprint(),
        "schedule_sha256": schedule_file["schedule_sha256"],
        "freeze_receipt_sha256": freeze_receipt["freeze_receipt_sha256"],
        "candidate_count": len(candidates),
        "scheduled_branches": len(schedule),
        "token_cache_logical_key": cache_artifact.logical_key,
        "token_cache_artifact_sha256": cache_artifact.artifact_sha256,
        **(
            {}
            if inventory_binding is None
            else {
                **inventory_binding,
                "proposal_population_status": proposal_population_status,
            }
        ),
    }


def _load_schedule(path: Path) -> tuple[BranchScheduleEntry, ...]:
    value = _read_json(path)
    if set(value) != {
        "schema_version",
        "schedule_id",
        "manifest_sha256",
        "entries",
        "schedule_sha256",
    }:
        raise ValueError("branch schedule fields differ from the frozen schema")
    body = {key: child for key, child in value.items() if key != "schedule_sha256"}
    if value.get("schema_version") != SCHEDULE_FILE_SCHEMA or canonical_sha256(
        body
    ) != value.get("schedule_sha256"):
        raise ValueError("branch schedule digest mismatch")
    rows = tuple(BranchScheduleEntry.from_mapping(item) for item in value["entries"])
    if not rows or value["schedule_id"] != rows[0].schedule_id:
        raise ValueError("branch schedule identity mismatch")
    if value["manifest_sha256"] != rows[0].manifest_sha256:
        raise ValueError("branch schedule changed the manifest digest")
    if any(row.schedule_id != rows[0].schedule_id for row in rows):
        raise ValueError("branch schedule mixes schedule identities")
    return rows


def _validate_proposal_audit(
    *,
    freeze_dir: Path,
    manifest: DecisionGroupManifest,
) -> str:
    value = _read_json(freeze_dir / "proposal_audit.json")
    expected_keys = {
        "schema_version",
        "context_fingerprint",
        "proposer_id",
        "proposal_request_sha256",
        "proposal",
        "grounding_overlays",
        "proposal_audit_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError("Qwen proposal audit fields differ from schema")
    body = {
        key: child for key, child in value.items() if key != "proposal_audit_sha256"
    }
    digest = canonical_sha256(body)
    if (
        value["schema_version"] != "method-v1-qwen-proposal-audit-v1"
        or value["proposal_audit_sha256"] != digest
    ):
        raise ValueError("Qwen proposal audit digest mismatch")
    expected_bindings = {
        "context_fingerprint": manifest.context.fingerprint(),
        "proposer_id": manifest.proposal_provider_id,
        "proposal_request_sha256": manifest.proposal_request_sha256,
    }
    for name, expected in expected_bindings.items():
        if value[name] != expected:
            raise ValueError(f"Qwen proposal audit changed {name}")
    proposal = value["proposal"]
    if not isinstance(proposal, Mapping):
        raise TypeError("Qwen proposal audit payload must be a mapping")
    processed_size = proposal.get("processed_image_size")
    if (
        not isinstance(processed_size, list)
        or len(processed_size) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in processed_size
        )
    ):
        raise ValueError("Qwen proposal audit lost processed image size")
    latest_index = manifest.context.latest_frame_index("agentview")
    frames = [
        frame
        for frame in manifest.context.frames
        if frame.camera == "agentview" and frame.frame_index == latest_index
    ]
    if len(frames) != 1:
        raise ValueError("manifest has no unique current proposal frame")
    reparsed = parse_qwen_proposal_json(
        str(proposal.get("raw_response", "")),
        context=manifest.context,
        frame_id=frames[0].frame_id,
        processed_width=int(processed_size[0]),
        processed_height=int(processed_size[1]),
    )
    if reparsed.audit_dict() != dict(proposal):
        raise ValueError("Qwen proposal audit does not reproduce from raw output")
    if tuple(item.fingerprint() for item in reparsed.candidates) != tuple(
        item.fingerprint() for item in manifest.candidates
    ):
        raise ValueError("Qwen proposal audit candidate set differs from manifest")
    overlays = value["grounding_overlays"]
    if not isinstance(overlays, list) or len(overlays) != len(manifest.candidates):
        raise ValueError("grounding overlay coverage differs from candidate set")
    for candidate, overlay in zip(manifest.candidates, overlays, strict=True):
        if not isinstance(overlay, Mapping) or set(overlay) != {
            "candidate_id",
            "candidate_fingerprint",
            "path",
            "file_sha256",
        }:
            raise ValueError("grounding overlay record fields differ from schema")
        if (
            overlay["candidate_id"] != candidate.candidate_id
            or overlay["candidate_fingerprint"] != candidate.fingerprint()
        ):
            raise ValueError("grounding overlay changed candidate identity")
        relative = Path(str(overlay["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("grounding overlay path escapes the decision freeze")
        overlay_path = (freeze_dir / relative).resolve()
        if not overlay_path.is_relative_to(freeze_dir.resolve()):
            raise ValueError("grounding overlay path escapes the decision freeze")
        if _file_sha256(overlay_path) != overlay["file_sha256"]:
            raise ValueError("grounding overlay file digest mismatch")
    return digest


def _validate_manifest_against_resolved(
    manifest: DecisionGroupManifest, resolved: Mapping[str, Any]
) -> None:
    """Bind a decision manifest to the one resolved Method-V1 configuration."""

    qwen_identity = QwenProviderIdentity(
        transformers_version=QWEN25VL_TRANSFORMERS_VERSION
    )
    expected_token_provider = (
        f"{qwen_identity.provider_id}+public-state=proprio8+budget=1"
    )
    expected_proposer = qwen_proposer_id(
        provider_id=qwen_identity.provider_id,
        proposal_camera="agentview",
        enabled_primitives=(Primitive.DIRECT, Primitive.OPEN),
        max_candidates=6,
        max_per_primitive=3,
    )
    expected = {
        "experiment_id": resolved["experiment_id"],
        "configuration_sha256": resolved["resolved_identity_sha256"],
        "proposal_provider_id": expected_proposer,
        "token_provider_id": expected_token_provider,
        "outcome_contract_sha256": resolved["outcome_contract_sha256"],
    }
    observed = {
        "experiment_id": manifest.experiment_id,
        "configuration_sha256": manifest.configuration_sha256,
        "proposal_provider_id": manifest.proposal_provider_id,
        "token_provider_id": manifest.token_provider_id,
        "outcome_contract_sha256": manifest.outcome_contract.fingerprint(),
    }
    if observed != expected:
        raise ValueError(
            "decision manifest differs from resolved Method-V1 identity: "
            f"expected={expected}, observed={observed}"
        )


def _freeze_receipt_expected_keys(schema_version: Any) -> set[str]:
    keys = {
        "schema_version",
        "scene_spec_sha256",
        "prepared_sha256",
        "bddl_file_sha256",
        "init_states_file_sha256",
        "manifest_sha256",
        "schedule_sha256",
        "resolved_identity_sha256",
        "runtime_source_sha256",
        "proposal_audit_sha256",
        "token_cache_logical_key",
        "token_cache_artifact_sha256",
        "token_cache_tensor_artifact_sha256",
        "token_cache_sidecar_artifact_sha256",
        "freeze_receipt_sha256",
    }
    if schema_version == INVENTORY_FREEZE_RECEIPT_SCHEMA:
        keys.update(
            {
                "reset_inventory_sha256",
                "inventory_row_sha256",
                "proposal_population_status",
            }
        )
    elif schema_version != FREEZE_RECEIPT_SCHEMA:
        raise ValueError("unsupported decision freeze receipt schema")
    return keys


def _validate_inventory_receipt_fields(
    receipt: Mapping[str, Any],
    *,
    manifest: DecisionGroupManifest,
) -> tuple[str | None, str | None, str | None]:
    if receipt["schema_version"] == FREEZE_RECEIPT_SCHEMA:
        return None, None, None
    inventory_sha256 = _lower_sha256(
        receipt["reset_inventory_sha256"], name="reset inventory SHA-256"
    )
    row_sha256 = _lower_sha256(
        receipt["inventory_row_sha256"], name="inventory row SHA-256"
    )
    primitive_set = {candidate.primitive for candidate in manifest.candidates}
    expected_status = (
        "VALID_CHOICE_SET"
        if {Primitive.DIRECT, Primitive.OPEN}.issubset(primitive_set)
        else "VALID_SINGLE_PRIMITIVE_SET"
    )
    if receipt["proposal_population_status"] != expected_status:
        raise ValueError("decision freeze proposal-population status changed")
    return inventory_sha256, row_sha256, expected_status


def _validate_freeze_receipt(
    *,
    freeze_dir: Path,
    spec: SceneResetSpec,
    spec_path: Path,
    manifest: DecisionGroupManifest,
) -> dict[str, Any]:
    receipt = _read_json(freeze_dir / "freeze_receipt.json")
    expected_keys = _freeze_receipt_expected_keys(receipt.get("schema_version"))
    if set(receipt) != expected_keys:
        raise ValueError("decision freeze receipt fields differ from schema")
    body = {
        key: value for key, value in receipt.items() if key != "freeze_receipt_sha256"
    }
    if canonical_sha256(body) != receipt["freeze_receipt_sha256"]:
        raise ValueError("decision freeze receipt digest mismatch")
    _validate_inventory_receipt_fields(receipt, manifest=manifest)
    schedule_file = _read_json(freeze_dir / "branch_schedule.json")
    resolved = validate_resolved_method_v1_identity(
        _read_json(freeze_dir / "resolved_method_v1.json")
    )
    _validate_manifest_against_resolved(manifest, resolved)
    bddl_file, init_states_file = _resolve_scene_paths(spec, spec_path=spec_path)
    observed = {
        "scene_spec_sha256": canonical_sha256(spec.to_dict()),
        "bddl_file_sha256": _file_sha256(bddl_file),
        "init_states_file_sha256": _file_sha256(init_states_file),
        "manifest_sha256": manifest.fingerprint(),
        "schedule_sha256": schedule_file.get("schedule_sha256"),
        "resolved_identity_sha256": resolved.get("resolved_identity_sha256"),
    }
    observed.update(
        {
            "runtime_source_sha256": _method_v1_runtime_source_sha256(),
            "proposal_audit_sha256": _validate_proposal_audit(
                freeze_dir=freeze_dir,
                manifest=manifest,
            ),
        }
    )
    cache_artifact = QwenFeatureCache(freeze_dir / "feature_cache").artifact_identity(
        manifest.token_cache_sha256,
        context=manifest.context,
        candidates=manifest.candidates,
        provider_id=manifest.token_provider_id,
    )
    observed.update(
        {
            "token_cache_logical_key": cache_artifact.logical_key,
            "token_cache_artifact_sha256": cache_artifact.artifact_sha256,
            "token_cache_tensor_artifact_sha256": (
                cache_artifact.tensor_artifact_sha256
            ),
            "token_cache_sidecar_artifact_sha256": (
                cache_artifact.sidecar_artifact_sha256
            ),
        }
    )
    for name, value in observed.items():
        if receipt[name] != value:
            raise ValueError(f"decision freeze changed {name}")
    return receipt


@dataclasses.dataclass(frozen=True)
class VerifiedDecisionFreeze:
    """Collector-owned decision freeze admitted from its sealed source files."""

    freeze_dir: Path
    manifest: DecisionGroupManifest
    schedule: tuple[BranchScheduleEntry, ...]
    freeze_receipt_sha256: str
    reset_inventory_sha256: str | None = None
    inventory_row_sha256: str | None = None
    proposal_population_status: str | None = None


@dataclasses.dataclass(frozen=True)
class VerifiedCollectionPlan:
    """Exact pre-outcome study population admitted from all decision freezes."""

    plan_path: Path
    collection_plan_sha256: str
    scorer_verifier_auth_key_id: str
    freezes: tuple[VerifiedDecisionFreeze, ...]
    document: Mapping[str, Any]


def load_verified_decision_freeze(
    path: str | Path,
) -> VerifiedDecisionFreeze:
    """Re-open and validate one collector decision freeze for data admission.

    Training and evaluation do not have to receive the original scene-spec path,
    but they must re-open every collector-owned artifact that is available in the
    freeze directory.  This verifies the receipt, manifest, schedule, resolved
    Method-V1 identity, proposal audit, physical token cache, and source identity.
    Scene files were checked before the single-use execution claim was created;
    every admitted attempt is independently re-opened by
    :func:`load_verified_collection_attempt` and its execution trace validator.
    """

    freeze_dir = Path(path).expanduser().resolve()
    manifest = DecisionGroupManifest.from_mapping(
        _read_json(freeze_dir / "decision_manifest.json")
    )
    schedule = _load_schedule(freeze_dir / "branch_schedule.json")
    validate_branch_schedule((manifest,), schedule)

    receipt = _read_json(freeze_dir / "freeze_receipt.json")
    expected_keys = _freeze_receipt_expected_keys(receipt.get("schema_version"))
    if set(receipt) != expected_keys:
        raise ValueError("decision freeze receipt fields differ from schema")
    body = {
        key: value for key, value in receipt.items() if key != "freeze_receipt_sha256"
    }
    if canonical_sha256(body) != receipt["freeze_receipt_sha256"]:
        raise ValueError("decision freeze receipt digest mismatch")
    for name in (
        "scene_spec_sha256",
        "prepared_sha256",
        "bddl_file_sha256",
        "init_states_file_sha256",
    ):
        value = receipt[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"decision freeze receipt has invalid {name}")

    schedule_file = _read_json(freeze_dir / "branch_schedule.json")
    resolved = validate_resolved_method_v1_identity(
        _read_json(freeze_dir / "resolved_method_v1.json")
    )
    _validate_manifest_against_resolved(manifest, resolved)
    inventory_sha256, row_sha256, population_status = (
        _validate_inventory_receipt_fields(receipt, manifest=manifest)
    )
    cache_artifact = QwenFeatureCache(freeze_dir / "feature_cache").artifact_identity(
        manifest.token_cache_sha256,
        context=manifest.context,
        candidates=manifest.candidates,
        provider_id=manifest.token_provider_id,
    )
    observed = {
        "manifest_sha256": manifest.fingerprint(),
        "schedule_sha256": schedule_file["schedule_sha256"],
        "resolved_identity_sha256": resolved["resolved_identity_sha256"],
        "runtime_source_sha256": _method_v1_runtime_source_sha256(),
        "proposal_audit_sha256": _validate_proposal_audit(
            freeze_dir=freeze_dir,
            manifest=manifest,
        ),
        "token_cache_logical_key": cache_artifact.logical_key,
        "token_cache_artifact_sha256": cache_artifact.artifact_sha256,
        "token_cache_tensor_artifact_sha256": cache_artifact.tensor_artifact_sha256,
        "token_cache_sidecar_artifact_sha256": (cache_artifact.sidecar_artifact_sha256),
    }
    for name, value in observed.items():
        if receipt[name] != value:
            raise ValueError(f"decision freeze changed {name}")
    return VerifiedDecisionFreeze(
        freeze_dir=freeze_dir,
        manifest=manifest,
        schedule=schedule,
        freeze_receipt_sha256=str(receipt["freeze_receipt_sha256"]),
        reset_inventory_sha256=inventory_sha256,
        inventory_row_sha256=row_sha256,
        proposal_population_status=population_status,
    )


def _lower_sha256(value: Any, *, name: str) -> str:
    result = str(value)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def scorer_auth_key_id(path: str | Path) -> str:
    """Return the public identity of a required, secret verifier key file."""

    source = Path(path).expanduser().resolve()
    try:
        secret = source.read_bytes()
    except OSError as error:
        raise ValueError(
            f"cannot read scorer verifier authentication key {source}"
        ) from error
    if len(secret) < 32:
        raise ValueError(
            "scorer verifier authentication key must contain at least 32 bytes"
        )
    return hashlib.sha256(secret).hexdigest()


def _collection_config_identity(
    freezes: Sequence[VerifiedDecisionFreeze],
    *,
    source_config_sha256: str,
) -> dict[str, str]:
    manifests = tuple(item.manifest for item in freezes)
    if not manifests:
        return {
            "identity_scope": "SOURCE_CONFIG_ONLY_NO_VALID_PROPOSAL",
            "source_config_sha256": _lower_sha256(
                source_config_sha256, name="source Method-V1 config SHA-256"
            ),
        }
    fields = {
        "experiment_id": {item.experiment_id for item in manifests},
        "configuration_sha256": {item.configuration_sha256 for item in manifests},
        "proposal_provider_id": {item.proposal_provider_id for item in manifests},
        "token_provider_id": {item.token_provider_id for item in manifests},
        "outcome_contract_sha256": {
            item.outcome_contract.fingerprint() for item in manifests
        },
    }
    mixed = sorted(name for name, values in fields.items() if len(values) != 1)
    if mixed:
        raise ValueError(
            f"collection plan cannot mix Method-V1 configuration identities: {mixed}"
        )
    return {
        **{name: next(iter(values)) for name, values in fields.items()},
        "source_config_sha256": _lower_sha256(
            source_config_sha256, name="source Method-V1 config SHA-256"
        ),
    }


def _collection_source_config_contract(
    config_path: str | Path,
    freezes: Sequence[VerifiedDecisionFreeze],
) -> tuple[str, dict[str, int], dict[str, dict[str, int]]]:
    """Re-open the source config and bind every freeze to its collection counts."""

    checked = load_method_v1_config(config_path)
    source_sha256 = canonical_sha256(checked)
    collection = checked.get("collection")
    if not isinstance(collection, Mapping):
        raise TypeError("Method-V1 source config collection section is missing")
    raw_counts = collection.get("reset_groups")
    split_names = ("train", "validation", "calibration", "test")
    if not isinstance(raw_counts, Mapping) or set(raw_counts) != set(split_names):
        raise ValueError("source config reset_groups must name all Method-V1 splits")
    counts: dict[str, int] = {}
    for split in split_names:
        value = raw_counts[split]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                "source config reset-group counts must be non-negative integers"
            )
        counts[split] = value
    for item in freezes:
        resolved = _read_json(item.freeze_dir / "resolved_method_v1.json")
        if resolved.get("source_config_sha256") != source_sha256:
            raise ValueError(
                "decision freeze source config differs from collection-plan config"
            )
    return source_sha256, counts, method_v1_information_stratum_counts(checked)


def _collection_plan_group(item: VerifiedDecisionFreeze) -> dict[str, Any]:
    manifest = item.manifest
    rows = item.schedule
    schedule_ids = tuple(dict.fromkeys(row.schedule_id for row in rows))
    return {
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest.fingerprint(),
        "freeze_receipt_sha256": item.freeze_receipt_sha256,
        "schedule_ids": list(schedule_ids),
        "schedule_sha256": canonical_sha256([row.to_dict() for row in rows]),
        "decision_group_id": manifest.decision_group_id,
        "initial_state_group": manifest.initial_state_group,
        "split_group_id": manifest.split_group_id,
        # Method V1 uses split_group_id as the family identity that keeps all
        # unobserved/hidden variants on one side of the split firewall.
        "hidden_variant_family_id": manifest.split_group_id,
        "information_stratum": manifest.information_stratum.value,
        "split": manifest.split,
        "scene_id": manifest.scene_id,
        "layout_id": manifest.layout_id,
        "reset_state_sha256": manifest.reset_state_sha256,
        "model_seeds": list(manifest.model_seeds),
        "candidate_count": len(manifest.candidates),
        "entry_ids": [row.entry_id for row in rows],
        "entry_count": len(rows),
    }


def _load_verified_proposal_failure(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() / "proposal_failure.json"
    value = _read_json(source)
    schema = value.get("schema_version")
    base_keys = {
        "schema_version",
        "status",
        "scene_spec_sha256",
        "context_fingerprint",
        "proposer_id",
        "failure",
        "runtime_source_sha256",
        "proposal_failure_sha256",
    }
    if schema == "method-v1-proposal-failure-v2":
        base_keys.update({"reset_inventory_sha256", "inventory_row_sha256"})
    elif schema != "method-v1-proposal-failure-v1":
        raise ValueError("unsupported proposal-failure schema")
    if set(value) != base_keys or value["status"] != "PROPOSAL_FAILURE":
        raise ValueError("proposal-failure fields or status differ from schema")
    body = {
        key: child for key, child in value.items() if key != "proposal_failure_sha256"
    }
    if canonical_sha256(body) != value["proposal_failure_sha256"]:
        raise ValueError("proposal-failure digest mismatch")
    for name in (
        "scene_spec_sha256",
        "runtime_source_sha256",
        "proposal_failure_sha256",
    ):
        _lower_sha256(value[name], name=f"proposal failure {name}")
    if schema == "method-v1-proposal-failure-v2":
        _lower_sha256(
            value["reset_inventory_sha256"], name="proposal failure inventory SHA-256"
        )
        _lower_sha256(
            value["inventory_row_sha256"], name="proposal failure row SHA-256"
        )
    return value


def _proposal_population_from_terminals(
    *,
    inventory: VerifiedResetInventory,
    freezes: Sequence[VerifiedDecisionFreeze],
    proposal_failure_dirs: Sequence[str | Path],
) -> dict[str, Any]:
    """Require exactly one immutable proposal terminal for every frozen reset."""

    inventory_by_row = {str(row["inventory_row_sha256"]): row for row in inventory.rows}
    terminals: dict[str, dict[str, Any]] = {}

    def admit(row_sha256: str, terminal: dict[str, Any]) -> None:
        if row_sha256 not in inventory_by_row:
            raise ValueError("proposal terminal is not a reset-inventory member")
        if row_sha256 in terminals:
            raise ValueError("reset inventory row has duplicate/replacement terminals")
        terminals[row_sha256] = terminal

    for freeze in freezes:
        if (
            freeze.reset_inventory_sha256 != inventory.inventory_sha256
            or freeze.inventory_row_sha256 is None
            or freeze.proposal_population_status
            not in {"VALID_CHOICE_SET", "VALID_SINGLE_PRIMITIVE_SET"}
        ):
            raise ValueError("decision freeze is not bound to this reset inventory")
        row = inventory_by_row.get(freeze.inventory_row_sha256)
        manifest = freeze.manifest
        expected_public = {
            "scene_id": manifest.scene_id,
            "layout_id": manifest.layout_id,
            "initial_state_group": manifest.initial_state_group,
            "decision_group_id": manifest.decision_group_id,
            "split_group_id": manifest.split_group_id,
            "split": manifest.split,
            "information_stratum": manifest.information_stratum.value,
            "prompt": manifest.context.prompt,
            "reset_state_sha256": manifest.reset_state_sha256,
            "model_seeds": list(manifest.model_seeds),
        }
        if row is None or any(
            row[name] != value for name, value in expected_public.items()
        ):
            raise ValueError(
                "decision freeze public identity differs from inventory row"
            )
        admit(
            freeze.inventory_row_sha256,
            {
                "row_id": row["row_id"],
                "inventory_row_sha256": freeze.inventory_row_sha256,
                "decision_group_id": row["decision_group_id"],
                "split_group_id": row["split_group_id"],
                "split": row["split"],
                "information_stratum": row["information_stratum"],
                "terminal_status": freeze.proposal_population_status,
                "terminal_artifact_sha256": freeze.freeze_receipt_sha256,
                "manifest_sha256": manifest.fingerprint(),
            },
        )
    failure_roots = tuple(
        Path(path).expanduser().resolve() for path in proposal_failure_dirs
    )
    if len(set(failure_roots)) != len(failure_roots):
        raise ValueError("proposal-failure directories must be unique")
    for root in failure_roots:
        failure = _load_verified_proposal_failure(root)
        if failure["schema_version"] != "method-v1-proposal-failure-v2":
            raise ValueError(
                "inventory plans require inventory-bound proposal failures"
            )
        if failure["reset_inventory_sha256"] != inventory.inventory_sha256:
            raise ValueError("proposal failure is bound to a different reset inventory")
        row_sha256 = str(failure["inventory_row_sha256"])
        row = inventory_by_row.get(row_sha256)
        if row is None or failure["scene_spec_sha256"] != row["scene_spec_sha256"]:
            raise ValueError(
                "proposal failure scene identity differs from inventory row"
            )
        admit(
            row_sha256,
            {
                "row_id": row["row_id"],
                "inventory_row_sha256": row_sha256,
                "decision_group_id": row["decision_group_id"],
                "split_group_id": row["split_group_id"],
                "split": row["split"],
                "information_stratum": row["information_stratum"],
                "terminal_status": "PROPOSAL_FAILURE",
                "terminal_artifact_sha256": failure["proposal_failure_sha256"],
                "manifest_sha256": None,
            },
        )
    missing = sorted(set(inventory_by_row) - set(terminals))
    if missing:
        raise ValueError(
            f"reset inventory has {len(missing)} rows without a proposal terminal"
        )
    status_names = (
        "VALID_CHOICE_SET",
        "VALID_SINGLE_PRIMITIVE_SET",
        "PROPOSAL_FAILURE",
    )
    terminal_rows = sorted(terminals.values(), key=lambda row: str(row["row_id"]))
    status_counts = {
        name: sum(row["terminal_status"] == name for row in terminal_rows)
        for name in status_names
    }
    return {
        "reset_inventory_sha256": inventory.inventory_sha256,
        "population_group_count": len(terminal_rows),
        "proposal_covered_group_count": (
            status_counts["VALID_CHOICE_SET"]
            + status_counts["VALID_SINGLE_PRIMITIVE_SET"]
        ),
        "choice_eligible_group_count": status_counts["VALID_CHOICE_SET"],
        "status_counts": status_counts,
        "terminal_rows": terminal_rows,
    }


def _validate_serialized_proposal_population(
    value: Mapping[str, Any],
    *,
    freezes: Sequence[VerifiedDecisionFreeze],
) -> None:
    expected_keys = {
        "reset_inventory_sha256",
        "population_group_count",
        "proposal_covered_group_count",
        "choice_eligible_group_count",
        "status_counts",
        "terminal_rows",
    }
    if set(value) != expected_keys:
        raise ValueError("proposal-population fields differ from schema")
    inventory_sha256 = _lower_sha256(
        value["reset_inventory_sha256"], name="proposal-population inventory SHA-256"
    )
    rows = value["terminal_rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("proposal population must contain terminal rows")
    row_fields = {
        "row_id",
        "inventory_row_sha256",
        "decision_group_id",
        "split_group_id",
        "split",
        "information_stratum",
        "terminal_status",
        "terminal_artifact_sha256",
        "manifest_sha256",
    }
    if any(not isinstance(row, Mapping) or set(row) != row_fields for row in rows):
        raise ValueError("proposal-population terminal row fields differ from schema")
    row_digests = [
        _lower_sha256(row["inventory_row_sha256"], name="terminal row SHA-256")
        for row in rows
    ]
    if len(set(row_digests)) != len(row_digests):
        raise ValueError("proposal population contains duplicate/replacement rows")
    statuses = (
        "VALID_CHOICE_SET",
        "VALID_SINGLE_PRIMITIVE_SET",
        "PROPOSAL_FAILURE",
    )
    observed = {
        status: sum(row["terminal_status"] == status for row in rows)
        for status in statuses
    }
    if any(row["terminal_status"] not in statuses for row in rows):
        raise ValueError("proposal population contains an unknown terminal status")
    status_counts = value["status_counts"]
    if (
        not isinstance(status_counts, Mapping)
        or set(status_counts) != set(statuses)
        or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in status_counts.values()
        )
    ):
        raise ValueError("proposal-population status counts differ from schema")
    row_ids: set[str] = set()
    decision_group_ids: set[str] = set()
    terminal_artifacts: set[str] = set()
    split_group_owner: dict[str, str] = {}
    for row in rows:
        for name in ("row_id", "decision_group_id", "split_group_id"):
            if _clean_text(row[name], name=f"proposal population {name}") != row[name]:
                raise ValueError(f"proposal population {name} is not canonical")
        if row["row_id"] != f"reset:{row['decision_group_id']}":
            raise ValueError("proposal population row/decision identity changed")
        if row["row_id"] in row_ids or row["decision_group_id"] in decision_group_ids:
            raise ValueError("proposal population contains duplicate public identities")
        row_ids.add(row["row_id"])
        decision_group_ids.add(row["decision_group_id"])
        if row["split"] not in {"train", "validation", "calibration", "test"}:
            raise ValueError("proposal population contains an invalid split")
        previous = split_group_owner.setdefault(row["split_group_id"], row["split"])
        if previous != row["split"]:
            raise ValueError("proposal population split group crosses data splits")
        try:
            InformationStratum(str(row["information_stratum"]))
        except ValueError as error:
            raise ValueError(
                "proposal population contains an invalid information stratum"
            ) from error
        terminal_artifact = _lower_sha256(
            row["terminal_artifact_sha256"],
            name="proposal terminal artifact SHA-256",
        )
        if terminal_artifact in terminal_artifacts:
            raise ValueError("proposal population reuses one terminal artifact")
        terminal_artifacts.add(terminal_artifact)
        if row["terminal_status"] == "PROPOSAL_FAILURE":
            if row["manifest_sha256"] is not None:
                raise ValueError("proposal failure must not claim a decision manifest")
        else:
            _lower_sha256(
                row["manifest_sha256"], name="proposal terminal manifest SHA-256"
            )
    if [row["row_id"] for row in rows] != sorted(row_ids):
        raise ValueError("proposal population terminal rows are not canonical-sorted")
    aggregate_counts = (
        value["population_group_count"],
        value["proposal_covered_group_count"],
        value["choice_eligible_group_count"],
    )
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0
        for count in aggregate_counts
    ) or (
        value["population_group_count"] != len(rows)
        or status_counts != observed
        or value["proposal_covered_group_count"]
        != observed["VALID_CHOICE_SET"] + observed["VALID_SINGLE_PRIMITIVE_SET"]
        or value["choice_eligible_group_count"] != observed["VALID_CHOICE_SET"]
    ):
        raise ValueError("proposal-population aggregate counts are inconsistent")
    terminals_by_row = {row["inventory_row_sha256"]: row for row in rows}
    valid_terminal_rows = {
        row["inventory_row_sha256"]
        for row in rows
        if row["terminal_status"] in {"VALID_CHOICE_SET", "VALID_SINGLE_PRIMITIVE_SET"}
    }
    freeze_rows = {freeze.inventory_row_sha256 for freeze in freezes}
    if None in freeze_rows or len(freeze_rows) != len(tuple(freezes)):
        raise ValueError("collection freezes do not have unique inventory rows")
    if valid_terminal_rows != freeze_rows:
        raise ValueError(
            "every serialized VALID proposal terminal must have exactly one "
            "supplied decision freeze"
        )
    for freeze in freezes:
        if (
            freeze.reset_inventory_sha256 != inventory_sha256
            or freeze.inventory_row_sha256 not in terminals_by_row
        ):
            raise ValueError("collection freeze is outside proposal population")
        terminal = terminals_by_row[freeze.inventory_row_sha256]
        if (
            terminal["terminal_status"] != freeze.proposal_population_status
            or terminal["terminal_artifact_sha256"] != freeze.freeze_receipt_sha256
            or terminal["manifest_sha256"] != freeze.manifest.fingerprint()
        ):
            raise ValueError("collection freeze differs from proposal terminal row")


def _collection_plan_body(
    *,
    plan_id: str,
    scorer_verifier_auth_key_id: str,
    freezes: Sequence[VerifiedDecisionFreeze],
    source_config_sha256: str,
    configured_split_counts: Mapping[str, int],
    configured_stratum_counts: Mapping[str, Mapping[str, int]],
    proposal_population: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = tuple(freezes)
    if not rows and (
        proposal_population is None
        or int(proposal_population.get("proposal_covered_group_count", -1)) != 0
    ):
        raise ValueError(
            "collection plan requires a decision freeze unless every inventory "
            "row is an immutable proposal failure"
        )
    roots = tuple(item.freeze_dir.resolve() for item in rows)
    if len(set(roots)) != len(roots):
        raise ValueError("collection plan freeze directories must be unique")
    if rows:
        validate_manifest_splits(tuple(item.manifest for item in rows))
    if proposal_population is not None:
        _validate_serialized_proposal_population(
            proposal_population,
            freezes=rows,
        )
    auth_key_id = _lower_sha256(
        scorer_verifier_auth_key_id,
        name="scorer verifier authentication key ID",
    )
    groups = sorted(
        (_collection_plan_group(item) for item in rows),
        key=lambda value: str(value["manifest_id"]),
    )
    split_names = ("train", "validation", "calibration", "test")
    configured_counts = {
        split: int(configured_split_counts[split]) for split in split_names
    }
    population_rows = (
        groups
        if proposal_population is None
        else list(proposal_population["terminal_rows"])
    )
    observed_reset_counts = (
        {
            split: len(
                {row["initial_state_group"] for row in groups if row["split"] == split}
            )
            for split in split_names
        }
        if proposal_population is None
        else {
            split: sum(row["split"] == split for row in population_rows)
            for split in split_names
        }
    )
    if observed_reset_counts != configured_counts:
        raise ValueError(
            "collection-plan reset population does not match source-config "
            "reset_groups; "
            f"expected={configured_counts}, observed={observed_reset_counts}"
        )
    if proposal_population is None:
        stratum_counts = validate_manifest_information_strata(
            tuple(item.manifest for item in rows),
            expected_counts=configured_stratum_counts,
        )
    else:
        stratum_counts = {
            split: {
                item.value: sum(
                    row["split"] == split and row["information_stratum"] == item.value
                    for row in population_rows
                )
                for item in InformationStratum
            }
            for split in split_names
        }
        expected_strata = {
            split: {
                item.value: int(configured_stratum_counts[split][item.value])
                for item in InformationStratum
            }
            for split in split_names
        }
        if stratum_counts != expected_strata:
            raise ValueError("proposal population strata differ from source config")
    expected_split_counts = {
        split: {
            "reset_groups": configured_counts[split],
            "decision_groups": sum(row["split"] == split for row in groups),
            "split_groups": len(
                {row["split_group_id"] for row in groups if row["split"] == split}
            ),
            "schedule_entries": sum(
                int(row["entry_count"]) for row in groups if row["split"] == split
            ),
            "information_strata": stratum_counts[split],
            **(
                {}
                if proposal_population is None
                else {
                    "proposal_failures": sum(
                        row["split"] == split
                        and row["terminal_status"] == "PROPOSAL_FAILURE"
                        for row in population_rows
                    ),
                    "choice_eligible_groups": sum(
                        row["split"] == split
                        and row["terminal_status"] == "VALID_CHOICE_SET"
                        for row in population_rows
                    ),
                    "single_primitive_groups": sum(
                        row["split"] == split
                        and row["terminal_status"] == "VALID_SINGLE_PRIMITIVE_SET"
                        for row in population_rows
                    ),
                }
            ),
        }
        for split in split_names
    }
    return {
        "schema_version": (
            COLLECTION_PLAN_SCHEMA
            if proposal_population is None
            else INVENTORY_COLLECTION_PLAN_SCHEMA
        ),
        "plan_id": _clean_text(plan_id, name="collection plan ID"),
        "pre_outcome_freeze": True,
        "config_identity": _collection_config_identity(
            rows, source_config_sha256=source_config_sha256
        ),
        "scorer_verifier_auth_key_id": auth_key_id,
        "infrastructure_failure_policy": dict(_COLLECTION_INFRASTRUCTURE_POLICY),
        "expected_split_counts": expected_split_counts,
        "group_count": len(groups),
        "schedule_entry_count": sum(int(row["entry_count"]) for row in groups),
        "groups": groups,
        **(
            {}
            if proposal_population is None
            else {"proposal_population": dict(proposal_population)}
        ),
    }


def freeze_collection_plan(
    *,
    freeze_dirs: Sequence[str | Path],
    config_path: str | Path,
    plan_id: str,
    scorer_auth_key_path: str | Path,
    output: str | Path,
    reset_inventory_path: str | Path | None = None,
    proposal_failure_dirs: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Freeze the complete study population before any schedule row is consumed."""

    roots = tuple(Path(path).expanduser().resolve() for path in freeze_dirs)
    if not roots and reset_inventory_path is None:
        raise ValueError(
            "collection plan requires --freeze-dir unless --reset-inventory "
            "proves an all-proposal-failure population"
        )
    if len(set(roots)) != len(roots):
        raise ValueError("collection plan freeze directories must be unique")
    for root in roots:
        claims = root / "execution_claims"
        if claims.exists() and any(path.is_file() for path in claims.rglob("*")):
            raise ValueError(
                "collection plan must be frozen before every execution claim/outcome"
            )
    freezes = tuple(load_verified_decision_freeze(root) for root in roots)
    source_config_sha256, configured_counts, configured_strata = (
        _collection_source_config_contract(config_path, freezes)
    )
    inventory = (
        None
        if reset_inventory_path is None
        else load_verified_reset_inventory(
            reset_inventory_path,
            config_path=config_path,
        )
    )
    if inventory is None and proposal_failure_dirs:
        raise ValueError("proposal failures require a reset inventory")
    proposal_population = (
        None
        if inventory is None
        else _proposal_population_from_terminals(
            inventory=inventory,
            freezes=freezes,
            proposal_failure_dirs=proposal_failure_dirs,
        )
    )
    body = _collection_plan_body(
        plan_id=plan_id,
        scorer_verifier_auth_key_id=scorer_auth_key_id(scorer_auth_key_path),
        freezes=freezes,
        source_config_sha256=source_config_sha256,
        configured_split_counts=configured_counts,
        configured_stratum_counts=configured_strata,
        proposal_population=proposal_population,
    )
    document = {**body, "collection_plan_sha256": canonical_sha256(body)}
    destination = Path(output).expanduser().resolve()
    _write_json_once(destination, document)
    # Re-open all source freezes and the serialized bytes before reporting a
    # usable plan. This also catches path normalization or serialization drift.
    return dict(
        load_verified_collection_plan(
            destination,
            freeze_dirs=roots,
            config_path=config_path,
            reset_inventory_path=reset_inventory_path,
            proposal_failure_dirs=proposal_failure_dirs,
        ).document
    )


def load_verified_collection_plan(
    path: str | Path,
    *,
    freeze_dirs: Sequence[str | Path],
    config_path: str | Path,
    expected_auth_key_id: str | None = None,
    reset_inventory_path: str | Path | None = None,
    proposal_failure_dirs: Sequence[str | Path] = (),
) -> VerifiedCollectionPlan:
    """Re-open one plan and prove exact equality with every named freeze."""

    source = Path(path).expanduser().resolve()
    document = _read_json(source)
    expected_keys = {
        "schema_version",
        "plan_id",
        "pre_outcome_freeze",
        "config_identity",
        "scorer_verifier_auth_key_id",
        "infrastructure_failure_policy",
        "expected_split_counts",
        "group_count",
        "schedule_entry_count",
        "groups",
        "collection_plan_sha256",
    }
    schema_version = document.get("schema_version")
    if schema_version == INVENTORY_COLLECTION_PLAN_SCHEMA:
        expected_keys.add("proposal_population")
    elif schema_version != COLLECTION_PLAN_SCHEMA:
        raise ValueError("unsupported collection-plan schema")
    if set(document) != expected_keys:
        raise ValueError("collection plan fields differ from the frozen schema")
    body = {
        key: value for key, value in document.items() if key != "collection_plan_sha256"
    }
    plan_sha256 = _lower_sha256(
        document["collection_plan_sha256"], name="collection plan SHA-256"
    )
    if (
        document["pre_outcome_freeze"] is not True
        or canonical_sha256(body) != plan_sha256
    ):
        raise ValueError("collection plan digest or pre-outcome identity is invalid")
    raw_groups = document["groups"]
    if not isinstance(raw_groups, list):
        raise TypeError("collection plan groups must be a list")
    if not raw_groups and schema_version != INVENTORY_COLLECTION_PLAN_SCHEMA:
        raise ValueError("legacy collection plan groups must be non-empty")
    supplied_roots = tuple(Path(item).expanduser().resolve() for item in freeze_dirs)
    if not supplied_roots and schema_version != INVENTORY_COLLECTION_PLAN_SCHEMA:
        raise ValueError("all collection-plan freeze directories must be supplied")
    if len(set(supplied_roots)) != len(supplied_roots):
        raise ValueError("supplied collection freeze directories must be unique")
    freezes = tuple(load_verified_decision_freeze(root) for root in supplied_roots)
    source_config_sha256, configured_counts, configured_strata = (
        _collection_source_config_contract(config_path, freezes)
    )
    if schema_version == INVENTORY_COLLECTION_PLAN_SCHEMA:
        if reset_inventory_path is None:
            if proposal_failure_dirs:
                raise ValueError(
                    "proposal-failure directories require the reset inventory"
                )
            raw_population = document["proposal_population"]
            if not isinstance(raw_population, Mapping):
                raise TypeError("proposal population must be a mapping")
            proposal_population = dict(raw_population)
            _validate_serialized_proposal_population(
                proposal_population,
                freezes=freezes,
            )
        else:
            inventory = load_verified_reset_inventory(
                reset_inventory_path,
                config_path=config_path,
            )
            proposal_population = _proposal_population_from_terminals(
                inventory=inventory,
                freezes=freezes,
                proposal_failure_dirs=proposal_failure_dirs,
            )
    else:
        if reset_inventory_path is not None or proposal_failure_dirs:
            raise ValueError(
                "legacy collection plan cannot receive inventory terminals"
            )
        proposal_population = None
    rebuilt = _collection_plan_body(
        plan_id=str(document["plan_id"]),
        scorer_verifier_auth_key_id=str(document["scorer_verifier_auth_key_id"]),
        freezes=freezes,
        source_config_sha256=source_config_sha256,
        configured_split_counts=configured_counts,
        configured_stratum_counts=configured_strata,
        proposal_population=proposal_population,
    )
    if body != rebuilt:
        raise ValueError(
            "collection plan differs from re-opened freeze groups, splits, or identities"
        )
    auth_key_id = _lower_sha256(
        document["scorer_verifier_auth_key_id"],
        name="scorer verifier authentication key ID",
    )
    if expected_auth_key_id is not None and auth_key_id != _lower_sha256(
        expected_auth_key_id, name="expected scorer verifier authentication key ID"
    ):
        raise ValueError(
            "scorer verifier authentication key differs from collection plan"
        )
    return VerifiedCollectionPlan(
        plan_path=source,
        collection_plan_sha256=plan_sha256,
        scorer_verifier_auth_key_id=auth_key_id,
        freezes=freezes,
        document=document,
    )


def _claim_execution_once(
    *,
    freeze_dir: Path,
    output_root: Path,
    entry: BranchScheduleEntry,
    collection_plan_sha256: str,
    scorer_verifier_auth_key_id: str,
    selection_sha256: str | None = None,
) -> Path:
    """Atomically consume one frozen branch before any outcome-bearing work."""

    claims = freeze_dir / "execution_claims"
    claim = claims / f"{entry.entry_id}.json"
    body = {
        "schema_version": EXECUTION_CLAIM_SCHEMA,
        "entry_id": entry.entry_id,
        "schedule_id": entry.schedule_id,
        "execution_index": entry.execution_index,
        "output_root": str(output_root.resolve()),
        "selection_sha256": selection_sha256,
        "collection_plan_sha256": _lower_sha256(
            collection_plan_sha256, name="collection plan SHA-256"
        ),
        "scorer_verifier_auth_key_id": _lower_sha256(
            scorer_verifier_auth_key_id,
            name="scorer verifier authentication key ID",
        ),
    }
    payload = {**body, "execution_claim_sha256": canonical_sha256(body)}
    _write_json_once(claim, payload)
    return claim


def load_verified_execution_claim(
    *,
    freeze_dir: str | Path,
    entry: BranchScheduleEntry,
    collection_plan_sha256: str,
    scorer_verifier_auth_key_id: str,
) -> Mapping[str, Any]:
    """Verify that one attempt was consumed under the frozen global plan."""

    root = Path(freeze_dir).expanduser().resolve()
    claim = _read_json(root / "execution_claims" / f"{entry.entry_id}.json")
    expected_keys = {
        "schema_version",
        "entry_id",
        "schedule_id",
        "execution_index",
        "output_root",
        "selection_sha256",
        "collection_plan_sha256",
        "scorer_verifier_auth_key_id",
        "execution_claim_sha256",
    }
    if set(claim) != expected_keys:
        raise ValueError("execution claim fields differ from the frozen schema")
    body = {
        key: value for key, value in claim.items() if key != "execution_claim_sha256"
    }
    if (
        claim["schema_version"] != EXECUTION_CLAIM_SCHEMA
        or canonical_sha256(body) != claim["execution_claim_sha256"]
    ):
        raise ValueError("execution claim digest mismatch")
    expected = {
        "entry_id": entry.entry_id,
        "schedule_id": entry.schedule_id,
        "execution_index": entry.execution_index,
        "collection_plan_sha256": _lower_sha256(
            collection_plan_sha256, name="collection plan SHA-256"
        ),
        "scorer_verifier_auth_key_id": _lower_sha256(
            scorer_verifier_auth_key_id,
            name="scorer verifier authentication key ID",
        ),
    }
    for name, value in expected.items():
        if claim[name] != value:
            raise ValueError(f"execution claim changes frozen {name}")
    output_root = Path(str(claim["output_root"])).expanduser().resolve()
    if str(output_root) != claim["output_root"]:
        raise ValueError("execution claim output root is not canonical")
    selection = claim["selection_sha256"]
    if selection is not None:
        _lower_sha256(selection, name="selection SHA-256")
    return claim


class _DirectOnlyProposer:
    def __init__(self, proposer: Qwen25VLProposer) -> None:
        self.proposer = proposer
        self.proposal_failed = False
        self.audit_records: list[dict[str, Any]] = []

    @property
    def proposer_id(self) -> str:
        return self.proposer.proposer_id

    def propose_direct(
        self,
        context: PolicyContext,
        *,
        max_candidates: int,
        deterministic_seed: int,
    ) -> Sequence[GroundedIntervention]:
        if deterministic_seed != 0:
            raise ValueError("greedy Qwen continuation seed changed")
        request = {
            "context_fingerprint": context.fingerprint(),
            "proposer_id": self.proposer_id,
            "max_candidates": max_candidates,
            "deterministic_seed": deterministic_seed,
        }
        try:
            result = self.proposer.propose(context)
        except ProposalFailure as error:
            self.proposal_failed = True
            self.audit_records.append(
                {
                    **request,
                    "status": "PROPOSAL_FAILURE",
                    "failure": error.audit_dict(),
                }
            )
            return ()
        candidates = tuple(
            item for item in result.candidates if item.primitive is Primitive.DIRECT
        )
        selected = candidates[:max_candidates]
        self.audit_records.append(
            {
                **request,
                "status": "PROPOSED",
                "proposal": result.audit_dict(),
                "all_valid_candidates": [
                    item.policy_payload() for item in result.candidates
                ],
                "eligible_direct_candidates": [
                    item.policy_payload() for item in selected
                ],
                "selected_top1_fingerprint": (
                    None if not selected else selected[0].fingerprint()
                ),
            }
        )
        return selected


def _runtime_provider_id() -> str:
    return QwenProviderIdentity(
        transformers_version=QWEN25VL_TRANSFORMERS_VERSION
    ).provider_id


def _reproduce_initial_context(
    *,
    spec: SceneResetSpec,
    environment: LiberoE1Environment,
    store: RGBFrameStore,
) -> PolicyContext:
    return public_context_from_observation(
        prompt=spec.prompt,
        observation=environment.public_observation,
        frame_store=store,
        frame_namespace=f"{spec.decision_group_id}:initial",
        frame_index=0,
    )


def _label_bearing_branch(
    *,
    entry: BranchScheduleEntry,
    manifest: DecisionGroupManifest,
    final_context: PolicyContext,
    initial_frame_count: int,
    observed_outcome: bool,
    execution_status: ExecutionStatus,
    receipt_id: str,
    diagnostics: Mapping[str, Any],
    private_metadata: Mapping[str, Any],
) -> ObservedBranch:
    post_frames = final_context.frames[initial_frame_count:]
    if not post_frames:
        raise ValueError("label-bearing execution lacks a real post-action RGB frame")
    return ObservedBranch(
        branch_id=entry.entry_id,
        initial_state_group=entry.initial_state_group,
        decision_group_id=entry.decision_group_id,
        split=entry.split,
        reset_state_sha256=entry.reset_state_sha256,
        repeat_index=entry.repeat_index,
        context=manifest.context,
        executed_intervention=manifest.candidate(entry.candidate_id),
        post_action_frames=post_frames,
        outcome_contract=manifest.outcome_contract,
        observed_outcome=bool(observed_outcome),
        execution_status=execution_status,
        execution_receipt_id=receipt_id,
        diagnostics=diagnostics,
        private_evaluator_metadata=private_metadata,
    )


def run_schedule_entry(
    *,
    scene_spec_path: Path,
    freeze_dir: Path,
    collection_plan_path: Path,
    collection_freeze_dirs: Sequence[Path],
    collection_config_path: Path,
    scorer_auth_key_path: Path,
    execution_index: int,
    molmo_endpoint: str,
    qwen_endpoint: str,
    output_root: Path,
    selection_path: Path | None = None,
    scorer_endpoint: str = "http://127.0.0.1:8005",
) -> CollectionAttempt:
    """Execute one single-use real branch without overwriting an attempt."""

    freeze_dir = freeze_dir.expanduser().resolve()
    auth_key_id = scorer_auth_key_id(scorer_auth_key_path)
    collection_plan = load_verified_collection_plan(
        collection_plan_path,
        freeze_dirs=collection_freeze_dirs,
        config_path=collection_config_path,
        expected_auth_key_id=auth_key_id,
    )
    plan_matches = [
        item for item in collection_plan.freezes if item.freeze_dir == freeze_dir
    ]
    if len(plan_matches) != 1:
        raise ValueError("decision freeze is not a unique member of collection plan")
    spec = SceneResetSpec.from_mapping(_read_json(scene_spec_path))
    manifest = DecisionGroupManifest.from_mapping(
        _read_json(freeze_dir / "decision_manifest.json")
    )
    schedule = _load_schedule(freeze_dir / "branch_schedule.json")
    matches = [item for item in schedule if item.execution_index == execution_index]
    if len(matches) != 1:
        raise ValueError("execution_index is outside the frozen schedule")
    entry = matches[0]
    planned_groups = [
        row
        for row in collection_plan.document["groups"]
        if row["manifest_id"] == manifest.manifest_id
    ]
    if len(planned_groups) != 1 or entry.entry_id not in planned_groups[0]["entry_ids"]:
        raise ValueError("schedule entry is not frozen in the collection plan")
    if (
        entry.manifest_sha256 != manifest.fingerprint()
        or manifest.decision_group_id != spec.decision_group_id
    ):
        raise ValueError("scene, manifest, and schedule identities disagree")
    _validate_freeze_receipt(
        freeze_dir=freeze_dir,
        spec=spec,
        spec_path=scene_spec_path,
        manifest=manifest,
    )
    resolved = validate_resolved_method_v1_identity(
        _read_json(freeze_dir / "resolved_method_v1.json")
    )
    continuation_identity = FixedContinuationIdentity(
        **resolved["continuation"]["identity"]
    )
    executor_identity = MolmoAct2ServerIdentity.from_mapping(
        resolved["executor_identity"]
    )
    if entry.outcome_contract_sha256 != manifest.outcome_contract.fingerprint():
        raise ValueError("schedule changed the frozen outcome contract")
    selection_sha256: str | None = None
    if selection_path is not None:
        # The frozen LIBERO runtime has PyTorch 1.11, whereas scorer/cache
        # replay uses the learned-model environment (PyTorch >=2.11). Keep the
        # modern checkpoint loader out of this process. The identity-bound
        # service replays the exact selection before the branch is consumed.
        scorer = ScorerVerificationHTTPClient(
            scorer_endpoint,
            auth_key_path=scorer_auth_key_path,
        )
        scorer.health()
        verified_selection = scorer.verify(
            selection_path=selection_path,
            manifest_path=freeze_dir / "decision_manifest.json",
            schedule_path=freeze_dir / "branch_schedule.json",
            entry_id=entry.entry_id,
        )
        if verified_selection["scorer_verifier_auth_key_id"] != auth_key_id:
            raise ValueError(
                "learned checkpoint training provenance uses a different "
                "scorer-verifier authentication key"
            )
        selection_sha256 = str(verified_selection["artifact_sha256"])

    # Force both frozen services to become usable before consuming a single-use
    # execution row. In particular, Qwen's /ready endpoint loads the checkpoint
    # now instead of discovering missing weights or OOM only after OPEN changed
    # the physical state.
    client = MolmoAct2HTTPClient(molmo_endpoint, expected_identity=executor_identity)
    client.health()
    qwen = QwenProposalHTTPClient(
        qwen_endpoint,
        expected_provider_id=_runtime_provider_id(),
    )
    qwen.ready()
    bddl, init_states = _resolve_scene_paths(spec, spec_path=scene_spec_path)
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    branch_dir = output_root / entry.entry_id
    if branch_dir.exists():
        raise FileExistsError(f"refusing to overwrite branch artifact {branch_dir}")

    _claim_execution_once(
        freeze_dir=freeze_dir,
        output_root=output_root,
        entry=entry,
        collection_plan_sha256=collection_plan.collection_plan_sha256,
        scorer_verifier_auth_key_id=auth_key_id,
        selection_sha256=selection_sha256,
    )

    executor: LiberoMolmoBudgetedExecutor | None = None
    try:
        branch_dir.mkdir(parents=True, exist_ok=False)
        _write_json_once(
            branch_dir / "started.json",
            {
                "status": "STARTED",
                "entry": entry.to_dict(),
                "manifest_sha256": manifest.fingerprint(),
                "selection_sha256": selection_sha256,
                "collection_plan_sha256": collection_plan.collection_plan_sha256,
                "scorer_verifier_auth_key_id": auth_key_id,
            },
        )
        frame_store = RGBFrameStore(branch_dir / "public_frames")
        with LiberoE1Environment(
            bddl_file=bddl,
            init_states_file=init_states,
            init_state_index=spec.init_state_index,
            env_seed=spec.env_seed,
            expected_reset_state_sha256=spec.reset_state_sha256,
            image_size=spec.image_size,
            settle_steps=spec.settle_steps,
            control_mode=spec.control_mode,
        ) as environment:
            initial = _reproduce_initial_context(
                spec=spec, environment=environment, store=frame_store
            )
            if initial != manifest.context:
                raise ValueError("live reset does not reproduce frozen public context")
            executor = LiberoMolmoBudgetedExecutor(
                environment=environment,
                client=client,
                frame_store=frame_store,
                branch_namespace=entry.entry_id,
            )
            if executor.executor_id != continuation_identity.executor_id:
                raise ValueError("live Method-V1 executor identity mismatch")
            continuation_proposer = Qwen25VLProposer(
                runtime=qwen,
                frame_store=frame_store,
                enabled_primitives=(Primitive.DIRECT,),
                max_candidates=3,
                max_per_primitive=3,
            )
            direct_only = _DirectOnlyProposer(continuation_proposer)
            fixed = FixedHorizonContinuation(
                identity=continuation_identity,
                proposer=direct_only,
                executor=executor,
            )
            candidate = manifest.candidate(entry.candidate_id)
            try:
                result = fixed.run(
                    context=initial,
                    selected_candidate=candidate,
                    model_seed=entry.model_seed,
                )
                final_context = result.final_context
                outcome = environment.predicate(spec.evaluator_predicate)
                public_trace = {
                    "fixed_horizon": result.to_dict(),
                    "stage_traces": dict(executor.public_traces),
                    "continuation_proposal_audit": list(direct_only.audit_records),
                }
                status = ExecutionStatus.COMPLETED
                receipt_id = canonical_sha256(
                    [stage.receipt.receipt_id for stage in result.stages]
                )
                diagnostics = {
                    "fixed_horizon_status": result.status.value,
                    "control_steps_used": result.control_steps_used,
                    "model_calls": sum(
                        stage.control_steps_used // 10 for stage in result.stages
                    ),
                    "policy_output_failure": False,
                    "continuation_abstained": (
                        result.status
                        is FixedHorizonStatus.OPEN_WITH_NO_DIRECT_CANDIDATE
                    ),
                    "continuation_proposal_failure": direct_only.proposal_failed,
                    "selection_sha256": selection_sha256,
                    "collection_plan_sha256": (collection_plan.collection_plan_sha256),
                    "scorer_verifier_auth_key_id": auth_key_id,
                }
            except PolicyExecutionFailure as error:
                final_context = error.final_context
                outcome = environment.predicate(spec.evaluator_predicate)
                stage_traces = dict(executor.public_traces)
                public_trace = {
                    "status": "POLICY_OUTPUT_FAILURE",
                    "failed_stage": dict(error.public_trace),
                    "stage_traces": stage_traces,
                    "continuation_proposal_audit": list(direct_only.audit_records),
                }
                status = ExecutionStatus.FAILED
                receipt_id = canonical_sha256(
                    {
                        "entry_id": entry.entry_id,
                        "policy_failure": public_trace,
                    }
                )
                diagnostics = {
                    "fixed_horizon_status": "POLICY_OUTPUT_FAILURE",
                    "control_steps_used": environment.step_count,
                    "model_calls": sum(
                        len(trace.get("chunks", ())) for trace in stage_traces.values()
                    ),
                    "policy_output_failure": True,
                    "continuation_proposal_failure": direct_only.proposal_failed,
                    "selection_sha256": selection_sha256,
                    "collection_plan_sha256": (collection_plan.collection_plan_sha256),
                    "scorer_verifier_auth_key_id": auth_key_id,
                }
            private = {
                "evaluator_predicate": list(spec.evaluator_predicate),
                "final_task_success": bool(outcome),
                "policy_input_changed_by_evaluator": False,
            }
            branch = _label_bearing_branch(
                entry=entry,
                manifest=manifest,
                final_context=final_context,
                initial_frame_count=len(initial.frames),
                observed_outcome=outcome,
                execution_status=status,
                receipt_id=receipt_id,
                diagnostics=diagnostics,
                private_metadata=private,
            )
        _write_json_once(branch_dir / "public_execution_trace.json", public_trace)
        private_dir = branch_dir / "private"
        _write_json_once(private_dir / "evaluator_sidecar.json", private)
        _write_json_once(
            branch_dir / "observed_branch.public.json",
            branch.to_dict(include_private=False),
        )
        artifact_tree = _tree_sha256(branch_dir)
        attempt = CollectionAttempt(
            attempt_id=entry.entry_id,
            schedule_entry=entry,
            status=CollectionAttemptStatus.OUTCOME_EVALUATED,
            artifact_tree_sha256=artifact_tree,
            branch=branch,
        )
        _write_json_once(
            branch_dir / "attempt.json",
            attempt.to_dict(include_private_branch=False),
        )
        return attempt
    # This is the outer experiment boundary: every otherwise-unclassified
    # service, simulator, filesystem, or identity failure must consume the
    # single-use row as an unlabelled infrastructure failure. Policy-output
    # failures are handled above because their physical prefix is evaluable.
    except Exception as error:  # noqa: BLE001
        # The claim has consumed this row. Best-effort creation of the branch
        # directory closes it as an unlabelled infrastructure failure even if
        # startup artifact creation was itself the point of failure.
        branch_dir.mkdir(parents=True, exist_ok=True)
        infrastructure_trace: dict[str, Any] | None = None
        public_trace_sha256: str | None = None
        if executor is not None and executor.public_traces:
            infrastructure_trace = {
                "status": "INFRASTRUCTURE_FAILURE",
                "stage_traces": dict(executor.public_traces),
            }
            trace_path = branch_dir / "public_execution_trace.json"
            if trace_path.exists():
                public_trace_sha256 = canonical_sha256(_read_json(trace_path))
            else:
                _write_json_once(trace_path, infrastructure_trace)
                public_trace_sha256 = canonical_sha256(infrastructure_trace)
        failure_message = str(error) or type(error).__name__
        failure_payload = {
            "status": "INFRASTRUCTURE_FAILURE",
            "failure_type": type(error).__name__,
            "message": failure_message,
            "traceback": traceback.format_exc(),
            "selection_sha256": selection_sha256,
            "collection_plan_sha256": collection_plan.collection_plan_sha256,
            "scorer_verifier_auth_key_id": auth_key_id,
            "public_trace_sha256": public_trace_sha256,
        }
        failure_sha = canonical_sha256(failure_payload)
        _write_json_once(branch_dir / "infrastructure_failure.json", failure_payload)
        artifact_tree = _tree_sha256(branch_dir)
        attempt = CollectionAttempt(
            attempt_id=entry.entry_id,
            schedule_entry=entry,
            status=CollectionAttemptStatus.INFRASTRUCTURE_FAILURE,
            artifact_tree_sha256=artifact_tree,
            infrastructure_failure=InfrastructureFailure(
                failure_type=type(error).__name__,
                message=failure_message,
                diagnostics_sha256=failure_sha,
            ),
        )
        _write_json_once(branch_dir / "attempt.json", attempt.to_dict())
        return attempt


def finalize_collection(
    *,
    collection_plan_path: Path,
    collection_config_path: Path,
    freeze_dirs: Sequence[Path],
    attempt_paths: Sequence[Path],
    dataset_id: str,
    require_complete: bool,
    output: Path,
) -> dict[str, Any]:
    from .train_outcomes import (
        build_method_v1_outcome_dataset,
        require_receipt_backed_dataset,
    )

    admitted = build_method_v1_outcome_dataset(
        collection_plan_path=collection_plan_path,
        collection_config_path=collection_config_path,
        freeze_dirs=freeze_dirs,
        attempt_paths=attempt_paths,
        dataset_id=dataset_id,
        require_complete=require_complete,
    )
    dataset = require_receipt_backed_dataset(admitted)
    canonical_dataset = dataset.to_dict()
    _write_json_once(output, canonical_dataset)
    return canonical_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    render = subcommands.add_parser("render")
    render.add_argument("--scene-spec", required=True, type=Path)
    render.add_argument("--output-dir", required=True, type=Path)

    freeze_inventory = subcommands.add_parser("freeze-inventory")
    freeze_inventory.add_argument(
        "--scene-spec", action="append", required=True, type=Path
    )
    freeze_inventory.add_argument("--config", required=True, type=Path)
    freeze_inventory.add_argument("--inventory-id", required=True)
    freeze_inventory.add_argument("--output", required=True, type=Path)

    verify_inventory = subcommands.add_parser("verify-inventory")
    verify_inventory.add_argument(
        "--scene-spec", action="append", required=True, type=Path
    )
    verify_inventory.add_argument("--config", required=True, type=Path)
    verify_inventory.add_argument("--inventory", required=True, type=Path)

    freeze = subcommands.add_parser("freeze")
    freeze.add_argument("--scene-spec", required=True, type=Path)
    freeze.add_argument("--prepared-dir", required=True, type=Path)
    freeze.add_argument("--config", required=True, type=Path)
    freeze.add_argument("--executor-identity", required=True, type=Path)
    freeze.add_argument("--output-dir", required=True, type=Path)
    freeze.add_argument("--qwen-device-map", default="auto")
    freeze.add_argument("--local-files-only", action="store_true")
    freeze.add_argument("--reset-inventory", type=Path)

    freeze_plan = subcommands.add_parser("freeze-plan")
    freeze_plan.add_argument("--freeze-dir", action="append", default=[], type=Path)
    freeze_plan.add_argument("--config", required=True, type=Path)
    freeze_plan.add_argument("--plan-id", required=True)
    freeze_plan.add_argument("--scorer-auth-key-file", required=True, type=Path)
    freeze_plan.add_argument("--output", required=True, type=Path)
    freeze_plan.add_argument("--reset-inventory", type=Path)
    freeze_plan.add_argument(
        "--proposal-failure-dir", action="append", default=[], type=Path
    )

    run = subcommands.add_parser("run")
    run.add_argument("--scene-spec", required=True, type=Path)
    run.add_argument("--freeze-dir", required=True, type=Path)
    run.add_argument(
        "--collection-freeze-dir", action="append", required=True, type=Path
    )
    run.add_argument("--collection-plan", required=True, type=Path)
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--scorer-auth-key-file", required=True, type=Path)
    run.add_argument("--execution-index", required=True, type=int)
    run.add_argument("--molmo-endpoint", default="http://127.0.0.1:8003")
    run.add_argument("--qwen-endpoint", default="http://127.0.0.1:8004")
    run.add_argument("--scorer-endpoint", default="http://127.0.0.1:8005")
    run.add_argument("--output-root", required=True, type=Path)
    run.add_argument("--selection", type=Path)

    finalize = subcommands.add_parser("finalize")
    finalize.add_argument("--collection-plan", required=True, type=Path)
    finalize.add_argument("--config", required=True, type=Path)
    finalize.add_argument("--freeze-dir", action="append", required=True, type=Path)
    finalize.add_argument("--attempt", action="append", required=True, type=Path)
    finalize.add_argument("--dataset-id", required=True)
    finalize.add_argument("--require-complete", action="store_true")
    finalize.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "render":
        result = render_public_reset(
            scene_spec=args.scene_spec.resolve(),
            output_dir=args.output_dir.resolve(),
        )
    elif args.command == "freeze-inventory":
        result = freeze_reset_inventory(
            scene_spec_paths=tuple(path.resolve() for path in args.scene_spec),
            config_path=args.config.resolve(),
            inventory_id=args.inventory_id,
            output=args.output.resolve(),
        )
    elif args.command == "verify-inventory":
        verified = load_verified_reset_inventory(
            args.inventory.resolve(),
            config_path=args.config.resolve(),
            scene_spec_paths=tuple(path.resolve() for path in args.scene_spec),
        )
        result = {
            "status": "VERIFIED",
            "reset_inventory_sha256": verified.inventory_sha256,
            "row_count": len(verified.rows),
            "source_config_sha256": verified.source_config_sha256,
        }
    elif args.command == "freeze":
        result = freeze_decision_group(
            scene_spec_path=args.scene_spec.resolve(),
            prepared_dir=args.prepared_dir.resolve(),
            config_path=args.config.resolve(),
            executor_identity_path=args.executor_identity.resolve(),
            output_dir=args.output_dir.resolve(),
            qwen_device_map=(
                None if args.qwen_device_map.lower() == "none" else args.qwen_device_map
            ),
            local_files_only=args.local_files_only,
            reset_inventory_path=(
                None if args.reset_inventory is None else args.reset_inventory.resolve()
            ),
        )
    elif args.command == "freeze-plan":
        result = freeze_collection_plan(
            freeze_dirs=tuple(path.resolve() for path in args.freeze_dir),
            config_path=args.config.resolve(),
            plan_id=args.plan_id,
            scorer_auth_key_path=args.scorer_auth_key_file.resolve(),
            output=args.output.resolve(),
            reset_inventory_path=(
                None if args.reset_inventory is None else args.reset_inventory.resolve()
            ),
            proposal_failure_dirs=tuple(
                path.resolve() for path in args.proposal_failure_dir
            ),
        )
    elif args.command == "run":
        result = run_schedule_entry(
            scene_spec_path=args.scene_spec.resolve(),
            freeze_dir=args.freeze_dir.resolve(),
            collection_plan_path=args.collection_plan.resolve(),
            collection_freeze_dirs=tuple(
                path.resolve() for path in args.collection_freeze_dir
            ),
            collection_config_path=args.config.resolve(),
            scorer_auth_key_path=args.scorer_auth_key_file.resolve(),
            execution_index=args.execution_index,
            molmo_endpoint=args.molmo_endpoint,
            qwen_endpoint=args.qwen_endpoint,
            output_root=args.output_root.resolve(),
            selection_path=(
                None if args.selection is None else args.selection.resolve()
            ),
            scorer_endpoint=args.scorer_endpoint,
        ).to_dict(include_private_branch=False)
    else:
        result = finalize_collection(
            collection_plan_path=args.collection_plan.resolve(),
            collection_config_path=args.config.resolve(),
            freeze_dirs=tuple(path.resolve() for path in args.freeze_dir),
            attempt_paths=tuple(path.resolve() for path in args.attempt),
            dataset_id=args.dataset_id,
            require_complete=args.require_complete,
            output=args.output.resolve(),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
