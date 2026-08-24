"""Prospective, outcome-free S03 public-input preparation contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


RUNBOOK_PATH = Path("docs/s03_perception_decision_runbook.md")
RUNBOOK_SHA256 = "8e11538a64c5bd302c5d915c236862b78e44a8007de52eb33f9012f317c44afa"
S02_SCHEDULE_PATH = Path(
    "results/method/piu_open_primitive_qualification_schedule_v1.json"
)
S02_SCHEDULE_SHA256 = "45d41b75fd982b977a0c9c5f82f63c2f33b686fd1512308cacf3cc690847e762"
S02_CERTIFICATE_PATH = Path("results/method/piu_open_primitive_certificate_v1.json")
S02_CERTIFICATE_SHA256 = "5291b6450230b5c8d36a28a805676cf61f655583431f2e9c53b0d548872cbd75"
S02_OUTCOME_INDEX_PATH = Path(
    "results/method/piu_open_primitive_qualification_outcomes_v1.jsonl"
)
S02_OUTCOME_INDEX_SHA256 = "cdecdd72379d1425dd54dae32dcc38bdf4a9aad3e72c27810913de050d3f3ac3"
CANDIDATE_REGISTRY_PATH = Path(
    "configs/experiments/original_drawer_candidate_set.yaml"
)
CANDIDATE_REGISTRY_SHA256 = (
    "8580c622ac25fcddde9707b4fec2d82b0e4652f5af53b608084fe3bd716a2ab4"
)
DAG_AMENDMENT_PATH = Path("configs/experiments/piu_empirical_stage_dag_v2.yaml")
VALIDATOR_PATH = Path("src/piu/s03_preparation.py")
BUILDER_PATH = Path("scripts/evaluation/build_piu_s03_preparation.py")

MANIFEST_PATH = Path("results/method/piu_s03_perception_decision_input_manifest_v1.json")
SCHEDULE_PATH = Path("results/method/piu_s03_perception_decision_schedule_v1.json")
CERTIFICATE_PATH = Path(
    "results/method/piu_s03_perception_decision_certificate_v1.json"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PROMPTS = {
    "ACT": "Place the cream cheese in the basket.",
    "OPEN": "Place the butter in the basket.",
    "STOP": "Place the milk in the basket.",
}
_SUBTESTS = {
    "A": "A_INFORMATION_EFFECT",
    "B": "B_DECISION_ROUTING",
    "C": "C_CLOSED_LOOP_TRANSITION",
}
_VALIDATOR_ROLES = {
    "A": "information_effect_input",
    "B": "decision_routing_input",
    "C": "closed_loop_transition_input",
}
_ALLOWED_POLICY_FIELDS = [
    "prompt",
    "candidate_registry",
    "observation_sequence[].phase",
    "observation_sequence[].rgb.agentview",
    "observation_sequence[].rgb.wrist",
    "observation_sequence[].public_action_history",
]
_FORBIDDEN_POLICY_FIELDS = [
    "simulator_semantic_id",
    "simulator_instance_id",
    "simulator_segmentation",
    "target_mask",
    "object_pose",
    "oracle_target_marker",
    "oracle_target_location",
    "container_membership",
    "evaluator_only_fields",
    "task_predicate",
    "reward",
    "success",
    "failure",
]
_MANIFEST_RECORD_KEYS = {
    "record_id",
    "record_order",
    "subtest",
    "stratum",
    "linked_s02_index",
    "linked_s02_group",
    "prompt",
    "policy_visible_input",
    "public_input_digest",
    "allowed_policy_visible_fields",
    "privileged_fields_check",
    "s02_receipt_provenance",
    "expected_validator_role",
    "outcome_present",
    "inference_executed",
}
_SCHEDULE_RECORD_KEYS = {
    "schedule_index",
    "record_id",
    "manifest_record_sha256",
    "subtest",
    "stratum",
    "linked_s02_index",
    "prompt",
    "input_artifacts",
    "public_input_digest",
    "expected_validator_role",
    "outcome_present",
    "inference_executed",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def render_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


def _resolve(path: str | Path, *, repository_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repository_root / value


def _portable(path: Path, *, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _artifact(path: Path, *, repository_root: Path) -> dict[str, str]:
    resolved = _resolve(path, repository_root=repository_root)
    if not resolved.is_file():
        raise FileNotFoundError(f"required S03 input artifact is missing: {path}")
    return {
        "path": _portable(resolved, repository_root=repository_root),
        "sha256": sha256(resolved),
    }


def _frozen_artifact(
    path: Path, expected_sha256: str, *, repository_root: Path
) -> dict[str, str]:
    value = _artifact(path, repository_root=repository_root)
    if value["sha256"] != expected_sha256:
        raise ValueError(f"frozen upstream artifact changed: {path}")
    return value


def _load_s02_entries(*, repository_root: Path) -> list[dict[str, Any]]:
    _frozen_artifact(
        S02_SCHEDULE_PATH, S02_SCHEDULE_SHA256, repository_root=repository_root
    )
    schedule = json.loads(_resolve(S02_SCHEDULE_PATH, repository_root=repository_root).read_text())
    if (
        schedule.get("schema_version") != "piu.primitive-qualification-schedule.v1"
        or schedule.get("status")
        != "FROZEN_BEFORE_PRIMITIVE_QUALIFICATION_OUTCOMES"
        or schedule.get("outcomes_loaded") is not False
        or schedule.get("rollout_executed") is not False
        or schedule.get("pre_outcome_only") is not True
    ):
        raise ValueError("S02 schedule differs from its prospective input contract")
    entries = schedule.get("entries")
    if not isinstance(entries, list) or len(entries) != 124:
        raise ValueError("S02 schedule must contain exactly 124 entries")
    if [row.get("execution_index") for row in entries] != list(range(124)):
        raise ValueError("S02 execution indices are not the complete frozen order")
    if len({str(row.get("initial_state_group", "")) for row in entries}) != 124:
        raise ValueError("S02 schedule groups are not unique")
    return [dict(row) for row in entries]


def _receipt_reference(
    entry: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    relative = str(entry.get("expected_execution_receipt", ""))
    expected_prefix = "runs/piu_open_primitive_qualification_v1/"
    if not relative.startswith(expected_prefix) or not relative.endswith("/report.json"):
        raise ValueError("S02 receipt path escapes the frozen qualification root")
    return {
        **_artifact(Path(relative), repository_root=repository_root),
        "controller_visible": False,
        "semantic_content_loaded_for_input_construction": False,
    }


def _public_rgb(
    entry: Mapping[str, Any], phase: str, *, repository_root: Path
) -> dict[str, dict[str, str]]:
    receipt = _resolve(
        str(entry["expected_execution_receipt"]), repository_root=repository_root
    )
    names = {
        "pre": ("00_before_agentview.png", "00_before_wrist.png"),
        "post": ("05_returned_home_agentview.png", "05_returned_home_wrist.png"),
    }
    agent, wrist = names[phase]
    assets = receipt.parent / "assets"
    return {
        "agentview": _artifact(assets / agent, repository_root=repository_root),
        "wrist": _artifact(assets / wrist, repository_root=repository_root),
    }


def _public_history(
    entry: Mapping[str, Any], *, recorded: bool, repository_root: Path
) -> dict[str, Any]:
    if not recorded:
        return {"mode": "EMPTY", "artifact": None}
    receipt = _resolve(
        str(entry["expected_execution_receipt"]), repository_root=repository_root
    )
    return {
        "mode": "RECORDED_S02_OPEN_ACTIONS",
        "artifact": _artifact(
            receipt.parent / "assets/public_action_history.json",
            repository_root=repository_root,
        ),
    }


def _observation(
    entry: Mapping[str, Any], phase: str, *, repository_root: Path
) -> dict[str, Any]:
    return {
        "phase": phase,
        "rgb": _public_rgb(entry, phase, repository_root=repository_root),
        "public_action_history": _public_history(
            entry, recorded=phase == "post", repository_root=repository_root
        ),
    }


def _record_specs() -> list[tuple[str, str, str]]:
    specs = [("A", "PAIRED_PRE_POST", _PROMPTS["OPEN"])] * 124
    for stratum in ("ACT", "OPEN", "STOP"):
        specs.extend([("B", stratum, _PROMPTS[stratum])] * 124)
    specs.extend([("C", "OBSERVE_OPEN_REOBSERVE", _PROMPTS["OPEN"])] * 124)
    return specs


def _record_id(subtest: str, stratum: str, index: int) -> str:
    if subtest == "A":
        return f"s03-a-{index:03d}"
    if subtest == "B":
        return f"s03-b-{stratum.lower()}-{index:03d}"
    return f"s03-c-{index:03d}"


def _policy_input(
    entry: Mapping[str, Any], subtest: str, stratum: str, prompt: str, *, repository_root: Path
) -> dict[str, Any]:
    if subtest in {"A", "C"}:
        phases = ("pre", "post")
    elif stratum in {"ACT", "OPEN"}:
        phases = ("pre",)
    else:
        phases = ("post",)
    return {
        "prompt": prompt,
        "candidate_registry": _frozen_artifact(
            CANDIDATE_REGISTRY_PATH,
            CANDIDATE_REGISTRY_SHA256,
            repository_root=repository_root,
        ),
        "observation_sequence": [
            _observation(entry, phase, repository_root=repository_root)
            for phase in phases
        ],
    }


def _receipt_tree(
    entries: Sequence[Mapping[str, Any]], *, repository_root: Path
) -> dict[str, Any]:
    rows = []
    for entry in entries:
        ref = _receipt_reference(entry, repository_root=repository_root)
        rows.append(
            {
                "execution_index": int(entry["execution_index"]),
                "path": ref["path"],
                "sha256": ref["sha256"],
            }
        )
    return {
        "algorithm": "sha256_of_canonical_json_ordered_receipt_refs",
        "entry_count": 124,
        "root_sha256": canonical_sha256(rows),
        "controller_visible": False,
        "semantic_content_loaded_for_input_construction": False,
    }


def _upstream(
    entries: Sequence[Mapping[str, Any]], *, repository_root: Path
) -> dict[str, Any]:
    outcome_index = _frozen_artifact(
        S02_OUTCOME_INDEX_PATH,
        S02_OUTCOME_INDEX_SHA256,
        repository_root=repository_root,
    )
    return {
        "runbook": _frozen_artifact(
            RUNBOOK_PATH, RUNBOOK_SHA256, repository_root=repository_root
        ),
        "dag_amendment": _artifact(DAG_AMENDMENT_PATH, repository_root=repository_root),
        "preparation_validator": _artifact(VALIDATOR_PATH, repository_root=repository_root),
        "preparation_builder": _artifact(BUILDER_PATH, repository_root=repository_root),
        "s02_schedule": _frozen_artifact(
            S02_SCHEDULE_PATH,
            S02_SCHEDULE_SHA256,
            repository_root=repository_root,
        ),
        "s02_certificate": _frozen_artifact(
            S02_CERTIFICATE_PATH,
            S02_CERTIFICATE_SHA256,
            repository_root=repository_root,
        ),
        "s02_outcome_index": {
            **outcome_index,
            "controller_visible": False,
            "semantic_content_loaded_for_input_construction": False,
        },
        "s02_execution_receipt_tree": _receipt_tree(
            entries, repository_root=repository_root
        ),
        "candidate_registry": _frozen_artifact(
            CANDIDATE_REGISTRY_PATH,
            CANDIDATE_REGISTRY_SHA256,
            repository_root=repository_root,
        ),
    }


def build_s03_input_manifest(*, repository_root: Path) -> dict[str, Any]:
    """Build the deterministic 620-record manifest without running inference."""

    entries = _load_s02_entries(repository_root=repository_root)
    specs = _record_specs()
    records: list[dict[str, Any]] = []
    for order, (subtest, stratum, prompt) in enumerate(specs):
        index = order % 124
        entry = entries[index]
        policy_input = _policy_input(
            entry, subtest, stratum, prompt, repository_root=repository_root
        )
        records.append(
            {
                "record_id": _record_id(subtest, stratum, index),
                "record_order": order,
                "subtest": _SUBTESTS[subtest],
                "stratum": stratum,
                "linked_s02_index": index,
                "linked_s02_group": str(entry["initial_state_group"]),
                "prompt": prompt,
                "policy_visible_input": policy_input,
                "public_input_digest": canonical_sha256(policy_input),
                "allowed_policy_visible_fields": list(_ALLOWED_POLICY_FIELDS),
                "privileged_fields_check": {
                    "status": "PASS",
                    "scope": "policy_visible_input",
                    "forbidden_fields_present": [],
                    "online_oracle_inputs": [],
                },
                "s02_receipt_provenance": _receipt_reference(
                    entry, repository_root=repository_root
                ),
                "expected_validator_role": _VALIDATOR_ROLES[subtest],
                "outcome_present": False,
                "inference_executed": False,
            }
        )
    return {
        "schema_version": "piu.s03-perception-decision-input-manifest.v1",
        "status": "FROZEN_BEFORE_S03_OUTCOMES",
        "claim_scope": "PUBLIC_INPUT_FREEZE_ONLY_NOT_PERFORMANCE_EVIDENCE",
        "upstream": _upstream(entries, repository_root=repository_root),
        "record_order": [
            "A_INFORMATION_EFFECT:linked_s02_index_0_to_123",
            "B_DECISION_ROUTING:ACT:linked_s02_index_0_to_123",
            "B_DECISION_ROUTING:OPEN:linked_s02_index_0_to_123",
            "B_DECISION_ROUTING:STOP:linked_s02_index_0_to_123",
            "C_CLOSED_LOOP_TRANSITION:linked_s02_index_0_to_123",
        ],
        "counts": {
            "total": 620,
            "A_INFORMATION_EFFECT": 124,
            "B_DECISION_ROUTING": {"ACT": 124, "OPEN": 124, "STOP": 124},
            "C_CLOSED_LOOP_TRANSITION": 124,
        },
        "policy_input_contract": {
            "allowed_policy_visible_fields": list(_ALLOWED_POLICY_FIELDS),
            "forbidden_privileged_fields": list(_FORBIDDEN_POLICY_FIELDS),
            "online_oracle_inputs": [],
            "evaluator_private_inputs_loaded": False,
        },
        "records": records,
        "outcome_present": False,
        "inference_executed": False,
        "rollout_executed": False,
        "paper_claim_ready": False,
    }


def build_s03_offline_schedule(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
    repository_root: Path,
) -> dict[str, Any]:
    """Bind exact manifest inputs and order without executing any S03 model."""

    records = []
    for row in manifest["records"]:
        records.append(
            {
                "schedule_index": int(row["record_order"]),
                "record_id": row["record_id"],
                "manifest_record_sha256": canonical_sha256(row),
                "subtest": row["subtest"],
                "stratum": row["stratum"],
                "linked_s02_index": row["linked_s02_index"],
                "prompt": row["prompt"],
                "input_artifacts": copy.deepcopy(row["policy_visible_input"]),
                "public_input_digest": row["public_input_digest"],
                "expected_validator_role": row["expected_validator_role"],
                "outcome_present": False,
                "inference_executed": False,
            }
        )
    return {
        "schema_version": "piu.s03-perception-decision-schedule.v1",
        "status": "FROZEN_BEFORE_S03_OUTCOMES",
        "claim_scope": "OFFLINE_INPUT_SCHEDULE_ONLY_NOT_PERFORMANCE_EVIDENCE",
        "input_manifest": {
            "path": _portable(manifest_path, repository_root=repository_root),
            "sha256": manifest_sha256,
        },
        "upstream": copy.deepcopy(manifest["upstream"]),
        "record_order": copy.deepcopy(manifest["record_order"]),
        "counts": copy.deepcopy(manifest["counts"]),
        "records": records,
        "outcome_present": False,
        "inference_executed": False,
        "rollout_executed": False,
        "predictions_present": False,
        "labels_present": False,
        "certificate_present": False,
        "paper_claim_ready": False,
    }


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} must be a mapping")
    return value


def _verify_reference(value: Any, location: str, *, repository_root: Path) -> Path:
    ref = _mapping(value, location)
    if set(ref) != {"path", "sha256"}:
        raise ValueError(f"{location} is not an exact artifact reference")
    digest = str(ref.get("sha256", ""))
    path = _resolve(str(ref.get("path", "")), repository_root=repository_root)
    if not _SHA256.fullmatch(digest) or not path.is_file() or sha256(path) != digest:
        raise ValueError(f"{location} differs from its referenced bytes")
    return path


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _scan_policy_keys(value: Any, *, location: str = "policy_visible_input") -> None:
    forbidden = {_normalized_key(item) for item in _FORBIDDEN_POLICY_FIELDS}
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normalized_key(str(key))
            if normalized in forbidden:
                raise ValueError(f"{location} contains forbidden field {key!r}")
            _scan_policy_keys(child, location=f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _scan_policy_keys(child, location=f"{location}[{index}]")


def _validate_policy_input(value: Any, *, repository_root: Path) -> None:
    policy = _mapping(value, "policy_visible_input")
    if set(policy) != {"prompt", "candidate_registry", "observation_sequence"}:
        raise ValueError("policy-visible input fields differ from the frozen allowlist")
    if not isinstance(policy["prompt"], str) or not policy["prompt"].strip():
        raise ValueError("policy-visible prompt is empty")
    _verify_reference(
        policy["candidate_registry"], "candidate_registry", repository_root=repository_root
    )
    observations = policy["observation_sequence"]
    if not isinstance(observations, list) or not observations:
        raise ValueError("policy-visible observation sequence is empty")
    phases = []
    for index, raw in enumerate(observations):
        observation = _mapping(raw, f"observation_sequence[{index}]")
        if set(observation) != {"phase", "rgb", "public_action_history"}:
            raise ValueError("public observation fields differ from the frozen schema")
        phase = str(observation["phase"])
        if phase not in {"pre", "post"} or phase in phases:
            raise ValueError("public observation phase is invalid or duplicated")
        phases.append(phase)
        rgb = _mapping(observation["rgb"], "public RGB")
        if set(rgb) != {"agentview", "wrist"}:
            raise ValueError("public RGB must contain exactly agentview and wrist")
        for view in ("agentview", "wrist"):
            path = _verify_reference(
                rgb[view], f"public RGB {phase}.{view}", repository_root=repository_root
            )
            if path.suffix.lower() != ".png" or "/assets/" not in path.as_posix():
                raise ValueError("public RGB reference is not a direct PNG asset")
        history = _mapping(observation["public_action_history"], "public action history")
        if set(history) != {"mode", "artifact"}:
            raise ValueError("public action history fields differ from the frozen schema")
        expected_mode = "EMPTY" if phase == "pre" else "RECORDED_S02_OPEN_ACTIONS"
        if history["mode"] != expected_mode:
            raise ValueError("public history mode differs from observation phase")
        if phase == "pre":
            if history["artifact"] is not None:
                raise ValueError("pre observation must not receive future action history")
        else:
            history_path = _verify_reference(
                history["artifact"], "public action history", repository_root=repository_root
            )
            if history_path.name != "public_action_history.json":
                raise ValueError("post history is not the direct public action artifact")
    _scan_policy_keys(policy)


def _validate_record_counts(records: Sequence[Mapping[str, Any]]) -> None:
    if len(records) != 620:
        raise ValueError("S03 preparation must contain exactly 620 records")
    if [row.get("record_order") for row in records] != list(range(620)):
        raise ValueError("S03 manifest record order is not exact and contiguous")
    ids = [str(row.get("record_id", "")) for row in records]
    if any(not item for item in ids) or len(set(ids)) != 620:
        raise ValueError("S03 record IDs must be nonempty and unique")
    by_subtest = Counter(str(row.get("subtest")) for row in records)
    if by_subtest != Counter(
        {"A_INFORMATION_EFFECT": 124, "B_DECISION_ROUTING": 372, "C_CLOSED_LOOP_TRANSITION": 124}
    ):
        raise ValueError("S03 subtest counts differ from the frozen design")
    b_counts = Counter(
        str(row.get("stratum"))
        for row in records
        if row.get("subtest") == "B_DECISION_ROUTING"
    )
    if b_counts != Counter({"ACT": 124, "OPEN": 124, "STOP": 124}):
        raise ValueError("S03 routing strata differ from 124 ACT/OPEN/STOP records")
    indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in records:
        indices[(str(row.get("subtest")), str(row.get("stratum")))].append(
            int(row.get("linked_s02_index", -1))
        )
    if any(values != list(range(124)) for values in indices.values()):
        raise ValueError("an S03 subtest/stratum filtered or reordered S02 indices")
    if not all(101 in values and 104 in values for values in indices.values()):
        raise ValueError("S03 preparation omitted frozen S02 indices 101 or 104")


def validate_s03_input_manifest(
    path: Path, *, repository_root: Path
) -> dict[str, Any]:
    value = json.loads(path.read_text())
    manifest = _mapping(value, "S03 input manifest")
    required_top = {
        "schema_version", "status", "claim_scope", "upstream", "record_order",
        "counts", "policy_input_contract", "records", "outcome_present",
        "inference_executed", "rollout_executed", "paper_claim_ready",
    }
    if set(manifest) != required_top:
        raise ValueError("S03 input manifest fields differ from the frozen schema")
    if (
        manifest["schema_version"] != "piu.s03-perception-decision-input-manifest.v1"
        or manifest["status"] != "FROZEN_BEFORE_S03_OUTCOMES"
        or any(
            manifest[name] is not False
            for name in ("outcome_present", "inference_executed", "rollout_executed", "paper_claim_ready")
        )
    ):
        raise ValueError("S03 input manifest crossed the pre-outcome boundary")
    records = manifest["records"]
    if not isinstance(records, list) or any(not isinstance(row, Mapping) for row in records):
        raise TypeError("S03 manifest records must be mappings")
    _validate_record_counts(records)
    for row in records:
        if set(row) != _MANIFEST_RECORD_KEYS:
            raise ValueError("S03 manifest record fields differ from the frozen schema")
        if row["outcome_present"] is not False or row["inference_executed"] is not False:
            raise ValueError("S03 manifest record contains outcome-bearing state")
        if row["prompt"] != row["policy_visible_input"].get("prompt"):
            raise ValueError("S03 record prompt differs from its policy input")
        if row["allowed_policy_visible_fields"] != _ALLOWED_POLICY_FIELDS:
            raise ValueError("S03 record policy allowlist changed")
        if row["privileged_fields_check"] != {
            "status": "PASS", "scope": "policy_visible_input",
            "forbidden_fields_present": [], "online_oracle_inputs": [],
        }:
            raise ValueError("S03 privileged-field audit did not pass cleanly")
        _validate_policy_input(row["policy_visible_input"], repository_root=repository_root)
        if row["public_input_digest"] != canonical_sha256(row["policy_visible_input"]):
            raise ValueError("S03 public input digest is missing or stale")
    expected = build_s03_input_manifest(repository_root=repository_root)
    if manifest != expected:
        raise ValueError("S03 input manifest differs from deterministic frozen inputs")
    return dict(manifest)


def validate_s03_offline_schedule(
    path: Path, *, repository_root: Path
) -> dict[str, Any]:
    value = json.loads(path.read_text())
    schedule = _mapping(value, "S03 offline schedule")
    required_top = {
        "schema_version", "status", "claim_scope", "input_manifest", "upstream",
        "record_order", "counts", "records", "outcome_present",
        "inference_executed", "rollout_executed", "predictions_present",
        "labels_present", "certificate_present", "paper_claim_ready",
    }
    if set(schedule) != required_top:
        raise ValueError("S03 schedule fields differ from the frozen schema")
    if (
        schedule["schema_version"] != "piu.s03-perception-decision-schedule.v1"
        or schedule["status"] != "FROZEN_BEFORE_S03_OUTCOMES"
        or any(
            schedule[name] is not False
            for name in (
                "outcome_present", "inference_executed", "rollout_executed",
                "predictions_present", "labels_present", "certificate_present",
                "paper_claim_ready",
            )
        )
    ):
        raise ValueError("S03 schedule crossed the pre-outcome boundary")
    manifest_path = _verify_reference(
        schedule["input_manifest"], "S03 input manifest", repository_root=repository_root
    )
    manifest = validate_s03_input_manifest(manifest_path, repository_root=repository_root)
    records = schedule["records"]
    if not isinstance(records, list) or any(not isinstance(row, Mapping) for row in records):
        raise TypeError("S03 schedule records must be mappings")
    if len(records) != 620 or [row.get("schedule_index") for row in records] != list(range(620)):
        raise ValueError("S03 schedule must freeze exactly 620 ordered records")
    for schedule_row, manifest_row in zip(records, manifest["records"], strict=True):
        if set(schedule_row) != _SCHEDULE_RECORD_KEYS:
            raise ValueError("S03 schedule record contains unfrozen or outcome fields")
        if schedule_row["outcome_present"] is not False or schedule_row["inference_executed"] is not False:
            raise ValueError("S03 schedule record contains outcome-bearing state")
        if schedule_row["manifest_record_sha256"] != canonical_sha256(manifest_row):
            raise ValueError("S03 schedule record is not bound to its manifest record")
        if schedule_row["input_artifacts"] != manifest_row["policy_visible_input"]:
            raise ValueError("S03 scheduled inputs differ from the sanitized manifest")
        _validate_policy_input(schedule_row["input_artifacts"], repository_root=repository_root)
    expected = build_s03_offline_schedule(
        manifest,
        manifest_path=manifest_path,
        manifest_sha256=sha256(manifest_path),
        repository_root=repository_root,
    )
    if schedule != expected:
        raise ValueError("S03 schedule differs from deterministic manifest binding")
    return dict(schedule)


def validate_s03_public_outcome_certificate(
    path: Path, *, schedule_path: Path, repository_root: Path
) -> dict[str, Any]:
    """Keep the future public S03 DAG gate closed without a full certificate."""

    value = json.loads(path.read_text())
    certificate = _mapping(value, "S03 outcome certificate")
    required = {
        "schema_version", "status", "schedule", "complete_frozen_denominator",
        "record_count", "results", "public_input_firewall_passed",
        "formal_method_claim", "paper_claim_ready", "s02_artifacts_modified",
        "new_physical_rollout_executed",
    }
    if set(certificate) != required:
        raise ValueError("S03 certificate fields differ from the future gate schema")
    if certificate["schema_version"] != "piu.s03-perception-decision-certificate.v1":
        raise ValueError("unsupported S03 outcome certificate schema")
    if certificate["status"] not in {
        "DEVELOPMENT_MECHANISM_SUPPORTED",
        "DEVELOPMENT_MECHANISM_NOT_SUPPORTED",
        "INVALID_INPUT_FIREWALL",
    }:
        raise ValueError("S03 certificate has an unknown terminal status")
    referenced_schedule = _verify_reference(
        certificate["schedule"], "S03 certificate schedule", repository_root=repository_root
    )
    if referenced_schedule.resolve() != schedule_path.resolve():
        raise ValueError("S03 certificate references a different schedule")
    validate_s03_offline_schedule(referenced_schedule, repository_root=repository_root)
    if certificate["complete_frozen_denominator"] is not True or certificate["record_count"] != 620:
        raise ValueError("S03 certificate does not cover the complete denominator")
    results = _mapping(certificate["results"], "S03 certificate results")
    expected_results = {
        "information_effect": "piu.s03-information-effect-result.v1",
        "decision_routing": "piu.s03-decision-routing-result.v1",
        "closed_loop_transition": "piu.s03-closed-loop-transition-result.v1",
    }
    if set(results) != set(expected_results):
        raise ValueError("S03 certificate lacks one or more subtest results")
    for name, schema in expected_results.items():
        result_path = _verify_reference(
            results[name], f"S03 {name} result", repository_root=repository_root
        )
        result = json.loads(result_path.read_text())
        if result.get("schema_version") != schema:
            raise ValueError(f"S03 {name} result schema is unsupported")
    if (
        certificate["public_input_firewall_passed"] is not True
        or certificate["formal_method_claim"] is not False
        or certificate["paper_claim_ready"] is not False
        or certificate["s02_artifacts_modified"] is not False
        or certificate["new_physical_rollout_executed"] is not False
    ):
        raise ValueError("S03 certificate crosses its development/public-input scope")
    return dict(certificate)
