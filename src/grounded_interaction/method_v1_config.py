"""Resolved, fail-closed configuration for Method-V1 experiments.

The human-edited experiments/method_v1.yaml uses JSON syntax (and is therefore
valid YAML 1.2), so the E1-frozen root dependency lock does not need a YAML
parser. Outcome collection consumes a resolved JSON identity that binds the
source config, both frozen models, token preprocessing, serializer,
continuation, and outcome contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .continuation import FixedContinuationIdentity
from .contracts import (
    OutcomeContract,
    Primitive,
    canonical_json_bytes,
    canonical_sha256,
)
from .method_v1_data import validate_information_stratum_counts
from .method_v1_runtime import method_v1_executor_id
from .molmoact2 import MolmoAct2ServerIdentity
from .proposals import QWEN_DIRECT_ONLY_SYSTEM_PROMPT, qwen_proposer_id
from .qwen_provider import QwenProviderIdentity

METHOD_V1_CONFIG_SCHEMA = "method-v1-config-v1"
METHOD_V1_RESOLVED_SCHEMA = "method-v1-resolved-identity-v1"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
QWEN_TRANSFORMERS_VERSION = "4.57.6"
METHOD_V1_SERIALIZER_ID = "grounded-precise-text-v1"


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read strict JSON configuration {source}") from error
    if not isinstance(value, dict):
        raise TypeError("configuration root must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], *, expected: set[str], name: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _require_int(value: Any, *, name: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_method_v1_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate scientific invariants that may not change inside one dataset."""

    _exact_keys(
        value,
        expected={
            "schema_version",
            "experiment_id",
            "models",
            "primitives",
            "execution",
            "continuation",
            "outcome_contract",
            "collection",
            "model",
            "training",
            "calibration",
            "evaluation",
        },
        name="Method-V1 config",
    )
    if value.get("schema_version") != METHOD_V1_CONFIG_SCHEMA:
        raise ValueError("unexpected Method-V1 config schema")
    models = value.get("models")
    primitives = value.get("primitives")
    execution = value.get("execution")
    continuation = value.get("continuation")
    outcome = value.get("outcome_contract")
    collection = value.get("collection")
    if not all(
        isinstance(item, Mapping)
        for item in (models, primitives, execution, continuation, outcome, collection)
    ):
        raise TypeError(
            "models/primitives/execution/continuation/outcome/collection must be objects"
        )

    proposer = models["proposer_provider"]
    executor = models["executor"]
    if proposer["model"] != QWEN_MODEL_ID:
        raise ValueError("Method-V1 is frozen to Qwen2.5-VL-3B-Instruct")
    if proposer["revision"] != QWEN_MODEL_REVISION:
        raise ValueError("Qwen checkpoint revision is not the resolved revision")
    if proposer["transformers_version"] != QWEN_TRANSFORMERS_VERSION:
        raise ValueError("Method-V1 requires the verified Transformers version")
    if proposer["processor_use_fast"] is not False:
        raise ValueError("Method-V1 freezes the slow Qwen processor")
    qwen_identity = QwenProviderIdentity(transformers_version=QWEN_TRANSFORMERS_VERSION)
    expected_loading = {
        "torch_dtype": qwen_identity.torch_dtype,
        "quantization_mode": qwen_identity.quantization_mode,
        "load_mode": qwen_identity.load_mode,
    }
    for name, expected in expected_loading.items():
        if proposer.get(name) != expected:
            raise ValueError(f"Qwen provider loading identity changed {name}")
    generation = proposer["generation"]
    if generation != {"do_sample": False, "num_beams": 1, "max_new_tokens": 512}:
        raise ValueError("Qwen proposal decoding must be one-pass greedy")
    if proposer["coordinate_contract"] != "qwen25-smart-resized-absolute-xyxy-v1":
        raise ValueError("Qwen bbox coordinate contract changed")
    if _require_int(proposer["max_candidates"], name="max_candidates") != 6:
        raise ValueError("Method-V1 supports at most six candidates")
    if _require_int(proposer["max_per_primitive"], name="max_per_primitive") != 3:
        raise ValueError("Method-V1 supports at most three per primitive")

    if list(primitives["enabled"]) != ["DIRECT", "OPEN"]:
        raise ValueError("Method-V1 enables only DIRECT and OPEN")
    if list(primitives["onehot_order"]) != ["DIRECT", "OPEN"]:
        raise ValueError("primitive one-hot order is part of model identity")

    total = _require_int(execution["total_control_steps"], name="total_control_steps")
    direct = _require_int(execution["direct_steps"], name="direct_steps")
    opening = _require_int(execution["open_steps"], name="open_steps")
    follow = _require_int(
        execution["continuation_direct_steps"], name="continuation_direct_steps"
    )
    if total != 300 or direct != total or opening + follow != total:
        raise ValueError("Method-V1 requires DIRECT=300 and OPEN=100+DIRECT=200")
    if _require_int(execution["chunk_replan_steps"], name="chunk_replan_steps") != 10:
        raise ValueError("MolmoAct2 must replan every ten control steps")
    if execution["switch_rule"] != "fixed_budget_no_private_predicate":
        raise ValueError("OPEN switching may not read private predicates")
    if continuation["learned_scorer_calls"] != 0:
        raise ValueError("the fixed continuation may not call the learned scorer")
    if continuation["allow_information_primitives"] is not False:
        raise ValueError("continuation may propose DIRECT only")
    if outcome["outcome_name"] != "full_task_with_fixed_continuation_v1":
        raise ValueError("Method-V1 cannot use an E1 contact label")
    if outcome["horizon"] != total:
        raise ValueError("outcome horizon and execution budget differ")
    if executor["conditioning_mode"] != "precise_text":
        raise ValueError("development Method-V1 is fixed to precise_text")
    if executor["action_horizon"] != 10 or executor["flow_steps"] != 10:
        raise ValueError("MolmoAct2 action semantics changed")

    _exact_keys(
        collection,
        expected={
            "repeats_per_candidate",
            "paired_model_seeds",
            "reset_groups",
            "information_stratum_counts",
            "split_unit",
        },
        name="Method-V1 collection config",
    )
    if collection["repeats_per_candidate"] != 2:
        raise ValueError("Method-V1 requires two paired repeats per candidate")
    if collection["paired_model_seeds"] is not True:
        raise ValueError("Method-V1 requires paired model seeds")
    if collection["split_unit"] != "scene_layout_hidden_family":
        raise ValueError("Method-V1 split-unit contract changed")
    validate_information_stratum_counts(
        collection["information_stratum_counts"],
        reset_groups=collection["reset_groups"],
    )

    canonical_json_bytes(value)
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def load_method_v1_config(path: str | Path) -> dict[str, Any]:
    return validate_method_v1_config(_read_json(path))


def method_v1_information_stratum_counts(
    config: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    """Return the exact pre-outcome information-condition allocation."""

    checked = validate_method_v1_config(config)
    collection = checked["collection"]
    return validate_information_stratum_counts(
        collection["information_stratum_counts"],
        reset_groups=collection["reset_groups"],
    )


def method_v1_outcome_contract(
    config: Mapping[str, Any], *, executor_id: str
) -> OutcomeContract:
    validate_method_v1_config(config)
    return _fixed_continuation_identity(executor_id=executor_id).outcome_contract()


def _fixed_continuation_identity(*, executor_id: str) -> FixedContinuationIdentity:
    qwen_identity = QwenProviderIdentity(transformers_version=QWEN_TRANSFORMERS_VERSION)
    proposer_id = qwen_proposer_id(
        provider_id=qwen_identity.provider_id,
        proposal_camera="agentview",
        enabled_primitives=(Primitive.DIRECT,),
        max_candidates=3,
        max_per_primitive=3,
    )
    return FixedContinuationIdentity(
        proposer_id=proposer_id,
        proposer_model_id=QWEN_MODEL_ID,
        proposer_revision=QWEN_MODEL_REVISION,
        proposer_prompt_sha256=hashlib.sha256(
            QWEN_DIRECT_ONLY_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        proposer_seed=0,
        executor_id=executor_id,
        serializer_id=METHOD_V1_SERIALIZER_ID,
    )


def resolve_method_v1_identity(
    config: Mapping[str, Any], *, executor_identity: MolmoAct2ServerIdentity
) -> dict[str, Any]:
    """Create the immutable identity all collectors/trainers must agree on."""

    checked = validate_method_v1_config(config)
    executor = checked["models"]["executor"]
    if (
        executor_identity.checkpoint_id != executor["model"]
        or executor_identity.checkpoint_revision != executor["revision"]
        or executor_identity.upstream_code_revision
        != executor["upstream_code_revision"]
    ):
        raise ValueError("MolmoAct2 identity does not match source config")
    continuation = _fixed_continuation_identity(
        executor_id=method_v1_executor_id(executor_identity.digest),
    )
    contract = continuation.outcome_contract()
    payload = {
        "schema_version": METHOD_V1_RESOLVED_SCHEMA,
        "experiment_id": checked["experiment_id"],
        "source_config_sha256": canonical_sha256(checked),
        "qwen": {
            "model_id": QWEN_MODEL_ID,
            "revision": QWEN_MODEL_REVISION,
            "transformers_version": QWEN_TRANSFORMERS_VERSION,
            "processor_use_fast": False,
            "attention_implementation": checked["models"]["proposer_provider"][
                "attention_implementation"
            ],
            "torch_dtype": checked["models"]["proposer_provider"]["torch_dtype"],
            "quantization_mode": checked["models"]["proposer_provider"][
                "quantization_mode"
            ],
            "load_mode": checked["models"]["proposer_provider"]["load_mode"],
            "coordinate_contract": checked["models"]["proposer_provider"][
                "coordinate_contract"
            ],
        },
        "executor_identity": executor_identity.to_dict(),
        "executor_identity_sha256": executor_identity.digest,
        "serializer_id": METHOD_V1_SERIALIZER_ID,
        "continuation": {
            "config": checked["continuation"],
            "identity": continuation.to_dict(),
            "policy_id": continuation.policy_id,
        },
        "execution": checked["execution"],
        "collection": checked["collection"],
        "outcome_contract": contract.to_dict(),
        "outcome_contract_sha256": contract.fingerprint(),
        "primitive_onehot_order": checked["primitives"]["onehot_order"],
    }
    return {**payload, "resolved_identity_sha256": canonical_sha256(payload)}


def validate_resolved_method_v1_identity(
    value: Mapping[str, Any], *, source_config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "experiment_id",
        "source_config_sha256",
        "qwen",
        "executor_identity",
        "executor_identity_sha256",
        "serializer_id",
        "continuation",
        "execution",
        "collection",
        "outcome_contract",
        "outcome_contract_sha256",
        "primitive_onehot_order",
        "resolved_identity_sha256",
    }
    _exact_keys(value, expected=expected_keys, name="resolved Method-V1 identity")
    if value["schema_version"] != METHOD_V1_RESOLVED_SCHEMA:
        raise ValueError("unexpected resolved identity schema")
    body = {
        key: child for key, child in value.items() if key != "resolved_identity_sha256"
    }
    if canonical_sha256(body) != value["resolved_identity_sha256"]:
        raise ValueError("resolved Method-V1 identity digest mismatch")
    executor = MolmoAct2ServerIdentity.from_mapping(value["executor_identity"])
    if executor.digest != value["executor_identity_sha256"]:
        raise ValueError("resolved MolmoAct2 identity digest mismatch")
    contract = OutcomeContract.from_mapping(value["outcome_contract"])
    if contract.fingerprint() != value["outcome_contract_sha256"]:
        raise ValueError("resolved outcome contract digest mismatch")
    if source_config is not None:
        checked = validate_method_v1_config(source_config)
        regenerated = resolve_method_v1_identity(checked, executor_identity=executor)
        if regenerated != dict(value):
            raise ValueError("resolved identity is not reproducible from source config")
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--executor-identity", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = load_method_v1_config(args.config)
    raw_identity = _read_json(args.executor_identity)
    identity_payload = raw_identity.get("executor", raw_identity)
    executor_identity = MolmoAct2ServerIdentity.from_mapping(identity_payload)
    resolved = resolve_method_v1_identity(config, executor_identity=executor_identity)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite resolved identity: {output}")
    output.write_bytes(canonical_json_bytes(resolved) + b"\n")
    print(json.dumps(resolved, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
