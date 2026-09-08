from __future__ import annotations

import json
from pathlib import Path

import pytest

from grounded_interaction.method_v1_config import (
    load_method_v1_config,
    resolve_method_v1_identity,
    validate_method_v1_config,
    validate_resolved_method_v1_identity,
)
from grounded_interaction.molmoact2 import MolmoAct2ServerIdentity

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "method_v1.yaml"
E1_PLAN = ROOT / "experiments" / "e1_referent_ceiling" / "pilot_state0_v1.json"


def test_method_v1_config_freezes_one_decision_and_fixed_continuation() -> None:
    config = load_method_v1_config(CONFIG)
    assert config["primitives"]["enabled"] == ["DIRECT", "OPEN"]
    assert config["execution"]["direct_steps"] == 300
    assert (
        config["execution"]["open_steps"]
        + config["execution"]["continuation_direct_steps"]
        == 300
    )
    assert config["continuation"]["learned_scorer_calls"] == 0
    assert (
        config["outcome_contract"]["outcome_name"]
        == "full_task_with_fixed_continuation_v1"
    )


def test_config_rejects_private_switch_and_e1_contact_outcome() -> None:
    config = load_method_v1_config(CONFIG)
    config["execution"]["switch_rule"] = "switch_when_private_drawer_open"
    with pytest.raises(ValueError, match="private"):
        validate_method_v1_config(config)

    config = load_method_v1_config(CONFIG)
    config["outcome_contract"]["outcome_name"] = "first_contact_intended"
    with pytest.raises(ValueError, match="E1"):
        validate_method_v1_config(config)


def test_resolved_identity_is_reproducible_and_tamper_evident() -> None:
    config = load_method_v1_config(CONFIG)
    plan = json.loads(E1_PLAN.read_text(encoding="utf-8"))
    executor = MolmoAct2ServerIdentity.from_mapping(plan["executor"])
    resolved = resolve_method_v1_identity(config, executor_identity=executor)
    assert (
        validate_resolved_method_v1_identity(resolved, source_config=config)[
            "resolved_identity_sha256"
        ]
        == resolved["resolved_identity_sha256"]
    )
    resolved["execution"]["open_steps"] = 99
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_resolved_method_v1_identity(resolved)
