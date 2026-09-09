from __future__ import annotations

import copy
import dataclasses

import pytest

from grounded_interaction.psr.config import (
    DEFAULT_PSR_CONFIG,
    PSRConfig,
    load_psr_config,
    validate_native_architecture,
)
from grounded_interaction.psr.types import (
    ExecutedIntent,
    IntentCandidate,
    PublicHistory,
    PublicObservation,
    RGBReference,
    advance_public_history,
    candidate_seed,
)


def _rgb(name: str, character: str) -> RGBReference:
    return RGBReference(name, character * 64, 256, 256)


def _observation(step: int) -> PublicObservation:
    char = "a" if step == 0 else ("b" if step == 10 else "c")
    return PublicObservation(
        agentview_rgb=_rgb(f"agent-{step}", char),
        wrist_rgb=_rgb(f"wrist-{step}", char),
        robot_state=(0.0,) * 8,
        control_step=step,
    )


def _history() -> PublicHistory:
    return PublicHistory(
        task="Put the butter in the basket.",
        current=_observation(0),
        previous=(),
        executed=(),
        remaining_control_steps=300,
    )


def test_real_yaml_is_the_exact_frozen_v1_config() -> None:
    loaded = load_psr_config("experiments/psr_v1.yaml")
    assert loaded.to_dict() == DEFAULT_PSR_CONFIG
    assert loaded.fingerprint == PSRConfig(DEFAULT_PSR_CONFIG).fingerprint


def test_changed_v1_protocol_requires_a_new_identity() -> None:
    changed = copy.deepcopy(DEFAULT_PSR_CONFIG)
    changed["execution"]["intent_window_steps"] = 49
    with pytest.raises(ValueError, match="frozen V1"):
        PSRConfig(changed)


def test_native_dimensions_are_discovered_and_checked() -> None:
    config = PSRConfig(DEFAULT_PSR_CONFIG)
    architecture = validate_native_architecture(
        config,
        {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_key_value_heads": 8,
            "action_horizon": 10,
            "action_dim": 7,
        },
    )
    assert architecture.hidden_size == 2048
    with pytest.raises(ValueError, match="action horizon"):
        validate_native_architecture(
            config, dataclasses.replace(architecture, action_horizon=5).__dict__
        )


def test_public_observation_requires_exactly_eight_finite_state_values() -> None:
    with pytest.raises(ValueError, match="exactly 8"):
        dataclasses.replace(_observation(0), robot_state=(0.0,) * 7)
    with pytest.raises(ValueError, match="finite"):
        dataclasses.replace(_observation(0), robot_state=(0.0,) * 7 + (float("nan"),))


def test_public_history_has_no_evaluator_or_future_fields() -> None:
    payload = _history().to_dict()
    rendered = str(payload).lower()
    for forbidden in (
        "reset",
        "scene_id",
        "future",
        "reward",
        "task_success",
        "oracle",
    ):
        assert forbidden not in rendered
    assert set(payload) == {
        "task",
        "current",
        "previous",
        "executed",
        "remaining_control_steps",
    }


def test_history_rejects_more_than_two_previous_boundaries() -> None:
    with pytest.raises(ValueError, match="at most two"):
        PublicHistory(
            task="Move the object.",
            current=_observation(30),
            previous=(_observation(0), _observation(10), _observation(20)),
            executed=(),
            remaining_control_steps=270,
        )


def test_candidate_identity_and_seed_do_not_depend_on_list_position() -> None:
    first = IntentCandidate("Open the drawer", (10, 11), "conditioned")
    same = IntentCandidate("  Open   the drawer  ", (10, 11), "conditioned")
    native = IntentCandidate("Open the drawer", (10, 11), "native")
    assert first.candidate_id == same.candidate_id
    assert first.candidate_id != native.candidate_id
    assert candidate_seed(17, 42, 50, first) == candidate_seed(17, 42, 50, same)


def test_history_advances_only_real_observations_and_fixed_boundaries() -> None:
    event = ExecutedIntent(
        text="Inspect the label",
        execution_route="conditioned",
        start_step=0,
        end_step=10,
        actions_ref="actions-sha256-abc",
        public_status="completed",
    )
    after_chunk = advance_public_history(
        _history(),
        _observation(10),
        remaining_control_steps=290,
        completed_intent=event,
        high_level_boundary=False,
    )
    assert after_chunk.previous == ()
    assert after_chunk.executed == (event,)
    at_boundary = advance_public_history(
        after_chunk,
        _observation(20),
        remaining_control_steps=280,
        high_level_boundary=True,
        boundary_observation=_observation(0),
    )
    assert at_boundary.previous == (_observation(0),)
    assert PublicHistory.from_mapping(at_boundary.to_dict()) == at_boundary
