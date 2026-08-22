from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from calibrated_interaction.calibration import (
    BinaryEffectCalibration,
    LACCalibrator,
    TemperatureScaler,
    softmax,
)
from calibrated_interaction.capabilities import CapabilityRegistry
from calibrated_interaction.contracts import CandidateAction, EffectFactor, Primitive
from calibrated_interaction.controller import (
    CalibratedSelector,
    DecisionKind,
    ModelPrediction,
)
from calibrated_interaction.data import (
    CounterfactualSample,
    assert_policy_input_clean,
    validate_group_splits,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/original_drawer_calibrated_replay.json"


def _candidates() -> tuple[CandidateAction, ...]:
    return (
        CandidateAction(
            "direct", Primitive.DIRECT, "the requested package", "the basket"
        ),
        CandidateAction(
            "open", Primitive.OPEN, "the middle drawer", purpose="inspect inside"
        ),
        CandidateAction("stop", Primitive.STOP, None),
    )


def _fixture_selector() -> CalibratedSelector:
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    route = dict(value["route_calibration"])
    route["labels"] = ["direct", "open", "stop"]
    return CalibratedSelector(
        LACCalibrator.from_dict(route),
        BinaryEffectCalibration.from_dict(value["effect_calibration"]),
    )


def _effects(*, useful_open: bool = True) -> dict[str, dict[str, float]]:
    base = {factor.value: 0.05 for factor in EffectFactor}
    direct = {**base, EffectFactor.EXECUTION_SUCCEEDED.value: 0.92}
    opened = {
        **base,
        EffectFactor.EXECUTION_SUCCEEDED.value: 0.94,
        EffectFactor.AMBIGUITY_REDUCED.value: 0.93 if useful_open else 0.50,
    }
    stop = {**base, EffectFactor.EXECUTION_SUCCEEDED.value: 0.99}
    return {"direct": direct, "open": opened, "stop": stop}


def test_capability_registry_rejects_bad_grounding_and_serializes_only_subtask() -> (
    None
):
    registry = CapabilityRegistry.load(ROOT / "configs/capabilities/pi05_libero.yaml")
    candidate = CandidateAction("open", Primitive.OPEN, "the upper cabinet")
    assert (
        registry.serialize(candidate, task_prompt="Find the tea")
        == "Open the upper cabinet."
    )
    with pytest.raises(ValueError, match="STOP"):
        CandidateAction("stop", Primitive.STOP, "a hidden target")
    with pytest.raises(ValueError, match="normalized"):
        CandidateAction(
            "open", Primitive.OPEN, "drawer", reference_region_xyxy=(0, 0, 4, 5)
        )


def test_temperature_scaling_is_fitted_on_nll_not_verbal_confidence() -> None:
    logits = np.asarray([[8.0, 0.0], [7.0, 0.0], [6.0, 0.0], [5.0, 0.0]])
    labels = np.asarray([0, 0, 1, 1])
    before = -np.log(softmax(logits)[np.arange(4), labels]).mean()
    scaler = TemperatureScaler.fit(logits, labels)
    after = -np.log(scaler.probabilities(logits)[np.arange(4), labels]).mean()
    assert scaler.temperature > 1.0
    assert after < before


def test_lac_can_abstain_on_empty_set_instead_of_inventing_a_top_one() -> None:
    calibrator = LACCalibrator(
        alpha=0.1,
        threshold=0.1,
        labels=("a", "b", "c"),
        calibration_size=100,
        split_id="held-out",
    )
    assert calibrator.predict({"a": 0.4, "b": 0.35, "c": 0.25}) == ()


def test_effect_bonferroni_protects_only_decision_critical_family() -> None:
    probabilities = np.full((24, len(EffectFactor)), 0.9)
    labels = np.ones_like(probabilities, dtype=np.int64)
    calibration = BinaryEffectCalibration.fit(
        probabilities,
        labels,
        joint_alpha=0.1,
        split_id="held-out",
        decision_factors=(
            EffectFactor.EXECUTION_SUCCEEDED,
            EffectFactor.AMBIGUITY_REDUCED,
        ),
    )
    assert calibration.calibrators[EffectFactor.EXECUTION_SUCCEEDED].alpha == 0.05
    assert calibration.calibrators[EffectFactor.AMBIGUITY_REDUCED].alpha == 0.05
    assert calibration.calibrators[EffectFactor.TARGET_CONFIRMED].alpha == 0.1


def test_selector_executes_singleton_route_only_with_reliable_effect() -> None:
    selector = _fixture_selector()
    prediction = ModelPrediction(
        candidate_ids=("direct", "open", "stop"),
        route_probabilities={"direct": 0.1, "open": 0.8, "stop": 0.1},
        effect_positive_probabilities=_effects(),
    )
    decision = selector.select(_candidates(), prediction)
    assert decision.kind is DecisionKind.EXECUTE
    assert decision.candidate_id == "open"

    unreliable = ModelPrediction(
        candidate_ids=prediction.candidate_ids,
        route_probabilities=prediction.route_probabilities,
        effect_positive_probabilities=_effects(useful_open=False),
    )
    assert selector.select(_candidates(), unreliable).kind is DecisionKind.ABSTAIN


def test_selector_can_use_unique_calibrated_information_action_in_multi_set() -> None:
    selector = _fixture_selector()
    selector.route_calibrator = LACCalibrator(
        alpha=0.1,
        threshold=0.7,
        labels=("direct", "open", "stop"),
        calibration_size=100,
        split_id="held-out",
    )
    prediction = ModelPrediction(
        candidate_ids=("direct", "open", "stop"),
        route_probabilities={"direct": 0.38, "open": 0.42, "stop": 0.20},
        effect_positive_probabilities=_effects(),
    )
    decision = selector.select(_candidates(), prediction)
    assert len(decision.route_prediction_set) == 2
    assert decision.candidate_id == "open"


def test_policy_projection_excludes_privileged_evaluator_metadata() -> None:
    sample = CounterfactualSample.from_mapping(
        {
            "episode_id": "episode-1",
            "initial_state_id": "state-1",
            "prompt": "Place the requested package in the basket",
            "observation_frames": ["before.png"],
            "history": [],
            "candidate_actions": [candidate.to_dict() for candidate in _candidates()],
            "executed_candidate": "open",
            "post_action_frames": ["after.png"],
            "effect_labels": {factor.value: False for factor in EffectFactor},
            "route_label": "open",
            "task_success": False,
            "privileged_metadata_for_evaluation_only": {
                "semantic_ids": [4, 7],
                "target_pose": [0.1, 0.2, 0.3],
            },
        }
    )
    assert "privileged_metadata_for_evaluation_only" not in sample.policy_input()
    with pytest.raises(ValueError, match="privileged"):
        assert_policy_input_clean({"history": [{"sim_state": [1, 2]}]})


def test_counterfactual_branches_cannot_cross_splits() -> None:
    validate_group_splits(
        [
            {"initial_state_id": "s1", "split": "train"},
            {"initial_state_id": "s1", "split": "train"},
            {"initial_state_id": "s2", "split": "calibration"},
        ]
    )
    with pytest.raises(ValueError, match="counterfactual leakage"):
        validate_group_splits(
            [
                {"initial_state_id": "s1", "split": "train"},
                {"initial_state_id": "s1", "split": "test"},
            ]
        )


def test_original_drawer_replay_is_open_reobserve_direct(tmp_path: Path) -> None:
    output = tmp_path / "trace.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/pipeline/run_calibrated_replay.py"),
            "--scenario",
            str(ROOT / "configs/scenarios/original_drawer.yaml"),
            "--replay",
            str(FIXTURE),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    trace = json.loads(output.read_text(encoding="utf-8"))
    assert trace["terminal"] == "BACKEND_TERMINAL"
    assert [step["decision"]["candidate_id"] for step in trace["steps"]] == [
        "open_middle_drawer",
        "direct_butter_to_basket",
    ]
    assert trace["online_oracle_inputs"] == []
    assert "not model or task-success evidence" in trace["validation_scope"]
