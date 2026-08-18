from __future__ import annotations

import numpy as np

from interaction_uncertainty.candidates import DrawerV0CandidateGenerator
from interaction_uncertainty.contracts import (
    ActionEffectForecast,
    FactDistribution,
    ObjectNode,
    Primitive,
    ScenePacket,
    TaskBelief,
    UnknownRegion,
)
from interaction_uncertainty.optimizer import InformationUtilityOptimizer
from interaction_uncertainty.scene_memory import SceneMemory
from interaction_uncertainty.sidecar import class_conditional_set, fixed_project
from interaction_uncertainty.task_parser import FrozenRetrievalTaskParser


def belief(location, conformal):
    return TaskBelief(
        prompt="Place the butter in the basket",
        facts=(FactDistribution("target_location", location, 1.0),),
        node_uncertainty={"drawer_middle_interior": location["middle_drawer"]},
        model_stamp="test",
        conformal_sets={"target_location": conformal},
    )


def scene():
    return ScenePacket(
        frame_id="f0",
        prompt="Place the butter in the basket",
        objects=(
            ObjectNode("drawer_middle", {"drawer": 1.0}, affordances=frozenset({"openable"})),
            ObjectNode("prompt_target", {"butter": 1.0}, affordances=frozenset({"task_target"})),
        ),
        unknown_regions=(
            UnknownRegion(
                "drawer_middle_interior",
                "drawer_middle",
                "UNOBSERVED_CONTAINER_INTERIOR",
                0.0,
                Primitive.OPEN_TO_INSPECT,
            ),
        ),
        public_robot_state=(0.0,) * 8,
    )


def future(prompt, values):
    return TaskBelief(
        prompt=prompt,
        facts=(FactDistribution("target_location", values, 1.0),),
        node_uncertainty={},
        model_stamp="test",
    )


def test_hidden_prompt_generates_open_and_never_not_found():
    task = FrozenRetrievalTaskParser().parse("Place the butter in the basket")
    current = belief(
        {
            "visible_workspace": 0.05,
            "middle_drawer": 0.80,
            "other_unsearched_region": 0.10,
            "absent": 0.05,
        },
        ("middle_drawer",),
    )
    candidates = DrawerV0CandidateGenerator().generate(
        task=task, scene=scene(), belief=current, memory=SceneMemory()
    )
    assert {item.primitive for item in candidates} == {
        Primitive.OPEN_TO_INSPECT,
        Primitive.ABSTAIN,
    }


def test_empty_is_local_and_does_not_make_not_found_valid():
    task = FrozenRetrievalTaskParser().parse("Place the butter in the basket")
    memory = SceneMemory(searched_regions={"drawer_middle_interior"})
    current = belief(
        {
            "visible_workspace": 0.05,
            "middle_drawer": 0.01,
            "other_unsearched_region": 0.84,
            "absent": 0.10,
        },
        ("other_unsearched_region",),
    )
    candidates = DrawerV0CandidateGenerator().generate(
        task=task, scene=scene(), belief=current, memory=memory
    )
    assert [item.primitive for item in candidates] == [Primitive.ABSTAIN]


def test_optimizer_prefers_positive_information_value():
    task = FrozenRetrievalTaskParser().parse("Place the butter in the basket")
    current = belief(
        {
            "visible_workspace": 0.10,
            "middle_drawer": 0.70,
            "other_unsearched_region": 0.15,
            "absent": 0.05,
        },
        ("middle_drawer",),
    )
    candidates = DrawerV0CandidateGenerator().generate(
        task=task, scene=scene(), belief=current, memory=SceneMemory()
    )
    opening = next(item for item in candidates if item.primitive is Primitive.OPEN_TO_INSPECT)
    low = {
        "visible_workspace": 0.97,
        "middle_drawer": 0.01,
        "other_unsearched_region": 0.01,
        "absent": 0.01,
    }
    forecast = ActionEffectForecast(
        candidate_id=opening.candidate_id,
        outcome_probabilities={"FAILED": 0.05, "REVEALED": 0.70, "EMPTY": 0.25},
        future_beliefs={key: future(current.prompt, low) for key in ("FAILED", "REVEALED", "EMPTY")},
        execution_success_probability=0.95,
        expected_task_progress=0.4,
        model_stamp="test",
    )
    decision = InformationUtilityOptimizer().select(
        belief=current,
        candidates=candidates,
        forecasts={opening.candidate_id: forecast},
        learned_progress={},
    )
    assert decision.selected.primitive is Primitive.OPEN_TO_INSPECT
    assert decision.utilities[opening.candidate_id]["expected_information_gain"] > 0.0


def test_projection_and_conformal_are_deterministic():
    values = np.arange(24, dtype=np.float32).reshape(3, 8)
    np.testing.assert_array_equal(
        fixed_project(values, output_dimension=4, seed=3),
        fixed_project(values, output_dimension=4, seed=3),
    )
    assert fixed_project(values[0], output_dimension=4, seed=3).shape == (4,)
    assert class_conditional_set(
        np.asarray([0.1, 0.8, 0.05, 0.05]),
        labels=("a", "b", "c", "d"),
        thresholds={"a": 0.2, "b": 0.3, "c": -1.0, "d": -1.0},
    ) == ("b",)


def test_scene_packet_rejects_oracle_inputs():
    try:
        ScenePacket(
            frame_id="bad",
            prompt="Place the butter in the basket",
            objects=(),
            unknown_regions=(),
            public_robot_state=(),
            online_oracle_inputs=("segmentation",),
        )
    except ValueError as error:
        assert "oracle" in str(error)
    else:
        raise AssertionError("oracle input was accepted")
