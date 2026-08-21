import json

import pytest

from interaction_uncertainty.vlm_reasoner import (
    SemanticAction,
    SemanticAssessment,
    extract_json_object,
    select_semantic_action,
    validate_executor_subtask,
)


BOX = "wrist_food_package_00"
DRAWER = "agentview_drawer_00"
BUTTER = "wrist_butter_00"


def region(
    object_id: str,
    *,
    target: float,
    similar: float,
    insufficient: float,
    resolution: float,
    graspable: float = 0.05,
) -> dict:
    other = 1.0 - target - similar - insufficient
    return {
        "object_id": object_id,
        "prompt_relevance": 0.95,
        "identity_belief": {
            "target": target,
            "visually_similar_non_target": similar,
            "other": other,
            "insufficient_visual_evidence": insufficient,
        },
        "graspability_belief": {
            "GRASPABLE": graspable,
            "NOT_GRASPABLE": 0.05,
            "INSUFFICIENT_EVIDENCE": 0.95 - graspable,
        },
        "resolution_uncertainty": resolution,
        "occlusion_uncertainty": 0.10,
        "state_uncertainty": 0.05,
        "move_closer_effect_probability": 0.90,
        "reason": "target-like food package",
    }


def option(
    action: str,
    target_id: str,
    *,
    posterior: float,
    progress: float,
    cost: float,
    success: float = 0.90,
) -> dict:
    return {
        "action": action,
        "target_id": target_id,
        "applicable_probability": 0.95,
        "execution_success_probability": success,
        "outcome_distribution": {
            "FAILED": 0.10,
            "IDENTITY_RESOLVED" if action == "MOVE_CLOSER" else "TARGET_REVEALED": 0.90,
        },
        "expected_posterior_uncertainty": posterior,
        "expected_task_progress": progress,
        "normalized_cost": cost,
        "normalized_risk": 0.01,
        "semantic_subtask": f"execute {action} on {target_id}",
        "reason": f"counterfactual {action}",
    }


def parse(value: dict, ids: list[str]) -> SemanticAssessment:
    value.setdefault("destination_query", "basket")
    value.setdefault("goal_relation", "inside")
    value.setdefault(
        "required_facts", ["target_identity", "target_location", "target_accessibility"]
    )
    return SemanticAssessment.from_mapping(value, allowed_object_ids=ids)


def scene(*ids: str) -> list[dict]:
    return [
        {"object_id": object_id, "visible_area": 900, "bbox_xyxy": [10, 10, 50, 50]}
        for object_id in ids
    ]


def test_visible_ambiguous_package_selects_move_closer_before_drawer() -> None:
    assessment = parse(
        {
            "target_query": "butter",
            "target_location_belief": {
                BOX: 0.50,
                DRAWER: 0.35,
                "OTHER_UNSEARCHED": 0.10,
                "ABSENT": 0.05,
            },
            "regions": [
                region(BOX, target=0.45, similar=0.40, insufficient=0.10, resolution=0.75)
            ],
            "unobserved_regions": [
                {
                    "object_id": DRAWER,
                    "target_probability": 0.35,
                    "inspectability": 0.90,
                    "reason": "closed drawer",
                }
            ],
            "interaction_options": [
                option("MOVE_CLOSER", BOX, posterior=0.25, progress=0.03, cost=0.05),
                option("OPEN_CONTAINER", DRAWER, posterior=0.45, progress=0.05, cost=0.24),
            ],
            "advisory_action": "MOVE_CLOSER",
            "advisory_target_id": BOX,
            "summary": "food box is ambiguous; drawer remains unsearched",
        },
        [BOX, DRAWER],
    )
    decision = select_semantic_action(assessment, scene_objects=scene(BOX, DRAWER))
    assert decision.action is SemanticAction.MOVE_CLOSER
    assert decision.target_id == BOX


def test_resolved_cream_cheese_shifts_belief_and_selects_open() -> None:
    assessment = parse(
        {
            "target_query": "butter",
            "target_location_belief": {
                BOX: 0.02,
                DRAWER: 0.83,
                "OTHER_UNSEARCHED": 0.05,
                "ABSENT": 0.10,
            },
            "regions": [
                region(BOX, target=0.01, similar=0.97, insufficient=0.01, resolution=0.05)
            ],
            "unobserved_regions": [
                {
                    "object_id": DRAWER,
                    "target_probability": 0.83,
                    "inspectability": 0.95,
                    "reason": "only high-probability registered region remains",
                }
            ],
            "interaction_options": [
                option("MOVE_CLOSER", BOX, posterior=0.52, progress=0.00, cost=0.05),
                option("OPEN_CONTAINER", DRAWER, posterior=0.12, progress=0.20, cost=0.20),
            ],
            "advisory_action": "OPEN_CONTAINER",
            "advisory_target_id": DRAWER,
            "summary": "visible box is cream cheese; inspect drawer",
        },
        [BOX, DRAWER],
    )
    decision = select_semantic_action(assessment, scene_objects=scene(BOX, DRAWER))
    assert decision.action is SemanticAction.OPEN_CONTAINER
    assert decision.target_id == DRAWER


def test_revealed_graspable_target_selects_act() -> None:
    butter = region(
        BUTTER,
        target=0.94,
        similar=0.02,
        insufficient=0.02,
        resolution=0.08,
        graspable=0.92,
    )
    butter["graspability_belief"] = {
        "GRASPABLE": 0.92,
        "NOT_GRASPABLE": 0.04,
        "INSUFFICIENT_EVIDENCE": 0.04,
    }
    act = option("ACT", BUTTER, posterior=0.02, progress=0.95, cost=0.08)
    act["outcome_distribution"] = {"FAILED": 0.10, "TASK_COMPLETED": 0.90}
    assessment = parse(
        {
            "target_query": "butter",
            "target_location_belief": {
                BUTTER: 0.92,
                "OTHER_UNSEARCHED": 0.03,
                "ABSENT": 0.05,
            },
            "regions": [butter],
            "unobserved_regions": [],
            "interaction_options": [act],
            "advisory_action": "ACT",
            "advisory_target_id": BUTTER,
            "summary": "butter is clear and graspable",
        },
        [BUTTER],
    )
    decision = select_semantic_action(assessment, scene_objects=scene(BUTTER))
    assert decision.action is SemanticAction.ACT


def test_stop_is_abstain_while_registered_region_remains_unsearched() -> None:
    assessment = parse(
        {
            "target_query": "butter",
            "target_location_belief": {
                DRAWER: 0.35,
                "OTHER_UNSEARCHED": 0.20,
                "ABSENT": 0.45,
            },
            "regions": [],
            "unobserved_regions": [
                {
                    "object_id": DRAWER,
                    "target_probability": 0.35,
                    "inspectability": 0.95,
                    "reason": "drawer remains unsearched",
                }
            ],
            "interaction_options": [],
            "advisory_action": "STOP",
            "advisory_target_id": None,
            "summary": "no physical action was proposed",
        },
        [DRAWER],
    )
    decision = select_semantic_action(assessment, scene_objects=scene(DRAWER))
    assert decision.action is SemanticAction.STOP
    assert decision.stop_reason == "ABSTAIN"


def test_stop_reports_not_found_only_after_exhaustion_and_absent_argmax() -> None:
    assessment = parse(
        {
            "target_query": "butter",
            "target_location_belief": {
                DRAWER: 0.05,
                "OTHER_UNSEARCHED": 0.10,
                "ABSENT": 0.85,
            },
            "regions": [],
            "unobserved_regions": [],
            "interaction_options": [],
            "advisory_action": "STOP",
            "advisory_target_id": None,
            "summary": "registered search domain is exhausted",
            "search_domain_exhausted": True,
        },
        [DRAWER],
    )
    decision = select_semantic_action(assessment, scene_objects=scene(DRAWER))
    assert decision.action is SemanticAction.STOP
    assert decision.stop_reason == "NOT_FOUND"


def test_stop_pays_unresolved_task_failure_cost() -> None:
    move = option("MOVE_CLOSER", BUTTER, posterior=0.55, progress=0.0, cost=0.20)
    move["outcome_distribution"] = {
        "FAILED": 0.20,
        "IDENTITY_RESOLVED": 0.80,
    }
    assessment = parse(
        {
            "target_query": "butter",
            "target_location_belief": {
                BUTTER: 0.50,
                "OTHER_UNSEARCHED": 0.40,
                "ABSENT": 0.10,
            },
            "regions": [
                region(
                    BUTTER,
                    target=0.40,
                    similar=0.35,
                    insufficient=0.20,
                    resolution=0.80,
                    graspable=0.10,
                )
            ],
            "unobserved_regions": [],
            "interaction_options": [move],
            "advisory_action": "MOVE_CLOSER",
            "advisory_target_id": BUTTER,
            "summary": "target-relevant evidence remains unresolved",
        },
        [BUTTER],
    )
    decision = select_semantic_action(assessment, scene_objects=scene(BUTTER))
    assert decision.stop_utility == pytest.approx(-decision.task_uncertainty)
    assert decision.option_utilities[0]["utility"] < 0.0
    assert decision.option_utilities[0]["utility"] > decision.stop_utility
    assert decision.action is SemanticAction.MOVE_CLOSER


def test_json_extractor_rejects_trailing_prose() -> None:
    assert extract_json_object("```json\n{\"a\": 1}\n```") == {"a": 1}
    with pytest.raises(ValueError):
        extract_json_object(json.dumps({"a": 1}) + " explanation")


def test_non_executable_vlm_subtask_falls_back_to_registered_hint() -> None:
    subtask, source = validate_executor_subtask(
        action=SemanticAction.MOVE_CLOSER,
        generated="Observation remains unchanged.",
        registered_hint="Move the wrist camera closer to region W4 and observe it clearly.",
    )
    assert subtask.startswith("Move the wrist camera closer")
    assert source == "registered_hint_repaired_non_executable_qwen_text"


def test_executable_grounded_act_subtask_is_retained() -> None:
    subtask, source = validate_executor_subtask(
        action=SemanticAction.ACT,
        generated="Pick up the visible butter package and place it in the basket.",
        registered_hint="Place the butter in the basket.",
    )
    assert subtask.startswith("Pick up the visible butter")
    assert source == "qwen_generated_schema_valid"
