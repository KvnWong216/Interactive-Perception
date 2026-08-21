import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/pipeline"))
sys.path.insert(0, str(ROOT / "src"))

from infer import (  # noqa: E402
    link_revealed_target_tracks,
    registered_action_candidates,
    update_belief_with_revealed_tracks,
)


ACTION_REGISTRY = {
    action: {"execution_success": 0.8, "cost": 0.1, "risk": 0.05}
    for action in (
        "MOVE_CLOSER",
        "NEXT_BEST_VIEW",
        "REMOVE_OCCLUDER",
        "OPEN_CONTAINER",
        "ACT",
    )
}


def region(object_id: str, view: str, relevance: float, target: float) -> dict:
    return {
        "object_id": object_id,
        "view": view,
        "identity_belief": {
            "target": target,
            "visually_similar_non_target": 0.1,
            "insufficient_visual_evidence": 0.1,
            "other": 0.8 - target,
        },
        "graspability_belief": {
            "GRASPABLE": 0.1,
            "NOT_GRASPABLE": 0.4,
            "INSUFFICIENT_EVIDENCE": 0.5,
        },
        "prompt_relevance": relevance,
        "identity_entropy": 0.7,
        "resolution_uncertainty": 0.8,
        "occlusion_uncertainty": 0.2,
        "state_uncertainty": 0.0,
        "uncertainty_mass": 0.6,
    }


def test_revealed_track_binds_new_cross_view_package_without_oracle() -> None:
    wrist_target = region("wrist_package", "wrist", 0.45, 0.55)
    wrist_old = region("wrist_handle", "wrist", 1.0, 0.70)
    agent_target = region("agent_package", "agentview", 0.95, 0.90)
    field = {
        "task_spec": {"target": "butter", "prompt": "Place butter in basket"},
        "regions": [wrist_target, wrist_old, agent_target],
        "unobserved_regions": [
            {
                "object_id": "drawer",
                "target_probability": 0.2,
                "inspectability": 0.9,
            }
        ],
        "target_location_belief": {
            "wrist_package": 0.15,
            "wrist_handle": 0.35,
            "agent_package": 0.30,
            "drawer": 0.18,
            "ABSENT": 0.02,
        },
        "task_uncertainty": 0.8,
    }
    nodes = [
        {
            "object_id": "wrist_package",
            "display_id": "W04",
            "view": "wrist",
            "bbox_xyxy": [70, 0, 100, 30],
            "visible_area": 600,
            "label_candidates": {"food package": 0.6, "butter": 0.4},
            "cross_view_best_match": {
                "object_id": "agent_package",
                "dino_cosine_similarity": 0.6,
            },
        },
        {
            "object_id": "wrist_handle",
            "display_id": "W08",
            "view": "wrist",
            "bbox_xyxy": [60, 120, 75, 135],
            "visible_area": 100,
            "label_candidates": {"butter": 1.0},
            "temporal_association_score": 0.95,
            "cross_view_best_match": {
                "object_id": "agent_package",
                "dino_cosine_similarity": 0.5,
            },
        },
        {
            "object_id": "agent_package",
            "display_id": "A08",
            "view": "agentview",
            "bbox_xyxy": [110, 120, 140, 150],
            "visible_area": 500,
            "label_candidates": {"butter": 1.0},
        },
        {
            "object_id": "drawer",
            "display_id": "A09",
            "view": "agentview",
            "bbox_xyxy": [0, 70, 220, 220],
            "visible_area": 10000,
            "label_candidates": {"drawer": 1.0},
        },
    ]
    tracks = link_revealed_target_tracks(
        field=field,
        scene_objects=nodes,
        executed_action="OPEN_CONTAINER",
        public_outcome="EVIDENCE_ACQUIRED",
    )
    assert tracks[0]["wrist_object_id"] == "wrist_package"
    updated = update_belief_with_revealed_tracks(field, tracks)
    assert max(
        updated["target_location_belief"],
        key=updated["target_location_belief"].get,
    ) == "wrist_package"
    candidates = registered_action_candidates(
        field=updated,
        scene_objects=nodes,
        action_registry=ACTION_REGISTRY,
        confirmed_track_ids=["wrist_package"],
    )
    target_pairs = {
        (row["action"], row["target_id"])
        for row in candidates
        if row["action"] in {"MOVE_CLOSER", "NEXT_BEST_VIEW", "ACT"}
    }
    assert target_pairs == {
        ("MOVE_CLOSER", "wrist_package"),
        ("NEXT_BEST_VIEW", "wrist_package"),
        ("ACT", "wrist_package"),
    }
    assert not any(row["action"] == "OPEN_CONTAINER" for row in candidates)
