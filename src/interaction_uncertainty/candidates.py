"""Hard-valid candidate generation from public scene state and memory."""

from __future__ import annotations

from .contracts import CandidateAction, Primitive, ScenePacket, TaskBelief, TaskSpec
from .scene_memory import SceneMemory


class DrawerV0CandidateGenerator:
    """Registered T01 action schema; it contains no hidden target state."""

    def generate(
        self,
        *,
        task: TaskSpec,
        scene: ScenePacket,
        belief: TaskBelief,
        memory: SceneMemory,
    ) -> tuple[CandidateAction, ...]:
        location = belief.fact("target_location")
        plausible = set(belief.conformal_sets.get("target_location", ()))
        objects = {node.object_id: node for node in scene.objects}
        candidates: list[CandidateAction] = []
        target_node = objects.get("prompt_target")
        if target_node is not None and "visible_workspace" in plausible:
            candidates.append(
                CandidateAction(
                    candidate_id="direct_act:prompt_target",
                    primitive=Primitive.DIRECT_ACT,
                    target_id="prompt_target",
                    subtask=f"Pick up the {task.target}",
                    stop_condition="target is grasped or the action visibly fails",
                    cost=0.10,
                    physical_risk=0.10,
                )
            )
        for region in scene.unknown_regions:
            if (
                region.accessible_via is Primitive.OPEN_TO_INSPECT
                and region.region_id not in memory.searched_regions
                and "middle_drawer" in plausible
                and memory.attempt_counts[
                    f"open_to_inspect:{region.parent_object_id}"
                ]
                < 1
            ):
                candidates.append(
                    CandidateAction(
                        candidate_id=f"open_to_inspect:{region.parent_object_id}",
                        primitive=Primitive.OPEN_TO_INSPECT,
                        target_id=region.parent_object_id,
                        subtask="Open the middle layer of the drawer",
                        stop_condition="the opening attempt ends and six public observations are available",
                        cost=0.18,
                        physical_risk=0.05,
                    )
                )
        if scene.unknown_regions and all(
            region.region_id in memory.searched_regions for region in scene.unknown_regions
        ) and "other_unsearched_region" not in plausible:
            candidates.append(
                CandidateAction(
                    candidate_id="stop_not_found",
                    primitive=Primitive.STOP_NOT_FOUND,
                    target_id=None,
                    subtask="Report that the target was not found",
                    stop_condition="all registered relevant regions are visually certified searched",
                    cost=0.0,
                    physical_risk=0.0,
                )
            )
        candidates.append(
            CandidateAction(
                candidate_id="abstain",
                primitive=Primitive.ABSTAIN,
                target_id=None,
                subtask="Abstain safely",
                stop_condition="no reliable hard-valid action is available",
                cost=0.0,
                physical_risk=0.0,
            )
        )
        return tuple(candidates)
