"""Counterfactual sample schema and policy-input leakage firewall."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import CandidateAction, EffectFactor, validate_candidate_set

_FORBIDDEN_POLICY_FRAGMENTS = (
    "privileged",
    "semantic_id",
    "segmentation",
    "ground_truth",
    "target_pose",
    "object_pose",
    "sim_state",
    "task_predicate",
    "joint_qpos",
)


def assert_policy_input_clean(value: Any, *, path: str = "policy_input") -> None:
    """Reject evaluator-private fields recursively before model tokenization."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if any(fragment in normalized for fragment in _FORBIDDEN_POLICY_FRAGMENTS):
                raise ValueError(f"privileged policy field at {path}.{key}")
            assert_policy_input_clean(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            assert_policy_input_clean(child, path=f"{path}[{index}]")


@dataclasses.dataclass(frozen=True)
class CounterfactualSample:
    """One candidate fork; privileged metadata is evaluator-only by construction."""

    episode_id: str
    initial_state_id: str
    prompt: str
    observation_frames: tuple[str, ...]
    history: tuple[Mapping[str, Any], ...]
    candidate_actions: tuple[CandidateAction, ...]
    executed_candidate: str
    post_action_frames: tuple[str, ...]
    effect_labels: Mapping[EffectFactor, bool]
    route_label: str
    task_success: bool
    privileged_metadata_for_evaluation_only: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CounterfactualSample:
        candidates = validate_candidate_set(
            [
                CandidateAction.from_mapping(row)
                for row in value.get("candidate_actions", ())
            ]
        )
        identifiers = {candidate.candidate_id for candidate in candidates}
        executed = str(value.get("executed_candidate", ""))
        route_label = str(value.get("route_label", ""))
        if executed not in identifiers or route_label not in identifiers:
            raise ValueError(
                "executed_candidate and route_label must reference candidate IDs"
            )
        labels = {
            EffectFactor(name): bool(label)
            for name, label in value.get("effect_labels", {}).items()
        }
        if set(labels) != set(EffectFactor):
            raise ValueError("effect_labels must contain every effect factor")
        sample = cls(
            episode_id=str(value.get("episode_id", "")),
            initial_state_id=str(value.get("initial_state_id", "")),
            prompt=str(value.get("prompt", "")),
            observation_frames=tuple(
                str(path) for path in value.get("observation_frames", ())
            ),
            history=tuple(value.get("history", ())),
            candidate_actions=candidates,
            executed_candidate=executed,
            post_action_frames=tuple(
                str(path) for path in value.get("post_action_frames", ())
            ),
            effect_labels=labels,
            route_label=route_label,
            task_success=bool(value.get("task_success", False)),
            privileged_metadata_for_evaluation_only=dict(
                value.get("privileged_metadata_for_evaluation_only", {})
            ),
        )
        if (
            not sample.episode_id
            or not sample.initial_state_id
            or not sample.prompt.strip()
        ):
            raise ValueError("episode_id, initial_state_id, and prompt are required")
        if not sample.observation_frames or not sample.post_action_frames:
            raise ValueError("pre- and post-action public frames are required")
        assert_policy_input_clean(sample.policy_input())
        return sample

    def policy_input(self) -> dict[str, Any]:
        value = {
            "prompt": self.prompt,
            "observation_frames": list(self.observation_frames),
            "history": [dict(row) for row in self.history],
            "candidate_actions": [
                candidate.to_dict() for candidate in self.candidate_actions
            ],
        }
        assert_policy_input_clean(value)
        return value

    def supervision(self) -> dict[str, Any]:
        return {
            "executed_candidate": self.executed_candidate,
            "post_action_frames": list(self.post_action_frames),
            "effect_labels": {
                factor.value: label for factor, label in self.effect_labels.items()
            },
            "route_label": self.route_label,
            "task_success": self.task_success,
        }


def validate_group_splits(rows: Sequence[Mapping[str, Any]]) -> None:
    """Keep every counterfactual branch of an initial state in one split."""

    state_to_split: dict[str, str] = {}
    for row in rows:
        state_id, split = (
            str(row.get("initial_state_id", "")),
            str(row.get("split", "")),
        )
        if not state_id or not split:
            raise ValueError("split rows require initial_state_id and split")
        previous = state_to_split.setdefault(state_id, split)
        if previous != split:
            raise ValueError(
                f"counterfactual leakage: state {state_id!r} occurs in {previous} and {split}"
            )
