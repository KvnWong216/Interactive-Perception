"""Leakage-controlled public transition and evaluator-sidecar contracts."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any


class Split(str, Enum):
    """Dataset roles; retrospective evidence can never become a test split."""

    TRAIN = "train"
    DEVELOPMENT = "development"
    RETROSPECTIVE_DEVELOPMENT = "retrospective_development"
    CALIBRATION = "calibration"
    SEALED_TEST = "sealed_test"


_FORBIDDEN_POLICY_KEYS = (
    "privileged",
    "evaluator",
    "segmentation",
    "semantic_id",
    "instance_id",
    "target_pose",
    "object_pose",
    "ground_truth",
    "task_predicate",
    "joint_qpos",
    "oracle",
    "target_mask",
)


def _clean_text(value: Any, *, name: str) -> str:
    result = " ".join(str(value or "").split())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def assert_public_policy_value(value: Any, *, path: str = "policy_input") -> None:
    """Recursively reject evaluator-only fields before model tokenization."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if any(fragment in normalized for fragment in _FORBIDDEN_POLICY_KEYS):
                raise ValueError(f"evaluator-only policy field at {path}.{key}")
            assert_public_policy_value(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            assert_public_policy_value(child, path=f"{path}[{index}]")


@dataclasses.dataclass(frozen=True)
class PublicTransition:
    """Only information that a deployed PIU model is permitted to consume."""

    sample_id: str
    initial_state_group: str
    split: Split
    prompt: str
    observations: Mapping[str, Any]
    public_action_history: Mapping[str, Any]
    candidate_actions: tuple[Mapping[str, Any], ...]
    online_oracle_inputs: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PublicTransition:
        if value.get("schema_version") != "piu.public-transition.v1":
            raise ValueError("unsupported public transition schema")
        row = cls(
            sample_id=_clean_text(value.get("sample_id"), name="sample_id"),
            initial_state_group=_clean_text(
                value.get("initial_state_group"), name="initial_state_group"
            ),
            split=Split(str(value.get("split", ""))),
            prompt=_clean_text(value.get("prompt"), name="prompt"),
            observations=dict(value.get("observations", {})),
            public_action_history=dict(value.get("public_action_history", {})),
            candidate_actions=tuple(value.get("candidate_actions", ())),
            online_oracle_inputs=tuple(value.get("online_oracle_inputs", ())),
        )
        if set(row.observations) != {"pre_interaction", "post_interaction"}:
            raise ValueError("observations require pre_interaction and post_interaction")
        if not row.candidate_actions:
            raise ValueError("at least one public candidate action is required")
        if row.online_oracle_inputs:
            raise ValueError("public transition declares online oracle inputs")
        assert_public_policy_value(row.policy_input())
        return row

    def policy_input(self) -> dict[str, Any]:
        """Project away provenance identifiers and every evaluator-side field."""

        result = {
            "prompt": self.prompt,
            "observations": dict(self.observations),
            "public_action_history": dict(self.public_action_history),
            "candidate_actions": [dict(row) for row in self.candidate_actions],
        }
        assert_public_policy_value(result)
        return result


def _nullable_bool(value: Any, *, name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean or null")
    return value


def _pixel_counts(value: Any, *, name: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty camera mapping")
    result = {str(key): int(count) for key, count in value.items()}
    if any(count < 0 for count in result.values()):
        raise ValueError(f"{name} cannot contain negative pixels")
    return result


@dataclasses.dataclass(frozen=True)
class EvaluatorSidecar:
    """Privileged outcomes stored separately from the policy observation."""

    sample_id: str
    initial_state_group: str
    split: Split
    sufficiency_decision_correct: bool | None
    interaction_selection_correct: bool | None
    primitive_execution_success: bool
    target_visible_pixels_pre: Mapping[str, int]
    target_visible_pixels_post: Mapping[str, int]
    information_acquired: bool
    target_identity_resolved: bool | None
    target_grasp_contact: bool
    wrong_object_grasp_contact: bool
    target_maximum_lift_m: float
    target_destination_final: bool
    task_success: bool
    provenance: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EvaluatorSidecar:
        if value.get("schema_version") != "piu.evaluator-sidecar.v1":
            raise ValueError("unsupported evaluator sidecar schema")
        lift = float(value.get("target_maximum_lift_m", float("nan")))
        if not math.isfinite(lift) or lift < 0.0:
            raise ValueError("target_maximum_lift_m must be finite and non-negative")
        pre = _pixel_counts(
            value.get("target_visible_pixels_pre"),
            name="target_visible_pixels_pre",
        )
        post = _pixel_counts(
            value.get("target_visible_pixels_post"),
            name="target_visible_pixels_post",
        )
        acquired = value.get("information_acquired")
        if not isinstance(acquired, bool):
            raise TypeError("information_acquired must be boolean")
        observed_acquisition = max(pre.values()) == 0 and max(post.values()) > 0
        if acquired != observed_acquisition:
            raise ValueError(
                "information_acquired must mean zero pre pixels and nonempty post mask"
            )
        return cls(
            sample_id=_clean_text(value.get("sample_id"), name="sample_id"),
            initial_state_group=_clean_text(
                value.get("initial_state_group"), name="initial_state_group"
            ),
            split=Split(str(value.get("split", ""))),
            sufficiency_decision_correct=_nullable_bool(
                value.get("sufficiency_decision_correct"),
                name="sufficiency_decision_correct",
            ),
            interaction_selection_correct=_nullable_bool(
                value.get("interaction_selection_correct"),
                name="interaction_selection_correct",
            ),
            primitive_execution_success=bool(
                value.get("primitive_execution_success", False)
            ),
            target_visible_pixels_pre=pre,
            target_visible_pixels_post=post,
            information_acquired=acquired,
            target_identity_resolved=_nullable_bool(
                value.get("target_identity_resolved"),
                name="target_identity_resolved",
            ),
            target_grasp_contact=bool(value.get("target_grasp_contact", False)),
            wrong_object_grasp_contact=bool(
                value.get("wrong_object_grasp_contact", False)
            ),
            target_maximum_lift_m=lift,
            target_destination_final=bool(
                value.get("target_destination_final", False)
            ),
            task_success=bool(value.get("task_success", False)),
            provenance=dict(value.get("provenance", {})),
        )

    def stage_outcomes(self) -> dict[str, bool | None]:
        """Return the six-stage evidence without inventing missing labels."""

        return {
            "L0_information_sufficiency": self.sufficiency_decision_correct,
            "L1_interaction_selection": self.interaction_selection_correct,
            "L2_primitive_execution": self.primitive_execution_success,
            "L3_information_acquisition": self.information_acquired,
            # This is a downstream utilization endpoint, not a spatial binding label.
            "L4_information_utilization": self.target_grasp_contact,
            "L5_task_completion": self.task_success,
        }


def validate_public_sidecar_pair(
    public: PublicTransition, sidecar: EvaluatorSidecar
) -> None:
    """Require exact identity and split agreement across the leakage firewall."""

    keys = ("sample_id", "initial_state_group", "split")
    for key in keys:
        if getattr(public, key) != getattr(sidecar, key):
            raise ValueError(f"public/evaluator mismatch for {key}")


def validate_group_splits(rows: Sequence[PublicTransition | EvaluatorSidecar]) -> None:
    """Keep all prompt/action branches of one initial state in one split."""

    observed: dict[str, Split] = {}
    for row in rows:
        previous = observed.setdefault(row.initial_state_group, row.split)
        if previous is not row.split:
            raise ValueError(
                f"group leakage: {row.initial_state_group!r} occurs in "
                f"{previous.value} and {row.split.value}"
            )


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def load_public_transitions(path: Path) -> list[PublicTransition]:
    rows = [PublicTransition.from_mapping(row) for row in _load_jsonl(path)]
    validate_group_splits(rows)
    return rows


def load_evaluator_sidecars(path: Path) -> list[EvaluatorSidecar]:
    rows = [EvaluatorSidecar.from_mapping(row) for row in _load_jsonl(path)]
    validate_group_splits(rows)
    return rows
