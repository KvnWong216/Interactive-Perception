"""End-to-end exercise of the rollout loop against a stand-in environment.

LIBERO needs MuJoCo, a GPU-capable renderer, and a large asset tree, none of
which belong in a unit test. The fake below implements only the surface
``run_episode`` actually touches -- the observation keys, the segmentation
lookup, the free-joint reads used for anchors, and the step/done protocol -- so
the loop's control flow, probe cadence, trace schema, and outcome scoring can be
checked without a simulator.

It verifies plumbing, not physics. Nothing here says anything about how a real
policy behaves in a real scene.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from interactive_perception.anchors import AnchorSpec
from interactive_perception.policy_client import ScriptedStubPolicy
from interactive_perception.rollout import RolloutConfig, run_episode, write_trace

RESOLUTION = 32


class _FakeModel:
    def __init__(self, sites: dict[str, int]) -> None:
        self._sites = sites

    def site_name2id(self, name: str) -> int:
        return self._sites[name]

    def body_name2id(self, name: str) -> int:
        raise KeyError(name)


class _FakeData:
    def __init__(self, joints: dict[str, np.ndarray], sites: np.ndarray) -> None:
        self._joints = joints
        self.site_xpos = sites

    def get_joint_qpos(self, name: str) -> np.ndarray:
        return self._joints[name]


class _FakeSim:
    def __init__(self, model: _FakeModel, data: _FakeData) -> None:
        self.model = model
        self.data = data

    def forward(self) -> None:
        return None


class _FakeInner:
    def __init__(self, sim: _FakeSim, objects: dict[str, Any]) -> None:
        self.sim = sim
        self.objects_dict = objects


class _Joint:
    def __init__(self, name: str) -> None:
        self.joints = [name]


class FakeLiberoEnv:
    """Minimal stand-in exposing only what the rollout loop reads.

    ``target_visible_after`` controls when the hidden target enters the
    segmentation map, which is how the information-endpoint logic and the
    premature-commit rule get exercised on both sides of the transition.
    """

    def __init__(self, *, target_visible_after: int | None = None) -> None:
        self.target_visible_after = target_visible_after
        self.instance_to_id = {"butter_1": 7, "basket_1": 8}
        self.steps = 0
        self._eef = np.array([0.0, 0.0, 1.0])

        joints = {
            "butter_1_joint0": np.array([0.3, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0]),
            "basket_1_joint0": np.array([-0.3, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0]),
            "drawer_joint": np.array([-0.05]),
        }
        sites = np.array([[0.0, 0.4, 0.9]])
        self.env = _FakeInner(
            _FakeSim(_FakeModel({"cabinet_top": 0}), _FakeData(joints, sites)),
            {"butter_1": _Joint("butter_1_joint0"), "basket_1": _Joint("basket_1_joint0")},
        )

    def seed(self, value: int) -> None:
        self._seed = value

    def _obs(self) -> dict[str, Any]:
        segmentation = np.zeros((RESOLUTION, RESOLUTION, 1), dtype=np.int32)
        if (
            self.target_visible_after is not None
            and self.steps >= self.target_visible_after
        ):
            segmentation[:4, :4, 0] = 7
        return {
            "agentview_image": np.full((RESOLUTION, RESOLUTION, 3), 120, dtype=np.uint8),
            "robot0_eye_in_hand_image": np.full(
                (RESOLUTION, RESOLUTION, 3), 90, dtype=np.uint8
            ),
            "robot0_eef_pos": self._eef.copy(),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.array([0.02, -0.02]),
            "agentview_segmentation_instance": segmentation,
        }

    def reset(self) -> dict[str, Any]:
        self.steps = 0
        return self._obs()

    def step(self, action) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self.steps += 1
        self._eef = self._eef + np.asarray(action, dtype=np.float64)[:3]
        return self._obs(), 0.0, False, {}

    def close(self) -> None:
        return None


TASK = {
    "id": "T01_drawer_retrieval",
    "target": "butter_1",
    "reveal_joint_name": "drawer_joint",
    "expected_terminal": "TASK_SUCCESS",
}

ANCHORS = [
    AnchorSpec.from_dict({"label": "butter", "role": "task_target", "ref": "butter_1"}),
    AnchorSpec.from_dict({"label": "basket", "role": "placement", "ref": "basket_1"}),
    AnchorSpec.from_dict(
        {"label": "drawer", "role": "occluder", "kind": "site", "ref": "cabinet_top"}
    ),
]

CONFIG = RolloutConfig(
    max_steps=60, num_steps_wait=4, replan_steps=5, probe_samples=4, probe_every=10
)


def _run(env: FakeLiberoEnv, **kwargs):
    return run_episode(
        env=env,
        policy=ScriptedStubPolicy(goal_position=(0.3, 0.0, 0.9), seed=0),
        task=TASK,
        prompt="Place the butter in the wicker basket.",
        prompt_variant="implicit",
        anchor_specs=ANCHORS,
        seed=0,
        config=CONFIG,
        **kwargs,
    )


def test_rollout_completes_and_produces_probe_records() -> None:
    outcome, records = _run(FakeLiberoEnv())
    assert outcome.error is None
    assert outcome.steps == CONFIG.max_steps + CONFIG.num_steps_wait
    assert records, "probe cadence produced no records"
    assert all(0.0 <= item.vacuity <= 1.0 for item in records)
    assert all(0.0 <= item.dissonance <= 1.0 for item in records)


def test_probe_cadence_matches_configuration() -> None:
    _, records = _run(FakeLiberoEnv())
    steps = [item.step for item in records]
    assert steps[0] == CONFIG.num_steps_wait
    gaps = {b - a for a, b in zip(steps, steps[1:])}
    assert gaps == {CONFIG.probe_every}


def test_hidden_target_scores_as_endpoint_not_reached() -> None:
    outcome, _ = _run(FakeLiberoEnv())
    assert outcome.information_endpoint_reached is False
    assert outcome.max_target_visible_pixels == 0
    assert outcome.steps_to_endpoint is None
    # No abstention channel exists, so the episode must terminate by acting.
    assert outcome.terminal_decision == "ACT_NO_ABSTENTION"
    assert outcome.correct_terminal_decision is False


def test_revealed_target_records_the_endpoint_step() -> None:
    outcome, _ = _run(FakeLiberoEnv(target_visible_after=20))
    assert outcome.information_endpoint_reached is True
    assert outcome.max_target_visible_pixels == 16
    assert outcome.steps_to_endpoint is not None


def test_committing_before_the_endpoint_counts_as_premature() -> None:
    """The stub drives straight at the target, so it commits with no evidence."""

    outcome, _ = _run(FakeLiberoEnv())
    assert outcome.first_committed_step is not None
    assert outcome.committed_before_endpoint is True
    assert outcome.premature_commit is True


def test_not_found_evidence_stays_zero_across_the_episode() -> None:
    outcome, records = _run(FakeLiberoEnv())
    assert outcome.not_found_evidence == pytest.approx(0.0)
    assert all(item.primitive_evidence["NOT_FOUND"] == 0.0 for item in records)


def test_frames_are_captured_for_the_demo_renderer() -> None:
    frames: list[np.ndarray] = []
    _run(FakeLiberoEnv(), frames=frames)
    assert len(frames) == CONFIG.max_steps
    assert frames[0].shape == (RESOLUTION, RESOLUTION, 3)


def test_trace_round_trips_as_jsonl(tmp_path) -> None:
    outcome, records = _run(FakeLiberoEnv())
    path = tmp_path / "trace.jsonl"
    write_trace(path, outcome=outcome, records=records, metadata={"policy": "stub"})

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["kind"] == "episode"
    assert lines[0]["metadata"]["policy"] == "stub"
    assert lines[0]["premature_commit"] is True
    assert len(lines) == len(records) + 1
    assert {line["kind"] for line in lines[1:]} == {"step"}
    assert set(lines[1]["primitive_evidence"]) == {
        "ACT",
        "NOT_FOUND",
        "ROTATE",
        "MOVE_CLOSER",
        "REMOVE_OCCLUDER",
    }


def test_environment_errors_are_recorded_not_swallowed() -> None:
    class Broken(FakeLiberoEnv):
        def step(self, action):
            if self.steps > 8:
                raise RuntimeError("mujoco exploded")
            return super().step(action)

    outcome, _ = _run(Broken())
    assert outcome.error is not None
    assert "mujoco exploded" in outcome.error
