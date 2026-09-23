"""Behavioral checks for geometry, causality, gradients and real prefix feedback."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from grounded_interaction.predictive_vla.backend import temporal_attention_bias
from grounded_interaction.predictive_vla.config import VLAConfig
from grounded_interaction.predictive_vla.data import (
    DATA_SCHEMA,
    TrainingExample,
    Trajectory,
    TrajectoryDataset,
)
from grounded_interaction.predictive_vla.geometry import GeometryAdapter, backproject
from grounded_interaction.predictive_vla.model import (
    ActionConditionedPredictor,
    patch_prediction_loss,
)
from grounded_interaction.predictive_vla.native import (
    ResidualLinear,
    postprocess_actions,
)
from grounded_interaction.predictive_vla.runtime import parse_response, run_episode
from grounded_interaction.predictive_vla.training import joint_loss
from grounded_interaction.predictive_vla.types import (
    AppliedAction,
    CameraFrame,
    Observation,
    PolicyContext,
)


def observation(step, color=0):
    frame = CameraFrame(np.full((4, 4, 3), color, dtype=np.uint8))
    return Observation(step, (frame, frame), np.zeros(8))


def context(step=0):
    return PolicyContext("find the cup", (observation(step),), (), 300 - step)


def test_geometry_uses_calibration_and_preserves_unknown_depth():
    depth = np.array([[2.0, np.nan], [0.0, 3.0]], dtype=np.float32)
    t = np.eye(4)
    t[:3, 3] = [1, 2, 3]
    frame = CameraFrame(np.zeros((2, 2, 3), dtype=np.uint8), depth, np.eye(3), t)
    xyz, valid = backproject(frame, np.array([[0, 0], [1, 0], [0, 1], [1, 1], [-1, 0]]))
    np.testing.assert_allclose(xyz[[0, 3]], [[1, 2, 5], [4, 5, 6]])
    assert valid.tolist() == [True, False, False, True, False]
    missing = CameraFrame(frame.rgb)
    assert not backproject(missing, np.array([[0, 0]]))[1].any()


def test_geometry_residual_starts_at_native_and_receives_gradients():
    adapter = GeometryAdapter(8)
    xyz = torch.tensor([[1.0, 2.0, 3.0], [float("nan"), 0.0, 0.0]])
    mask = torch.tensor([True, False])
    output = adapter(xyz, mask)
    assert torch.equal(output, torch.zeros_like(output))
    output.sum().backward()
    assert adapter.project[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        adapter.project[-1].weight.normal_()
    assert not torch.equal(
        adapter(torch.zeros(2, 3), mask)[0], adapter(torch.zeros(2, 3), mask)[1]
    )


def test_block_attention_never_reads_later_observations_or_future_text():
    times = torch.tensor([[0, 0, 1, 1, 1]])
    visual = torch.tensor([[True, True, True, False, False]])
    allowed = temporal_attention_bias(times, visual, torch.float32)[0, 0].eq(0)
    assert allowed[0, 1]  # same-time visual tokens can fuse
    assert not allowed[:2, 2:].any()
    assert allowed[3, :4].all()
    assert not allowed[3, 4]  # answer tokens remain autoregressive
    assert allowed.diagonal().all()


def test_history_rejects_unexecuted_actions_and_retains_actual_prefix():
    with pytest.raises(ValueError, match="actually applied"):
        PolicyContext(
            "find cup", (observation(0),), (AppliedAction(0, np.zeros(7)),), 300
        )
    old = context()
    actual = tuple(AppliedAction(i, np.full(7, i)) for i in range(3))
    new = old.advance(observation(3), actual, max_frames=2)
    assert new.remaining_steps == 297
    assert [a.step for a in new.applied_actions] == [0, 1, 2]
    latest = new.advance(observation(4), (AppliedAction(3, np.zeros(7)),), max_frames=1)
    assert latest.applied_actions == ()
    assert latest.current.step == 4


def predictor_inputs():
    return (
        torch.randn(1, 4, 12, requires_grad=True),
        torch.tensor([[True, True, True, False]]),
        torch.randn(1, 3, 7, requires_grad=True),
        torch.tensor([[True, True, False]]),
        torch.rand(1, 5, 4),
        torch.tensor([2]),
    )


def test_prediction_reads_history_and_ordered_actions_without_target_gradients():
    torch.manual_seed(3)
    predictor = ActionConditionedPredictor(12, width=16, heads=4, layers=2)
    shared, cm, actions, am, positions, horizon = predictor_inputs()
    output = predictor(shared, cm, actions, am, positions, horizon)
    target = torch.randn_like(output, requires_grad=True)
    patch_prediction_loss(output, target, torch.ones(1, 5, dtype=torch.bool)).backward()
    assert target.grad is None
    assert shared.grad[:, :3].abs().sum() > 0
    assert actions.grad[:, :2].abs().sum() > 0
    assert not shared.grad[:, 3].any() and not actions.grad[:, 2].any()
    reversed_actions = actions.detach().clone()
    reversed_actions[:, :2] = reversed_actions[:, :2].flip(1)
    changed = predictor(shared, cm, reversed_actions, am, positions, horizon)
    assert not torch.allclose(output, changed)
    # Masked padding is inert, including NaNs.
    dirty_shared, dirty_actions = shared.detach().clone(), actions.detach().clone()
    dirty_shared[:, 3] = float("nan")
    dirty_actions[:, 2] = float("nan")
    torch.testing.assert_close(
        output, predictor(dirty_shared, cm, dirty_actions, am, positions, horizon)
    )


def test_future_queries_are_independent_and_identify_crop_level():
    torch.manual_seed(7)
    predictor = ActionConditionedPredictor(12, width=16, heads=4, layers=2)
    shared, cm, actions, am, positions, horizon = predictor_inputs()
    output = predictor(shared, cm, actions, am, positions, horizon)
    alone = predictor(shared, cm, actions, am, positions[:, :1], horizon)
    torch.testing.assert_close(output[:, :1], alone)
    levels = positions[:, :1].repeat(1, 2, 1)
    levels[:, 0, -1] = 0
    levels[:, 1, -1] = 1
    result = predictor(shared, cm, actions, am, levels, horizon)
    assert not torch.allclose(result[:, 0], result[:, 1])


def test_prediction_rejects_misaligned_horizon():
    model = ActionConditionedPredictor(12, width=16, heads=4)
    values = list(predictor_inputs())
    values[-1] = torch.tensor([3])
    with pytest.raises(ValueError, match="actual action prefix"):
        model(*values)


def test_zero_lora_preserves_native_bypass_after_updates():
    base = torch.nn.Linear(4, 5)
    enabled = True
    layer = ResidualLinear(base, rank=2, alpha=4.0, enabled=lambda: enabled)
    x = torch.randn(3, 4)
    torch.testing.assert_close(layer(x), base(x), rtol=0, atol=0)
    layer(x).sum().backward()
    assert layer.up.weight.grad.abs().sum() > 0
    assert base.weight.grad is None
    with torch.no_grad():
        layer.up.weight.add_(1.0)
    assert not torch.equal(layer(x), base(x))
    enabled = False
    assert torch.equal(layer(x), base(x))


def test_action_postprocessing_preserves_native_window_before_unnormalizing():
    outer = SimpleNamespace(
        config=SimpleNamespace(n_obs_steps=2),
        _slice_action_dim=lambda a, d: a[..., :d],
        _slice_action_chunk=lambda a, obs, n: a[:, obs - 1 : obs - 1 + n],
    )
    stats = SimpleNamespace(unnormalize_action=lambda a, tag: a + 100)
    actions = torch.arange(54).reshape(1, 6, 9)
    result = postprocess_actions(
        outer=outer,
        actions=actions,
        stats=stats,
        tag="libero",
        action_dim=7,
        n_action_steps=3,
    )
    torch.testing.assert_close(result, actions[:, 1:4, :7] + 100)


def write_trajectory(path, *, color=0, length=13):
    np.savez(
        path,
        agent_rgb=np.full((length + 1, 4, 4, 3), color, dtype=np.uint8),
        wrist_rgb=np.zeros((length + 1, 4, 4, 3), dtype=np.uint8),
        states=np.zeros((length + 1, 8), dtype=np.float32),
        actions=np.zeros((length, 7), dtype=np.float32),
    )


def test_partial_final_chunk_uses_deployment_history_boundaries(tmp_path):
    path = tmp_path / "episode.npz"
    write_trajectory(path)
    trajectory = Trajectory(
        path, {"task": "find cup", "final_response": "DONE: cup"}, total_steps=300
    )
    examples = list(trajectory.examples(VLAConfig(execute_steps=10)))
    assert [e.context.current.step for e in examples] == [0, 10, 13]
    assert [o.step for o in examples[-1].context.observations] == [0, 10, 13]
    assert [a.step for a in examples[-1].context.applied_actions] == list(range(13))
    assert examples[-2].future_observation.step == 13
    assert len(examples[-2].actual_future_actions) == 3


def test_dataset_rejects_family_leakage_before_loading_policy(tmp_path):
    rows = []
    for i, split in enumerate(("train", "validation")):
        path = tmp_path / f"ep{i}.npz"
        write_trajectory(path, color=i)
        rows.append(
            {
                "episode_id": str(i),
                "reset_family": "same-family",
                "split": split,
                "task": "find cup",
                "path": path.name,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": DATA_SCHEMA, "episodes": rows}))
    with pytest.raises(ValueError, match="crosses data splits"):
        TrajectoryDataset(manifest, VLAConfig())


def test_parallel_data_validation_preserves_counts_and_rejects_bad_states(tmp_path):
    rows = []
    for index, split in enumerate(("train", "validation", "test")):
        path = tmp_path / f"episode{index}.npz"
        write_trajectory(path, length=index + 2)
        rows.append(
            {
                "episode_id": str(index),
                "reset_family": str(index),
                "split": split,
                "task": "find cup",
                "path": path.name,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": DATA_SCHEMA, "episodes": rows}))
    dataset = TrajectoryDataset(manifest, VLAConfig())
    sequential = dataset.validate(workers=1)
    assert sequential["actions"] == 9 and sequential["episodes"] == 3
    assert dataset.validate(workers=3) == sequential
    audit = tmp_path / "final_audit.json"
    audit.write_text(
        json.dumps({"passed": True, "manual_seed": 17, "summary": sequential})
    )
    assert dataset.read_validation_report(audit) == sequential
    audit.write_text(
        json.dumps({"passed": True, "manual_seed": 29, "summary": sequential})
    )
    with pytest.raises(ValueError, match="manual_seed"):
        dataset.read_validation_report(audit)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["states"][0, 0] = np.nan
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="real 8-D states"):
        dataset.validate(workers=3)


def test_joint_prediction_never_passes_future_to_policy_encoder():
    cfg = VLAConfig(language_weight=0.0)
    current = context()
    example = TrainingExample(current, np.ones((2, 7)), observation(2, 200), None)
    shared = torch.randn(1, 3, 12, requires_grad=True)
    calls = []

    class Backend:
        config = cfg

        def encode(self, c):
            assert c is current and c.current.step == 0
            calls.append("public_encode")
            return SimpleNamespace(
                hidden=shared, ids=torch.ones((1, 3), dtype=torch.long)
            )

        def flow_loss(self, encoded, actions):
            return encoded.hidden.square().mean()

        def normalize_actions(self, actions):
            return torch.tensor(actions, dtype=torch.float32)

        def target(self, obs, task):
            assert obs is example.future_observation and obs.step == 2
            calls.append("separate_teacher")
            return (
                torch.ones(1, 4, 12),
                torch.rand(1, 4, 4),
                torch.ones(1, 4, dtype=torch.bool),
            )

    loss, metrics = joint_loss(
        Backend(), ActionConditionedPredictor(12, width=16, heads=4), example
    )
    loss.backward()
    assert calls == ["public_encode", "separate_teacher"]
    assert set(metrics) == {"action", "prediction"} and shared.grad.abs().sum() > 0


def test_runtime_reobserves_and_records_actual_actions_before_answering():
    seen = []

    class Policy:
        def reset(self):
            pass

        def respond(self, c):
            seen.append(c.current.step)
            return "CONTINUE" if c.current.step == 0 else "DONE: a cup"

        def act(self, c, *, seed, steps):
            return np.zeros((steps, 7))

    class Environment:
        step_count = 0

        def step(self, action):
            self.step_count += 1
            applied = action.copy()
            applied[-1] = 1.0  # simulator canonicalization, not proposed 0
            return observation(self.step_count), applied

    result = run_episode(
        policy=Policy(),
        environment=Environment(),
        context=context(),
        config=VLAConfig(),
    )
    assert seen == [0, 5] and result.answer == "a cup"
    assert len(result.actions) == 5 and all(a.value[-1] == 1.0 for a in result.actions)
    assert result.context.remaining_steps == 295


def test_training_predicts_longer_than_the_deployment_prefix(tmp_path):
    path = tmp_path / "episode.npz"
    write_trajectory(path, length=20)
    trajectory = Trajectory(path, {"task": "find cup"}, total_steps=300)
    examples = list(trajectory.examples(VLAConfig()))
    assert [e.context.current.step for e in examples] == [0, 5, 10, 15]
    assert [len(e.actual_future_actions) for e in examples] == [10, 10, 10, 5]
    assert [a.step for a in examples[1].context.applied_actions] == list(range(5))


@pytest.mark.parametrize("value", ["DONE:", "maybe finished", "CONTINUE and answer"])
def test_incomplete_or_ambiguous_language_cannot_stop(value):
    with pytest.raises(ValueError):
        parse_response(value)
