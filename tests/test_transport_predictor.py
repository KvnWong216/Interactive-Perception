"""Transport identity, provenance boundary, order, padding and learning checks."""

import pytest
import torch

from grounded_interaction.predictive_vla.config import VLAConfig
from grounded_interaction.predictive_vla.transport import (
    TransportPredictor,
    native_grids,
)


def inputs():
    torch.manual_seed(17)
    axis = torch.tensor([0.1, 0.45, 0.9])
    v, u = torch.meshgrid(axis, axis, indexing="ij")
    uv = torch.stack((u.flatten(), v.flatten()), -1)
    p = torch.cat(
        [
            torch.cat((uv, torch.full((9, 1), float(i)), torch.zeros(9, 1)), 1)
            for i in [0, 1]
        ]
    )[None]
    return {
        "shared": torch.randn(1, 4, 12, requires_grad=True),
        "context_valid": torch.tensor([[True, True, True, False]]),
        "actions": torch.randn(1, 3, 7, requires_grad=True),
        "action_valid": torch.tensor([[True, True, False]]),
        "positions": p,
        "horizon": torch.tensor([2]),
        "current": torch.randn(1, 18, 12, requires_grad=True),
        "current_valid": torch.ones(1, 18, dtype=torch.bool),
        "state": torch.randn(1, 8),
    }


def test_transport_identity_current_detached_then_ordered_gradient():
    x = inputs()
    m = TransportPredictor(12, width=16)
    initial = m(**x)
    torch.testing.assert_close(
        initial,
        torch.nn.functional.layer_norm(x["current"].detach(), (12,)),
        atol=1e-6,
        rtol=1e-6,
    )
    initial.square().sum().backward()
    assert x["current"].grad is None
    assert m.output[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        m.output[-1].weight.normal_(std=0.01)
    m.zero_grad()
    x = inputs()
    out = m(**x)
    out.square().mean().backward()
    assert x["shared"].grad[:, :3].abs().sum() > 0
    assert x["actions"].grad[:, :2].abs().sum() > 0
    assert not x["shared"].grad[:, 3].any() and not x["actions"].grad[:, 2].any()
    reversed_actions = x["actions"].detach().clone()
    reversed_actions[:, :2] = reversed_actions[:, :2].flip(1)
    assert not torch.allclose(out, m(**{**x, "actions": reversed_actions}))
    for k in ["shared", "actions"]:
        dirty = x[k].detach().clone()
        dirty[:, -1] = float("nan")
        torch.testing.assert_close(out, m(**{**x, k: dirty}))


def test_transport_rejects_wrong_grid_and_horizon():
    x = inputs()
    m = TransportPredictor(12, width=16)
    with pytest.raises(ValueError, match="prefix"):
        m(**{**x, "horizon": torch.tensor([3])})
    p = x["positions"].clone()
    p[:, 0, 3] = 1
    with pytest.raises(ValueError, match="uncropped"):
        m(**{**x, "positions": p})
    p = x["positions"][0].clone()
    p[[0, 1]] = p[[1, 0]]
    with pytest.raises(ValueError, match="row-major"):
        native_grids(p)
    mask = x["current_valid"].clone()
    mask[:, 0] = False
    with pytest.raises(ValueError, match="complete"):
        m(**{**x, "current_valid": mask})


def test_old_configuration_defaults_remain_absolute():
    assert VLAConfig().predictor_kind == "absolute"
    with pytest.raises(ValueError, match="predictor kind"):
        VLAConfig(predictor_kind="guess")


@pytest.mark.parametrize("kind", ["transport", "local_transport"])
def test_joint_transport_keeps_future_out_of_policy_and_current_teacher(kind):
    from types import SimpleNamespace

    import numpy as np

    from grounded_interaction.predictive_vla.training import joint_loss, make_predictor

    x = inputs()
    now = SimpleNamespace(state=np.zeros(8, dtype=np.float32))
    future = object()
    context = SimpleNamespace(current=now, task="move")
    observations = []
    encoded = []

    class Backend:
        config = VLAConfig(
            predictor_kind=kind,
            predictor_dim=16,
            predictor_heads=4,
            language_weight=0,
        )
        capabilities = SimpleNamespace(hidden_dim=12)
        device = torch.device("cpu")

        def encode(self, c):
            encoded.append(c)
            return SimpleNamespace(
                hidden=x["shared"], ids=torch.ones(1, 4, dtype=torch.long)
            )

        def flow_loss(self, shared, actions):
            return shared.hidden.square().mean()

        def normalize_actions(self, actions):
            return torch.tensor(actions, dtype=torch.float32)

        def target(self, observation, task):
            observations.append(observation)
            return (
                x["current"].detach() + (0 if observation is now else 1),
                x["positions"],
                x["current_valid"],
            )

    b = Backend()
    head = make_predictor(b)
    example = SimpleNamespace(
        context=context,
        future_observation=future,
        actual_future_actions=np.zeros((2, 7)),
        response=None,
    )
    loss, parts = joint_loss(b, head, example)
    assert torch.isfinite(loss)
    assert encoded == [context] and observations == [future, now]
    assert set(parts) == {"action", "prediction"}


def test_local_transport_keeps_identity_masks_and_teacher_boundary():
    from grounded_interaction.predictive_vla.transport import LocalTransportPredictor

    x = inputs()
    model = LocalTransportPredictor(12, width=16)
    initial = model(**x)
    torch.testing.assert_close(
        initial,
        torch.nn.functional.layer_norm(x["current"].detach(), (12,)),
        atol=1e-6,
        rtol=1e-6,
    )
    # A nonconstant spatial target activates the newly introduced readout.
    target = initial.detach().roll(1, dims=1)
    (initial - target).square().mean().backward()
    assert x["current"].grad is None
    assert model.local_output[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.local_output[-1].weight.normal_(std=0.01)
    model.zero_grad()
    x = inputs()
    output = model(**x)
    output.square().mean().backward()
    assert x["current"].grad is None
    assert x["actions"].grad[:, :2].abs().sum() > 0
    assert not x["actions"].grad[:, 2:].any()
    assert not x["shared"].grad[:, 3:].any()
    dirty = x["actions"].detach().clone()
    dirty[:, -1] = float("nan")
    torch.testing.assert_close(output, model(**{**x, "actions": dirty}))


@pytest.mark.parametrize("kind", ["local_residual", "patch_residual"])
def test_appearance_correction_starts_at_copy_and_detaches_teacher(kind):
    from grounded_interaction.predictive_vla.transport import PREDICTOR_TYPES

    x = inputs()
    model = PREDICTOR_TYPES[kind](12, width=16)
    result = model(**x)
    expected = torch.nn.functional.layer_norm(x["current"].detach(), (12,))
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
    (result - expected.roll(1, dims=1)).square().mean().backward()
    assert x["current"].grad is None
    assert model.residual_output[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.residual_output[-1].weight.normal_(std=0.01)
    model.zero_grad()
    x = inputs()
    model(**x).square().mean().backward()
    assert x["current"].grad is None
    assert x["actions"].grad[:, :2].abs().sum() > 0
    assert not x["actions"].grad[:, 2:].any()
    assert x["shared"].grad[:, :3].abs().sum() > 0
    assert not x["shared"].grad[:, 3:].any()
