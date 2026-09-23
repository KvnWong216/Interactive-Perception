"""Check experiment controls before queued GPU head fits."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_real_predictor_debug import Candidate

from grounded_interaction.predictive_vla.model import ActionConditionedPredictor


def inputs():
    torch.manual_seed(17)
    return {
        "shared": torch.randn(1, 6, 32),
        "actions": torch.randn(1, 4, 7),
        "donor_actions": torch.randn(1, 4, 7),
        "current": torch.randn(1, 8, 32),
        "positions": torch.randn(1, 8, 4),
    }


def test_original_diagnostic_path_exactly_preserves_production_head():
    x = inputs()
    base = ActionConditionedPredictor(32, width=32, heads=4, layers=2, time_scale=1000)
    y = base(
        x["shared"],
        torch.ones(1, 6, dtype=torch.bool),
        x["actions"],
        torch.ones(1, 4, dtype=torch.bool),
        x["positions"],
        torch.tensor([4]),
    )
    torch.testing.assert_close(Candidate(base, "original")(x), y, rtol=0, atol=0)


@pytest.mark.parametrize(
    "variant",
    [
        "current_absolute_mixed",
        "current_residual_mixed",
        "current_absolute_separate",
        "current_residual_separate",
    ],
)
def test_factorial_keeps_current_and_action_gradient_paths(variant):
    x = inputs()
    x["current"].requires_grad_()
    x["actions"].requires_grad_()
    model = Candidate(
        ActionConditionedPredictor(32, width=32, heads=4, layers=2, time_scale=1000),
        variant,
    )
    output = model(x)
    output.square().mean().backward()
    assert output.shape == (1, 8, 32)
    assert x["current"].grad.abs().sum() > 0
    assert x["actions"].grad.abs().sum() > 0
    assert not torch.allclose(output, model(x, wrong=True))


@pytest.mark.parametrize(
    "variant",
    ["fourier_queries", "linear_query_control", "fourier_separate", "separate_actions"],
)
def test_additional_queries_and_fusion_have_action_gradients(variant):
    x = inputs()
    x["actions"].requires_grad_()
    model = Candidate(
        ActionConditionedPredictor(32, width=32, heads=4, layers=2, time_scale=1000),
        variant,
    )
    model(x).square().mean().backward()
    assert x["actions"].grad.abs().sum() > 0
    if hasattr(model, "position"):
        assert model.position.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("variant", ["fourier_queries", "linear_query_control"])
def test_zero_query_projection_preserves_initial_predictions(variant):
    x = inputs()
    base = ActionConditionedPredictor(32, width=32, heads=4, layers=2, time_scale=1000)
    torch.testing.assert_close(
        Candidate(base, variant)(x), Candidate(base, "original")(x), rtol=0, atol=0
    )


def test_real_fork_pair_score_rejects_copy_and_accepts_oracle():
    from fork_predictor_lab import evaluate, normalize

    torch.manual_seed(17)
    current = torch.randn(1, 4, 32)
    rows = [
        {
            "current": current,
            "target": torch.randn(1, 4, 32),
            "valid": torch.ones(1, 4, dtype=torch.bool),
        }
        for _ in range(6)
    ]
    cases = [{"case": {"case_id": 0}, "inputs": rows, "repeat_noise_max": 0}]
    oracle = evaluate(lambda x: normalize(x["target"]), cases)
    blind = evaluate(lambda x: normalize(x["current"]), cases)
    assert oracle["mean_loss"] == 0 and oracle["pair_assignment_accuracy"] == 1
    assert oracle["separation_ratio"] == pytest.approx(1)
    assert blind["pair_tie_fraction"] == 1 and blind["separation_ratio"] == 0


def test_effect_loss_ignores_common_error_but_preserves_action_differences():
    from train_fork_effect_control import effect_loss

    torch.manual_seed(17)
    target = torch.randn(6, 4, 32)
    common = torch.randn(1, 4, 32)
    assert effect_loss(target + common, target, 0.5) < 1e-12
    assert effect_loss(target.flip(0), target, 0.5) > 0.1
    shared = torch.randn(1, 4, 32, requires_grad=True)
    loss = effect_loss(shared.expand(6, -1, -1), target, 0.5)
    loss.backward()
    torch.testing.assert_close(shared.grad, torch.zeros_like(shared), rtol=0, atol=1e-7)
