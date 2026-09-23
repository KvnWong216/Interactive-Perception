"""Known-answer controls for the experiment's causal diagnostics."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_predictor_capacity import spectrum_bound
from predictor_diagnostic_lab import make_pairs, pair_metrics, predict, stack

from grounded_interaction.predictive_vla.model import ActionConditionedPredictor


def test_action_pair_shares_every_observation_and_oracle_requires_correct_action():
    records = make_pairs(families=2)
    row = records[0]
    for key in ("shared", "current", "positions"):
        torch.testing.assert_close(row[key][0], row[key][1], rtol=0, atol=0)
    metrics = pair_metrics(row["target"], row["target"])
    assert metrics["loss"] == 0 and metrics["strict_pair_accuracy"] == 1
    assert metrics["empirical_action_blind_bound"] > 0
    midpoint = row["target"].mean(0, keepdim=True).expand_as(row["target"])
    blind = pair_metrics(midpoint, row["target"])
    assert blind["tie_fraction"] == 1
    assert blind["loss"] == pytest.approx(blind["empirical_action_blind_bound"])
    with pytest.raises(AssertionError):
        torch.testing.assert_close(records[0]["current"], records[1]["current"])


def test_static_negative_control_cannot_claim_action_use():
    row = make_pairs(families=2, static=True)[0]
    result = pair_metrics(row["current"], row["target"])
    assert result["loss"] == pytest.approx(0, abs=1e-10)
    assert result["tie_fraction"] == 1 and result["strict_pair_accuracy"] == 0
    assert result["empirical_action_blind_bound"] == 0


def test_blind_network_is_invariant_to_paired_actions():
    torch.manual_seed(17)
    model = ActionConditionedPredictor(32, width=32, heads=4, layers=2)
    data = stack(make_pairs(families=2))
    output = predict(model, data, blind=True).reshape(2, 2, 16, 32)
    torch.testing.assert_close(output[:, 0], output[:, 1], rtol=0, atol=0)


def test_rank_bound_detects_known_bottleneck():
    matrix = torch.eye(4)
    rank1 = spectrum_bound(matrix, 1)
    assert rank1["best_affine_rank_mse_bound"] == pytest.approx(2 / 16)
    assert rank1["effective_rank"] == pytest.approx(3)
    assert spectrum_bound(matrix, 3)["best_affine_rank_mse_bound"] < 1e-12
