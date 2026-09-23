"""A low average error or action ranking alone cannot authorize joint training."""

import json
from types import SimpleNamespace

import pytest
import torch

from grounded_interaction.predictive_vla.qualification import (
    load_qualified_predictor,
    mechanism_gate,
)


def test_copy_gain_alone_does_not_pass_mechanism_gate():
    live = {
        "loss": 0.08,
        "copy_loss": 0.1,
        "swapped_loss": 0.09,
        "action_variance_explained": 0.005,
    }
    checks = mechanism_gate(live, {"loss": 0.09})
    assert checks["beats_copy_10_percent"] and checks["wrong_action_cost_5_percent"]
    assert not checks["explains_action_variance_10_percent"]
    live["action_variance_explained"] = 0.2
    assert all(mechanism_gate(live, {"loss": 0.09}).values())
    live["loss"] = float("nan")
    with pytest.raises(ValueError, match="metrics"):
        mechanism_gate(live, {"loss": 0.09})


def test_development_report_cannot_mutate_predictor(tmp_path):
    report = tmp_path / "dev.json"
    report.write_text(json.dumps({"eligible_for_joint_training": False}))
    model = torch.nn.Linear(2, 2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError, match="independent transport"):
        load_qualified_predictor(
            tmp_path / "absent.pt",
            report,
            model,
            tmp_path / "policy.pt",
            SimpleNamespace(),
        )
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k])


@pytest.mark.parametrize("kind", ["affine", "local"])
def test_qualified_head_restores_exactly_and_checks_policy_metadata(tmp_path, kind):
    from grounded_interaction.predictive_vla.transport import (
        LocalTransportPredictor,
        TransportPredictor,
    )

    Model = LocalTransportPredictor if kind == "local" else TransportPredictor

    torch.manual_seed(17)
    source = Model(12, width=16)
    model = Model(12, width=16)
    head_path = tmp_path / "head.pt"
    policy_path = tmp_path / "policy.pt"
    report_path = tmp_path / "qualified.json"
    torch.save(
        {
            "schema": "transport-head-v1",
            "model_kind": kind,
            "manual_seed": 17,
            "normalize_context": True,
            "fusion": "control_residual",
            "width": 16,
            "time_scale": 300,
            "predictor": source.state_dict(),
        },
        head_path,
    )
    live = {
        "loss": 0.08,
        "copy_loss": 0.1,
        "swapped_loss": 0.1,
        "action_variance_explained": 0.2,
    }
    report = {
        "schema": "transport-qualification-v1",
        "model_kind": kind,
        "passed": True,
        "manual_seeds": [17, 29, 43],
        "confirmation_families": 3,
        "selected_head": str(head_path.resolve()),
        "source_policy": {
            "checkpoint": str(policy_path.resolve()),
            "updates": 1500,
            "manual_seed": 17,
            "config": {},
        },
        "disjoint_families_and_episodes": True,
        "fresh_confirmation": True,
        "seed_results": [
            {"manual_seed": seed, "live": live, "blind": {"loss": 0.09}}
            for seed in [17, 29, 43]
        ],
    }
    report_path.write_text(json.dumps(report))
    config = SimpleNamespace(
        predictor_dim=16,
        total_steps=300,
        predictor_kind="local_transport" if kind == "local" else "transport",
    )
    metadata = {"source_updates": 1500, "source_manual_seed": 17, "source_config": {}}
    load_qualified_predictor(
        head_path, report_path, model, policy_path, config, policy_metadata=metadata
    )
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, source.state_dict()[k], rtol=0, atol=0)
    with pytest.raises(ValueError, match="actual policy"):
        load_qualified_predictor(
            head_path,
            report_path,
            model,
            policy_path,
            config,
            policy_metadata={**metadata, "source_updates": 2000},
        )
