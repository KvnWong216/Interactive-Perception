"""Verify actual optimization budgets and recovery across an interrupted batch."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from grounded_interaction.predictive_vla.config import VLAConfig
from grounded_interaction.predictive_vla.training import (
    CHECKPOINT_SCHEMA,
    load_checkpoint,
    train,
)


class Backend:
    config = VLAConfig(
        geometry=False,
        prediction_weight=0,
        language_weight=0,
        accumulation=2,
        learning_rate=0.01,
        vlm_learning_rate=0.01,
        action_expert_learning_rate=0.01,
    )
    device = torch.device("cpu")

    def __init__(self):
        torch.manual_seed(91)
        self.model = torch.nn.Module()
        self.model.transformer = torch.nn.Module()
        self.model.transformer.blocks = torch.nn.Sequential(torch.nn.Linear(2, 2))
        self.model.added = torch.nn.Linear(2, 2)
        self.model.action_expert = torch.nn.Linear(2, 1)
        self.model.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)

    def train(self, mode):
        self.model.train(mode)

    def encode(self, context):
        return self.model.transformer.blocks(context) + self.model.added(context)

    def flow_loss(self, encoded, actions):
        prediction = self.model.action_expert(encoded)
        return (
            (prediction - float(actions[0, 0]) + torch.rand_like(prediction) * 0.1)
            .square()
            .mean()
        )


class Dataset:
    def __init__(self, *, interrupt=False):
        self.interrupt = interrupt

    def validate(self):
        return {
            "splits": {"train": 1, "validation": 1},
            "episodes": 2,
            "rgbd_episodes": 0,
        }

    def examples(self, split, *, seed):
        for i in range(9 if split == "train" else 3):
            if self.interrupt and split == "train" and i == 5:
                raise RuntimeError("injected data interruption")
            yield SimpleNamespace(
                context=torch.tensor([[float(i) / 10, 1.0]]),
                actual_future_actions=np.ones((1, 7)) * i / 10,
                response=None,
            )


@pytest.mark.parametrize("corruption", ["missing", "nonfinite", "shape"])
def test_predictor_restore_rejects_invalid_head_before_mutating_policy(tmp_path, corruption):
    torch.manual_seed(17)
    config = VLAConfig(prediction_weight=0.1, language_weight=0)
    policy, predictor = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    backend = SimpleNamespace(model=policy, config=config)
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    head = {k: v.clone() for k, v in predictor.state_dict().items()}
    if corruption == "missing":
        head = None
    elif corruption == "nonfinite":
        head["weight"][0, 0] = float("nan")
    else:
        head["weight"] = head["weight"][:1]
    path = tmp_path / "bad.pt"
    torch.save({"schema": CHECKPOINT_SCHEMA, "config": config.to_dict(),
                "manual_seed": config.manual_seed, "adapters": {k: v+1 for k, v in before.items()},
                "predictor": head}, path)
    with pytest.raises(ValueError, match="predictor"):
        load_checkpoint(path, backend, predictor)
    for k, v in policy.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)


def test_valid_stage2_policy_only_restore_still_checks_saved_head(tmp_path):
    torch.manual_seed(17)
    config = VLAConfig(prediction_weight=0.1, language_weight=0)
    source, target = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    path = tmp_path / "valid.pt"
    torch.save({"schema": CHECKPOINT_SCHEMA, "config": config.to_dict(),
                "manual_seed": config.manual_seed, "adapters": source.state_dict(),
                "predictor": torch.nn.Linear(2, 2).state_dict()}, path)
    load_checkpoint(path, SimpleNamespace(model=target, config=config))
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, source.state_dict()[name], rtol=0, atol=0)


def test_resume_matches_uninterrupted_updates_and_retains_full_expert(tmp_path):
    options = {
        "max_updates": 5,
        "warmup_updates": 2,
        "eval_every": 1,
        "validation_examples": 3,
    }
    reference = Backend()
    result = train(reference, Dataset(), output=tmp_path / "reference", **options)
    assert result["updates"] == 5
    interrupted = Backend()
    with pytest.raises(RuntimeError, match="injected data interruption"):
        train(
            interrupted, Dataset(interrupt=True), output=tmp_path / "resume", **options
        )
    recovered = Backend()
    result = train(
        recovered,
        Dataset(),
        output=tmp_path / "resume",
        resume=tmp_path / "resume/last.pt",
        **options,
    )
    assert result["updates"] == 5
    for name, tensor in reference.model.state_dict().items():
        torch.testing.assert_close(
            tensor, recovered.model.state_dict()[name], rtol=0, atol=0
        )
    payload = torch.load(tmp_path / "resume/last.pt", weights_only=True)
    assert any("action_expert" in key for key in payload["adapters"])
    assert "frozen" not in payload["adapters"]
    assert payload["predictor"] is None
    metadata = json.loads((tmp_path / "resume/run_metadata.json").read_text())
    assert metadata["effective_batch"] == 2
    assert metadata["trainable_parameters"]["action_expert"] > 0
    with pytest.raises(ValueError, match="schedule differs"):
        train(
            Backend(),
            Dataset(),
            output=tmp_path / "resume",
            resume=tmp_path / "resume/last.pt",
            **{**options, "max_updates": 6},
        )
