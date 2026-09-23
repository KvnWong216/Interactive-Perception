"""Check actual distributed optimizer semantics against a single-process batch."""

from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from grounded_interaction.predictive_vla.config import VLAConfig
from grounded_interaction.predictive_vla.parallel import (
    assert_replicas_equal,
    average_gradients,
    prediction_scale,
    sharded_examples,
)
from grounded_interaction.predictive_vla.training import (
    CHECKPOINT_SCHEMA,
    warm_start_policy,
)


def _worker(rank, rendezvous, result):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4)
    parameters = [torch.nn.Parameter(torch.tensor([1.0, 2.0])) for _ in range(3)]
    optimizer = torch.optim.AdamW(parameters, lr=0.01, weight_decay=0.1)
    for _ in range(rank + 1):
        loss = (parameters[0] * (rank + 1)).sum()
        if rank == 0:
            loss = loss + parameters[1].square().sum()
        loss.backward()
    assert average_gradients(parameters, rank + 1) == 10
    torch.testing.assert_close(parameters[0].grad, torch.full((2,), 3.0))
    torch.testing.assert_close(parameters[1].grad, torch.tensor([0.2, 0.4]))
    assert parameters[2].grad is None
    optimizer.step()
    assert_replicas_equal(parameters)
    if rank == 0:
        torch.save([p.detach() for p in parameters], result)
    dist.destroy_process_group()


def test_four_rank_optimizer_matches_global_mean(tmp_path):
    rendezvous = "file://" + str(tmp_path / "rendezvous")
    result = str(tmp_path / "parameters.pt")
    mp.spawn(_worker, args=(rendezvous, result), nprocs=4, join=True)
    reference = [torch.nn.Parameter(torch.tensor([1.0, 2.0])) for _ in range(3)]
    optimizer = torch.optim.AdamW(reference, lr=0.01, weight_decay=0.1)
    losses = []
    for rank in range(4):
        for _ in range(rank + 1):
            loss = (reference[0] * (rank + 1)).sum()
            if rank == 0:
                loss = loss + reference[1].square().sum()
            losses.append(loss)
    (sum(losses) / len(losses)).backward()
    optimizer.step()
    for actual, expected in zip(
        torch.load(result, weights_only=True), reference, strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_disjoint_rank_windows_and_prediction_ramp():
    shards = [list(sharded_examples(range(11), rank, 4)) for rank in range(4)]
    assert shards == [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert prediction_scale(0, 200, 0.1) == 0
    assert prediction_scale(100, 200, 0.1) == 0.05
    assert prediction_scale(300, 200, 0.1) == 0.1


def test_warm_start_exact_transfer_and_architecture_rejection(tmp_path):
    from types import SimpleNamespace

    source = VLAConfig(prediction_weight=0, language_weight=0)
    model = torch.nn.Linear(3, 2)
    expected = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    path = tmp_path / "best.pt"
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "config": source.to_dict(),
            "manual_seed": source.manual_seed,
            "updates": 1500,
            "adapters": expected,
            "predictor": None,
        },
        path,
    )
    backend = SimpleNamespace(
        model=torch.nn.Linear(3, 2),
        config=replace(source, prediction_weight=0.1, accumulation=8),
    )
    report = warm_start_policy(path, backend)
    assert report["exact_restoration"] and not report["optimizer_restored"]
    for name, value in backend.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    backend.config = replace(backend.config, history_frames=2)
    with pytest.raises(ValueError, match="architecture differs"):
        warm_start_policy(path, backend)
