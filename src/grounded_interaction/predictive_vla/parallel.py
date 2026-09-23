"""Synchronous data parallelism for native VLM/AE calls outside model.forward.

Accumulate locally, then reduce every used gradient before clipping/Adam.
This avoids wrapping a model in DDP while silently bypassing its forward path.
"""

from itertools import islice

import torch
import torch.distributed as dist


def sharded_examples(examples, rank, world_size):
    """Disjoint deterministic windows; discard at most world_size-1 per epoch."""
    iterator = iter(examples)
    while block := list(islice(iterator, world_size)):
        if len(block) != world_size:
            return
        yield block[rank]


def buckets(parameters, limit=4_000_000):
    group, size = [], 0
    for parameter in parameters:
        if group and size + parameter.numel() > limit:
            yield group
            group, size = [], 0
        group.append(parameter)
        size += parameter.numel()
    if group:
        yield group


def average_gradients(parameters, local_count):
    """Reduce summed losses to the exact global example mean, including tails.

    A parameter unused on one rank contributes zero; unused on every rank
    retains grad=None, so Adam momentum/weight decay does not update it.
    """
    device = parameters[0].device
    active = torch.tensor(
        [p.grad is not None for p in parameters], device=device, dtype=torch.int32
    )
    count = torch.tensor(float(local_count), device=device)
    dist.all_reduce(active, op=dist.ReduceOp.MAX)
    dist.all_reduce(count)
    if count.item() <= 0:
        raise ValueError("empty distributed gradient batch")
    used = [
        p for p, present in zip(parameters, active.tolist(), strict=True) if present
    ]
    for group in buckets(used):
        for parameter in group:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
        flat = torch.cat([p.grad.reshape(-1) for p in group])
        dist.all_reduce(flat)
        flat.div_(count)
        offset = 0
        for parameter in group:
            parameter.grad.copy_(
                flat[offset : offset + parameter.numel()].view_as(parameter)
            )
            offset += parameter.numel()
    return int(count.item())


@torch.no_grad()
def assert_replicas_equal(parameters):
    """Compare actual values to rank zero; no digests or sampling."""
    equal = torch.ones((), device=parameters[0].device, dtype=torch.int32)
    for group in buckets(parameters):
        local = torch.cat([p.detach().reshape(-1) for p in group])
        reference = local.clone()
        dist.broadcast(reference, src=0)
        if not torch.equal(local, reference):
            equal.zero_()
    dist.all_reduce(equal, op=dist.ReduceOp.MIN)
    if not equal.item():
        raise RuntimeError("distributed parameter replicas diverged")


def prediction_scale(update, warmup_updates, maximum):
    """First update is action-only, reaching maximum after the ramp."""
    return maximum * min(1.0, update / warmup_updates) if warmup_updates else maximum
