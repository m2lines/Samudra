# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Small DDP building blocks for the future native-quarter runner.

The runner owns process-group initialization, data readers, optimizer/schedule,
gradient checks/clipping, checkpoint I/O, and fail-fast worker lifecycle. Wrap
the core model with ``distributed_task_loss`` and call
``accumulate_task_gradients`` after ``optimizer.zero_grad(set_to_none=True)``.
Objectives receive that core model explicitly: never close over another model
or call ``ddp.module`` for training forwards. Each callback returns the scalar
loss for one example, with unchanged task-specific loss weights.

All ranks must use the same task, completed task-update count and global batch
size. Eight global examples become two per rank at world size four; the first
forward AND backward run inside no_sync. The returned loss and synchronized
gradients are global means. Task callbacks must have consistent parameter usage
within an update; unused task adapters remain None and avoid AdamW updates.

Checkpoint only at completed optimizer-update boundaries. Save model, optimizer,
task counters, configuration/provenance and ``gather_rank_rng`` together. Restore
RNG after constructing/loading everything, immediately before the next update.
This restores Python, global NumPy, torch CPU and one explicitly chosen CUDA
device per rank. Custom Generator objects and worker/prefetch state are owned by
the runner. Exact replay requires the same topology and deterministic kernels;
equivalence across rank counts is numerical, not bitwise.
"""

import random
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from samudra.experiments.task_schedule import sample_indices


@dataclass(frozen=True)
class MicroSample:
    global_micro: int
    sample_index: int
    mask_seed: int
    coverage_index: int


def rank_samples(
    *,
    task: str,
    dataset_size: int,
    observation_size: int,
    seed: int,
    completed_task_updates: int,
    rank: int,
    world_size: int,
    global_batch: int = 8,
) -> tuple[MicroSample, ...]:
    """Partition the existing sampler and masking stream, without rank reseeding."""
    if task not in ("om4", "observation"):
        raise ValueError("Expected om4 or observation task")
    if (
        world_size < 1
        or not 0 <= rank < world_size
        or global_batch < 1
        or global_batch % world_size
        or observation_size < 1
    ):
        raise ValueError("Global batch must divide into nonempty equal rank batches")
    task_seed = seed + (100000 if task == "om4" else 0)
    indices = sample_indices(
        dataset_size, task_seed, completed_task_updates, global_batch
    )
    offset = completed_task_updates * global_batch
    return tuple(
        MicroSample(
            global_micro=micro,
            sample_index=indices[micro],
            mask_seed=task_seed + 1000000 + offset + micro,
            coverage_index=(offset + micro) % observation_size,
        )
        for micro in range(rank, global_batch, world_size)
    )


Objective = Callable[[nn.Module, Any], torch.Tensor]


class TaskLoss(nn.Module):
    """Make the complete task loss a DDP forward, including checkpointed calls."""

    def __init__(self, model: nn.Module, objectives: Mapping[str, Objective]):
        super().__init__()
        if not objectives or set(objectives) - {"om4", "observation"}:
            raise ValueError("Supply observation and/or OM4 objective callbacks")
        self.model = model
        self.objectives = dict(objectives)

    def forward(self, task: str, batch: Any) -> torch.Tensor:
        loss = self.objectives[task](self.model, batch)
        if loss.ndim != 0 or not loss.requires_grad:
            raise ValueError("Task callback must return a differentiable scalar loss")
        return loss


def distributed_task_loss(
    model: nn.Module,
    objectives: Mapping[str, Objective],
    *,
    device_ids: list[int] | None = None,
    process_group=None,
) -> DDP:
    """Wrap a ready model; do not initialize distributed state or move devices."""
    if not dist.is_initialized():
        raise RuntimeError("Initialize the process group before wrapping task loss")
    return DDP(
        TaskLoss(model, objectives),
        device_ids=device_ids,
        process_group=process_group,
        find_unused_parameters=True,
        broadcast_buffers=False,
    )


def accumulate_task_gradients(
    ddp: DDP,
    task: str,
    samples: Sequence[MicroSample],
    load_sample: Callable[[MicroSample], Any],
    *,
    global_batch: int = 8,
    autocast: Callable[[], AbstractContextManager] = nullcontext,
) -> float:
    """Accumulate a global mean; caller subsequently checks/clips/steps once.

    Loader/forward exceptions require the runner's fail-fast launcher to stop
    peer ranks. Nonfinite scalar losses are reduced before backward so every
    rank fails together. No optimizer step or sampling counter is advanced here.
    """
    group = ddp.process_group
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if (
        global_batch < 1
        or global_batch % world
        or len(samples) != global_batch // world
        or [s.global_micro for s in samples] != list(range(rank, global_batch, world))
    ):
        raise ValueError("Local samples must be the rank-strided global batch")
    if any(parameter.grad is not None for parameter in ddp.parameters()):
        raise ValueError("Zero gradients with set_to_none=True before each update")
    ddp.train()
    total = torch.zeros((), dtype=torch.float64, device=next(ddp.parameters()).device)
    for local_micro, sample in enumerate(samples):
        context = ddp.no_sync() if local_micro < len(samples) - 1 else nullcontext()
        with context:
            batch = load_sample(sample)
            with autocast():
                loss = ddp(task, batch)
            finite = torch.isfinite(loss.detach()).to(torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=group)
            if not finite.item():
                raise FloatingPointError("Nonfinite task loss on at least one rank")
            (loss / len(samples)).backward()
            total += loss.detach().double() / len(samples)
        del loss, batch
    dist.all_reduce(total, group=group)
    return float(total / world)


def capture_rank_rng(*, cuda_device=None, process_group=None) -> dict[str, Any]:
    """Capture this rank only; explicitly name CUDA device to include its RNG."""
    distributed = dist.is_initialized()
    state = {
        "rank": dist.get_rank(process_group) if distributed else 0,
        "world_size": dist.get_world_size(process_group) if distributed else 1,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": None,
    }
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        state["torch_cuda"] = torch.cuda.get_rng_state(cuda_device).cpu().clone()
    return state


def gather_rank_rng(*, cuda_device=None, process_group=None) -> dict[str, Any]:
    """Collect small per-rank RNG records on all ranks, for one atomic checkpoint."""
    local = capture_rank_rng(cuda_device=cuda_device, process_group=process_group)
    states: list[Any] = [None] * local["world_size"]
    if dist.is_initialized():
        dist.all_gather_object(states, local, group=process_group)
    else:
        states[0] = local
    return {"version": 1, "world_size": local["world_size"], "ranks": states}


def restore_rank_rng(record, *, cuda_device=None, process_group=None) -> None:
    """Restore only the current rank, rejecting changed topology or CUDA scope."""
    rank = dist.get_rank(process_group) if dist.is_initialized() else 0
    world = dist.get_world_size(process_group) if dist.is_initialized() else 1
    states = record.get("ranks", [])
    if (
        record.get("version") != 1
        or record.get("world_size") != world
        or len(states) != world
        or any(s["rank"] != i or s["world_size"] != world for i, s in enumerate(states))
    ):
        raise ValueError("RNG checkpoint rank topology differs")
    state = states[rank]
    if (state["torch_cuda"] is None) != (cuda_device is None):
        raise ValueError("Restore must use the captured CUDA RNG scope")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if cuda_device is not None:
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), cuda_device)
