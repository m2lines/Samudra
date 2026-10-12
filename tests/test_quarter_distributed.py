# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import datetime
import io
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.checkpoint import checkpoint

from samudra.experiments.quarter_distributed import (
    TaskLoss,
    accumulate_task_gradients,
    distributed_task_loss,
    gather_rank_rng,
    rank_samples,
    restore_rank_rng,
)
from samudra.experiments.task_schedule import sample_indices


@pytest.mark.parametrize("task", ["om4", "observation"])
@pytest.mark.parametrize("world", [4, 8])
def test_partition_matches_historical_eight_example_stream(task, world):
    all_samples = [
        sample
        for rank in range(world)
        for sample in rank_samples(
            task=task,
            dataset_size=11,
            observation_size=7,
            seed=1729,
            completed_task_updates=5,
            rank=rank,
            world_size=world,
        )
    ]
    ordered = sorted(all_samples, key=lambda sample: sample.global_micro)
    seed = 1729 + (100000 if task == "om4" else 0)
    assert [s.sample_index for s in ordered] == sample_indices(11, seed, 5, 8)
    assert [s.mask_seed for s in ordered] == list(range(seed + 1000040, seed + 1000048))
    assert [s.coverage_index for s in ordered] == [i % 7 for i in range(40, 48)]
    assert [s.global_micro for s in all_samples] == (
        [0, 4, 1, 5, 2, 6, 3, 7] if world == 4 else list(range(8))
    )


@pytest.mark.parametrize(
    "world,rank,batch", [(3, 0, 8), (4, 4, 8), (4, -1, 8), (4, 0, 2), (0, 0, 8)]
)
def test_bad_partition_is_rejected(world, rank, batch):
    with pytest.raises(ValueError):
        rank_samples(
            task="om4",
            dataset_size=11,
            observation_size=7,
            seed=1729,
            completed_task_updates=0,
            rank=rank,
            world_size=world,
            global_batch=batch,
        )


class ToyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(3, 4)
        self.input_adapters = nn.ModuleDict(
            {task: nn.Linear(3, 3) for task in ("om4", "observation")}
        )
        self.era5_adapter = nn.Linear(3, 3)
        self.heads = nn.ModuleDict(
            {task: nn.Linear(4, 1) for task in ("om4", "observation")}
        )
        self.auxiliary_head = nn.Linear(4, 1)


def example(sample):
    return (
        torch.tensor(
            [
                [
                    sample.sample_index / 11,
                    sample.coverage_index / 7,
                    sample.mask_seed % 17 / 17,
                ]
            ],
            dtype=torch.float64,
        ),
        torch.tensor([[sample.sample_index % 3 / 3]], dtype=torch.float64),
    )


def task_objective(core, batch, task, stochastic=False):
    value, target = batch
    if stochastic:
        value = (
            value
            + random.random()
            + float(np.random.rand())
            + torch.rand((), dtype=value.dtype)
        )
    if task == "observation":
        value = core.era5_adapter(value)
    value = core.input_adapters[task](value)
    # Exercise the non-reentrant activation checkpoint path used by the ocean model.
    hidden = checkpoint(core.shared, value, use_reentrant=False).tanh()
    loss = (core.heads[task](hidden) - target).square().mean()
    if task == "om4":
        loss = loss + 0.01 * (core.auxiliary_head(hidden) - 0.3).square().mean()
    return loss


def equal_state(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            equal_state(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            equal_state(a, b)
    else:
        assert actual == expected


def distributed_worker(rank, rendezvous, output, world):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=world,
        timeout=datetime.timedelta(seconds=90),
    )
    try:
        torch.manual_seed(73)
        core = ToyCore().double()
        reference = copy.deepcopy(core)
        sync_flags = []
        stochastic = False

        def observation(model, batch):
            sync_flags.append(ddp.require_backward_grad_sync)
            return task_objective(model, batch, "observation", stochastic)

        def om4(model, batch):
            sync_flags.append(ddp.require_backward_grad_sync)
            return task_objective(model, batch, "om4", stochastic)

        ddp = distributed_task_loss(core, {"observation": observation, "om4": om4})
        optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-4, weight_decay=0.01)
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=1e-4, weight_decay=0.01
        )
        for count, task in enumerate(("om4", "observation", "om4")):
            arguments: dict[str, Any] = dict(
                task=task,
                dataset_size=11,
                observation_size=7,
                seed=1729,
                completed_task_updates=count,
            )
            local = rank_samples(**arguments, rank=rank, world_size=world)
            whole = rank_samples(**arguments, rank=0, world_size=1)
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            sync_flags.clear()
            loss = accumulate_task_gradients(ddp, task, local, example)
            assert sync_flags == ([False, True] if world == 4 else [True])
            expected_loss = 0.0
            for sample in whole:
                scalar = task_objective(reference, example(sample), task)
                (scalar / 8).backward()
                expected_loss += float(scalar.detach()) / 8
            assert loss == pytest.approx(expected_loss, abs=1e-13)
            for (name, parameter), (ref_name, ref_parameter) in zip(
                core.named_parameters(), reference.named_parameters(), strict=True
            ):
                assert name == ref_name
                if ref_parameter.grad is None:
                    assert parameter.grad is None, name
                else:
                    torch.testing.assert_close(
                        parameter.grad, ref_parameter.grad, atol=1e-13, rtol=1e-12
                    )
            other = "observation" if task == "om4" else "om4"
            assert all(p.grad is None for p in core.input_adapters[other].parameters())
            if task == "observation":
                assert all(p.grad is None for p in core.auxiliary_head.parameters())
            else:
                assert all(p.grad is None for p in core.era5_adapter.parameters())
            optimizer.step()
            reference_optimizer.step()
            for actual, expected in zip(
                core.parameters(), reference.parameters(), strict=True
            ):
                torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)

        # Serialized checkpoint -> stochastic next update -> restore -> same update.
        # Every rank has distinct Python/NumPy/torch streams, so accidentally
        # restoring rank zero on all workers cannot pass this replay check.
        stochastic = True
        random.seed(200 + rank)
        np.random.seed(300 + rank)
        torch.manual_seed(400 + rank)
        payload = dict(
            model=copy.deepcopy(ddp.module.state_dict()),
            optimizer=copy.deepcopy(optimizer.state_dict()),
            rng=gather_rank_rng(),
        )
        stream = io.BytesIO()
        torch.save(payload, stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=False)
        assert saved["rng"]["ranks"][rank]["rank"] == rank
        states = saved["rng"]["ranks"]
        assert not torch.equal(states[0]["torch_cpu"], states[1]["torch_cpu"])
        local = rank_samples(
            task="observation",
            dataset_size=11,
            observation_size=7,
            seed=1729,
            completed_task_updates=7,
            rank=rank,
            world_size=world,
        )
        optimizer.zero_grad(set_to_none=True)
        native_loss = accumulate_task_gradients(ddp, "observation", local, example)
        optimizer.step()
        expected_model = copy.deepcopy(ddp.module.state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        ddp.module.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        random.random(), np.random.rand(), torch.rand(4)
        restore_rank_rng(saved["rng"])
        optimizer.zero_grad(set_to_none=True)
        replay_loss = accumulate_task_gradients(ddp, "observation", local, example)
        optimizer.step()
        assert replay_loss == native_loss
        equal_state(ddp.module.state_dict(), expected_model)
        equal_state(optimizer.state_dict(), expected_optimizer)

        # A bad loss on one worker is announced before backward on all workers.
        ddp.module.objectives["observation"] = lambda model, batch: task_objective(
            model, batch, "observation"
        ) * (float("nan") if rank == 2 else 1)
        optimizer.zero_grad(set_to_none=True)
        with pytest.raises(FloatingPointError, match="at least one rank"):
            accumulate_task_gradients(
                ddp, "observation", local, example, autocast=nullcontext
            )
        Path(output, f"rank-{rank}.ok").write_text(
            "gradient parity; unused routing; exact serialized RNG replay; collective failure\n"
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [4, 8])
def test_global_batch_unused_parameters_and_serialized_resume(tmp_path, world):
    mp.spawn(
        distributed_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path), world),
        nprocs=world,
        join=True,
    )
    assert len(list(tmp_path.glob("rank-*.ok"))) == world


def test_rng_scope_and_topology_rejected_before_mutation():
    record = gather_rank_rng()
    before = torch.get_rng_state().clone()
    wrong = copy.deepcopy(record)
    wrong["world_size"] = 4
    with pytest.raises(ValueError, match="topology"):
        restore_rank_rng(wrong)
    with pytest.raises(ValueError, match="CUDA RNG scope"):
        restore_rank_rng(record, cuda_device=0)
    assert torch.equal(before, torch.get_rng_state())


def test_wrapper_rejects_detached_loss():
    wrapped = TaskLoss(nn.Linear(1, 1), {"observation": lambda *_: torch.tensor(1.0)})
    with pytest.raises(ValueError, match="differentiable scalar"):
        wrapped("observation", None)


@pytest.mark.cuda
def test_serialized_cuda_rng_restores_cpu_and_device_draws():
    if not torch.cuda.is_available():
        pytest.skip("CUDA RNG qualification requires a CUDA device")
    device = torch.cuda.current_device()
    random.seed(171)
    np.random.seed(172)
    torch.manual_seed(173)
    stream = io.BytesIO()
    torch.save(gather_rank_rng(cuda_device=device), stream)
    stream.seek(0)
    checkpoint_state = torch.load(stream, weights_only=False)

    def draws():
        return (
            random.random(),
            np.random.rand(),
            torch.rand(5),
            torch.rand(5, device=device).cpu(),
        )

    expected = draws()
    draws()
    restore_rank_rng(checkpoint_state, cuda_device=device)
    equal_state(draws(), expected)
