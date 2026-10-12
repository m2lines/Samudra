# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU distributed correctness of batch-eight updates and replay checkpointing."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from samudra.experiments.diffusion_interventions import LatentReplay
from samudra.experiments.diffusion_scratch_distributed import (
    capture_rank_state,
    gather_rank_states,
    learning_rate,
    owned_replay_slots,
    restore_rank_state,
    sum_gradients,
    validate_rank_states,
)
from samudra.experiments.task_schedule import TaskSchedule


def _toy_model():
    return torch.nn.ParameterList(
        [
            torch.nn.Parameter(torch.tensor([0.3, -0.2])),
            torch.nn.Parameter(torch.tensor([0.7])),
            torch.nn.Parameter(torch.tensor([1.1])),
            torch.nn.Parameter(torch.tensor([2.0])),  # globally unused
        ]
    )


def _micro_loss(model, micro):
    # Both ranks share p0, but only one rank uses each p1/p2. Merely reducing
    # locally non-None gradients would deadlock or average the wrong parameters.
    return (model[0].sum() * (micro + 1) - 1).square() + model[
        1 + micro % 2
    ].square().sum()


def _replay(rank, world):
    replay = LatentReplay.__new__(LatentReplay)
    replay.wave = SimpleNamespace(device=torch.device("cpu"))
    replay.slots = {
        slot: dict(
            index=slot * 10,
            lead=12 + slot * 6,
            uses=2,
            latent=torch.tensor([float(slot)]),
            reference=torch.tensor([0.5]),
        )
        for slot in owned_replay_slots(rank, world)
    }
    return replay


def _distributed_worker(rank, world, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world
    )
    model = _toy_model()
    optimizer = torch.optim.AdamW(model, lr=0.01, weight_decay=0.1)
    for micro in range(rank, 8, world):
        (_micro_loss(model, micro) / 8).backward()
    sum_gradients(model)
    observed = [p.grad.clone() if p.grad is not None else None for p in model]
    optimizer.step()
    replay = _replay(rank, world)
    states = gather_rank_states(
        capture_rank_state(rank, world, replay, torch.device("cpu")), world, rank
    )
    if rank == 0:
        validate_rank_states(states, world)
        torch.save(
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                rank_states=states,
            ),
            Path(output) / "checkpoint.pt",
        )
    dist.barrier()
    saved = torch.load(Path(output) / "checkpoint.pt", weights_only=False)
    validate_rank_states(saved["rank_states"], world)
    restored = _toy_model()
    restored.load_state_dict(saved["model"])
    restored_optimizer = torch.optim.AdamW(restored, lr=0.01, weight_decay=0.1)
    restored_optimizer.load_state_dict(saved["optimizer"])
    restored_replay = _replay(rank, world)
    restored_replay.slots = {}
    restore_rank_state(saved["rank_states"][rank], restored_replay, torch.device("cpu"))
    for slot, state in replay.state_dict().items():
        for key, value in state.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(
                    value, restored_replay.slots[slot][key], rtol=0, atol=0
                )
            else:
                assert value == restored_replay.slots[slot][key]
    # One further optimizer update must match uninterrupted execution, including
    # Adam moments and non-decay of globally unused parameters.
    for current, opt in ((model, optimizer), (restored, restored_optimizer)):
        opt.zero_grad(set_to_none=True)
        for micro in range(rank, 8, world):
            (_micro_loss(current, micro) / 8).backward()
        sum_gradients(current)
        opt.step()
    for expected, actual in zip(model, restored, strict=True):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    torch.save(
        dict(gradients=observed, model=model.state_dict()),
        Path(output) / f"rank-{rank}.pt",
    )
    dist.destroy_process_group()


def test_distributed_mean_and_optimizer_replay_restart(tmp_path):
    mp.start_processes(
        _distributed_worker,
        args=(2, str(tmp_path / "rdzv"), str(tmp_path)),
        nprocs=2,
        join=True,
        start_method="fork",
    )
    serial = _toy_model()
    optimizer = torch.optim.AdamW(serial, lr=0.01, weight_decay=0.1)
    expected_gradients = None
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(8):
            (_micro_loss(serial, micro) / 8).backward()
        if step == 0:
            expected_gradients = [
                p.grad.clone() if p.grad is not None else None for p in serial
            ]
        optimizer.step()
    assert expected_gradients is not None
    for rank in range(2):
        result = torch.load(tmp_path / f"rank-{rank}.pt", weights_only=False)
        for expected, actual in zip(
            expected_gradients, result["gradients"], strict=True
        ):
            if expected is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        for key, value in serial.state_dict().items():
            torch.testing.assert_close(
                result["model"][key], value, atol=1e-6, rtol=1e-6
            )
        assert float(result["model"]["3"]) == 2.0


def test_replay_ownership_rejects_rank_swaps_and_foreign_slots():
    for world in (1, 2, 4, 8):
        states = [
            dict(
                rank=rank,
                owned_slots=owned_replay_slots(rank, world),
                replay={slot: {} for slot in owned_replay_slots(rank, world)},
            )
            for rank in range(world)
        ]
        validate_rank_states(states, world)
        assert sorted(slot for state in states for slot in state["replay"]) == [
            0,
            1,
            2,
            3,
        ]
    states[0]["replay"][0] = {}
    with pytest.raises(ValueError, match="foreign"):
        validate_rank_states(states, 8)
    with pytest.raises(ValueError, match="order"):
        validate_rank_states(states[::-1], 8)


def test_original_quadratic_exposure_and_cooldown():
    schedule = TaskSchedule(10160, 10160, "mixed")
    milestone = next(
        step
        for step in range(schedule.total)
        if schedule.counts(step)["observation"] == 2000
    )
    assert milestone == 7561
    assert schedule.counts(milestone) == {"om4": 5561, "observation": 2000}
    assert schedule.counts(schedule.total) == {"om4": 10160, "observation": 10160}
    assert learning_rate(0, schedule.total) == 1e-4
    assert learning_rate(17780, schedule.total) == 1e-4
    assert learning_rate(20319, schedule.total) == 1e-6
    assert learning_rate(19000, schedule.total) < learning_rate(18000, schedule.total)
