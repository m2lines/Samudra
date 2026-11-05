# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare the actual accumulated DDP update with one full global batch."""

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from samudra.skyline.coverage import corrected_mean
from samudra.skyline.ema import restore_ema
from samudra.skyline.model import Vocabulary
from samudra.skyline.optimization import accumulate_gradients


class BatchNetwork(torch.nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(3, 4), torch.nn.SiLU(), torch.nn.Linear(4, outputs)
        )

    def forward(self, batch):
        return self.layers(batch["values"])


def model_and_data(objective):
    torch.manual_seed(741)
    model = BatchNetwork(7 if objective == "absolute_ce" else 1).double()
    batch = dict(
        values=torch.arange(48, dtype=torch.float64).reshape(16, 3) / 11,
        target=torch.linspace(-0.8, 0.8, 16, dtype=torch.float64),
        depth=torch.arange(16) % 19,
        importance=torch.linspace(0.25, 1.75, 16, dtype=torch.float64),
    )
    return model, batch, Vocabulary(np.full(19, 2.0), 7)


def ddp_accumulation_worker(rank, init_path, output, objective):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method="file://" + init_path, rank=rank, world_size=4
    )
    try:
        model, batch, vocab = model_and_data(objective)
        ddp = DistributedDataParallel(model)
        counter = dict(calls=0)

        def counted_reduce(state, bucket):
            state["calls"] += 1
            group = dist.group.WORLD
            assert group is not None
            return default_hooks.allreduce_hook(group, bucket)

        ddp.register_comm_hook(counter, counted_reduce)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
        ema = restore_ema(model, dict(decay=0.999, faster_decay_at_start=True))
        microbatches = iter(
            [
                {k: v[start : start + 2] for k, v in batch.items()}
                for start in (rank * 4, rank * 4 + 2)
            ]
        )
        optimizer.zero_grad(set_to_none=True)
        stats = accumulate_gradients(
            ddp,
            lambda: next(microbatches),
            objective,
            vocab,
            torch.device("cpu"),
            accumulation=2,
        )
        gradients = [p.grad.clone() for p in model.parameters()]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        ema(model)
        torch.save(
            dict(
                model=model.state_dict(),
                gradients=gradients,
                ema=ema.get_state(True),
                reductions=counter["calls"],
                stats=stats,
            ),
            f"{output}/rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("objective", ["absolute_ce", "regression"])
def test_four_rank_accumulation_matches_full_batch_update_and_ema(tmp_path, objective):
    mp.spawn(
        ddp_accumulation_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path), objective),
        nprocs=4,
        join=True,
    )
    model, batch, vocab = model_and_data(objective)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    ema = restore_ema(model, dict(decay=0.999, faster_decay_at_start=True))
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch)
    losses = (
        F.cross_entropy(
            logits, vocab.encode(batch["target"], batch["depth"]), reduction="none"
        )
        if objective == "absolute_ce"
        else F.mse_loss(logits.squeeze(-1), batch["target"], reduction="none")
    )
    loss = corrected_mean(losses, batch["importance"])
    loss.backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    ema(model)
    stats = []
    for rank in range(4):
        state = torch.load(tmp_path / f"rank-{rank}.pt", weights_only=False)
        assert (
            state["reductions"] == 1
        )  # Intermediate forward AND backward were in no_sync.
        assert int(state["ema"]["num_updates"]) == 1
        for actual, expected in zip(state["gradients"], gradients, strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(
                state["model"][key], value, rtol=1e-10, atol=1e-12
            )
        for key, value in ema.get_state(True)["ema_params"].items():
            torch.testing.assert_close(
                state["ema"]["ema_params"][key], value, rtol=1e-10, atol=1e-12
            )
        stats.append(state["stats"])
    expected = torch.stack(
        (loss.detach(), batch["importance"].mean(), batch["importance"].square().mean())
    ).float()
    torch.testing.assert_close(torch.stack(stats).mean(0), expected)


def test_accumulation_rejects_unequal_microbatches():
    model, batch, vocab = model_and_data("regression")
    microbatches = iter([{k: v[:n] for k, v in batch.items()} for n in (2, 1)])
    with pytest.raises(ValueError, match="equal nonempty"):
        accumulate_gradients(
            model,
            lambda: next(microbatches),
            "regression",
            vocab,
            torch.device("cpu"),
            2,
        )
