"""DDP must all-reduce once per step, not once per forward/backward chunk.

A face microbatch is split into chunks because no GPU holds nine 752^2 tiles'
activations at once. Each chunk runs its own backward, so without `no_sync`
DDP all-reduces the full parameter set on every one of them -- three to five
rank-wide syncs per step where one is needed.

Nothing else in the suite can catch that. `test_face_training_e2e` runs on one
process and never builds a DDP model at all, so the regression this file
guards is invisible to it: the gradients stay correct (all-reduce is linear),
only the wall clock moves.
"""

import dataclasses
import os
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from ocean_emulators.train import Trainer

TILES = 6
CHUNKS = [(0, 1), (2, 3), (4, 5)]  # three chunks of two tiles
CHANNELS, SIZE = 2, 4


# --------------------------------------------------------------------------
# The decision itself
# --------------------------------------------------------------------------


def _trainer(
    *, use_no_sync: bool = True, static_graph: bool = False, accumulation: int = 1
):
    """Just the attributes `_ddp_withholds_allreduce` reads."""
    stub = object.__new__(Trainer)
    stub.ddp_use_no_sync_for_accumulation = use_no_sync
    stub.ddp_static_graph = static_graph
    stub.gradient_accumulation_steps = accumulation
    return stub


def test_chunks_withhold_the_allreduce_even_without_accumulation() -> None:
    """The regression. One accumulation step is the 1-face config."""
    assert _trainer(accumulation=1)._ddp_withholds_allreduce(object(), per_chunk=True)


def test_a_single_backward_still_needs_accumulation_to_withhold() -> None:
    """The curriculum path has no chunks, so only accumulation makes a boundary."""
    trainer = _trainer(accumulation=1)
    assert not trainer._ddp_withholds_allreduce(object(), per_chunk=False)
    assert _trainer(accumulation=2)._ddp_withholds_allreduce(object(), per_chunk=False)


@pytest.mark.parametrize("per_chunk", [True, False])
def test_no_ddp_no_static_graph_and_the_opt_out_all_win(per_chunk: bool) -> None:
    assert not _trainer()._ddp_withholds_allreduce(None, per_chunk=per_chunk)
    assert not _trainer(static_graph=True)._ddp_withholds_allreduce(
        object(), per_chunk=per_chunk
    )
    assert not _trainer(use_no_sync=False)._ddp_withholds_allreduce(
        object(), per_chunk=per_chunk
    )


# --------------------------------------------------------------------------
# What DDP actually does, over gloo
# --------------------------------------------------------------------------


@dataclasses.dataclass
class _Batch:
    """The slice of `TrainData` that `_replay_forward_backward` touches."""

    inputs: torch.Tensor
    labels: torch.Tensor

    def get_input(self, _index: int) -> torch.Tensor:
        return self.inputs

    def get_label(self, _index: int) -> torch.Tensor:
        return self.labels

    def slice_batch(self, start: int, stop: int) -> "_Batch":
        return _Batch(self.inputs[start:stop], self.labels[start:stop])


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(CHANNELS, CHANNELS, 1)

    def forward(self, batch: _Batch):
        return (self.conv(batch.get_input(0)),)


def _run_one_step(*, use_no_sync: bool, seed: int):
    """One `_replay_forward_backward` over three chunks. Returns (syncs, grads)."""
    torch.manual_seed(seed)
    model = _Model()
    ddp = DistributedDataParallel(model)

    counter = {"buckets": 0}

    def hook(state, bucket):
        state["buckets"] += 1
        future: torch.futures.Future = torch.futures.Future()
        buffer = bucket.buffer()
        dist.all_reduce(buffer)
        buffer /= dist.get_world_size()
        future.set_result(buffer)
        return future

    ddp.register_comm_hook(counter, hook)

    generator = torch.Generator().manual_seed(seed + dist.get_rank())
    data = _Batch(
        torch.randn(TILES, CHANNELS, SIZE, SIZE, generator=generator),
        torch.randn(TILES, CHANNELS, SIZE, SIZE, generator=generator),
    )

    trainer = object.__new__(Trainer)
    trainer.model = ddp
    trainer.fp_ctx = types.SimpleNamespace(
        local_tiles=tuple(range(TILES)), chunks=CHUNKS
    )
    trainer.tile_wet_masks = None
    trainer.replay_cfg = types.SimpleNamespace(blend_before_backward=False)
    trainer._loss_denominator_is_fixed = True
    trainer.train_loss_fn = lambda pred, target, sample_weight=None: (
        (pred - target) ** 2
    ).mean(dim=(0, 2, 3))

    prepared = types.SimpleNamespace(
        request=types.SimpleNamespace(train_slots=[]),
    )
    trainer._replay_forward_backward(
        data,
        prepared,
        scale=1.0,
        ddp_model=ddp,
        use_no_sync=use_no_sync,
        sync_gradients=True,
    )
    grads = [parameter.grad.clone() for parameter in model.parameters()]
    return counter["buckets"], grads


def _worker(rank: int, world_size: int, port: str, queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        withheld, withheld_grads = _run_one_step(use_no_sync=True, seed=0)
        every, every_grads = _run_one_step(use_no_sync=False, seed=0)
        queue.put(
            (
                rank,
                withheld,
                every,
                [
                    bool(torch.allclose(a, b, atol=1e-6))
                    for a, b in zip(withheld_grads, every_grads)
                ],
            )
        )
    except BaseException as error:  # surfaced by the parent, not swallowed
        queue.put(("error", rank, repr(error)))
        raise
    finally:
        dist.destroy_process_group()


def test_one_allreduce_per_step_not_one_per_chunk() -> None:
    """Three chunks, one sync -- and the same gradients either way.

    The bucket count is asserted as a ratio rather than an absolute, because
    how many buckets DDP splits a model into is its business and it rebuilds
    them after the first iteration. Three times as many syncs for the same
    work is the defect; 3x1 or 3x2 are both it.
    """
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_worker, args=(rank, 2, "29744", queue))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=180) for _ in processes]
    for process in processes:
        process.join(timeout=60)

    errors = [row for row in results if row[0] == "error"]
    assert not errors, errors

    for _rank, withheld, every, grads_match in results:
        assert withheld >= 1
        assert every == len(CHUNKS) * withheld, (
            f"expected {len(CHUNKS)}x the syncs without no_sync, "
            f"got {every} against {withheld}"
        )
        # All-reduce is linear, so withholding it must not move the answer.
        assert all(grads_match)
