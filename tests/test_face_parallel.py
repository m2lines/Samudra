"""Sharding one face-sized replay row across ranks.

The properties that matter are not about speed. Every rank must run the same
number of forward/backward chunks or DDP deadlocks; the ranks must agree about
divergence or their buffers drift apart; and the loss denominator must not
depend on which tiles landed on which rank, or the face's two all-land tiles
quietly reweight everything.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ocean_emulators.face_parallel import (
    FaceParallelContext,
    assign_tiles,
    face_group_is_shardable,
    local_tile_windows,
    split_into_chunks,
)
from ocean_emulators.tiling import (
    TileSpec,
    build_group_layout,
    face_tile_windows,
    ownership_masks,
    tile_catalog_from_windows,
)

EXTENT = 240
TILE = 40
OVERLAP = 4


def face_layout():
    return build_group_layout(
        tile_catalog_from_windows(
            face_tile_windows(1, extent=EXTENT, tile=TILE, overlap=OVERLAP)
        )
    )


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------


@pytest.mark.parametrize("world_size", [1, 4, 6, 9, 12, 36])
def test_every_tile_is_owned_by_exactly_one_rank(world_size: int) -> None:
    owned: list[int] = []
    for rank in range(world_size):
        assignment = assign_tiles(36, world_size, rank)
        assert all(assignment.owns(tile) for tile in assignment.local_tiles)
        owned.extend(assignment.local_tiles)
    assert sorted(owned) == list(range(36))


@pytest.mark.parametrize("world_size", [1, 4, 6, 9, 12, 36])
def test_every_rank_holds_the_same_number_of_tiles(world_size: int) -> None:
    """Unequal counts mean unequal chunk counts, and a rank that never reaches
    the gradient all-reduce the others are blocked in."""
    counts = {assign_tiles(36, world_size, r).tiles_per_rank for r in range(world_size)}
    assert counts == {36 // world_size}


def test_a_world_that_does_not_divide_the_face_is_rejected_with_the_options() -> None:
    """36 tiles over the 8-GPU nodes is the case this will actually hit."""
    with pytest.raises(ValueError, match="do not divide") as caught:
        assign_tiles(36, 8, 0)
    assert "deadlock" in str(caught.value)
    assert "[1, 2, 3, 4, 6, 9, 12, 18, 36]" in str(caught.value)


def test_ranks_get_contiguous_blocks_for_chunk_locality() -> None:
    """Scattered tiles would multiply the store chunks a rank has to decode."""
    tiles = assign_tiles(36, 4, 0).local_tiles
    assert tiles == (0, 1, 2, 6, 7, 8, 12, 13, 14)


def test_a_rank_outside_the_world_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside a world"):
        assign_tiles(36, 4, 4)


def test_local_windows_follow_the_ranks_own_ordering() -> None:
    windows = face_tile_windows(1, extent=EXTENT, tile=TILE, overlap=OVERLAP)
    assignment = assign_tiles(36, 4, 1)
    local = local_tile_windows(windows, assignment)
    assert local == [windows[tile] for tile in assignment.local_tiles]


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "per_chunk", "expected"),
    [
        (9, 3, [3, 3, 3]),
        (9, 4, [3, 3, 3]),  # evened out rather than 4/4/1
        (4, 4, [4]),
        (6, 3, [3, 3]),
        (3, 3, [3]),
        (5, 2, [2, 2, 1]),  # the short chunk lands last
        (1, 3, [1]),
    ],
)
def test_chunks_are_evened_out_rather_than_greedily_packed(
    count: int, per_chunk: int, expected: list[int]
) -> None:
    """Greedy 4/4/1 has the same chunk count as 3/3/3 but a higher peak and a
    last chunk that wastes most of the GPU."""
    chunks = split_into_chunks(count, per_chunk)
    assert [len(chunk) for chunk in chunks] == expected
    assert [position for chunk in chunks for position in chunk] == list(range(count))
    assert max(len(chunk) for chunk in chunks) <= max(per_chunk, 1)


@pytest.mark.parametrize("world_size", [4, 6, 9, 12])
def test_every_rank_runs_the_same_number_of_chunks(world_size: int) -> None:
    counts = {
        len(split_into_chunks(assign_tiles(36, world_size, r).tiles_per_rank, 3))
        for r in range(world_size)
    }
    assert len(counts) == 1


def test_a_rank_owning_nothing_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one tile"):
        split_into_chunks(0, 3)


# --------------------------------------------------------------------------
# Shardability
# --------------------------------------------------------------------------


def test_a_uniform_single_face_group_is_shardable() -> None:
    assert face_group_is_shardable(face_layout(), 4) == (True, "")


def test_mixed_tile_shapes_are_reported_rather_than_silently_broken() -> None:
    """The shrink-the-boundary-tile geometry lands here: it is the failure the
    clamped origins exist to avoid."""
    tiles = [
        TileSpec(tile_id=0, dataset_index=0, face=1, i_start=0, i_end=48,
                 j_start=0, j_end=48, owned=(0, 48, 0, 48)),
        TileSpec(tile_id=1, dataset_index=1, face=1, i_start=40, i_end=80,
                 j_start=0, j_end=48, owned=(0, 48, 0, 40)),
    ]
    ok, reason = face_group_is_shardable(build_group_layout(tiles), 2)
    assert not ok
    assert "shapes" in reason


def test_an_indivisible_world_is_reported() -> None:
    ok, reason = face_group_is_shardable(face_layout(), 8)
    assert not ok
    assert "do not divide" in reason


# --------------------------------------------------------------------------
# Single-rank context
# --------------------------------------------------------------------------


def test_one_rank_owns_the_whole_face_and_needs_no_collective() -> None:
    context = FaceParallelContext(face_layout(), world_size=1, rank=0, tiles_per_chunk=9)
    assert context.local_tiles == tuple(range(36))
    assert context.blender.exchanges == 0
    assert context.agree(False) is False
    assert context.agree(True) is True


def test_local_ownership_masks_are_the_ranks_slice_of_the_global_ones() -> None:
    layout = face_layout()
    context = FaceParallelContext(layout, world_size=4, rank=2, tiles_per_chunk=3)
    assert torch.equal(
        context.local_ownership_masks(),
        ownership_masks(layout)[list(context.local_tiles)],
    )


def test_ownership_masks_partition_the_face_across_ranks() -> None:
    """Every cell scored exactly once, summed over all four ranks."""
    layout = face_layout()
    total = 0.0
    for rank in range(4):
        context = FaceParallelContext(layout, world_size=4, rank=rank, tiles_per_chunk=3)
        total += float(context.local_ownership_masks().sum())
    assert total == float(EXTENT * EXTENT)


def test_the_denominator_of_a_single_rank_is_the_whole_face_count() -> None:
    layout = face_layout()
    context = FaceParallelContext(layout, world_size=1, rank=0, tiles_per_chunk=9)
    channels = 3
    wet = torch.ones(36, channels, TILE + 2 * OVERLAP, TILE + 2 * OVERLAP)
    local = wet * context.local_ownership_masks()
    denominator = context.global_wet_denominator(local)
    assert denominator.tolist() == pytest.approx([EXTENT * EXTENT] * channels)


def test_a_mis_sized_wet_mask_is_rejected() -> None:
    context = FaceParallelContext(face_layout(), world_size=4, rank=0, tiles_per_chunk=3)
    with pytest.raises(ValueError, match="owns 9"):
        context.global_wet_denominator(torch.ones(36, 2, 48, 48))


# --------------------------------------------------------------------------
# Multi-rank behaviour
# --------------------------------------------------------------------------


def _worker(rank: int, world_size: int, port: str, queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        layout = face_layout()
        context = FaceParallelContext(
            layout, world_size=world_size, rank=rank, tiles_per_chunk=3
        )
        channels = 3
        size = TILE + 2 * OVERLAP
        # Land only on the tiles of rank 0, the worst case for per-rank
        # normalization: its wet count is a fraction of everyone else's.
        wet = torch.ones(len(context.local_tiles), channels, size, size)
        if rank == 0:
            wet[:, :, : size // 2, :] = 0.0
        local = wet * context.local_ownership_masks()
        queue.put(
            (
                rank,
                context.num_chunks,
                context.global_wet_denominator(local).tolist(),
                context.agree(rank == world_size - 1),
                context.agree(False),
            )
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [4, 9])
def test_ranks_agree_on_chunks_denominator_and_divergence(world_size: int) -> None:
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_worker, args=(rank, world_size, str(29700 + world_size), queue)
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    try:
        collected = [queue.get(timeout=300) for _ in range(world_size)]
    finally:
        for process in processes:
            process.join(timeout=60)
            if process.is_alive():
                process.terminate()
    assert all(process.exitcode == 0 for process in processes)

    chunk_counts = {entry[1] for entry in collected}
    assert len(chunk_counts) == 1, "unequal chunk counts would deadlock DDP"

    denominators = [tuple(entry[2]) for entry in collected]
    assert len(set(denominators)) == 1, "the denominator must not depend on the rank"

    # One rank raising divergence must carry every rank with it, and a clean
    # step must stay clean everywhere.
    assert all(entry[3] for entry in collected)
    assert not any(entry[4] for entry in collected)


def test_the_denominator_is_the_same_whichever_rank_holds_the_land() -> None:
    """The point of a face-wide denominator: the answer cannot depend on the
    assignment, so land never has to be balanced across ranks."""
    layout = face_layout()
    channels = 2
    size = TILE + 2 * OVERLAP
    single = FaceParallelContext(layout, world_size=1, rank=0, tiles_per_chunk=9)
    wet = torch.ones(36, channels, size, size)
    wet[:6, :, : size // 2, :] = 0.0  # a band of land across the first six tiles
    expected = single.global_wet_denominator(wet * single.local_ownership_masks())

    # The same field, summed rank by rank exactly as the collective would.
    # `expected` came from a world of one, so it is already the face total.
    total = torch.zeros(channels, dtype=torch.float64)
    for rank in range(4):
        context = FaceParallelContext(layout, world_size=4, rank=rank, tiles_per_chunk=3)
        local = wet[list(context.local_tiles)] * context.local_ownership_masks()
        total += local.to(torch.float64).sum(dim=(0, 2, 3))
    assert total.tolist() == pytest.approx(expected.to(torch.float64).tolist())


# --------------------------------------------------------------------------
# Face-wide loss normalization
# --------------------------------------------------------------------------


def _norm_worker(rank: int, world_size: int, port: str, queue) -> None:
    import numpy as np
    import xarray as xr

    from ocean_emulators.config import GradientLossConfig, build_loss_fn
    from ocean_emulators.constants import TensorMap
    from ocean_emulators.utils.multiton import MultitonScope

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        with MultitonScope():
            tensor_map = TensorMap.init_instance("all", "all_fw_noeta")
            channels = len(tensor_map.prognostic_var_names)
            size = TILE + 2 * OVERLAP
            layout = face_layout()
            context = FaceParallelContext(
                layout, world_size=world_size, rank=rank, tiles_per_chunk=3
            )
            wet = torch.ones(channels, size, size)
            # Land concentrated on the low-numbered tiles, so how much of it a
            # rank holds depends entirely on the assignment.
            face_masks = torch.ones(layout.num_tiles, channels, size, size)
            face_masks[:8] = 0.0
            face_masks[8:14, :, : size // 2] = 0.0
            local = face_masks[list(context.local_tiles)]

            denominator, z_norms = context.global_loss_norms(
                wet, local, tiles=local.shape[0]
            )

            generator = torch.Generator().manual_seed(7)
            pred = torch.randn(
                local.shape[0], channels, size, size, generator=generator
            )
            target = torch.randn(
                local.shape[0], channels, size, size, generator=generator
            )
            config = GradientLossConfig(
                type=["gradient_h", "gradient_z"],
                metric="mse_mae",
                lambda_h=0.1,
                lambda_z=0.1,
            )
            loss_fn = build_loss_fn(
                config,
                wet,
                xr.DataArray(np.linspace(-60.0, 60.0, size), dims="lat"),
                torch.device("cpu"),
                channels,
                "constant",
                denominator=denominator,
                gradient_z_norms=z_norms,
            )
            # Each rank's chunks, summed: its share of the face.
            share = sum(
                loss_fn(
                    pred[chunk[0] : chunk[-1] + 1],
                    target[chunk[0] : chunk[-1] + 1],
                    sample_weight=local[chunk[0] : chunk[-1] + 1],
                )
                for chunk in context.chunks
            )
            # `.tolist()`, not the tensor: a tensor crosses an mp.Queue as a
            # shared-memory handle, and the sender can exit before the parent
            # maps it, which surfaces as a bare FileNotFoundError.
            queue.put((rank, denominator.tolist(), share.tolist()))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [4, 9])
def test_the_face_denominator_is_identical_on_every_rank(world_size: int) -> None:
    """Every rank divides by the same face-wide counts, so a rank holding the
    all-land tiles no longer has its few wet cells weighted up."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_norm_worker, args=(rank, world_size, str(29800 + world_size), queue)
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    try:
        collected = [queue.get(timeout=300) for _ in range(world_size)]
    finally:
        for process in processes:
            process.join(timeout=60)
            if process.is_alive():
                process.terminate()
    assert all(process.exitcode == 0 for process in processes)

    denominators = {tuple(entry[1]) for entry in collected}
    assert len(denominators) == 1, "the denominator must not depend on the rank"

    # DDP averages the per-rank losses, and that mean is the face-wide score.
    # Each rank's share is finite and they are not all equal -- a rank holding
    # only land contributes nothing, which is the point.
    shares = [torch.tensor(entry[2]) for entry in collected]
    assert all(torch.isfinite(share).all() for share in shares)
    assert len({round(float(share.sum()), 6) for share in shares}) > 1
