"""The tile-sharded blender must agree with the single-process one.

`TileBlender` scatters every tile onto a canonical grid; for a 36-tile face
that grid does not fit on one GPU, so `DistributedTileBlender` accumulates on
each tile's own extent instead and trades only the shared bands. That is an
algebraic identity, not an approximation, so these tests hold it to bitwise
equality -- a tolerance here would hide exactly the ordering bug the accumulate
-in-source-order design exists to prevent.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ocean_emulators.tiling import (
    DistributedTileBlender,
    TileBlender,
    build_group_layout,
    contiguous_tile_blocks,
    face_tile_windows,
    halo_ops,
    tile_catalog_from_windows,
)

WINDOWS = ["quintic", "kbd"]

# A shrunk face: 6x6 tiles, so every overlap regime the real 4320-cell face has
# (interior seam, wider clamped boundary seam, four-tile corner) is present.
EXTENT = 240
TILE = 40
OVERLAP = 4


def face_layout(*, extent: int = EXTENT, tile: int = TILE, overlap: int = OVERLAP):
    windows = face_tile_windows(1, extent=extent, tile=tile, overlap=overlap)
    return build_group_layout(tile_catalog_from_windows(windows))


def tile_ranks_from_blocks(num_tiles: int, world_size: int) -> list[int]:
    ranks = [0] * num_tiles
    for rank, block in enumerate(contiguous_tile_blocks(num_tiles, world_size)):
        for tile in block:
            ranks[tile] = rank
    return ranks


def random_tiles(layout, channels: int = 3, *, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    height, width = layout.tiles[0].shape
    return torch.randn(
        layout.num_tiles, channels, height, width, generator=generator,
        dtype=torch.float64,
    )


# --------------------------------------------------------------------------
# Geometry of the exchange
# --------------------------------------------------------------------------


def test_halo_ops_are_symmetric_and_address_the_same_cells() -> None:
    """Every contribution has a mirror, and both name the same physical band."""
    layout = face_layout()
    ops = halo_ops(layout)
    pairs = {(op.src_tile, op.dst_tile) for op in ops}
    assert pairs == {(b, a) for a, b in pairs}

    by_pair = {(op.src_tile, op.dst_tile): op for op in ops}
    for (src, dst), op in by_pair.items():
        mirror = by_pair[(dst, src)]
        assert op.src_box == mirror.dst_box
        assert op.dst_box == mirror.src_box
        assert op.shape == mirror.shape


def test_halo_ops_are_ordered_by_source_then_destination() -> None:
    """NCCL matches point-to-point ops by posting order, so the list order is
    part of the contract, not an implementation detail."""
    ops = halo_ops(face_layout())
    keys = [(op.src_tile, op.dst_tile) for op in ops]
    assert keys == sorted(keys)


def test_interior_tile_has_eight_neighbours_and_a_corner_has_three() -> None:
    layout = face_layout()
    counts = {}
    for op in halo_ops(layout):
        counts[op.dst_tile] = counts.get(op.dst_tile, 0) + 1
    assert counts[14] == 8  # interior of a 6x6 grid
    assert counts[0] == 3  # face corner: right, below, diagonal
    assert counts[5] == 3


def test_contiguous_blocks_partition_the_grid() -> None:
    for world_size in (1, 4, 9, 36):
        blocks = contiguous_tile_blocks(36, world_size)
        assert len(blocks) == world_size
        assert sorted(tile for block in blocks for tile in block) == list(range(36))
        assert len({len(block) for block in blocks}) == 1


def test_contiguous_blocks_reject_a_world_that_does_not_divide_the_grid() -> None:
    with pytest.raises(ValueError, match="do not divide evenly"):
        contiguous_tile_blocks(36, 8)


# --------------------------------------------------------------------------
# Single-process equivalence
# --------------------------------------------------------------------------


@pytest.mark.parametrize("window", WINDOWS)
def test_all_tiles_local_reproduces_tileblender_bitwise(window: str) -> None:
    """With one rank there is no exchange, so any difference is pure
    accumulation order -- which is what this pins down."""
    layout = face_layout()
    tiles = random_tiles(layout)
    reference = TileBlender(layout, window=window, dtype=torch.float64).blend(tiles)
    sharded = DistributedTileBlender(
        layout,
        tile_ranks=[0] * layout.num_tiles,
        rank=0,
        window=window,
        dtype=torch.float64,
    ).blend(tiles)
    assert torch.equal(sharded, reference)


@pytest.mark.parametrize("window", WINDOWS)
def test_single_tile_group_is_the_identity(window: str) -> None:
    layout = face_layout(extent=40, tile=40, overlap=0)
    tiles = random_tiles(layout)
    blender = DistributedTileBlender(
        layout, tile_ranks=[0], rank=0, window=window, dtype=torch.float64
    )
    assert torch.equal(blender.blend(tiles), tiles)


def test_denominator_matches_tileblenders_on_every_local_tile() -> None:
    """The denominator is pure geometry; if it drifts, every seam is scaled
    wrong and nothing else in the pipeline would notice."""
    layout = face_layout()
    reference = TileBlender(layout, dtype=torch.float64)
    for rank in range(4):
        blender = DistributedTileBlender(
            layout,
            tile_ranks=tile_ranks_from_blocks(layout.num_tiles, 4),
            rank=rank,
            dtype=torch.float64,
        )
        for slot, tile_index in enumerate(blender.local_tiles):
            j0, j1, i0, i1 = layout.canonical_bounds(layout.tiles[tile_index])
            assert torch.equal(
                blender.denominator[slot, 0],
                reference.denominator[0, 0, j0:j1, i0:i1],
            )


def test_blend_supports_a_leading_batch_dimension() -> None:
    layout = face_layout()
    tiles = random_tiles(layout)
    blender = DistributedTileBlender(
        layout, tile_ranks=[0] * layout.num_tiles, rank=0, dtype=torch.float64
    )
    single = blender.blend(tiles)
    batched = blender.blend(torch.stack([tiles, tiles + 1.0]))
    assert batched.shape == (2, *single.shape)
    assert torch.equal(batched[0], single)


def test_blender_rejects_a_rank_that_owns_nothing() -> None:
    layout = face_layout()
    with pytest.raises(ValueError, match="owns no tile"):
        DistributedTileBlender(layout, tile_ranks=[0] * layout.num_tiles, rank=1)


def test_blender_rejects_a_tile_ranks_of_the_wrong_length() -> None:
    layout = face_layout()
    with pytest.raises(ValueError, match="tile_ranks has"):
        DistributedTileBlender(layout, tile_ranks=[0, 0], rank=0)


def test_blender_rejects_the_wrong_number_of_local_tiles() -> None:
    layout = face_layout()
    blender = DistributedTileBlender(
        layout, tile_ranks=tile_ranks_from_blocks(layout.num_tiles, 4), rank=0
    )
    with pytest.raises(ValueError, match="owns 9 tile"):
        blender.blend(random_tiles(layout).to(torch.float32))


def test_cross_rank_overlaps_without_a_process_group_fail_loudly() -> None:
    """Silently skipping the exchange would leave the seams unreconciled and
    the run would just train slightly wrong."""
    layout = face_layout()
    blender = DistributedTileBlender(
        layout,
        tile_ranks=tile_ranks_from_blocks(layout.num_tiles, 4),
        rank=0,
        dtype=torch.float64,
    )
    local = random_tiles(layout)[list(blender.local_tiles)]
    with pytest.raises(RuntimeError, match="not initialized"):
        blender.blend(local)


# --------------------------------------------------------------------------
# Multi-process equivalence
# --------------------------------------------------------------------------


def _worker(rank: int, world_size: int, port: str, queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        layout = face_layout()
        tiles = random_tiles(layout)
        ranks = tile_ranks_from_blocks(layout.num_tiles, world_size)
        results = {}
        for window in WINDOWS:
            blender = DistributedTileBlender(
                layout,
                tile_ranks=ranks,
                rank=rank,
                window=window,
                dtype=torch.float64,
            )
            local = tiles[list(blender.local_tiles)]
            results[window] = (list(blender.local_tiles), blender.blend(local))
        queue.put((rank, results))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [4, 9])
def test_sharded_blend_reproduces_the_single_process_blend(world_size: int) -> None:
    """The real test: N ranks, each holding a block of the face, must rebuild
    exactly what one process holding all 36 tiles would have produced.

    Both windows go through one spawn group because the cost here is starting
    interpreters, not blending.
    """
    layout = face_layout()
    tiles = random_tiles(layout)
    reference = {
        window: TileBlender(layout, window=window, dtype=torch.float64).blend(tiles)
        for window in WINDOWS
    }

    context = mp.get_context("spawn")
    queue = context.Queue()
    port = str(29500 + world_size)
    processes = [
        context.Process(target=_worker, args=(rank, world_size, port, queue))
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

    for _, results in collected:
        for window, (local_tiles, blended) in results.items():
            for slot, tile_index in enumerate(local_tiles):
                assert torch.equal(blended[slot], reference[window][tile_index]), (
                    f"{window}: tile {tile_index} differs from the single-process blend"
                )
