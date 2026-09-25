"""Scoring a sharded face as if it were one field.

The reference everything is held to is the obvious computation: stitch the
face together from each tile's owned cells, then take the area-weighted RMSE
and the wet-cell mean of the error over it. Whatever the split across ranks,
`FaceScorer` has to return that -- in particular it must not become a mean of
per-rank or per-tile values, which would weigh a mostly-land tile like an open
ocean one.
"""

import os
import threading
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ocean_emulators.face_parallel import assign_tiles
from ocean_emulators.face_validation import FaceScorer, ReadAhead
from ocean_emulators.tiling import (
    build_group_layout,
    face_tile_windows,
    ownership_masks,
    tile_catalog_from_windows,
)

EXTENT = 240
TILE = 40
OVERLAP = 4
CHANNELS = 3


def _face():
    """Random tiles over a face with uneven land and uneven cell areas."""
    windows = face_tile_windows(1, extent=EXTENT, tile=TILE, overlap=OVERLAP)
    layout = build_group_layout(tile_catalog_from_windows(windows))
    generator = torch.Generator().manual_seed(0)
    wet = torch.rand((CHANNELS, EXTENT, EXTENT), generator=generator) > 0.3
    wet[:, :TILE, :TILE] = False  # one tile all land
    wet[2] = False  # and one channel with no wet cell anywhere
    area = 1.0 + torch.rand((EXTENT, EXTENT), generator=generator)
    size = TILE + 2 * OVERLAP
    prediction = torch.randn((36, CHANNELS, size, size), generator=generator)
    target = torch.randn((36, CHANNELS, size, size), generator=generator)
    return windows, layout, wet, area, prediction, target


def _cut(field, window):
    _, i0, i1, j0, j1 = window
    return field[..., j0:j1, i0:i1]


def _stitched_reference():
    """RMSE and MAE of the stitched face, computed directly."""
    windows, layout, wet, area, prediction, target = _face()
    owned = ownership_masks(layout).bool()
    error = torch.full((CHANNELS, EXTENT, EXTENT), float("nan"))
    for tile, window in enumerate(windows):
        _, i0, i1, j0, j1 = window
        view = error[:, j0:j1, i0:i1]
        mask = owned[tile].expand_as(view)
        view[mask] = (prediction[tile] - target[tile])[mask]
    assert not torch.isnan(error).any(), "ownership must cover the face"
    weights = torch.where(wet, area, 0.0).double()
    squared = (error.double() ** 2 * weights).sum(dim=(-2, -1))
    absolute = (error.double().abs() * torch.where(wet, 1.0, 0.0)).sum(dim=(-2, -1))
    total_area = weights.sum(dim=(-2, -1))
    rmse = torch.where(total_area > 0, (squared / total_area).sqrt(), torch.nan)
    mae = absolute / torch.where(wet, 1.0, 0.0).sum(dim=(-2, -1)).clamp_min(1e-8)
    return rmse.float(), mae.float()


def _local_scorer(rank: int, world_size: int, reduce_sum):
    """This rank's scorer and tiles, set up the way the trainer does it."""
    windows, layout, wet, area, prediction, target = _face()
    tiles = list(assign_tiles(36, world_size, rank).local_tiles)
    owned = ownership_masks(layout).bool()[tiles]
    tile_wet = torch.stack([_cut(wet, windows[t]) for t in tiles])
    weight = tile_wet & owned
    # A mean-absolute-error share: fixed denominators, face-wide, carrying the
    # 1/world_size that `global_loss_norms` puts in for DDP's mean.
    denominator = reduce_sum(weight.double().sum(dim=(0, 2, 3))) / world_size

    def loss_fn(pred, targ, sample_weight):
        numerator = ((pred - targ).abs() * sample_weight.float()).sum(dim=(0, 2, 3))
        return (numerator.double() / denominator.clamp_min(1e-8)).float()

    scorer = FaceScorer(
        loss_fn=loss_fn,
        weight=weight,
        wet=torch.ones((CHANNELS, *weight.shape[-2:]), dtype=torch.bool),
        area=torch.stack([_cut(area, windows[t]) for t in tiles]).unsqueeze(1),
        world_size=world_size,
        reduce_sum=reduce_sum,
    )
    return scorer, prediction[tiles], target[tiles]


def test_one_rank_scores_the_stitched_face() -> None:
    rmse, mae = _stitched_reference()
    scorer, prediction, target = _local_scorer(0, 1, lambda tensor: tensor)
    metrics = scorer.score(prediction, target)
    assert torch.allclose(metrics.rmse_per_channel, rmse, equal_nan=True, rtol=1e-5)
    assert torch.allclose(metrics.loss_per_channel[:2], mae[:2], rtol=1e-5)
    # A channel with no wet cell is NaN, and dropped from the channel mean.
    assert torch.isnan(metrics.rmse_per_channel[2])
    assert torch.isclose(metrics.rmse, rmse[:2].mean(), rtol=1e-5)


def test_pooling_is_not_a_mean_of_tile_rmses() -> None:
    """The distinction the pooled definition exists for."""
    rmse, _ = _stitched_reference()
    windows, layout, wet, area, prediction, target = _face()
    owned = ownership_masks(layout).bool()
    per_tile = []
    for tile, window in enumerate(windows):
        mask = _cut(wet, window) & owned[tile]
        weights = torch.where(mask, _cut(area, window), 0.0)
        squared = ((prediction[tile] - target[tile]) ** 2 * weights).sum(dim=(-2, -1))
        per_tile.append((squared / weights.sum(dim=(-2, -1))).sqrt())
    tile_mean = torch.stack(per_tile).nanmean(dim=0)
    assert not torch.allclose(tile_mean[:2], rmse[:2], rtol=1e-3)


def _worker(rank: int, world_size: int, port: str, queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:

        def reduce_sum(tensor):
            tensor = tensor.clone()
            dist.all_reduce(tensor)
            return tensor

        scorer, prediction, target = _local_scorer(rank, world_size, reduce_sum)
        metrics = scorer.score(prediction, target)
        # Plain lists: a tensor sent from a spawned child dies with it.
        queue.put(
            (
                rank,
                metrics.rmse_per_channel.tolist(),
                metrics.loss_per_channel.tolist(),
                float(metrics.rmse),
            )
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [4, 9])
def test_every_rank_reports_the_stitched_face(world_size: int) -> None:
    rmse, mae = _stitched_reference()
    context = mp.get_context("spawn")
    queue = context.Queue()
    port = str(29600 + world_size)
    processes = [
        context.Process(target=_worker, args=(rank, world_size, port, queue))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=120) for _ in processes]
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0

    for rank, rank_rmse, rank_loss, rank_mean in results:
        rank_rmse, rank_loss = torch.tensor(rank_rmse), torch.tensor(rank_loss)
        rank_mean = torch.tensor(rank_mean)
        assert torch.allclose(rank_rmse, rmse, equal_nan=True, rtol=1e-5), rank
        assert torch.allclose(rank_loss[:2], mae[:2], rtol=1e-5), rank
        assert torch.isclose(rank_mean, rmse[:2].mean(), rtol=1e-5), rank


def test_read_ahead_keeps_order_and_bounds_the_lookahead() -> None:
    started: list[int] = []
    lock = threading.Lock()

    def read(index: int):
        with lock:
            started.append(index)
        time.sleep(0.01)
        return index

    reads = [lambda index=index: read(index) for index in range(6)]
    seen = []
    for value in ReadAhead(reads, depth=2):
        # Never more than `depth` reads started beyond what was consumed.
        assert len(started) <= len(seen) + 2 + 1
        seen.append(value)
    assert seen == list(range(6))


def test_read_ahead_stops_reading_when_the_consumer_stops() -> None:
    started: list[int] = []
    reads = [lambda index=index: started.append(index) or index for index in range(50)]
    iterator = iter(ReadAhead(reads, depth=2))
    assert next(iterator) == 0
    iterator.close()
    assert len(started) <= 3


def test_read_ahead_surfaces_a_failed_read() -> None:
    def broken():
        raise OSError("disk went away")

    with pytest.raises(OSError, match="disk went away"):
        list(ReadAhead([lambda: 0, broken, lambda: 2], depth=2))
