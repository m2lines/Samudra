"""The chunk-streaming reader must deliver exactly what per-tile reads deliver.

It exists only to be faster, so the whole value of it rests on producing
byte-identical tiles. The equivalence tests here run against a synthetic packed
cache chunked and tiled like the real one, so they catch an off-by-one in the
scatter without needing the 37 TB face cache.
"""

import json

import numpy as np
import pytest
import torch
import xarray as xr

from ocean_emulators.chunk_reader import GroupChunkReader, chunk_plan
from ocean_emulators.tiling import face_tile_windows

# A shrunk face with the real one's proportions: chunks on the tile grid, and
# windows that overhang their chunk on both sides.
EXTENT = 240
TILE = 40
OVERLAP = 4
SIZE = TILE + 2 * OVERLAP
CHANNELS = ["U_0", "U_1", "V_0", "V_1", "Eta"]
BOUNDARY = ["oceTAUX", "oceTAUY"]
TIMES = 3


@pytest.fixture(scope="module")
def packed_store(tmp_path_factory) -> str:
    """A packed cache shaped like `llc-train-ready-v1`, small enough to test."""
    path = tmp_path_factory.mktemp("cache") / "face.zarr"
    generator = np.random.default_rng(0)
    prognostic = generator.standard_normal(
        (TIMES, len(CHANNELS), EXTENT, EXTENT)
    ).astype(np.float16)
    boundary = generator.standard_normal(
        (TIMES, len(BOUNDARY), EXTENT, EXTENT)
    ).astype(np.float16)
    # Land is NaN in the real cache, so the scatter has to carry NaN faithfully.
    prognostic[:, :, :5, :5] = np.nan

    data = xr.Dataset(
        {
            "prognostic": (("time", "prognostic_channel", "y", "x"), prognostic),
            "boundary": (("time", "boundary_channel", "y", "x"), boundary),
        },
        coords={
            "time": np.arange(TIMES, dtype="int64"),
            "y": np.arange(EXTENT, dtype="int32"),
            "x": np.arange(EXTENT, dtype="int32"),
        },
        attrs={
            "prognostic_channel_names_json": json.dumps(CHANNELS),
            "boundary_channel_names_json": json.dumps(BOUNDARY),
            "llc_face": 1,
        },
    )
    data.to_zarr(
        path,
        encoding={
            "prognostic": {"chunks": (1, len(CHANNELS), TILE, TILE)},
            "boundary": {"chunks": (1, len(BOUNDARY), TILE, TILE)},
        },
    )
    return str(path)


def face_windows():
    return face_tile_windows(1, extent=EXTENT, tile=TILE, overlap=OVERLAP)


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def test_each_chunk_appears_once_however_many_tiles_want_it() -> None:
    """The entire point: 36 overlapping windows, 36 chunk decodes at most."""
    local = [(i0, i1, j0, j1) for _, i0, i1, j0, j1 in face_windows()]
    plan = chunk_plan(local, chunk_rows=TILE, chunk_cols=TILE)
    assert len(plan) == (EXTENT // TILE) ** 2
    assert len(set(plan)) == len(plan)


def test_the_plan_covers_every_cell_of_every_tile_exactly_once() -> None:
    """A gap leaves uninitialized memory in a training target; an overlap means
    the scatter is writing the same cell twice from different chunks."""
    local = [(i0, i1, j0, j1) for _, i0, i1, j0, j1 in face_windows()]
    plan = chunk_plan(local, chunk_rows=TILE, chunk_cols=TILE)
    written = [np.zeros((j1 - j0, i1 - i0), dtype=int) for i0, i1, j0, j1 in local]
    for copies in plan.values():
        for copy in copies:
            written[copy.tile][copy.destination[0], copy.destination[1]] += 1
    for tile, counts in enumerate(written):
        assert counts.min() == 1 and counts.max() == 1, f"tile {tile}"


def test_source_and_destination_boxes_have_the_same_shape() -> None:
    local = [(i0, i1, j0, j1) for _, i0, i1, j0, j1 in face_windows()]
    for copies in chunk_plan(local, chunk_rows=TILE, chunk_cols=TILE).values():
        for copy in copies:
            source = tuple(s.stop - s.start for s in copy.source)
            destination = tuple(s.stop - s.start for s in copy.destination)
            assert source == destination


def test_a_rank_sized_block_touches_far_fewer_chunks_than_the_naive_read() -> None:
    """The 3x3 block a rank owns is why chunk locality drives the assignment."""
    windows = face_windows()
    block = [windows[row * 6 + col] for row in range(3) for col in range(3)]
    local = [(i0, i1, j0, j1) for _, i0, i1, j0, j1 in block]
    plan = chunk_plan(local, chunk_rows=TILE, chunk_cols=TILE)
    naive = sum(
        len(range(j0 // TILE, (j1 - 1) // TILE + 1))
        * len(range(i0 // TILE, (i1 - 1) // TILE + 1))
        for i0, i1, j0, j1 in local
    )
    assert len(plan) == 16
    # Column footprints across the face are [2, 3, 3, 3, 3, 2] chunks, so the
    # first three tiles of a row span 8 chunk columns and the block spans 8x8.
    assert naive == 64


def test_an_empty_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty window"):
        chunk_plan([(5, 5, 0, 10)], chunk_rows=4, chunk_cols=4)


# --------------------------------------------------------------------------
# Equivalence with the per-tile read
# --------------------------------------------------------------------------


def naive_tile(store: str, window, names, prefix: str, row: int) -> torch.Tensor:
    """What `TorchTrainDataset._read_prognostic` produces for one tile: the
    store sliced per tile, keeping the length-1 time axis."""
    _, i0, i1, j0, j1 = window
    data = xr.open_zarr(store, chunks=None, decode_times=False)
    available = json.loads(data.attrs[f"{prefix}_channel_names_json"])
    indices = [available.index(name) for name in names]
    array = (
        data[prefix]
        .isel({f"{prefix}_channel": indices})
        .isel(time=row, y=slice(j0, j1), x=slice(i0, i1))
    )
    return torch.from_numpy(array.to_numpy()).unsqueeze(0)


@pytest.mark.parametrize("threads", [1, 4])
def test_every_tile_matches_the_per_tile_read_exactly(packed_store, threads) -> None:
    windows = face_windows()
    reader = GroupChunkReader(
        packed_store,
        prefix="prognostic",
        windows=windows,
        var_names=CHANNELS,
        threads=threads,
    )
    for row in range(TIMES):
        tiles = reader.read(row)
        assert len(tiles) == len(windows)
        for index, window in enumerate(windows):
            want = naive_tile(packed_store, window, CHANNELS, "prognostic", row)
            assert tiles[index].shape == want.shape
            assert torch.equal(tiles[index].nan_to_num(-1.0), want.nan_to_num(-1.0)), (
                f"tile {index} at row {row}"
            )
    reader.close()


def test_the_store_dtype_survives_so_this_is_a_drop_in(packed_store) -> None:
    """The per-tile path keeps float16; casting here would double host traffic
    and quietly change what the trainer pins."""
    reader = GroupChunkReader(
        packed_store, prefix="prognostic", windows=face_windows(), var_names=CHANNELS
    )
    assert reader.read(0)[0].dtype == torch.float16
    reader.close()


def test_land_nans_are_carried_through_rather_than_filled(packed_store) -> None:
    """Land is NaN in the cache and the masking happens downstream; a reader
    that zero-filled would make land indistinguishable from cold water."""
    reader = GroupChunkReader(
        packed_store, prefix="prognostic", windows=face_windows(), var_names=CHANNELS
    )
    assert torch.isnan(reader.read(0)[0]).any()
    reader.close()


def test_a_channel_subset_is_selected_and_ordered_as_asked(packed_store) -> None:
    """Channel order is the model's input order, so a reordering here is a
    silent variable swap."""
    wanted = ["Eta", "U_1"]
    reader = GroupChunkReader(
        packed_store, prefix="prognostic", windows=face_windows(), var_names=wanted
    )
    tiles = reader.read(1)
    assert tiles[0].shape[1] == 2
    for index, window in enumerate(face_windows()):
        want = naive_tile(packed_store, window, wanted, "prognostic", 1)
        assert torch.equal(
            tiles[index].nan_to_num(-1.0), want.nan_to_num(-1.0)
        ), f"tile {index}"
    reader.close()


def test_boundary_channels_read_through_the_same_path(packed_store) -> None:
    reader = GroupChunkReader(
        packed_store, prefix="boundary", windows=face_windows(), var_names=BOUNDARY
    )
    tiles = reader.read(2)
    for index, window in enumerate(face_windows()):
        want = naive_tile(packed_store, window, BOUNDARY, "boundary", 2)
        assert torch.equal(tiles[index], want)
    reader.close()


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_a_window_outside_the_store_is_rejected(packed_store) -> None:
    with pytest.raises(ValueError, match="falls outside"):
        GroupChunkReader(
            packed_store,
            prefix="prognostic",
            windows=[(1, 0, SIZE, EXTENT - 8, EXTENT + SIZE)],
            var_names=CHANNELS,
        )


def test_an_unknown_channel_is_rejected(packed_store) -> None:
    with pytest.raises(KeyError, match="missing requested channels"):
        GroupChunkReader(
            packed_store,
            prefix="prognostic",
            windows=face_windows(),
            var_names=["Theta_0"],
        )


def test_a_row_past_the_end_is_rejected(packed_store) -> None:
    reader = GroupChunkReader(
        packed_store, prefix="prognostic", windows=face_windows(), var_names=CHANNELS
    )
    with pytest.raises(IndexError, match="outside"):
        reader.read(TIMES)
    reader.close()


def test_a_timestamp_resolves_to_its_store_row(packed_store) -> None:
    """The caller's dataset is sliced to a train window, so its positions are
    not store rows; matching on the timestamp is what bridges that."""
    reader = GroupChunkReader(
        packed_store, prefix="prognostic", windows=face_windows(), var_names=CHANNELS
    )
    assert reader.store_row(np.int64(2)) == 2
    with pytest.raises(ValueError, match="matches 0 rows"):
        reader.store_row(np.int64(99))
    reader.close()


# --------------------------------------------------------------------------
# Against the real store
# --------------------------------------------------------------------------

FACE_CACHE = (
    "/orcd/data/abodner/002/cody/LLC_patch/face_1/"
    "LLC4320_face1_i0-4320_j0-4320.zarr"
)


@pytest.mark.manual
def test_matches_the_per_tile_read_on_the_real_face_cache() -> None:
    """The synthetic store above has the right shape but not the real one's
    chunk sizes, dtype quirks or land pattern. Reads ~2.5 GB, hence `manual`:

        uv run pytest -m manual -k real_face_cache
    """
    metadata = xr.open_zarr(FACE_CACHE, chunks=None, decode_times=False)
    names = json.loads(metadata.attrs["prognostic_channel_names_json"])
    windows = face_tile_windows(1)
    # A clamped face corner and a fully interior tile: both overlap regimes.
    picked = [windows[0], windows[14]]

    reader = GroupChunkReader(
        FACE_CACHE, prefix="prognostic", windows=picked, var_names=names, threads=8
    )
    row = 6000
    for tile, window in zip(reader.read(row), picked, strict=True):
        _, i0, i1, j0, j1 = window
        want = torch.from_numpy(
            metadata["prognostic"]
            .isel(time=row, y=slice(j0, j1), x=slice(i0, i1))
            .to_numpy()
        ).unsqueeze(0)
        assert tile.shape == (1, 205, 752, 752)
        assert tile.dtype == torch.float16
        assert torch.equal(tile.nan_to_num(-7.0), want.nan_to_num(-7.0))
    assert reader.store_row(metadata["time"].to_numpy()[row]) == row
    reader.close()


# --------------------------------------------------------------------------
# Both arrays together
# --------------------------------------------------------------------------


def test_group_frame_reader_serves_both_arrays_from_one_store(packed_store) -> None:
    from ocean_emulators.chunk_reader import GroupFrameReader

    windows = face_windows()
    reader = GroupFrameReader(
        packed_store,
        windows=windows,
        prognostic_var_names=CHANNELS,
        boundary_var_names=BOUNDARY,
        threads=4,
    )
    timestamp = np.int64(1)
    for tiles, names, prefix in (
        (reader.read_prognostic(timestamp), CHANNELS, "prognostic"),
        (reader.read_boundary(timestamp), BOUNDARY, "boundary"),
    ):
        assert len(tiles) == len(windows)
        for index, window in enumerate(windows):
            want = naive_tile(packed_store, window, names, prefix, int(timestamp))
            assert torch.equal(
                tiles[index].nan_to_num(-1.0), want.nan_to_num(-1.0)
            ), f"{prefix} tile {index}"
    reader.close()


def test_the_reader_reports_what_the_per_tile_path_would_have_cost(
    packed_store,
) -> None:
    """The saving is the reason this exists, so it is logged; pin the numbers
    that log reports."""
    from ocean_emulators.chunk_reader import GroupFrameReader

    reader = GroupFrameReader(
        packed_store,
        windows=face_windows(),
        prognostic_var_names=CHANNELS,
        boundary_var_names=BOUNDARY,
        threads=2,
    )
    # 36 distinct chunks per array; one read per tile would take 256.
    assert reader.chunks_per_frame == 72
    assert reader.naive_chunks_per_frame == 512
    reader.close()


def test_a_rank_sized_block_reads_only_its_own_chunks(packed_store) -> None:
    """A rank opens the reader on ITS tiles, not the face's, so it never
    decodes a chunk only its neighbours need."""
    from ocean_emulators.chunk_reader import GroupFrameReader
    from ocean_emulators.face_parallel import assign_tiles, local_tile_windows

    windows = face_windows()
    assignment = assign_tiles(len(windows), 4, 0)
    reader = GroupFrameReader(
        packed_store,
        windows=local_tile_windows(windows, assignment),
        prognostic_var_names=CHANNELS,
        boundary_var_names=BOUNDARY,
        threads=2,
    )
    assert reader.num_tiles == 9
    assert reader.prognostic.num_chunks == 16
    reader.close()
