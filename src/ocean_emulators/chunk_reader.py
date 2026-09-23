"""Read a whole tile group's frame by decoding each store chunk once.

The per-tile reader in `datasets.py` is right for a group of one, or for a
directory of pre-cut caches where each tile is its own store. It is the wrong
shape for a face cut out of one packed cache, because the tiles overlap and
their windows straddle the store's chunk grid.

A tile window is ``[720c - overlap, 720c + 720 + overlap)``. It starts inside
chunk ``c-1`` and ends inside chunk ``c+1``, so an interior tile intersects
3x3 = 9 chunks. Asking zarr for the window makes it fetch and inflate all nine
and then throw most of each away; over a 36-tile face that is 256 chunk decodes
to deliver 36 chunks of distinct data.

This module inverts the loop. It walks *chunks* rather than tiles: each chunk
the group needs is read exactly once and scattered into every tile buffer that
overlaps it, then dropped. A rank holding a contiguous 3x3 block of tiles
touches 16 chunks instead of 81, and measured end to end that is 5x faster than
the per-tile path (`scripts/_bench/face_read.py`).

The face is never materialized. Peak memory is the tile buffers plus one
inflated chunk per thread, so a rank pays for the nine tiles it owns rather
than for the whole 4320^2 field.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import xarray as xr
import zarr

from ocean_emulators.utils.data import (
    _packed_channel_indices,
    _packed_channel_names,
    _with_julian_time_coord,
)

logger = logging.getLogger(__name__)

#: Reading more chunks at once than this made the whole face SLOWER in
#: `scripts/_bench/face_read.py` -- 4 ranks x 15 threads took 7.85 s against
#: 2.91 s for 4 x 8 on identical work. The store, not the core count, sets
#: this, so callers size their threads against the total across ranks.
SUGGESTED_TOTAL_READ_THREADS = 32


@dataclasses.dataclass(frozen=True)
class ChunkCopy:
    """The piece of one chunk that belongs to one tile."""

    #: Position in the reader's window list.
    tile: int
    #: Where it lands in that tile's buffer, as ``(rows, cols)``.
    destination: tuple[slice, slice]
    #: Where it comes from inside the inflated chunk.
    source: tuple[slice, slice]


def chunk_plan(
    windows: Sequence[tuple[int, int, int, int]],
    *,
    chunk_rows: int,
    chunk_cols: int,
) -> dict[tuple[int, int], list[ChunkCopy]]:
    """Map each store chunk to the tile pieces it feeds.

    ``windows`` are ``(i_start, i_end, j_start, j_end)`` in the store's own
    index space -- the caller has already subtracted the store's origin, so a
    cropped cache and a whole-face one look the same here.

    Keys are ``(chunk_row, chunk_col)``. A chunk appears once no matter how
    many tiles want it, which is the entire point.
    """
    if chunk_rows < 1 or chunk_cols < 1:
        raise ValueError(f"chunk shape must be positive, got {(chunk_rows, chunk_cols)}")

    plan: dict[tuple[int, int], list[ChunkCopy]] = {}
    for tile, (i0, i1, j0, j1) in enumerate(windows):
        if i1 <= i0 or j1 <= j0:
            raise ValueError(f"Tile {tile} has an empty window {(i0, i1, j0, j1)}")
        for chunk_row in range(j0 // chunk_rows, (j1 - 1) // chunk_rows + 1):
            for chunk_col in range(i0 // chunk_cols, (i1 - 1) // chunk_cols + 1):
                base_j, base_i = chunk_row * chunk_rows, chunk_col * chunk_cols
                # The overlap of tile and chunk, in store coordinates.
                shared_j0, shared_j1 = max(j0, base_j), min(j1, base_j + chunk_rows)
                shared_i0, shared_i1 = max(i0, base_i), min(i1, base_i + chunk_cols)
                plan.setdefault((chunk_row, chunk_col), []).append(
                    ChunkCopy(
                        tile=tile,
                        destination=(
                            slice(shared_j0 - j0, shared_j1 - j0),
                            slice(shared_i0 - i0, shared_i1 - i0),
                        ),
                        source=(
                            slice(shared_j0 - base_j, shared_j1 - base_j),
                            slice(shared_i0 - base_i, shared_i1 - base_i),
                        ),
                    )
                )
    return plan


class GroupChunkReader:
    """One packed array, read a whole tile group at a time.

    Produces exactly what `TorchTrainDataset._read_prognostic` produces for each
    tile -- a ``[1, C, H, W]`` tensor in the store's own dtype, channels in the
    requested order -- so it is a drop-in for the per-tile read and
    `tests/test_chunk_reader.py` holds it to that.
    """

    def __init__(
        self,
        store: str,
        *,
        prefix: str,
        windows: Sequence[tuple[int, int, int, int, int]],
        var_names: Sequence[str],
        threads: int = 8,
    ) -> None:
        if not windows:
            raise ValueError("GroupChunkReader needs at least one tile window")
        if threads < 1:
            raise ValueError(f"threads must be >= 1, got {threads}")

        self.store = str(store)
        self.prefix = prefix
        self.threads = threads

        group = zarr.open(self.store, mode="r")
        self.array = group[prefix]
        if self.array.ndim != 4:
            raise ValueError(
                f"{prefix} in {store} has {self.array.ndim} dimensions; the "
                "packed layout this reader understands is [time, channel, y, x]."
            )

        # Decode the time axis exactly as `DataSource` does, or a timestamp
        # from a dataset will never match a row here: the packed caches store
        # hours-since-epoch integers, xarray turns them into datetime64, and
        # `_with_julian_time_coord` turns those into cftime Julian objects.
        metadata = _with_julian_time_coord(xr.open_zarr(self.store, chunks=None))
        self.channel_indices = _packed_channel_indices(
            _packed_channel_names(metadata, prefix),
            list(var_names),
            prefix=prefix,
        )[0]
        # Absolute LLC indices the store starts at. A whole-face cache starts at
        # zero and a pre-cut one does not, and the windows are absolute either
        # way, so this is what keeps both correct.
        self.origin_j = int(metadata["y"].to_numpy()[0])
        self.origin_i = int(metadata["x"].to_numpy()[0])
        self.store_times = metadata["time"].to_numpy()
        metadata.close()

        local: list[tuple[int, int, int, int]] = []
        height, width = self.array.shape[-2:]
        for tile, window in enumerate(windows):
            _, i0, i1, j0, j1 = window
            i0, i1 = i0 - self.origin_i, i1 - self.origin_i
            j0, j1 = j0 - self.origin_j, j1 - self.origin_j
            if not (0 <= j0 < j1 <= height and 0 <= i0 < i1 <= width):
                raise ValueError(
                    f"Tile {tile} window {window} falls outside {store}, whose "
                    f"extent is j[{self.origin_j}:{self.origin_j + height}) "
                    f"i[{self.origin_i}:{self.origin_i + width})"
                )
            local.append((i0, i1, j0, j1))
        self.windows = tuple(local)
        self.tile_shapes = tuple((j1 - j0, i1 - i0) for i0, i1, j0, j1 in local)

        chunk_rows, chunk_cols = self.array.chunks[-2:]
        self.plan = chunk_plan(local, chunk_rows=chunk_rows, chunk_cols=chunk_cols)
        self.chunk_shape = (chunk_rows, chunk_cols)
        self._pool = ThreadPoolExecutor(
            max_workers=threads, thread_name_prefix=f"chunk-{prefix}"
        )
        logger.info(
            "GroupChunkReader(%s): %d tiles, %d chunk decode(s) per frame "
            "(%d distinct chunks would be the floor), %d thread(s)",
            prefix,
            len(self.windows),
            len(self.plan),
            len(self.plan),
            threads,
        )

    @property
    def num_chunks(self) -> int:
        """Chunk decodes per frame, reading each needed chunk exactly once."""
        return len(self.plan)

    @property
    def naive_chunks(self) -> int:
        """What one-read-per-tile would decode, for the same tiles."""
        rows, cols = self.chunk_shape
        return sum(
            len(range(j0 // rows, (j1 - 1) // rows + 1))
            * len(range(i0 // cols, (i1 - 1) // cols + 1))
            for i0, i1, j0, j1 in self.windows
        )

    def store_row(self, timestamp) -> int:
        """Where a timestamp sits on the store's own time axis.

        A `DataSource` is sliced to a train or val window before any dataset
        sees it, so a position there is not a row here. Matching on the
        timestamp rather than carrying an offset is what keeps this right when
        the caller's dataset was sliced, strided, or built from a different
        store entirely.
        """
        matches = np.flatnonzero(self.store_times == timestamp)
        if matches.size != 1:
            raise ValueError(
                f"Timestamp {timestamp!r} ({type(timestamp).__name__}) matches "
                f"{matches.size} rows of {self.store}'s time axis, which holds "
                f"{type(self.store_times[0]).__name__} from "
                f"{self.store_times[0]!r} to {self.store_times[-1]!r}; expected "
                "exactly one. A mismatch in TYPE rather than value means the "
                "two sides decoded the time axis differently."
            )
        return int(matches[0])

    def read(self, store_row: int) -> list[torch.Tensor]:
        """Every tile's ``[1, C, H, W]`` frame at one store row."""
        if not 0 <= store_row < self.array.shape[0]:
            raise IndexError(
                f"store row {store_row} is outside {self.store}'s "
                f"{self.array.shape[0]} timestamps"
            )
        channels = len(self.channel_indices)
        buffers = [
            np.empty((channels, height, width), dtype=self.array.dtype)
            for height, width in self.tile_shapes
        ]
        # Contiguous channel runs are the norm (a packed cache stores variables
        # in blocks), and slicing beats fancy indexing into a fresh array.
        selection = self._channel_selection()

        def fill(item: tuple[tuple[int, int], list[ChunkCopy]]) -> None:
            (chunk_row, chunk_col), copies = item
            base_j = chunk_row * self.chunk_shape[0]
            base_i = chunk_col * self.chunk_shape[1]
            chunk = self.array[
                store_row,
                selection,
                base_j : base_j + self.chunk_shape[0],
                base_i : base_i + self.chunk_shape[1],
            ]
            for copy in copies:
                buffers[copy.tile][:, copy.destination[0], copy.destination[1]] = chunk[
                    :, copy.source[0], copy.source[1]
                ]

        # Surface the first worker error rather than leaving a half-filled
        # buffer to become a silently wrong training target.
        for outcome in [self._pool.submit(fill, item) for item in self.plan.items()]:
            outcome.result()

        return [torch.from_numpy(buffer).unsqueeze(0) for buffer in buffers]

    def _channel_selection(self):
        indices = self.channel_indices
        contiguous = indices == list(range(indices[0], indices[0] + len(indices)))
        if contiguous:
            return slice(indices[0], indices[0] + len(indices))
        return indices

    def close(self) -> None:
        self._pool.shutdown(wait=False)

    def __del__(self) -> None:
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)


class GroupFrameReader:
    """Both arrays of a packed cache, read a whole tile group at a time.

    The per-tile `DataSource` path is one read per tile per array. For a face
    that is 36 reads over windows that overlap each other and straddle the
    chunk grid; this is one chunk-streaming pass per array instead.

    Timestamps, not positions, address a frame. A `DataSource` is sliced to a
    train or val window before any dataset sees it, so its index 0 is not the
    store's, and a stride makes the two drift further apart.
    """

    def __init__(
        self,
        store: str,
        *,
        windows: Sequence[tuple[int, int, int, int, int]],
        prognostic_var_names: Sequence[str],
        boundary_var_names: Sequence[str],
        threads: int = 8,
    ) -> None:
        self.prognostic = GroupChunkReader(
            store,
            prefix="prognostic",
            windows=windows,
            var_names=prognostic_var_names,
            threads=threads,
        )
        self.boundary = GroupChunkReader(
            store,
            prefix="boundary",
            windows=windows,
            var_names=boundary_var_names,
            threads=threads,
        )
        self.num_tiles = len(windows)

    def read_prognostic(self, timestamp) -> list[torch.Tensor]:
        return self.prognostic.read(self.prognostic.store_row(timestamp))

    def read_boundary(self, timestamp) -> list[torch.Tensor]:
        return self.boundary.read(self.boundary.store_row(timestamp))

    @property
    def chunks_per_frame(self) -> int:
        return self.prognostic.num_chunks + self.boundary.num_chunks

    @property
    def naive_chunks_per_frame(self) -> int:
        return self.prognostic.naive_chunks + self.boundary.naive_chunks

    def close(self) -> None:
        self.prognostic.close()
        self.boundary.close()
