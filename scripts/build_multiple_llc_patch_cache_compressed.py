#!/usr/bin/env python
"""Chunk-first builder for several LLC patch caches at once.

The existing builder is *cache-first*: one process per output cache, each
opening whatever source chunks its own tile needs. Run four of them over a 2x2
tile block and the same source chunks get read, decompressed and thrown away
several times over -- 36 reads of 720x720 chunks per 3D variable per timestep
where 16 distinct chunks would do, and four full-globe reads of every 2D
variable where one would do.

This builder inverts the loop. It works out which source chunks the whole set of
tiles needs, opens each of them **once** per timestep, and scatters the pieces
into every cache that wants them. Reads drop to the theoretical minimum: each
source byte is read exactly once no matter how many caches overlap it.

The saving grows with the number of tiles. Over a 2x2 block it is ~2.6x. Over a
face or the globe, where every 720x720 chunk feeds up to four tiles and one 2D
chunk feeds all of them, it approaches "read the source store once", which is
the floor for any tool.

Output format is byte-compatible with `llc-train-ready-v1` from
build_llc_patch_cache_compressed_train_val.py: same channel order, same packed
layout, same stats/masks/attrs, same one-chunk-per-timestep chunking. Channel
order and stats naming are imported from that builder's shared module rather
than re-derived, so the two cannot drift.

Parallelism is by TIME, not by cache. `--init` creates every store once, then
any number of `--fill` jobs write disjoint time ranges into them concurrently.
Zarr keys a chunk by its time index, so disjoint time ranges touch disjoint
files and never race. Splitting by time adds no redundant reads, unlike
splitting by cache.

    # once
    build_multiple_llc_patch_cache_compressed.py --init ...
    # then N of these in parallel, each a quarter of the time axis
    build_multiple_llc_patch_cache_compressed.py --fill --time-index-start 0    --time-index-stop 2578 ...
    build_multiple_llc_patch_cache_compressed.py --fill --time-index-start 2578 --time-index-stop 5156 ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time as _time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numcodecs
import numpy as np
import xarray as xr
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_llc_patch_cache_uncompressed_train_val import (  # noqa: E402
    DEFAULT_BOUNDARY_CHANNELS,
    DEFAULT_PROGNOSTIC_CHANNELS,
    get_numpy_float_type,
    stats_var_name,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

CACHE_FORMAT = "llc-train-ready-v1"
#: LLC's horizontal chunk edge. Every 3D source array is chunked (1, k, 1, 720, 720).
SOURCE_CHUNK = 720
#: Axis-name vocabulary, matching the source store's `_ARRAY_DIMENSIONS`.
LEVEL_DIMS = ("k", "k_p1", "k_l", "k_u", "lev")
ROW_DIMS = ("j", "j_g", "y", "lat")
COL_DIMS = ("i", "i_g", "x", "lon")


# ---------------------------------------------------------------- tiles


@dataclass(frozen=True)
class TileSpec:
    """One output cache: a window on one LLC face."""

    name: str
    face: int
    i_start: int
    i_end: int
    j_start: int
    j_end: int

    @property
    def height(self) -> int:
        return self.j_end - self.j_start

    @property
    def width(self) -> int:
        return self.i_end - self.i_start

    def chunk_span(self) -> list[tuple[int, int]]:
        """The (j, i) source chunks this tile overlaps."""
        js = range(self.j_start // SOURCE_CHUNK, (self.j_end - 1) // SOURCE_CHUNK + 1)
        return [
            (jc, ic)
            for jc in js
            for ic in range(
                self.i_start // SOURCE_CHUNK, (self.i_end - 1) // SOURCE_CHUNK + 1
            )
        ]


def parse_tiles(raw: str) -> list[TileSpec]:
    """Tiles from JSON: a list of [face, i_start, i_end, j_start, j_end].

    A file path is read as JSON too, so a large global tiling need not be pasted
    onto a command line.
    """
    text = Path(raw).read_text() if Path(raw).is_file() else raw
    entries = json.loads(text)
    tiles = []
    for entry in entries:
        if isinstance(entry, dict):
            face, i0, i1, j0, j1 = (
                entry["face"], entry["i_start"], entry["i_end"],
                entry["j_start"], entry["j_end"],
            )
        else:
            if len(entry) != 5:
                raise ValueError(
                    f"Each tile must be [face, i_start, i_end, j_start, j_end]; got {entry}"
                )
            face, i0, i1, j0, j1 = (int(v) for v in entry)
        tiles.append(
            TileSpec(
                name=f"LLC4320_face{face}_i{i0}-{i1}_j{j0}-{j1}",
                face=int(face), i_start=int(i0), i_end=int(i1),
                j_start=int(j0), j_end=int(j1),
            )
        )
    if not tiles:
        raise ValueError("No tiles given")
    shapes = {(t.height, t.width) for t in tiles}
    if len(shapes) != 1:
        raise ValueError(f"All tiles must share one shape; got {sorted(shapes)}")
    return tiles


def plan_chunks(tiles: list[TileSpec]) -> dict[int, list[tuple[int, int]]]:
    """Distinct 720-grid chunks per face, for the face list and the log line.

    The actual reads are planned per array by `SourceArray.blocks_for`, because
    LLC's 2D arrays are chunked globe-at-a-time rather than 720x720.
    """
    per_face: dict[int, set[tuple[int, int]]] = {}
    for tile in tiles:
        per_face.setdefault(tile.face, set()).update(tile.chunk_span())
    return {face: sorted(chunks) for face, chunks in per_face.items()}


# ------------------------------------------------------- source access


@dataclass(frozen=True)
class SourceArray:
    """A source array plus where each LLC axis sits in its dimension order."""

    array: zarr.Array
    time: int
    level: int | None
    face: int | None
    row: int
    col: int
    ndim: int
    #: This array's OWN horizontal chunk edge. Not a constant: LLC's 3D arrays
    #: are chunked 720x720 while its 2D arrays put the whole globe in one chunk.
    #: Planning reads on a hardcoded 720 grid decompresses a 2D chunk once per
    #: cell -- 16 full-globe inflations to serve a 2x2 tile block, which is the
    #: exact waste this builder exists to remove.
    row_chunk: int
    col_chunk: int

    def blocks_for(self, tiles: list["TileSpec"], face: int) -> list[tuple[int, int]]:
        """Origins of the distinct chunk-aligned blocks covering `tiles`."""
        rows, cols = self.array.shape[self.row], self.array.shape[self.col]
        origins = set()
        for tile in tiles:
            if tile.face != face:
                continue
            for j in range(
                (tile.j_start // self.row_chunk) * self.row_chunk,
                min(tile.j_end, rows), self.row_chunk,
            ):
                for i in range(
                    (tile.i_start // self.col_chunk) * self.col_chunk,
                    min(tile.i_end, cols), self.col_chunk,
                ):
                    origins.add((j, i))
        return sorted(origins)

    def read_origin(self, t: int, face: int, j0: int, i0: int) -> np.ndarray:
        rows, cols = self.array.shape[self.row], self.array.shape[self.col]
        return self.read_block(
            t, face, j0, min(j0 + self.row_chunk, rows),
            i0, min(i0 + self.col_chunk, cols),
        )

    def read_block(
        self, t: int, face: int, j0: int, j1: int, i0: int, i1: int
    ) -> np.ndarray:
        """`[level, j, i]` (or `[j, i]` when the array has no vertical axis)."""
        index: list[object] = [slice(None)] * self.ndim
        index[self.time] = t
        if self.face is not None:
            index[self.face] = face
        index[self.row] = slice(j0, j1)
        index[self.col] = slice(i0, i1)
        return self.array[tuple(index)]


def open_source_array(group: zarr.Group, name: str) -> SourceArray:
    array = group[name]
    dims = array.attrs.get("_ARRAY_DIMENSIONS")
    if dims is None:
        raise KeyError(f"Source array {name} has no _ARRAY_DIMENSIONS")

    def find(candidates) -> int | None:
        for position, dim in enumerate(dims):
            if dim in candidates:
                return position
        return None

    time_axis = find(("time",))
    row, col = find(ROW_DIMS), find(COL_DIMS)
    if row is None or col is None:
        raise KeyError(f"Source array {name} has dims {dims}; expected j/i axes")
    return SourceArray(
        array=array,
        time=-1 if time_axis is None else time_axis,
        level=find(LEVEL_DIMS),
        face=find(("face",)),
        row=row,
        col=col,
        ndim=len(dims),
        row_chunk=array.chunks[row],
        col_chunk=array.chunks[col],
    )


def split_channel(channel: str) -> tuple[str, int | None]:
    """`Theta_7` -> `("Theta", 7)`; `Eta` -> `("Eta", None)`."""
    base, _, level = channel.rpartition("_")
    if base and level.isdigit():
        return base, int(level)
    return channel, None


# ----------------------------------------------------------- the store


def build_compressor(args: argparse.Namespace):
    shuffle = {
        "shuffle": numcodecs.Blosc.SHUFFLE,
        "bitshuffle": numcodecs.Blosc.BITSHUFFLE,
        "noshuffle": numcodecs.Blosc.NOSHUFFLE,
    }[args.shuffle]
    return numcodecs.Blosc(cname=args.compressor, clevel=args.compression_level, shuffle=shuffle)


def create_array(group, name, shape, chunks, dtype, dims, compressor=None):
    array = group.create_dataset(
        name, shape=shape, chunks=chunks, dtype=dtype,
        compressor=compressor, overwrite=True,
    )
    array.attrs["_ARRAY_DIMENSIONS"] = list(dims)
    return array


def init_store(
    path: Path,
    tile: TileSpec,
    *,
    args: argparse.Namespace,
    prognostic_channels: list[str],
    boundary_channels: list[str],
    times: np.ndarray,
    time_attrs: dict,
    train_count: int,
    val_count: int,
    statics: dict[str, np.ndarray],
    means: xr.Dataset,
    stds: xr.Dataset,
) -> None:
    """Create one cache with its full time extent and every static array."""
    float_dtype = get_numpy_float_type(args.float_type)
    compressor = build_compressor(args)
    n_prog, n_bound = len(prognostic_channels), len(boundary_channels)
    n_time, height, width = len(times), tile.height, tile.width

    group = zarr.open_group(str(path), mode="w")

    create_array(group, "prognostic", (n_time, n_prog, height, width),
                 (args.time_chunk, n_prog, height, width), float_dtype,
                 ("time", "prognostic_channel", "y", "x"), compressor)
    create_array(group, "boundary", (n_time, n_bound, height, width),
                 (args.time_chunk, n_bound, height, width), float_dtype,
                 ("time", "boundary_channel", "y", "x"), compressor)

    # Stats, in the same order as the channels and under the same names the
    # existing builder resolves, so a cache from either tool normalizes alike.
    for prefix, channels, source in (
        ("prognostic", prognostic_channels, means),
        ("boundary", boundary_channels, means),
    ):
        values = np.asarray(
            [source[stats_var_name(c, source)].squeeze().item() for c in channels],
            dtype=float_dtype,
        )
        create_array(group, f"{prefix}_mean", values.shape, values.shape,
                     float_dtype, (f"{prefix}_channel",))[:] = values
    for prefix, channels in (("prognostic", prognostic_channels),
                             ("boundary", boundary_channels)):
        values = np.asarray(
            [stds[stats_var_name(c, stds)].squeeze().item() for c in channels],
            dtype=float_dtype,
        )
        create_array(group, f"{prefix}_std", values.shape, values.shape,
                     float_dtype, (f"{prefix}_channel",))[:] = values

    # Masks: a 3D channel takes its own level, a surface channel takes level 0.
    # W lives on cell interfaces and borrows the mask of the cell below its
    # face, which is what the existing builder does.
    tile_mask = statics["mask"]
    prognostic_mask = np.stack(
        [tile_mask[level if level is not None else 0] for _, level in
         (split_channel(c) for c in prognostic_channels)]
    ).astype("i1")
    boundary_mask = np.repeat(tile_mask[0][None].astype("i1"), n_bound, axis=0)
    create_array(group, "prognostic_mask", prognostic_mask.shape,
                 prognostic_mask.shape, "i1",
                 ("prognostic_channel", "y", "x"))[:] = prognostic_mask
    create_array(group, "boundary_mask", boundary_mask.shape, boundary_mask.shape,
                 "i1", ("boundary_channel", "y", "x"))[:] = boundary_mask

    for name in ("XC", "YC", "rA"):
        values = statics[name].astype(np.float32)
        create_array(group, name, values.shape, values.shape, "f4", ("y", "x"))[:] = values

    create_array(group, "time", (n_time,), (min(n_time, 1024),), times.dtype,
                 ("time",))[:] = times
    group["time"].attrs.update(time_attrs)
    create_array(group, "y", (height,), (height,), "i2", ("y",))[:] = np.arange(
        tile.j_start, tile.j_end, dtype=np.int16)
    create_array(group, "x", (width,), (width,), "i2", ("x",))[:] = np.arange(
        tile.i_start, tile.i_end, dtype=np.int16)
    create_array(group, "prognostic_channel", (n_prog,), (n_prog,), "i4",
                 ("prognostic_channel",))[:] = np.arange(n_prog, dtype=np.int32)
    create_array(group, "boundary_channel", (n_bound,), (n_bound,), "i4",
                 ("boundary_channel",))[:] = np.arange(n_bound, dtype=np.int32)

    group.attrs.update({
        "cache_format": CACHE_FORMAT,
        "built_by": "build_multiple_llc_patch_cache_compressed.py",
        "source_path": str(args.source),
        "means_path": str(args.means),
        "stds_path": str(args.stds),
        "train_start": args.train_start, "train_end": args.train_end,
        "val_start": args.val_start, "val_end": args.val_end,
        "train_time_count": int(train_count), "val_time_count": int(val_count),
        "time_chunk": args.time_chunk,
        "float_type": args.float_type,
        "compression_codec": args.compressor,
        "compression_level": args.compression_level,
        "compression_shuffle": args.shuffle,
        "compression_target_vars": "prognostic,boundary",
        "prognostic_channel_count": n_prog,
        "boundary_channel_count": n_bound,
        "prognostic_channel_names_json": json.dumps(prognostic_channels),
        "boundary_channel_names_json": json.dumps(boundary_channels),
        "llc_face": tile.face,
        "llc_i_start": tile.i_start, "llc_i_end": tile.i_end,
        "llc_j_start": tile.j_start, "llc_j_end": tile.j_end,
    })
    zarr.consolidate_metadata(str(path))


# ------------------------------------------------------------- filling


def gather_tile(
    blocks: dict[tuple[int, int], np.ndarray], tile: TileSpec, levels: int | None
) -> np.ndarray:
    """Assemble one tile's window out of the blocks covering it.

    `blocks` is keyed by each block's absolute (j, i) origin, and its extent is
    read off the block itself -- so this works whether a block is one 720x720
    LLC chunk or a whole 4320x4320 globe plane.
    """
    shape = (
        (tile.height, tile.width) if levels is None
        else (levels, tile.height, tile.width)
    )
    out = np.empty(shape, dtype=np.float32)
    for (j0, i0), block in blocks.items():
        # Overlap of this block with the tile, in absolute LLC indices.
        aj0, aj1 = max(j0, tile.j_start), min(j0 + block.shape[-2], tile.j_end)
        ai0, ai1 = max(i0, tile.i_start), min(i0 + block.shape[-1], tile.i_end)
        if aj0 >= aj1 or ai0 >= ai1:
            continue
        src = (..., slice(aj0 - j0, aj1 - j0), slice(ai0 - i0, ai1 - i0))
        dst = (..., slice(aj0 - tile.j_start, aj1 - tile.j_start),
               slice(ai0 - tile.i_start, ai1 - tile.i_start))
        out[dst] = block[src]
    return out


def fill_time_range(
    tiles: list[TileSpec],
    stores: dict[str, zarr.Group],
    source: zarr.Group,
    *,
    args: argparse.Namespace,
    prognostic_channels: list[str],
    boundary_channels: list[str],
    time_indices: np.ndarray,
    start: int,
    stop: int,
) -> None:
    """Write `[start, stop)` of the output time axis into every cache."""
    float_dtype = get_numpy_float_type(args.float_type)
    chunk_plan = plan_chunks(tiles)
    faces = sorted(chunk_plan)

    # Group channels by source array so a 3D array is read once for all its
    # levels, exactly as the packed layout wants them.
    def group_channels(channels):
        order: list[str] = []
        levels: dict[str, list[tuple[int, int | None]]] = {}
        for position, channel in enumerate(channels):
            base, level = split_channel(channel)
            if base not in levels:
                levels[base] = []
                order.append(base)
            levels[base].append((position, level))
        return order, levels

    prog_order, prog_levels = group_channels(prognostic_channels)
    bound_order, bound_levels = group_channels(boundary_channels)
    arrays = {
        name: open_source_array(source, name)
        for name in dict.fromkeys(prog_order + bound_order)
    }

    n_prog, n_bound = len(prognostic_channels), len(boundary_channels)
    buffers = {
        t.name: (
            np.empty((n_prog, t.height, t.width), dtype=float_dtype),
            np.empty((n_bound, t.height, t.width), dtype=float_dtype),
        )
        for t in tiles
    }

    pool = ThreadPoolExecutor(max_workers=args.workers) if args.workers > 1 else None

    def scatter(base: str, face: int, blocks: dict[tuple[int, int], np.ndarray]) -> None:
        """Push one variable's blocks into every tile that overlaps them."""
        array = arrays[base]
        levels = None if array.level is None else next(iter(blocks.values())).shape[0]
        slots = [
            (which, positions)
            for which, table in ((0, prog_levels), (1, bound_levels))
            if (positions := table.get(base)) is not None
        ]
        for tile in tiles:
            if tile.face != face:
                continue
            window = gather_tile(blocks, tile, levels)
            for which, positions in slots:
                buffer = buffers[tile.name][which]
                for position, level in positions:
                    buffer[position] = (
                        window if level is None else window[level]
                    ).astype(float_dtype, copy=False)

    # A variable feeding both packed arrays -- Eta is a prognostic *and* a
    # boundary channel -- must still be read once.
    all_bases = list(dict.fromkeys(prog_order + bound_order))
    volume_bases = [b for b in all_bases if arrays[b].level is not None]
    surface_bases = [b for b in all_bases if arrays[b].level is None]

    blocks_per_var = max(
        (len(arrays[b].blocks_for(tiles, f)) for b in volume_bases for f in faces),
        default=1,
    )
    group_size = max(1, -(-args.workers // max(blocks_per_var, 1)))
    logger.info(
        "read plan: %d block(s) per depth variable, %d variable(s) in flight, "
        "%d surface block(s) -- peak scratch ~%.1f GiB",
        blocks_per_var, group_size, len(surface_bases),
        group_size * blocks_per_var * 105 / 1024,
    )

    started = _time.perf_counter()
    try:
        for offset, out_index in enumerate(range(start, stop)):
            source_index = int(time_indices[out_index])

            # Depth-resolved variables. One variable offers only
            # `blocks_per_var` parallel reads -- 16 for a 2x2 block of 1104
            # tiles -- so with more threads than that, reading one variable at a
            # time leaves most of them idle. Read `group_size` variables at once
            # instead, chosen so the work list just covers the thread pool.
            # Peak scratch stays bounded at group_size x blocks_per_var chunks.
            for start_base in range(0, len(volume_bases), group_size):
                group = volume_bases[start_base:start_base + group_size]
                jobs = [
                    (base, face, origin)
                    for base in group for face in faces
                    for origin in arrays[base].blocks_for(tiles, face)
                ]
                def read_volume(job, t=source_index):
                    base, face, origin = job
                    return job, arrays[base].read_origin(t, face, *origin)
                pairs = pool.map(read_volume, jobs) if pool else map(read_volume, jobs)
                grouped: dict[tuple[str, int], dict[tuple[int, int], np.ndarray]] = {}
                for (base, face, origin), block in pairs:
                    grouped.setdefault((base, face), {})[origin] = block
                for (base, face), blocks in grouped.items():
                    scatter(base, face, blocks)
                del grouped

            # Surface variables: ONE chunk holds the whole globe, so there is no
            # parallelism within a variable -- parallelise across variables
            # instead, and inflate each of those chunks exactly once.
            surface_jobs = [
                (base, face, origin)
                for base in surface_bases
                for face in faces
                for origin in arrays[base].blocks_for(tiles, face)
            ]
            def read_surface(job, t=source_index):
                base, face, origin = job
                return job, arrays[base].read_origin(t, face, *origin)
            pairs = pool.map(read_surface, surface_jobs) if pool else map(read_surface, surface_jobs)
            surface_blocks: dict[tuple[str, int], dict[tuple[int, int], np.ndarray]] = {}
            for (base, face, origin), block in pairs:
                surface_blocks.setdefault((base, face), {})[origin] = block
            for (base, face), blocks in surface_blocks.items():
                scatter(base, face, blocks)
            del surface_blocks

            for tile in tiles:
                prognostic, boundary = buffers[tile.name]
                stores[tile.name]["prognostic"][out_index] = prognostic
                stores[tile.name]["boundary"][out_index] = boundary

            if (offset + 1) % args.log_every == 0 or out_index == stop - 1:
                done = offset + 1
                rate = (_time.perf_counter() - started) / done
                logger.info(
                    "time %d/%d (store index %d) | %.1f s/step | eta %.1f h",
                    done, stop - start, source_index, rate,
                    rate * (stop - start - done) / 3600,
                )
    finally:
        if pool is not None:
            pool.shutdown()


# ---------------------------------------------------------------- main


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--means", type=Path, required=True)
    p.add_argument("--stds", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--tiles", required=True,
                   help="JSON list of [face, i_start, i_end, j_start, j_end], or a path to one")
    p.add_argument("--name-suffix", default="",
                   help="Appended to each cache directory name, before .zarr")
    p.add_argument("--train-start", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--val-start", required=True)
    p.add_argument("--val-end", required=True)
    p.add_argument("--prognostic-channels", default=None,
                   help="JSON list; defaults to PROGNOSTIC_VARS['all']")
    p.add_argument("--boundary-channels", default=None,
                   help="JSON list; defaults to BOUNDARY_VARS['all']")
    p.add_argument("--float-type", default="float16")
    p.add_argument("--time-chunk", type=int, default=1)
    p.add_argument("--compressor", default="lz4")
    p.add_argument("--compression-level", type=int, default=5)
    p.add_argument("--shuffle", default="shuffle",
                   choices=("shuffle", "bitshuffle", "noshuffle"))
    p.add_argument("--workers", type=int, default=8,
                   help="Threads for source chunk reads (blosc releases the GIL)")
    p.add_argument("--init", action="store_true", help="Create the stores and their statics")
    p.add_argument("--fill", action="store_true", help="Write a time range into existing stores")
    p.add_argument("--time-index-start", type=int, default=0)
    p.add_argument("--time-index-stop", type=int, default=None,
                   help="Exclusive; defaults to the end of the selected time axis")
    p.add_argument("--time-splits", type=int, default=1,
                   help="Split the time axis into this many equal ranges (for job arrays)")
    p.add_argument("--time-split-index", type=int, default=None,
                   help="Which split this job fills, 0-based. Overrides --time-index-*")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if not args.init and not args.fill:
        args.init = args.fill = True
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    tiles = parse_tiles(args.tiles)
    prognostic_channels = (
        json.loads(args.prognostic_channels) if args.prognostic_channels
        else list(DEFAULT_PROGNOSTIC_CHANNELS)
    )
    boundary_channels = (
        json.loads(args.boundary_channels) if args.boundary_channels
        else list(DEFAULT_BOUNDARY_CHANNELS)
    )

    chunk_plan = plan_chunks(tiles)
    distinct = sum(len(v) for v in chunk_plan.values())
    naive = sum(len(t.chunk_span()) for t in tiles)
    logger.info("%d tile(s) of %dx%d", len(tiles), tiles[0].height, tiles[0].width)
    logger.info(
        "source chunks per 3D variable per timestep: %d distinct vs %d cache-first "
        "-> %.2fx fewer reads", distinct, naive, naive / distinct,
    )
    logger.info("channels: %d prognostic, %d boundary",
                len(prognostic_channels), len(boundary_channels))

    # The time axis is resolved through xarray so the train/val window semantics
    # are identical to the existing builder (label slicing, inclusive of both ends).
    data = xr.open_zarr(args.source, chunks={}, consolidated=False)
    train = data.sel(time=slice(args.train_start, args.train_end)).time.values
    val = data.sel(time=slice(args.val_start, args.val_end)).time.values
    if train.size == 0 or val.size == 0:
        raise ValueError("Train or val time selection is empty")
    combined = np.unique(np.concatenate([train, val]))
    lookup = {value: index for index, value in enumerate(data.time.values)}
    time_indices = np.array([lookup[value] for value in combined], dtype=np.int64)
    logger.info("Selected %d train + %d val = %d unique times",
                train.size, val.size, time_indices.size)

    source = zarr.open_group(str(args.source), mode="r")
    raw_time = source["time"]
    # Encode the time axis exactly as xarray does when the per-cache builder
    # writes it -- integer offsets from the first sample -- so a cache from
    # either builder is byte-identical rather than merely equivalent once
    # decoded. The source's own encoding (seconds from an epoch three days
    # before the first sample) would decode the same but compare different.
    import cftime

    decoded = cftime.num2date(
        raw_time[:][time_indices],
        raw_time.attrs["units"],
        calendar=raw_time.attrs.get("calendar", "standard"),
        only_use_cftime_datetimes=False,
    )
    origin = decoded[0]
    offsets = np.array([(d - origin).total_seconds() for d in decoded], dtype=np.int64)
    stamp = origin.strftime("%Y-%m-%d %H:%M:%S")
    if np.all(offsets % 3600 == 0):
        times, unit = offsets // 3600, "hours"
    else:
        times, unit = offsets, "seconds"
    time_attrs = {
        "units": f"{unit} since {stamp}",
        "calendar": "proleptic_gregorian",
    }

    if args.time_split_index is not None:
        # Contiguous, equal-as-possible ranges. Disjoint by construction, and a
        # zarr chunk is keyed by time index, so parallel fills never collide.
        if not 0 <= args.time_split_index < args.time_splits:
            raise ValueError(
                f"--time-split-index must be in [0, {args.time_splits})"
            )
        edges = np.linspace(0, time_indices.size, args.time_splits + 1).astype(int)
        args.time_index_start = int(edges[args.time_split_index])
        stop = int(edges[args.time_split_index + 1])
        logger.info("Split %d/%d -> time [%d:%d)", args.time_split_index + 1,
                    args.time_splits, args.time_index_start, stop)
    else:
        stop = args.time_index_stop if args.time_index_stop is not None else time_indices.size
    stop = min(stop, time_indices.size)
    if args.time_index_start >= stop:
        raise ValueError(f"Empty time range [{args.time_index_start}:{stop})")

    args.output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        t.name: args.output_root / f"{t.name}{args.name_suffix}.zarr" for t in tiles
    }

    if args.dry_run:
        for tile in tiles:
            logger.info("[dry-run] %s -> %s", tile.name, paths[tile.name])
        logger.info("[dry-run] would write time [%d:%d) of %d",
                    args.time_index_start, stop, time_indices.size)
        return

    if args.init:
        # Statics come from the same chunk-first read: the mask cube and the
        # grid fields are chunked on the same 720 grid as everything else.
        means = xr.open_zarr(args.means)
        stds = xr.open_zarr(args.stds)
        mask_source = open_source_array(source, "mask_c")
        grid_sources = {n: open_source_array(source, n) for n in ("XC", "YC", "rA")}
        for tile in tiles:
            path = paths[tile.name]
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"{path} exists; pass --overwrite")
            logger.info("Init %s", path)
            # Same chunk-first read as the time loop, on each array's own grid.
            def read_static(array, tile=tile):
                blocks = {
                    origin: array.read_origin(0, tile.face, *origin)
                    for origin in array.blocks_for([tile], tile.face)
                }
                levels = None if array.level is None else next(iter(blocks.values())).shape[0]
                return gather_tile(blocks, tile, levels)

            mask = read_static(mask_source).astype(bool)
            statics = {"mask": mask}
            for name, array in grid_sources.items():
                statics[name] = read_static(array)
            init_store(
                path, tile, args=args,
                prognostic_channels=prognostic_channels,
                boundary_channels=boundary_channels,
                times=times, time_attrs=time_attrs,
                train_count=train.size, val_count=val.size,
                statics=statics, means=means, stds=stds,
            )

    if args.fill:
        stores = {t.name: zarr.open_group(str(paths[t.name]), mode="r+") for t in tiles}
        logger.info("Filling time [%d:%d) of %d into %d cache(s)",
                    args.time_index_start, stop, time_indices.size, len(tiles))
        fill_time_range(
            tiles, stores, source, args=args,
            prognostic_channels=prognostic_channels,
            boundary_channels=boundary_channels,
            time_indices=time_indices,
            start=args.time_index_start, stop=stop,
        )
        # Deliberately NOT consolidating here. Init already wrote .zmetadata and
        # a fill changes no metadata -- only chunk contents -- so re-consolidating
        # is redundant, and with several time-split jobs writing the same stores
        # it would have them all rewriting one .zmetadata file concurrently.
    logger.info("Done.")


if __name__ == "__main__":
    main()
